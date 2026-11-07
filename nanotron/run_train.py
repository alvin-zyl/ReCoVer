"""
Nanotron training script.

Usage:
```
export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=8 run_train.py --config-file examples/config_tiny_llama.yaml
```
"""

import argparse, os
from typing import Dict, cast

import numpy as np
from nanotron import logging
from nanotron.config import (
    DataArgs,
    DatasetStageArgs,
    NanosetDatasetsArgs,
    PretrainDatasetsArgs,
    get_config_from_file,
    apply_config_overrides
)
from nanotron.data.dataloader_builder import build_nanoset_dataloader
from nanotron.dataloader import (
    clm_process,
    dummy_infinite_data_generator,
    get_datasets,
    get_datasets_from_disk,
    get_train_dataloader,
)
from nanotron.helpers import (
    compute_remain_train_steps_of_a_data_stage_from_ckp,
    get_consumed_train_samples_of_a_data_stage_from_ckp,
)
from nanotron.logging import log_rank
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.trainer import DistributedTrainer
from nanotron.utils import main_rank_first, get_args
from torch.utils.data import DataLoader

try:
    from huggingface_hub import __version__ as hf_hub_version
    from transformers import AutoTokenizer
    from transformers import __version__ as tf_version
except ImportError:
    hf_hub_version = None
    tf_version = None

logger = logging.get_logger(__name__)


def get_dataloader_from_data_stage(
    trainer: DistributedTrainer,
    data: DataArgs,
    consumed_train_samples: int,
    num_remaining_train_steps: int,
):
    """
    Returns a dataloader for a given data stage.

    data: The data configuration for the current stage.
    consumed_train_samples: The number of samples consumed by the model in the this stage (each stage starts from zero).
    num_remaining_train_steps: The number of remaining training steps for this stage.
    """
    assert (
        consumed_train_samples >= 0
    ), "consumed_train_samples should be greater than 0"
    assert (
        num_remaining_train_steps >= 0
    ), "num_remaining_train_steps should be greater than 0"

    # First, we need to know which ranks to feed the dataloader to
    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)

    # Case 1: Dummy data generator
    if data.dataset is None:
        log_rank(
            "Using dummy data generator", logger=logger, level=logging.INFO, rank=0
        )
        dataloader = dummy_infinite_data_generator(
            micro_batch_size=trainer.micro_batch_size,
            sequence_length=trainer.sequence_length,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            vocab_size=trainer.model_config.vocab_size,
            seed=data.seed,
            parallel_context=trainer.parallel_context,
        )()

    # Case 2: HuggingFace datasets
    elif isinstance(data.dataset, PretrainDatasetsArgs):
        log_rank("Using `datasets` library", logger=logger, level=logging.INFO, rank=0)

        # We need to the 1st device to process dataset and cache it, then other devices load from cache
        with main_rank_first(trainer.parallel_context.world_pg):
            # TODO @nouamanetazi: this may timeout before 1st device finishes processing dataset. Can we have a ctxmanager to modify timeout?
            # TODO: generalise to include  for validation/test splits

            # We load the raw dataset
            if data.dataset.load_from_disk:
                train_dataset = get_datasets_from_disk(
                    hf_dataset_path=data.dataset.hf_dataset_or_datasets,
                    splits=data.dataset.hf_dataset_splits,
                )["train"]
            else:
                raw_dataset = get_datasets(
                    hf_dataset_or_datasets=data.dataset.hf_dataset_or_datasets,
                    hf_dataset_config_name=data.dataset.hf_dataset_config_name,
                    splits=data.dataset.hf_dataset_splits,
                    streaming=data.dataset.streaming,
                )["train"]

                tokenizer_path = trainer.config.tokenizer.tokenizer_name_or_path
                log_rank(
                    f"Loading tokenizer from {tokenizer_path} and transformers/hf_hub versions {tf_version, hf_hub_version}",
                    logger=logger,
                    level=logging.INFO,
                    rank=0,
                )

                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "left"
                print(f"Tokenizer max model length: {tokenizer.model_max_length}")

                # Check that tokenizer's vocab size is smaller than the model's vocab size
                assert (
                    tokenizer.vocab_size <= trainer.model_config.vocab_size
                ), f"Tokenizer's vocab size ({tokenizer.vocab_size}) is larger than the model's vocab size ({trainer.model_config.vocab_size})"

                # We apply the Causal Language Modeling preprocessing
                train_dataset = clm_process(
                    raw_dataset=raw_dataset,
                    tokenizer=tokenizer,
                    text_column_name=data.dataset.text_column_name,
                    dataset_processing_num_proc_per_process=data.dataset.dataset_processing_num_proc_per_process,
                    dataset_overwrite_cache=data.dataset.dataset_overwrite_cache,
                    sequence_length=trainer.sequence_length,
                )

            # We load the processed dataset on the ranks requiring it
            dataloader = get_train_dataloader(
                train_dataset=train_dataset,
                sequence_length=trainer.sequence_length,
                parallel_context=trainer.parallel_context,
                input_pp_rank=input_pp_rank,
                output_pp_rank=output_pp_rank,
                micro_batch_size=trainer.micro_batch_size,
                consumed_train_samples=consumed_train_samples,
                dataloader_num_workers=data.num_loading_workers,
                seed_worker=data.seed,
                dataloader_drop_last=True,
            )

            # Check if we have enough samples for train_steps (skip for streaming datasets)
            if not data.dataset.streaming:
                total_tokens_dataset = len(dataloader.dataset) * trainer.sequence_length
                num_tokens_needed_for_training = (
                    num_remaining_train_steps
                    * trainer.global_batch_size
                    * trainer.sequence_length
                )
                assert num_tokens_needed_for_training <= total_tokens_dataset, (
                    f"Dataset is too small for steps ({total_tokens_dataset} < {num_tokens_needed_for_training}), "
                    f"Try train_steps<={len(dataloader.dataset) // trainer.global_batch_size + trainer.iteration_step}"
                )
            else:
                log_rank(
                    "Streaming mode enabled: skipping dataset size check. Ensure your dataset is large enough for training.",
                    logger=logger,
                    level=logging.WARNING,
                    rank=0,
                )

    # Case 3: Nanosets
    elif isinstance(data.dataset, NanosetDatasetsArgs):
        # Get tokenizer cardinality
        tokenizer = AutoTokenizer.from_pretrained(
            trainer.config.tokenizer.tokenizer_name_or_path
        )
        token_size = 4 if len(tokenizer) > np.iinfo(np.uint16).max + 1 else 2
        del tokenizer
        # Create Nanoset
        from nanotron.data.nanoset import Nanoset

        with main_rank_first(trainer.parallel_context.world_pg):
            train_dataset = Nanoset(
                dataset_folders=data.dataset.dataset_folder,
                dataset_weights=data.dataset.dataset_weights,
                sequence_length=trainer.sequence_length,
                token_size=token_size,
                train_split_num_samples=trainer.config.tokens.train_steps
                * trainer.global_batch_size,
                random_seed=data.seed,
            )

        # Prepare dataloader
        train_dataloader = build_nanoset_dataloader(
            train_dataset,
            trainer.sequence_length,
            parallel_context=trainer.parallel_context,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            micro_batch_size=trainer.micro_batch_size,
            consumed_train_samples=consumed_train_samples,
            dataloader_num_workers=data.num_loading_workers,
            dataloader_drop_last=True,
        )

        return train_dataloader
    else:
        raise ValueError(
            f"Unhandled case of `self.config.data.dataset`. Got: {data.dataset}"
        )

    return dataloader


def detect_validation_split(dataset_args: PretrainDatasetsArgs) -> str:
    """
    Auto-detect validation split name from dataset.
    Tries 'validation' > 'val' > 'test' in order.

    Args:
        dataset_args: Dataset configuration arguments

    Returns:
        str: Name of the validation split found, or None if not found
    """
    import os

    if dataset_args.streaming:
        # For streaming datasets, try to load and handle exceptions
        for split_name in ["validation", "val", "test"]:
            try:
                from datasets import load_dataset

                test_ds = load_dataset(
                    dataset_args.hf_dataset_or_datasets,
                    dataset_args.hf_dataset_config_name,
                    split=split_name,
                    streaming=True,
                )
                # If we can get an iterator, the split exists
                next(iter(test_ds))
                return split_name
            except Exception:
                continue
        return None
    else:
        # For non-streaming, check available splits
        if dataset_args.load_from_disk:
            base_path = dataset_args.hf_dataset_or_datasets
            for split_name in ["validation", "val", "test"]:
                split_path = os.path.join(base_path, split_name)
                if os.path.exists(split_path):
                    return split_name
        else:
            # Try loading each split
            for split_name in ["validation", "val", "test"]:
                try:
                    from datasets import load_dataset

                    load_dataset(
                        dataset_args.hf_dataset_or_datasets,
                        dataset_args.hf_dataset_config_name,
                        split=split_name,
                    )
                    return split_name
                except Exception:
                    continue
        return None


def get_validation_dataloader_from_data_stage(
    trainer: DistributedTrainer,
    data: DataArgs,
) -> DataLoader:
    """
    Returns a validation dataloader for a given data stage.

    Args:
        trainer: The distributed trainer instance
        data: The data configuration for the current stage

    Returns:
        Optional[DataLoader]: Validation dataloader or None if no validation split found
    """
    # First check if validation is enabled in config
    if (
        trainer.config.tokens.val_check_interval <= 0
        or trainer.config.tokens.limit_val_batches <= 0
    ):
        log_rank(
            "Validation disabled: val_check_interval or limit_val_batches not set properly",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )
        return None

    # Get input/output ranks
    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)

    # Case 1: Dummy data generator - no validation
    if data.dataset is None:
        log_rank(
            "Dummy data generator: no validation dataset",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )
        return None

    # Case 2: HuggingFace datasets
    elif isinstance(data.dataset, PretrainDatasetsArgs):
        # Detect validation split
        val_split = detect_validation_split(data.dataset)
        if val_split is None:
            log_rank(
                "No validation split found in dataset (tried 'validation', 'val', 'test')",
                logger=logger,
                level=logging.WARNING,
                rank=0,
            )
            return None

        log_rank(
            f"Using validation split: '{val_split}'",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        tokenizer_path = trainer.config.tokenizer.tokenizer_name_or_path

        with main_rank_first(trainer.parallel_context.world_pg):
            # Load validation dataset
            if data.dataset.load_from_disk:
                val_dataset = get_datasets_from_disk(
                    hf_dataset_path=data.dataset.hf_dataset_or_datasets,
                    splits=val_split,
                )[val_split]
            else:
                raw_val_dataset = get_datasets(
                    hf_dataset_or_datasets=data.dataset.hf_dataset_or_datasets,
                    hf_dataset_config_name=data.dataset.hf_dataset_config_name,
                    splits=val_split,
                    streaming=data.dataset.streaming,
                )[val_split]

                # Tokenize validation dataset
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "left"

                val_dataset = clm_process(
                    raw_dataset=raw_val_dataset,
                    tokenizer=tokenizer,
                    text_column_name=data.dataset.text_column_name,
                    dataset_processing_num_proc_per_process=data.dataset.dataset_processing_num_proc_per_process,
                    dataset_overwrite_cache=data.dataset.dataset_overwrite_cache,
                    sequence_length=trainer.sequence_length,
                )

            # Create validation dataloader (no consumed_samples, always start from beginning)
            val_dataloader = get_train_dataloader(
                train_dataset=val_dataset,
                sequence_length=trainer.sequence_length,
                parallel_context=trainer.parallel_context,
                input_pp_rank=input_pp_rank,
                output_pp_rank=output_pp_rank,
                micro_batch_size=trainer.micro_batch_size,
                consumed_train_samples=0,  # Always start from beginning for validation
                dataloader_num_workers=data.num_loading_workers,
                seed_worker=data.seed,
                dataloader_drop_last=True,
            )

            return val_dataloader

    # Case 3: Nanosets - not yet supported
    elif isinstance(data.dataset, NanosetDatasetsArgs):
        log_rank(
            "Nanoset datasets: validation not yet supported for Nanosets",
            logger=logger,
            level=logging.WARNING,
            rank=0,
        )
        return None

    return None


def get_dataloader(trainer: DistributedTrainer) -> Dict[str, Dict[str, DataLoader]]:
    """
    Returns a dictionary with train and optional validation dataloaders.

    Structure:
    {
        "stage_name": {
            "train": <train_dataloader>,
            "validation": <val_dataloader> or None
        },
        ...
    }
    """
    dataloaders = {}

    for stage_idx, stage in enumerate(trainer.config.data_stages):
        # NOTE: we only create the dataloader for the first stage,
        # then we lazy initialize the dataloader for the other stages
        stage = cast(DatasetStageArgs, stage)
        consumed_train_samples = get_consumed_train_samples_of_a_data_stage_from_ckp(
            stage, trainer.metadata
        )
        assert (
            consumed_train_samples is not None
        ), f"Cannot find consumed_train_samples for stage {stage.start_training_step} in the checkpoint"

        num_remaining_train_steps = compute_remain_train_steps_of_a_data_stage_from_ckp(
            stage, trainer.config, trainer.metadata
        )
        log_rank(
            f"[Training Plan] Stage {stage.name} has {num_remaining_train_steps} remaining training steps and has consumed {consumed_train_samples} samples",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        # Create training dataloader (lazy or immediate)
        train_dataloader = (
            get_dataloader_from_data_stage(
                trainer,
                stage.data,
                consumed_train_samples=consumed_train_samples,
                num_remaining_train_steps=num_remaining_train_steps,
            )
            if stage_idx == 0
            else lambda stage=stage, consumed=consumed_train_samples, remaining=num_remaining_train_steps: get_dataloader_from_data_stage(
                trainer,
                stage.data,
                consumed_train_samples=consumed,
                num_remaining_train_steps=remaining,
            )
        )

        # Create validation dataloader (only for first stage, lazy for others)
        val_dataloader = (
            get_validation_dataloader_from_data_stage(trainer, stage.data)
            if stage_idx == 0
            else lambda stage=stage: get_validation_dataloader_from_data_stage(
                trainer, stage.data
            )
        )

        dataloaders[stage.name] = {
            "train": train_dataloader,
            "validation": val_dataloader,
        }
    return dataloaders


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file

    # Load config from file
    config = get_config_from_file(config_file)

    # Apply command line overrides
    config = apply_config_overrides(config, args)

    # Apply generic dotted-path overrides
    if args.override:
        from nanotron.config.config import apply_generic_overrides
        config = apply_generic_overrides(config, args.override)

    # Load trainer and data
    trainer = DistributedTrainer(config)
    dataloader = get_dataloader(trainer)

    # Train
    trainer.train(dataloader)
