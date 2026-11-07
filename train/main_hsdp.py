import os
import sys
import time
import json
import random
import argparse
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.distributed.checkpoint as dcp
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM
from transformers import default_data_collator
import datasets
import datasets.distributed
import wandb
from loguru import logger
from pretraining_utils import training_utils, args_utils
from pretraining_utils.dataloader import PreprocessedIterableDataset
import datetime, pdb, pickle
from torch.profiler import profile, ProfilerActivity

try:
    import ulfm_collectives as ULFM
    from ulfm_collectives.training_manager import ULFMTrainingManager
    from ulfm_collectives.failure_simulator import FailureSimulator, set_failure_simulator
    from ulfm_collectives.failure import (
        FailureSchedule,
        ParallelismSpec,
        generate as generate_failure_schedule,
    )
    from ulfm_collectives.hsdp_groups import (
        compute_hsdp_layout,
        replica_ranks,
        replica0_ranks,
        replicate_peer_ranks,
    )
    from ulfm_collectives.hsdp_training_manager import HSDPULFMTrainingManager
    from ulfm_collectives.side_collectives import ulfm_all_reduce_isolated
    from ulfm_collectives.policy import GradRestoreMode
    _ULFM_AVAILABLE = True
except ImportError:
    _ULFM_AVAILABLE = False

transformers.logging.set_verbosity_error()


torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

import math


def _nanotron_style_clip_grad_norm(parameters, max_norm, pg, norm_type=2.0):
    """Nanotron's clip_grad_norm pattern.

    Pre-condition: gradients are already DP-synced — every rank's grad shard
    matches its DP-counterpart's. The remaining axis to reduce over is the
    parameter-sharding axis (mp_pg in nanotron / shard_pg in HSDP).

    For norm_type==2:
      1. Each rank computes Σ_p ||p.grad||² over its local shard of params.
      2. dist.all_reduce(SUM, group=pg) → global Σ ||p.grad||².
      3. sqrt → true global L2 norm.
      4. Apply uniform clip_coef = max_norm / total_norm to every grad on
         this rank. Same coef every rank → preserves global gradient direction.
    """
    parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    device = grads[0].device if grads else torch.device("cuda")

    if norm_type == math.inf:
        if grads:
            local_norm = torch.max(torch.stack([
                torch.linalg.vector_norm(g.detach(), ord=math.inf, dtype=torch.float32)
                for g in grads
            ]))
        else:
            local_norm = torch.zeros([], dtype=torch.float32, device=device)
        if pg is not None:
            dist.all_reduce(local_norm, op=dist.ReduceOp.MAX, group=pg)
        total_norm = local_norm
    else:
        if grads:
            local_pow = torch.linalg.vector_norm(
                torch.stack([
                    torch.linalg.vector_norm(
                        g.detach(), ord=norm_type, dtype=torch.float32
                    )
                    for g in grads
                ]),
                ord=norm_type,
                dtype=torch.float32,
            ).pow(norm_type)
        else:
            local_pow = torch.zeros([], dtype=torch.float32, device=device)
        if pg is not None:
            dist.all_reduce(local_pow, op=dist.ReduceOp.SUM, group=pg)
        total_norm = local_pow.pow(1.0 / norm_type)

    if max_norm is not None and max_norm != float("inf"):
        clip_coef = max_norm / (total_norm + 1.0e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for g in grads:
            g.detach().mul_(clip_coef_clamped.to(g.device))

    return total_norm


class LMWrapper(nn.Module):
    """Wraps a causal LM to accept a single batch dict, as required by ULFMTrainingManager."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model(**batch)


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, default=None, help="W&B run name (default: the model config's file name)")
    parser.add_argument("--wandb_project", type=str, default="ulfm", help="W&B project")
    parser.add_argument("--model_config", type=str, required=True, help="Llama model config (JSON), e.g. configs/llama_60m.json")
    parser.add_argument("--offline_mode", default=False, action="store_true", help="read the pre-tokenized dataset from --dataset_path instead of streaming allenai/c4 from the Hugging Face Hub")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.environ.get("RECOVER_DATASET_PATH"),
        help="Pre-tokenized C4 (seq_len 4096) saved with datasets.save_to_disk; "
             "required with --offline_mode. Defaults to $RECOVER_DATASET_PATH.",
    )
    parser.add_argument("--continue_from", type=str, default=None, help="resume from this checkpoint directory (<save_dir>/model_<step>)")
    parser.add_argument("--batch_size", type=int, required=True, help="microbatch size per GPU, in sequences")
    parser.add_argument("--gradient_accumulation", type=int, default=1, help="microbatches per GPU per update step. Derived from --total_batch_size when that is given")
    parser.add_argument("--total_batch_size", type=int, default=None, help="global batch per update step, in sequences: batch_size x gradient_accumulation x world size")
    parser.add_argument("--max_length", type=int, default=256, help="sequence length in tokens. With --offline_mode, the 4096-token sequences are cut to this length")
    parser.add_argument("--optimizer", default="adamw", help="adam, adamw, sgd, adafactor, adam8bit or adam8bit_per_layer")
    parser.add_argument("--lr", type=float, default=1e-4, help="peak learning rate")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_restarts"],
        help="learning-rate schedule after warmup",
    )
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="final learning rate as a fraction of --lr")
    parser.add_argument("--activation_checkpointing", action="store_true", help="recompute activations in backward to save memory")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--warmup_steps", type=int, default=1_000, help="learning-rate warmup, in update steps")
    parser.add_argument("--eval_every", type=int, default=5_000, help="evaluate every N update steps")
    parser.add_argument(
        "--num_training_steps",
        type=int,
        default=10_000,
        help="Number of **update steps** to train for. "
        "Notice that gradient accumulation is taken into account.",
    )
    parser.add_argument(
        "--max_train_tokens",
        type=training_utils.max_train_tokens_to_number,
        default=None,
        help="Number of tokens to train on. Overwrites num_training_steps. "
        "You can use M and B suffixes, e.g. 100M or 1B.",
    )
    parser.add_argument("--save_every", type=int, default=10_000, help="save a checkpoint every N update steps")
    parser.add_argument("--save_dir", type=str, default=None, help="checkpoint directory (default: checkpoints/<model config>-<date and time>)")
    parser.add_argument("--tags", type=str, default=None, help="comma-separated W&B tags")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_bf16_supported() else "float32",
        help="model dtype. With --mixed_precision, the model stays in float32 and FSDP computes in bfloat16",
    )
    parser.add_argument("--workers", type=int, default=8, help="data-loader worker processes")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--name", type=str, default="test", help="not used")
    parser.add_argument("--grad_clipping", type=float, default=0.0, help="clip the gradient norm to this value. 0 disables clipping")
    # beta1 for adafactor
    parser.add_argument("--beta1", type=float, default=0.0, help="momentum for sgd, beta1 for adafactor")
    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true", help="train on one GPU without torch.distributed (no fault tolerance)")
    parser.add_argument(
        "--backend",
        type=str,
        default="ulfm",
        choices=["nccl", "ulfm"],
        help="Cross-replica backend: 'nccl' (baseline) or 'ulfm' (fault-tolerant).",
    )
    parser.add_argument(
        "--mixed_precision",
        default=False,
        action="store_true",
        help="Enable FSDP native mixed precision: bf16 compute, fp32 reduce, "
             "fp32 master weights. Requires the model to remain in fp32; "
             "when set, the --dtype bf16 cast below is skipped.",
    )
    parser.add_argument(
        "--hsdp_shard_size",
        type=int,
        default=None,
        help="Ranks per FSDP replica. Default: torch.cuda.device_count(). "
             "world_size must be divisible by this value. With --backend nccl it "
             "must be 2 or more (FSDP1 bug pytorch/pytorch#154888).",
    )
    # --------------------------------------------------------------
    # Failure schedule (deterministic replica-aware).
    # Either load a pre-generated YAML via --failure_schedule, or generate
    # inline with --failure_count + --failure_step_range + --failure_locations.
    # The two are mutually exclusive. If neither is given, no failures are injected.
    # --------------------------------------------------------------
    parser.add_argument(
        "--failure_schedule",
        type=str,
        default=None,
        help="Path to a YAML failure schedule (see scripts/generate_failure_schedule.py).",
    )
    parser.add_argument(
        "--failure_count",
        type=int,
        default=0,
        help="Inline generator: number of replicas to kill. 0 disables injection.",
    )
    parser.add_argument(
        "--failure_seed",
        type=int,
        default=0,
        help="Inline generator seed. All ranks must pass the same value.",
    )
    parser.add_argument(
        "--failure_step_range",
        type=int,
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help="Inline generator: half-open [START, END) minibatch index range for kills.",
    )
    parser.add_argument(
        "--failure_sampling",
        choices=("iid", "stratified"),
        default="stratified",
        help="Inline generator: step-sampling mode.",
    )
    parser.add_argument(
        "--failure_locations",
        nargs="+",
        default=None,
        metavar="NAME:WEIGHT",
        help="Inline generator: location:weight pairs (e.g. grad-sync:first:0.6 forward:0.4). "
             "Locations: forward, backward, grad-sync:first, grad-sync:mid, grad-sync:last, optimizer (see ulfm_collectives/failure/locations.py).",
    )
    parser.add_argument(
        "--ulfm_verbose",
        default=False,
        action="store_true",
        help="Enable verbose ULFM logging: turns on ProcessGroupULFM C++ "
             "verbose mode (set_ulfm_verbose_logging(True)) and bumps the "
             "Python 'ulfm_collectives' logger to DEBUG. Mirrors nanotron's "
             "--ulfm-verbose flag.",
    )
    parser.add_argument(
        "--failure_exclude_replicas",
        type=str,
        default="0",
        help="Comma-separated replica ids to spare from selection "
             "(default '0' spares the replica that contains global rank 0, "
             "which hosts wandb). Pass empty string to allow all replicas.",
    )

    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)
    return args


@torch.no_grad()
def evaluate_model(
    model,
    preprocess_batched,
    pad_idx,
    global_rank,
    world_size,
    device,
    batch_size,
    dataloader=None,
    shard_pg=None,
    replicate_pg=None,
    num_live_replicas=None,
):
    _time = time.time()
    if dataloader is None:
        val_data = datasets.load_dataset(
            "allenai/c4", "en", split="validation", streaming=True
        )  # DGX
        val_data = val_data.shuffle(seed=42)
        if is_main_process():
            logger.info(
                f"Loaded validation dataset in {time.time() - _time:.2f} seconds"
            )

        if not args.single_gpu:
            val_data = datasets.distributed.split_dataset_by_node(
                val_data, rank=global_rank, world_size=world_size
            )

        val_data_mapped = val_data.map(
            preprocess_batched,
            batched=True,
            remove_columns=["text", "timestamp", "url"],
        )
        val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(
            val_data_mapped, batch_size
        )

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    if is_main_process():
        logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in (
        val_data_mapped.batch(batch_size=batch_size)
        if dataloader is None
        else dataloader
    ):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        batch["labels"] = (
            batch["input_ids"].clone() if "labels" not in batch else batch["labels"]
        )
        batch["labels"][batch["labels"] == pad_idx] = -100

        # Same call as the training loop: the HSDP model is
        # FSDP(LMWrapper(...)), whose forward takes the batch dict itself.
        if args.single_gpu:
            loss = model(**batch).loss
        else:
            loss = model(batch).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches

    if args.backend == "ulfm" and replicate_pg is not None and num_live_replicas:
        # After a failure the default (world) group still contains the dead
        # ranks, so a collective on it fails with MPIX_ERR_PROC_FAILED.
        # Reduce over the same groups as the training loop instead: shard_pg
        # (NCCL, within a replica) and replicate_pg (ULFM, repaired on
        # failure), then average over the replicas that are still alive.
        # replicate_pg here is the eval group (same ranks as the training
        # replicate_pg): a failure found by this reduction stays in the
        # training group for the next step (see side_collectives).
        dist.all_reduce(total_loss, op=dist.ReduceOp.AVG, group=shard_pg)
        ulfm_all_reduce_isolated(total_loss, dist.ReduceOp.SUM, replicate_pg)
        total_loss = total_loss.item() / replicate_pg.current_size()
    else:
        # Gather losses across all GPUs
        gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
        dist.all_gather(gathered_losses, total_loss)
        total_loss = sum([t.item() for t in gathered_losses]) / world_size

    return total_loss, evaluated_on_tokens


def hsdp_ulfm_is_saver(replica_id, replicate_pg, device):
    """ULFM only: True on the lowest-numbered replica that is still alive.

    Every replica holds a complete copy of the sharded model and optimizer,
    so the shards one replica writes over its shard_pg form a full
    checkpoint. Unlike a save over the default group, this still works after
    a failure (the default group keeps the dead ranks). The ULFM all-reduce
    also lines the surviving replicas up before the save.
    """
    lowest = torch.tensor([replica_id], dtype=torch.int64, device=device)
    # replicate_pg is the eval group (see evaluate_model).
    ulfm_all_reduce_isolated(lowest, dist.ReduceOp.MIN, replicate_pg)
    return int(lowest.item()) == replica_id

def build_hsdp_groups(world_size: int, shard_size: int, backend: str):
    """Build (shard_pg, replicate_pg, layout, gate_pg) for the current rank.

    shard_pg is always NCCL. replicate_pg matches `backend`
    ('nccl' or 'ulfm'). Every group is created on every rank (required by
    PyTorch dist.new_group), but each rank only belongs to one of each kind.
    """
    layout = compute_hsdp_layout(world_size=world_size, shard_size=shard_size)
    my_rank = dist.get_rank()

    _timeout_sec = os.getenv("NCCL_PG_TIMEOUT_SECONDS")
    _nccl_timeout = (
        datetime.timedelta(seconds=float(_timeout_sec))
        if _timeout_sec is not None
        else None
    )

    # --- Shard groups: one per replica ---
    shard_pg = None
    for rid in range(layout.num_replicas):
        ranks = replica_ranks(rid, shard_size)
        pg = dist.new_group(ranks=ranks, backend="nccl", timeout=_nccl_timeout)
        if my_rank in ranks:
            shard_pg = pg

    # --- Replicate groups: one per intra-replica offset ---
    replicate_pg = None
    for offset in range(shard_size):
        ranks = replicate_peer_ranks(offset, shard_size, layout.num_replicas)
        # backend=None on the ULFM side inherits the world backend (ulfm);
        # backend='nccl' on the baseline side forces NCCL explicitly.
        if backend == "ulfm":
            pg = dist.new_group(ranks=ranks, backend=None)
        else:
            pg = dist.new_group(ranks=ranks, backend="nccl", timeout=_nccl_timeout)
        if my_rank in ranks:
            replicate_pg = pg

    # --- Replica-consistency gate: shard_pg's ranks, longer timeout ---
    # Mirrors nanotron's mp_pg (NCCL_MP_TIMEOUT_SECONDS). After a failure
    # before the cross-replica reduction (forward, backward, optimizer), the
    # rank paired with the dead one learns of it at once and waits here for
    # its shard-mates, which stay in the ULFM all-reduce until the failed
    # replica's other ranks are gone: their NCCL_PG_TIMEOUT_SECONDS, the
    # watchdog's delay before abort, and process teardown. With shard_pg's
    # own timeout this rank times out first and takes its healthy replica
    # down. Falls back to NCCL_PG_TIMEOUT_SECONDS when unset.
    gate_pg = shard_pg
    if backend == "ulfm":
        _gate_sec = os.getenv("NCCL_MP_TIMEOUT_SECONDS")
        _gate_timeout = (
            datetime.timedelta(seconds=float(_gate_sec))
            if _gate_sec is not None
            else _nccl_timeout
        )
        for rid in range(layout.num_replicas):
            ranks = replica_ranks(rid, shard_size)
            pg = dist.new_group(ranks=ranks, backend="nccl", timeout=_gate_timeout)
            if my_rank in ranks:
                gate_pg = pg

    assert shard_pg is not None and replicate_pg is not None
    return shard_pg, replicate_pg, layout, gate_pg


def _build_failure_simulator(args, world_size: int, shard_size: int):
    """Return a configured FailureSimulator.

    Resolves the schedule from either --failure_schedule (YAML file) or the
    inline --failure_count / --failure_step_range / --failure_locations
    flags. The two sources are mutually exclusive. When no failures are
    requested, returns a simulator with an empty schedule so downstream
    ``begin_minibatch`` / ``may_fail_here`` calls become no-ops.
    """
    spec = ParallelismSpec(kind="hsdp", world_size=world_size, shard_size=shard_size)

    if args.failure_schedule and args.failure_count > 0:
        raise ValueError(
            "--failure_schedule and --failure_count are mutually exclusive; "
            "pick one source for the failure schedule."
        )

    if args.failure_schedule:
        schedule = FailureSchedule.load(args.failure_schedule)
        schedule.assert_matches_topology(spec)
        return FailureSimulator(schedule=schedule)

    if args.failure_count <= 0:
        empty = FailureSchedule(parallelism=spec, generator_config=None, entries=())
        return FailureSimulator(schedule=empty, enabled=False)

    if args.failure_step_range is None:
        raise ValueError(
            "--failure_count > 0 requires --failure_step_range START END."
        )
    if not args.failure_locations:
        raise ValueError(
            "--failure_count > 0 requires --failure_locations NAME:WEIGHT ..."
        )

    weights = {}
    for item in args.failure_locations:
        if ":" not in item:
            raise ValueError(
                f"--failure_locations entry {item!r} must be NAME:WEIGHT."
            )
        name, weight = item.rsplit(":", 1)
        weights[name] = float(weight)

    exclude_str = (args.failure_exclude_replicas or "").strip()
    exclude_ids = [int(x) for x in exclude_str.split(",") if x.strip()] if exclude_str else []

    start, end = args.failure_step_range
    schedule = generate_failure_schedule(
        parallelism=spec,
        seed=args.failure_seed,
        num_failures=args.failure_count,
        step_range=(start, end),
        location_weights=weights,
        sampling=args.failure_sampling,
        exclude_replica_ids=exclude_ids,
    )
    return FailureSimulator(schedule=schedule)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.backend == "ulfm":
        # PyTorch hardcodes a Store() placeholder for the ULFM backend and ignores
        # any store= argument. Work around: init first, then build a real TCPStore
        # using MPI barriers for coordination, then monkey-patch _world.pg_map so
        # subsequent new_group(backend="nccl") can exchange ncclUniqueIds.
        import datetime
        dist.init_process_group(backend="ulfm")
        _rank = dist.get_rank()
        _world_size = dist.get_world_size()
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = int(os.environ.get("MASTER_PORT", "29500"))
        if _rank == 0:
            store = dist.TCPStore(
                host_name=master_addr,
                port=master_port,
                world_size=_world_size,
                is_master=True,
                timeout=dist.default_pg_timeout,
                wait_for_workers=False,
            )
        dist.barrier()
        if _rank != 0:
            store = dist.TCPStore(
                host_name=master_addr,
                port=master_port,
                world_size=_world_size,
                is_master=False,
                timeout=dist.default_pg_timeout,
            )
        dist.barrier()
        import torch.distributed.distributed_c10d as _c10d
        _default_pg = _c10d._get_default_group()
        _backend_str, _ = _c10d._world.pg_map[_default_pg]
        _c10d._world.pg_map[_default_pg] = (_backend_str, store)
    else:
        dist.init_process_group(backend=args.backend)

    # assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    # global_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID")))
    # local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_PROCID")))
    # world_size = int(os.environ["WORLD_SIZE"])

    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK",
                                    str(global_rank % max(torch.cuda.device_count(), 1))))
    torch.cuda.set_device(local_rank)

    if not args.single_gpu:
        if args.hsdp_shard_size is None:
            shard_size = max(torch.cuda.device_count(), 1)
        else:
            shard_size = args.hsdp_shard_size
        if args.backend == "nccl" and shard_size == 1 and world_size > 1:
            # FSDP1 replaces HYBRID_SHARD with NO_SHARD when the shard group has one
            # process; NO_SHARD reduces only within that group, so the replicas would
            # train on unsynchronized gradients (pytorch/pytorch#154888). The ULFM path
            # reduces across replicas itself and is not affected.
            raise ValueError(
                "--backend nccl needs --hsdp_shard_size 2 or more. With one GPU per replica, "
                "FSDP1 no longer reduces gradients across replicas (pytorch/pytorch#154888)"
            )
        # Mirror nanotron's ParallelContext: build a derived world_pg via
        # dist.new_group(range(W), backend=...) — sibling of the default
        # group at MPI_COMM_WORLD, dedicated to world-level operations
        # (e.g. ULFM consensus). shard_pg / replicate_pg are also
        # MPI_COMM_WORLD-rooted siblings (PyTorch's new_group always
        # validates ranks against the default group).
        world_pg = dist.new_group(
            ranks=list(range(world_size)),
            backend=dist.get_backend(),
        )
        shard_pg, replicate_pg, hsdp_layout, gate_pg = build_hsdp_groups(
            world_size=world_size, shard_size=shard_size, backend=args.backend
        )
        # ULFM: a second group with replicate_pg's ranks, for the all-reduces
        # outside training steps (evaluation, saver election); see
        # ulfm_collectives.side_collectives.
        eval_replicate_pg = None
        if args.backend == "ulfm":
            for offset in range(shard_size):
                ranks = replicate_peer_ranks(offset, shard_size, hsdp_layout.num_replicas)
                pg = dist.new_group(ranks=ranks, backend=None)
                if global_rank in ranks:
                    eval_replicate_pg = pg
        logger.info(
            f"HSDP layout: num_replicas={hsdp_layout.num_replicas}, "
            f"shard_size={shard_size}, my replica={hsdp_layout.replica_id_of(global_rank)}, "
            f"my shard_rank={hsdp_layout.shard_rank_of(global_rank)}, my global_rank={global_rank}"
        )
    else:
        world_pg = None
        shard_pg = replicate_pg = hsdp_layout = gate_pg = eval_replicate_pg = None

    logger.info(
        f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}"
    )

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    if _ULFM_AVAILABLE and not args.single_gpu and args.backend == "ulfm":
        if args.ulfm_verbose:
            import logging as _logging
            ULFM.set_ulfm_verbose_logging(True)
            _logging.getLogger("ulfm_collectives").setLevel(_logging.DEBUG)
            if global_rank == 0:
                logger.info("ULFM verbose logging enabled (C++ + Python DEBUG)")
        sim = _build_failure_simulator(args, world_size=world_size, shard_size=shard_size)
        if sim is not None:
            if global_rank == 0:
                for _line in sim.describe().splitlines():
                    logger.info(_line)
            set_failure_simulator(sim)
            sim.initialize(rank=global_rank, world_size=world_size)
    else:
        sim = None

    if args.total_batch_size is not None:
        assert (
            args.total_batch_size % world_size == 0
        ), "total_batch_size must be divisible by world_size"
        args.gradient_accumulation = args.total_batch_size // (
            args.batch_size * world_size
        )
        if is_main_process():
            logger.info(
                f"{args.gradient_accumulation}-{world_size}-{args.total_batch_size}-{args.batch_size}"
            )
        assert (
            args.gradient_accumulation > 0
        ), "gradient_accumulation must be greater than 0"

    assert (
        args.gradient_accumulation * args.batch_size * world_size
        == args.total_batch_size
    ), "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # Loguru is used for script-level messages (loss / update step). Only the
    # lead rank prints them, so the "Update step / loss" line prints once:
    # global rank 0, or with ULFM once it has failed, the first shard of the
    # lowest live replica. The lead also writes to wandb if it has a run
    # (wandb.init runs on global rank 0 only).
    if args.single_gpu or args.backend != "ulfm":
        is_lead = lambda: global_rank == 0
    else:
        # current_rank() is the rank in replicate_pg as repaired after
        # failures, where the survivors keep their order: 0 is the lowest
        # live replica.
        is_lead = lambda: (
            hsdp_layout.shard_rank_of(global_rank) == 0
            and replicate_pg.current_rank() == 0
        )
    logger.remove()
    logger.add(sys.stderr, filter=lambda _record: is_lead())

    def writes_metadata(is_saver: bool) -> bool:
        """The rank that writes a checkpoint's metadata (scheduler, steps):
        with ULFM the first shard of the replica that wrote the shards (see
        hsdp_ulfm_is_saver), which is not global rank 0 once that has
        failed; else global rank 0."""
        if args.single_gpu or args.backend != "ulfm":
            return global_rank == 0
        return is_saver and hsdp_layout.shard_rank_of(global_rank) == 0

    # Stdlib logging is used by ulfm_collectives (orchestrator,
    # training_manager, hooks). Configure the root logger on EVERY rank with
    # an INFO-level StreamHandler so messages like "Restore completed",
    # "Communicator repaired", etc. surface from every rank that emits them.
    # Without this, stdlib's lastResort handler suppresses everything below
    # WARNING. --ulfm_verbose later may further bump ulfm_collectives to DEBUG.
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format=f"%(asctime)s [%(levelname)s|rank{global_rank}] %(name)s: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    if global_rank == 0:
        model_name = args.model_config.split("/")[1]
        model_name = model_name.split(".")[0]
        run_name = (
            f"{model_name}"
            if args.run_name is None
            else args.run_name
        )
        wandb.init(project=args.wandb_project, name=run_name)

    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    if args.offline_mode:
        if args.dataset_path is None:
            raise ValueError("--offline_mode needs --dataset_path (or $RECOVER_DATASET_PATH)")
        logger.info(f"Loading tokenized data from {args.dataset_path}")
        data = datasets.load_from_disk(args.dataset_path)
        logger.info("Finished loading from disk")
    else:
        data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)

        seed_for_shuffle = 42
        logger.info(f"Shuffling data with seed {seed_for_shuffle}")
        data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)

    if not args.single_gpu:
        if args.offline_mode:
            train_data: datasets.Dataset = data["train"]
            train_data = datasets.distributed.split_dataset_by_node(
                train_data,
                rank=global_rank,
                world_size=world_size,
            )
            eval_data = data["validation"]
            eval_data = datasets.distributed.split_dataset_by_node(
                eval_data,
                rank=global_rank,
                world_size=world_size,
            )
        else:
            data = datasets.distributed.split_dataset_by_node(
                data,
                rank=global_rank,
                world_size=world_size,
            )

    # tokenizer = AutoTokenizer.from_pretrained(
    #     "t5-base", model_max_length=args.max_length
    # )

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    if args.offline_mode:
        # The on-disk dataset is pre-tokenized at seq_len=4096. If the user
        # wants a shorter --max_length, slice each example down to that length
        # at collate time — no reprocessing pass over the dataset needed.
        # If max_length >= the on-disk length, this is a no-op.
        def _truncating_collator(features):
            for f in features:
                for k in ("input_ids", "attention_mask", "labels", "token_type_ids"):
                    v = f.get(k)
                    if v is None:
                        continue
                    # Works for tensors, numpy arrays, and python lists.
                    if hasattr(v, "shape"):
                        if v.shape and v.shape[-1] > args.max_length:
                            f[k] = v[..., : args.max_length]
                    elif isinstance(v, (list, tuple)) and len(v) > args.max_length:
                        f[k] = v[: args.max_length]
            return default_data_collator(features)

        _collate = _truncating_collator if args.max_length < 4096 else default_data_collator

        train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            num_workers=args.workers,
            collate_fn=_collate,
            shuffle=True,
        )
        eval_dataloader = torch.utils.data.DataLoader(
            eval_data,
            batch_size=args.batch_size,
            num_workers=args.workers,
            collate_fn=_collate,
            shuffle=True,
        )
    # else:
    #     # it doesn't matter which tokenizer we use, because we train from scratch
    #     # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice

    #     dataset = PreprocessedIterableDataset(
    #         data, tokenizer, batch_size=args.batch_size, max_length=args.max_length
    #     )
    #     train_dataloader = torch.utils.data.DataLoader(
    #         dataset, batch_size=None, num_workers=args.workers
    #     )
    #     eval_dataloader = None

    # Always build the architecture from --model_config. When resuming, we
    # load weights AFTER FSDP wrapping via dcp.load (sharded distributed
    # checkpoint), since save_every now writes sharded checkpoints — they
    # are not directly loadable via HF from_pretrained.
    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(
            f"Will resume from sharded checkpoint at {args.continue_from} "
            "(architecture rebuilt from --model_config; weights loaded post-FSDP wrap)"
        )
    else:
        logger.info("*" * 40)
        logger.info("Building model from scratch")

    model_config = AutoConfig.from_pretrained(args.model_config)
    model = AutoModelForCausalLM.from_config(model_config)

    if args.dtype in ["bf16", "bfloat16"] and not args.mixed_precision:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    if args.activation_checkpointing:
        # Non-reentrant: FSDP post-backward hook fires once per flat_param per
        # backward. Reentrant mode can fire it multiple times, which breaks the
        # sync-step fold in _reduce_grad (would double-count on the 2nd fire).
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if not args.single_gpu:
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LlamaDecoderLayer},
        )
        mp_policy = None
        if args.mixed_precision:
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
                keep_low_precision_grads=False,
            )
        model = FullyShardedDataParallel(
            LMWrapper(model),
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            process_group=(shard_pg, replicate_pg),
            auto_wrap_policy=wrap_policy,
            device_id=local_rank,
            use_orig_params=True,
            mixed_precision=mp_policy,
        )

    # Resume model weights from a sharded checkpoint. dcp.load is a
    # collective — every rank must enter the state_dict_type and dcp.load
    # block. The pattern: build a template state_dict by reading the current
    # (freshly initialized) sharded params, hand it to dcp.load to overwrite
    # in-place from disk, then load_state_dict to reapply.
    if args.continue_from is not None:
        if args.single_gpu:
            sd = torch.load(
                os.path.join(args.continue_from, "pytorch_model.bin"),
                map_location="cpu",
            )
            model.load_state_dict(sd)
        else:
            with FullyShardedDataParallel.state_dict_type(
                model, StateDictType.SHARDED_STATE_DICT
            ):
                sharded_sd = {"model": model.state_dict()}
                dcp.load(
                    state_dict=sharded_sd,
                    storage_reader=dcp.FileSystemReader(args.continue_from),
                )
                model.load_state_dict(sharded_sd["model"])
        logger.info(f"Loaded sharded model weights from {args.continue_from}")

    global_step = 0
    update_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    # ====== starting config ======= #

    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    layer_wise_flag = True if "per_layer" in args.optimizer.lower() else False

    optimizer = training_utils.build_optimizer(model, trainable_params, args)
    if layer_wise_flag:
        if not isinstance(optimizer, dict):
            raise ValueError("Layer-wise optimizer is not properly constructed.")

    if not layer_wise_flag:
        scheduler = training_utils.get_scheculer(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )

    if args.continue_from is not None:
        optimizer_checkpoint = torch.load(
            os.path.join(args.continue_from, "optimizer.pt"), map_location="cpu"
        )
        scheduler.load_state_dict(optimizer_checkpoint["scheduler"])
        # Load sharded optimizer state via FSDP's distributed API. Each rank
        # reads its own shard from the dcp checkpoint dir; the result is
        # rekeyed to flat-param ids and applied via optimizer.load_state_dict.
        if args.single_gpu and "optimizer" in optimizer_checkpoint:
            optimizer.load_state_dict(optimizer_checkpoint["optimizer"])
        elif not args.single_gpu:
            with FullyShardedDataParallel.state_dict_type(
                model, StateDictType.SHARDED_STATE_DICT
            ):
                optim_template = FullyShardedDataParallel.optim_state_dict(
                    model, optimizer
                )
                state_dict_to_load = {"optim": optim_template}
                dcp.load(
                    state_dict=state_dict_to_load,
                    storage_reader=dcp.FileSystemReader(args.continue_from),
                )
                flattened_osd = FullyShardedDataParallel.optim_state_dict_to_load(
                    model, optimizer, state_dict_to_load["optim"]
                )
                optimizer.load_state_dict(flattened_osd)
        logger.info(f"Optimizer and scheduler restored from {args.continue_from}")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(
                f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}"
            )
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(
                f"Will train for {args.num_training_steps - update_step} update steps"
            )

    scheduler_start_step = update_step

    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(
        f"All params: \n{[n for n,p in model.named_parameters() if p.requires_grad]}\n"
    )
    logger.info(
        f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M"
    )
    logger.info(
        f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M"
    )
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")

    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update(
        {
            "max_lr": run_config.pop(
                "lr"
            ),  # rename lr to max_lr to avoid conflicts with scheduler
            "total_params_M": n_total_params / 1_000_000,
            "dataset": "allenai/c4",
            "model": model_config.to_dict(),
            "world_size": world_size,
            "device": str(device),
        }
    )

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now")  # save current script

    lm_criterion = lambda output, _: output.loss

    training_manager = None
    if not args.single_gpu and args.backend == "ulfm":
        if not _ULFM_AVAILABLE:
            raise RuntimeError(
                "ulfm_collectives not available; cannot run --backend ulfm without it."
            )
        training_manager = HSDPULFMTrainingManager(
            fsdp_model=model,
            replicate_pg=replicate_pg,
            world_pg=world_pg,
            grad_accum_steps=args.gradient_accumulation,
            policy_type="static",
            initial_world_size=hsdp_layout.num_replicas,
        )
        training_manager.txn.set_replica_map(hsdp_layout.replica_id_of, global_rank)

    # global steps and others are defined above
    pad_idx = -1
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step

    # ##############################
    # TRAINING LOOP
    # ##############################

    max_memory = torch.cuda.max_memory_allocated()
    if global_rank == 0:
        logger.info(f"Maximum memory allocated before training: {max_memory} bytes\n")
    torch.cuda.reset_peak_memory_stats()

    if not args.single_gpu:
        dist.barrier()

    if global_rank == 0:
        print(f"Rank {global_rank} starting training loop.")

    # Helper: bring a microbatch to device, set labels, count tokens.
    def _prepare_batch(batch):
        batch = {k: v.to(device) for k, v in batch.items()}
        batch["labels"] = (
            batch["input_ids"].clone() if "labels" not in batch else batch["labels"]
        )
        batch["labels"][batch["labels"] == pad_idx] = -100
        return batch

    # Outer loop is per UPDATE STEP; inner loop is gradient accumulation.
    # For ULFM, the inner microbatch loop is wrapped by a restore-mode loop
    # that may run extra microbatches at a policy boundary (mirrors nanotron).
    data_iter = iter(train_dataloader)
    batch_idx = -1  # cumulative microbatch counter (matches old `batch_idx`)

    # Resume: skip microbatches already seen in earlier update_steps.
    if update_step > 0:
        batches_to_skip = update_step * args.gradient_accumulation
        for _ in range(batches_to_skip):
            try:
                next(data_iter)
                batch_idx += 1
            except StopIteration:
                break

    while update_step < args.num_training_steps:
        # ============================================================
        # Single-GPU / NCCL backend: simple inner microbatch loop
        # ============================================================
        if args.single_gpu or args.backend == "nccl":
            loss = None
            # Already pre-scaled (loss / N) per microbatch — collect those so
            # the local sum is the per-rank mean over microbatches; then a
            # global AVG (SUM-allreduce / world_size) gives the DP-mean.
            losses_local_scaled = []
            data_exhausted = False
            for micro_idx in range(args.gradient_accumulation):
                try:
                    raw_batch = next(data_iter)
                except StopIteration:
                    data_exhausted = True
                    break
                batch_idx += 1
                global_step += 1
                local_step += 1
                batch = _prepare_batch(raw_batch)
                tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

                if args.single_gpu:
                    loss = model(**batch).loss
                else:
                    loss = model(batch).loss
                scaled_loss = loss / args.gradient_accumulation
                scaled_loss.backward()
                losses_local_scaled.append(scaled_loss.detach().to(dtype=torch.float32))

            if data_exhausted:
                break
            
            # Nanotron-style global L2 clip. Grads are already DP-synced
            # (FSDP HYBRID_SHARD sums cross-replica during backward); we just
            # need to all_reduce squared norms across shard_pg (the
            # parameter-sharding axis) so every rank gets the same global
            # total_norm and applies the same clip coefficient.
            if args.single_gpu:
                grad_norm = _nanotron_style_clip_grad_norm(
                    trainable_params,
                    args.grad_clipping if args.grad_clipping != 0.0 else float("inf"),
                    pg=None,
                )
            else:
                grad_norm = _nanotron_style_clip_grad_norm(
                    trainable_params,
                    args.grad_clipping if args.grad_clipping != 0.0 else float("inf"),
                    pg=shard_pg,
                )
            grad_norm = float(grad_norm)
            if not layer_wise_flag:
                optimizer.step()
                optimizer.zero_grad()

        # ============================================================
        # ULFM backend: nanotron-style restore-mode loop around the
        # microbatch loop. On failure, query orchestrator for extra
        # microbatches and run another pass before optimizer step.
        # ============================================================
        else:
            is_first_pass = True
            loss = None
            # ULFM path doesn't pre-scale loss in microbatch_step. Collect
            # RAW per-microbatch losses across all passes (initial + extended);
            # later we SUM-allreduce across world and divide by total
            # microbatches = sum-of-(n_micro per pass) × live_world.
            losses_local_raw = []
            data_exhausted = False

            sim.begin_minibatch(update_step)
            while True:
                if is_first_pass:
                    n_micro = training_manager.get_effective_n_microbatches()
                else:
                    n_micro = training_manager.get_n_extra_microbatches()

                training_manager.prepare_iteration(is_first_pass=is_first_pass)
                world_work = training_manager.on_world_consensus()
                # Extended pass: take in failures the world consensus found
                # that this replicate group's decision does not include.
                if not is_first_pass and training_manager.reconcile_extended_pass(world_work):
                    n_micro = training_manager.get_n_extra_microbatches()

                # Inner microbatch loop (forward + backward only)
                for micro_idx in range(n_micro):
                    try:
                        raw_batch = next(data_iter)
                    except StopIteration:
                        data_exhausted = True
                        break
                    batch_idx += 1
                    global_step += 1
                    local_step += 1
                    batch = _prepare_batch(raw_batch)
                    tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

                    # Extended pass: wait for non-blocking restore before the
                    # first microbatch's backward.
                    if not is_first_pass and micro_idx == 0:
                        training_manager.wait_restore_before_backward()

                    with sim.may_fail_here("forward"):
                        loss = training_manager.microbatch_step(
                            batch_idx, micro_idx, n_micro,
                            batch, None, lm_criterion,
                        )
                    losses_local_raw.append(
                        torch.tensor(float(loss), device=device, dtype=torch.float32)
                    )

                if data_exhausted:
                    break

                is_first_pass = False

                # Cross-replica allreduce on each FSDP unit's grad shard.
                training_manager.fire_cross_replica_allreduces()
                # Intra-replica barrier: ensure all shard-mates within a
                # replica observe the same outcome of the cross-replica
                # reduce before consensus / restore-mode dispatch. NCCL
                # collective on gate_pg (shard_pg's ranks, longer timeout;
                # see build_hsdp_groups), no MPI traffic.
                if gate_pg is not None:
                    dist.barrier(group=gate_pg, device_ids=[torch.cuda.current_device()])
                training_manager.on_consensus_step()

                mode = training_manager.get_restore_mode()
                if mode == GradRestoreMode.SKIP:
                    break  # success, proceed to optimizer step
                if mode == GradRestoreMode.NON_BLOCKING:
                    # Policy boundary: async restore + extra microbatches.
                    training_manager.start_nonblocking_restore()
                    continue

                # BLOCKING: blocking restore (re-reduce) loop
                crossed_boundary = False
                blocking_attempts = 0
                while training_manager.get_restore_mode() == GradRestoreMode.BLOCKING:
                    training_manager.start_blocking_restore()
                    blocking_attempts += 1
                    training_manager.on_world_consensus()

                    if gate_pg is not None:
                        dist.barrier(group=gate_pg, device_ids=[torch.cuda.current_device()])
                    training_manager.on_consensus_step()

                    if training_manager.is_at_policy_boundary():
                        training_manager.start_nonblocking_restore()
                        crossed_boundary = True
                        break
                    elif training_manager.get_restore_mode() == GradRestoreMode.BLOCKING:
                        logger.warning(f"[Rank {global_rank}] Failure happend during re-reduction; retrying")
                        if blocking_attempts > 3:
                            raise RuntimeError(
                                f"[Rank {global_rank}] Blocking restore retry limit exceeded"
                            )
                        else:
                            continue  # retry blocking restore
                if crossed_boundary:
                    continue  # outer: run extra microbatches
                break  # restored without crossing boundary, proceed

            if data_exhausted:
                break

            # Post-loop: normalize, clip, optimizer step.
            # Nanotron-style global L2 clip on shard_pg. Grads are already
            # DP-synced (fire_cross_replica_allreduces + normalize_gradients).
            training_manager.normalize_gradients()
            grad_norm = _nanotron_style_clip_grad_norm(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clipping if args.grad_clipping != 0.0 else float("inf"),
                pg=shard_pg,
            )
            grad_norm = float(grad_norm)
        
        # DP-averaged loss for display (mirrors nanotron's smoothness).
        # Two paths to the same result:
        #   NCCL path: scaled_loss = loss / N is already collected, so the
        #     local sum IS the per-rank mean over microbatches. SUM-allreduce
        #     then divide by live_world = global mean.
        #   ULFM path: raw losses collected (no pre-scale). Sum locally, then
        #     SUM-allreduce, then divide by total microbatches across the
        #     world = (n_micro_processed × live_world).
        # Use SUM + manual divide so MPI/ULFM (which lacks ReduceOp.AVG) works.
        if args.single_gpu:
            if losses_local_scaled:
                loss_for_log = float(torch.stack(losses_local_scaled).sum())
            else:
                loss_for_log = 0.0
        elif args.backend == "nccl":
            if losses_local_scaled:
                _local_sum = torch.stack(losses_local_scaled).sum().to(
                    device=device, dtype=torch.float32
                )
                dist.all_reduce(_local_sum, op=dist.ReduceOp.SUM)
                loss_for_log = float(_local_sum) / world_size
            else:
                loss_for_log = 0.0
        else:  
            # ulfm — constant divisor (initial total microbatches across
            # world). Mirrors the constant grad-divisor philosophy: failure
            # recovery doesn't rescale; surviving ranks contribute, missing
            # ranks contribute 0, displayed loss stays directly comparable
            # to the failure-free / NCCL run.
            if losses_local_raw:
                _local_sum = torch.stack(losses_local_raw).sum().to(
                    device=device, dtype=torch.float32
                )
                dist.all_reduce(_local_sum, op=dist.ReduceOp.AVG, group=shard_pg)
                dist.ulfm_all_reduce(_local_sum, op=dist.ReduceOp.SUM, group=replicate_pg)
                loss_for_log = float(_local_sum) / training_manager._get_grad_div_factor()
                
                training_manager.optimizer_step(optimizer)
            else:
                loss_for_log = 0.0

        if is_lead():
            logger.info(
                f"Update step {update_step}/{args.num_training_steps}, global step {global_step}, loss: {loss_for_log:.4f}"
            )

        if not layer_wise_flag:
            scheduler.step()

        update_step += 1
        # Match nanotron: sync GPU before timing so iter_time includes all
        # outstanding kernels (FSDP reshard, optimizer step, etc.).
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        update_time = time.time() - update_time

        # save checkpoint by save_every
        # Sharded distributed checkpoint: each rank writes its own shard
        # to the same directory. No single rank holds the full model in
        # memory. Resume via dcp.load with the same FSDP wrapping.
        if (
            local_step > args.gradient_accumulation
            and update_step % args.save_every == 0
        ):
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            if is_lead():
                logger.info(
                    f"Saving sharded model checkpoint to {current_model_directory}, update step {update_step}"
                )
                os.makedirs(current_model_directory, exist_ok=True)
            is_saver = False
            if not args.single_gpu:
                if args.backend == "ulfm":
                    is_saver = hsdp_ulfm_is_saver(
                        hsdp_layout.replica_id_of(global_rank), eval_replicate_pg, device
                    )
                    if is_saver:
                        os.makedirs(current_model_directory, exist_ok=True)
                else:
                    dist.barrier()  # ensure dir exists before all ranks write

            if args.single_gpu:
                # No FSDP — single rank, save HF-style.
                model.save_pretrained(
                    current_model_directory, max_shard_size="100GB"
                )
            else:
                # All ranks must enter state_dict_type, optim_state_dict, and
                # dcp.save (collective). Sharded model + sharded optimizer.
                with FullyShardedDataParallel.state_dict_type(
                    model, StateDictType.SHARDED_STATE_DICT
                ):
                    sharded_sd = {
                        "model": model.state_dict(),
                        "optim": FullyShardedDataParallel.optim_state_dict(
                            model,
                            optimizer,
                            # ULFM: its key merge must not use the default
                            # group, which may hold dead ranks.
                            group=shard_pg if args.backend == "ulfm" else None,
                        ),
                    }
                if args.backend == "ulfm":
                    # One surviving replica writes over its shard_pg (see
                    # hsdp_ulfm_is_saver); the others wait for it.
                    if is_saver:
                        dcp.save(
                            state_dict=sharded_sd,
                            storage_writer=dcp.FileSystemWriter(current_model_directory),
                            process_group=shard_pg,
                        )
                    dist.ulfm_all_reduce(
                        torch.zeros(1, device=device), group=replicate_pg
                    )
                else:
                    dcp.save(
                        state_dict=sharded_sd,
                        storage_writer=dcp.FileSystemWriter(current_model_directory),
                    )

            # Scheduler + training metadata are small: one rank writes them,
            # the first shard of the saving replica with ULFM, else rank 0.
            # Optimizer state lives in the sharded dcp checkpoint above.
            if writes_metadata(is_saver):
                optimizer_checkpoint = {
                    "scheduler": scheduler.state_dict(),
                    "update_step": update_step,
                    "global_step": global_step,
                    "config": run_config,
                    "wandb": wandb.run.dir if wandb.run is not None else None,
                    "dtype": args.dtype,
                }
                torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

                training_state_checkpoint = {
                    "global_step": global_step,
                    "update_step": update_step,
                    "tokens_seen": tokens_seen,
                    "tokens_seen_before": tokens_seen_before,
                    "update_time": update_time,
                }
                with open(f"{current_model_directory}/training_state.json", "w") as f:
                    json.dump(training_state_checkpoint, f, indent=4)

                # save wandb related info
                if wandb.run is not None:
                    wandb_info = {
                        "wandb_id": wandb.run.id,
                    }
                    with open(f"{args.save_dir}/wandb.json", "w") as f:
                        json.dump(wandb_info, f, indent=4)

        # evaluation
        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            model.eval()
            total_loss, evaluated_on_tokens = evaluate_model(
                model,
                preprocess_batched,
                pad_idx,
                global_rank,
                world_size,
                device,
                args.batch_size,
                eval_dataloader,
                shard_pg=shard_pg,
                replicate_pg=(eval_replicate_pg if args.backend == "ulfm" else replicate_pg),
                num_live_replicas=(
                    training_manager.txn.curr_world_size
                    if args.backend == "ulfm" and training_manager is not None
                    else None
                ),
            )
            if is_lead() and wandb.run is not None:
                wandb.log(
                    {
                        "final_eval_loss": total_loss,
                        "final_eval_perplexity": np.exp(total_loss),
                        "final_eval_tokens": evaluated_on_tokens,
                    },
                    step=global_step,
                )
            logger.info(
                f"Eval loss and perplexity at step {update_step}: {total_loss}, {np.exp(total_loss)}"
            )
            model.train()

        if not layer_wise_flag:
            lr = optimizer.param_groups[0]["lr"]
        else:
            lr = list(optimizer.values())[0].param_groups[0]["lr"]
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        max_memory = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        # Nanotron-style throughput. Uses global_batch_size × seq_len (no pad
        # adjustment) so values are directly comparable with nanotron runs.
        elapsed_time_per_iteration_ms = update_time * 1000.0
        tokens_per_iter = args.total_batch_size * args.max_length
        tokens_per_sec = tokens_per_iter / update_time if update_time > 0 else 0.0
        consumed_tokens = update_step * tokens_per_iter
        # Live total GPUs: for ULFM, replicate_pg shrinks under failures while
        # shard_pg stays stable (failures are per-replica). For NCCL it's static.
        if args.backend == "ulfm" and training_manager is not None:
            total_gpus = training_manager.txn.curr_world_size * shard_size
        else:
            total_gpus = world_size
        tokens_per_sec_per_gpu = tokens_per_sec / total_gpus if total_gpus > 0 else 0.0

        # ULFM-specific metrics (mirrors nanotron_ulfm trainer_ulfm.py:498-513).
        ulfm_metrics = {}
        if args.backend == "ulfm" and training_manager is not None:
            txn = training_manager.txn
            curr_grad_accum = txn.curr_grad_accum_steps
            curr_dp_size = txn.curr_world_size
            gbs = txn.effective_batch_size
            total_workload = curr_dp_size * curr_grad_accum
            redundancy = (
                1.0 - (gbs / total_workload) if total_workload > 0 else 0.0
            )
            ulfm_metrics = {
                "ulfm/num_majors": txn.num_major_procs,
                "ulfm/num_minors": txn.num_minor_procs,
                "ulfm/num_major_spares": txn.num_major_spare_procs,
                "ulfm/num_minor_spares": txn.num_minor_spare_procs,
                "ulfm/curr_grad_accum": curr_grad_accum,
                "ulfm/curr_minor_grad_accum": txn.minor_proc_grad_accum_steps,
                "ulfm/redundancy": redundancy,
            }

        if is_lead() and wandb.run is not None:
            wandb.log(
                {
                    "loss": loss_for_log,
                    "lr": lr,
                    "update_step": update_step,
                    "consumed_tokens": consumed_tokens,
                    "elapsed_time_per_iteration_ms": elapsed_time_per_iteration_ms,
                    "tokens_per_sec": tokens_per_sec,
                    "tokens_per_sec_per_gpu": tokens_per_sec_per_gpu,
                    "total_gpus": total_gpus,
                    "global_batch_size": args.total_batch_size,
                    "tokens_seen": tokens_seen,
                    "throughput_tokens": tokens_in_update / update_time,
                    "throughput_examples": args.total_batch_size / update_time,
                    "throughput_batches": batches_in_update / update_time,
                    "gradnorm": grad_norm,
                    "max_memory": max_memory,
                    **ulfm_metrics,
                },
                step=global_step,
            )

        update_time = time.time()

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")

    current_model_directory = f"{args.save_dir}/model_{update_step}"
    # Final sharded save: all ranks must enter the FSDP state_dict_type and
    # dcp.save (collective). The directory + log are the lead's (is_lead).
    save_final = not os.path.exists(current_model_directory)
    is_saver = False
    if not args.single_gpu:
        # Every rank must take the decision above before the lead creates the
        # directory. A rank that checked afterwards would skip the collective
        # save below while the others wait in it.
        if args.backend == "ulfm":
            # Not the default group, which may hold dead ranks: the ULFM
            # all-reduce lines up the replicas, shard_pg each replica.
            is_saver = hsdp_ulfm_is_saver(
                hsdp_layout.replica_id_of(global_rank), eval_replicate_pg, device
            )
            dist.barrier(group=shard_pg, device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()
    if save_final and is_lead():
        logger.info(
            f"Saving final sharded model checkpoint to {current_model_directory}, update step {update_step}"
        )
        os.makedirs(current_model_directory, exist_ok=True)
    if save_final and not args.single_gpu:
        if args.backend == "ulfm":
            if is_saver:
                os.makedirs(current_model_directory, exist_ok=True)
        else:
            dist.barrier()
        with FullyShardedDataParallel.state_dict_type(
            model, StateDictType.SHARDED_STATE_DICT
        ):
            sharded_sd = {
                "model": model.state_dict(),
                "optim": FullyShardedDataParallel.optim_state_dict(
                    model,
                    optimizer,
                    # ULFM: its key merge must not use the default group,
                    # which may hold dead ranks.
                    group=shard_pg if args.backend == "ulfm" else None,
                ),
            }
        if args.backend == "ulfm":
            # One surviving replica writes over its shard_pg (see
            # hsdp_ulfm_is_saver); the others wait for it.
            if is_saver:
                dcp.save(
                    state_dict=sharded_sd,
                    storage_writer=dcp.FileSystemWriter(current_model_directory),
                    process_group=shard_pg,
                )
            dist.ulfm_all_reduce(torch.zeros(1, device=device), group=replicate_pg)
        else:
            dcp.save(
                state_dict=sharded_sd,
                storage_writer=dcp.FileSystemWriter(current_model_directory),
            )
    elif save_final and args.single_gpu and global_rank == 0:
        model.save_pretrained(current_model_directory)

    if save_final and writes_metadata(is_saver):

        # Optimizer state lives in the sharded dcp checkpoint above; only
        # save scheduler + metadata here.
        optimizer_checkpoint = {
            "scheduler": scheduler.state_dict(),
            "update_step": update_step,
            "global_step": global_step,
            "config": run_config,
            "wandb": wandb.run.dir if wandb.run is not None else None,
            "dtype": args.dtype,
        }
        torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
        }
        with open(f"{current_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    total_loss, evaluated_on_tokens = evaluate_model(
        model,
        preprocess_batched,
        pad_idx,
        global_rank,
        world_size,
        device,
        args.batch_size,
        eval_dataloader,
        shard_pg=shard_pg,
        replicate_pg=(eval_replicate_pg if args.backend == "ulfm" else replicate_pg),
        num_live_replicas=(
            training_manager.txn.curr_world_size
            if args.backend == "ulfm" and training_manager is not None
            else None
        ),
    )

    if is_lead():
        if wandb.run is not None:
            wandb.log(
                {
                    "final_eval_loss": total_loss,
                    "final_eval_perplexity": np.exp(total_loss),
                    "final_eval_tokens": evaluated_on_tokens,
                },
                step=global_step,
            )
        logger.info(
            f"Eval loss and perplexity at step {update_step}: {total_loss}, {np.exp(total_loss)}"
        )

    logger.info("Script finished successfully")
    if is_lead():
        print("Finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    if args.backend == "ulfm":
        from ulfm_collectives.fail_stop import install_fail_stop

        install_fail_stop()
    main(args)
