import functools
import inspect
import os
import random
import socket
import argparse
from contextlib import ExitStack, contextmanager
from typing import ContextManager, List, Optional

import torch
from packaging import version
from torch import nn
from torch.utils.checkpoint import checkpoint

from nanotron import distributed as dist


class Singleton(type):
    """
    Singleton metaclass.
    Create objects using this class as the metaclass to enable singleton behaviour.
    For instance:
    ```
    class Logger(metaclass=Singleton):
      ...
    ```
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ContextManagers:
    """
    Wrapper for `contextlib.ExitStack` which enters a collection of context managers. Adaptation of `ContextManagers`
    in the `transformers` library.
    """

    def __init__(self, context_managers: List[ContextManager]):
        self.context_managers = context_managers
        self.stack = ExitStack()

    def __enter__(self):
        for context_manager in self.context_managers:
            self.stack.enter_context(context_manager)

    def __exit__(self, *args, **kwargs):
        self.stack.__exit__(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({[context_manager.gen.__qualname__ for context_manager in self.context_managers]})"


@contextmanager
def main_rank_first(group: dist.ProcessGroup):
    """Context manager that executes the code in the context with the rank zero of the group going first."""
    is_main = dist.get_rank(group) == 0
    if is_main:
        yield

    dist.barrier(group)

    if not is_main:
        yield


@contextmanager
def local_ranks_zero_first(group: Optional[dist.ProcessGroup] = None):
    """Context manager that executes the code in the context with all the local rank zero of the group going first.
    Useful to run only once per node first (e.g. to create local files, etc)
    """
    is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0
    if is_main:
        yield

    dist.barrier(group)

    if not is_main:
        yield


def checkpoint_method(attr_name: str):
    """Decorator to checkpoint a method of a class."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _self = args[0]
            checkpoint_activated = getattr(_self, attr_name)
            if checkpoint_activated:
                all_args = list(args)
                signature_params = inspect.signature(func).parameters
                # Parameters are ordered in the function definition order: https://docs.python.org/3/library/inspect.html#inspect.Signature.parameters
                for i, (arg_name, arg_value) in enumerate(signature_params.items()):
                    if arg_value.kind in [inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL]:
                        raise NotImplementedError(
                            "Checkpointing of functions with *args or **kwargs is not supported."
                        )
                    if i < len(args):
                        continue
                    if arg_name not in kwargs:
                        assert (
                            arg_value.default is not inspect.Parameter.empty
                        ), f"Missing argument {arg_name} from {kwargs} for {func.__name__}"
                        all_args.append(arg_value.default)
                    else:
                        all_args.append(kwargs[arg_name])
                assert len(all_args) == len(signature_params), f"Missing arguments for {func.__name__}"
                # TODO @nouamanetazi: we pass `self`(which is module) to checkpoint, so it's stored in `ctx.inputs` whereas some other methods create a custom fwd and pass only tensors without `self`. Need to investigate which is better
                return checkpoint(func, *all_args)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def get_parameter_and_parent_module(target: str, root_module: nn.Module):
    module_path, _, param_name = target.rpartition(".")

    mod: torch.nn.Module = root_module.get_submodule(module_path)

    if not hasattr(mod, param_name):
        raise AttributeError(mod._get_name() + " has no attribute `" + param_name + "`")

    param: torch.nn.Parameter = getattr(mod, param_name)

    if not isinstance(param, torch.nn.Parameter):
        raise AttributeError("`" + param_name + "` is not an " "nn.Parameter")

    return param, mod, param_name


def get_untyped_storage(tensor: torch.Tensor) -> torch.UntypedStorage:
    if version.parse(torch.__version__) >= version.parse("2.0"):
        return tensor.untyped_storage()
    else:
        return tensor.storage().untyped()


def tensor_from_untyped_storage(untyped_storage: torch.UntypedStorage, dtype: torch.dtype):
    # TODO @thomasw21: Figure out what's the best Pytorch way of building a tensor from a storage.
    device = untyped_storage.device
    tensor = torch.empty([], dtype=dtype, device=device)
    tensor.set_(source=untyped_storage)
    return tensor


def find_free_port(min_port: int = 2000, max_port: int = 65000) -> int:
    while True:
        port = random.randint(min_port, max_port)
        try:
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("localhost", port))
                return port
        except OSError:
            continue


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to the YAML or python config file",
    )

    # General config overrides
    parser.add_argument(
        "--run", type=str, default=None, help="Override run name (general.run)"
    )
    parser.add_argument(
        "--tag", type=str, default=None, help="Suffix for run name (general.tag)"
    )
    parser.add_argument(
        "--entity", type=str, default=None, help="Override wandb entity name (general.entity)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Override project name (general.project)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override random seed (general.seed)"
    )
    parser.add_argument(
        "--hf-dataset-or-datasets",
        type=str,
        nargs="+",
        default=None,
        help="Override HF dataset or datasets for each data stage (data_stages.data.dataset.hf_dataset_or_datasets)",
    )

    # Checkpoint config overrides
    parser.add_argument(
        "--checkpoints-path",
        type=str,
        default=None,
        help="Override checkpoint save path (checkpoints.checkpoints_path)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Override checkpoint interval (checkpoints.checkpoint_interval)",
    )
    parser.add_argument(
        "--resume-checkpoint-path",
        type=str,
        default=None,
        help="Override resume checkpoint path (checkpoints.resume_checkpoint_path)",
    )
    parser.add_argument(
        "--save-initial-state",
        action="store_true",
        default=None,
        help="Override save initial state (checkpoints.save_initial_state)",
    )
    parser.add_argument(
        "--save-final-state",
        action="store_true",
        default=None,
        help="Override save final state (checkpoints.save_final_state)",
    )

    # Optimizer config overrides
    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=None,
        help="Override learning rate (optimizer.learning_rate_scheduler.learning_rate)",
    )
    parser.add_argument(
        "--min-decay-lr",
        type=float,
        default=None,
        help="Override min decay learning rate (optimizer.learning_rate_scheduler.min_decay_lr)",
    )
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=None,
        help="Override learning rate warmup steps (optimizer.learning_rate_scheduler.lr_warmup_steps)",
    )

    # Token config overrides
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Override micro batch size (tokens.micro_batch_size)",
    )
    parser.add_argument(
        "--batch-accumulation-per-replica",
        type=int,
        default=None,
        help="Override batch accumulation (tokens.batch_accumulation_per_replica)",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=None,
        help="Override train steps (tokens.train_steps)",
    )
    parser.add_argument(
        "--val-check-interval",
        type=int,
        default=None,
        help="Override validation check interval (tokens.val_check_interval)",
    )
    parser.add_argument(
        "--dp",
        type=int,
        default=None,
        help="Override dp (parallelism.dp)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=None,
        help="Override tp (parallelism.tp)",
    )
    parser.add_argument(
        "--pp",
        type=int,
        default=None,
        help="Override pp (parallelism.pp)",
    )
    parser.add_argument(
        "--disable-ddp",
        action="store_true",
        default=None,
        help="Override parallelism.disable_ddp: skip wrapping the model in DDP",
    )

    # ULFM options
    parser.add_argument(
        "--ulfm-verbose",
        action="store_true",
        default=False,
        help="Enable verbose debug logging in the ULFM backend",
    )

    # Failure simulator options (deterministic replica-aware schedule).
    # Either load a pre-generated YAML via --failure-schedule, or generate
    # inline with --failure-count + --failure-step-range + --failure-locations.
    # The two sources are mutually exclusive. If neither is given, no failures
    # are injected.
    parser.add_argument(
        "--failure-schedule",
        type=str,
        default=None,
        help="Path to a pre-generated failure schedule YAML "
             "(see scripts/generate_failure_schedule.py).",
    )
    parser.add_argument(
        "--failure-count",
        type=int,
        default=0,
        help="Inline generator: number of DP replicas to kill. 0 disables injection.",
    )
    parser.add_argument(
        "--failure-seed",
        type=int,
        default=42,
        help="Inline generator seed (must be identical on every rank).",
    )
    parser.add_argument(
        "--failure-step-range",
        type=int,
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help="Inline generator: half-open [START, END) minibatch index range for kills.",
    )
    parser.add_argument(
        "--failure-sampling",
        choices=("iid", "stratified"),
        default="stratified",
        help="Inline generator: step sampling mode.",
    )
    parser.add_argument(
        "--failure-locations",
        nargs="+",
        default=None,
        metavar="NAME:WEIGHT",
        help="Inline generator: location:weight pairs "
             "(e.g. grad-sync:first:0.6 forward:0.4). "
             "Locations: forward, backward, grad-sync:first, grad-sync:mid, grad-sync:last, optimizer (see ulfm_collectives/failure/locations.py).",
    )
    parser.add_argument(
        "--failure-exclude-replicas",
        type=str,
        default="0",
        help="Comma-separated DP replica ids to spare from selection "
             "(default '0' spares the dp_rank=0 replica that hosts wandb). "
             "Pass empty string to allow all replicas.",
    )

    # Generic dotted-path overrides (e.g. --override optimizer.optimizer_factory.muon_mode=sgd)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override any config field via dotted path (e.g. optimizer.optimizer_factory.muon_mode=sgd)",
    )

    return parser.parse_args()
