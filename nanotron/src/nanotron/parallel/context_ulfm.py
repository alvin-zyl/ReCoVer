# src/nanotron/parallel/context_ulfm.py
"""
ULFMParallelContext: ParallelContext variant that initializes ULFM as the world
backend and pins TP/PP/EP/MP subgroups to explicit NCCL.

The DP process group inherits the ULFM default backend so the ULFM comm hook
can intercept gradient allreduces for fault-tolerant DP training.
"""

import datetime
import os

import numpy as np

import nanotron.distributed as dist
from nanotron.distributed import initialize_torch_distributed_ulfm
from nanotron.parallel.context import ParallelContext


class ULFMParallelContext(ParallelContext):
    """
    Parallel context that uses ULFM for the world/DP groups and NCCL for TP/PP/EP.

    Initialization order:
      1. Call initialize_torch_distributed_ulfm() so dist.is_initialized() is True
         before the parent __init__ can call the NCCL initializer.
      2. Call super().__init__(..., backend="ulfm") which creates world_pg as ULFM.
      3. Override _init_parallel_groups() to pass backend="nccl" explicitly to all
         groups except dp_pg, which gets no explicit backend (inherits ULFM).
    """

    def __init__(
        self,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        data_parallel_size: int,
        expert_parallel_size: int = 1,
    ):
        # Pre-initialize with ULFM so the parent's guard (if not dist.is_initialized())
        # skips the NCCL initialize_torch_distributed() call.
        if not dist.is_initialized():
            initialize_torch_distributed_ulfm()

        # Pass backend="ulfm" to bypass the assert and so world_pg uses ULFM.
        super().__init__(
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            data_parallel_size=data_parallel_size,
            expert_parallel_size=expert_parallel_size,
            backend="ulfm",
        )

    def _create_uncached_group(self, all_groups_ranks, backend, timeout):
        """create_new_group without the rank-set cache (world_ranks_to_pg)."""
        dist.barrier()
        rank = dist.get_rank()
        new_group_containing_rank = None
        kwargs = {"timeout": timeout} if timeout is not None else {}
        for group_ranks in all_groups_ranks:
            new_group = dist.new_group(ranks=group_ranks, backend=backend, **kwargs)
            if rank in group_ranks:
                new_group_containing_rank = new_group
        dist.barrier()
        return new_group_containing_rank

    def _init_parallel_groups(self):
        """
        Same layout as parent but TP/PP/EP/MP groups use explicit NCCL backend.
        dp_pg is created without a backend argument so it inherits ULFM.

        Set NCCL_PG_TIMEOUT_SECONDS env var to shorten NCCL watchdog timeout for
        testing (e.g. NCCL_PG_TIMEOUT_SECONDS=10).  Default: PyTorch's default (30 min).
        """
        dist.barrier()
        world_size = dist.get_world_size()
        ranks = np.arange(0, world_size).reshape(
            (
                self.expert_parallel_size,
                self.pipeline_parallel_size,
                self.data_parallel_size,
                self.tensor_parallel_size,
            )
        )
        self.world_ranks_to_pg = {}

        _timeout_sec = os.getenv("NCCL_PG_TIMEOUT_SECONDS")
        _nccl_timeout = (
            datetime.timedelta(seconds=float(_timeout_sec))
            if _timeout_sec is not None
            else None
        )

        # TP: NCCL (intra-replica, no failure expected)
        self.tp_pg = self.create_new_group(
            ranks.transpose((0, 1, 2, 3)).reshape((-1, self.tensor_parallel_size)),
            backend="nccl",
            timeout=_nccl_timeout,
        )
        # DP: ULFM (cross-replica, needs fault tolerance) — no backend → inherits ULFM
        self.dp_pg = self.create_new_group(
            ranks.transpose((3, 0, 1, 2)).reshape((-1, self.data_parallel_size)),
            backend=None,
        )
        # PP: NCCL (cross-replica pipeline stages)
        # Longer timeout: when a rank in the other replica dies, the surviving
        # replica's PP partner must wait for finalize_backward (blocked on MPI
        # futures) before sending pipeline grads.  Use NCCL_PP_TIMEOUT_SECONDS
        # to give it enough headroom; falls back to NCCL_PG_TIMEOUT_SECONDS.
        _pp_timeout_sec = os.getenv("NCCL_PP_TIMEOUT_SECONDS")
        _pp_timeout = (
            datetime.timedelta(seconds=float(_pp_timeout_sec))
            if _pp_timeout_sec is not None
            else _nccl_timeout
        )
        self.pp_pg = self.create_new_group(
            ranks.transpose((2, 3, 0, 1)).reshape((-1, self.pipeline_parallel_size)),
            backend="nccl",
            timeout=_pp_timeout,
        )
        # Expert: NCCL
        self.expert_pg = self.create_new_group(
            ranks.transpose((1, 2, 3, 0)).reshape((-1, self.expert_parallel_size)),
            backend="nccl",
            timeout=_nccl_timeout,
        )
        # Model parallel (TP+PP+EP for a given DP rank): NCCL
        # Longer timeout than other NCCL groups: surviving replicas must wait
        # for NCCL watchdog + MPI failure detection before DP partners arrive.
        _mp_timeout_sec = os.getenv("NCCL_MP_TIMEOUT_SECONDS")
        _mp_timeout = (
            datetime.timedelta(seconds=float(_mp_timeout_sec))
            if _mp_timeout_sec is not None
            else _nccl_timeout
        )
        # Not through create_new_group: it caches groups by rank set, and with
        # tp=1 (or pp=1) these ranks are a PP (or TP) group's, so mp_pg would
        # be that group and the replica-consistency gate would get its
        # shorter timeout.
        self.mp_pg = self._create_uncached_group(
            [ranks[:, :, dp_rank, :].reshape(-1) for dp_rank in range(self.data_parallel_size)],
            backend="nccl",
            timeout=_mp_timeout,
        )
        # TP+Expert combined: NCCL
        self.tp_and_expert_pg = self.create_new_group(
            [
                ranks[:, pp_rank, dp_rank, :].reshape(-1)
                for pp_rank in range(self.pipeline_parallel_size)
                for dp_rank in range(self.data_parallel_size)
            ],
            backend="nccl",
            timeout=_nccl_timeout,
        )
        self.world_rank_matrix: np.ndarray = ranks
        # Same ranks as dp_pg, for all-reduces outside the training step
        # (validation); see ulfm_collectives.side_collectives. Created last so
        # that the groups above are created exactly as before.
        self.dp_eval_pg = self._create_uncached_group(
            ranks.transpose((3, 0, 1, 2)).reshape((-1, self.data_parallel_size)),
            backend=None,
            timeout=None,
        )
