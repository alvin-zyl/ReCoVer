#!/usr/bin/env python3
"""
ULFM Communication Hook for PyTorch DDP

This module provides a reusable communication hook for fault-tolerant distributed training
with ULFM (User Level Failure Mitigation). The hook integrates with PyTorch DDP and the
StepTxnOrchestrator for gradient snapshotting and recovery.
"""

import contextlib
import os
import signal
import logging
import torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .policy import FaultTolerancePolicy

try:
    import ulfm_collectives as ULFM
except ImportError:
    raise ImportError(
        "ULFM collectives extension not found. Please build the extension first."
    )

from .orchestrator import StepTxnOrchestrator
from .failure_simulator import get_failure_simulator

logger = logging.getLogger(__name__)


@dataclass
class HookState:
    """State object passed to the ULFM communication hook."""

    pg: Union[ULFM.ProcessGroupULFM, dist.ProcessGroup]
    orchestrator: StepTxnOrchestrator


def create_ulfm_recovery_hook(ulfm_opts: ULFM.ULFMOptions = None):
    """
    Create a ULFM communication hook with comprehensive failure recovery logic.

    Now policy-driven: consults the FaultTolerancePolicy for all fault tolerance decisions.
    The policy is accessed from the orchestrator (via hstate.orchestrator.policy).

    This hook integrates with the StepTxnOrchestrator to:
    - Snapshot gradients before reduction
    - Detect and handle process failures
    - Consult policy for quiesce/repair decisions
    - Track successful reductions for restoration decisions

    Args:
        failure_strategy: "continue", "restart", or "abort" (legacy, kept for ULFM options)

    Returns:
        Callable hook function compatible with DDP.register_comm_hook()
        The hook expects state to be a HookState object with pg and orchestrator.
        The orchestrator must have a policy attribute.

    Example:
        >>> from ulfm_hook import create_ulfm_recovery_hook, HookState
        >>> from policy import create_policy
        >>> pg = dist.group.WORLD
        >>> rank = dist.get_rank()
        >>> policy = create_policy("adaptive", ...)
        >>> orchestrator = StepTxnOrchestrator(rank=rank, policy=policy)
        >>> hook = create_ulfm_recovery_hook("continue")
        >>> hstate = HookState(pg=pg, orchestrator=orchestrator)
        >>> ddp_model.register_comm_hook(state=hstate, hook=hook)
    """

    """ULFMOptions setup with AllreduceOp SUM for gradient reduction."""
    opts = torch.distributed.AllreduceOptions()
    opts.reduceOp = torch.distributed.ReduceOp.SUM
    ulfm_opts = ulfm_opts if ulfm_opts is not None else ULFM.ULFMOptions()

    def hook(hstate: HookState, bucket: dist.GradBucket):
        """ULFM communication hook with comprehensive recovery logic."""
        pg = hstate.pg
        orch = hstate.orchestrator
        bucket_index = bucket.index()

        logger.debug(
            f"[Rank {orch._rank}] Hook entered for bucket {bucket_index}, "
            f"numel={bucket.buffer().numel()}, dtype={bucket.buffer().dtype}"
        )

        # 1) If a previous failure quiesced comms, NOOP this bucket
        if getattr(pg, "is_quiesced", lambda: False)():
            logger.warning(
                f"[Rank {orch._rank}] Communicator is quiesced before submitting MPI request."
            )
            fut = torch.futures.Future()
            fut.set_result(bucket.buffer())
            return fut

        # 2) Snapshot the entire bucket buffer (pre-reduce)
        logger.debug(
            f"[Rank {orch._rank}] Snapshotting bucket {bucket_index} before allreduce"
        )
        orch.on_bucket_snapshot(bucket.buffer(), bucket_index, pg)

        # work = dist.ulfm_all_reduce(bucket.buffer(), async_op=True, ulfm_opts=ulfm_opts)
        logger.debug(
            f"[Rank {orch._rank}] Submitting ulfm_allreduce for bucket {bucket_index}"
        )
        work = pg.ulfm_allreduce([bucket.buffer()], opts, ulfm_opts)

        def on_done(fut):
            logger.debug(
                f"[Rank {orch._rank}] Allreduce completed for bucket {bucket_index}, "
                f"entering work completion handler"
            )
            # Use orchestrator's unified entry point for handling work completion
            # This encapsulates all failure detection, policy consultation, and recovery logic
            # get_failure_simulator() is called here (not at hook-registration time) so that
            # simulators set after DDP construction are picked up correctly.
            _sim = get_failure_simulator()
            # Buckets complete asynchronously here and their number is not
            # known up front, so only grad-sync:first is supported.
            ctx = _sim.may_fail_here("grad-sync:first") if _sim is not None else contextlib.nullcontext()
            with ctx:
                orch.handle_work_completion(
                    work=work,
                    bucket_index=bucket_index,
                )

            # Increment hook invocation counter
            orch.increment_hook_counter()
            logger.debug(
                f"[Rank {orch._rank}] Hook done for bucket {bucket_index}, "
                f"hook_count={orch._hook_invocation_counter}"
            )

            return fut.value()[0]

        return work.get_future().then(on_done)

    return hook


def create_ulfm_deferred_hook(ulfm_opts: ULFM.ULFMOptions = None):
    """
    Create a deferred ULFM communication hook for pipeline-parallel training.

    The hook fires during the last microbatch's backward pass (per DDP bucketing)
    but does NOT submit MPI work.  Instead it:
      1. Snapshots the bucket buffer (for failure restoration)
      2. Queues the bucket reference on the orchestrator's deferred list
      3. Returns an immediately-resolved Future so finalize_backward never blocks

    The actual ULFM allreduce is fired later via
    ``orchestrator.fire_deferred_allreduces()`` after the pipeline stage and
    replica-consistency gate complete.

    Returns:
        Callable hook compatible with DDP.register_comm_hook()
    """
    ulfm_opts = ulfm_opts if ulfm_opts is not None else ULFM.ULFMOptions()

    def hook(hstate: HookState, bucket: dist.GradBucket):
        pg = hstate.pg
        orch = hstate.orchestrator
        bucket_index = bucket.index()

        logger.debug(
            f"[Rank {orch._rank}] Deferred hook entered for bucket {bucket_index}, "
            f"numel={bucket.buffer().numel()}, dtype={bucket.buffer().dtype}"
        )

        # If comms are quiesced, return immediately (same as recovery hook)
        if getattr(pg, "is_quiesced", lambda: False)():
            logger.warning(
                f"[Rank {orch._rank}] Communicator quiesced — skipping bucket {bucket_index}"
            )
            fut = torch.futures.Future()
            fut.set_result(bucket.buffer())
            return fut

        # Snapshot for restoration on failure (clone runs on current CUDA stream,
        # ordered after the backward that produced this bucket's gradients —
        # no cuda.synchronize needed here)
        orch.on_bucket_snapshot(bucket.buffer(), bucket_index, pg)

        # Queue for deferred allreduce (fired after PP completes)
        orch.queue_deferred_bucket(bucket.buffer(), bucket_index)

        # Return pre-resolved Future with the *same* bucket buffer tensor.
        # finalize_backward's alias check (bucket_view_in.is_alias_of(bucket_view_out))
        # passes → no copy → effectively a no-op.
        _sim = get_failure_simulator()
        ctx = _sim.may_fail_here("backward") if _sim is not None else contextlib.nullcontext()
        with ctx:
            fut = torch.futures.Future()
            fut.set_result(bucket.buffer())
            return fut

    return hook


def create_ulfm_fp32_deferred_hook(accumulator, param_id_to_name: dict):
    """
    Create a deferred ULFM hook that accumulates bf16 grads into fp32 buffers.

    Exploits the fact that a DDP bucket's fp32 grad views form a contiguous
    slice of the accumulator's `_contiguous_fp32_grad_buffer` (DDP fills
    buckets in reverse param-registration order, never skipping). The hook:
      1. Converts bf16 bucket grads → fp32 in the accumulator's buffer (add_)
      2. Computes the [min_offset, min_offset+total_numel) slice covering the
         bucket's params and asserts it is gap-free
      3. Snapshots that slice (clone) for failure rollback
      4. Queues the slice view (aliased to the real accumulator storage) for
         deferred ULFM allreduce — allreduce lands in place, no scatter-back
      5. Returns pre-resolved Future (bf16 bucket unchanged for DDP)

    Args:
        accumulator: FP32GradientAccumulator instance
        param_id_to_name: dict mapping id(param) → param name in accumulator

    Returns:
        Callable hook compatible with DDP.register_comm_hook()
    """

    def hook(hstate: HookState, bucket: dist.GradBucket):
        pg = hstate.pg
        orch = hstate.orchestrator
        bucket_index = bucket.index()

        logger.debug(
            f"[Rank {orch._rank}] FP32 deferred hook entered for bucket {bucket_index}, "
            f"numel={bucket.buffer().numel()}, dtype={bucket.buffer().dtype}"
        )

        # 1. Accumulate bf16 grads → fp32 buffer (unconditional: local work,
        #    must happen even if comm is quiesced so the accumulator carries
        #    every micro's contribution)
        for param, grad in zip(bucket.parameters(), bucket.gradients()):
            name = param_id_to_name[id(param)]
            fp32_grad_buffer = accumulator.get_grad_buffer(name)
            fp32_grad_buffer.add_(grad.view_as(fp32_grad_buffer))

        # If comm is quiesced we skip snapshot + queue: no allreduce will run
        # for this bucket this step, so there is nothing to roll back to.
        if getattr(pg, "is_quiesced", lambda: False)():
            logger.warning(
                f"[Rank {orch._rank}] Communicator quiesced — skipping snapshot/queue for bucket {bucket_index}"
            )
            fut = torch.futures.Future()
            fut.set_result(bucket.buffer())
            return fut

        # 2. Compute contiguous slice of _contiguous_fp32_grad_buffer for this bucket
        base = accumulator._contiguous_fp32_grad_buffer
        base_ptr = base.data_ptr()
        element_size = base.element_size()

        min_off = None
        max_end = 0
        total_numel = 0
        for p in bucket.parameters():
            v = accumulator.get_grad_buffer(param_id_to_name[id(p)]).view(-1)
            el_off = (v.data_ptr() - base_ptr) // element_size
            if min_off is None or el_off < min_off:
                min_off = el_off
            if el_off + v.numel() > max_end:
                max_end = el_off + v.numel()
            total_numel += v.numel()

        assert max_end - min_off == total_numel, (
            f"[Rank {orch._rank}] DDP bucket {bucket_index} is not contiguous in "
            f"_contiguous_fp32_grad_buffer: span={max_end - min_off} vs sum(numel)={total_numel}. "
            f"The slice shortcut requires DDP bucketing that preserves param-registration contiguity."
        )

        bucket_slice = base.narrow(0, min_off, total_numel)

        # 3. Snapshot the slice for failure rollback (restore writes back into
        #    the accumulator's real storage since bucket_slice aliases it)
        orch.on_bucket_snapshot(bucket_slice, bucket_index, pg)

        # 4. Queue the slice view for deferred allreduce — ULFM allreduce lands
        #    in place in _contiguous_fp32_grad_buffer via this view. No scatter-back.
        orch.queue_deferred_bucket(bucket_slice, bucket_index)

        # 5. Return pre-resolved Future (bf16 bucket unchanged for DDP)
        _sim = get_failure_simulator()
        ctx = _sim.may_fail_here("backward") if _sim is not None else contextlib.nullcontext()
        with ctx:
            fut = torch.futures.Future()
            fut.set_result(bucket.buffer())
            return fut

    return hook


@dataclass
class HSDPHookState:
    """State for the FSDP1 HYBRID_SHARD ULFM hook.

    Unlike DDP, FSDP fires the hook per FSDP unit (not per bucket). We assign
    a stable unit index in registration order — FSDP unit order is
    deterministic across ranks, so every rank sees the same indices.
    """

    pg: "Union[ULFM.ProcessGroupULFM, dist.ProcessGroup]"
    orchestrator: StepTxnOrchestrator
    _unit_counter: int = 0

    def next_unit_index(self) -> int:
        idx = self._unit_counter
        self._unit_counter += 1
        return idx

    def reset_unit_counter(self) -> None:
        """Call between training steps so unit indices restart at 0."""
        self._unit_counter = 0


def create_ulfm_hsdp_hook(ulfm_opts: ULFM.ULFMOptions = None):
    """
    Create an FSDP1 HYBRID_SHARD ULFM comm hook.

    FSDP fires this after the intra-replica reduce-scatter. The hook's job
    is to perform the cross-replica allreduce on the shard grad. On failure,
    the orchestrator snapshots and restores the bf16 shard grad in place.

    Hook signature per FSDP1: (state, grad_shard: torch.Tensor) -> Future[Tensor].

    Up/down-casting to fp32 for MPI happens inside the C++ ulfm_allreduce;
    Python-side stays in bf16.

    Returns:
        Callable hook compatible with FullyShardedDataParallel.register_comm_hook()
    """
    opts = torch.distributed.AllreduceOptions()
    opts.reduceOp = torch.distributed.ReduceOp.SUM
    ulfm_opts = ulfm_opts if ulfm_opts is not None else ULFM.ULFMOptions()

    def hook(state: HSDPHookState, grad_shard: torch.Tensor):
        """Deferred hook (mirrors nanotron pattern): snapshot + queue, NO MPI.

        FSDP fires this on the sync microstep with ``flat_param._saved_grad_shard``
        (the local-shard accumulator across all microsteps in the window).
        We snapshot it for failure rollback and queue it on the orchestrator's
        ``_deferred_buckets``. The actual ULFM allreduce is fired later by
        ``HSDPULFMTrainingManager._fire_cross_replica_allreduces`` →
        ``txn.fire_deferred_allreduces()``, AFTER backward returns. Returning
        a pre-resolved Future means FSDP's deferred-drain (if any) is a no-op
        and no MPI traffic happens during backward.
        """
        pg = state.pg
        orch = state.orchestrator
        unit_index = state.next_unit_index()

        logger.debug(
            f"[Rank {orch._rank}] HSDP deferred hook entered for unit {unit_index}, "
            f"numel={grad_shard.numel()}, dtype={grad_shard.dtype}"
        )

        # If comm is quiesced from a prior failure: skip snapshot + queue.
        # No allreduce will run for this unit this step → nothing to roll back.
        if getattr(pg, "is_quiesced", lambda: False)():
            logger.warning(
                f"[Rank {orch._rank}] replicate_pg quiesced — skipping snapshot/queue for unit {unit_index}."
            )
            fut = torch.futures.Future()
            fut.set_result(grad_shard)
            return fut

        # Snapshot the accumulator for restore on failure.
        orch.on_bucket_snapshot(grad_shard, unit_index, pg)

        # Queue for deferred ULFM allreduce. The buffer reference here is
        # _saved_grad_shard, which becomes flat_param.grad after FSDP's
        # _finalize_params runs at the end of backward — same Python tensor
        # object, so the queued reference stays valid.
        orch.queue_deferred_bucket(grad_shard, unit_index)

        # Pre-resolved Future — FSDP's drain will not block on this.
        _sim = get_failure_simulator()
        ctx = _sim.may_fail_here("backward") if _sim is not None else contextlib.nullcontext()
        with ctx:
            fut = torch.futures.Future()
            fut.set_result(grad_shard)
            return fut

    return hook
