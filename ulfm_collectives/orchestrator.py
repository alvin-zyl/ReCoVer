import logging
import threading
from typing import List, Optional, Set, Tuple, TYPE_CHECKING

import torch
import torch.distributed as dist
import ulfm_collectives as ULFM

from .policy import GradRestoreMode
from .ulfm_work_types import ULFMWorkType
from .failure_simulator import get_failure_simulator

from .policy import (
    FailureEvent,
    PolicyDecision,
    FaultTolerancePolicy,
    StaticWorldPolicy,
)


logger = logging.getLogger(__name__)


class StepTxnOrchestrator:
    """
    Process-local coordinator for transactional steps across many process groups.

    This orchestrator owns the policy and provides a unified entry point for all
    failure handling logic through handle_work_completion().

    Responsibilities:
      - snapshot gradients before allreduce
      - track which buckets completed under the current communicator epoch
      - detect failures in ULFM work completions
      - consult policy for recovery decisions
      - apply recovery strategies (quiesce, repair, restore)
      - drive gradient restoration (blocking or non-blocking)

    Unified Entry Point:
      - handle_work_completion(work, bucket_index): Call this after each allreduce
        to handle both success and failure cases. It encapsulates all failure
        detection, policy consultation, and recovery logic.
    """

    def __init__(
        self,
        rank: int,
        pg: ULFM.ProcessGroupULFM = None,
        policy: "FaultTolerancePolicy" = None,
        ulfm_opts: ULFM.ULFMOptions = None,
    ) -> None:
        self._rank = rank
        self._pg = pg if pg is not None else dist.group.WORLD
        self._policy = policy
        if ulfm_opts is None:
            ulfm_opts = ULFM.ULFMOptions()
            logger.warning(
                f"[Rank {self._rank}] No ULFMOptions provided; using defaults: "
                f"auto_repair={ulfm_opts.auto_repair}, track_rank_types={ulfm_opts.track_rank_types}"
            )
        self._ulfm_opts = ulfm_opts

        # Failure / txn state
        self._need_restore = threading.Event()

        # Restoration bookkeeping
        self._restore_streams: dict = {}
        self._restore_event = None
        self._restore_plan = GradRestoreMode.SKIP
        self._restore_started = False

        # Snapshots: (grad_view_ref, snapshot_tensor, pg_epoch_at_snapshot, bucket_idx)
        self._snapshots: List[Tuple[torch.Tensor, torch.Tensor, int, int]] = []

        # Buckets already reduced in the current epoch
        self._buckets_reduced_current_epoch: Set[int] = set()
        self._buckets_nooped_current_step: Set[int] = set()

        # Optimizer decision flags
        self._at_policy_boundary = False
        self._num_policy_boundary_steps = 0

        # Hook invocation counter (used to detect gradient corruption)
        self._hook_invocation_counter = 0

        # Deferred bucket queue: populated by the deferred hook during backward,
        # fired after the pipeline stage completes via fire_deferred_allreduces().
        self._deferred_buckets: List[Tuple[torch.Tensor, int]] = []

        # Replicas this DP group has dropped through its own failure events,
        # and how to find a world rank's replica (set_replica_map). Used to
        # reconcile with the world consensus at the start of an extended pass.
        self._replica_of = None
        self._my_replica = None
        self._excluded_replicas: Set[int] = set()

        # Training progression (microbatch index / total in accumulation window / macrobatch index)
        self._current_microbatch_idx = 0
        self._total_microbatches = 0
        self._current_macrobatch_idx = 0
        self.initialize()

    # ------------------------------------------------------------------ #
    # Policy access
    # ------------------------------------------------------------------ #
    def initialize(self):
        self.dp_pg.set_target_contribution(self.curr_grad_accum_steps)

    @property
    def policy(self) -> "FaultTolerancePolicy":
        """Get the fault tolerance policy."""
        return self._policy

    @property
    def dp_pg(self) -> ULFM.ProcessGroupULFM:
        """Get the data parallel process group."""
        return self._pg

    @property
    def curr_world_size(self) -> int:
        """Get the current world size from the process group."""
        return self.dp_pg.current_size()

    @property
    def ulfm_opts(self) -> ULFM.ULFMOptions:
        """Get the ULFM options used by this orchestrator."""
        return self._ulfm_opts

    @property
    def at_policy_boundary(self) -> bool:
        """Check if we are at a policy boundary."""
        return self._at_policy_boundary

    @property
    def num_policy_boundary_steps(self) -> int:
        """Number of extra microbatches needed at a policy boundary."""
        return self._num_policy_boundary_steps

    @property
    def curr_grad_accum_steps(self) -> int:
        """Get the number of grad accumulation steps for major procs."""
        return self.policy.current_grad_accum_steps

    @property
    def minor_proc_grad_accum_steps(self) -> int:
        """Get the number of grad accumulation steps for minor procs."""
        return self.policy.get_minor_proc_grad_accum_steps()

    @property
    def is_minor(self) -> bool:
        """Check if this rank is currently a minor rank."""
        return self.dp_pg.is_minor()

    @property
    def is_boundary_minor(self) -> bool:
        """Check if this rank is currently a boundary minor rank."""
        return self.dp_pg.is_boundary_minor()

    @property
    def num_major_procs(self) -> int:
        """Get the current number of major procs from the process group."""
        return self.dp_pg.get_num_major_procs()

    @property
    def num_major_spare_procs(self) -> int:
        """Get the current number of major spare procs from the process group."""
        return self.dp_pg.get_num_major_spare_procs()

    @property
    def num_minor_spare_procs(self) -> int:
        """Get the current number of minor spare procs from the process group."""
        return self.dp_pg.get_num_minor_spare_procs()

    @property
    def num_minor_procs(self) -> int:
        """Get the current number of minor procs from the process group."""
        return self.dp_pg.get_num_minor_procs()

    @property
    def should_zero_grad(self) -> bool:
        """Check if this rank should zero gradients at the last policy boundary step."""
        if not self.dp_pg.should_contribute():
            return True
        else:
            self.dp_pg.increment_contributed()
            return False

    @property
    def effective_batch_size(self) -> int:
        """Get the effective batch size considering grad accumulation and FT policy."""
        if isinstance(self.policy, StaticWorldPolicy):
            return self.policy.target_batch_size
        else:
            return self.curr_world_size * self.curr_grad_accum_steps

    @property
    def policy_sanity_check_passed(self) -> bool:
        """Check if the current policy configuration is sane as in process group."""
        num_major_checked = self.num_major_procs == self.policy.get_num_major_procs()
        num_minor_checked = self.num_minor_procs == self.policy.get_num_minor_procs()
        num_major_spares_checked = (
            self.num_major_spare_procs == self.policy.get_num_major_spare_procs()
        )
        num_minor_spares_checked = (
            self.num_minor_spare_procs == self.policy.get_num_minor_spare_procs()
        )
        return (
            num_major_checked
            and num_minor_checked
            and num_major_spares_checked
            and num_minor_spares_checked
        )

    def detect_policy_boundary(self, at_boundary: bool) -> None:
        if at_boundary and not self._at_policy_boundary:
            self._at_policy_boundary = True

    # ------------------------------------------------------------------ #
    # Registration and progress
    # ------------------------------------------------------------------ #
    def update_progress(
        self, microbatch_idx: int, total_microbatches: int, macrobatch_idx: int
    ) -> None:
        """Called from the training loop to report where we are in the window."""
        self._current_microbatch_idx = microbatch_idx
        self._total_microbatches = total_microbatches
        self._current_macrobatch_idx = macrobatch_idx

    def get_progress(self) -> Tuple[int, int, int]:
        """Return (current_microbatch_idx, total_microbatches, current_macrobatch_idx)."""
        return (
            self._current_microbatch_idx,
            self._total_microbatches,
            self._current_macrobatch_idx,
        )

    # ------------------------------------------------------------------ #
    # Bucket lifecycle
    # ------------------------------------------------------------------ #

    def reset_hook_counter(self) -> None:
        self._hook_invocation_counter = 0

    def get_hook_counter(self) -> int:
        return self._hook_invocation_counter

    def increment_hook_counter(self) -> None:
        self._hook_invocation_counter += 1

    def on_bucket_snapshot(
        self,
        bucket_buffer: torch.Tensor,
        bucket_index: int,
        pg: ULFM.ProcessGroupULFM,
    ) -> None:
        """
        Snapshot the full bucket gradient buffer before reduction.
        """
        current_epoch = pg.worldEpoch()
        snapshot = bucket_buffer.detach().clone()
        self._snapshots.append((bucket_buffer, snapshot, current_epoch, bucket_index))
        logger.debug(
            f"[Rank {self._rank}] Bucket {bucket_index} snapshot captured at pg_epoch {current_epoch}"
        )

    def _on_bucket_reduction_success(
        self,
        bucket_index: int,
    ) -> None:
        """
        Mark a bucket as successfully reduced in the current communicator epoch.
        """
        self._buckets_reduced_current_epoch.add(bucket_index)
        logger.debug(f"[Rank {self._rank}] Bucket {bucket_index} successfully reduced")

    def _on_recovery_success(self) -> None:
        """
        Called after a successful recovery.
        """
        self._policy.on_recovery()

    # ------------------------------------------------------------------ #
    # Failure handling - Unified entry point
    # ------------------------------------------------------------------ #

    def handle_work_completion(
        self,
        work: ULFM.WorkULFM,
        bucket_index: Optional[int] = None,
        work_type: Optional[ULFMWorkType] = ULFMWorkType.GRADIENT_REDUCTION,
    ) -> None:
        """
        Unified entry point for handling work completion (success or failure).

        This method encapsulates all the complex logic for:
        - Detecting failures in ULFM work
        - Consulting the policy for recovery decisions
        - Applying orchestrator-side responses (quiesce, restoration planning)
        - Performing manual or auto-repair of communicators
        - Tracking successful bucket reductions

        Args:
            work: ULFM work object from allreduce operation
            bucket_index: Index of the gradient bucket being processed

        Raises:
            RuntimeError: If manual communicator repair fails
            Exception: Re-raises any exception encountered during failure handling
        """
        # Import FailureEvent here to avoid circular imports
        from .policy import FailureEvent

        # A work could be marked as NOOP even after comm being repaired
        if (
            hasattr(work, "was_noop")
            and work.was_noop()
            and work_type == ULFMWorkType.GRADIENT_REDUCTION
        ):
            noop_bucket = True
            self._buckets_nooped_current_step.add(bucket_index)
            logger.warning(
                f"[Rank {self._rank}] bucket {bucket_index} was marked NOOP."
            )
        else:
            noop_bucket = False

        # Check for failures in the work completion
        if hasattr(work, "has_failures") and work.has_failures():
            try:
                failed_ranks = work.get_failed_ranks()

                # Top-level failure detection (always visible)
                logger.warning(
                    f"[Rank {self._rank}] Communication failures detected in ranks: {failed_ranks}"
                )
                if self._replica_of is not None and hasattr(work, "get_failed_world_ranks"):
                    for world_rank in work.get_failed_world_ranks():
                        self._excluded_replicas.add(self._replica_of(world_rank))

                # Create FailureEvent for policy
                current_microbatch_idx, total_microbatches, _ = self.get_progress()
                if total_microbatches <= 0:
                    total_microbatches = (
                        self.policy.grad_accum_steps if self.policy else 1
                    )

                # Unpack failure stats and current counts from work
                (
                    failed_major,
                    failed_minor,
                    failed_major_spares,
                    failed_minor_spares,
                    failed_boundary_minors,
                    at_policy_boundary,
                ) = work.get_failure_stats()
                (
                    curr_majors,
                    curr_minors,
                    curr_major_spares,
                    curr_minor_spares,
                    curr_boundary_minors,
                    curr_contributed,
                    curr_boundary_contributed,
                ) = work.get_current_counts()

                failure_event = FailureEvent(
                    failed_ranks=failed_ranks,
                    current_microbatch_idx=current_microbatch_idx,
                    total_microbatches=total_microbatches,
                    world_epoch=self.dp_pg.worldEpoch(),
                    curr_rank=self.dp_pg.current_rank(),
                    curr_size=self.dp_pg.current_size(),
                    failed_major=failed_major,
                    failed_minor=failed_minor,
                    failed_major_spares=failed_major_spares,
                    failed_minor_spares=failed_minor_spares,
                    failed_boundary_minors=failed_boundary_minors,
                    curr_num_major_procs=curr_majors,
                    curr_num_minor_procs=curr_minors,
                    curr_num_major_spares=curr_major_spares,
                    curr_num_minor_spares=curr_minor_spares,
                    curr_num_boundary_minor_procs=curr_boundary_minors,
                    curr_contributed=curr_contributed,
                    curr_boundary_contributed=curr_boundary_contributed,
                    at_policy_boundary=at_policy_boundary,
                )

                # Consult policy for decision
                decision = self.policy.on_failure(failure_event)

                # Detailed policy decision trace (INFO level)
                logger.info(
                    f"[Rank {self._rank}] Policy decision: {decision.failure_response.value}, "
                    f"quiesce={decision.should_quiesce}, manual_repair={decision.should_manual_repair}"
                )

                # Apply orchestrator-side response (quiesce + restoration plan)
                self._apply_policy_decision(
                    decision=decision,
                    failure_event=failure_event,
                )

                # Handle communicator repair
                if decision.should_manual_repair:
                    # Manual repair (when auto_repair is disabled)
                    if self.dp_pg.repair_communicator():
                        logger.info(
                            f"[Rank {self._rank}] Manual communicator repair successful"
                        )
                        self._on_recovery_success()
                    else:
                        logger.error(
                            f"[Rank {self._rank}] Manual communicator repair failed"
                        )
                        raise RuntimeError("Failed to repair communicator")
                else:
                    # Auto-repair handles it, just verify
                    if not self.dp_pg.check_for_failures():
                        logger.info(
                            f"[Rank {self._rank}] Auto-repair handled failures successfully"
                        )
                        self._on_recovery_success()

            except Exception as e:
                logger.error(f"[Rank {self._rank}] Error handling work completion: {e}")
                raise
            return False
        else:
            if not noop_bucket:
                if bucket_index is not None:
                    # No failures - mark bucket as successfully reduced in current epoch
                    self._on_bucket_reduction_success(bucket_index)
                return True
            return False

    def _apply_policy_decision(
        self,
        *,
        decision: "PolicyDecision",
        failure_event: "FailureEvent",
    ) -> None:
        """
        Apply policy decision after the ULFM hook observes a failure.

        This is an internal method called by handle_work_completion().
        """
        restore_mode = decision.grad_restore_mode or GradRestoreMode.SKIP
        grads_corrupted = self._hook_invocation_counter > 0

        self._restore_started = False

        self.detect_policy_boundary(decision.at_policy_boundary)
        if self.at_policy_boundary:
            self._num_policy_boundary_steps = decision.num_policy_boundary_steps or 0
            # Fold any contributions already made during a prior boundary
            # extended pass into the regular counter (no-op if this is a
            # fresh boundary) before installing the new split, so the next
            # extension stacks cleanly on top.
            self.dp_pg.merge_boundary_contributed()
            self.dp_pg.set_boundary_minor_split(
                decision.num_nonzero_grad_procs,
                decision.num_policy_boundary_steps,
                decision.num_policy_boundary_steps - 1,
            )

        if restore_mode != GradRestoreMode.SKIP:
            self._restore_plan = restore_mode
            self._need_restore.set()
        else:
            self._restore_plan = GradRestoreMode.SKIP
            self._need_restore.clear()

        # Top-level failure notification (always visible)
        if decision.should_quiesce:
            logger.warning(
                f"[Rank {self._rank}] Failure detected; quiescing PGs "
                f"(failed_ranks={failure_event.failed_ranks}, restore_mode={self._restore_plan})"
            )
            self._set_quiesce(True)
        else:
            logger.warning(
                f"[Rank {self._rank}] Failure detected; continuing without quiesce "
                f"(restore_mode={self._restore_plan}, failed_ranks={failure_event.failed_ranks})"
            )

        if grads_corrupted:
            if self._restore_plan != GradRestoreMode.SKIP:
                logger.warning(
                    f"[Rank {self._rank}] Buckets reduced before failure: {self._hook_invocation_counter}; "
                    f"marking gradients for restoration"
                )
            else:
                logger.warning(
                    f"[Rank {self._rank}] Buckets reduced before failure: {self._hook_invocation_counter}; "
                    f"but not restoring due to policy"
                )
        elif not grads_corrupted:
            logger.info(
                f"[Rank {self._rank}] Failure occurred before gradient reduction; no restoration needed"
            )

        # Always reset per-epoch bookkeeping after a failure; the repaired communicator
        # increments worldEpoch() so we need to re-verify each bucket.
        self._buckets_reduced_current_epoch.clear()

    def _set_quiesce(self, value: bool) -> None:
        try:
            self.dp_pg.set_quiesce(value)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Restoration control
    # ------------------------------------------------------------------ #

    def get_restore_plan(self) -> GradRestoreMode:
        return self._restore_plan

    def restore_gradients_non_blocking(self):
        """
        Launch non-blocking gradient restoration (used at policy boundaries).
        """
        current_epoch = self.dp_pg.worldEpoch()

        snapshots_to_restore = [
            (view, snap, epoch, bucket_idx)
            for (view, snap, epoch, bucket_idx) in self._snapshots
            if epoch < current_epoch
            and bucket_idx not in self._buckets_nooped_current_step
        ]

        if not snapshots_to_restore:
            logger.info(
                f"[Rank {self._rank}] No snapshots need restoration - all buckets are up to date"
            )
            self._need_restore.clear()
            self._restore_plan = GradRestoreMode.SKIP
            # Nothing was reduced in this pass: the remaining snapshots hold
            # the buffers' current (local) values. Drop them and this pass's
            # NOOP marks, as below.
            self._snapshots.clear()
            self._buckets_nooped_current_step.clear()
            return

        use_cuda = torch.cuda.is_available()
        devices = sorted(
            {view.device for (view, _, _, _) in snapshots_to_restore},
            key=lambda d: (d.type, getattr(d, "index", -1)),
        )

        if use_cuda:
            for device in devices:
                if device.type == "cuda" and device not in self._restore_streams:
                    self._restore_streams[device] = torch.cuda.Stream(device=device)

        restored_count = 0
        for view, snap, _, _ in snapshots_to_restore:
            device = view.device
            if device.type == "cuda" and use_cuda:
                stream = self._restore_streams[device]
                with torch.cuda.stream(stream):
                    view.copy_(
                        snap.to(dtype=view.dtype, device=device, non_blocking=True),
                        non_blocking=True,
                    )
            else:
                view.copy_(snap.to(dtype=view.dtype, device=device))
            restored_count += 1

        self._restore_started = True

        if use_cuda:
            self._restore_event = torch.cuda.Event()
            for stream in self._restore_streams.values():
                self._restore_event.record(stream)
        else:
            self._restore_event = None

        logger.info(
            f"[Rank {self._rank}] Non-blocking restore scheduled - {restored_count} bucket gradients "
            f"from epoch < {current_epoch}"
        )

        # All existing snapshots are stale after a policy-boundary rollback:
        # the extended pass about to run will repopulate _snapshots with fresh
        # entries. Clearing here prevents blocking restore (if it runs after
        # the extended pass also fails) from re-popping these and overwriting
        # the extended pass's contribution.
        self._snapshots.clear()
        # The NOOP marks belong to the pass just rolled back as well. Kept,
        # they would exclude the same bucket indices from the next rollback,
        # after the extended pass has reduced those buckets, and the reduced
        # sums would be summed again.
        self._buckets_nooped_current_step.clear()

        self._set_quiesce(False)
        return

    def restore_gradients_blocking(
        self,
        re_reduce: bool = True,
        allow_internal_retry: bool = True,
    ) -> None:
        """
        Blocking gradient restoration before optimizer.step().

        Args:
            re_reduce: After rolling each bucket back to its snapshot, re-issue
                a ulfm_allreduce to produce the reduced value in place.
            allow_internal_retry: Primary gate for internal retry on re-
                reduction failure. When True (legacy), a re-reduction failure
                is handled by the existing boundary-aware branch: retry if not
                at a policy boundary, return if at one. When False, any
                re-reduction failure returns control to the caller
                immediately, regardless of boundary state. Pass False when the
                caller cross-synchronizes DP groups after this call (e.g. a
                replica barrier + ULFM DP barrier that propagates the failure
                to late-discoverer DP groups) — otherwise early-discoverer
                ranks retry-to-success and set restore_plan=SKIP while
                late-discoverer ranks set restore_plan=BLOCKING and re-enter
                alone, and the next collective deadlocks (MPI has no timeout).
        """
        current_epoch = self.dp_pg.worldEpoch()

        snapshots_to_restore = [
            (view, snap, epoch, bucket_idx)
            for (view, snap, epoch, bucket_idx) in self._snapshots
            if epoch < current_epoch
            and bucket_idx not in self._buckets_reduced_current_epoch
        ]

        if not snapshots_to_restore:
            logger.info(
                f"[Rank {self._rank}] No snapshots need restoration - all buckets are up to date"
            )
            self._need_restore.clear()
            self._restore_plan = GradRestoreMode.SKIP
            return

        if re_reduce:
            opts = torch.distributed.AllreduceOptions()
            opts.reduceOp = torch.distributed.ReduceOp.SUM
            ulfm_opts = ULFM.ULFMOptions()
            ulfm_opts.auto_repair = self.policy.enable_auto_repair

        restored_count = 0
        successfully_reduced = set()
        while snapshots_to_restore:
            view, snap, epoch, bucket_idx = snapshots_to_restore.pop()
            view.copy_(snap.to(dtype=view.dtype, device=view.device))
            restored_count += 1
            logger.debug(
                f"[Rank {self._rank}] Restored bucket {bucket_idx} for re-reduction, "
                f"old_epoch={epoch}, current epoch={current_epoch}, restored_count={restored_count}"
            )

            if re_reduce:
                work = self.dp_pg.ulfm_allreduce([view], opts=opts, ulfm_opts=ulfm_opts)
                work.wait()
                succeed = self.handle_work_completion(
                    work=work,
                    bucket_index=bucket_idx,
                )

                if not succeed:
                    if allow_internal_retry and not self.at_policy_boundary:
                        logger.warning(
                            f"[Rank {self._rank}] Failure during re-reduction of bucket {bucket_idx}, "
                            f"not crossing policy boundary, retrying restoration and re-reduction"
                        )
                        snapshots_to_restore = [
                            (view, snap, epoch, bucket_idx)
                            for (view, snap, epoch, bucket_idx) in self._snapshots
                            if epoch < current_epoch
                            and bucket_idx not in self._buckets_reduced_current_epoch
                        ]
                        successfully_reduced.clear()
                        restored_count = 0
                        continue
                    else:
                        # Either at policy boundary (legacy early-return) or
                        # internal retry disabled — hand control back to caller.
                        return
                else:
                    successfully_reduced.add(bucket_idx)
            else:
                successfully_reduced.add(bucket_idx)

        self._buckets_reduced_current_epoch.update(successfully_reduced)
        action = "restored and re-reduced" if re_reduce else "restored"
        logger.info(
            f"[Rank {self._rank}] Blocking restore completed - {restored_count} bucket gradients {action} "
            f"from epoch < {current_epoch}"
        )

        self._set_quiesce(False)
        self._restore_plan = GradRestoreMode.SKIP
        self._restore_started = True
        return

    def wait_restore_before_backward(self) -> None:
        """
        Synchronize non-blocking restoration before starting backward().
        """
        if self._restore_event is not None:
            torch.cuda.current_stream().wait_event(self._restore_event)
            self._restore_event = None
        if self._restore_started and self._restore_plan == GradRestoreMode.NON_BLOCKING:
            self._restore_plan = GradRestoreMode.SKIP
            self._need_restore.clear()
            logger.info(
                f"[Rank {self._rank}] Restore completed (synced before backward)"
            )

    def on_grad_sync_step_consensus(self) -> bool:
        consensus_ulfm_opts = self.ulfm_opts.copy()
        consensus_ulfm_opts.consensus_on_rank_types = False
        work = self.dp_pg.consensus(consensus_ulfm_opts)
        work.wait()
        succeed = self.handle_work_completion(
            work=work, work_type=ULFMWorkType.CONSENSUS
        )
        return succeed

    def set_replica_map(self, replica_of, my_world_rank: int) -> None:
        """``replica_of(world_rank)`` gives the replica (data-parallel index)
        of a world rank."""
        self._replica_of = replica_of
        self._my_replica = replica_of(my_world_rank)

    def reconcile_with_world_failures(self, failed_world_ranks) -> bool:
        """Called after the world consensus that starts an extended pass,
        with the world ranks that consensus found failed. Returns True if
        this DP group re-decided (the caller then re-reads the number of
        extra microbatches).

        Each DP group decides on the failures it has seen itself. A replica
        that broke after some groups had already decided is in the decision
        of the groups that decided later, but not in the decision of the
        early ones, and the replica's surviving ranks have already passed the
        replica-consistency gate. The groups would then run different
        numbers of extra microbatches. The world consensus sees every
        failure, so here:
        - a rank whose own replica lost a rank leaves as a failed process
          (the replica is the unit that survives or fails), and
        - a group that has not dropped some failed replica runs one DP
          consensus, which finds the failure (that replica's rank in this
          group is dead or leaving) and makes the decision anew, from the
          same counts as the other groups.
        No microbatch and no reduction has run since this group's last
        rollback, so the rollback below only resets the plan, and the
        quiesce latch set by the decision is released.

        Nothing happens when every failed replica is already dropped by
        this group, e.g. when all failures of the step happened before
        their replicas passed the gate.
        """
        if self._replica_of is None or not failed_world_ranks:
            return False
        failed_replicas = {self._replica_of(r) for r in failed_world_ranks}
        if self._my_replica in failed_replicas:
            from .fail_stop import kill_self
            kill_self(
                f"replica {self._my_replica} lost rank(s) "
                f"{sorted(r for r in failed_world_ranks if self._replica_of(r) == self._my_replica)}; "
                f"leaving with it"
            )
        new_replicas = failed_replicas - self._excluded_replicas
        if not new_replicas:
            return False
        logger.warning(
            f"[Rank {self._rank}] World consensus found failed replica(s) "
            f"{sorted(new_replicas)} that this DP group has not dropped; "
            f"deciding again before the extended pass"
        )
        if self.on_grad_sync_step_consensus():
            logger.warning(
                f"[Rank {self._rank}] The DP consensus found no failure; "
                f"keeping the current decision"
            )
            return False
        self.restore_gradients_non_blocking()
        self._set_quiesce(False)
        return True

    # ------------------------------------------------------------------ #
    # Iteration lifecycle
    # ------------------------------------------------------------------ #

    def on_iteration_start(self) -> None:
        self.reset_hook_counter()
        self.policy.on_window_start()

    def on_grad_sync_step_prepare(self) -> None:
        self._set_quiesce(False)

    def _on_policy_advancement(self) -> None:
        rank_type_counts = self.policy.advance_policy()
        self.dp_pg.set_major_minor_split_with_spares(*rank_type_counts)
        self.dp_pg.update_rank_type_counts(*rank_type_counts)
        self.dp_pg.reset_policy_boundary()
        self.dp_pg.reset_boundary_minor()
        self.dp_pg.set_target_contribution(
            self.curr_grad_accum_steps, self.minor_proc_grad_accum_steps
        )

    def _on_step_committed_across_policy_boundary(self) -> None:
        """
        Called after optimizer.step() is committed across a policy boundary.
        """
        self._on_policy_advancement()

    def mark_iteration_end(self) -> bool:
        """
        Called after backward() to determine whether optimizer.step() should run.
        Returns True if the step was skipped.
        """
        self._snapshots.clear()
        self._buckets_reduced_current_epoch.clear()
        self._buckets_nooped_current_step.clear()
        self._restore_streams.clear()
        self._restore_event = None
        self._restore_plan = GradRestoreMode.SKIP
        self._restore_started = False
        self._need_restore.clear()
        self._set_quiesce(False)
        self._at_policy_boundary = False
        self._num_policy_boundary_steps = 0
        self.dp_pg.reset_contributed()

    def after_successful_commit(self) -> None:
        """Training loop must call this after a successful optimizer.step()."""
        # if self.at_policy_boundary:
        if self.at_policy_boundary:
            self._on_step_committed_across_policy_boundary()
            logger.debug(
                f"[Rank {self._rank}] Advanced policy after last step at policy boundary"
            )
        self.mark_iteration_end()

    # ------------------------------------------------------------------ #
    # Deferred bucket allreduce (for pipeline-parallel training)
    # ------------------------------------------------------------------ #

    def queue_deferred_bucket(self, buffer: torch.Tensor, bucket_index: int) -> None:
        """Queue a bucket for deferred allreduce (called by deferred hook)."""
        self._deferred_buckets.append((buffer, bucket_index))
        logger.debug(
            f"[Rank {self._rank}] Queued deferred bucket {bucket_index} "
            f"(total queued: {len(self._deferred_buckets)})"
        )

    def fire_deferred_allreduces(self) -> bool:
        """
        Fire all deferred bucket allreduces.

        Call after the pipeline stage and replica-consistency gate complete.
        Each queued buffer is allreduced in-place via ULFM. For the fp32
        accumulator path, the queued buffer is a view into the accumulator's
        _contiguous_fp32_grad_buffer, so the allreduce result lands directly
        in the accumulator storage — no scatter-back.

        Returns True if all buckets succeeded, False if any failure occurred.
        """
        if not self._deferred_buckets:
            logger.debug(f"[Rank {self._rank}] No deferred buckets to allreduce")
            return True

        opts = torch.distributed.AllreduceOptions()
        opts.reduceOp = torch.distributed.ReduceOp.SUM
        ulfm_opts = self._ulfm_opts

        logger.debug(
            f"[Rank {self._rank}] Firing {len(self._deferred_buckets)} deferred allreduces"
        )

        any_failure = False
        num_buckets = len(self._deferred_buckets)
        for bucket_pos, (buffer, bucket_index) in enumerate(self._deferred_buckets):
            work = self.dp_pg.ulfm_allreduce([buffer], opts=opts, ulfm_opts=ulfm_opts)
            work.wait()

            # grad-sync:<first|mid|last> failure injection: this bucket's
            # cross-replica all-reduce has completed, later ones are pending.
            _sim = get_failure_simulator()
            if _sim is not None:
                _sim.check_grad_sync(bucket_pos, num_buckets)
            success = self.handle_work_completion(work, bucket_index=bucket_index)

            self.increment_hook_counter()

            if not success:
                any_failure = True

        self._deferred_buckets.clear()
        return not any_failure

    def clear_deferred_buckets(self) -> None:
        """Clear the deferred bucket queue (e.g. on iteration reset)."""
        self._deferred_buckets.clear()
