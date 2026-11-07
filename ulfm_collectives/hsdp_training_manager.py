"""HSDPULFMTrainingManager: ULFMTrainingManager adapted for FSDP1 HYBRID_SHARD.

Differences from the DDP-based parent:
  - Model is already FSDP-wrapped before being passed in (no internal DDP wrap).
  - Orchestrator's process group is replicate_pg (cross-replica), not world.
  - Registers create_ulfm_hsdp_hook instead of create_ulfm_recovery_hook.

Everything else — policy, orchestrator, train_step microbatch state machine,
no_sync, restore modes, optimizer commit — is inherited unchanged.
"""

import logging
import contextlib
import torch
import torch.distributed as dist
from torch.distributed.fsdp._traversal_utils import _get_fsdp_states

import ulfm_collectives as ULFM
from .training_manager import ULFMTrainingManager
from .orchestrator import StepTxnOrchestrator
from .policy import create_policy, GradRestoreMode
from .ulfm_hook import HSDPHookState, create_ulfm_hsdp_hook
from .failure_simulator import get_failure_simulator

logger = logging.getLogger(__name__)


class HSDPULFMTrainingManager(ULFMTrainingManager):
    """ULFMTrainingManager for FSDP1 HYBRID_SHARD.

    Args:
        fsdp_model: FullyShardedDataParallel(hf_model, sharding_strategy=HYBRID_SHARD, ...)
        replicate_pg: cross-replica ULFM process group
        grad_accum_steps: number of microbatches per optimizer step
        policy_type: "static" or "adaptive"
        **policy_kwargs: forwarded to create_policy
    """

    def __init__(
        self,
        fsdp_model,
        replicate_pg,
        world_pg: "ULFM.ProcessGroupULFM" = None,
        grad_accum_steps: int = 1,
        policy_type: str = "static",
        **policy_kwargs,
    ):
        if not isinstance(replicate_pg, ULFM.ProcessGroupULFM):
            raise ValueError(
                f"replicate_pg must be a ProcessGroupULFM, got {type(replicate_pg)}"
            )

        self.failure_strategy = "continue"
        self.process_group = replicate_pg
        # Derived world PG (sibling of replicate_pg / shard_pg, all rooted at
        # MPI_COMM_WORLD), mirroring nanotron's ParallelContext.world_pg. Used
        # by on_world_consensus(); None disables that step.
        self.world_pg = world_pg

        policy = create_policy(
            policy_type=policy_type,
            initial_grad_accum_steps=grad_accum_steps,
            enable_auto_repair=True,
            **policy_kwargs,
        )

        ulfm_opts = self._create_ulfm_opts(policy_type)

        rank = dist.get_rank()
        self.txn = StepTxnOrchestrator(
            rank=rank, pg=replicate_pg, policy=policy, ulfm_opts=ulfm_opts
        )

        # Already-wrapped FSDP model. Keep the attribute name `ddp_model` so
        # inherited train_step() code (which calls self.ddp_model.no_sync(),
        # self.ddp_model(data), self.ddp_model.parameters()) works unchanged —
        # FSDP1 provides all three.
        self.ddp_model = fsdp_model

        # FSDP defaults to dividing grads by (shard × replicate). We keep its
        # SHARD scaling (shard_pg is stable under our failure model — failures
        # kill replicas, not shards) and strip out the replicate factor, so
        # the parent's _get_grad_div_factor (= target_replicate × grad_accum
        # under StaticWorldPolicy) is the only replicate-axis divisor and
        # stays constant across rank failures.
        self._strip_fsdp_replicate_scaling(fsdp_model)

        self._hook_state = HSDPHookState(pg=replicate_pg, orchestrator=self.txn)
        self._register_ulfm_hook(ulfm_opts=ulfm_opts)

        self._micro_in_window = 0

        logger.debug(
            f"[Rank {rank}] HSDPULFMTrainingManager initialized: "
            f"policy={policy_type}, grad_accum={grad_accum_steps}"
        )

    @staticmethod
    def _strip_fsdp_replicate_scaling(fsdp_model) -> None:
        """Set predivide=1, postdivide=shard_size on every FSDP unit so
        FSDP divides only by the (stable) shard dimension; the replicate-axis
        divisor is then applied solely by the parent's _get_grad_div_factor
        (= target_replicate × grad_accum under StaticWorldPolicy, constant
        across failures).
        """
        states = _get_fsdp_states(fsdp_model)
        if not states:
            raise RuntimeError("HSDP manager: no FSDP states found in model")
        for state in states:
            if state.process_group is None:
                raise RuntimeError(
                    "HSDP manager: FSDP state missing shard process_group"
                )
            shard_size = state.process_group.size()
            state._gradient_predivide_factor = 1.0
            state._gradient_postdivide_factor = float(shard_size)

    def _register_ulfm_hook(self, ulfm_opts):
        """Register the FSDP-shaped deferred ULFM hook.

        The hook fires on the sync microstep with ``flat_param._saved_grad_shard``
        (the local-shard accumulator). It snapshots + queues the buffer on the
        orchestrator and returns a pre-resolved Future — NO MPI is issued
        during backward. ``_fire_cross_replica_allreduces`` later drains the
        orchestrator's queue and does the actual ULFM allreduces.
        """
        hook = create_ulfm_hsdp_hook(ulfm_opts=ulfm_opts)
        self.ddp_model.register_comm_hook(state=self._hook_state, hook=hook)
        logger.debug(
            f"[Rank {self.txn._rank}] HSDP deferred ULFM hook registered on replicate_pg"
        )
        self._hook = hook

    def _fire_cross_replica_allreduces(self) -> None:
        """Drain the orchestrator's deferred-bucket queue: for each queued
        ``_saved_grad_shard`` (snapshotted + queued by the hook during
        backward's sync microstep), fire ULFM allreduce, wait, and run
        ``handle_work_completion`` (preceded by the ``grad-sync:*``
        failure-injection point).

        Mirrors nanotron's pattern: hook fires during backward but does NO
        MPI; this method runs AFTER backward and does all MPI sequentially.
        No overlap with shard_pg's NCCL ops → no CUDA-aware MPI starvation.
        """
        self.txn.fire_deferred_allreduces()

    # ------------------------------------------------------------------
    # Loop-driven API (mirrors NanotronULFMTrainingManager). The trainer
    # (main_hsdp.py) drives the outer iteration / inner microbatch loops,
    # the per-iteration restore-mode dispatch, optimizer step, and grad
    # clipping. The manager exposes microbatch_step + query/action helpers.
    # ------------------------------------------------------------------

    # ---- Query methods ----

    def get_effective_n_microbatches(self) -> int:
        """Microbatches for the first pass of an iteration (policy-current grad_accum)."""
        return self.txn.curr_grad_accum_steps

    def get_n_extra_microbatches(self) -> int:
        """Microbatches for an extended pass at a policy boundary."""
        return self.txn.num_policy_boundary_steps

    def get_restore_mode(self) -> GradRestoreMode:
        return self._get_restore_mode()

    def is_at_policy_boundary(self) -> bool:
        return self.txn.at_policy_boundary

    # ---- Lifecycle ----

    def prepare_iteration(self, is_first_pass: bool) -> None:
        """Called once at the start of an iteration pass (first or extended)."""
        if is_first_pass:
            self._notify_window_start()
        self._on_grad_sync_step()
        self._hook_state.reset_unit_counter()

    def on_world_consensus(self):
        """Global ULFM consensus on world_pg; returns its work (None without world_pg)."""
        if self.world_pg is None:
            return
        ulfm_opts = ULFM.ULFMOptions(auto_repair=True)
        work = self.world_pg.consensus(ulfm_opts)
        work.wait()
        return work

    def on_consensus_step(self) -> None:
        """Cross-DP consensus barrier; clears quiesce on success."""
        self._on_consensus_step()

    # ---- Microbatch step (forward + backward only) ----

    def microbatch_step(
        self,
        batch_idx: int,
        micro_idx: int,
        n_micro: int,
        data,
        target,
        criterion,
        scaler=None,
    ):
        """Run one microbatch's forward + backward. The trainer drives the
        per-iteration loop, so this method does NOT touch the optimizer or
        the cross-replica reduce. The last microstep (micro_idx == n_micro-1)
        runs in sync mode (FSDP _reduce_grad runs intra-shard reduce_scatter
        + accumulates into _saved_grad_shard); earlier microsteps run under
        no_sync (still reduce_scatter+accumulate after the fp32-fold patch,
        but no autograd-side reshard wait).
        """
        self.txn.update_progress(
            microbatch_idx=micro_idx,
            total_microbatches=n_micro,
            macrobatch_idx=batch_idx,
        )
        is_last = (micro_idx == n_micro - 1)
        if is_last:
            ctx = contextlib.nullcontext()
            self._on_grad_sync_step()
        else:
            ctx = self.ddp_model.no_sync()

        with ctx:
            output = self.ddp_model(data)
            loss = self._may_zero_grad(criterion(output, target))
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()

        self._on_microbatch_complete(micro_idx)
        return float(loss.detach())

    # ---- Cross-replica allreduce (after inner microbatch loop) ----

    def fire_cross_replica_allreduces(self) -> None:
        """Public entry point — see _fire_cross_replica_allreduces for details."""
        self._fire_cross_replica_allreduces()

    # ---- Restore actions ----

    def start_blocking_restore(self) -> None:
        self._start_restore_gradients_blocking(allow_internal_retry=False)

    def start_nonblocking_restore(self) -> None:
        self._start_restore_gradients_non_blocking()

    def wait_restore_before_backward(self) -> None:
        self._wait_restore_before_backward()

    # ---- Optimizer-step helpers ----

    def normalize_gradients(self) -> None:
        """Apply the manager's grad_div_factor to every param.grad in-place."""
        div = self._get_grad_div_factor()
        for p in self.ddp_model.parameters():
            if p.grad is not None:
                p.grad.div_(div)

    def optimizer_step(self, optimizer, scaler=None) -> None:
        """Run optimizer.step (with optional scaler), zero_grad, and notify
        the orchestrator that the step was committed. Caller is responsible
        for normalize_gradients() and grad clipping BEFORE calling this.
        """
        # ``optimizer`` failure: the step's cross-replica collectives (gradient
        # sync, loss logging) are done; what follows is local. Survivors find
        # the failure at the next step's consensus / gradient sync.
        _sim = get_failure_simulator()
        if _sim is not None:
            with _sim.may_fail_here("optimizer"):
                pass
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        optimizer.zero_grad(set_to_none=False)
        self._on_step_committed()

    # ---- Diagnostics ----

    def compute_grad_norm(self) -> float:
        """Sum of per-param L2 norms on this rank's local shard. Mirrors the
        NCCL baseline; not a true global norm but comparable across runs."""
        total = 0.0
        for p in self.ddp_model.parameters():
            if p.grad is not None:
                total += float(torch.norm(p.grad.detach()).cpu())
        return total
