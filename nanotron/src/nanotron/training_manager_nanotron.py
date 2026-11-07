# src/nanotron/training_manager_nanotron.py
"""
NanotronULFMTrainingManager: Adapts ULFMTrainingManager for nanotron's
pipeline-engine-based training loop.

Gradient sync strategy: DEFERRED PER-BUCKET ALLREDUCE via DDP comm hook.
  - The pipeline engine uses no_sync() on early microbatches and
    ddp_trigger_sync_in_bwd() on the last microbatch.
  - On the last microbatch, DDP hooks fire per-bucket. Our deferred hook
    snapshots each bucket and returns a pre-resolved Future (no blocking).
  - After the pipeline completes, fire_deferred_allreduces() triggers the
    actual ULFM allreduces per-bucket via the orchestrator.
  - Per-bucket failure handling through the orchestrator's handle_work_completion.

The trainer drives the loop — this manager exposes query/action methods
that the training_step while-loop uses to decide what to do next:
  - get_restore_mode(), is_at_policy_boundary(), get_n_extra_microbatches()
  - start_blocking_restore(), start_nonblocking_restore(), wait_restore_before_backward()
  - normalize_gradients(), optimizer_step(), on_consensus_step(), fire_deferred_allreduces()
"""

import logging

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import ulfm_collectives as ULFM
from ulfm_collectives.training_manager import ULFMTrainingManager
from ulfm_collectives.orchestrator import StepTxnOrchestrator
from ulfm_collectives.failure_simulator import get_failure_simulator
from ulfm_collectives.policy import create_policy, GradRestoreMode

logger = logging.getLogger(__name__)


class NanotronULFMTrainingManager(ULFMTrainingManager):
    """
    ULFMTrainingManager adapted for nanotron's pipeline-engine training loop.

    Usage pattern in ULFMDistributedTrainer.training_step() while-loop:

        manager.prepare_iteration(...)
        while True:
            n_micro = manager.get_effective_n_microbatches() or get_n_extra_microbatches()
            pipeline_engine.train_batch_iter(n_micro, ...)
            ok = manager.fire_deferred_allreduces()
            manager.on_consensus_step()
            mode = manager.get_restore_mode()
            if mode == SKIP: break
            if mode == BLOCKING:
                manager.start_blocking_restore()
                if manager.is_at_policy_boundary():
                    manager.start_nonblocking_restore()
                    continue
                break
            if mode == NON_BLOCKING:
                manager.start_nonblocking_restore()
                continue
        manager.normalize_and_step(optimizer)
    """

    def __init__(
        self,
        ddp_model: DistributedDataParallel,
        dp_pg: ULFM.ProcessGroupULFM,
        grad_accum_steps: int = 1,
        policy_type: str = "static",
        grad_accumulator=None,
        **policy_kwargs,
    ):
        if not isinstance(dp_pg, ULFM.ProcessGroupULFM):
            raise ValueError(
                f"dp_pg must be a ProcessGroupULFM, got {type(dp_pg)}. "
                "Ensure you are using ULFMParallelContext."
            )

        self.failure_strategy = "continue"
        self.process_group = dp_pg

        policy = create_policy(
            policy_type=policy_type,
            initial_grad_accum_steps=grad_accum_steps,
            enable_auto_repair=True,
            **policy_kwargs,
        )

        ulfm_opts = self._create_ulfm_opts(policy_type)

        rank = dist.get_rank()
        self.txn = StepTxnOrchestrator(
            rank=rank, pg=dp_pg, policy=policy, ulfm_opts=ulfm_opts
        )

        self.ddp_model = ddp_model
        self._micro_in_window = 0
        self._fp32_accumulator = grad_accumulator

        # Register deferred comm hook — snapshots + queues per-bucket,
        # returns pre-resolved Futures so finalize_backward never blocks.
        if grad_accumulator is not None:
            self._register_fp32_deferred_hook(ulfm_opts, grad_accumulator)
        else:
            self._register_deferred_hook(ulfm_opts)

        logger.debug(
            f"[Rank {rank}] NanotronULFMTrainingManager initialized "
            f"(deferred hook registered, fp32_accum={'yes' if grad_accumulator else 'no'}): "
            f"policy={policy_type}, grad_accum={grad_accum_steps}"
        )

    def _register_deferred_hook(self, ulfm_opts):
        from ulfm_collectives.ulfm_hook import create_ulfm_deferred_hook, HookState

        hstate = HookState(pg=self.process_group, orchestrator=self.txn)
        hook = create_ulfm_deferred_hook(ulfm_opts=ulfm_opts)
        self.ddp_model.register_comm_hook(state=hstate, hook=hook)

    def _register_fp32_deferred_hook(self, ulfm_opts, accumulator):
        from ulfm_collectives.ulfm_hook import create_ulfm_fp32_deferred_hook, HookState

        unwrapped = self.ddp_model.module
        module_id_to_prefix = {
            id(module): f"{module_name}."
            for module_name, module in unwrapped.named_modules()
        }
        module_id_to_prefix[id(unwrapped)] = ""

        # Build param_id→name map matching the accumulator's naming convention.
        # Tied parameters use their tied name (same as init_optimizer_and_grad_accumulator).
        param_id_to_name = {
            id(param): (
                param.get_tied_info().get_full_name_from_module_id_to_prefix(
                    module_id_to_prefix=module_id_to_prefix
                )
                if param.is_tied
                else name
            )
            for name, param in unwrapped.named_parameters()
        }

        hstate = HookState(pg=self.process_group, orchestrator=self.txn)
        hook = create_ulfm_fp32_deferred_hook(accumulator, param_id_to_name)
        self.ddp_model.register_comm_hook(state=hstate, hook=hook)
        # Prevent double-adding in accumulator._accumulate_grad()
        accumulator._is_accumulation_sync_step = True

    # ------------------------------------------------------------------
    # Policy query interface (read by trainer each iteration)
    # ------------------------------------------------------------------

    def get_effective_n_microbatches(self) -> int:
        """Current grad_accum_steps from policy (may change after failure)."""
        return self.txn.curr_grad_accum_steps

    def get_effective_dp_size(self) -> int:
        """Current surviving DP world size."""
        return self.txn.curr_world_size

    def get_restore_mode(self) -> GradRestoreMode:
        """Current restore plan set by handle_work_completion."""
        return self._get_restore_mode()

    def is_at_policy_boundary(self) -> bool:
        """Whether the orchestrator is at a policy boundary."""
        return self.txn.at_policy_boundary

    def get_n_extra_microbatches(self) -> int:
        """Number of extra microbatches needed at a policy boundary."""
        return self.txn.num_policy_boundary_steps

    def should_zero_grad(self) -> bool:
        """At policy boundary, some procs must zero grads on last extended step."""
        return self.txn.should_zero_grad

    # ------------------------------------------------------------------
    # Nanotron lifecycle interface
    # ------------------------------------------------------------------

    def prepare_iteration(self, iteration_step: int, n_microbatches: int, is_first_pass: bool) -> None:
        """Called once at the start of each update step (not on extended passes)."""
        if is_first_pass:
            self._notify_window_start()
        self._on_grad_sync_step()

    def fire_deferred_allreduces(self) -> bool:
        """
        Post-pipeline per-bucket ULFM gradient allreduce.

        Returns True if all buckets succeeded, False if any failure occurred.
        On failure, fp32 scatter-back is skipped so the accumulator retains
        locally-accumulated (correct) values.
        """
        return self.txn.fire_deferred_allreduces()

    # ------------------------------------------------------------------
    # Actions (called by the training_step loop)
    # ------------------------------------------------------------------

    def start_blocking_restore(self, allow_internal_retry: bool = True):
        """Blocking gradient restoration with re-reduction.

        Args:
            allow_internal_retry: If True, orchestrator retries re-reductions
                internally until success or policy boundary. Pass False when
                the caller cross-synchronizes DP groups after this call, so
                all ranks take the same outer-loop path on re-reduction
                failure (see orchestrator.restore_gradients_blocking).
        """
        self._start_restore_gradients_blocking(allow_internal_retry=allow_internal_retry)

    def start_nonblocking_restore(self):
        """Start async gradient restoration (overlaps with next forward pass)."""
        self._start_restore_gradients_non_blocking()

    def wait_restore_before_backward(self):
        """Wait for non-blocking restore to complete (called before last backward)."""
        self._wait_restore_before_backward()

    def on_consensus_step(self):
        """Run ULFM consensus collective across DP ranks."""
        self._on_consensus_step()

    def on_grad_sync_step(self):
        """Prepare for gradient sync (set quiesce=False)."""
        self._on_grad_sync_step()

    def normalize_gradients(self):
        div_factor = self._get_grad_div_factor()

        if self._fp32_accumulator is not None:
            # FP32 path: normalize the contiguous fp32 grad buffer
            self._fp32_accumulator._contiguous_fp32_grad_buffer.div_(div_factor)
            # Assign fp32 grads to fp32 params for optimizer
            for name in self._fp32_accumulator.parameters:
                fp32_param = self._fp32_accumulator.parameters[name]["fp32"]
                fp32_param.grad = self._fp32_accumulator.get_grad_buffer(name)
        else:
            # BF16 path: normalize param.grad directly
            for p in self.ddp_model.parameters():
                if p.grad is not None:
                    p.grad.div_(div_factor)

    def optimizer_step(self, optimizer):
        """Normalize gradients, run optimizer.step(), commit, reset."""

        # ``optimizer`` failure: the step's cross-replica collectives (gradient
        # sync, loss logging) are done; what follows is local. Survivors find
        # the failure at the next step's consensus / gradient sync.
        _sim = get_failure_simulator()
        if _sim is not None:
            with _sim.may_fail_here("optimizer"):
                pass

        optimizer.step()

        if self._fp32_accumulator is not None:
            # Copy fp32 weights back to bf16 model params
            self._fp32_accumulator.step()
            self._fp32_accumulator.zero_grad()
        else:
            optimizer.zero_grad()

        self._on_step_committed()
        self._micro_in_window = 0
