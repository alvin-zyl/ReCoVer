#!/usr/bin/env python3
"""
Advanced ULFM DDP Example with Generalized Failure Recovery

This example demonstrates the new generalized failure recovery system:
1. WorkULFM for per-operation failure detection
2. ProcessGroup-level recovery methods
3. Flexible failure handling strategies
4. Integration with PyTorch DDP training loops
"""

import contextlib
import logging
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from typing import Union

try:
    import ulfm_collectives as ULFM
except ImportError:
    raise ImportError(
        "ULFM collectives extension not found. Please build the extension first."
    )
from .orchestrator import StepTxnOrchestrator
from .ulfm_hook import create_ulfm_recovery_hook, HookState
from .policy import create_policy, GradRestoreMode

# Logger will be configured based on CLI arguments
logger = logging.getLogger(__name__)


class ULFMTrainingManager:
    """
    Training manager that handles ULFM failure recovery in a structured way.

    Now policy-driven: delegates fault tolerance decisions to a FaultTolerancePolicy.
    """

    def __init__(
        self,
        model,
        grad_accum_steps=1,
        failure_strategy="continue",
        enable_auto_repair=True,
        policy_type="adaptive",  # "adaptive" or "fixed",
        **kwargs,
    ):
        self.failure_strategy = failure_strategy
        self.process_group: Union[ULFM.ProcessGroupULFM, dist.ProcessGroup] = (
            dist.group.WORLD
        )

        # Verify we have ULFM process group
        if not isinstance(self.process_group, ULFM.ProcessGroupULFM):
            raise ValueError("Must use ULFM backend: dist.init_process_group('ulfm')")

        # Create fault tolerance policy
        policy = create_policy(
            policy_type=policy_type,
            initial_grad_accum_steps=grad_accum_steps,
            enable_auto_repair=enable_auto_repair,
            **kwargs,
        )

        if policy_type == "static":
            ulfm_opts = ULFM.ULFMOptions(
                auto_repair=True,
                track_rank_types=True,
            )
        else:
            ulfm_opts = ULFM.ULFMOptions(
                auto_repair=True,
                track_rank_types=False,
            )

        # Create orchestrator for gradient management (with policy)
        rank = dist.get_rank()
        ulfm_opts = self._create_ulfm_opts(policy_type)
        self.txn = StepTxnOrchestrator(
            rank=rank, pg=self.process_group, policy=policy, ulfm_opts=ulfm_opts
        )

        # Create DDP model with ULFM hook
        self.ddp_model = DistributedDataParallel(model)
        self._register_ulfm_hook(ulfm_opts=ulfm_opts)

        self._micro_in_window = 0  # counts [0..K-1] within an accumulation window

        logger.debug(
            f"[Rank {rank}] ULFMTrainingManager initialized with policy={policy_type}, "
            f"auto_repair={enable_auto_repair}, strategy={failure_strategy}"
        )

    def _register_ulfm_hook(self, ulfm_opts):
        """Register ULFM communication hook with recovery logic."""
        hstate = HookState(pg=self.process_group, orchestrator=self.txn)
        hook = create_ulfm_recovery_hook(ulfm_opts=ulfm_opts)
        self.ddp_model.register_comm_hook(state=hstate, hook=hook)
        logger.debug(
            f"[Rank {self.txn._rank}] ULFM hook registered with policy type: {type(self.policy).__name__}"
        )

        # Store hook for statistics access
        self._hook = hook

    @property
    def policy(self):
        """Get the fault tolerance policy from orchestrator."""
        return self.txn.policy

    @property
    def _is_at_grad_sync_step(self):
        return (self._micro_in_window + 1) == self._get_grad_accum_steps()

    def _create_ulfm_opts(self, policy_type):
        if policy_type == "static":
            ulfm_opts = ULFM.ULFMOptions(
                auto_repair=True,
                track_rank_types=True,
            )
        else:
            ulfm_opts = ULFM.ULFMOptions(
                auto_repair=True,
                track_rank_types=False,
            )
        return ulfm_opts

    def _get_grad_accum_steps(self):
        """Get current gradient accumulation steps from policy."""
        return self.txn.curr_grad_accum_steps

    def _get_grad_div_factor(self):
        """Get target world size adjusted for grad accumulation from policy."""
        return self.txn.effective_batch_size

    def _notify_window_start(self):
        """Notify policy that a new accumulation window is starting."""
        self.txn.on_iteration_start()

    def _on_microbatch_complete(self, microbatch_idx):
        """Notify policy that a microbatch completed and get decision."""
        return self.policy.on_microbatch_complete(microbatch_idx)

    def _get_restore_mode(self):
        return self.txn.get_restore_plan()

    def _start_restore_gradients_non_blocking(self):
        self.txn.restore_gradients_non_blocking()

    def _wait_restore_before_backward(self):
        self.txn.wait_restore_before_backward()

    def _may_zero_grad(self, loss):
        if self.txn.should_zero_grad:
            logger.info(
                f"[Rank {self.txn._rank}] Zeroing out gradients on micro batch idx "
                f"{self._micro_in_window} as per policy."
            )
            loss *= 0.0
        return loss

    def _start_restore_gradients_blocking(self, allow_internal_retry: bool = True):
        self.txn.restore_gradients_blocking(allow_internal_retry=allow_internal_retry)

    def _on_step_committed(self):
        self.txn.after_successful_commit()

    def _on_consensus_step(self):
        succeed = self.txn.on_grad_sync_step_consensus()
        if not succeed:
            logger.warning(
                f"[Rank {self.txn._rank}] Failures was detected in consensus step."
            )

    def _on_grad_sync_step(self):
        self.txn.on_grad_sync_step_prepare()

    def reconcile_extended_pass(self, world_work) -> bool:
        """After the world consensus that starts an extended pass: align this
        DP group's decision with the failures that consensus found
        (StepTxnOrchestrator.reconcile_with_world_failures). Returns True if
        the decision changed; the caller then re-reads the extra microbatches."""
        if world_work is None or not world_work.has_failures():
            return False
        return self.txn.reconcile_with_world_failures(world_work.get_failed_world_ranks())

    def train_step(self, batch_idx, data, target, criterion, optimizer, scaler=None):
        """
        One microbatch step; handles grad-accum windows and ULFM healing.
        Now policy-driven: consults policy for all fault tolerance decisions.

        Restoration logic (after failure detected by hook):
        (a) At POLICY boundary (policy.on_failure returns at_iteration_boundary=True):
            - Skip current optimizer step (need_extra_microbatch=True)
            - Add one more grad acc step as "extended step"
            - During forward of extended step: async restore corrupted gradients
            - Wait for restoration, do backward, then optimizer step

        (b) NOT at policy boundary (at_iteration_boundary=False):
            - Continue current iteration normally
            - Before optimizer step: restore gradients blockingly if needed
            - Then do optimizer step

        Returns: (loss_value, stepped_bool)
        """
        self.txn.update_progress(
            microbatch_idx=self._micro_in_window,
            total_microbatches=self._get_grad_accum_steps(),
            macrobatch_idx=batch_idx,
        )

        # === Window start: initialize ===
        if self._micro_in_window == 0:
            # Notify policy of window start
            self._notify_window_start()

            # Zero gradients at start of window
            optimizer.zero_grad(set_to_none=False)

        # === Check if orchestrator flagged need for restoration (set by hook) ===
        # If at policy boundary: use non-blocking restoration during forward
        # (This flag is set by hook when policy.on_failure() returns at_iteration_boundary=True)
        restore_mode = self._get_restore_mode()

        if restore_mode == GradRestoreMode.NON_BLOCKING:
            # At policy boundary: Start async restoration during forward
            logger.info(
                f"[Rank {self.txn._rank}] At policy boundary - starting non-blocking grad restoration"
            )
            self._start_restore_gradients_non_blocking()

        # === Backward (use no_sync on non-last microbatches) ===
        if self._is_at_grad_sync_step:
            ctx = contextlib.nullcontext()
            self._on_grad_sync_step()
        else:
            ctx = self.ddp_model.no_sync()

        with ctx:
            logger.debug(
                f"[Rank {self.txn._rank}] Backward pass at microbatch {self._micro_in_window} "
                f"no_sync={not self._is_at_grad_sync_step}"
            )
            # === Forward ===
            output = self.ddp_model(data)

            # === Wait for async restoration if it was started ===
            if restore_mode == GradRestoreMode.NON_BLOCKING:
                self._wait_restore_before_backward()
                logger.debug(
                    f"[Rank {self.txn._rank}] Non-blocking restoration completed before backward"
                )

            loss = self._may_zero_grad(criterion(output, target))
            if isinstance(ctx, contextlib.nullcontext):
                self._on_consensus_step()
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()

        # === After backward: check with policy if we should commit ===
        state = self._on_microbatch_complete(self._micro_in_window)
        restore_mode = self._get_restore_mode()
        stepped = False

        # === Decide whether to commit optimizer step ===
        if state.at_iteration_boundary:
            logger.debug(
                f"[Rank {self.txn._rank}] Microbatch index: {self._micro_in_window}, grad_acc_step: {self._get_grad_accum_steps()}, "
                f"at iteration boundary"
            )
            self._on_grad_sync_step()

            # If NOT at policy boundary but need restoration: blocking restore before optimizer
            if restore_mode == GradRestoreMode.BLOCKING:
                logger.info(
                    f"[Rank {self.txn._rank}] Not at policy boundary - blocking grad restoration before optimizer"
                )
                self._start_restore_gradients_blocking()
                logger.debug(f"[Rank {self.txn._rank}] Blocking restoration finished.")

            # Optimizer step
            if scaler is None:
                for p in self.ddp_model.parameters():
                    if p.grad is not None:
                        p.grad.div_(self._get_grad_div_factor())
                optimizer.step()
            else:
                if hasattr(scaler, "unscale_"):
                    scaler.unscale_(optimizer)
                for p in self.ddp_model.parameters():
                    if p.grad is not None:
                        p.grad.div_(self._get_grad_div_factor())
                scaler.step(optimizer)
                scaler.update()

            optimizer.zero_grad(set_to_none=False)
            stepped = True

            # Notify orchestrator
            self._on_step_committed()

            # Reset for next window
            self._micro_in_window = 0

        else:
            # Not at window boundary: continue accumulation
            self._micro_in_window += 1

        return float(loss.detach()), stepped

    def get_recovery_stats(self):
        """Get failure and recovery statistics from policy."""
        # Statistics are now tracked centrally in the policy object
        return self.policy.get_stats()
