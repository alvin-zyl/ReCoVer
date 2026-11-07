#!/usr/bin/env python3
"""
Fault Tolerance Policy Manager

This module encapsulates all fault tolerance decision-making logic, making the
training manager agnostic to specific fault tolerance strategies.

The Policy determines:
1. When failures require quiescing vs. just communicator repair
2. How to adjust gradient accumulation windows based on failure patterns
3. Whether gradient restoration should be blocking or non-blocking
4. Whether a step can be committed or needs additional forward/backward passes
"""

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class GradRestoreMode(Enum):
    """How gradients should be restored after failure."""

    BLOCKING = "blocking"  # Synchronous restoration with re-reduction
    NON_BLOCKING = "non_blocking"  # Asynchronous restoration overlapped with forward
    SKIP = "skip"  # No restoration needed


class FailureResponse(Enum):
    """How to respond to a detected failure."""

    QUIESCE_AND_RESTORE = (
        "quiesce_and_restore"  # Stop all comms, restore grads next iter
    )
    REPAIR_AND_CONTINUE = "repair_and_continue"  # Just repair comm, keep going
    ABORT = "abort"  # Unrecoverable failure


@dataclass
class PolicyDecision:
    """
    Decision output from the policy about how to handle current state.
    """

    # Failure handling
    failure_response: FailureResponse
    should_quiesce: bool
    should_manual_repair: bool

    # Gradient management
    grad_restore_mode: GradRestoreMode

    # State tracking
    at_policy_boundary: (
        bool  # Are we at the boundary of policy where re-configuration is needed?
    )

    # Optional fields for advanced policies (fixed world)
    num_policy_boundary_steps: Optional[int] = None  # Extra steps needed at boundary
    num_nonzero_grad_procs: Optional[int] = (
        None  # Num procs with zero grads at the last boundary step
    )


@dataclass
class PolicyState:
    """Current status of the policy for per-step/window instructions."""

    at_iteration_boundary: bool  # Is current step at a step boundary?
    failures_this_window: int  # Failures encountered in current accumulation window
    total_failures: int  # Total failures encountered so far
    total_recoveries: int  # Total successful recoveries so far


@dataclass
class FailureEvent:
    """Information about a detected failure."""

    failed_ranks: List[int]
    current_microbatch_idx: int  # Which microbatch in accumulation window
    total_microbatches: int  # Total microbatches in window
    world_epoch: int  # ProcessGroup epoch (increments on repair)
    curr_rank: int  # Current rank after repairs
    curr_size: int  # Current world size after repairs
    failed_major: int  # Number of failed major processes
    failed_minor: int  # Number of failed minor processes
    failed_major_spares: int  # Number of failed major spare processes
    failed_minor_spares: int  # Number of failed minor spare processes
    failed_boundary_minors: int  # Number of failed boundary minor processes
    curr_num_major_procs: int  # Current number of major processes
    curr_num_minor_procs: int  # Current number of minor processes
    curr_num_major_spares: int  # Current number of major spare processes
    curr_num_minor_spares: int  # Current number of minor spare processes
    curr_contributed: int  # Number of gradients globally contributed by procs that are not being zeroed
    at_policy_boundary: bool = False  # Whether failure occurred at policy boundary
    curr_num_boundary_minor_procs: Optional[int] = (
        None  # Current number of boundary minor processes
    )
    curr_boundary_contributed: int = 0  # Global sum of boundary-phase gradient contributions


class FaultTolerancePolicy:
    """
    Base class for fault tolerance policies.

    Subclass this to implement different fault tolerance strategies.
    """

    def __init__(
        self, initial_grad_accum_steps: int = 1, enable_auto_repair: bool = True
    ):
        """
        Args:
            initial_grad_accum_steps: Initial gradient accumulation window size
            enable_auto_repair: Whether to enable automatic communicator repair
        """
        self.grad_accum_steps = initial_grad_accum_steps
        self.enable_auto_repair = enable_auto_repair

        # State tracking
        self._current_microbatch = 0
        self._failures_this_window = 0
        self._recoveries_this_window = 0
        self._total_failures = 0
        self._total_recoveries = 0

    @property
    def current_grad_accum_steps(self) -> int:
        """Get the current gradient accumulation steps."""
        return self.grad_accum_steps

    def on_window_start(self):
        """
        Called at the start of each accumulation window.
        """
        self._current_microbatch = 0
        self._failures_this_window = 0

    def on_failure(self, failure_event: FailureEvent) -> PolicyDecision:
        """
        Decide how to handle a failure event.

        Args:
            failure_event: Information about the failure
            orchestrator: StepTxnOrchestrator for state queries and actions

        Returns:
            PolicyDecision with instructions for training manager
        """
        raise NotImplementedError("Subclasses must implement on_failure()")

    def on_recovery(self) -> None:
        """
        Called after a successful recovery.
        """
        self._recoveries_this_window += 1
        self._total_recoveries += 1

    def on_microbatch_complete(self, microbatch_idx: int) -> PolicyState:
        """
        Called after each microbatch completes successfully.

        Args:
            microbatch_idx: Index of completed microbatch in window

        Returns:
            PolicyDecision about whether to commit or continue accumulating
        """
        self._current_microbatch = microbatch_idx

        at_boundary = (microbatch_idx + 1) >= self.current_grad_accum_steps

        return PolicyState(
            at_iteration_boundary=at_boundary,
            failures_this_window=self._failures_this_window,
            total_failures=self._total_failures,
            total_recoveries=self._total_recoveries,
        )

    def get_stats(self):
        """Get policy statistics."""
        return {
            "total_failures": self._total_failures,
            "total_recoveries": self._total_recoveries,
            "success_rate": self._total_recoveries / max(1, self._total_failures),
            "current_grad_accum_steps": self.grad_accum_steps,
        }

    def update_policy(self):
        """
        Update internal policy state if needed at failure when boundary not crossed.
        Default implementation does nothing.
        """
        pass

    def advance_policy(self):
        """
        Advance internal policy state after completing a policy boundary adjustment.
        Default implementation does nothing.
        """
        raise NotImplementedError("Subclasses must implement advance_policy()")

    def get_num_major_procs(self) -> int:
        """
        Get the number of major processes as per current policy.
        Default implementation returns total processes (no minor/major split).
        """
        return NotImplementedError("Subclasses must implement get_num_major_procs()")

    def get_num_minor_procs(self) -> int:
        """
        Get the number of minor processes as per current policy.
        Default implementation returns zero (no minor/major split).
        """
        return NotImplementedError("Subclasses must implement get_num_minor_procs()")

    def get_num_major_spare_procs(self) -> int:
        """
        Get the number of major spare processes as per current policy.
        Default implementation returns zero (no spares).
        """
        return NotImplementedError(
            "Subclasses must implement get_num_major_spare_procs()"
        )

    def get_num_minor_spare_procs(self) -> int:
        """
        Get the number of minor spare processes as per current policy.
        Default implementation returns zero (no spares).
        """
        return NotImplementedError(
            "Subclasses must implement get_num_minor_spare_procs()"
        )

    def get_minor_proc_grad_accum_steps(self) -> int:
        """
        Get the gradient accumulation steps for minor processes as per current policy.
        Default implementation returns zero (no minor/major split).
        """
        return NotImplementedError(
            "Subclasses must implement get_minor_proc_grad_accum_steps()"
        )


class AdaptiveWorldPolicy(FaultTolerancePolicy):
    """
    Adaptive World-Size Policy: Simple repair and continue, agnostic to world size changes.

    This policy:
    - Accepts any number of process failures
    - Simply repairs communicator and continues training
    - Adjusts gradients for new world size automatically
    - No gradient restoration needed (always uses current surviving processes)
    - Suitable for elastic training where world size can vary

    Strategy:
    - On failure: Repair communicator, continue with survivors
    - No quiescing or gradient restoration
    - Training adapts to whatever world size remains
    """

    def __init__(
        self, initial_grad_accum_steps: int = 1, enable_auto_repair: bool = True
    ):
        super().__init__(initial_grad_accum_steps, enable_auto_repair)

    def on_failure(self, failure_event: FailureEvent) -> PolicyDecision:
        """
        Handle failure with simple repair and continue strategy.

        No matter when or how many failures occur, just repair and keep going.
        """
        self._total_failures += 1
        self._failures_this_window += 1

        num_failed = len(failure_event.failed_ranks)

        # Policy decision logging (INFO level - detailed trace)
        logger.info(
            f"[AdaptiveWorldPolicy] Failure detected: {num_failed} processes failed, "
            f"gradient restoration mode is blocking."
        )

        # Always just repair and continue - no quiesce, no restoration
        decision = PolicyDecision(
            failure_response=FailureResponse.REPAIR_AND_CONTINUE,
            should_quiesce=False,
            should_manual_repair=not self.enable_auto_repair,
            grad_restore_mode=GradRestoreMode.BLOCKING,
            at_policy_boundary=False,  # Not used in this policy
        )

        return decision

    def get_minor_proc_grad_accum_steps(self):
        return self.grad_accum_steps


class StaticWorldPolicy(FaultTolerancePolicy):
    """
    Fixed World-Size Policy: Complex policy that tries to maintain a target world size.

    This policy has configurable "boundaries" that define acceptable operating conditions:
    - min_world_size: Minimum acceptable number of processes
    - max_failures_per_window: Maximum failures allowed before taking corrective action
    - target_world_size: Desired world size to maintain

    When within boundaries:
    - Gradients may be corrupted, need restoration
    - Quiesce and restore to maintain consistency

    When crossing boundaries:
    - Policy may need to adjust grad_accum_steps
    - May need to trigger more aggressive recovery (e.g., restart failed processes)
    - Different restoration strategies based on severity
    """

    def __init__(
        self,
        initial_grad_accum_steps: int = 1,
        enable_auto_repair: bool = True,
        initial_world_size: Optional[int] = None,
        target_world_size: Optional[int] = None,
        adaptive_grad_accum: bool = False,
    ):
        """
        Args:
            target_world_size: Target world size to maintain (None = use initial world size)
            min_world_size: Minimum acceptable world size (None = no minimum)
            max_failures_per_window: Max failures per window before boundary crossed
            adaptive_grad_accum: Adjust grad accumulation based on world size changes
        """
        super().__init__(initial_grad_accum_steps, enable_auto_repair)

        self.target_world_size = target_world_size
        self.adaptive_grad_accum = adaptive_grad_accum
        assert initial_world_size is not None, "initial_world_size must be provided"
        if self.target_world_size is None:
            self.target_world_size = initial_world_size
            logger.warning(
                f"[FixedWorldSizePolicy] No target_world_size specified, using initial_world_size={initial_world_size}"
            )
            if not self.adaptive_grad_accum:
                self.adaptive_grad_accum = True
                logger.warning(
                    "[FixedWorldSizePolicy] Enabling adaptive_grad_accum since target_world_size equals initial_world_size"
                )

        assert (
            self.target_world_size <= initial_world_size
        ), f"target_world_size ({self.target_world_size}) cannot exceed initial_world_size ({initial_world_size})"

        self._initial_grad_accum_steps = initial_grad_accum_steps
        # Reference baseline used by _on_policy_boundary as the starting point
        # for the next extension. Defaults to self.grad_accum_steps; promoted
        # to self._current_grad_accum_steps only when a failure is observed
        # during an active boundary extended pass (boundary_contributed > 0),
        # so stacked boundaries extend on top of the already-extended window.
        self._boundary_ref_grad_accum_steps = initial_grad_accum_steps
        self._current_grad_accum_steps = (
            initial_grad_accum_steps  # subject to change at policy boundaries
        )
        self._current_world_size = (
            initial_world_size  # Will be updated on first failure
        )
        self._minor_proc_grad_accum_steps = initial_grad_accum_steps
        self._num_major_spares = self._current_world_size - self.target_world_size
        self._num_minor_spares = 0

    @property
    def target_batch_size(self) -> int:
        """Get the effective target batch size considering grad accumulation."""
        return self.target_world_size * self._initial_grad_accum_steps

    @property
    def current_grad_accum_steps(self):
        return self._current_grad_accum_steps

    def on_failure(self, failure_event: FailureEvent) -> PolicyDecision:
        """
        Handle failure with boundary-aware logic.

        Decision tree:
        2. Check if crossing policy boundaries (world size, failure budget)
           -> If yes: More aggressive action (quiesce, restore, maybe adjust config)
           -> If no: Standard quiesce and restore
        """
        self._total_failures += 1
        self._failures_this_window += 1
        self._current_world_size = failure_event.curr_size

        # Check if we're at a boundary condition
        at_policy_boundary = failure_event.at_policy_boundary
        if at_policy_boundary:
            num_policy_boundary_steps, num_nonzero_grad_procs = (
                self._on_policy_boundary(failure_event)
            )
            decision = PolicyDecision(
                failure_response=FailureResponse.REPAIR_AND_CONTINUE,
                should_quiesce=True,
                should_manual_repair=not self.enable_auto_repair,
                grad_restore_mode=GradRestoreMode.NON_BLOCKING,
                at_policy_boundary=at_policy_boundary,
                num_policy_boundary_steps=num_policy_boundary_steps,
                num_nonzero_grad_procs=num_nonzero_grad_procs,
            )
        else:
            self.update_policy(failure_event)
            decision = PolicyDecision(
                failure_response=FailureResponse.REPAIR_AND_CONTINUE,
                should_quiesce=False,
                should_manual_repair=not self.enable_auto_repair,
                grad_restore_mode=GradRestoreMode.BLOCKING,
                at_policy_boundary=at_policy_boundary,
            )

        return decision

    def update_policy(self, failure_event: FailureEvent):
        """
        Update internal policy state if needed at failure when boundary not crossed.
        Default implementation does nothing.
        """
        self._current_world_size = failure_event.curr_size
        self._num_major_spares = failure_event.curr_num_major_spares
        self._num_minor_spares = failure_event.curr_num_minor_spares
        logger.info(
            f"[StaticWorldPolicy] Updated policy state after non-boundary failure: current world size {self._current_world_size}, "
            f"number of major spares: {self._num_major_spares}, number of minor spares: {self._num_minor_spares}"
        )

    def _on_policy_boundary(self, failure_event: FailureEvent):
        """
        Handle actions needed when at a policy boundary.
        """
        # If this failure was observed during an active boundary extended
        # pass, fold the boundary-phase contributions into curr_contributed
        # and use the already-extended _current_grad_accum_steps as the new
        # reference baseline. Otherwise start from the regular grad_accum_steps.
        if failure_event.curr_boundary_contributed > 0:
            failure_event.curr_contributed += failure_event.curr_boundary_contributed
            failure_event.curr_boundary_contributed = 0
            self._boundary_ref_grad_accum_steps = self._current_grad_accum_steps
        else:
            self._boundary_ref_grad_accum_steps = self.grad_accum_steps

        target_world_size_with_acc = (
            self.target_world_size * self._initial_grad_accum_steps
        )

        num_policy_boundary_steps = 1
        while (
            failure_event.curr_contributed
            + failure_event.curr_size * num_policy_boundary_steps
        ) < target_world_size_with_acc:
            num_policy_boundary_steps += 1

        self._current_grad_accum_steps = (
            self._boundary_ref_grad_accum_steps + num_policy_boundary_steps
        )

        num_zero_grad_procs = (
            failure_event.curr_contributed
            + failure_event.curr_size * num_policy_boundary_steps
            - target_world_size_with_acc
        )
        logger.warning(
            f"[StaticWorldPolicy] At policy boundary: current world size {failure_event.curr_size}, "
            f"target batch size {target_world_size_with_acc}, current gradients (global) on hand: {failure_event.curr_contributed}, "
            f"temporarily adjusting grad_accum_steps by {num_policy_boundary_steps}: "
            f"{self._boundary_ref_grad_accum_steps} -> {self._current_grad_accum_steps}. "
            f"Number of zero-grad procs at the last boundary step: {num_zero_grad_procs}."
        )
        num_nonzero_grad_procs = failure_event.curr_size - num_zero_grad_procs
        return num_policy_boundary_steps, num_nonzero_grad_procs

    def advance_policy(self):
        """
        Advance internal policy state after completing a policy boundary adjustment.
        """
        target_batch_size = (
            self.target_world_size * self._initial_grad_accum_steps
        )  # This is fixed
        while self._current_world_size * self.grad_accum_steps < target_batch_size:
            self.grad_accum_steps += (
                1  # Increase grad accum steps to be >= target batch size
            )
        
        # Important: _current_grad_accum_steps could be larger at the boundary, need to sync with advanced policy
        self._current_grad_accum_steps = self.grad_accum_steps

        min_num_major_procs = (
            target_batch_size // self.grad_accum_steps
        )  # Mimimum procs for the new grad accum steps
        # Adjust minor proc grad accum steps to fill the gap
        self._minor_proc_grad_accum_steps = (
            target_batch_size - min_num_major_procs * self.grad_accum_steps
        )
        actual_batch_size = (
            min_num_major_procs * self.grad_accum_steps
            + self._minor_proc_grad_accum_steps
        )
        assert actual_batch_size == target_batch_size, (
            f"Invalid policy advancement, target batch size {target_batch_size} != actual batch size {actual_batch_size} "
            f"(num major procs: {min_num_major_procs}, grad_accum: {self.grad_accum_steps}); "
            + f"minor procs grad_accum: {self._minor_proc_grad_accum_steps})"
            if self._minor_proc_grad_accum_steps
            else ""
        )

        num_minors = 1 if self._minor_proc_grad_accum_steps > 0 else 0

        # Infer number of spares if possible
        if min_num_major_procs + num_minors < self._current_world_size:
            total_num_spares = self._current_world_size - (
                min_num_major_procs + num_minors
            )
            _num_major_spares = total_num_spares
            _num_minor_spares = 0
            while (
                num_minors
                and _num_major_spares > 1
                and _num_major_spares > int(total_num_spares * 0.8)
            ):
                _num_minor_spares += 1
                _num_major_spares -= 1
            self._num_major_spares = _num_major_spares
            self._num_minor_spares = _num_minor_spares
        else:
            self._num_major_spares = 0
            self._num_minor_spares = 0

        logger.warning(
            f"[StaticWorldPolicy] Advanced policy after boundary adjustment: world size: {self._current_world_size}, "
            f"global batch size: {actual_batch_size}, number of major procs: {min_num_major_procs}, "
            f"gradient accumlation steps: {self.grad_accum_steps}, "
            + (
                f"number of minor procs: 1, gradient accumulation steps for minor procs: {self._minor_proc_grad_accum_steps}, "
                if num_minors else "number of minor procs: 0, "
            )
            + f"number of major spares: {self._num_major_spares}, number of minor spares: {self._num_minor_spares}"
        )
        return (
            min_num_major_procs,
            num_minors,
            self._num_major_spares,
            self._num_minor_spares,
        )

    def get_num_major_procs(self) -> int:
        """
        Get the number of major processes as per current policy.
        """
        target_batch_size = (
            self.target_world_size * self._initial_grad_accum_steps
        )  # This is fixed
        min_num_major_procs = target_batch_size // self.grad_accum_steps
        return min_num_major_procs

    def get_num_minor_procs(self) -> int:
        """
        Get the number of minor processes as per current policy.
        """
        return 1 if self._minor_proc_grad_accum_steps > 0 else 0

    def get_num_major_spare_procs(self) -> int:
        """
        Get the number of major spare processes as per current policy.
        """
        return self._num_major_spares

    def get_num_minor_spare_procs(self) -> int:
        """
        Get the number of minor spare processes as per current policy.
        """
        return self._num_minor_spares

    def get_minor_proc_grad_accum_steps(self) -> int:
        """
        Get the gradient accumulation steps for minor processes as per current policy.
        """
        return self._minor_proc_grad_accum_steps


# Factory function for easy policy creation
def create_policy(policy_type: str = "adaptive", **kwargs) -> FaultTolerancePolicy:
    """
    Create a fault tolerance policy.

    Args:
        policy_type: "adaptive" (AdaptiveWorldSizePolicy) or "fixed" (FixedWorldSizePolicy)
        **kwargs: Policy-specific arguments

    Returns:
        FaultTolerancePolicy instance
    """
    policies = {
        "adaptive": AdaptiveWorldPolicy,
        "static": StaticWorldPolicy,
    }

    if policy_type not in policies:
        raise ValueError(
            f"Unknown policy type: {policy_type}. Choose from {list(policies.keys())}"
        )

    return policies[policy_type](**kwargs)
