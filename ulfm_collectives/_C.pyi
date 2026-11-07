# mypy: allow-untyped-defs
# mypy: disable-error-code="type-arg"
from datetime import timedelta
from enum import Enum
from typing import Any, Optional, Union, Tuple

import torch
from torch import Tensor
from torch._C._distributed_c10d import ProcessGroup, Work, AllreduceOptions, Reducer

# This module is defined in csrc/bindings.cpp

def createProcessGroupULFM(ranks: list[int]) -> ProcessGroupULFM: ...

class ULFMFailureHandlingStrategy(Enum):
    CONTINUE_WITH_SURVIVORS = ...
    RESTART_FAILED_PROCESSES = ...
    ABORT_ON_FAILURE = ...

class ULFMOptions:
    auto_repair: bool
    track_rank_types: bool
    auto_elect: bool  # Default True
    consensus_on_rank_types: bool  # Default False
    failure_strategy: ULFMFailureHandlingStrategy
    max_retries: int
    retry_delay_ms: int

    def __init__(
        self,
        auto_repair: bool = False,
        track_rank_types: bool = False,
        auto_elect: bool = True,
        consensus_on_rank_types: bool = False
    ) -> None: ...

    def copy(self) -> "ULFMOptions":
        """Create a copy of this ULFMOptions object."""
        ...

class WorkULFM(Work):
    """ULFM-aware Work class for failure detection (no recovery logic)."""
    def has_failures(self) -> bool:
        """Check if this work detected any failures."""
        ...

    def get_failed_ranks(self) -> list[int]:
        """Get list of ranks that failed during this work."""
        ...

    def was_noop(self) -> bool:
        """Check if this work was marked as a no-op due to failures."""
        ...

    def markNoop(self) -> None:
        """Mark this work as a no-op (used internally for failure handling)."""
        ...

    def get_failure_stats(self) -> Tuple[int, int, int, int, int, bool]:
        """Get failure stats as tuple.

        Returns:
            (failed_majors, failed_minors, failed_major_spares, failed_minor_spares, failed_boundary_minors, at_policy_boundary)
        """
        ...

    def get_current_counts(self) -> Tuple[int, int, int, int, int, int, int]:
        """Get current rank type counts as tuple.

        Returns:
            (majors, minors, major_spares, minor_spares, boundary_minors,
             contributed, boundary_contributed)
            where contributed / boundary_contributed are the global sums of
            regular-phase and boundary-phase gradient contributions across all
            surviving ranks at the time count_rank_types was called.
        """
        ...

class ProcessGroupULFM(ProcessGroup):
    def ulfm_allreduce(
        self,
        tensors: list[Tensor],
        opts: AllreduceOptions = ...,
        ulfm_opts: ULFMOptions = ...
    ) -> WorkULFM: ...
    
    # Legacy recovery methods (backward compatibility)
    def repair_communicator(self) -> bool:
        """Legacy method: repair the MPI communicator after failures."""
        ...
    
    def notify_all_ranks_of_failure(self) -> None:
        """Legacy method: notify all ranks of detected failures."""
        ...
    
    def check_for_failures(self) -> bool:
        """Legacy method: check if there are any detected failures."""
        ...
    
    # New modular failure recovery system
    def detect_and_recover_failures(
        self,
        auto_repair: bool = True
    ) -> Tuple[bool, list[int]]:
        """
        Comprehensive 5-step failure detection and recovery workflow.

        Follows ULFM best practices:
        1. Notice failure (comm_agree first)
        2. Get failed ranks (while communicator is revoked)
        3. Ack failures (MUST come before collective operations)
        4. Agree on failed ranks (now safe after ack)
        5. Repair communicator if needed and requested

        Args:
            auto_repair: Whether to automatically repair the communicator

        Returns:
            Tuple of (success: bool, failed_ranks: list[int])
        """
        ...

    def consensus(
        self,
        ulfm_opts: ULFMOptions = ...
    ) -> WorkULFM:
        """
        Perform consensus operation for failure detection and recovery.

        This is a collective operation that all ranks must call. It performs
        failure detection and optionally recovers the communicator based on
        the provided ULFM options.

        Args:
            ulfm_opts: Options controlling failure handling behavior

        Returns:
            WorkULFM object that can be queried for failures via has_failures()
            and get_failed_ranks()
        """
        ...

    # State management methods
    def set_quiesce(self, v: bool) -> None:
        """Set the quiesce state of the process group."""
        ...
    
    def is_quiesced(self) -> bool:
        """Check if the process group is currently quiesced."""
        ...
    
    def worldEpoch(self) -> int:
        """Get the current world epoch (increments after communicator repairs)."""
        ...

    def current_rank(self) -> int:
        """Get the current rank (may change after communicator repairs)."""
        ...

    def current_size(self) -> int:
        """Get the current world size (may change after communicator repairs)."""
        ...

    def set_minor(self) -> None:
        """Set this rank as a minor rank."""
        ...

    def reset_minor(self) -> None:
        """Reset this rank to not be a minor rank."""
        ...

    def is_minor(self) -> bool:
        """Check if this rank is a minor rank."""
        ...

    def set_major_minor_split(self, boundary: int) -> None:
        """Set the major/minor split: ranks < boundary are major, ranks >= boundary are minor."""
        ...

    def set_major_minor_split_with_spares(self, num_majors: int, num_minors: int, num_major_spares: int, num_minor_spares: int) -> None:
        """Set rank type based on explicit counts.

        Layout: [major workers | major spares | minor workers | minor spares]
        - Ranks 0 to (num_majors - 1): Major workers
        - Ranks num_majors to (num_majors + num_major_spares - 1): Major spares
        - Ranks (num_majors + num_major_spares) to (num_majors + num_major_spares + num_minors - 1): Minor workers
        - Remaining ranks: Minor spares
        """
        ...

    def set_spare(self) -> None:
        """Set this rank as a spare rank."""
        ...

    def reset_spare(self) -> None:
        """Reset this rank to not be a spare rank."""
        ...

    def is_spare(self) -> bool:
        """Check if this rank is a spare rank."""
        ...

    def set_boundary_minor(self) -> None:
        """Set this rank as a boundary minor rank."""
        ...

    def reset_boundary_minor(self) -> None:
        """Reset this rank to not be a boundary minor rank."""
        ...

    def is_boundary_minor(self) -> bool:
        """Check if this rank is a boundary minor rank."""
        ...

    def set_boundary_minor_split(
        self,
        num_boundary_majors: int,
        boundary_major_workload: int,
        boundary_minor_workload: int,
    ) -> None:
        """Set the boundary minor split and directly populate the boundary-phase
        target contribution for this rank.

        Ranks < num_boundary_majors are not boundary minor; their
        boundary_target_contribution is set to boundary_major_workload.
        Ranks >= num_boundary_majors are boundary minor; their
        boundary_target_contribution is set to boundary_minor_workload.

        boundary_contributed_ is reset by reset_contributed() at iteration
        end, so this is safe to call multiple times within the same boundary
        (e.g. after cascading failures).

        Raises if either workload is negative.
        """
        ...

    def is_at_policy_boundary(self) -> bool:
        """Check if PG has reached policy boundary.

        This is a sticky flag: once true, stays true until explicitly reset.
        Returns True if a worker failed with no matching spares available.
        """
        ...

    def reset_policy_boundary(self) -> None:
        """Reset PG-level policy boundary flag.

        Call this after a policy change (e.g., advancing policy).
        """
        ...

    def update_rank_type_counts(
        self,
        majors: int,
        minors: int,
        major_spares: int,
        minor_spares: int,
        boundary_minors: int = 0,
    ) -> None:
        """Update rank type counts directly (without MPI communication)."""
        ...

    def get_num_major_procs(self) -> int:
        """Get count of major workers."""
        ...

    def get_num_minor_procs(self) -> int:
        """Get count of minor workers."""
        ...

    def get_num_major_spare_procs(self) -> int:
        """Get count of major spares."""
        ...

    def get_num_minor_spare_procs(self) -> int:
        """Get count of minor spares."""
        ...

    def get_num_boundary_minor_procs(self) -> int:
        """Get count of boundary minor ranks."""
        ...

    def get_contributed(self) -> int:
        """Get local count of gradient contributions made by this rank."""
        ...

    def increment_contributed(self) -> None:
        """Increment the local gradient contribution counter.

        Call from the Python control plane each time this rank's gradient
        was not zeroed before the allreduce (i.e. the rank is a real contributor).
        """
        ...

    def reset_contributed(self) -> None:
        """Reset the local gradient contribution counter to zero."""
        ...

    def get_target_contribution(self) -> int:
        """Get the target contribution value for this rank."""
        ...

    def set_target_contribution(self, major_value: int, minor_value: int = -1) -> None:
        """Set the target contribution based on rank type.

        major_value is applied to major ranks. minor_value is applied to minor
        ranks; if omitted (or <=0), minor ranks use major_value. Raises if
        either effective value is <= 0.
        """
        ...

    def increment_target_contribution(self, delta: int = 1) -> None:
        """Increment the target contribution by a positive delta (default 1).

        Raises if delta <= 0.
        """
        ...

    def get_boundary_contributed(self) -> int:
        """Get local count of gradient contributions made during the boundary phase."""
        ...

    def reset_boundary_contributed(self) -> None:
        """Reset the boundary-phase gradient contribution counter to zero."""
        ...

    def merge_boundary_contributed(self) -> None:
        """Fold boundary_contributed into contributed and zero both
        boundary_contributed and boundary_target_contribution.

        Call before issuing a new set_boundary_minor_split at a nested
        boundary so the contributions already made during the previous
        extended pass are accounted for by the regular counter and the
        next extension stacks on top cleanly.
        """
        ...

    def get_boundary_target_contribution(self) -> int:
        """Get the boundary-phase target contribution value for this rank."""
        ...

    def set_boundary_target_contribution(self, value: int) -> None:
        """Set the boundary-phase target contribution (non-negative).

        Raises if value < 0.
        """
        ...

    def should_contribute(self) -> bool:
        """Return True if this rank still owes a contribution.

        During the extended pass at a policy boundary (is_at_policy_boundary()
        is True) compares boundary_contributed < boundary_target_contribution;
        otherwise compares contributed < target_contribution.
        """
        ...

    def elect_promotion(self, failed_majors: int, failed_minors: int) -> bool:
        """Elect spare promotion via collective.

        Automatically promotes spares to replace failed workers:
        - Major spares (non-minor spares) replace failed majors
        - Minor spares replace failed minors

        Args:
            failed_majors: Number of failed major workers to replace
            failed_minors: Number of failed minor workers to replace

        Returns:
            True if THIS rank was promoted from spare to worker
        """
        ...

    def record_and_handling_failure(
        self,
        failed_ranks: list[int],
        ulfm_opts: ULFMOptions,
        ulfm_work: WorkULFM,
    ) -> None:
        """Combined helper: track rank types, compute failures, auto-elect, and record.

        Encapsulates the entire failure handling workflow for reuse:
        1. If track_rank_types is enabled, snapshot old counts and count survivors
        2. Compute failed counts (old - new)
        3. Check if at policy boundary (worker failed with no matching spares)
        4. If auto_elect is enabled and NOT at boundary, elect spare promotion
        5. Update stored rank type counts
        6. Record failure info in the WorkULFM object

        Args:
            failed_ranks: List of ranks that failed
            ulfm_opts: Options controlling failure handling behavior
            ulfm_work: WorkULFM object to record failure info into
        """
        ...


# ULFM logging control
def set_ulfm_verbose_logging(verbose: bool) -> None:
    """Enable or disable verbose ULFM logging."""
    ...

def is_ulfm_verbose_logging() -> bool:
    """Check if verbose ULFM logging is enabled."""
    ...
