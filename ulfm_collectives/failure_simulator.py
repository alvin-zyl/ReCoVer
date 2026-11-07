#!/usr/bin/env python3
"""Deterministic replica-aware failure simulator.

The simulator consumes a pre-computed :class:`FailureSchedule` (loaded from
YAML or produced at init time by :mod:`ulfm_collectives.failure.generator`)
and SIGKILLs the targeted rank at the scheduled step/location.

Public API (unchanged from prior revisions; internals rewritten):

    sim = FailureSimulator(schedule)
    set_failure_simulator(sim)
    sim.initialize(rank=dist.get_rank(), world_size=dist.get_world_size())

    @sim.may_fail("forward_pass")
    def forward(x): ...

    for step in range(num_steps):
        sim.begin_minibatch(step)
        output = forward(x)
        with sim.may_fail_here("backward"):
            loss.backward()
"""

import logging
import os
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Dict, List, Optional, Set

from .failure.locations import GRAD_SYNC_POSITIONS, canonical_location
from .failure.schedule import FailureEntry, FailureSchedule
from .failure.topology import expected_world_size, replicas_for

logger = logging.getLogger(__name__)


@dataclass
class _State:
    rank: int = -1
    world_size: int = -1
    step: int = -1
    armed_entry: Optional[FailureEntry] = None


@dataclass
class FailureRecord:
    location: str
    rank: int
    step: int
    timestamp: float = field(default_factory=time.time)


class FailureSimulator:
    """Schedule-driven failure injector.

    Parameters
    ----------
    schedule:
        The schedule to execute. Every rank must construct a simulator with
        the *same* schedule (either loaded from the same file or produced by
        the generator with identical inputs).
    enabled:
        If False, the simulator registers locations and logs but never kills.
    """

    def __init__(self, schedule: FailureSchedule, enabled: bool = True) -> None:
        self.schedule = schedule
        self.enabled = enabled

        self._lock = threading.RLock()
        self._state = _State()
        self._initialized = False
        self._has_failed = False

        # Entries assigned to *this* rank (resolved in initialize()).
        self._my_entries: Dict[int, FailureEntry] = {}

        # Locations known to the simulator (decorator / context manager).
        self._registered_locations: Set[str] = set()
        self._location_scan_done = False

        self._history: List[FailureRecord] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, rank: int, world_size: int) -> None:
        with self._lock:
            expected = expected_world_size(self.schedule.parallelism)
            if world_size != expected:
                raise RuntimeError(
                    f"schedule expects world_size={expected} "
                    f"(parallelism={self.schedule.parallelism.to_dict()}), "
                    f"but runtime world_size={world_size}"
                )

            self._state.rank = rank
            self._state.world_size = world_size

            replicas = replicas_for(self.schedule.parallelism)
            my_entries: Dict[int, FailureEntry] = {}
            for entry in self.schedule.entries:
                global_rank = replicas[entry.replica_id][entry.local_rank]
                if global_rank == rank:
                    # By construction each replica appears at most once in the
                    # schedule, so at most one entry can land on this rank.
                    # Assert to catch construction-time regressions.
                    assert entry.step not in my_entries, (
                        f"two scheduled entries target the same step on rank {rank}"
                    )
                    my_entries[entry.step] = entry
            self._my_entries = my_entries
            self._initialized = True

            if rank == 0:
                self._log_schedule_banner(replicas)

            if my_entries:
                for step, e in sorted(my_entries.items()):
                    logger.info(
                        "[Rank %d] assigned failure: step=%d location=%s "
                        "(replica=%d, local_rank=%d)",
                        rank,
                        step,
                        e.location,
                        e.replica_id,
                        e.local_rank,
                    )
            else:
                logger.info("[Rank %d] no failures assigned", rank)

    # ------------------------------------------------------------------
    # Location registration (decorator / context manager)
    # ------------------------------------------------------------------

    def register_location(self, location: str) -> None:
        with self._lock:
            self._registered_locations.add(location)

    def get_registered_locations(self) -> Set[str]:
        with self._lock:
            return set(self._registered_locations)

    def may_fail(self, location: str):
        """Decorator. Registers ``location`` at decoration time; checks on call."""
        self.register_location(location)

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                self.check(location)
                return func(*args, **kwargs)

            return wrapper

        return decorator

    @contextmanager
    def may_fail_here(self, location: str):
        """Context manager. Registers ``location`` on first ``__enter__``; checks on entry."""
        self.register_location(location)
        self.check(location)
        yield

    # ------------------------------------------------------------------
    # Per-step arming / checking
    # ------------------------------------------------------------------

    def begin_minibatch(self, step: int) -> None:
        """Arm this rank for ``step`` if an entry is scheduled at that step.

        Also performs a one-shot best-effort scan for scheduled locations
        that are not in the registry. Warnings may over-warn for
        ``may_fail_here`` locations that register only on first loop entry.
        """
        with self._lock:
            self._state.step = step
            # Don't unconditionally clear armed_entry — an arm whose scheduled
            # step lands on a no_sync microstep (where the matching location
            # never gets reached, e.g., "grad-sync:*" with the fp32 fold
            # that fires the hook only on sync microsteps) must persist into
            # the next microbatch so the next sync-step hook can consume it.
            # check(location) gates only on location, not step, so persisting
            # is safe; new entries below still overwrite if scheduled.

            if not self.enabled or not self._initialized:
                return
            if self._has_failed:
                return

            if not self._location_scan_done:
                self._location_scan_done = True
                self._warn_unregistered_locations()

            entry = self._my_entries.get(step)
            if entry is not None:
                self._state.armed_entry = entry
                logger.debug(
                    "[Rank %d] step=%d armed for location=%s",
                    self._state.rank,
                    step,
                    entry.location,
                )

    def describe(self) -> str:
        """Return a human-readable, multi-line dump of the full schedule.

        Includes parallelism, generator config (if present), a per-entry
        table with global-rank resolution, and the full YAML form for
        copy/paste reproducibility. Safe to call any time (does not require
        :meth:`initialize`).
        """
        replicas = replicas_for(self.schedule.parallelism)
        lines: List[str] = []
        sep = "=" * 72
        lines.append(sep)
        lines.append(
            f"FailureSimulator schedule ({len(self.schedule.entries)} entries)"
        )
        lines.append(f"  parallelism: {self.schedule.parallelism.to_dict()}")
        gc = self.schedule.generator_config
        if gc is not None:
            lines.append(
                f"  generator: seed={gc.seed} num_failures={gc.num_failures} "
                f"step_range={tuple(gc.step_range)} sampling={gc.sampling} "
                f"locations={dict(sorted(gc.location_weights.items()))} "
                f"exclude_replica_ids="
                f"{list(gc.exclude_replica_ids) if gc.exclude_replica_ids else []}"
            )
        else:
            lines.append("  generator: <loaded from file, no generator_config>")

        if self.schedule.entries:
            lines.append(
                f"  {'step':>6}  {'replica':>7}  {'local':>5}  "
                f"{'global':>6}  location"
            )
            for e in self.schedule.entries:
                gr = replicas[e.replica_id][e.local_rank]
                lines.append(
                    f"  {e.step:>6}  {e.replica_id:>7}  {e.local_rank:>5}  "
                    f"{gr:>6}  {e.location}"
                )
        else:
            lines.append("  (no failures scheduled)")

        lines.append("--- schedule YAML (begin) ---")
        for yl in self.schedule.to_yaml_str().rstrip("\n").splitlines():
            lines.append(f"  {yl}")
        lines.append("--- schedule YAML (end) ---")
        lines.append(sep)
        return "\n".join(lines)

    def _log_schedule_banner(self, replicas) -> None:
        """Emit the full schedule dump to the stdlib logger at INFO.

        Call sites that use a different logger (loguru, nanotron log_rank)
        should additionally call ``describe()`` and log the return value via
        their own logger so the schedule appears in the expected output stream.
        """
        for line in self.describe().splitlines():
            logger.info("%s", line)

    def _warn_unregistered_locations(self) -> None:
        scheduled = {canonical_location(e.location) for e in self.schedule.entries}
        missing = scheduled - {
            canonical_location(loc) for loc in self._registered_locations
        }
        if missing:
            logger.warning(
                "FailureSimulator: scheduled locations not yet registered: %s "
                "(decorators register at import; may_fail_here registers on first "
                "__enter__ — this warning may be premature for the latter)",
                sorted(missing),
            )

    def check(self, location: str) -> bool:
        """If this rank is armed for ``location`` at the current step, SIGKILL.

        Returns False when no failure is injected. When a failure is injected,
        the process is killed and control never returns.
        """
        with self._lock:
            armed = self._state.armed_entry
            if armed is None or canonical_location(armed.location) != canonical_location(
                location
            ):
                return False
            self._inject(armed)
            return True  # unreachable

    def check_grad_sync(self, bucket_pos: int, num_buckets: int) -> bool:
        """``grad-sync:<first|mid|last>``: call right after bucket ``bucket_pos``
        (0-based, in firing order) of ``num_buckets`` finished its cross-replica
        all-reduce. ``mid`` is bucket ``num_buckets // 2``; with one bucket all
        three positions are the same point.
        """
        for pos in GRAD_SYNC_POSITIONS:
            self.register_location(f"grad-sync:{pos}")
        with self._lock:
            armed = self._state.armed_entry
            if armed is None:
                return False
            location = canonical_location(armed.location)
            if not location.startswith("grad-sync:"):
                return False
            target = {
                "first": 0,
                "mid": num_buckets // 2,
                "last": num_buckets - 1,
            }.get(location.split(":", 1)[1])
            if target is None or bucket_pos != target:
                return False
            self._inject(armed)
            return True  # unreachable

    def _inject(self, entry: FailureEntry) -> None:
        """Record and SIGKILL. Caller must hold ``self._lock``."""
        self._history.append(
            FailureRecord(
                location=entry.location,
                rank=self._state.rank,
                step=entry.step,
            )
        )
        self._has_failed = True
        logger.warning(
            "[Rank %d] INJECTING FAILURE at location=%s step=%d (replica=%d, local_rank=%d)",
            self._state.rank,
            canonical_location(entry.location)
            + (
                f" (scheduled as {entry.location})"
                if canonical_location(entry.location) != entry.location
                else ""
            ),
            entry.step,
            entry.replica_id,
            entry.local_rank,
        )
        os.kill(os.getpid(), signal.SIGKILL)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def has_failed(self) -> bool:
        with self._lock:
            return self._has_failed

    @property
    def my_entries(self) -> Dict[int, FailureEntry]:
        with self._lock:
            return dict(self._my_entries)

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "parallelism": self.schedule.parallelism.to_dict(),
                "generator_config": (
                    self.schedule.generator_config.to_dict()
                    if self.schedule.generator_config is not None
                    else None
                ),
                "num_scheduled_entries": len(self.schedule.entries),
                "my_entries": [
                    {
                        "step": e.step,
                        "replica_id": e.replica_id,
                        "local_rank": e.local_rank,
                        "location": e.location,
                    }
                    for e in sorted(self._my_entries.values(), key=lambda e: e.step)
                ],
                "registered_locations": sorted(self._registered_locations),
                "has_failed": self._has_failed,
                "history": [
                    {
                        "location": r.location,
                        "rank": r.rank,
                        "step": r.step,
                        "timestamp": r.timestamp,
                    }
                    for r in self._history
                ],
            }

    def reset(self) -> None:
        """Re-resolve per-rank entries and clear history. Schedule is unchanged."""
        with self._lock:
            self._has_failed = False
            self._history.clear()
            self._state.step = -1
            self._state.armed_entry = None
            if self._initialized and self._state.rank >= 0:
                rank = self._state.rank
                world_size = self._state.world_size
                self._initialized = False
                self._my_entries = {}
                self._location_scan_done = False
                # Release the lock to call initialize (acquires it again).
        if self._state.rank >= 0 and self._state.world_size >= 0 and not self._initialized:
            self.initialize(rank=self._state.rank, world_size=self._state.world_size)

    def __repr__(self) -> str:
        return (
            f"FailureSimulator(entries={len(self.schedule.entries)}, "
            f"rank={self._state.rank}, enabled={self.enabled})"
        )


# ----------------------------------------------------------------------
# Global singleton (kept for backwards compatibility with call sites)
# ----------------------------------------------------------------------

_simulator: Optional[FailureSimulator] = None


def get_failure_simulator() -> Optional[FailureSimulator]:
    return _simulator


def set_failure_simulator(sim: Optional[FailureSimulator]) -> None:
    global _simulator
    _simulator = sim


def clear_failure_simulator() -> None:
    global _simulator
    _simulator = None
