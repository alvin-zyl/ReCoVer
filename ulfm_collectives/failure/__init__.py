"""Deterministic replica-aware failure schedule.

This subpackage contains the data model, YAML (de)serialization, rank/replica
topology, and the pure-function schedule generator used by both the standalone
CLI (``scripts/generate_failure_schedule.py``) and the runtime
``FailureSimulator``.
"""

from .schedule import (
    FailureEntry,
    FailureSchedule,
    GeneratorConfig,
    ParallelismSpec,
)
from .topology import replicas_3d, replicas_hsdp, replicas_for
from .generator import generate

__all__ = [
    "FailureEntry",
    "FailureSchedule",
    "GeneratorConfig",
    "ParallelismSpec",
    "replicas_3d",
    "replicas_hsdp",
    "replicas_for",
    "generate",
]
