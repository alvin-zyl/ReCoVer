"""Replica topology functions.

Pure Python + numpy. No torch, no MPI, no distributed imports. These are the
same layouts used by ``nanotron/src/nanotron/parallel/context_ulfm.py`` (3D)
and ``main_hsdp.py`` (HSDP), reproduced here so the standalone generator can
compute them without launching training.
"""

from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from .schedule import ParallelismSpec


def replicas_3d(tp: int, pp: int, dp: int, ep: int) -> List[List[int]]:
    """Replica layout for 3D (+expert) parallelism.

    Mirrors ``context_ulfm.py``::

        ranks = arange(world).reshape((ep, pp, dp, tp))

    A replica is one DP slice: all ranks sharing the same ``dp`` index. Inner
    list is ordered by ``(ep, pp, tp)`` so ``local_rank`` is stable.
    """
    if min(tp, pp, dp, ep) <= 0:
        raise ValueError(
            f"all parallelism dims must be positive (got tp={tp}, pp={pp}, dp={dp}, ep={ep})"
        )
    world = tp * pp * dp * ep
    ranks = np.arange(world).reshape((ep, pp, dp, tp))
    return [ranks[:, :, dp_idx, :].reshape(-1).tolist() for dp_idx in range(dp)]


def replicas_hsdp(world_size: int, shard_size: int) -> List[List[int]]:
    """Replica layout for HSDP.

    Mirrors ``main_hsdp.py``: contiguous blocks of ``shard_size`` ranks form
    one shard group, which is the fault-unit (a failure anywhere in a shard
    brings down the whole replica).
    """
    if world_size <= 0 or shard_size <= 0:
        raise ValueError(
            f"world_size and shard_size must be positive (got {world_size}, {shard_size})"
        )
    if world_size % shard_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by shard_size ({shard_size})"
        )
    num_replicas = world_size // shard_size
    return [list(range(r * shard_size, (r + 1) * shard_size)) for r in range(num_replicas)]


def replicas_for(spec: "ParallelismSpec") -> List[List[int]]:
    """Dispatch ``replicas_3d`` / ``replicas_hsdp`` based on ``spec.kind``."""
    # Local import to avoid a circular dependency at module import time.
    from .schedule import ParallelismSpec  # noqa: F401

    if spec.kind == "3d":
        return replicas_3d(tp=spec.tp, pp=spec.pp, dp=spec.dp, ep=spec.ep)
    if spec.kind == "hsdp":
        return replicas_hsdp(world_size=spec.world_size, shard_size=spec.shard_size)
    raise ValueError(f"unknown parallelism kind: {spec.kind!r}")


def expected_world_size(spec: "ParallelismSpec") -> int:
    """Total ranks implied by ``spec``."""
    if spec.kind == "3d":
        return spec.tp * spec.pp * spec.dp * spec.ep
    if spec.kind == "hsdp":
        return spec.world_size
    raise ValueError(f"unknown parallelism kind: {spec.kind!r}")
