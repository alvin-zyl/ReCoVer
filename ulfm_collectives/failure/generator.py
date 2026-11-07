"""Deterministic failure-schedule generator.

Pure function of its inputs — no distributed state, no env vars, no wall
clock. Running with ``-np 1`` as a standalone tool produces byte-identical
output to calling from any rank during training init.
"""

import random
from typing import Dict, Iterable, Literal, Optional, Set, Tuple

from .schedule import FailureSchedule, GeneratorConfig, ParallelismSpec, FailureEntry
from .topology import replicas_for


_SamplingMode = Literal["iid", "stratified"]


def _validate_inputs(
    replicas_count: int,
    num_failures: int,
    step_range: Tuple[int, int],
    location_weights: Dict[str, float],
    sampling: str,
    eligible_replicas: int,
) -> None:
    if num_failures < 0:
        raise ValueError(f"num_failures must be >= 0 (got {num_failures})")
    if num_failures > replicas_count:
        raise ValueError(
            f"num_failures ({num_failures}) exceeds number of replicas ({replicas_count}); "
            "one kill per replica is the max"
        )
    if num_failures > eligible_replicas:
        raise ValueError(
            f"num_failures ({num_failures}) exceeds eligible replicas "
            f"({eligible_replicas}) after exclusion"
        )
    if len(step_range) != 2 or step_range[1] <= step_range[0]:
        raise ValueError(f"step_range must be (start, end) with end > start (got {step_range})")
    if num_failures == 0:
        return
    if sampling not in ("iid", "stratified"):
        raise ValueError(f"sampling must be 'iid' or 'stratified' (got {sampling!r})")
    if sampling == "stratified":
        width = step_range[1] - step_range[0]
        if width < num_failures:
            raise ValueError(
                f"stratified sampling requires step_range width ({width}) >= num_failures "
                f"({num_failures}); increase step_range or drop to iid"
            )
    if not location_weights:
        raise ValueError("location_weights must be non-empty when num_failures > 0")
    for name, w in location_weights.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"location name must be a non-empty string (got {name!r})")
        if w <= 0:
            raise ValueError(f"location weight for {name!r} must be > 0 (got {w})")


def _stratified_buckets(
    start: int, end: int, num: int
) -> Tuple[Tuple[int, int], ...]:
    """Partition ``[start, end)`` into ``num`` contiguous buckets.

    Buckets may differ in width by at most one when ``(end-start) % num != 0``;
    the extra steps go to the earliest buckets.
    """
    width = end - start
    base = width // num
    remainder = width % num
    buckets = []
    cursor = start
    for i in range(num):
        extra = 1 if i < remainder else 0
        nxt = cursor + base + extra
        buckets.append((cursor, nxt))
        cursor = nxt
    return tuple(buckets)


def generate(
    parallelism: ParallelismSpec,
    seed: int,
    num_failures: int,
    step_range: Tuple[int, int],
    location_weights: Dict[str, float],
    sampling: _SamplingMode = "stratified",
    exclude_replica_ids: Optional[Iterable[int]] = None,
) -> FailureSchedule:
    """Produce a deterministic ``FailureSchedule`` from the given inputs.

    Args:
        parallelism: topology descriptor. Determines replica count and sizes.
        seed: single integer consumed by a single ``random.Random`` instance.
        num_failures: exact number of replicas to schedule a kill on. Must be
            ``<= num_replicas - len(exclude_replica_ids)``.
        step_range: ``(start, end)``, half-open. Kill steps are drawn from
            ``[start, end)``.
        location_weights: mapping of location name to positive weight; weights
            are normalized internally. Iteration uses sorted keys so dict
            insertion order does not affect the output.
        sampling: ``"stratified"`` (default) partitions ``step_range`` into
            ``num_failures`` buckets and draws one step per bucket;
            ``"iid"`` draws each step independently from the full range
            (collisions across different replicas are allowed).
        exclude_replica_ids: replica ids that must not be selected (e.g. the
            replica hosting wandb). Order-independent — stored and iterated
            as a sorted set for deterministic output.
    """
    replicas = replicas_for(parallelism)
    excluded: Set[int] = set()
    if exclude_replica_ids is not None:
        for rid in exclude_replica_ids:
            rid = int(rid)
            if rid < 0 or rid >= len(replicas):
                raise ValueError(
                    f"exclude_replica_ids contains {rid}, out of bounds "
                    f"[0, {len(replicas)})"
                )
            excluded.add(rid)
    eligible_ids = [r for r in range(len(replicas)) if r not in excluded]
    _validate_inputs(
        len(replicas),
        num_failures,
        step_range,
        location_weights,
        sampling,
        len(eligible_ids),
    )

    if num_failures == 0:
        cfg = GeneratorConfig(
            seed=seed,
            num_failures=0,
            step_range=tuple(step_range),
            sampling=sampling,
            # Preserve whatever the caller supplied; may be empty.
            location_weights=dict(location_weights),
            exclude_replica_ids=tuple(sorted(excluded)),
        )
        return FailureSchedule(parallelism=parallelism, generator_config=cfg, entries=())

    rng = random.Random(seed)

    # Step 4: pick replicas from the eligible subset. Preserve sample order
    # (do not sort) so the bucket-to-replica mapping in stratified mode is
    # shuffled rather than monotonic in replica_id.
    chosen_replicas = rng.sample(eligible_ids, num_failures)

    # Step 5: draw steps, local_rank, location for each chosen replica in sample order.
    sorted_locs = sorted(location_weights.keys())
    sorted_weights = [location_weights[k] for k in sorted_locs]

    start, end = int(step_range[0]), int(step_range[1])
    if sampling == "stratified":
        buckets = _stratified_buckets(start, end, num_failures)
    else:
        buckets = tuple((start, end) for _ in range(num_failures))

    entries = []
    for idx, replica_id in enumerate(chosen_replicas):
        bucket_start, bucket_end = buckets[idx]
        step = rng.randrange(bucket_start, bucket_end)
        local_rank = rng.randrange(0, len(replicas[replica_id]))
        location = rng.choices(sorted_locs, weights=sorted_weights, k=1)[0]
        entries.append(
            FailureEntry(
                step=step,
                replica_id=replica_id,
                local_rank=local_rank,
                location=location,
            )
        )

    cfg = GeneratorConfig(
        seed=seed,
        num_failures=num_failures,
        step_range=(start, end),
        sampling=sampling,
        location_weights=dict(location_weights),
        exclude_replica_ids=tuple(sorted(excluded)),
    )

    return FailureSchedule(
        parallelism=parallelism,
        generator_config=cfg,
        entries=tuple(entries),
    )
