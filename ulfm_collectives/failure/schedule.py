"""Failure schedule data model + YAML (de)serialization.

Classes are frozen dataclasses; the schedule and its inputs are immutable
after construction. YAML is the only on-disk format.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Literal, Optional, Tuple

import yaml

from .locations import validate_location


_SamplingMode = Literal["iid", "stratified"]
_Kind = Literal["3d", "hsdp"]


@dataclass(frozen=True)
class FailureEntry:
    step: int
    replica_id: int
    local_rank: int
    location: str


@dataclass(frozen=True)
class ParallelismSpec:
    kind: _Kind
    # 3d fields (None for hsdp)
    tp: Optional[int] = None
    pp: Optional[int] = None
    dp: Optional[int] = None
    ep: Optional[int] = None
    # hsdp fields (None for 3d)
    world_size: Optional[int] = None
    shard_size: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind == "3d":
            missing = [
                name
                for name, val in (
                    ("tp", self.tp),
                    ("pp", self.pp),
                    ("dp", self.dp),
                    ("ep", self.ep),
                )
                if val is None
            ]
            if missing:
                raise ValueError(
                    f"ParallelismSpec(kind='3d') missing required dims: {missing}"
                )
            if any(v <= 0 for v in (self.tp, self.pp, self.dp, self.ep)):
                raise ValueError(
                    f"all 3d dims must be positive (got tp={self.tp}, pp={self.pp}, "
                    f"dp={self.dp}, ep={self.ep})"
                )
            if self.world_size is not None or self.shard_size is not None:
                raise ValueError(
                    "ParallelismSpec(kind='3d') must not set world_size or shard_size"
                )
        elif self.kind == "hsdp":
            if self.world_size is None or self.shard_size is None:
                raise ValueError(
                    "ParallelismSpec(kind='hsdp') requires world_size and shard_size"
                )
            if self.world_size <= 0 or self.shard_size <= 0:
                raise ValueError("hsdp world_size and shard_size must be positive")
            if self.world_size % self.shard_size != 0:
                raise ValueError(
                    f"world_size ({self.world_size}) must be divisible by "
                    f"shard_size ({self.shard_size})"
                )
            if any(
                v is not None for v in (self.tp, self.pp, self.dp, self.ep)
            ):
                raise ValueError(
                    "ParallelismSpec(kind='hsdp') must not set tp/pp/dp/ep"
                )
        else:
            raise ValueError(f"unknown parallelism kind: {self.kind!r}")

    def to_dict(self) -> Dict:
        if self.kind == "3d":
            return {"kind": "3d", "tp": self.tp, "pp": self.pp, "dp": self.dp, "ep": self.ep}
        return {"kind": "hsdp", "world_size": self.world_size, "shard_size": self.shard_size}


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int
    num_failures: int
    step_range: Tuple[int, int]
    sampling: _SamplingMode = "stratified"
    location_weights: Dict[str, float] = field(default_factory=dict)
    exclude_replica_ids: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.num_failures < 0:
            raise ValueError(f"num_failures must be >= 0 (got {self.num_failures})")
        if len(self.step_range) != 2 or self.step_range[1] <= self.step_range[0]:
            raise ValueError(
                f"step_range must be (start, end) with end > start (got {self.step_range})"
            )
        if self.sampling not in ("iid", "stratified"):
            raise ValueError(f"sampling must be 'iid' or 'stratified' (got {self.sampling!r})")
        if self.num_failures > 0:
            if not self.location_weights:
                raise ValueError("location_weights must be non-empty when num_failures > 0")
            for name, w in self.location_weights.items():
                if not isinstance(name, str) or not name:
                    raise ValueError(f"location name must be a non-empty string (got {name!r})")
                validate_location(name)
                if w <= 0:
                    raise ValueError(
                        f"location weight for {name!r} must be > 0 (got {w})"
                    )
        # Canonicalize exclude_replica_ids to sorted, unique tuple of ints.
        canon = tuple(sorted({int(r) for r in self.exclude_replica_ids}))
        for rid in canon:
            if rid < 0:
                raise ValueError(
                    f"exclude_replica_ids entries must be >= 0 (got {rid})"
                )
        object.__setattr__(self, "exclude_replica_ids", canon)

    def to_dict(self) -> Dict:
        out: Dict = {
            "seed": self.seed,
            "num_failures": self.num_failures,
            "step_range": list(self.step_range),
            "sampling": self.sampling,
            # Preserve user-specified key order when round-tripping through YAML:
            # sort for a canonical on-disk form.
            "location_weights": {k: self.location_weights[k] for k in sorted(self.location_weights)},
        }
        if self.exclude_replica_ids:
            out["exclude_replica_ids"] = list(self.exclude_replica_ids)
        return out


@dataclass(frozen=True)
class FailureSchedule:
    parallelism: ParallelismSpec
    generator_config: Optional[GeneratorConfig]
    entries: Tuple[FailureEntry, ...]

    def __post_init__(self) -> None:
        # Enforce sort order (step, replica_id).
        sorted_entries = tuple(
            sorted(self.entries, key=lambda e: (e.step, e.replica_id))
        )
        object.__setattr__(self, "entries", sorted_entries)

        # One-kill-per-replica invariant.
        seen_replicas = set()
        for e in sorted_entries:
            if e.replica_id in seen_replicas:
                raise ValueError(
                    f"duplicate replica_id {e.replica_id} in schedule "
                    "(at most one kill per replica)"
                )
            seen_replicas.add(e.replica_id)

        # Bound-check each entry against parallelism.
        # Local import to avoid cycle at module import.
        from .topology import replicas_for

        replicas = replicas_for(self.parallelism)
        num_replicas = len(replicas)
        for e in sorted_entries:
            if not (0 <= e.replica_id < num_replicas):
                raise ValueError(
                    f"replica_id {e.replica_id} out of bounds [0, {num_replicas})"
                )
            replica_size = len(replicas[e.replica_id])
            if not (0 <= e.local_rank < replica_size):
                raise ValueError(
                    f"local_rank {e.local_rank} out of bounds [0, {replica_size}) "
                    f"for replica {e.replica_id}"
                )
            if not isinstance(e.location, str) or not e.location:
                raise ValueError(f"location must be a non-empty string (got {e.location!r})")
            validate_location(e.location)
            if e.step < 0:
                raise ValueError(f"step must be >= 0 (got {e.step})")

    def global_rank_of(self, entry: FailureEntry) -> int:
        """Resolve ``(replica_id, local_rank)`` → global rank via the topology."""
        from .topology import replicas_for

        return replicas_for(self.parallelism)[entry.replica_id][entry.local_rank]

    def entries_for_rank(self, global_rank: int) -> List[FailureEntry]:
        from .topology import replicas_for

        replicas = replicas_for(self.parallelism)
        out: List[FailureEntry] = []
        for e in self.entries:
            if replicas[e.replica_id][e.local_rank] == global_rank:
                out.append(e)
        return out

    def assert_matches_topology(self, spec: ParallelismSpec) -> None:
        if self.parallelism != spec:
            raise RuntimeError(
                f"schedule topology {self.parallelism.to_dict()} does not match "
                f"runtime topology {spec.to_dict()}"
            )

    # ------------------------------------------------------------------
    # YAML I/O
    # ------------------------------------------------------------------

    def to_yaml_dict(self) -> Dict:
        out: Dict = {"parallelism": self.parallelism.to_dict()}
        if self.generator_config is not None:
            out["generator_config"] = self.generator_config.to_dict()
        out["entries"] = [
            {
                "step": e.step,
                "replica_id": e.replica_id,
                "local_rank": e.local_rank,
                "location": e.location,
            }
            for e in self.entries
        ]
        return out

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(
                self.to_yaml_dict(),
                f,
                sort_keys=False,
                default_flow_style=None,
            )

    def to_yaml_str(self) -> str:
        return yaml.safe_dump(
            self.to_yaml_dict(),
            sort_keys=False,
            default_flow_style=None,
        )

    @classmethod
    def load(cls, path: str) -> "FailureSchedule":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls.from_yaml_dict(raw)

    @classmethod
    def from_yaml_dict(cls, raw: Dict) -> "FailureSchedule":
        if not isinstance(raw, dict):
            raise ValueError("schedule YAML must be a mapping at the top level")
        for key in ("parallelism", "entries"):
            if key not in raw:
                raise ValueError(f"schedule YAML missing required key: {key!r}")

        p = raw["parallelism"]
        if not isinstance(p, dict) or "kind" not in p:
            raise ValueError("parallelism block must be a mapping with a 'kind' field")
        spec = ParallelismSpec(**p)

        gc_raw = raw.get("generator_config")
        gc: Optional[GeneratorConfig] = None
        if gc_raw is not None:
            if not isinstance(gc_raw, dict):
                raise ValueError("generator_config must be a mapping")
            step_range = gc_raw.get("step_range")
            if not (isinstance(step_range, (list, tuple)) and len(step_range) == 2):
                raise ValueError("generator_config.step_range must be a 2-element list")
            gc = GeneratorConfig(
                seed=int(gc_raw["seed"]),
                num_failures=int(gc_raw["num_failures"]),
                step_range=(int(step_range[0]), int(step_range[1])),
                sampling=gc_raw.get("sampling", "stratified"),
                location_weights=dict(gc_raw.get("location_weights") or {}),
                exclude_replica_ids=tuple(
                    int(r) for r in (gc_raw.get("exclude_replica_ids") or ())
                ),
            )

        entries_raw = raw.get("entries") or []
        if not isinstance(entries_raw, list):
            raise ValueError("entries must be a list")
        entries: List[FailureEntry] = []
        for i, e in enumerate(entries_raw):
            if not isinstance(e, dict):
                raise ValueError(f"entry {i} must be a mapping")
            for key in ("step", "replica_id", "local_rank", "location"):
                if key not in e:
                    raise ValueError(f"entry {i} missing required key: {key!r}")
            entries.append(
                FailureEntry(
                    step=int(e["step"]),
                    replica_id=int(e["replica_id"]),
                    local_rank=int(e["local_rank"]),
                    location=str(e["location"]),
                )
            )

        return cls(parallelism=spec, generator_config=gc, entries=tuple(entries))
