"""Pure helpers for the HSDP 2D rank layout. No distributed imports."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class HSDPLayout:
    world_size: int
    shard_size: int

    @property
    def num_replicas(self) -> int:
        return self.world_size // self.shard_size

    def replica_id_of(self, rank: int) -> int:
        return rank // self.shard_size

    def shard_rank_of(self, rank: int) -> int:
        return rank % self.shard_size


def compute_hsdp_layout(world_size: int, shard_size: int) -> HSDPLayout:
    if shard_size <= 0 or world_size <= 0:
        raise ValueError("world_size and shard_size must be positive")
    if world_size % shard_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by shard_size ({shard_size})"
        )
    return HSDPLayout(world_size=world_size, shard_size=shard_size)


def replica_ranks(replica_id: int, shard_size: int) -> List[int]:
    """Ranks belonging to `replica_id` (contiguous block)."""
    start = replica_id * shard_size
    return list(range(start, start + shard_size))


def replicate_peer_ranks(offset: int, shard_size: int, num_replicas: int) -> List[int]:
    """Ranks that share the same intra-replica offset across all replicas."""
    return [offset + r * shard_size for r in range(num_replicas)]


def replica0_ranks(shard_size: int) -> List[int]:
    return list(range(shard_size))
