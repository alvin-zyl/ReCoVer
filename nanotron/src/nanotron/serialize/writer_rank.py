"""Which rank writes what in a checkpoint after a ULFM group lost ranks.

Nanotron lets rank 0 of dp_pg (or of world_pg) write what is replicated
across that group. With ULFM, dp_pg and world_pg shrink when ranks fail, but
torch.distributed keeps the ranks they had when created: once rank 0 has
failed, no rank is 0 and nothing would be written. ``writer_rank`` is the
rank in the repaired communicator instead, so the lowest surviving rank
writes; it equals ``dist.get_rank`` for every other process group, and for a
ULFM group as long as no rank below this one has failed.
``root_rank`` is the creation-time rank of that lowest surviving rank,
for collectives rooted at it (0 for every other group, and while rank 0 is
alive).
"""

from nanotron import distributed as dist


def writer_rank(pg) -> int:
    current_rank = getattr(pg, "current_rank", None)
    return current_rank() if current_rank is not None else dist.get_rank(pg)


def root_rank(pg) -> int:
    original_rank_of = getattr(pg, "original_rank_of", None)
    return original_rank_of(0) if original_rank_of is not None else 0
