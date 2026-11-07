import pytest
from ulfm_collectives.hsdp_groups import (
    compute_hsdp_layout,
    replica_ranks,
    replicate_peer_ranks,
    replica0_ranks,
)


def test_layout_divisibility():
    with pytest.raises(ValueError, match="divisible"):
        compute_hsdp_layout(world_size=5, shard_size=2)


def test_layout_values():
    layout = compute_hsdp_layout(world_size=8, shard_size=2)
    assert layout.num_replicas == 4
    assert layout.shard_size == 2


def test_replica_ranks_contiguous():
    # shard_size=2, world=8: replica 0=[0,1], 1=[2,3], 2=[4,5], 3=[6,7]
    assert replica_ranks(replica_id=0, shard_size=2) == [0, 1]
    assert replica_ranks(replica_id=2, shard_size=2) == [4, 5]


def test_replicate_peer_ranks():
    # shard_size=2, num_replicas=4: offset 0 -> [0,2,4,6], offset 1 -> [1,3,5,7]
    assert replicate_peer_ranks(offset=0, shard_size=2, num_replicas=4) == [0, 2, 4, 6]
    assert replicate_peer_ranks(offset=1, shard_size=2, num_replicas=4) == [1, 3, 5, 7]


def test_replica0_ranks():
    assert replica0_ranks(shard_size=2) == [0, 1]
    assert replica0_ranks(shard_size=4) == [0, 1, 2, 3]


def test_rank_decomposition_roundtrip():
    layout = compute_hsdp_layout(world_size=8, shard_size=2)
    for rank in range(8):
        rid = layout.replica_id_of(rank)
        sr = layout.shard_rank_of(rank)
        assert 0 <= rid < 4
        assert 0 <= sr < 2
        assert rank in replica_ranks(rid, 2)
        assert rank in replicate_peer_ranks(sr, 2, 4)
