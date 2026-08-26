import pytest

from vllm.engine.block_manager import BlockManager
from vllm.engine.sequence import Sequence


BLOCK_SIZE = 4


def make_seq(token_ids: list[int]) -> Sequence:
    """
    Change this function if your Sequence constructor has a different
    signature.
    """
    return Sequence(token_ids)


def simulate_forward(manager: BlockManager, seq: Sequence):
    """
    Pretend every currently known token has completed its forward pass
    and therefore has valid K/V.
    """
    seq.num_computed_tokens = len(seq)
    manager.hash_blocks(seq)


def assert_manager_consistent(manager: BlockManager):
    """Check invariants that must always hold."""
    free_ids = set(manager.free_block_ids)

    for block in manager.blocks:
        assert block.ref_count >= 0

        if block.ref_count == 0:
            assert block.block_id in free_ids
        else:
            assert block.block_id not in free_ids

    for block_hash, block_id in manager.hash_to_block_id.items():
        assert manager.blocks[block_id].hash == block_hash


@pytest.mark.parametrize("num_tokens", range(1, 21))
def test_normal_allocation(num_tokens):
    manager = BlockManager(num_blocks=10, block_size=BLOCK_SIZE)
    seq = make_seq(list(range(num_tokens)))

    result = manager.allocate(seq)

    expected_blocks = (
        num_tokens + BLOCK_SIZE - 1
    ) // BLOCK_SIZE

    assert result == 0
    assert len(seq.block_list) == expected_blocks
    assert len(manager.free_block_ids) == 10 - expected_blocks
    assert seq.num_computed_tokens == 0
    assert seq.num_hashed_tokens == 0

    for block_id in seq.block_list:
        assert manager.blocks[block_id].ref_count == 1
        assert block_id not in manager.free_block_ids

    assert_manager_consistent(manager)


def test_prefix_sharing():
    manager = BlockManager(num_blocks=8, block_size=BLOCK_SIZE)

    seq1 = make_seq([
        1, 2, 3, 4,
        5, 6, 7, 8,
        11,
    ])

    assert manager.allocate(seq1) == 0
    simulate_forward(manager, seq1)

    seq1_block_ids = seq1.block_list.copy()

    first_hash = manager.blocks[seq1_block_ids[0]].hash
    second_hash = manager.blocks[seq1_block_ids[1]].hash

    assert first_hash is not None
    assert second_hash is not None

    # The third block is partial, so it should not be hashed.
    assert manager.blocks[seq1_block_ids[2]].hash is None

    manager.deallocate(seq1)

    seq2 = make_seq([
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10,
    ])

    assert manager.allocate(seq2) == 0

    # The first two physical blocks should be reused.
    assert seq2.block_list[:2] == seq1_block_ids[:2]

    # Their hashes should still be the same.
    assert manager.blocks[seq2.block_list[0]].hash == first_hash
    assert manager.blocks[seq2.block_list[1]].hash == second_hash

    # Eight prompt tokens were found in the prefix cache.
    assert seq2.num_computed_tokens == 8
    assert seq2.num_hashed_tokens == 8

    # The third block is newly allocated and still incomplete.
    assert manager.blocks[seq2.block_list[2]].hash is None

    assert_manager_consistent(manager)


def test_hash_chain_diverges_at_third_block():
    manager = BlockManager(num_blocks=8, block_size=BLOCK_SIZE)

    first_block = [1, 2, 3, 4]
    second_block = [5, 6, 7, 8]

    third_block_a = [11, 12, 13, 14]
    third_block_b = [9, 10, 15, 16]

    hash_a0 = manager.get_hash(first_block)
    hash_a1 = manager.get_hash(second_block, hash_a0)
    hash_a2 = manager.get_hash(third_block_a, hash_a1)

    hash_b0 = manager.get_hash(first_block)
    hash_b1 = manager.get_hash(second_block, hash_b0)
    hash_b2 = manager.get_hash(third_block_b, hash_b1)

    assert hash_a0 == hash_b0
    assert hash_a1 == hash_b1
    assert hash_a2 != hash_b2


def test_deallocation_frees_blocks():
    manager = BlockManager(num_blocks=5, block_size=BLOCK_SIZE)

    seq = make_seq([
        1, 2, 3, 4,
        5, 6, 7, 8,
        9,
    ])

    assert manager.allocate(seq) == 0
    simulate_forward(manager, seq)

    block_ids = seq.block_list.copy()

    manager.deallocate(seq)

    assert seq.block_list == []
    assert seq.num_computed_tokens == 0
    assert seq.num_hashed_tokens == 0

    for block_id in block_ids:
        block = manager.blocks[block_id]

        assert block.ref_count == 0
        assert block_id in manager.free_block_ids

    # Full blocks remain cached after deallocation.
    assert manager.blocks[block_ids[0]].hash is not None
    assert manager.blocks[block_ids[1]].hash is not None

    # Partial final block is not cached.
    assert manager.blocks[block_ids[2]].hash is None

    assert_manager_consistent(manager)


def test_inactive_cached_block_is_reactivated():
    manager = BlockManager(num_blocks=3, block_size=BLOCK_SIZE)

    first = make_seq([
        0, 1, 2, 3,
        99,
    ])

    assert manager.allocate(first) == 0
    simulate_forward(manager, first)

    cached_block_id = first.block_list[0]

    # Initial allocation should have used physical block 0.
    assert cached_block_id == 0
    assert manager.blocks[cached_block_id].hash is not None

    manager.deallocate(first)

    # The cached block is inactive, so it is in the free queue.
    assert cached_block_id in manager.free_block_ids
    assert manager.blocks[cached_block_id].ref_count == 0

    # Cached blocks should be at the end of the eviction queue.
    assert next(reversed(manager.free_block_ids)) == cached_block_id

    second = make_seq([
        0, 1, 2, 3,
        100,
    ])

    assert manager.allocate(second) == 0

    # Cache hit should remove physical block 0 from the free queue.
    assert cached_block_id not in manager.free_block_ids

    # The cached physical block should be attached to the new sequence.
    assert second.block_list[0] == cached_block_id
    assert manager.blocks[cached_block_id].ref_count == 1

    # Four tokens already have cached K/V.
    assert second.num_computed_tokens == 4
    assert second.num_hashed_tokens == 4

    assert_manager_consistent(manager)


def test_failed_allocation_is_atomic():
    manager = BlockManager(num_blocks=1, block_size=BLOCK_SIZE)

    # Five tokens require two blocks, but only one exists.
    seq = make_seq([1, 2, 3, 4, 5])

    free_before = list(manager.free_block_ids)
    refs_before = [
        block.ref_count
        for block in manager.blocks
    ]

    result = manager.allocate(seq)

    assert result == -1

    # Failed allocation must not modify anything.
    assert seq.block_list == []
    assert seq.num_computed_tokens == 0
    assert seq.num_hashed_tokens == 0

    assert list(manager.free_block_ids) == free_before

    assert [
        block.ref_count
        for block in manager.blocks
    ] == refs_before

    assert_manager_consistent(manager)


def test_active_prefix_reference_count():
    manager = BlockManager(num_blocks=5, block_size=BLOCK_SIZE)

    seq1 = make_seq([
        1, 2, 3, 4,
        5,
    ])

    assert manager.allocate(seq1) == 0
    simulate_forward(manager, seq1)

    shared_block_id = seq1.block_list[0]

    assert manager.blocks[shared_block_id].ref_count == 1
    assert shared_block_id not in manager.free_block_ids

    seq2 = make_seq([
        1, 2, 3, 4,
        6,
    ])

    assert manager.allocate(seq2) == 0

    # Both sequences should point to the same cached physical block.
    assert seq2.block_list[0] == shared_block_id
    assert manager.blocks[shared_block_id].ref_count == 2

    manager.deallocate(seq1)

    # seq2 still owns the shared block.
    assert manager.blocks[shared_block_id].ref_count == 1
    assert shared_block_id not in manager.free_block_ids

    manager.deallocate(seq2)

    # Nobody owns it anymore, so it returns to the free queue.
    assert manager.blocks[shared_block_id].ref_count == 0
    assert shared_block_id in manager.free_block_ids

    assert_manager_consistent(manager)


def test_only_computed_full_blocks_are_hashed():
    manager = BlockManager(num_blocks=4, block_size=BLOCK_SIZE)

    seq = make_seq([
        1, 2, 3, 4,
        5, 6, 7, 8,
    ])

    assert manager.allocate(seq) == 0

    # Eight token IDs are known, but only seven have valid K/V.
    seq.num_computed_tokens = 7
    manager.hash_blocks(seq)

    first_block = manager.blocks[seq.block_list[0]]
    second_block = manager.blocks[seq.block_list[1]]

    assert first_block.hash is not None
    assert first_block.token_ids == [1, 2, 3, 4]

    # Token 8 has not completed its forward pass, so this block is incomplete.
    assert second_block.hash is None
    assert second_block.token_ids == []

    assert seq.num_hashed_tokens == 4

    assert_manager_consistent(manager)