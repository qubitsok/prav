use prav_core::arena::Arena;
use prav_core::decoder::DecodingState;
use prav_core::topology::SquareGrid;

#[test]
fn test_internal_sparse_reset() {
    extern crate std;
    let mut memory = std::vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

    // Manually touch some blocks (simulating activity)
    decoder.blocks_state[0].boundary = 0xFFFF;
    decoder.mark_block_dirty(0);

    decoder.blocks_state[1].occupied = 0xAAAA;
    decoder.mark_block_dirty(1);

    decoder.defect_mask[4] = 0x1234;
    decoder.mark_block_dirty(4);

    // Verify bits set in dirty mask
    // Blocks 0, 1, 4 are all in word 0.
    assert_ne!(decoder.block_dirty_mask[0] & (1 << 0), 0);
    assert_ne!(decoder.block_dirty_mask[0] & (1 << 1), 0);
    assert_ne!(decoder.block_dirty_mask[0] & (1 << 4), 0);

    assert_ne!(decoder.blocks_state[0].boundary, 0);
    assert_ne!(decoder.blocks_state[1].occupied, 0);
    assert_ne!(decoder.defect_mask[4], 0);

    // Perform reset
    decoder.sparse_reset();

    // Verify everything is clean
    assert_eq!(decoder.blocks_state[0].boundary, 0);
    assert_eq!(decoder.blocks_state[1].occupied, 0);
    assert_eq!(decoder.defect_mask[4], 0);

    // Verify dirty mask is cleared
    // block 0, 1, 4 are in first u64 of mask (indices < 64)
    assert_eq!(decoder.block_dirty_mask[0], 0);
}

#[test]
fn test_dirty_tracking_optimization() {
    extern crate std;
    let mut memory = std::vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

    // Initial state: nothing touched
    assert_eq!(decoder.block_dirty_mask[0], 0);

    // 1. Check FIND DOES mark dirty (because of path compression)
    // We manually set up a chain: 0 -> 1 -> 2
    decoder.parents[0] = 1;
    decoder.parents[1] = 2;
    decoder.parents[2] = 2;

    // find(0) should compress path 0->2 and MARK block 0 dirty
    // 0 is in block 0.
    decoder.find(0);

    // Check path compression happened
    assert_eq!(decoder.parents[0], 2);

    // Block 0 should be dirty
    assert_ne!(decoder.block_dirty_mask[0] & 1, 0);

    // 2. Check UNION DOES mark dirty
    // union(2, 3) -> merges block 0 (node 2) and block 0 (node 3)
    // Both are in block 0.
    unsafe {
        decoder.union(2, 3);
    }

    // Still block 0 dirty (no new blocks)
    assert_ne!(decoder.block_dirty_mask[0] & 1, 0);
}
