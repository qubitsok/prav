#![cfg(test)]

use prav_core::arena::Arena;
use prav_core::decoder::state::{BlockStateHot, DecodingState, FLAG_VALID_FULL};
use prav_core::topology::SquareGrid;
use std::mem::{align_of, size_of};

#[test]
fn test_block_state_layout() {
    assert_eq!(
        size_of::<BlockStateHot>(),
        64,
        "BlockStateHot must be exactly 64 bytes"
    );
    assert_eq!(
        align_of::<BlockStateHot>(),
        64,
        "BlockStateHot must be 64-byte aligned"
    );
}

#[test]
fn test_flag_valid_full_initialization() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

    // Initial state: valid_mask is !0 for internal blocks
    for (i, block) in decoder.blocks_state.iter().enumerate() {
        if i < decoder.blocks_state.len() - 1 {
            // Last block might be partial depending on grid size
            // For 32x32 stride 32, we have 16 blocks (1024 nodes).
            // Blocks are fully valid.
            assert_eq!(
                block.flags & FLAG_VALID_FULL,
                FLAG_VALID_FULL,
                "Block {} should be marked FULL VALID",
                i
            );
            assert_eq!(block.effective_mask, !0, "Effective mask should be !0");
        }
    }
}

#[test]
fn test_hot_cold_sync_on_erasure_load() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

    let erasure_pattern = 0xAA55AA55AA55AA55;
    let erasures = vec![erasure_pattern];

    decoder.load_erasures(&erasures);

    let block_hot = &decoder.blocks_state[0];
    let block_cold = &decoder.blocks_state[0];

    assert_eq!(block_cold.erasure_mask, erasure_pattern);
    assert_eq!(
        block_hot.effective_mask,
        block_cold.valid_mask & !erasure_pattern
    );

    // FLAG_VALID_FULL depends on VALID MASK, not effective mask.
    // If valid mask was !0, it should remain FLAG_VALID_FULL even if erased.
    // (Optimization allows skipping cold load of valid_mask for boundary checks,
    // erasures are handled via effective_mask which is hot).
    assert_eq!(block_hot.flags & FLAG_VALID_FULL, FLAG_VALID_FULL);
}

#[test]
fn test_partial_validity_flag() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    // 33x32 grid.
    // Stride Y = 64 (next pow 2 of 33).
    // Width 33.
    let decoder = DecodingState::<SquareGrid, 64>::new(&mut arena, 33, 32, 1);

    let block0 = &decoder.blocks_state[0];
    let block0_static = &decoder.blocks_state[0];

    assert_ne!(
        block0_static.valid_mask, !0,
        "Block 0 should have padding bits"
    );
    assert_eq!(
        block0.flags & FLAG_VALID_FULL,
        0,
        "Block 0 should NOT be marked FULL VALID"
    );
}

#[test]
fn test_block_root_reset() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

    // Set root
    decoder.blocks_state[0].root = 123;

    decoder.mark_block_dirty(0);
    decoder.sparse_reset();

    assert_eq!(
        decoder.blocks_state[0].root,
        u32::MAX,
        "Root should be reset to MAX (None)"
    );
}
