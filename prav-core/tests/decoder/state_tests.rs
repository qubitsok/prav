//! Unit tests for DecodingState initialization and state management.
//!
//! Tests the core decoder state structure:
//! - Stride calculation and power-of-2 rounding
//! - Row mask generation for small strides
//! - Valid mask initialization for 2D and 3D grids
//! - Erasure loading and effective mask updates
//! - Sparse reset behavior and dirty tracking

use prav_core::arena::Arena;
use prav_core::decoder::{DecodingState, FLAG_VALID_FULL};
use prav_core::topology::SquareGrid;

/// Helper to create a decoder for testing
fn create_decoder<'a, const STRIDE_Y: usize>(
    arena: &mut Arena<'a>,
    width: usize,
    height: usize,
) -> DecodingState<'a, SquareGrid, STRIDE_Y> {
    DecodingState::<SquareGrid, STRIDE_Y>::new(arena, width, height, 1)
}

/// Helper to create a 3D decoder for testing
fn create_decoder_3d<'a, const STRIDE_Y: usize>(
    arena: &mut Arena<'a>,
    width: usize,
    height: usize,
    depth: usize,
) -> DecodingState<'a, SquareGrid, STRIDE_Y> {
    DecodingState::<SquareGrid, STRIDE_Y>::new(arena, width, height, depth)
}

// =============================================================================
// Stride Calculation Tests
// =============================================================================

#[test]
fn test_new_stride_8x8_grid() {
    // 8x8 grid: max_dim = 8, next_power_of_two = 8
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = create_decoder::<8>(&mut arena, 8, 8);

    assert_eq!(decoder.stride_y, 8, "8x8 grid should have stride_y = 8");
    assert_eq!(decoder.width, 8);
    assert_eq!(decoder.height, 8);
}

#[test]
fn test_new_stride_9x9_grid() {
    // 9x9 grid: max_dim = 9, next_power_of_two = 16
    let mut memory = vec![0u8; 2 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = create_decoder::<16>(&mut arena, 9, 9);

    assert_eq!(decoder.stride_y, 16, "9x9 grid should have stride_y = 16");
}

#[test]
fn test_new_stride_rectangular_grid() {
    // 4x12 grid: max_dim = 12, next_power_of_two = 16
    let mut memory = vec![0u8; 2 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = create_decoder::<16>(&mut arena, 4, 12);

    assert_eq!(decoder.stride_y, 16, "4x12 grid should have stride_y = 16");
    assert_eq!(decoder.width, 4);
    assert_eq!(decoder.height, 12);
}

#[test]
fn test_new_stride_33x33_grid() {
    // 33x33 grid: max_dim = 33, next_power_of_two = 64
    let mut memory = vec![0u8; 32 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = create_decoder::<64>(&mut arena, 33, 33);

    assert_eq!(decoder.stride_y, 64, "33x33 grid should have stride_y = 64");
}

// =============================================================================
// Row Mask Tests (for stride < 64)
// =============================================================================

#[test]
fn test_row_masks_stride_8() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = create_decoder::<8>(&mut arena, 8, 8);

    // For stride 8, row_start_mask should have bit 0, 8, 16, 24, 32, 40, 48, 56 set
    // row_end_mask should have bit 7, 15, 23, 31, 39, 47, 55, 63 set
    let expected_start: u64 = 0x0101_0101_0101_0101; // Every 8th bit starting at 0
    let expected_end: u64 = 0x8080_8080_8080_8080;   // Every 8th bit starting at 7

    assert_eq!(
        decoder.row_start_mask, expected_start,
        "row_start_mask incorrect for stride 8"
    );
    assert_eq!(
        decoder.row_end_mask, expected_end,
        "row_end_mask incorrect for stride 8"
    );
}

#[test]
fn test_row_masks_stride_16() {
    let mut memory = vec![0u8; 2 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = create_decoder::<16>(&mut arena, 16, 16);

    // For stride 16, row_start_mask should have bit 0, 16, 32, 48 set
    // row_end_mask should have bit 15, 31, 47, 63 set
    let expected_start: u64 = 0x0001_0001_0001_0001;
    let expected_end: u64 = 0x8000_8000_8000_8000;

    assert_eq!(
        decoder.row_start_mask, expected_start,
        "row_start_mask incorrect for stride 16"
    );
    assert_eq!(
        decoder.row_end_mask, expected_end,
        "row_end_mask incorrect for stride 16"
    );
}

#[test]
fn test_row_masks_stride_32() {
    let mut memory = vec![0u8; 8 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = create_decoder::<32>(&mut arena, 32, 32);

    // For stride 32, row_start_mask should have bit 0, 32 set
    // row_end_mask should have bit 31, 63 set
    let expected_start: u64 = 0x0000_0001_0000_0001;
    let expected_end: u64 = 0x8000_0000_8000_0000;

    assert_eq!(
        decoder.row_start_mask, expected_start,
        "row_start_mask incorrect for stride 32"
    );
    assert_eq!(
        decoder.row_end_mask, expected_end,
        "row_end_mask incorrect for stride 32"
    );
}

#[test]
fn test_row_masks_stride_64_or_larger() {
    let mut memory = vec![0u8; 32 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = create_decoder::<64>(&mut arena, 64, 64);

    // For stride >= 64, row masks should be 0 (one row per block)
    assert_eq!(
        decoder.row_start_mask, 0,
        "row_start_mask should be 0 for stride >= 64"
    );
    assert_eq!(
        decoder.row_end_mask, 0,
        "row_end_mask should be 0 for stride >= 64"
    );
}

// =============================================================================
// Valid Mask Initialization Tests
// =============================================================================

#[test]
fn test_valid_mask_full_block() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = create_decoder::<8>(&mut arena, 8, 8);

    // 8x8 grid with stride 8 fits exactly in one block (64 nodes)
    // Block 0 should have all bits set
    assert_eq!(
        decoder.blocks_state[0].valid_mask, !0u64,
        "Block 0 should have all valid bits set for 8x8 grid"
    );
    assert_ne!(
        decoder.blocks_state[0].flags & FLAG_VALID_FULL, 0,
        "FLAG_VALID_FULL should be set for full block"
    );
}

#[test]
fn test_valid_mask_partial_block() {
    // 6x4 grid: max_dim = 6, next_power_of_two = 8, so stride = 8
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = create_decoder::<8>(&mut arena, 6, 4);

    // 6x4 grid with stride 8
    // Valid nodes: (x,y) where x < 6 and y < 4
    // In block layout: rows 0-3 have first 6 bits set
    // Expected: bits 0-5, 8-13, 16-21, 24-29 (6 bits per row, 4 rows)
    let expected: u64 = 0x3F3F_3F3F;

    assert_eq!(
        decoder.blocks_state[0].valid_mask, expected,
        "Block 0 should have correct valid_mask for 6x4 grid"
    );
    assert_eq!(
        decoder.blocks_state[0].flags & FLAG_VALID_FULL, 0,
        "FLAG_VALID_FULL should NOT be set for partial block"
    );
}

#[test]
fn test_valid_mask_multi_block() {
    let mut memory = vec![0u8; 8 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = create_decoder::<32>(&mut arena, 32, 32);

    // 32x32 grid with stride 32 = 1024 nodes = 16 blocks
    // Each block should have specific valid_mask pattern

    // Check that we have multiple blocks
    assert!(decoder.blocks_state.len() >= 16, "Should have at least 16 blocks");

    // First block should be fully valid
    assert_eq!(
        decoder.blocks_state[0].valid_mask, !0u64,
        "First block should be fully valid"
    );
}

// =============================================================================
// 3D Grid Initialization Tests
// =============================================================================

#[test]
fn test_valid_mask_3d_grid() {
    // 8x8x4 grid: max_dim = 8, stride = 8
    let mut memory = vec![0u8; 32 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = create_decoder_3d::<8>(&mut arena, 8, 8, 4);

    // 8x8x4 grid = 256 nodes with 3D indexing

    // Count total valid nodes
    let mut total_valid = 0u64;
    for block in decoder.blocks_state.iter() {
        total_valid += block.valid_mask.count_ones() as u64;
    }

    // Should have 256 valid nodes for 8x8x4 grid
    assert_eq!(total_valid, 256, "8x8x4 grid should have 256 valid nodes");
}

// =============================================================================
// Effective Mask and Erasure Tests
// =============================================================================

#[test]
fn test_effective_mask_initial() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = create_decoder::<8>(&mut arena, 8, 8);

    // Initially, effective_mask should equal valid_mask (no erasures)
    assert_eq!(
        decoder.blocks_state[0].effective_mask,
        decoder.blocks_state[0].valid_mask,
        "effective_mask should equal valid_mask initially"
    );
    assert_eq!(
        decoder.blocks_state[0].erasure_mask, 0,
        "erasure_mask should be 0 initially"
    );
}

#[test]
fn test_load_erasures_basic() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<8>(&mut arena, 8, 8);

    // Mark some nodes as erased
    let erasures = [0b1010u64]; // Erase nodes 1 and 3

    decoder.load_erasures(&erasures);

    assert_eq!(
        decoder.blocks_state[0].erasure_mask, 0b1010,
        "erasure_mask should be set correctly"
    );

    // effective_mask = valid_mask & !erasure_mask
    let expected_effective = decoder.blocks_state[0].valid_mask & !0b1010u64;
    assert_eq!(
        decoder.blocks_state[0].effective_mask, expected_effective,
        "effective_mask should exclude erased nodes"
    );
}

#[test]
fn test_load_erasures_partial_array() {
    let mut memory = vec![0u8; 8 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<32>(&mut arena, 32, 32);

    // Only provide erasures for first 2 blocks
    let erasures = [0xFFu64, 0xF0u64];

    decoder.load_erasures(&erasures);

    // First two blocks should have erasures
    assert_eq!(decoder.blocks_state[0].erasure_mask, 0xFF);
    assert_eq!(decoder.blocks_state[1].erasure_mask, 0xF0);

    // Remaining blocks should have no erasures and full effective_mask
    for block in decoder.blocks_state.iter().skip(2) {
        assert_eq!(block.erasure_mask, 0, "Blocks beyond erasures array should have no erasures");
        assert_eq!(block.effective_mask, block.valid_mask, "effective_mask should equal valid_mask");
    }
}

#[test]
fn test_load_erasures_clears_previous() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<8>(&mut arena, 8, 8);

    // First load some erasures
    decoder.load_erasures(&[0xFFu64]);
    assert_eq!(decoder.blocks_state[0].erasure_mask, 0xFF);

    // Load different erasures - previous should be replaced via effective_mask recalculation
    decoder.load_erasures(&[0x0Fu64]);
    assert_eq!(decoder.blocks_state[0].erasure_mask, 0x0F);
}

// =============================================================================
// mark_block_dirty Tests
// =============================================================================

#[test]
fn test_mark_block_dirty_single() {
    let mut memory = vec![0u8; 8 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<32>(&mut arena, 32, 32);

    // Clear dirty mask
    decoder.block_dirty_mask.fill(0);

    // Mark block 5 dirty
    decoder.mark_block_dirty(5);

    // Check that bit 5 is set in word 0
    assert_ne!(
        decoder.block_dirty_mask[0] & (1 << 5), 0,
        "Block 5 should be marked dirty"
    );

    // Other bits should be clear
    assert_eq!(
        decoder.block_dirty_mask[0] & !(1 << 5), 0,
        "Only block 5 should be dirty"
    );
}

#[test]
fn test_mark_block_dirty_multiple_words() {
    let mut memory = vec![0u8; 32 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<64>(&mut arena, 64, 64);

    // Clear dirty mask
    decoder.block_dirty_mask.fill(0);

    // Mark block 65 dirty (should be bit 1 in word 1)
    decoder.mark_block_dirty(65);

    // Check word 1, bit 1
    assert_ne!(
        decoder.block_dirty_mask[1] & (1 << 1), 0,
        "Block 65 should be bit 1 in word 1"
    );
}

#[test]
fn test_mark_block_dirty_idempotent() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<8>(&mut arena, 8, 8);

    // Clear dirty mask
    decoder.block_dirty_mask.fill(0);

    // Mark same block dirty multiple times
    decoder.mark_block_dirty(0);
    let first_state = decoder.block_dirty_mask[0];

    decoder.mark_block_dirty(0);
    let second_state = decoder.block_dirty_mask[0];

    assert_eq!(first_state, second_state, "mark_block_dirty should be idempotent");
}

// =============================================================================
// is_small_grid Tests
// =============================================================================

#[test]
fn test_is_small_grid_true() {
    let mut memory = vec![0u8; 8 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = create_decoder::<32>(&mut arena, 32, 32);

    // 32x32 with stride 32 = 1024 nodes = 16 blocks + 1 boundary = 17 blocks
    // 17 <= 65, so should be small
    assert!(decoder.is_small_grid(), "32x32 grid should be small");
}

#[test]
fn test_is_small_grid_boundary_case() {
    // 64x64 with stride 64 = 4096 nodes = 64 blocks + 1 = 65 blocks
    // This is exactly at the boundary
    let mut memory = vec![0u8; 64 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = create_decoder::<64>(&mut arena, 64, 64);

    // 64 data blocks + 1 boundary block = 65 blocks
    assert!(decoder.is_small_grid(), "64x64 grid (65 blocks) should still be small");
}

#[test]
fn test_is_small_grid_false() {
    // Need a grid with > 65 blocks
    // 128x128 with stride 128 = 16384 nodes = 256 blocks
    let mut memory = vec![0u8; 256 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = create_decoder::<128>(&mut arena, 128, 128);

    assert!(!decoder.is_small_grid(), "128x128 grid should NOT be small");
}

// =============================================================================
// push_next Tests
// =============================================================================

#[test]
fn test_push_next_sets_bit() {
    let mut memory = vec![0u8; 8 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<32>(&mut arena, 32, 32);

    // Clear queued mask
    decoder.queued_mask.fill(0);

    // Push block 7 to next queue
    decoder.push_next(7);

    assert_ne!(
        decoder.queued_mask[0] & (1 << 7), 0,
        "Block 7 should be queued"
    );
}

#[test]
fn test_push_next_multiple_blocks() {
    let mut memory = vec![0u8; 8 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<32>(&mut arena, 32, 32);

    // Clear queued mask
    decoder.queued_mask.fill(0);

    // Push multiple blocks
    decoder.push_next(0);
    decoder.push_next(3);
    decoder.push_next(15);

    let expected = (1u64 << 0) | (1u64 << 3) | (1u64 << 15);
    assert_eq!(
        decoder.queued_mask[0] & expected, expected,
        "All pushed blocks should be queued"
    );
}

// =============================================================================
// sparse_reset Tests
// =============================================================================

#[test]
fn test_sparse_reset_clears_dirty_blocks() {
    let mut memory = vec![0u8; 8 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<32>(&mut arena, 32, 32);

    // Modify some state in block 0
    decoder.blocks_state[0].boundary = 0xDEAD;
    decoder.blocks_state[0].occupied = 0xBEEF;
    decoder.blocks_state[0].root = 42;
    decoder.defect_mask[0] = 0xCAFE;

    // Mark block 0 dirty
    decoder.block_dirty_mask[0] = 1;

    // Reset
    decoder.sparse_reset();

    // Block 0 should be cleared
    assert_eq!(decoder.blocks_state[0].boundary, 0, "boundary should be reset");
    assert_eq!(decoder.blocks_state[0].occupied, 0, "occupied should be reset");
    assert_eq!(decoder.blocks_state[0].root, u32::MAX, "root should be invalidated");
    assert_eq!(decoder.defect_mask[0], 0, "defect_mask should be cleared");

    // Dirty mask should be cleared
    assert_eq!(decoder.block_dirty_mask[0], 0, "dirty mask should be cleared");
}

#[test]
fn test_sparse_reset_preserves_clean_blocks() {
    let mut memory = vec![0u8; 8 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<32>(&mut arena, 32, 32);

    // Modify state in block 0 (will NOT mark as dirty)
    decoder.blocks_state[0].boundary = 0xDEAD;
    decoder.blocks_state[0].occupied = 0xBEEF;

    // Mark only block 1 as dirty, not block 0
    decoder.block_dirty_mask[0] = 0b10; // Only bit 1 (block 1) set

    // Reset
    decoder.sparse_reset();

    // Block 0 should NOT be touched (not dirty)
    assert_eq!(decoder.blocks_state[0].boundary, 0xDEAD, "clean block boundary should be preserved");
    assert_eq!(decoder.blocks_state[0].occupied, 0xBEEF, "clean block occupied should be preserved");
}

#[test]
fn test_sparse_reset_resets_parents() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<8>(&mut arena, 8, 8);

    // Modify parent pointers
    decoder.parents[0] = 10;
    decoder.parents[1] = 10;
    decoder.parents[2] = 10;

    // Mark block 0 dirty
    decoder.block_dirty_mask[0] = 1;

    // Reset
    decoder.sparse_reset();

    // Parents in block 0 should be self-referential
    assert_eq!(decoder.parents[0], 0, "parent[0] should be reset to 0");
    assert_eq!(decoder.parents[1], 1, "parent[1] should be reset to 1");
    assert_eq!(decoder.parents[2], 2, "parent[2] should be reset to 2");
}

#[test]
fn test_sparse_reset_preserves_boundary_node() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<8>(&mut arena, 8, 8);

    let boundary_idx = decoder.parents.len() - 1;
    let num_blocks = decoder.blocks_state.len();

    // Mark only valid blocks as dirty (not beyond the actual block count)
    // For 8x8 with stride 8 = 64 nodes = 1 data block + boundary
    decoder.block_dirty_mask[0] = (1u64 << num_blocks) - 1;

    // Reset
    decoder.sparse_reset();

    // Boundary node should still be self-referential
    assert_eq!(
        decoder.parents[boundary_idx], boundary_idx as u32,
        "Boundary node should be preserved after reset"
    );
}

#[test]
fn test_sparse_reset_clears_masks() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<8>(&mut arena, 8, 8);

    let num_blocks = decoder.blocks_state.len();

    // Set some state
    decoder.queued_mask[0] = 0xFF;
    decoder.active_mask[0] = 0xFF;
    decoder.active_block_mask = 0xFF;

    // Mark only valid blocks as dirty
    decoder.block_dirty_mask[0] = (1u64 << num_blocks) - 1;

    // Reset
    decoder.sparse_reset();

    // All masks should be cleared
    assert!(decoder.queued_mask.iter().all(|&w| w == 0), "queued_mask should be cleared");
    assert!(decoder.active_mask.iter().all(|&w| w == 0), "active_mask should be cleared");
    assert_eq!(decoder.active_block_mask, 0, "active_block_mask should be cleared");
}

// =============================================================================
// initialize_internal Tests
// =============================================================================

#[test]
fn test_initialize_internal_resets_state() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<8>(&mut arena, 8, 8);

    // Modify state
    decoder.blocks_state[0].boundary = 0xFFFF;
    decoder.blocks_state[0].occupied = 0xFFFF;
    decoder.defect_mask[0] = 0xFFFF;
    decoder.path_mark[0] = 0xFFFF;

    // Re-initialize
    decoder.initialize_internal();

    // State should be reset
    assert_eq!(decoder.blocks_state[0].boundary, 0);
    assert_eq!(decoder.blocks_state[0].occupied, 0);
    assert_eq!(decoder.blocks_state[0].root, u32::MAX);
    assert_eq!(decoder.defect_mask[0], 0);
    assert_eq!(decoder.path_mark[0], 0);
}

#[test]
fn test_initialize_internal_resets_parents() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<8>(&mut arena, 8, 8);

    // Modify parents
    for i in 0..decoder.parents.len() {
        decoder.parents[i] = 999;
    }

    // Re-initialize
    decoder.initialize_internal();

    // Parents should be self-referential
    for (i, &p) in decoder.parents.iter().enumerate() {
        assert_eq!(p, i as u32, "parent[{}] should be {}", i, i);
    }
}
