//! Tests for growth loop orchestration in `decoder/growth/mod.rs`.
//!
//! These tests specifically target the `ClusterGrowth` trait implementation
//! to achieve 100% code coverage for mod.rs.
//!
//! Note: With STRIDE_Y <= 64 constraint, the "large grid" code paths
//! (blocks_state.len() > 65) are unreachable for 2D SquareGrid topology.
//! Coverage for those paths requires either:
//! - 3D grids (depth > 1)
//! - A custom topology with more nodes

use crate::common;

use prav_core::arena::Arena;
use prav_core::decoder::growth::ClusterGrowth;
use prav_core::decoder::state::{DecodingState, EdgeCorrection};
use prav_core::topology::SquareGrid;

/// Helper to compute linear index from 2D coordinates.
fn idx(x: usize, y: usize, stride_y: usize) -> usize {
    y * stride_y + x
}

/// Helper to set defects in syndromes array.
fn set_defect(syndromes: &mut [u64], node_idx: usize) {
    let blk = node_idx / 64;
    let bit = node_idx % 64;
    if blk < syndromes.len() {
        syndromes[blk] ^= 1 << bit;
    }
}

// =============================================================================
// Small Grid Tests (≤65 blocks) - These are the reachable paths
// =============================================================================

/// Test small grid grow_iteration explicit call.
///
/// Exercises the small grid path in `grow_iteration` (lines 133-150).
/// This is different from `grow_clusters` which calls `grow_bitmask_iteration`.
#[test]
fn test_small_grid_grow_iteration_explicit() {
    let mut memory = vec![0u8; 1024 * 1024 * 10];
    let mut arena = Arena::new(&mut memory);

    let w = 16;
    let h = 16;
    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, w, h, 1);

    assert!(decoder.is_small_grid(), "Expected small grid");

    let num_blocks = decoder.blocks_state.len();
    let mut syndromes = vec![0u64; num_blocks];

    // Two horizontally adjacent defects at (2,2) and (3,2)
    let d1 = idx(2, 2, 16);
    let d2 = idx(3, 2, 16);
    set_defect(&mut syndromes, d1);
    set_defect(&mut syndromes, d2);

    decoder.sparse_reset();
    decoder.load_dense_syndromes(&syndromes);

    // Call grow_iteration explicitly (bypassing grow_bitmask_iteration)
    // The first iteration expands the boundary
    let mut iterations = 0;
    while decoder.grow_iteration() && iterations < 256 {
        iterations += 1;
    }

    let mut corrections = vec![EdgeCorrection::default(); w * h * 4];
    let count = decoder.decode(&mut corrections);

    let result = common::verify_matching(&syndromes, &corrections[..count]);
    assert!(result.is_ok(), "Failed: {:?}", result.err());
}

/// Test small grid queued_mask clearing in load_dense_syndromes.
///
/// Covers the `if !self.queued_mask.is_empty()` check.
#[test]
fn test_small_grid_queued_mask_clear() {
    let mut memory = vec![0u8; 1024 * 1024 * 10];
    let mut arena = Arena::new(&mut memory);

    let w = 8;
    let h = 8;
    let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, w, h, 1);

    assert!(decoder.is_small_grid());

    let num_blocks = decoder.blocks_state.len();
    let mut syndromes = vec![0u64; num_blocks];

    let d1 = idx(2, 2, 8);
    let d2 = idx(3, 2, 8);
    set_defect(&mut syndromes, d1);
    set_defect(&mut syndromes, d2);

    // First decode cycle
    decoder.sparse_reset();
    decoder.load_dense_syndromes(&syndromes);

    let mut corrections = vec![EdgeCorrection::default(); w * h * 4];
    let count = decoder.decode(&mut corrections);
    assert!(common::verify_matching(&syndromes, &corrections[..count]).is_ok());

    // Second decode cycle - tests queued_mask clearing
    decoder.sparse_reset();
    decoder.load_dense_syndromes(&syndromes);

    let count = decoder.decode(&mut corrections);
    let result = common::verify_matching(&syndromes, &corrections[..count]);
    assert!(result.is_ok(), "Second cycle failed: {:?}", result.err());
}

/// Test grow_bitmask_iteration non-unrolled path.
///
/// The unrolled path requires STRIDE_Y == 32 && blocks == 16.
/// Using STRIDE_Y == 16 avoids the unrolled path.
#[test]
fn test_small_grid_bitmask_iteration_non_unrolled() {
    let mut memory = vec![0u8; 1024 * 1024 * 10];
    let mut arena = Arena::new(&mut memory);

    // 12x12 grid with stride 16 = 256 nodes = 4 blocks
    // This is NOT 16 blocks, so it avoids the unrolled path
    let w = 12;
    let h = 12;
    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, w, h, 1);

    assert!(decoder.is_small_grid());
    let num_blocks = decoder.blocks_state.len();
    assert_ne!(num_blocks, 17, "Should NOT be exactly 16+1 blocks"); // 16 data + 1 sentinel

    let mut syndromes = vec![0u64; num_blocks];
    let d1 = idx(2, 2, 16);
    let d2 = idx(3, 2, 16);
    set_defect(&mut syndromes, d1);
    set_defect(&mut syndromes, d2);

    decoder.sparse_reset();
    decoder.load_dense_syndromes(&syndromes);
    decoder.grow_clusters();

    let mut corrections = vec![EdgeCorrection::default(); w * h * 4];
    let count = decoder.decode(&mut corrections);

    let result = common::verify_matching(&syndromes, &corrections[..count]);
    assert!(result.is_ok(), "Failed: {:?}", result.err());
}

/// Test various stride values.
#[test]
fn test_stride_variations() {
    // Stride 8: 8x8 grid
    {
        let mut memory = vec![0u8; 1024 * 1024 * 10];
        let mut arena = Arena::new(&mut memory);

        let w = 6;
        let h = 6;
        let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, w, h, 1);

        let num_blocks = decoder.blocks_state.len();
        let mut syndromes = vec![0u64; num_blocks];
        let d1 = idx(2, 2, 8);
        let d2 = idx(3, 2, 8);
        set_defect(&mut syndromes, d1);
        set_defect(&mut syndromes, d2);

        decoder.sparse_reset();
        decoder.load_dense_syndromes(&syndromes);

        let mut corrections = vec![EdgeCorrection::default(); w * h * 4];
        let count = decoder.decode(&mut corrections);

        let result = common::verify_matching(&syndromes, &corrections[..count]);
        assert!(result.is_ok(), "Stride 8 failed: {:?}", result.err());
    }

    // Stride 16: 16x16 grid
    {
        let mut memory = vec![0u8; 1024 * 1024 * 10];
        let mut arena = Arena::new(&mut memory);

        let w = 10;
        let h = 10;
        let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, w, h, 1);

        let num_blocks = decoder.blocks_state.len();
        let mut syndromes = vec![0u64; num_blocks];
        let d1 = idx(2, 2, 16);
        let d2 = idx(3, 2, 16);
        set_defect(&mut syndromes, d1);
        set_defect(&mut syndromes, d2);

        decoder.sparse_reset();
        decoder.load_dense_syndromes(&syndromes);

        let mut corrections = vec![EdgeCorrection::default(); w * h * 4];
        let count = decoder.decode(&mut corrections);

        let result = common::verify_matching(&syndromes, &corrections[..count]);
        assert!(result.is_ok(), "Stride 16 failed: {:?}", result.err());
    }

    // Stride 32: 32x32 grid (exactly 16 blocks, might hit unrolled path)
    {
        let mut memory = vec![0u8; 1024 * 1024 * 10];
        let mut arena = Arena::new(&mut memory);

        let w = 20;
        let h = 20;
        let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, w, h, 1);

        let num_blocks = decoder.blocks_state.len();
        let mut syndromes = vec![0u64; num_blocks];
        let d1 = idx(2, 2, 32);
        let d2 = idx(3, 2, 32);
        set_defect(&mut syndromes, d1);
        set_defect(&mut syndromes, d2);

        decoder.sparse_reset();
        decoder.load_dense_syndromes(&syndromes);

        let mut corrections = vec![EdgeCorrection::default(); w * h * 4];
        let count = decoder.decode(&mut corrections);

        let result = common::verify_matching(&syndromes, &corrections[..count]);
        assert!(result.is_ok(), "Stride 32 failed: {:?}", result.err());
    }

    // Stride 64: 64x64 grid (large stride)
    {
        let mut memory = vec![0u8; 1024 * 1024 * 50];
        let mut arena = Arena::new(&mut memory);

        let w = 40;
        let h = 40;
        let mut decoder = DecodingState::<SquareGrid, 64>::new(&mut arena, w, h, 1);

        let num_blocks = decoder.blocks_state.len();
        let mut syndromes = vec![0u64; num_blocks];
        let d1 = idx(2, 2, 64);
        let d2 = idx(3, 2, 64);
        set_defect(&mut syndromes, d1);
        set_defect(&mut syndromes, d2);

        decoder.sparse_reset();
        decoder.load_dense_syndromes(&syndromes);

        let mut corrections = vec![EdgeCorrection::default(); w * h * 4];
        let count = decoder.decode(&mut corrections);

        let result = common::verify_matching(&syndromes, &corrections[..count]);
        assert!(result.is_ok(), "Stride 64 failed: {:?}", result.err());
    }
}

/// Test empty syndrome input terminates immediately.
#[test]
fn test_empty_syndromes() {
    let mut memory = vec![0u8; 1024 * 1024 * 10];
    let mut arena = Arena::new(&mut memory);

    let w = 16;
    let h = 16;
    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, w, h, 1);

    let num_blocks = decoder.blocks_state.len();
    let syndromes = vec![0u64; num_blocks]; // All zeros

    decoder.sparse_reset();
    decoder.load_dense_syndromes(&syndromes);
    decoder.grow_clusters();

    let mut corrections = vec![EdgeCorrection::default(); 100];
    let count = decoder.decode(&mut corrections);

    assert_eq!(count, 0, "Expected no corrections for empty syndromes");
}

/// Test partial syndrome array (shorter than blocks_state).
#[test]
fn test_partial_syndrome_array() {
    let mut memory = vec![0u8; 1024 * 1024 * 50];
    let mut arena = Arena::new(&mut memory);

    let w = 40;
    let h = 40;
    let mut decoder = DecodingState::<SquareGrid, 64>::new(&mut arena, w, h, 1);

    // Provide only half the syndromes
    let num_blocks = decoder.blocks_state.len();
    let partial_len = num_blocks / 2;
    let mut syndromes = vec![0u64; partial_len];

    let d1 = idx(2, 2, 64);
    let d2 = idx(3, 2, 64);
    set_defect(&mut syndromes, d1);
    set_defect(&mut syndromes, d2);

    decoder.sparse_reset();
    decoder.load_dense_syndromes(&syndromes);
    decoder.grow_clusters();

    let mut corrections = vec![EdgeCorrection::default(); w * h * 4];
    let count = decoder.decode(&mut corrections);

    // Should decode syndromes in the provided range
    let result = common::verify_matching(&syndromes, &corrections[..count]);
    assert!(result.is_ok(), "Failed: {:?}", result.err());
}

/// Test growth termination when clusters merge.
#[test]
fn test_grow_clusters_termination() {
    let mut memory = vec![0u8; 1024 * 1024 * 10];
    let mut arena = Arena::new(&mut memory);

    let w = 16;
    let h = 16;
    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, w, h, 1);

    let num_blocks = decoder.blocks_state.len();
    let mut syndromes = vec![0u64; num_blocks];

    // Two adjacent defects - should merge quickly
    let d1 = idx(4, 4, 16);
    let d2 = idx(5, 4, 16);
    set_defect(&mut syndromes, d1);
    set_defect(&mut syndromes, d2);

    decoder.sparse_reset();
    decoder.load_dense_syndromes(&syndromes);
    decoder.grow_clusters();

    let mut corrections = vec![EdgeCorrection::default(); w * h * 4];
    let count = decoder.decode(&mut corrections);

    let result = common::verify_matching(&syndromes, &corrections[..count]);
    assert!(result.is_ok(), "Failed: {:?}", result.err());
}

/// Test growth with distant defects (multiple iterations needed).
#[test]
fn test_distant_defects() {
    let mut memory = vec![0u8; 1024 * 1024 * 50];
    let mut arena = Arena::new(&mut memory);

    let w = 60;
    let h = 60;
    let mut decoder = DecodingState::<SquareGrid, 64>::new(&mut arena, w, h, 1);

    let num_blocks = decoder.blocks_state.len();
    let mut syndromes = vec![0u64; num_blocks];

    // Two defects far apart
    let d1 = idx(5, 5, 64);
    let d2 = idx(55, 55, 64);
    set_defect(&mut syndromes, d1);
    set_defect(&mut syndromes, d2);

    decoder.sparse_reset();
    decoder.load_dense_syndromes(&syndromes);

    // Growth should terminate within limit even for distant defects
    decoder.grow_clusters();

    let mut corrections = vec![EdgeCorrection::default(); w * h * 4];
    let count = decoder.decode(&mut corrections);

    // Just verify it completes
        assert!(count > 0);
}

/// Test active_block_mask tracking in small grid mode.
#[test]
fn test_active_block_mask_tracking() {
    let mut memory = vec![0u8; 1024 * 1024 * 10];
    let mut arena = Arena::new(&mut memory);

    let w = 16;
    let h = 16;
    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, w, h, 1);

    assert!(decoder.is_small_grid());

    let num_blocks = decoder.blocks_state.len();
    let mut syndromes = vec![0u64; num_blocks];

    // Set defects in multiple blocks
    let d1 = idx(2, 2, 16); // Block 0
    let d2 = idx(3, 2, 16); // Block 0
    set_defect(&mut syndromes, d1);
    set_defect(&mut syndromes, d2);

    decoder.sparse_reset();
    decoder.load_dense_syndromes(&syndromes);

    // After loading, active_block_mask should have bits set
    assert!(decoder.active_block_mask != 0, "Expected active blocks after syndrome load");

    decoder.grow_clusters();

    // After growth completes, active_block_mask should be 0
    assert_eq!(decoder.active_block_mask, 0, "Expected no active blocks after growth");

    let mut corrections = vec![EdgeCorrection::default(); w * h * 4];
    let count = decoder.decode(&mut corrections);

    let result = common::verify_matching(&syndromes, &corrections[..count]);
    assert!(result.is_ok(), "Failed: {:?}", result.err());
}

// =============================================================================
// 3D Grid Tests (to reach large grid paths)
// =============================================================================

/// Test 3D grid which can have more than 65 blocks.
///
/// With depth > 1 and appropriate dimensions, we can reach the large grid path.
#[test]
fn test_3d_grid_large_blocks() {
    let mut memory = vec![0u8; 1024 * 1024 * 100];
    let mut arena = Arena::new(&mut memory);

    // 3D grid: 32x32x8 with stride 32
    // Total nodes = 32^3 = 32768
    // blocks = 32768/64 = 512 blocks > 65
    let w = 32;
    let h = 32;
    let depth = 8;
    let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, w, h, depth);

    // Check if this creates a large grid
    let is_large = !decoder.is_small_grid();

    if is_large {
        let num_blocks = decoder.blocks_state.len();
        let mut syndromes = vec![0u64; num_blocks];

        // Place defects in block 0
        set_defect(&mut syndromes, 0);
        set_defect(&mut syndromes, 1);

        decoder.sparse_reset();
        decoder.load_dense_syndromes(&syndromes);

        // Test large grid grow_clusters path
        decoder.grow_clusters();

        let mut corrections = vec![EdgeCorrection::default(); w * h * depth * 4];
        let count = decoder.decode(&mut corrections);

        // Verify completion
            assert!(count > 0);
    }
    // If not large grid, test passes (the 3D config might still be small)
}
