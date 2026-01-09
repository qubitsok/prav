//! Comprehensive tests for the sparse syndrome decoder module.
//!
//! Tests cover:
//! - SparseState data structure operations
//! - Defect extraction from dense syndromes
//! - Early termination with `all_defects_resolved`
//! - Full sparse decode path
//! - Two-defect fast path

use prav_core::arena::Arena;
use prav_core::decoder::sparse::{
    SPARSE_THRESHOLD, SparseState, all_defects_resolved, decode_sparse, decode_two_defects,
    extract_defects,
};
use prav_core::decoder::state::DecodingState;
use prav_core::decoder::types::EdgeCorrection;
use prav_core::topology::SquareGrid;

// ============================================================================
// SparseState Tests
// ============================================================================

#[test]
fn test_sparse_state_default() {
    let state = SparseState::default();
    assert!(state.is_empty());
    assert_eq!(state.len(), 0);
    assert_eq!(state.defects(), &[]);
}

#[test]
fn test_sparse_state_clear() {
    let mut state = SparseState::new();
    assert!(state.push(1));
    assert!(state.push(2));
    assert!(state.push(3));
    assert_eq!(state.len(), 3);

    state.clear();
    assert!(state.is_empty());
    assert_eq!(state.len(), 0);
}

#[test]
fn test_sparse_state_capacity_limit() {
    let mut state = SparseState::new();

    // Fill to capacity
    for i in 0..SPARSE_THRESHOLD {
        assert!(state.push(i as u32), "Should accept defect {}", i);
    }
    assert_eq!(state.len(), SPARSE_THRESHOLD);

    // Should reject when full
    assert!(!state.push(999));
    assert_eq!(state.len(), SPARSE_THRESHOLD);
}

#[test]
fn test_sparse_state_defects_slice() {
    let mut state = SparseState::new();
    state.push(10);
    state.push(20);
    state.push(30);

    let defects = state.defects();
    assert_eq!(defects.len(), 3);
    assert_eq!(defects[0], 10);
    assert_eq!(defects[1], 20);
    assert_eq!(defects[2], 30);
}

// ============================================================================
// extract_defects Tests
// ============================================================================

#[test]
fn test_extract_defects_single_word() {
    let syndromes = [0b1010_0101u64; 1]; // Bits 0, 2, 5, 7 set
    let result = extract_defects(&syndromes, 8, 64);
    assert!(result.is_some());
    let state = result.unwrap();
    assert_eq!(state.len(), 4);
    assert_eq!(state.defects(), &[0, 2, 5, 7]);
}

#[test]
fn test_extract_defects_multiple_words() {
    let mut syndromes = [0u64; 4];
    syndromes[0] = 1; // Defect at 0
    syndromes[1] = 1 << 32; // Defect at 64 + 32 = 96
    syndromes[3] = 1 << 63; // Defect at 192 + 63 = 255

    let result = extract_defects(&syndromes, 16, 256);
    assert!(result.is_some());
    let state = result.unwrap();
    assert_eq!(state.len(), 3);
    assert_eq!(state.defects(), &[0, 96, 255]);
}

#[test]
fn test_extract_defects_exactly_threshold() {
    // Create exactly SPARSE_THRESHOLD defects
    let words_needed = SPARSE_THRESHOLD.div_ceil(64);
    let mut syndromes = vec![0u64; words_needed];

    for i in 0..SPARSE_THRESHOLD {
        let word_idx = i / 64;
        let bit_idx = i % 64;
        syndromes[word_idx] |= 1 << bit_idx;
    }

    let result = extract_defects(&syndromes, 16, 256);
    assert!(result.is_some());
    assert_eq!(result.unwrap().len(), SPARSE_THRESHOLD);
}

#[test]
fn test_extract_defects_threshold_plus_one() {
    // Create SPARSE_THRESHOLD + 1 defects - should fail
    let words_needed = (SPARSE_THRESHOLD + 1).div_ceil(64);
    let mut syndromes = vec![0u64; words_needed];

    for i in 0..=SPARSE_THRESHOLD {
        let word_idx = i / 64;
        let bit_idx = i % 64;
        syndromes[word_idx] |= 1 << bit_idx;
    }

    let result = extract_defects(&syndromes, 16, 256);
    assert!(result.is_none());
}

// ============================================================================
// all_defects_resolved Tests
// ============================================================================

#[test]
fn test_all_defects_resolved_empty() {
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut buffer);
    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

    let sparse = SparseState::new();
    let boundary_node = (decoder.parents.len() - 1) as u32;

    assert!(all_defects_resolved(&mut decoder, &sparse, boundary_node));
}

#[test]
fn test_all_defects_resolved_single_defect_not_at_boundary() {
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut buffer);
    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

    let mut sparse = SparseState::new();
    sparse.push(10); // Single defect in middle of grid

    let boundary_node = (decoder.parents.len() - 1) as u32;

    // Odd number of defects not connected to boundary
    assert!(!all_defects_resolved(&mut decoder, &sparse, boundary_node));
}

#[test]
fn test_all_defects_resolved_single_defect_at_boundary() {
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut buffer);
    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

    let mut sparse = SparseState::new();
    sparse.push(10);

    let boundary_node = (decoder.parents.len() - 1) as u32;

    // Manually union the defect with boundary
    unsafe {
        decoder.union(10, boundary_node);
    }

    assert!(all_defects_resolved(&mut decoder, &sparse, boundary_node));
}

#[test]
fn test_all_defects_resolved_two_defects_same_cluster() {
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut buffer);
    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

    let mut sparse = SparseState::new();
    sparse.push(10);
    sparse.push(20);

    // Union them together
    unsafe {
        decoder.union(10, 20);
    }

    let boundary_node = (decoder.parents.len() - 1) as u32;

    // Even number of defects in same cluster -> resolved
    assert!(all_defects_resolved(&mut decoder, &sparse, boundary_node));
}

#[test]
fn test_all_defects_resolved_two_defects_different_clusters() {
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut buffer);
    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

    let mut sparse = SparseState::new();
    sparse.push(10);
    sparse.push(20);

    // Don't union them - they're in different clusters
    let boundary_node = (decoder.parents.len() - 1) as u32;

    assert!(!all_defects_resolved(&mut decoder, &sparse, boundary_node));
}

#[test]
fn test_all_defects_resolved_three_defects_need_boundary() {
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut buffer);
    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

    let mut sparse = SparseState::new();
    sparse.push(10);
    sparse.push(20);
    sparse.push(30);

    let boundary_node = (decoder.parents.len() - 1) as u32;

    // Union all three
    unsafe {
        decoder.union(10, 20);
        decoder.union(20, 30);
    }

    // Odd number of defects, same cluster, but NOT connected to boundary
    assert!(!all_defects_resolved(&mut decoder, &sparse, boundary_node));

    // Now connect to boundary
    unsafe {
        decoder.union(10, boundary_node);
    }

    // Should be resolved now
    assert!(all_defects_resolved(&mut decoder, &sparse, boundary_node));
}

// ============================================================================
// decode_sparse Tests
// ============================================================================

#[test]
fn test_decode_sparse_no_defects() {
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut buffer);
    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

    let syndromes = vec![0u64; decoder.blocks_state.len()];
    let mut corrections = vec![EdgeCorrection::default(); 100];

    let result = decode_sparse(&mut decoder, &syndromes, &mut corrections);
    assert_eq!(result, Some(0));
}

#[test]
fn test_decode_sparse_single_pair() {
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut buffer);
    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

    // Two adjacent defects at positions 0 and 1
    let mut syndromes = vec![0u64; decoder.blocks_state.len()];
    syndromes[0] = 0b11; // Bits 0 and 1

    let mut corrections = vec![EdgeCorrection::default(); 100];

    let result = decode_sparse(&mut decoder, &syndromes, &mut corrections);
    assert!(result.is_some());
    // Should produce at least one correction
    let count = result.unwrap();
    assert!(count >= 1, "Expected at least one correction for pair");
}

#[test]
fn test_decode_sparse_too_many_defects() {
    let mut buffer = vec![0u8; 2 * 1024 * 1024];
    let mut arena = Arena::new(&mut buffer);
    let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

    // Create more than SPARSE_THRESHOLD defects
    let mut syndromes = vec![!0u64; decoder.blocks_state.len()]; // All bits set
    syndromes.truncate(1); // But only in first word = 64 defects > 32 threshold

    // Actually let's set exactly the threshold+1 defects spread across words
    syndromes = vec![0u64; decoder.blocks_state.len()];
    for i in 0..=SPARSE_THRESHOLD {
        let word_idx = i / 64;
        let bit_idx = i % 64;
        if word_idx < syndromes.len() {
            syndromes[word_idx] |= 1 << bit_idx;
        }
    }

    let mut corrections = vec![EdgeCorrection::default(); 100];

    let result = decode_sparse(&mut decoder, &syndromes, &mut corrections);
    assert!(result.is_none(), "Should fail when too many defects");
}

#[test]
fn test_decode_sparse_distant_pair() {
    let mut buffer = vec![0u8; 2 * 1024 * 1024];
    let mut arena = Arena::new(&mut buffer);
    let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

    // Two defects far apart - should still work
    let mut syndromes = vec![0u64; decoder.blocks_state.len()];
    syndromes[0] = 1; // Position 0
    if syndromes.len() > 8 {
        syndromes[8] = 1; // Position 512 (8*64)
    }

    let mut corrections = vec![EdgeCorrection::default(); 200];

    let result = decode_sparse(&mut decoder, &syndromes, &mut corrections);
    assert!(result.is_some());
}

// ============================================================================
// decode_two_defects Tests
// ============================================================================

#[test]
fn test_decode_two_defects_adjacent() {
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut buffer);
    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

    let mut corrections = vec![EdgeCorrection::default(); 100];

    // Two adjacent defects at positions 0 and 1
    let count = decode_two_defects(&mut decoder, 0, 1, &mut corrections);

    // Should produce corrections connecting them
    assert!(count >= 1, "Expected corrections for adjacent pair");
}

#[test]
fn test_decode_two_defects_same_block() {
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut buffer);
    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

    let mut corrections = vec![EdgeCorrection::default(); 100];

    // Two defects in the same block (0-63), positions 0 and 10
    let count = decode_two_defects(&mut decoder, 0, 10, &mut corrections);

    assert!(count >= 1);
}

#[test]
fn test_decode_two_defects_different_blocks() {
    let mut buffer = vec![0u8; 2 * 1024 * 1024];
    let mut arena = Arena::new(&mut buffer);
    let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

    let mut corrections = vec![EdgeCorrection::default(); 200];

    // Defects in different blocks: block 0 (node 0) and block 2 (node 128)
    let count = decode_two_defects(&mut decoder, 0, 128, &mut corrections);

    // Should produce corrections connecting them across blocks
    assert!(count >= 1);
}

#[test]
fn test_decode_two_defects_same_node_degenerate() {
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut buffer);
    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

    let mut corrections = vec![EdgeCorrection::default(); 100];

    // Edge case: same node twice (degenerate case)
    let count = decode_two_defects(&mut decoder, 5, 5, &mut corrections);

    // Should handle gracefully (no crash)
    // Result may be 0 corrections since they trivially cancel
    assert!(count <= 1);
}
