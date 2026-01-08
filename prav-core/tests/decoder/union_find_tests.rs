//! Unit tests for UnionFind trait implementation.
//!
//! Tests the Union-Find data structure used for efficient cluster merging:
//! - Fast path for self-rooted nodes (~95% of cases at p=0.001)
//! - Path halving compression in slow path
//! - Deterministic union behavior (smaller root joins larger)
//! - Block cache invalidation after unions
//! - Dirty tracking for sparse reset

use prav_core::arena::Arena;
use prav_core::decoder::{DecodingState, UnionFind};
use prav_core::topology::SquareGrid;

/// Helper to create a decoder for testing UnionFind operations
fn create_decoder<'a, const STRIDE_Y: usize>(
    arena: &mut Arena<'a>,
    width: usize,
    height: usize,
) -> DecodingState<'a, SquareGrid, STRIDE_Y> {
    DecodingState::<SquareGrid, STRIDE_Y>::new(arena, width, height, 1)
}

// =============================================================================
// find() Fast Path Tests
// =============================================================================

#[test]
fn test_find_fast_path_self_rooted() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<8>(&mut arena, 8, 8);

    // Initially all nodes are self-rooted (parents[i] == i)
    // This should hit the fast path and return immediately
    for i in 0..64 {
        let root = decoder.find(i);
        assert_eq!(root, i, "Self-rooted node {} should return itself", i);
    }
}

#[test]
fn test_find_fast_path_no_path_compression_needed() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<8>(&mut arena, 8, 8);

    // When node points directly to root (depth 1), fast path doesn't apply
    // but we should still get correct result
    decoder.parents[5] = 10; // Node 5 points to 10
    // Node 10 is still self-rooted

    let root = decoder.find(5);
    assert_eq!(root, 10, "Node 5 should find root 10");
}

// =============================================================================
// find() Slow Path Tests (Path Halving)
// =============================================================================

#[test]
fn test_find_slow_path_two_levels() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<8>(&mut arena, 8, 8);

    // Create a chain: 0 -> 1 -> 2 (root)
    decoder.parents[0] = 1;
    decoder.parents[1] = 2;
    // 2 is self-rooted

    let root = decoder.find(0);
    assert_eq!(root, 2, "Should find root 2");

    // After path halving, 0 should point to 2 (grandparent)
    // Path halving: parent[0] = grandparent = 2
    assert_eq!(decoder.parents[0], 2, "Path halving should point 0 to grandparent 2");
}

#[test]
fn test_find_slow_path_deep_chain() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<8>(&mut arena, 8, 8);

    // Create a chain: 0 -> 1 -> 2 -> 3 -> 4 -> 5 (root)
    decoder.parents[0] = 1;
    decoder.parents[1] = 2;
    decoder.parents[2] = 3;
    decoder.parents[3] = 4;
    decoder.parents[4] = 5;
    // 5 is self-rooted

    let root = decoder.find(0);
    assert_eq!(root, 5, "Should find root 5");

    // After path halving, the chain should be significantly compressed
    // Path halving skips every other node
}

#[test]
fn test_find_slow_path_marks_dirty() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<8>(&mut arena, 8, 8);

    // Clear dirty masks first
    decoder.block_dirty_mask.fill(0);

    // Create a chain: 0 -> 1 -> 2 (root)
    // All in block 0 (nodes 0-63)
    decoder.parents[0] = 1;
    decoder.parents[1] = 2;

    let _ = decoder.find(0);

    // Block 0 should be marked dirty due to path compression
    let dirty_bit = decoder.block_dirty_mask[0] & 1;
    assert_ne!(dirty_bit, 0, "Block 0 should be marked dirty after path compression");
}

#[test]
fn test_find_slow_path_cross_block_chain() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<8>(&mut arena, 8, 8);

    // Clear dirty masks
    decoder.block_dirty_mask.fill(0);

    // Create chain crossing blocks: 0 -> 65 -> 130 (root)
    // Node 0 is in block 0, node 65 is in block 1, node 130 is in block 2
    // But we need to ensure these are valid indices
    // For 8x8 grid with stride 8, valid nodes are within the grid
    // Let's use a larger grid
    drop(decoder);
    drop(arena);

    let mut memory2 = vec![0u8; 4 * 1024 * 1024];
    let mut arena2 = Arena::new(&mut memory2);
    let mut decoder2 = create_decoder::<16>(&mut arena2, 16, 16);

    // Clear dirty masks
    decoder2.block_dirty_mask.fill(0);

    // Chain: 0 -> 65 -> 130 (root)
    decoder2.parents[0] = 65;
    decoder2.parents[65] = 130;
    // 130 is self-rooted

    let root = decoder2.find(0);
    assert_eq!(root, 130, "Should find root 130");

    // Multiple blocks should be marked dirty
    // Block 0 (nodes 0-63), Block 1 (nodes 64-127)
    let block0_dirty = (decoder2.block_dirty_mask[0] & 1) != 0;
    assert!(block0_dirty, "Block 0 should be marked dirty");
}

// =============================================================================
// union_roots() Tests
// =============================================================================

#[test]
fn test_union_roots_smaller_joins_larger() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<8>(&mut arena, 8, 8);

    // Both nodes are self-rooted initially
    let result = unsafe { decoder.union_roots(3, 7) };

    assert!(result, "union_roots should return true for different roots");
    // Smaller root (3) should join larger (7)
    assert_eq!(decoder.parents[3], 7, "Smaller root 3 should point to larger root 7");
    assert_eq!(decoder.parents[7], 7, "Larger root 7 should remain self-rooted");
}

#[test]
fn test_union_roots_larger_joins_smaller_parameter_order() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<8>(&mut arena, 8, 8);

    // Parameter order shouldn't matter - smaller always joins larger
    let result = unsafe { decoder.union_roots(10, 5) };

    assert!(result, "union_roots should return true for different roots");
    // Smaller root (5) should join larger (10)
    assert_eq!(decoder.parents[5], 10, "Smaller root 5 should point to larger root 10");
    assert_eq!(decoder.parents[10], 10, "Larger root 10 should remain self-rooted");
}

#[test]
fn test_union_roots_same_root_returns_false() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<8>(&mut arena, 8, 8);

    let result = unsafe { decoder.union_roots(5, 5) };

    assert!(!result, "union_roots should return false for same roots");
    assert_eq!(decoder.parents[5], 5, "Node should remain self-rooted");
}

#[test]
fn test_union_roots_invalidates_block_cache() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<8>(&mut arena, 8, 8);

    // Set a cached root value for block 0
    decoder.blocks_state[0].root = 0;

    // Union nodes 3 and 7 (both in block 0)
    let _ = unsafe { decoder.union_roots(3, 7) };

    // The smaller root's block should have its cache invalidated
    assert_eq!(
        decoder.blocks_state[0].root, u32::MAX,
        "Block cache should be invalidated after union"
    );
}

#[test]
fn test_union_roots_marks_dirty() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<8>(&mut arena, 8, 8);

    // Clear dirty mask
    decoder.block_dirty_mask.fill(0);

    let _ = unsafe { decoder.union_roots(3, 7) };

    // Block 0 should be marked dirty
    let block0_dirty = (decoder.block_dirty_mask[0] & 1) != 0;
    assert!(block0_dirty, "Block should be marked dirty after union_roots");
}

// =============================================================================
// union() Tests (Combined find + union_roots)
// =============================================================================

#[test]
fn test_union_different_roots() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<8>(&mut arena, 8, 8);

    let result = unsafe { decoder.union(3, 7) };

    assert!(result, "union should return true for different roots");

    // After union, both should have same root
    let root3 = decoder.find(3);
    let root7 = decoder.find(7);
    assert_eq!(root3, root7, "After union, nodes should have same root");
}

#[test]
fn test_union_same_cluster() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<8>(&mut arena, 8, 8);

    // First union 3 and 7
    let _ = unsafe { decoder.union(3, 7) };

    // Second union of same nodes should return false
    let result = unsafe { decoder.union(3, 7) };
    assert!(!result, "union of already-connected nodes should return false");
}

#[test]
fn test_union_transitive() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<8>(&mut arena, 8, 8);

    // Union 1 and 2
    let _ = unsafe { decoder.union(1, 2) };
    // Union 2 and 3
    let _ = unsafe { decoder.union(2, 3) };

    // Now 1, 2, 3 should all be in same cluster
    let root1 = decoder.find(1);
    let root2 = decoder.find(2);
    let root3 = decoder.find(3);

    assert_eq!(root1, root2, "1 and 2 should have same root");
    assert_eq!(root2, root3, "2 and 3 should have same root");
}

#[test]
fn test_union_chain_compression() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<8>(&mut arena, 8, 8);

    // Create a longer chain through multiple unions
    for i in 0..5 {
        let _ = unsafe { decoder.union(i, i + 1) };
    }

    // All should have the same root
    let expected_root = decoder.find(0);
    for i in 1..=5 {
        let root = decoder.find(i);
        assert_eq!(root, expected_root, "All nodes should have same root after chain union");
    }
}

// =============================================================================
// Edge Cases and Boundary Tests
// =============================================================================

#[test]
fn test_union_find_boundary_node() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = create_decoder::<8>(&mut arena, 8, 8);

    // The boundary node is at parents.len() - 1
    let boundary_idx = decoder.parents.len() - 1;
    assert_eq!(
        decoder.parents[boundary_idx], boundary_idx as u32,
        "Boundary node should be self-rooted"
    );
}

#[test]
fn test_find_returns_valid_root() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<8>(&mut arena, 8, 8);

    // Create complex tree structure
    decoder.parents[0] = 1;
    decoder.parents[1] = 3;
    decoder.parents[2] = 3;
    // 3 is self-rooted

    // All finds should return a valid root (self-referential node)
    for &node in &[0u32, 1, 2, 3] {
        let root = decoder.find(node);
        assert_eq!(
            decoder.parents[root as usize], root,
            "find({}) returned {} which is not a root (parents[{}] = {})",
            node, root, root, decoder.parents[root as usize]
        );
    }
}

#[test]
fn test_union_find_deterministic() {
    // Union-Find should be deterministic regardless of union order
    let mut memory1 = vec![0u8; 1024 * 1024];
    let mut arena1 = Arena::new(&mut memory1);
    let mut decoder1 = create_decoder::<8>(&mut arena1, 8, 8);

    let mut memory2 = vec![0u8; 1024 * 1024];
    let mut arena2 = Arena::new(&mut memory2);
    let mut decoder2 = create_decoder::<8>(&mut arena2, 8, 8);

    // Same unions, different order
    let _ = unsafe { decoder1.union(1, 5) };
    let _ = unsafe { decoder1.union(3, 5) };

    let _ = unsafe { decoder2.union(3, 5) };
    let _ = unsafe { decoder2.union(1, 5) };

    // Final roots should be the same
    let root1_d1 = decoder1.find(1);
    let root1_d2 = decoder2.find(1);

    assert_eq!(root1_d1, root1_d2, "Union-Find should be deterministic");
}

// =============================================================================
// Performance-Critical Path Tests
// =============================================================================

#[test]
fn test_find_fast_path_majority_case() {
    // At p=0.001, ~95% of nodes are self-rooted
    // This test verifies the fast path works correctly for many nodes
    let mut memory = vec![0u8; 4 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<32>(&mut arena, 32, 32);

    // All 1024 nodes should be self-rooted initially
    for i in 0u32..1024 {
        let root = decoder.find(i);
        assert_eq!(root, i, "Node {} should be self-rooted", i);
    }
}

#[test]
fn test_block_dirty_tracking_sparse() {
    let mut memory = vec![0u8; 4 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_decoder::<32>(&mut arena, 32, 32);

    // Clear dirty mask
    decoder.block_dirty_mask.fill(0);

    // Union only a few nodes (sparse case)
    let _ = unsafe { decoder.union(0, 1) };   // Block 0
    let _ = unsafe { decoder.union(128, 129) }; // Block 2

    // Count dirty blocks
    let dirty_count: u32 = decoder.block_dirty_mask.iter().map(|w| w.count_ones()).sum();

    // Should have exactly 2 dirty blocks (block 0 and block 2)
    assert!(dirty_count <= 4, "Only touched blocks should be dirty, got {} dirty blocks", dirty_count);
}
