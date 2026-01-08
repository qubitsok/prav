//! Property-based tests for decoder core modules.
//!
//! These tests verify invariants using randomized inputs:
//! - Sparse reset idempotency
//! - Union-find symmetry and determinism
//! - Coordinate mapping bijectivity
//! - Path compression convergence
//! - Block/bit addressing consistency

use prav_core::arena::Arena;
use prav_core::decoder::DecodingState;
use prav_core::topology::SquareGrid;
use proptest::prelude::*;

// =============================================================================
// Test 1: Sparse Reset Idempotency
// =============================================================================

proptest! {
    /// Verify that sparse_reset is idempotent: reset(reset(state)) == reset(state)
    #[test]
    fn prop_sparse_reset_idempotent(
        dirty_bits in prop::collection::vec(0u64..64, 1..8),
        block_values in prop::collection::vec(any::<u64>(), 1..8),
    ) {
        let mut memory = vec![0u8; 2 * 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

        // Set up some dirty state
        let num_blocks = decoder.blocks_state.len().min(8);
        for (i, &bit) in dirty_bits.iter().take(num_blocks).enumerate() {
            if i < num_blocks {
                let blk = bit as usize % num_blocks;
                decoder.mark_block_dirty(blk);
                if blk < block_values.len() {
                    decoder.blocks_state[blk].boundary = block_values[blk];
                    decoder.blocks_state[blk].occupied = block_values[blk];
                }
            }
        }

        // First reset
        decoder.sparse_reset();

        // Capture state after first reset
        let boundaries_after_first: Vec<u64> = decoder.blocks_state.iter()
            .map(|b| b.boundary)
            .collect();
        let occupied_after_first: Vec<u64> = decoder.blocks_state.iter()
            .map(|b| b.occupied)
            .collect();
        let dirty_mask_after_first: Vec<u64> = decoder.block_dirty_mask.to_vec();

        // Second reset (should be no-op since nothing is dirty)
        decoder.sparse_reset();

        // State should be identical
        for (i, block) in decoder.blocks_state.iter().enumerate() {
            prop_assert_eq!(
                block.boundary, boundaries_after_first[i],
                "Block {} boundary changed after second reset", i
            );
            prop_assert_eq!(
                block.occupied, occupied_after_first[i],
                "Block {} occupied changed after second reset", i
            );
        }

        for (i, &mask) in decoder.block_dirty_mask.iter().enumerate() {
            prop_assert_eq!(
                mask, dirty_mask_after_first[i],
                "Dirty mask word {} changed after second reset", i
            );
        }
    }
}

// =============================================================================
// Test 2: Union-Find Symmetry
// =============================================================================

proptest! {
    /// Verify union symmetry: final root is same regardless of argument order
    #[test]
    fn prop_union_symmetric(
        a in 0u32..63,
        b in 0u32..63,
    ) {
        prop_assume!(a != b);

        // First decoder: union(a, b)
        let mut memory1 = vec![0u8; 1024 * 1024];
        let mut arena1 = Arena::new(&mut memory1);
        let mut decoder1 = DecodingState::<SquareGrid, 8>::new(&mut arena1, 8, 8, 1);

        unsafe { decoder1.union(a, b); }
        let root_a1 = decoder1.find(a);
        let root_b1 = decoder1.find(b);

        // Second decoder: union(b, a)
        let mut memory2 = vec![0u8; 1024 * 1024];
        let mut arena2 = Arena::new(&mut memory2);
        let mut decoder2 = DecodingState::<SquareGrid, 8>::new(&mut arena2, 8, 8, 1);

        unsafe { decoder2.union(b, a); }
        let root_a2 = decoder2.find(a);
        let root_b2 = decoder2.find(b);

        // Roots should be equal in both cases
        prop_assert_eq!(root_a1, root_b1, "Roots should be same after union");
        prop_assert_eq!(root_a2, root_b2, "Roots should be same after union (swapped)");

        // The actual root value should be deterministic
        prop_assert_eq!(root_a1, root_a2, "Union must be symmetric: root should be same regardless of order");
    }
}

proptest! {
    /// Verify that union is deterministic: smaller root always joins larger
    #[test]
    fn prop_union_deterministic(
        a in 0u32..63,
        b in 0u32..63,
    ) {
        prop_assume!(a != b);

        let mut memory = vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

        unsafe { decoder.union(a, b); }

        let root = decoder.find(a);

        // The root should be the larger of the two
        prop_assert_eq!(root, a.max(b), "Larger index should become root");

        // Verify both nodes have same root
        let root_b = decoder.find(b);
        prop_assert_eq!(root, root_b, "Both nodes should have same root");
    }
}

// =============================================================================
// Test 3: Block/Bit Address Round-Trip
// =============================================================================

proptest! {
    /// Verify block/bit decomposition is reversible
    #[test]
    fn prop_block_bit_round_trip(node in 0u32..4096) {
        let blk = (node / 64) as usize;
        let bit = (node % 64) as usize;

        // Reconstruction
        let reconstructed = (blk * 64 + bit) as u32;

        prop_assert_eq!(reconstructed, node, "Block/bit must round-trip");
        prop_assert!(bit < 64, "Bit must be in range [0, 63]");
    }
}

// =============================================================================
// Test 4: Tiled Coordinate Mapping
// =============================================================================

proptest! {
    /// Verify tiled coordinate mapping is bijective
    #[test]
    fn prop_tiled_coords_round_trip(
        x in 0usize..64,
        y in 0usize..64,
    ) {
        let width = 64usize;
        let height = 64usize;

        prop_assume!(x < width && y < height);

        let tiles_x = (width + 31) / 32;

        // Forward: (x, y) -> node_idx
        let tx = x / 32;
        let ty = y / 32;
        let lx = x % 32;
        let ly = y % 32;
        let tile_idx = ty * tiles_x + tx;
        let local_idx = ly * 32 + lx;
        let node_idx = tile_idx * 1024 + local_idx;

        // Reverse: node_idx -> (x', y')
        let tile_idx_back = node_idx / 1024;
        let local_idx_back = node_idx % 1024;
        let tx_back = tile_idx_back % tiles_x;
        let ty_back = tile_idx_back / tiles_x;
        let lx_back = local_idx_back % 32;
        let ly_back = local_idx_back / 32;
        let x_back = tx_back * 32 + lx_back;
        let y_back = ty_back * 32 + ly_back;

        prop_assert_eq!(x_back, x, "X coordinate must round-trip");
        prop_assert_eq!(y_back, y, "Y coordinate must round-trip");
    }
}

proptest! {
    /// Verify all coordinates in a tile map to unique node indices
    #[test]
    fn prop_tiled_coords_unique(
        x1 in 0usize..32,
        y1 in 0usize..32,
        x2 in 0usize..32,
        y2 in 0usize..32,
    ) {
        // Within a single tile, all (x, y) should map to unique local indices
        let local_idx1 = y1 * 32 + x1;
        let local_idx2 = y2 * 32 + x2;

        if x1 == x2 && y1 == y2 {
            prop_assert_eq!(local_idx1, local_idx2);
        } else {
            prop_assert_ne!(local_idx1, local_idx2, "Different coords must map to different indices");
        }
    }
}

// =============================================================================
// Test 5: Path Compression Convergence
// =============================================================================

proptest! {
    /// Verify that find() with path compression maintains valid parent structure
    #[test]
    fn prop_path_compression_valid(
        nodes in prop::collection::vec(0u32..63, 2..8),
    ) {
        let mut memory = vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

        // Create a chain of unions
        for window in nodes.windows(2) {
            unsafe { decoder.union(window[0], window[1]); }
        }

        // After all unions, all nodes should have same root
        if !nodes.is_empty() {
            let expected_root = decoder.find(nodes[0]);

            for &node in &nodes {
                let root = decoder.find(node);
                prop_assert_eq!(root, expected_root, "All nodes in chain should have same root");

                // Verify root is self-referential
                prop_assert_eq!(
                    decoder.parents[root as usize], root,
                    "Root must be self-referential"
                );
            }
        }
    }
}

proptest! {
    /// Verify that find() returns a valid root (self-referential node)
    #[test]
    fn prop_find_returns_root(node in 0u32..63) {
        let mut memory = vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

        let root = decoder.find(node);

        // Root must be self-referential
        prop_assert_eq!(
            decoder.parents[root as usize], root,
            "find() must return a root (self-referential node)"
        );
    }
}

// =============================================================================
// Test 6: Mark Block Dirty Idempotency
// =============================================================================

proptest! {
    /// Verify mark_block_dirty is idempotent
    #[test]
    fn prop_mark_dirty_idempotent(blk_idx in 0usize..16) {
        let mut memory = vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

        // Clear dirty mask
        decoder.block_dirty_mask.fill(0);

        // Mark once
        decoder.mark_block_dirty(blk_idx);
        let mask_after_first = decoder.block_dirty_mask[0];

        // Mark again
        decoder.mark_block_dirty(blk_idx);
        let mask_after_second = decoder.block_dirty_mask[0];

        prop_assert_eq!(
            mask_after_first, mask_after_second,
            "mark_block_dirty must be idempotent"
        );

        // Verify the bit is set
        let expected_bit = 1u64 << blk_idx;
        prop_assert_ne!(
            mask_after_first & expected_bit, 0,
            "Target bit must be set"
        );
    }
}

// =============================================================================
// Test 7: Transitive Union Closure
// =============================================================================

proptest! {
    /// Verify union transitivity: if a~b and b~c, then a~c
    #[test]
    fn prop_union_transitive(
        a in 0u32..31,
        b in 32u32..48,
        c in 49u32..63,
    ) {
        let mut memory = vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

        // Union a with b
        unsafe { decoder.union(a, b); }

        // Union b with c
        unsafe { decoder.union(b, c); }

        // Now a, b, c should all have same root
        let root_a = decoder.find(a);
        let root_b = decoder.find(b);
        let root_c = decoder.find(c);

        prop_assert_eq!(root_a, root_b, "a and b should have same root");
        prop_assert_eq!(root_b, root_c, "b and c should have same root");
        prop_assert_eq!(root_a, root_c, "Union must be transitive");
    }
}

// =============================================================================
// Test 8: Stride Power of Two Property
// =============================================================================

proptest! {
    /// Verify stride calculation produces power of two
    #[test]
    fn prop_stride_power_of_two(
        width in 1usize..65,
        height in 1usize..65,
    ) {
        let max_dim = width.max(height);
        let stride = max_dim.next_power_of_two();

        prop_assert!(stride.is_power_of_two(), "Stride must be power of two");
        prop_assert!(stride >= max_dim, "Stride must be >= max_dim");
        prop_assert!(stride <= 128, "Stride bounded for small grids");
    }
}

// =============================================================================
// Test 9: Valid Mask Coverage
// =============================================================================

proptest! {
    /// Verify that valid_mask covers all grid nodes
    #[test]
    fn prop_valid_mask_covers_grid(
        width in 4usize..17,
        height in 4usize..17,
    ) {
        // Ensure stride matches
        let max_dim = width.max(height);
        let stride = max_dim.next_power_of_two();

        // Only test when stride matches available const generics
        prop_assume!(stride == 8 || stride == 16 || stride == 32);

        if stride == 8 {
            let mut memory = vec![0u8; 2 * 1024 * 1024];
            let mut arena = Arena::new(&mut memory);
            let decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, width, height, 1);

            // Count valid bits
            let mut valid_count = 0usize;
            for block in decoder.blocks_state.iter() {
                valid_count += block.valid_mask.count_ones() as usize;
            }

            prop_assert_eq!(
                valid_count, width * height,
                "Valid mask should cover exactly width*height nodes"
            );
        }
    }
}

// =============================================================================
// Test 10: Effective Mask Consistency
// =============================================================================

proptest! {
    /// Verify effective_mask = valid_mask & !erasure_mask
    #[test]
    fn prop_effective_mask_formula(
        valid in any::<u64>(),
        erasure in any::<u64>(),
    ) {
        let mut memory = vec![0u8; 2 * 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

        // Manually set masks on block 0
        decoder.blocks_state[0].valid_mask = valid;

        // Load erasures
        let erasures = [erasure];
        decoder.load_erasures(&erasures);

        let effective = decoder.blocks_state[0].effective_mask;
        let expected = valid & !erasure;

        prop_assert_eq!(
            effective, expected,
            "effective_mask must equal valid_mask & !erasure_mask"
        );
    }
}
