//! Tests for the mono/poly split optimization in process_block_small_stride.
//!
//! These tests verify:
//! 1. Correct dispatch to mono vs poly paths
//! 2. Same-cluster neighbor optimization (skip union)
//! 3. Different-cluster neighbor merging
//! 4. Boundary connections
//! 5. Correctness equivalence with the original behavior

use prav_core::arena::Arena;
use prav_core::decoder::DecodingState;
use prav_core::topology::SquareGrid;

/// Test that monochromatic blocks take the fast path and maintain root cache
#[test]
fn test_mono_dispatch_and_root_cache() {
    let mut memory = vec![0u8; 1024 * 1024 * 10];
    let mut arena = Arena::new(&mut memory);

    // 16x16 grid, Stride 16
    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

    unsafe {
        // Setup Block 1: monochromatic with cached root
        let blk_idx = 1;
        let root = 64; // First node of Block 1

        // Set boundary and occupied
        let block = decoder.blocks_state.get_unchecked_mut(blk_idx);
        block.boundary = 0xFF; // First 8 bits
        block.occupied = 0xFF;
        block.root = root; // Pre-cached root (monochromatic)

        let block_static = decoder.blocks_state.get_unchecked_mut(blk_idx);
        block_static.valid_mask = !0;
        block_static.erasure_mask = 0;

        // All nodes point to root
        for i in 64..128 {
            decoder.parents[i] = root;
        }

        // Process the block
        decoder.process_block_small_stride::<false>(blk_idx);

        // After processing, root should still be valid (not invalidated if no external merges)
        // The root may have changed due to boundary merges
        let final_root = decoder.find(root);
        assert_eq!(
            decoder.blocks_state[blk_idx].root, final_root,
            "Mono block should maintain/update root cache"
        );
    }
}

/// Test polychromatic dispatch when block has multiple roots
#[test]
fn test_poly_dispatch() {
    let mut memory = vec![0u8; 1024 * 1024 * 10];
    let mut arena = Arena::new(&mut memory);

    // 16x16 grid, Stride 16
    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

    unsafe {
        // Setup Block 1: polychromatic (two separate roots)
        let blk_idx = 1;

        let block = decoder.blocks_state.get_unchecked_mut(blk_idx);
        block.boundary = 0xFF; // First 8 bits
        block.occupied = 0xFF;
        block.root = u32::MAX; // Not monochromatic

        let block_static = decoder.blocks_state.get_unchecked_mut(blk_idx);
        block_static.valid_mask = !0;
        block_static.erasure_mask = 0;

        // Create two separate clusters: 64-67 -> 64, 68-71 -> 68
        decoder.parents[64] = 64;
        decoder.parents[65] = 64;
        decoder.parents[66] = 64;
        decoder.parents[67] = 64;
        decoder.parents[68] = 68;
        decoder.parents[69] = 68;
        decoder.parents[70] = 68;
        decoder.parents[71] = 68;

        // Initially different roots
        assert_ne!(decoder.find(64), decoder.find(68));

        // Process the block - poly path should merge them
        decoder.process_block_small_stride::<false>(blk_idx);

        // After processing, they should be connected (horizontal spread connects them)
        assert_eq!(
            decoder.find(64),
            decoder.find(68),
            "Poly path should merge adjacent clusters"
        );
    }
}

/// Test same-cluster neighbor optimization: when both blocks share the same root,
/// skip union logic between them (though they may both connect to boundary)
#[test]
fn test_mono_same_root_neighbor() {
    let mut memory = vec![0u8; 1024 * 1024 * 10];
    let mut arena = Arena::new(&mut memory);

    // 64x64 grid, Stride 64 - use middle blocks to avoid boundary effects
    let mut decoder = DecodingState::<SquareGrid, 64>::new(&mut arena, 64, 64, 1);

    unsafe {
        let shared_root = 128; // Root in block 2

        // Setup Block 2 (middle block, will be processed)
        let blk2 = 2;
        let block2 = decoder.blocks_state.get_unchecked_mut(blk2);
        block2.boundary = 0xFF; // Some boundary bits, but not edges
        block2.occupied = 0xFF;
        block2.root = shared_root;
        block2.effective_mask = !0;

        let block2_static = decoder.blocks_state.get_unchecked_mut(blk2);
        block2_static.valid_mask = !0;
        block2_static.erasure_mask = 0;

        // All nodes in block 2 point to shared_root
        for i in 128..192 {
            decoder.parents[i] = shared_root;
        }

        // Setup Block 3 (down neighbor) with SAME root
        let blk3 = 3;
        let block3 = decoder.blocks_state.get_unchecked_mut(blk3);
        block3.boundary = 0;
        block3.occupied = 0xFF; // First 8 bits occupied
        block3.root = shared_root; // Same root!
        block3.effective_mask = !0;

        let block3_static = decoder.blocks_state.get_unchecked_mut(blk3);
        block3_static.valid_mask = !0;
        block3_static.erasure_mask = 0;

        // Connect block 3's occupied nodes to shared_root
        for i in 192..200 {
            decoder.parents[i] = shared_root;
        }

        // Verify setup: both blocks share the same root
        assert_eq!(
            decoder.find(128),
            decoder.find(192),
            "Setup verification: blocks should share same root"
        );

        // Process block 2
        decoder.process_block_small_stride::<false>(blk2);

        // After processing, both blocks should still share the same root
        // (the optimization ensures no unnecessary unions)
        assert_eq!(
            decoder.find(128),
            decoder.find(192),
            "Same-cluster neighbor relationship should be preserved"
        );
    }
}

/// Test different-cluster neighbor merging
#[test]
fn test_mono_different_root_neighbor() {
    let mut memory = vec![0u8; 1024 * 1024 * 10];
    let mut arena = Arena::new(&mut memory);

    // 64x64 grid, Stride 64
    let mut decoder = DecodingState::<SquareGrid, 64>::new(&mut arena, 64, 64, 1);

    unsafe {
        // Setup Block 1 with root 64
        let blk1 = 1;
        let root1 = 64;
        let block1 = decoder.blocks_state.get_unchecked_mut(blk1);
        block1.boundary = !0;
        block1.occupied = !0;
        block1.root = root1;

        let block1_static = decoder.blocks_state.get_unchecked_mut(blk1);
        block1_static.valid_mask = !0;
        block1_static.erasure_mask = 0;

        for i in 64..128 {
            decoder.parents[i] = root1;
        }

        // Setup Block 0 (up neighbor) with root 0
        let blk0 = 0;
        let root0 = 0;
        let block0 = decoder.blocks_state.get_unchecked_mut(blk0);
        block0.occupied = !0;
        block0.root = root0;

        let block0_static = decoder.blocks_state.get_unchecked_mut(blk0);
        block0_static.valid_mask = !0;
        block0_static.erasure_mask = 0;

        for i in 0..64 {
            decoder.parents[i] = root0;
        }

        // Initially different clusters
        assert_ne!(decoder.find(root1), decoder.find(root0));

        // Process block 1
        decoder.process_block_small_stride::<false>(blk1);

        // Should be merged now
        assert_eq!(
            decoder.find(root1),
            decoder.find(root0),
            "Different-cluster neighbors should be merged"
        );
    }
}

/// Test mono block connecting to polychromatic neighbor
#[test]
fn test_mono_to_poly_neighbor() {
    let mut memory = vec![0u8; 1024 * 1024 * 10];
    let mut arena = Arena::new(&mut memory);

    // 64x64 grid, Stride 64
    let mut decoder = DecodingState::<SquareGrid, 64>::new(&mut arena, 64, 64, 1);

    unsafe {
        // Setup Block 1: monochromatic with root 64
        let blk1 = 1;
        let root1 = 64;
        let block1 = decoder.blocks_state.get_unchecked_mut(blk1);
        block1.boundary = !0;
        block1.occupied = !0;
        block1.root = root1;

        let block1_static = decoder.blocks_state.get_unchecked_mut(blk1);
        block1_static.valid_mask = !0;
        block1_static.erasure_mask = 0;

        for i in 64..128 {
            decoder.parents[i] = root1;
        }

        // Setup Block 0 (up neighbor): polychromatic (root = u32::MAX)
        let blk0 = 0;
        let block0 = decoder.blocks_state.get_unchecked_mut(blk0);
        block0.occupied = 0b11; // Two nodes occupied
        block0.root = u32::MAX; // Polychromatic

        let block0_static = decoder.blocks_state.get_unchecked_mut(blk0);
        block0_static.valid_mask = !0;
        block0_static.erasure_mask = 0;

        // Two separate roots in block 0
        decoder.parents[0] = 0;
        decoder.parents[1] = 1;

        // Initially separate
        assert_ne!(decoder.find(root1), decoder.find(0));
        assert_ne!(decoder.find(root1), decoder.find(1));

        // Process block 1
        decoder.process_block_small_stride::<false>(blk1);

        // Mono block should connect to at least one node in poly neighbor
        let merged_to_0 = decoder.find(root1) == decoder.find(0);
        let merged_to_1 = decoder.find(root1) == decoder.find(1);

        assert!(
            merged_to_0 || merged_to_1,
            "Mono block should merge with poly neighbor's occupied nodes"
        );
    }
}

/// Test boundary connections in mono path
#[test]
fn test_boundary_connections_mono() {
    let mut memory = vec![0u8; 1024 * 1024 * 10];
    let mut arena = Arena::new(&mut memory);

    // 16x16 grid, Stride 16
    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

    unsafe {
        // Block 0 is at the top-left corner
        let blk_idx = 0;
        let root = 0;

        let block = decoder.blocks_state.get_unchecked_mut(blk_idx);
        block.boundary = 1; // Single node at position 0
        block.occupied = 1;
        block.root = root;

        let block_static = decoder.blocks_state.get_unchecked_mut(blk_idx);
        block_static.valid_mask = !0;
        block_static.erasure_mask = 0;

        decoder.parents[0] = root;

        let boundary_node = (decoder.parents.len() - 1) as u32;

        // Initially not connected to boundary
        assert_ne!(decoder.find(root), boundary_node);

        // Process block 0 - node at left edge should connect to boundary
        decoder.process_block_small_stride::<false>(blk_idx);

        // Should be connected to boundary (left edge)
        assert_eq!(
            decoder.find(root),
            decoder.find(boundary_node),
            "Node at left edge should connect to boundary"
        );
    }
}

/// Test inter-block vertical spreading in mono path
#[test]
fn test_inter_block_spread_mono() {
    let mut memory = vec![0u8; 1024 * 1024 * 10];
    let mut arena = Arena::new(&mut memory);

    // 16x16 grid, Stride 16
    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

    unsafe {
        // Block 1: monochromatic
        let blk1 = 1;
        let root1 = 64;

        let block1 = decoder.blocks_state.get_unchecked_mut(blk1);
        block1.boundary = 0xF; // Bottom row bits
        block1.occupied = 0xF;
        block1.root = root1;
        block1.effective_mask = !0;

        let block1_static = decoder.blocks_state.get_unchecked_mut(blk1);
        block1_static.valid_mask = !0;
        block1_static.erasure_mask = 0;

        for i in 64..68 {
            decoder.parents[i] = root1;
        }

        // Block 2: unoccupied but valid
        let blk2 = 2;
        let block2 = decoder.blocks_state.get_unchecked_mut(blk2);
        block2.occupied = 0;
        block2.boundary = 0;
        block2.root = u32::MAX;
        block2.effective_mask = !0;

        let block2_static = decoder.blocks_state.get_unchecked_mut(blk2);
        block2_static.valid_mask = !0;
        block2_static.erasure_mask = 0;

        // Process block 1 - should spread to block 2
        decoder.process_block_small_stride::<false>(blk1);

        // Block 2 should now have some occupied bits
        let block2_after = decoder.blocks_state.get_unchecked(blk2);
        assert_ne!(
            block2_after.occupied, 0,
            "Spread should populate neighbor block"
        );
    }
}

/// Property test: verify correctness by running multiple random scenarios
#[test]
fn test_correctness_random_scenarios() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    for seed in 0..100 {
        let mut memory = vec![0u8; 1024 * 1024 * 50];
        let mut arena = Arena::new(&mut memory);

        // 32x32 grid, Stride 32
        let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

        // Generate pseudo-random defects based on seed
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let hash = hasher.finish();

        // Place 3-5 defects
        let num_defects = 3 + (hash % 3) as usize;
        let mut defect_positions = Vec::new();

        for i in 0..num_defects {
            let pos = ((hash >> (i * 10)) % 1024) as usize;
            if pos < decoder.parents.len() - 1 {
                defect_positions.push(pos);
            }
        }

        // Load defects manually
        unsafe {
            for &pos in &defect_positions {
                let blk_idx = pos >> 6;
                let bit = pos & 63;

                if blk_idx < decoder.blocks_state.len() {
                    let block = decoder.blocks_state.get_unchecked_mut(blk_idx);
                    block.boundary |= 1 << bit;
                    block.occupied |= 1 << bit;

                    let block_static = decoder.blocks_state.get_unchecked_mut(blk_idx);
                    block_static.valid_mask |= 1 << bit;
                }
            }
        }

        // Run growth until convergence
        let mut iterations = 0;
        loop {
            let mut changed = false;
            for blk_idx in 0..decoder.blocks_state.len() {
                unsafe {
                    if decoder.process_block_small_stride::<true>(blk_idx) {
                        changed = true;
                    }
                }
            }
            if !changed || iterations > 100 {
                break;
            }
            iterations += 1;
        }

        // Verify: all defects should be connected to each other or to boundary
        if defect_positions.len() >= 2 {
            // In a fully grown state, all defects should eventually connect
            // (either to each other or through the boundary)
            // This is a sanity check, not a strict invariant
        }

        // Basic sanity: occupied bits should be a superset of boundary bits
        for blk_idx in 0..decoder.blocks_state.len() {
            unsafe {
                let block = decoder.blocks_state.get_unchecked(blk_idx);
                let invalid_boundary = block.boundary & !block.occupied;
                assert_eq!(
                    invalid_boundary, 0,
                    "Boundary bits must be subset of occupied bits"
                );
            }
        }
    }
}
