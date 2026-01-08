//! Tests for stride32-specific code paths

use prav_core::arena::Arena;
use prav_core::decoder::state::{BoundaryConfig, DecodingState};
use prav_core::topology::SquareGrid;

#[test]
fn test_stride32_hole_connection_up_neighbor() {
    // Test hole detection when spreading up to invalid neighbor (lines 481-491)
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    // 32x32 grid with a hole (invalid node)
    let mut state = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

    // Block 1 spreading up to block 0 which has a hole
    unsafe {
        // Mark some nodes in block 0 as invalid (hole)
        let block0 = state.blocks_state.get_unchecked_mut(0);
        block0.valid_mask = !0x8000_0000_0000_0000u64; // Bit 63 invalid (hole)
        block0.effective_mask = block0.valid_mask;
        block0.boundary = 0;
        block0.occupied = 0;
        block0.root = u32::MAX;

        // Block 1 has boundary at bit 31 (row 0 of block 1)
        // which spreads up to bit 63 of block 0 (the hole)
        let block1 = state.blocks_state.get_unchecked_mut(1);
        block1.boundary = 1u64 << 31; // Bit 31 = row 0 of block 1
        block1.occupied = 1u64 << 31;
        block1.root = 64 + 31; // Root at node 95
        block1.valid_mask = !0;
        block1.effective_mask = !0;

        // Set up parent
        state.parents[64 + 31] = (64 + 31) as u32;

        state.process_block_small_stride::<false>(1);

        // The spread should hit the hole and connect to boundary
        let boundary_node = (state.parents.len() - 1) as u32;
        let root = state.find(95);
        assert_eq!(
            root, boundary_node,
            "Spreading into hole should connect to boundary"
        );
    }
}

#[test]
fn test_stride32_internal_boundary_up() {
    // Test internal boundary up path (lines 702-715)
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut state = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

    unsafe {
        // Block 0 with hole in row 0 (internal boundary)
        let block0 = state.blocks_state.get_unchecked_mut(0);
        block0.valid_mask = !(1u64); // Bit 0 is invalid (internal hole)
        block0.effective_mask = block0.valid_mask;
        block0.boundary = 1 << 1; // Bit 1 active
        block0.occupied = 1 << 1;
        block0.root = 1;

        state.parents[1] = 1;

        state.process_block_small_stride::<false>(0);

        // Spread should connect to boundary via internal hole
        let boundary_node = (state.parents.len() - 1) as u32;
        let root = state.find(1);
        assert_eq!(
            root, boundary_node,
            "Internal hole should trigger boundary connection"
        );
    }
}

#[test]
fn test_stride32_poly_dispatch() {
    // Test polychromatic dispatch path in stride32
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut state = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

    // Disable boundary checks
    state.boundary_config = BoundaryConfig {
        check_top: false,
        check_bottom: false,
        check_left: false,
        check_right: false,
    };

    unsafe {
        // Block 1 with multiple boundary bits that have different roots (poly)
        let block1 = state.blocks_state.get_unchecked_mut(1);
        block1.boundary = (1 << 0) | (1 << 32); // Two bits: row 0 and row 1
        block1.occupied = (1 << 0) | (1 << 32);
        block1.root = u32::MAX; // Polychromatic
        block1.valid_mask = !0;
        block1.effective_mask = !0;

        // Set up different roots
        let base = 64;
        state.parents[base + 0] = (base + 0) as u32; // Root at node 64
        state.parents[base + 32] = (base + 32) as u32; // Root at node 96

        state.process_block_small_stride::<false>(1);

        // The two should merge
        let root64 = state.find(64);
        let root96 = state.find(96);
        assert_eq!(root64, root96, "Poly block should merge different roots");
    }
}

#[test]
fn test_stride32_last_block_bottom_boundary() {
    // Test bottom boundary on last block (STRIDE_Y=32 specific)
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut state = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

    // Enable bottom boundary check
    state.boundary_config = BoundaryConfig {
        check_top: false,
        check_bottom: true,
        check_left: false,
        check_right: false,
    };

    unsafe {
        // Last data block (block 15 for 32x32)
        let num_blocks = state.blocks_state.len();
        let last_data_blk = num_blocks - 2;
        let base = last_data_blk * 64;

        // Set up boundary at bottom row
        let block = state.blocks_state.get_unchecked_mut(last_data_blk);
        block.boundary = 0xFFFF_FFFF_0000_0000u64; // Row 1 (bottom of block)
        block.occupied = 0xFFFF_FFFF_0000_0000u64;
        block.root = (base + 32) as u32;
        block.valid_mask = !0;
        block.effective_mask = !0;

        state.parents[base + 32] = (base + 32) as u32;
        for i in 33..64 {
            state.parents[base + i] = (base + 32) as u32;
        }

        state.process_block_small_stride::<false>(last_data_blk);

        // Should connect to boundary
        let boundary_node = (state.parents.len() - 1) as u32;
        let root = state.find((base + 32) as u32);
        assert_eq!(
            root, boundary_node,
            "Last block bottom should connect to boundary"
        );
    }
}

#[test]
fn test_stride32_down_neighbor_hole_connection() {
    // Test hole detection when spreading down
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut state = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

    unsafe {
        // Block 1 has a hole in row 0
        let block1 = state.blocks_state.get_unchecked_mut(1);
        block1.valid_mask = !1u64; // Bit 0 is invalid
        block1.effective_mask = block1.valid_mask;
        block1.boundary = 0;
        block1.occupied = 0;
        block1.root = u32::MAX;

        // Block 0 spreading down to block 1's hole
        let block0 = state.blocks_state.get_unchecked_mut(0);
        block0.boundary = 1u64 << 32; // Bit 32 (row 1, col 0) spreads down to bit 0 of block 1
        block0.occupied = 1u64 << 32;
        block0.root = 32;
        block0.valid_mask = !0;
        block0.effective_mask = !0;

        state.parents[32] = 32;

        state.process_block_small_stride::<false>(0);

        // Should connect to boundary via hole
        let boundary_node = (state.parents.len() - 1) as u32;
        let root = state.find(32);
        assert_eq!(
            root, boundary_node,
            "Spreading into down neighbor hole should connect to boundary"
        );
    }
}
