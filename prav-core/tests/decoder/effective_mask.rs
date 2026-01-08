#![cfg(test)]

use prav_core::arena::Arena;
use prav_core::decoder::growth::ClusterGrowth;
use prav_core::decoder::state::DecodingState;
use prav_core::topology::SquareGrid;

#[test]
fn test_effective_mask_initialization() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

    for i in 0..decoder.blocks_state.len() {
        assert_eq!(
            decoder.blocks_state[i].effective_mask, decoder.blocks_state[i].valid_mask,
            "Block {} mismatch",
            i
        );
    }
}

#[test]
fn test_effective_mask_update() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

    // Erase bits in first block
    let erasure_pattern = 0x0000_0000_FFFF_FFFF;
    let erasures = vec![erasure_pattern];
    decoder.load_erasures(&erasures);

    let valid = decoder.blocks_state[0].valid_mask;
    let expected = valid & !erasure_pattern;

    assert_eq!(decoder.blocks_state[0].erasure_mask, erasure_pattern);
    assert_eq!(
        decoder.blocks_state[0].effective_mask, expected,
        "Effective mask should be valid & !erasure"
    );

    // Check second block (not erased)
    assert_eq!(
        decoder.blocks_state[1].effective_mask,
        decoder.blocks_state[1].valid_mask
    );
}

#[test]
fn test_growth_hits_erasure() {
    // Verify that growth into an erasure is treated as a boundary hit (due to new optimization)
    // or at least handled gracefully.

    let mut memory = vec![0u8; 10 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

    // Setup:
    // Block 0: Active, occupied.
    // Block 1: Valid but erased.
    // We expect growth from Block 0 to "hit" Block 1.
    // Since Block 1 is erased, effective_mask[1] is 0.
    // Flow from 0 to 1 will check `andnot(m1, flow)`. m1 is 0. So `flow & !0` = flow.
    // So it registers a hit.

    let root = decoder.find(0);
    unsafe {
        let block0 = decoder.blocks_state.get_unchecked_mut(0);
        block0.occupied = !0; // Full block
        block0.boundary = !0; // Active boundary
        block0.root = root;

        // Erase block 1
        let erasures = vec![0, !0]; // Block 0 clean, Block 1 full erasure
        decoder.load_erasures(&erasures);

        decoder.active_block_mask = 1; // Activate Block 0
    }

    // Boundary node is the last parent
    let boundary_node = (decoder.parents.len() - 1) as u32;
    assert_ne!(
        decoder.find(root),
        boundary_node,
        "Root should not be boundary yet"
    );

    // Perform growth
    // Note: grow_iteration returns bool (any_expanded)
    let res = decoder.grow_iteration();
    assert!(res);

    // Check if root is now connected to boundary
    let new_root = decoder.find(root);
    assert_eq!(
        new_root, boundary_node,
        "Growth into erasure should trigger boundary connection"
    );
}
