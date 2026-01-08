use prav_core::arena::Arena;
use prav_core::decoder::{ClusterGrowth, DecodingState, EdgeCorrection, Peeling};
use prav_core::topology::SquareGrid;

#[test]
fn test_grow_iteration_explicit() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 4>::new(&mut arena, 4, 4, 1);

    // Setup a scenario where growth is needed
    // 4x4 grid. Block 0 (0..63) covers the whole grid (since 4x4=16 < 64).
    // Just inject a syndrome.
    let syndromes = vec![1]; // Defect at node 0
    decoder.load_dense_syndromes(&syndromes);

    // Verify internal state exposed via public API (for advanced users)
    if !decoder.is_small_grid() {
        // Check active_mask
        assert_ne!(decoder.active_mask[0] & 1, 0, "Block 0 should be active");
    } else {
        // Updated expectation: Small grids use active_block_mask optimization
        assert_ne!(decoder.active_block_mask, 0);
    }

    // Step 1: Grow
    let expanded = decoder.grow_iteration();
    assert!(expanded);

    // Check that growth actually happened (occupied set)
    assert_ne!(decoder.blocks_state[0].occupied, 0);
}

#[test]
fn test_compact_corrections_explicit() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

    let mut corrections = vec![EdgeCorrection::default(); 100];

    // Simulate edge emissions using emit_linear
    decoder.emit_linear(1, 2);
    decoder.emit_linear(3, 4);
    decoder.emit_linear(1, 2); // Duplicate (should toggle off)
    decoder.emit_linear(5, 6);
    decoder.emit_linear(3, 4); // Duplicate (should toggle off)
    decoder.emit_linear(3, 4); // Triplicate (should toggle on)

    let new_count = decoder.reconstruct_corrections(&mut corrections);

    // Expected:
    // (1,2) emitted twice -> 0
    // (3,4) emitted 3 times -> 1
    // (5,6) emitted once -> 1
    // Total 2.

    assert_eq!(new_count, 2);

    // Verify contents
    let result = &corrections[0..new_count];
    // Order depends on iteration order of hash map/bitmap, so we just check existence.
    // Note: (3,4) and (5,6).

    // Sort result for easier checking or just check existence
    let mut sorted_res = result.to_vec();
    sorted_res.sort();

    assert!(sorted_res.contains(&EdgeCorrection { u: 3, v: 4 }));
    assert!(sorted_res.contains(&EdgeCorrection { u: 5, v: 6 }));
    assert!(!sorted_res.contains(&EdgeCorrection { u: 1, v: 2 }));
}
