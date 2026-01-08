use prav_core::arena::Arena;
use prav_core::decoder::state::{DecodingState, EdgeCorrection};
use prav_core::topology::SquareGrid;

#[test]
fn test_trace_manhattan_basic() {
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut buffer);
    // 4x4x1 grid
    // Nodes: 16
    // Stride Y: 4
    let width = 4;
    let height = 4;
    let depth = 1;
    let mut decoder = DecodingState::<SquareGrid, 4>::new(&mut arena, width, height, depth);

    // Simulate a path: 0 -> 1 -> 5 -> 9 (defect at 0, root at 9)
    // 0 is (0,0), 1 is (1,0), 5 is (1,1), 9 is (1,2)
    // Parents:
    // 0 -> 1
    // 1 -> 5
    // 5 -> 9
    // 9 -> 9 (root)

    decoder.parents[0] = 1;
    decoder.parents[1] = 5;
    decoder.parents[5] = 9;
    decoder.parents[9] = 9;

    // We rely on peel_forest to populate path_mark from defect_mask.
    // peel_forest clears path_mark at start.
    // Set defect at node 0.
    decoder.defect_mask[0] = 1;

    let mut corrections = vec![EdgeCorrection { u: 0, v: 0 }; 10];

    // peel_forest will:
    // 1. Clear path_mark.
    // 2. Read defect_mask. Found defect at 0.
    // 3. Trace 0 -> 1 -> 5 -> 9. Mark 0, 1, 5.
    // 4. Iterate path_mark.
    //    Bit 0 set (node 0) -> trace_manhattan(0, 1). Emits (0,1).
    //    Bit 1 set (node 1) -> trace_manhattan(1, 5). Emits (1,5).
    //    Bit 5 set (node 5) -> trace_manhattan(5, 9). Emits (5,9).

    let count = decoder.peel_forest(&mut corrections);

    // Expected edges:
    // (0,1) -> from processing 0
    // (1,5) -> from processing 1
    // (5,9) -> from processing 5
    // (9,9) -> from processing 9, trace_manhattan(9,9) does nothing.

    // Note: peel_forest iterates path_mark.
    // Order depends on bit traversal (LSB first).
    // 0 -> trace(0, 1) -> emits (0, 1)
    // 1 -> trace(1, 5) -> emits (1, 5)
    // 5 -> trace(5, 9) -> emits (5, 9)
    // 9 -> trace(9, 9) -> nothing.

    assert_eq!(count, 3);

    let mut edges: Vec<(u32, u32)> = corrections[0..count].iter().map(|c| (c.u, c.v)).collect();
    edges.sort(); // Sort for comparison

    assert_eq!(edges[0], (0, 1));
    assert_eq!(edges[1], (1, 5));
    assert_eq!(edges[2], (5, 9));
}

#[test]
fn test_trace_manhattan_non_adjacent() {
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut buffer);
    // 4x4x1 grid
    let width = 4;
    let height = 4;
    let depth = 1;
    let mut decoder = DecodingState::<SquareGrid, 4>::new(&mut arena, width, height, depth);

    // Simulate "virtual" edge or multi-step parent
    // 0 -> 2 (skip 1). (0,0) -> (2,0)
    // distance 2.
    decoder.parents[0] = 2;
    decoder.parents[2] = 2; // Root

    decoder.defect_mask[0] = 1;

    let mut corrections = vec![EdgeCorrection { u: 0, v: 0 }; 10];
    let count = decoder.peel_forest(&mut corrections);

    // trace_manhattan(0, 2)
    // Should emit (0, 1) and (1, 2)
    assert_eq!(count, 2);
    let mut edges: Vec<(u32, u32)> = corrections[0..count].iter().map(|c| (c.u, c.v)).collect();
    edges.sort();
    assert_eq!(edges[0], (0, 1));
    assert_eq!(edges[1], (1, 2));
}
