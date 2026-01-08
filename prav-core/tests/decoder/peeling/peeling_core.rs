use prav_core::arena::Arena;
use prav_core::decoder::DecodingState;
use prav_core::decoder::peeling::Peeling;
use prav_core::topology::SquareGrid;
use prav_core::decoder::types::EdgeCorrection;

#[test]
fn test_peeling_trace_path_and_reconstruct() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);

    // 32x32 grid
    let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

    // Create a chain: 0 -> 1 -> 33 (1,1) -> 32 (0,1) -> Boundary (root)
    // Nodes:
    // 0: (0,0)
    // 1: (1,0)
    // 33: (1,1)
    // 32: (0,1)
    // Boundary is typically parents.len() - 1.

    let boundary_node = (decoder.parents.len() - 1) as u32;

    // Set up parents structure manually to simulate a union-find tree
    decoder.parents[0] = 1;
    decoder.parents[1] = 33;
    decoder.parents[33] = 32;
    decoder.parents[32] = boundary_node;
    
    // trace_path from 0 to boundary
    decoder.trace_path(0, boundary_node);

    // Check path_mark
    // The path is 0->1, 1->33, 33->32, 32->boundary.
    // trace_path marks nodes on the path from u up to (but not including?) 
    // Wait, trace_path loop:
    // curr = u;
    // loop {
    //   next = parents[curr];
    //   if curr == next { break; }
    //   mark(curr);
    //   curr = next;
    // }
    // So it marks 0, 1, 33, 32.
    
    let nodes_to_check = [0, 1, 33, 32];
    for &node in &nodes_to_check {
        let blk = (node as usize) / 64;
        let bit = (node as usize) % 64;
        assert_ne!(decoder.path_mark[blk] & (1 << bit), 0, "Node {} should be marked", node);
    }
    
    // Now simulate peel_forest's second pass logic manually or call it?
    // peel_forest logic:
    // for marked u: v = parents[u]. trace_manhattan(u, v).
    
    // 0 -> 1
    decoder.trace_manhattan(0, 1);
    // 1 -> 33
    decoder.trace_manhattan(1, 33);
    // 33 -> 32
    decoder.trace_manhattan(33, 32);
    // 32 -> boundary
    decoder.trace_manhattan(32, boundary_node);

    // Now reconstruct corrections
    let mut corrections = vec![EdgeCorrection::default(); 10];
    let count = decoder.reconstruct_corrections(&mut corrections);

    assert_eq!(count, 4);

    // Check edges.
    // 0->1: Horizontal edge.
    // 1->33: Vertical edge (+32).
    // 33->32: Horizontal edge (-1).
    // 32->Boundary.

    // Note: corrections are unordered in the list usually, or ordered by iteration order.
    
    let mut found_h1 = false; // 0-1
    let mut found_v1 = false; // 1-33
    let mut found_h2 = false; // 32-33
    let mut found_b = false;  // 32-Boundary

    for i in 0..count {
        let c = corrections[i];
        if c.v == u32::MAX {
            if c.u == 32 { found_b = true; }
        } else {
            let (u, v) = if c.u < c.v { (c.u, c.v) } else { (c.v, c.u) };
            if u == 0 && v == 1 { found_h1 = true; }
            if u == 1 && v == 33 { found_v1 = true; }
            if u == 32 && v == 33 { found_h2 = true; }
        }
    }

    assert!(found_h1, "Missing edge 0-1");
    assert!(found_v1, "Missing edge 1-33");
    assert!(found_h2, "Missing edge 32-33");
    assert!(found_b, "Missing boundary edge for 32");
}

#[test]
fn test_peeling_3d_manhattan() {
    let mut memory = vec![0u8; 1024 * 1024 * 2];
    let mut arena = Arena::new(&mut memory);

    // 8x8x8 grid? 
    // SquareGrid is usually 2D. We need a 3D topology or just use SquareGrid with depth > 1 if supported.
    // DecodingState has generic T: Topology. SquareGrid might be 2D only.
    // Let's check Topology trait.
    // For now, let's assume we can construct a 3D state if we set depth > 1.
    // The new method signature: new(arena, width, height, depth).
    
    let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 2);
    // width=8, height=8, depth=2. Stride Y=8. Stride Z=64.
    
    // u = (0,0,0) = 0
    // v = (1,1,1) = 1*64 + 1*8 + 1 = 73
    
    let u = 0;
    let v = 73;
    
    decoder.trace_manhattan(u, v);
    
    // Path should satisfy dx=1, dy=1, dz=1.
    // Edges:
    // (0,0,0)-(1,0,0) [Horizontal]
    // (1,0,0)-(1,1,0) [Vertical]
    // (1,1,0)-(1,1,1) [Z-edge]
    // Or some permutation depending on implementation order.
    // trace_manhattan order: X then Y then Z.
    
    // 1. X loop: (0,0,0) -> (1,0,0). u=0, next=1. Emit(0,1).
    // 2. Y loop: (1,0,0) -> (1,1,0). u=1, next=9. Emit(1,9).
    // 3. Z loop: (1,1,0) -> (1,1,1). u=9, next=73. Emit(9,73).
    
    let mut corrections = vec![EdgeCorrection::default(); 10];
    let count = decoder.reconstruct_corrections(&mut corrections);
    
    assert_eq!(count, 3);
    
    let mut found_x = false;
    let mut found_y = false;
    let mut found_z = false;
    
    for i in 0..count {
        let c = corrections[i];
        let (u, v) = if c.u < c.v { (c.u, c.v) } else { (c.v, c.u) };
        if u == 0 && v == 1 { found_x = true; }
        if u == 1 && v == 9 { found_y = true; }
        if u == 9 && v == 73 { found_z = true; }
    }
    
    assert!(found_x, "Missing X edge");
    assert!(found_y, "Missing Y edge");
    assert!(found_z, "Missing Z edge");
}
