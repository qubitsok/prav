use prav_core::arena::Arena;
use prav_core::decoder::DecodingState;
use prav_core::decoder::peeling::Peeling;
use prav_core::topology::SquareGrid;
use prav_core::decoder::types::EdgeCorrection;

#[test]
fn test_trace_bitmask_bfs_generic_stride_16() {
    // 1MB memory
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);

    // Stride 16. Width 16, Height 16.
    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

    // Pick two points.
    // (1,1) = 1*16 + 1 = 17
    // (2,2) = 2*16 + 2 = 34
    let _u = 17;
    let _v = 34;

    // We want to trace from u. We need to set up the grid such that BFS finds v.
    // trace_bitmask_bfs searches for *occupied* nodes in the same cluster that are NOT visited.
    // But wait, trace_bitmask_bfs_impl uses `bfs_queue` and `visited`.
    // It explores neighbors. If a neighbor is occupied and not visited, it adds to queue.
    // It stops when?
    // It seems to stop when it hits a boundary OR it explores everything.
    // Wait, let's re-read trace_bitmask_bfs_impl.
    
    // It sets `boundary_hit`.
    // If it hits boundary, it traces back from boundary to start.
    // It emits linear corrections.
    
    // So to test it, we should connect u to boundary via occupied nodes.
    // Boundary nodes are x=0, x=w-1, y=0, y=h-1.
    
    // Let's connect (1,1) -> (0,1) [Boundary]
    // u = 17 (1,1). Neighbor (0,1) = 1.
    // Node 1 is on boundary x=0? Yes, 1 % 16 = 1. Wait.
    // x = u % 16. y = u / 16.
    // 17: x=1, y=1.
    // 1: x=1, y=0. y=0 is boundary.
    
    // So if we make (1,0) occupied, BFS from (1,1) should find it.
    
    let u = 17; // (1,1)
    let b = 1;  // (1,0) - Boundary
    
    decoder.defect_mask[u as usize / 64] |= 1 << (u % 64); // Start node usually has defect or something
    
    // Mark nodes as occupied
    decoder.blocks_state[u as usize / 64].occupied |= 1 << (u % 64);
    decoder.blocks_state[b as usize / 64].occupied |= 1 << (b % 64);
    
    decoder.trace_bitmask_bfs(u);
    
    // Should have emitted correction for edge (1,1)-(1,0).
    // Edge between 1 and 17. 17-1 = 16 = STRIDE_Y. Vertical edge.
    // Low node is 1. Dir = 1.
    // idx = 1 * 3 + 1 = 4.
    // Word 0, bit 4.
    
    assert_ne!(decoder.edge_bitmap[0] & (1 << 4), 0, "Vertical edge (1,1)-(1,0) should be marked");
    
    // And since 1 is boundary, it should emit boundary correction for 1?
    // trace_bitmask_bfs_impl:
    // if boundary_hit != MAX:
    //    emit_linear(curr, MAX);
    //    ...
    
    // curr starts at boundary_hit (1).
    // emit_linear(1, MAX).
    // 1 is in block 0. bit 1.
    assert_ne!(decoder.boundary_bitmap[0] & (1 << 1), 0, "Boundary correction for (1,0) should be marked");
}

#[test]
fn test_trace_bitmask_bfs_generic_3d() {
    let mut memory = vec![0u8; 1024 * 1024 * 16];
    let mut arena = Arena::new(&mut memory);

    // Stride 32, but 3D.
    // Width 32, Height 4, Depth 4.
    // Stride Y = 32 (default for const generic).
    // We need to ensure logic uses generic path.
    // trace_bitmask_bfs checks: if STRIDE_Y == 32 && !is_3d
    
    // We make a 3D grid.
    let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 4, 4);
    
    // (1,1,1).
    // In 3D: z * stride_z + y * stride_y + x.
    // stride_z = stride_y * height?
    // DecodingState calculation for stride_z needs checking.
    // Usually stride_z is next power of 2 of (stride_y * height).
    // width=32 -> stride_y=32.
    // height=4. 32*4 = 128. stride_z = 128?
    
    // Let's assume stride_z is accessible via decoder.graph.stride_z
    let stride_z = decoder.graph.stride_z as u32;
    let stride_y = 32;
    
    // u = (1,1,1) = 1*stride_z + 1*stride_y + 1
    let u = stride_z + stride_y + 1;
    
    // Connect to boundary (1,1,0) -> z=0 is boundary.
    let b = stride_y + 1; // (1,1,0)
    
    decoder.blocks_state[u as usize / 64].occupied |= 1 << (u % 64);
    decoder.blocks_state[b as usize / 64].occupied |= 1 << (b % 64);
    
    decoder.trace_bitmask_bfs(u);
    
    // Should find path u -> b.
    // Edge u-b is Z-edge. diff = stride_z.
    // Low node b. Dir = 2 (Z).
    // idx = b * 3 + 2.
    
    let idx = (b as usize) * 3 + 2;
    let word = idx / 64;
    let bit = idx % 64;
    
    assert_ne!(decoder.edge_bitmap[word] & (1 << bit), 0, "Z edge should be marked");
    
    // Boundary correction at b (z=0 boundary)
    assert_ne!(decoder.boundary_bitmap[b as usize / 64] & (1 << (b % 64)), 0, "Boundary correction should be marked");
}

#[test]
fn test_trace_bfs_explicit() {
    // trace_bfs(u, v, mask) finds path from u to v using BFS, respecting mask (maybe?)
    // Actually looking at code:
    // mask argument is used in try_queue: if (mask & (1 << next)) != 0 ...
    // So it restricts traversal to nodes present in 'mask'.
    // Wait, mask is a u64. But 'next' is a node index (0..64 typically for local bfs?).
    // Ah, trace_bfs:
    // u_local = u % 64;
    // It seems trace_bfs is a LOCAL BFS within a block (or 64 neighbors)?
    // Let's check code:
    // let u_local = (u % 64) as usize;
    // let v_local = (v % 64) as usize;
    // ...
    // queue |= 1 << u_local;
    // ...
    // try_queue(... mask ... next ...)
    
    // So trace_bfs seems to find path between u and v assuming they are close or within 64-node window?
    // And `mask` specifies allowed nodes relative to something?
    // No, `try_queue` takes `next` (0..63).
    // `mask` is likely the "active nodes" in the block.
    
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);
    
    // Use block 0.
    // u = 0, v = 2.
    // Path 0 -> 1 -> 2.
    // mask needs bits 0, 1, 2 set.
    
    let u = 0;
    let v = 2;
    let mask = (1 << 0) | (1 << 1) | (1 << 2);
    
    decoder.trace_bfs(u, v, mask);
    
    // Should emit edges 0-1 and 1-2.
    // 0-1: u=0, dir=0 (diff 1). idx=0*3+0=0.
    // 1-2: u=1, dir=0. idx=1*3+0=3.
    
    assert_ne!(decoder.edge_bitmap[0] & (1 << 0), 0, "Edge 0-1");
    assert_ne!(decoder.edge_bitmap[0] & (1 << 3), 0, "Edge 1-2");
}

#[test]
fn test_reconstruct_corrections_boundary_dirty() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);
    
    // Manually mark a boundary dirty.
    let blk_idx = 0;
    let bit_idx = 5;
    let _u = 5;
    
    decoder.boundary_bitmap[blk_idx] |= 1 << bit_idx;
    // Mark as dirty
    decoder.boundary_dirty_mask[0] |= 1 << 0; // mask_idx 0, mask_bit 0 (since blk_idx=0)
    decoder.boundary_dirty_list[0] = 0;
    decoder.boundary_dirty_count = 1;
    
    let mut corrections = vec![EdgeCorrection::default(); 10];
    let count = decoder.reconstruct_corrections(&mut corrections);
    
    assert_eq!(count, 1);
    assert_eq!(corrections[0].u, 5);
    assert_eq!(corrections[0].v, u32::MAX);
    
    // Ensure dirty state is cleared
    assert_eq!(decoder.boundary_dirty_count, 0);
    assert_eq!(decoder.boundary_bitmap[0], 0);
    assert_eq!(decoder.boundary_dirty_mask[0], 0);
}
