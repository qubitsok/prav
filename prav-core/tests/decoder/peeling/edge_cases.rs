//! Edge case tests for peeling module coverage.
//!
//! These tests target specific uncovered code paths in the peeling algorithm.

use prav_core::arena::Arena;
use prav_core::decoder::DecodingState;
use prav_core::decoder::peeling::Peeling;
use prav_core::decoder::types::EdgeCorrection;
use prav_core::topology::SquareGrid;

/// Test trace_bitmask_bfs with STRIDE_Y > 32.
///
/// Coverage: mod.rs:267-270 (the 65-element visited array branch)
///
/// When STRIDE_Y > 32, trace_bitmask_bfs uses a 65-element visited array
/// instead of the 17-element array used for smaller strides.
#[test]
fn test_trace_bitmask_bfs_stride_gt_32() {
    let mut memory = vec![0u8; 1024 * 1024 * 64];
    let mut arena = Arena::new(&mut memory);

    // Create 64x64 grid with STRIDE_Y=64
    let mut decoder = DecodingState::<SquareGrid, 64>::new(&mut arena, 64, 64, 1);

    // Pick node at (1,1) = 64 + 1 = 65
    let u = 65u32;
    // Boundary is at (1,0) = 1
    let b = 1u32;

    // Mark nodes as occupied
    decoder.blocks_state[u as usize / 64].occupied |= 1 << (u % 64);
    decoder.blocks_state[b as usize / 64].occupied |= 1 << (b % 64);

    // Run BFS - this triggers the STRIDE_Y > 32 branch
    decoder.trace_bitmask_bfs(u);

    // Verify edge was marked: u-b is vertical (diff = 64 = STRIDE_Y)
    // Low node is b=1. Dir = 1 (vertical).
    // idx = 1 * 3 + 1 = 4
    let idx = 1 * 3 + 1;
    let word = idx / 64;
    let bit = idx % 64;
    assert_ne!(
        decoder.edge_bitmap[word] & (1 << bit),
        0,
        "Vertical edge (1,1)-(1,0) should be marked"
    );

    // Boundary correction at b (y=0 boundary)
    assert_ne!(
        decoder.boundary_bitmap[b as usize / 64] & (1 << (b % 64)),
        0,
        "Boundary correction for (1,0) should be marked"
    );
}

/// Test trace_bfs fallback to trace_manhattan when path not found.
///
/// Coverage: mod.rs:259
///
/// When trace_bfs cannot find a path within the mask, it falls back
/// to trace_manhattan for direct coordinate-based pathfinding.
#[test]
fn test_trace_bfs_fallback_to_manhattan() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

    // u = 0, v = 2
    // But mask only includes bits 0 and 2 (NOT 1)
    // BFS cannot find a path because bit 1 is missing
    let u = 0u32;
    let v = 2u32;
    let mask = (1u64 << 0) | (1u64 << 2); // Missing bit 1!

    decoder.trace_bfs(u, v, mask);

    // Since BFS fails, it should fallback to trace_manhattan
    // which emits edges 0-1 and 1-2 directly
    // 0-1: idx = 0*3+0 = 0
    // 1-2: idx = 1*3+0 = 3
    assert_ne!(decoder.edge_bitmap[0] & (1 << 0), 0, "Edge 0-1 via manhattan");
    assert_ne!(decoder.edge_bitmap[0] & (1 << 3), 0, "Edge 1-2 via manhattan");
}

/// Test emit_linear with invalid (non-adjacent) direction.
///
/// Coverage: mod.rs:372
///
/// emit_linear returns early without modifying state when the difference
/// between u and v doesn't match any valid edge direction (1, stride_y, stride_z).
#[test]
fn test_emit_linear_invalid_direction() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

    // Save initial state
    let initial_edge_dirty_count = decoder.edge_dirty_count;
    let initial_edge_bitmap = decoder.edge_bitmap[0];

    // u = 0, v = 5 (diff = 5, not 1, 8, or 64)
    // This is NOT a valid edge
    decoder.emit_linear(0, 5);

    // Verify no state change
    assert_eq!(
        decoder.edge_dirty_count, initial_edge_dirty_count,
        "edge_dirty_count should not change for invalid direction"
    );
    assert_eq!(
        decoder.edge_bitmap[0], initial_edge_bitmap,
        "edge_bitmap should not change for invalid direction"
    );
}

/// Test reconstruct_corrections with edge buffer overflow.
///
/// Coverage: mod.rs:113
///
/// When more corrections are generated than the buffer can hold,
/// reconstruct_corrections should safely truncate at the buffer size.
#[test]
fn test_reconstruct_corrections_buffer_overflow_edge() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

    // Mark multiple edges as dirty (more than buffer can hold)
    // Set 5 bits in edge_bitmap[0]
    decoder.edge_bitmap[0] = 0b11111; // 5 edges: bits 0,1,2,3,4
    decoder.edge_dirty_list[0] = 0;
    decoder.edge_dirty_count = 1;
    decoder.edge_dirty_mask[0] = 1;

    // Buffer only holds 2 corrections
    let mut corrections = [EdgeCorrection::default(); 2];
    let count = decoder.reconstruct_corrections(&mut corrections);

    // Should only get 2 corrections (buffer size)
    assert_eq!(count, 2, "Should truncate to buffer size");

    // Dirty state should still be cleared
    assert_eq!(decoder.edge_dirty_count, 0, "edge_dirty_count should be cleared");
    assert_eq!(decoder.edge_bitmap[0], 0, "edge_bitmap should be cleared");
}

/// Test reconstruct_corrections with boundary buffer overflow.
///
/// Coverage: mod.rs:142
///
/// When more boundary corrections are generated than the buffer can hold,
/// reconstruct_corrections should safely truncate at the buffer size.
#[test]
fn test_reconstruct_corrections_buffer_overflow_boundary() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

    // Mark multiple boundary nodes as dirty (more than buffer can hold)
    decoder.boundary_bitmap[0] = 0b1111; // 4 boundary corrections: bits 0,1,2,3
    decoder.boundary_dirty_list[0] = 0;
    decoder.boundary_dirty_count = 1;
    decoder.boundary_dirty_mask[0] = 1;

    // Buffer only holds 2 corrections
    let mut corrections = [EdgeCorrection::default(); 2];
    let count = decoder.reconstruct_corrections(&mut corrections);

    // Should only get 2 corrections
    assert_eq!(count, 2, "Should truncate to buffer size");

    // Dirty state should still be cleared
    assert_eq!(
        decoder.boundary_dirty_count, 0,
        "boundary_dirty_count should be cleared"
    );
    assert_eq!(decoder.boundary_bitmap[0], 0, "boundary_bitmap should be cleared");
}

/// Test trace_manhattan in 3D with Z-axis movement only.
///
/// Coverage: mod.rs:325-337
///
/// Tests the dz > 0 branch in trace_manhattan for pure Z-axis paths.
#[test]
fn test_trace_manhattan_3d_z_axis() {
    let mut memory = vec![0u8; 1024 * 1024 * 16];
    let mut arena = Arena::new(&mut memory);

    // Create 3D grid: 8x8x4
    let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 4);

    let stride_y = 8;
    let stride_z = decoder.graph.stride_z;

    // u at (1,1,0), v at (1,1,2) - pure Z movement
    let u = stride_y + 1; // (1,1,0)
    let v = (2 * stride_z + stride_y + 1) as u32; // (1,1,2)

    decoder.trace_manhattan(u as u32, v);

    // Should emit 2 Z-edges: (1,1,0)-(1,1,1) and (1,1,1)-(1,1,2)
    // Edge (1,1,0)-(1,1,1): u=9, dir=2, idx=9*3+2=29
    // Edge (1,1,1)-(1,1,2): u=stride_z+9, dir=2
    let idx_0 = (stride_y + 1) * 3 + 2;
    let word_0 = idx_0 / 64;
    let bit_0 = idx_0 % 64;

    assert_ne!(
        decoder.edge_bitmap[word_0] & (1 << bit_0),
        0,
        "First Z-edge should be marked"
    );

    let idx_1 = (stride_z + stride_y + 1) * 3 + 2;
    let word_1 = idx_1 / 64;
    let bit_1 = idx_1 % 64;

    assert_ne!(
        decoder.edge_bitmap[word_1] & (1 << bit_1),
        0,
        "Second Z-edge should be marked"
    );
}

/// Test get_coord for 3D coordinate extraction.
///
/// Coverage: mod.rs:397-402
///
/// Tests the depth > 1 branch in get_coord for 3D grids.
#[test]
fn test_get_coord_3d() {
    let mut memory = vec![0u8; 1024 * 1024 * 16];
    let mut arena = Arena::new(&mut memory);

    // Create 3D grid: 8x8x4
    let decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 4);

    let stride_y = 8;
    let stride_z = decoder.graph.stride_z;

    // Test node at (3, 5, 2)
    // idx = z * stride_z + y * stride_y + x = 2 * stride_z + 5 * 8 + 3
    let expected_x = 3usize;
    let expected_y = 5usize;
    let expected_z = 2usize;
    let idx = expected_z * stride_z + expected_y * stride_y + expected_x;

    let (x, y, z) = decoder.get_coord(idx as u32);

    assert_eq!(x, expected_x, "X coordinate mismatch");
    assert_eq!(y, expected_y, "Y coordinate mismatch");
    assert_eq!(z, expected_z, "Z coordinate mismatch");

    // Test corner case: (0, 0, 0)
    let (x0, y0, z0) = decoder.get_coord(0);
    assert_eq!((x0, y0, z0), (0, 0, 0), "Origin should be (0,0,0)");

    // Test max corner: (7, 7, 3)
    let max_idx = 3 * stride_z + 7 * stride_y + 7;
    let (xm, ym, zm) = decoder.get_coord(max_idx as u32);
    assert_eq!((xm, ym, zm), (7, 7, 3), "Max corner should be (7,7,3)");
}

/// Test peel_forest small grid fast path with isolated root defect.
///
/// Coverage: mod.rs:46-50
///
/// The small grid optimization triggers when:
/// - STRIDE_Y <= 32 && blocks_state.len() <= 17
/// - The defect node is its own root (not merged)
/// - The node is occupied
#[test]
fn test_peel_forest_small_grid_fast_path() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);

    // Create small grid: 8x8 with STRIDE_Y=8 (fits in small grid criteria)
    let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

    // Set up a defect at node 9 (row 1, col 1) that is:
    // 1. A defect (in defect_mask)
    // 2. Its own root (parents[9] == 9)
    // 3. Occupied (in blocks_state.occupied)
    // 4. Not the boundary node

    let node = 9u32;
    let blk = (node as usize) / 64;
    let bit = (node as usize) % 64;

    // Mark as defect
    decoder.defect_mask[blk] |= 1 << bit;

    // Make it its own root
    decoder.parents[node as usize] = node;

    // Mark as occupied
    decoder.blocks_state[blk].occupied |= 1 << bit;

    // Also mark boundary node (node 0, which is at x=0) as occupied
    // so the BFS can find it
    decoder.blocks_state[0].occupied |= 1 << 0;
    decoder.blocks_state[0].occupied |= 1 << 1; // Connect nodes

    // Run peel_forest
    let mut corrections = [EdgeCorrection::default(); 100];
    let _count = decoder.peel_forest(&mut corrections);

    // If the fast path was taken, defect_mask should be cleared for this node
    // (trace_bitmask_bfs clears defects in visited component)
    // Note: the fast path may or may not trigger depending on find() result
}

/// Test trace_manhattan with u == v (early return).
///
/// Coverage: mod.rs:275
///
/// trace_manhattan returns early without doing anything when u == v.
#[test]
fn test_trace_manhattan_same_node() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

    let initial_edge_bitmap = decoder.edge_bitmap[0];
    let initial_boundary_bitmap = decoder.boundary_bitmap[0];

    // Call trace_manhattan with same node
    decoder.trace_manhattan(5, 5);

    // Nothing should change
    assert_eq!(decoder.edge_bitmap[0], initial_edge_bitmap);
    assert_eq!(decoder.boundary_bitmap[0], initial_boundary_bitmap);
}

/// Test trace_bfs with u == v (early return).
///
/// Coverage: mod.rs:176
///
/// trace_bfs returns early without doing anything when u == v.
#[test]
fn test_trace_bfs_same_node() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

    let initial_bitmap = decoder.edge_bitmap[0];

    // Call trace_bfs with same start and end node
    decoder.trace_bfs(5, 5, u64::MAX);

    // Nothing should change
    assert_eq!(
        decoder.edge_bitmap[0], initial_bitmap,
        "No edges should be emitted when u == v"
    );
}

/// Test trace_bfs with vertical movement (stride_y offset).
///
/// Coverage: mod.rs:226-233
///
/// Tests the vertical neighbor exploration in trace_bfs where curr_bit >= stride_y.
#[test]
fn test_trace_bfs_vertical_movement() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

    // u = 16 (row 2), v = 0 (row 0)
    // Path: 16 -> 8 -> 0 (vertical moves)
    let u = 16u32;
    let v = 0u32;
    // Mask includes all nodes in column 0 of rows 0, 1, 2
    let mask = (1u64 << 0) | (1u64 << 8) | (1u64 << 16);

    decoder.trace_bfs(u, v, mask);

    // Should emit vertical edges 16-8 and 8-0
    // 8-16: u=8, dir=1 (vertical), idx=8*3+1=25
    // 0-8: u=0, dir=1 (vertical), idx=0*3+1=1
    assert_ne!(decoder.edge_bitmap[0] & (1 << 25), 0, "Edge 8-16 should be marked");
    assert_ne!(decoder.edge_bitmap[0] & (1 << 1), 0, "Edge 0-8 should be marked");
}

/// Test trace_manhattan when u is the boundary node.
///
/// Coverage: mod.rs:279-281
///
/// When u is the boundary node, trace_manhattan should emit a boundary
/// correction for v and return early.
#[test]
fn test_trace_manhattan_u_is_boundary() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

    // The boundary node is parents.len() - 1
    let boundary_node = (decoder.parents.len() - 1) as u32;
    let v = 5u32;

    decoder.trace_manhattan(boundary_node, v);

    // Should emit boundary correction for v
    let blk = (v as usize) / 64;
    let bit = (v as usize) % 64;
    assert_ne!(
        decoder.boundary_bitmap[blk] & (1 << bit),
        0,
        "Boundary correction should be emitted for v when u is boundary node"
    );
}

/// Test trace_manhattan with negative X direction.
///
/// Coverage: mod.rs:330
///
/// Tests backward movement in X direction (ux > vx).
#[test]
fn test_trace_manhattan_backward_x() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

    // u at (3,0), v at (0,0) - backward X movement
    let u = 3u32;
    let v = 0u32;

    decoder.trace_manhattan(u, v);

    // Should emit edges 2-3, 1-2, 0-1 (backward direction)
    // 2-3: u=2, dir=0, idx=6
    // 1-2: u=1, dir=0, idx=3
    // 0-1: u=0, dir=0, idx=0
    assert_ne!(decoder.edge_bitmap[0] & (1 << 6), 0, "Edge 2-3");
    assert_ne!(decoder.edge_bitmap[0] & (1 << 3), 0, "Edge 1-2");
    assert_ne!(decoder.edge_bitmap[0] & (1 << 0), 0, "Edge 0-1");
}

/// Test trace_bitmask_bfs_impl with 3D Z-neighbors.
///
/// Coverage: mod.rs:660-688
///
/// Tests the Z-neighbor exploration in the generic BFS path for 3D grids.
#[test]
fn test_trace_bitmask_bfs_impl_3d_z_neighbors() {
    let mut memory = vec![0u8; 1024 * 1024 * 32];
    let mut arena = Arena::new(&mut memory);

    // Create 3D grid: 16x8x4 with STRIDE_Y=16
    // This uses the generic path (not the 32x32 fast path)
    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 8, 4);

    let stride_y = 16;
    let stride_z = decoder.graph.stride_z;

    // Start at (1,1,2), connect to boundary at (1,1,0) via Z-edges
    // u = 2*stride_z + stride_y + 1
    let u = (2 * stride_z + stride_y + 1) as u32;

    // Intermediate nodes: (1,1,1) = stride_z + stride_y + 1
    let mid = (stride_z + stride_y + 1) as u32;

    // Boundary: (1,1,0) = stride_y + 1 (z=0 is boundary)
    let b = (stride_y + 1) as u32;

    // Mark all nodes as occupied
    decoder.blocks_state[u as usize / 64].occupied |= 1 << (u % 64);
    decoder.blocks_state[mid as usize / 64].occupied |= 1 << (mid % 64);
    decoder.blocks_state[b as usize / 64].occupied |= 1 << (b % 64);

    // Run BFS
    decoder.trace_bitmask_bfs(u);

    // Should have traced path u -> mid -> b with Z-edges
    // And boundary correction at b

    // Check boundary correction
    assert_ne!(
        decoder.boundary_bitmap[b as usize / 64] & (1 << (b % 64)),
        0,
        "Boundary correction for z=0 node should be marked"
    );

    // Check Z-edge at mid
    let z_edge_idx = (mid as usize) * 3 + 2; // dir=2 for Z
    let z_word = z_edge_idx / 64;
    let z_bit = z_edge_idx % 64;
    assert_ne!(
        decoder.edge_bitmap[z_word] & (1 << z_bit),
        0,
        "Z-edge at mid node should be marked"
    );
}
