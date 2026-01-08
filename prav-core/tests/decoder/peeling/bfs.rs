use prav_core::arena::Arena;
use prav_core::decoder::DecodingState;
use prav_core::decoder::peeling::Peeling;
use prav_core::topology::SquareGrid;

#[test]
fn test_trace_bitmask_bfs_simple_path() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);

    // 32x32 grid, Stride 32
    let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

    // Nodes: (1,1)=33, (1,2)=65, (1,3)=97
    let u1 = 33;
    let u2 = 65;
    let u3 = 97;

    // Set defects
    let blk1 = (u1 as usize) / 64;
    let bit1 = (u1 as usize) % 64;
    decoder.defect_mask[blk1] |= 1 << bit1;

    // Set occupied bits to form a path
    decoder.blocks_state[u1 as usize / 64].occupied |= 1 << (u1 % 64);
    decoder.blocks_state[u2 as usize / 64].occupied |= 1 << (u2 % 64);
    decoder.blocks_state[u3 as usize / 64].occupied |= 1 << (u3 % 64);

    // We need a path to boundary. (0,1)=32 is boundary node for x=0.
    let u0 = 32;
    decoder.blocks_state[u0 as usize / 64].occupied |= 1 << (u0 % 64);

    // Call BFS from u1
    decoder.trace_bitmask_bfs(u1);

    // Should have found path u1 -> u0 (boundary)
    // Edge (32, 33) is dir 0 (horizontal).
    // idx = 32 * 3 + 0 = 96.
    // word = 96 / 64 = 1. bit = 96 % 64 = 32.
    assert_ne!(decoder.edge_bitmap[1] & (1 << 32), 0);

    // Should have cleared defect u1
    assert_eq!(decoder.defect_mask[blk1] & (1 << bit1), 0);

    // Edges in path:
    // 32-33 (found above)
}

#[test]
fn test_trace_bitmask_bfs_32x32_all_dirs() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

    // Center point (15, 15) = 15 * 32 + 15 = 480 + 15 = 495
    let u = 495;

    // Test Left (-1)
    decoder.sparse_reset();
    decoder.defect_mask[u as usize / 64] |= 1 << (u % 64);
    decoder.blocks_state[u as usize / 64].occupied |= 1 << (u % 64);
    decoder.mark_block_dirty(u as usize / 64);
    for x in (0..15).rev() {
        let n = 15 * 32 + x;
        decoder.blocks_state[n as usize / 64].occupied |= 1 << (n % 64);
        decoder.mark_block_dirty(n as usize / 64);
    }
    decoder.trace_bitmask_bfs(u);
    // Path includes edge 494-495: idx = 494 * 3 + 0 = 1482. word 23, bit 10.
    assert_ne!(decoder.edge_bitmap[23] & (1 << 10), 0);

    // Test Right (+1)
    decoder.sparse_reset();
    decoder.defect_mask[u as usize / 64] |= 1 << (u % 64);
    decoder.blocks_state[u as usize / 64].occupied |= 1 << (u % 64);
    decoder.mark_block_dirty(u as usize / 64);
    for x in 16..32 {
        let n = 15 * 32 + x;
        decoder.blocks_state[n as usize / 64].occupied |= 1 << (n % 64);
        decoder.mark_block_dirty(n as usize / 64);
    }
    decoder.trace_bitmask_bfs(u);
    // Path includes edge 495-496: idx = 495 * 3 + 0 = 1485. word 23, bit 13.
    assert_ne!(decoder.edge_bitmap[23] & (1 << 13), 0);

    // Test Up (-32)
    decoder.sparse_reset();
    decoder.defect_mask[u as usize / 64] |= 1 << (u % 64);
    decoder.blocks_state[u as usize / 64].occupied |= 1 << (u % 64);
    decoder.mark_block_dirty(u as usize / 64);
    for y in (0..15).rev() {
        let n = y * 32 + 15;
        decoder.blocks_state[n as usize / 64].occupied |= 1 << (n % 64);
        decoder.mark_block_dirty(n as usize / 64);
    }
    decoder.trace_bitmask_bfs(u);
    // Path includes edge 463-495: idx = 463 * 3 + 1 = 1389 + 1 = 1390. word 21, bit 46.
    assert_ne!(decoder.edge_bitmap[21] & (1 << 46), 0);

    // Test Down (+32)
    decoder.sparse_reset();
    decoder.defect_mask[u as usize / 64] |= 1 << (u % 64);
    decoder.blocks_state[u as usize / 64].occupied |= 1 << (u % 64);
    decoder.mark_block_dirty(u as usize / 64);
    for y in 16..32 {
        let n = y * 32 + 15;
        decoder.blocks_state[n as usize / 64].occupied |= 1 << (n % 64);
        decoder.mark_block_dirty(n as usize / 64);
    }
    decoder.trace_bitmask_bfs(u);
    // Path includes edge 495-527: idx = 495 * 3 + 1 = 1485 + 1 = 1486. word 23, bit 14.
    assert_ne!(decoder.edge_bitmap[23] & (1 << 14), 0);
}

#[test]
fn test_trace_bitmask_bfs_large_grid() {
    let mut memory = vec![0u8; 1024 * 1024 * 2];
    let mut arena = Arena::new(&mut memory);

    // 64x64 grid, Stride 64
    let mut decoder = DecodingState::<SquareGrid, 64>::new(&mut arena, 64, 64, 1);

    // (1,1) = 1 * 64 + 1 = 65.
    let u1 = 65;
    let u0 = 64; // boundary

    decoder.defect_mask[u1 as usize / 64] |= 1 << (u1 % 64);
    decoder.blocks_state[u1 as usize / 64].occupied |= 1 << (u1 % 64);
    decoder.blocks_state[u0 as usize / 64].occupied |= 1 << (u0 % 64);

    decoder.trace_bitmask_bfs(u1);

    // Path 64-65: idx = 64 * 3 + 0 = 192. word 3, bit 0.
    assert_ne!(decoder.edge_bitmap[3] & (1 << 0), 0);
    assert_eq!(decoder.defect_mask[u1 as usize / 64] & (1 << (u1 % 64)), 0);
}
