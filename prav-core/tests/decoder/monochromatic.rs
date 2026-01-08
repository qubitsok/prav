use prav_core::arena::Arena;
use prav_core::decoder::DecodingState;
use prav_core::topology::SquareGrid;

#[test]
fn test_monochromatic_promotion_large_stride() {
    let mut memory = vec![0u8; 1024 * 1024 * 50]; // 50MB
    let mut arena = Arena::new(&mut memory);

    // 64x64 grid. Stride = 64. Large Stride logic.
    let w = 64;
    let h = 64;
    let mut decoder = DecodingState::<SquareGrid, 64>::new(&mut arena, w, h, 1);

    unsafe {
        let blk_idx = 1; // Row 1
        let block = decoder.blocks_state.get_unchecked_mut(blk_idx);
        block.occupied = !0; // All occupied
        block.boundary = !0; // All boundary
        decoder.blocks_state.get_unchecked_mut(blk_idx).valid_mask = !0;

        let root = 64; // First node of Block 1
        for i in 0..64 {
            decoder.parents[64 + i] = root;
        }

        // Block 1 is now effectively monochromatic (all point to 64).
        // But block_root is None initially.
        assert_eq!(decoder.blocks_state[blk_idx].root, u32::MAX);

        // Run process_block.
        // With unified small_stride logic, this block SHOULD be promoted to Mono automatically.
        decoder.process_block(blk_idx);

        // Check promotion (Expect 64)
        // Root might be 64 or whatever find(64) returns (which is 64).
        let expected_root = decoder.find(root);
        assert_eq!(decoder.blocks_state[blk_idx].root, expected_root);

        // Verify that if we explicitly SET it to mono, it STAYS mono.
        decoder.blocks_state[blk_idx].root = expected_root;
        decoder.process_block(blk_idx);
        assert_eq!(decoder.blocks_state[blk_idx].root, expected_root);

        // Test Fast Path Neighbor Interaction (Up)
        // Block 0 (Row 0) has a node that wants to connect.
        let blk_up = 0;
        let block_up = decoder.blocks_state.get_unchecked_mut(blk_up);
        block_up.occupied = 1; // Node 0
        block_up.boundary = 1;
        decoder.blocks_state.get_unchecked_mut(blk_up).valid_mask = !0;
        decoder.parents[0] = 0; // Root 0

        // Reset boundary for Block 1 so it spreads again
        let block = decoder.blocks_state.get_unchecked_mut(blk_idx);
        block.boundary = !0;

        // Block 1 (Mono Root 64) should connect to Block 0 (Node 0).

        // Connects via bit 0. Node 64 (Row 1, Col 0) connects to Node 0 (Row 0, Col 0).

        decoder.process_block(blk_idx);

        // Check if connected

        assert_eq!(decoder.find(64), decoder.find(0));
    }
}

#[test]

fn test_monochromatic_promotion_small_stride() {
    let mut memory = vec![0u8; 1024 * 1024 * 10];

    let mut arena = Arena::new(&mut memory);

    // 16x16 grid. Stride = 16. Small Stride logic.

    let w = 16;

    let h = 16;

    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, w, h, 1);

    unsafe {
        // Block 0 covers 64 nodes.

        // Since Stride=16, Block 0 covers 4 rows (0, 1, 2, 3).

        // Let's fill Block 0 completely.

        let blk_idx = 0;

        let block = decoder.blocks_state.get_unchecked_mut(blk_idx);

        block.occupied = !0;

        block.boundary = !0;

        decoder.blocks_state.get_unchecked_mut(blk_idx).valid_mask = !0;

        let root = 0;

        for i in 0..64 {
            decoder.parents[i] = root;
        }

        // Run process_block.
        // With O(1) cache optimization, it WILL auto-promote to monochromatic.
        // The root will be found during cache miss handling and cached.
        decoder.process_block(blk_idx);

        // After processing, the block should be promoted to monochromatic
        // The root may have changed due to boundary merges, so use find()
        let current_root = decoder.find(root);
        assert_eq!(decoder.blocks_state[blk_idx].root, current_root);

        // Test that root persists after another processing iteration
        decoder.blocks_state.get_unchecked_mut(blk_idx).boundary = !0;
        decoder.process_block(blk_idx);

        let expected = decoder.find(root);

        assert_eq!(decoder.blocks_state[blk_idx].root, expected);

        // Test interactions with neighbor (Block 1).

        // Block 1 starts at node 64.

        // Let's set node 64 occupied.

        let blk_down = 1;

        let block_down = decoder.blocks_state.get_unchecked_mut(blk_down);

        block_down.occupied = 1;

        block_down.boundary = 1;

        decoder.blocks_state.get_unchecked_mut(blk_down).valid_mask = !0;

        decoder.parents[64] = 64;

        // Reset boundary for Block 0

        let block = decoder.blocks_state.get_unchecked_mut(blk_idx);

        block.boundary = !0;

        // Run process_block(0). It should check Down neighbor (Block 1).

        // Node 48 (in Block 0) connects to Node 64 (in Block 1).

        decoder.process_block(blk_idx);

        assert_eq!(decoder.find(0), decoder.find(64));
    }
}

#[test]

fn test_hoisted_root_merge_logic() {
    let mut memory = vec![0u8; 1024 * 1024 * 10];

    let mut arena = Arena::new(&mut memory);

    // 64x64 grid, Stride 64 (Large Stride)

    let w = 64;

    let h = 64;

    let mut decoder = DecodingState::<SquareGrid, 64>::new(&mut arena, w, h, 1);

    unsafe {
        // Setup:

        // Block 1 (Row 1): Monochromatic, Root 64.

        // Block 0 (Row 0): Root 0.

        // Block 2 (Row 2): Root 128.

        // Block 1

        let blk_mono = 1;

        let root_mono = 64;

        let block = decoder.blocks_state.get_unchecked_mut(blk_mono);

        block.occupied = !0;

        block.boundary = !0; // Active boundary

        decoder.blocks_state.get_unchecked_mut(blk_mono).valid_mask = !0;

        for i in 0..64 {
            decoder.parents[64 + i] = root_mono;
        }

        decoder.blocks_state[blk_mono].root = root_mono; // Already promoted

        // Block 0 (Up Neighbor)

        let blk_up = 0;

        let root_up = 0;

        let block_up = decoder.blocks_state.get_unchecked_mut(blk_up);

        block_up.occupied = !0; // Occupied so it merges

        decoder.blocks_state.get_unchecked_mut(blk_up).valid_mask = !0;

        for i in 0..64 {
            decoder.parents[i] = root_up;
        }

        decoder.blocks_state[blk_up].root = root_up;

        // Block 2 (Down Neighbor)

        let blk_down = 2;

        let root_down = 128;

        let block_down = decoder.blocks_state.get_unchecked_mut(blk_down);

        block_down.occupied = !0; // Occupied so it merges

        decoder.blocks_state.get_unchecked_mut(blk_down).valid_mask = !0;

        for i in 0..64 {
            decoder.parents[128 + i] = root_down;
        }

        decoder.blocks_state[blk_down].root = root_down;

        // Verify initial separation

        assert_ne!(decoder.find(root_mono), decoder.find(root_up));

        assert_ne!(decoder.find(root_mono), decoder.find(root_down));


        decoder.parents[0] = 1000;

        decoder.parents[1000] = 1000; // 1000 is a root.


        decoder.blocks_state[blk_up].root = 1000;


        decoder.process_block_small_stride::<false>(blk_mono);

        assert_eq!(
            decoder.find(64),
            decoder.find(1000),
            "Mono should connect to Up"
        );

        assert_eq!(
            decoder.find(64),
            decoder.find(128),
            "Mono should connect to Down"
        );

        assert_eq!(
            decoder.find(1000),
            decoder.find(128),
            "Up and Down should be connected via Mono"
        );
    }
}
