use prav_core::arena::Arena;
use prav_core::decoder::DecodingState;
use prav_core::topology::SquareGrid;

#[test]
fn test_flood_fill_snaking_path() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    // 32x32 grid -> Stride 32.
    let mut state = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

    // Setup U-turn scenario in Block 0
    // Path: 0 (Row0 start) -> 32 (Row1 start) -> ... -> 63 (Row1 end) -> 31 (Row0 end)

    let blk_idx = 0;

    // Boundary: Start at 0
    let boundary_init = 1u64;

    // Mask (valid_mask & erasure_mask):
    // Allow 0 (to start)
    // Allow 32 (below 0)
    // Allow 32..63 (full Row 1)
    // Allow 31 (above 63)
    let mut mask = 0u64;
    mask |= 1 << 0; // Start
    mask |= 1 << 32; // Down
    mask |= 0xFFFFFFFF00000000; // All Row 1
    mask |= 1 << 31; // End target

    unsafe {
        let block = state.blocks_state.get_unchecked_mut(blk_idx);
        block.boundary = boundary_init;
        block.occupied = boundary_init; // Start occupied

        let block_static = state.blocks_state.get_unchecked_mut(blk_idx);
        block_static.valid_mask = mask; // Treat valid as mask
        block_static.erasure_mask = 0;
    }

    // Run growth
    unsafe {
        state.process_block_small_stride::<false>(blk_idx);
    }

    let occupied = unsafe { state.blocks_state.get_unchecked(blk_idx).occupied };

    // Check if bit 31 is set
    assert_eq!(
        occupied & (1 << 31),
        1 << 31,
        "Flood fill should reach bit 31 via U-turn"
    );

    // Check intermediate
    assert_eq!(
        occupied & (1 << 63),
        1 << 63,
        "Flood fill should traverse Row 1"
    );
}

#[test]
fn test_flood_fill_zig_zag() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut state = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

    // Zig-Zag:
    // 0->32. 32->33. 33->1. 1->2. 2->34.
    // 0 (0,0) -> 32 (0,1) -> 33 (1,1) -> 1 (1,0) -> 2 (2,0) -> 34 (2,1)

    let blk_idx = 0;
    let boundary_init = 1u64; // Bit 0

    let mut mask = 0u64;
    mask |= 1 << 0;
    mask |= 1 << 32;
    mask |= 1 << 33;
    mask |= 1 << 1;
    mask |= 1 << 2;
    mask |= 1 << 34;

    unsafe {
        let block = state.blocks_state.get_unchecked_mut(blk_idx);
        block.boundary = boundary_init;
        block.occupied = boundary_init;

        let block_static = state.blocks_state.get_unchecked_mut(blk_idx);
        block_static.valid_mask = mask;
        block_static.erasure_mask = 0;
    }

    unsafe {
        state.process_block_small_stride::<false>(blk_idx);
    }

    let occupied = unsafe { state.blocks_state.get_unchecked(blk_idx).occupied };
    assert_eq!(
        occupied & (1 << 34),
        1 << 34,
        "Flood fill should complete zig-zag"
    );
}
