use prav_core::arena::Arena;
use prav_core::decoder::DecodingState;
use prav_core::topology::SquareGrid;

#[test]
fn test_vertical_merge_explicit_loop() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut state = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

    let blk_idx = 0;
    let base_global = blk_idx * 64;

    // Setup Polychromatic state
    // Row 0: Bit 0 active. Root = 0.
    // Row 1: Bit 0 active (Index 32). Root = 32.
    // They are separate components initially.

    unsafe {
        let block = state.blocks_state.get_unchecked_mut(blk_idx);
        block.boundary = (1 << 0) | (1 << 32);
        block.occupied = (1 << 0) | (1 << 32);

        // Ensure roots are distinct
        state.parents[base_global + 0] = (base_global + 0) as u32;
        state.parents[base_global + 32] = (base_global + 32) as u32;

        // Restrict valid mask to only bits 0 and 32 to prevent spreading to bit 1
        state.blocks_state.get_unchecked_mut(blk_idx).valid_mask = (1 << 0) | (1 << 32);
    }

    // This should trigger process_block_small_stride_32 -> Polychromatic path -> vertical_pairs loop
    unsafe {
        state.process_block_small_stride::<false>(blk_idx);
    }

    let root0 = state.find((base_global + 0) as u32);
    let root32 = state.find((base_global + 32) as u32);

    assert_eq!(root0, root32, "Vertical pair (0, 32) should be merged");

    // Verify it didn't merge unrelated bits (e.g. bit 1)
    let root1 = state.find((base_global + 1) as u32);
    assert_ne!(root0, root1, "Unrelated nodes should not be merged");
}
