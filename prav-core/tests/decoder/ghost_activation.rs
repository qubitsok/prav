use prav_core::arena::Arena;
use prav_core::decoder::state::DecodingState;
use prav_core::topology::SquareGrid;

#[test]
fn test_ghost_activation_prevention_via_sparse_reset() {
    let width = 64;
    let height = 64;
    let buffer_size = 1024 * 1024 * 32;
    let mut buffer = vec![0u8; buffer_size];
    let mut arena = Arena::new(&mut buffer);

    let mut decoder = DecodingState::<SquareGrid, 64>::new(&mut arena, width, height, 1);

    // Setup: Simulate garbage in Block 0
    let blk_idx = 0;
    decoder.blocks_state[blk_idx].boundary = 1; // Bit 0 set
    decoder.blocks_state[blk_idx].occupied = 1;
    decoder.mark_block_dirty(blk_idx);

    // Perform reset
    decoder.sparse_reset();

    unsafe {
        let expanded = decoder.process_block(blk_idx);

        assert!(
            !expanded,
            "Ghost activation detected! Stale boundary data was processed after reset."
        );
        assert_eq!(
            decoder.blocks_state[blk_idx].occupied, 0,
            "Occupied value should be 0 after reset"
        );
    }
}
