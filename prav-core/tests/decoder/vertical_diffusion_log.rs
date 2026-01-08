use prav_core::arena::Arena;
use prav_core::decoder::state::DecodingState;
use prav_core::topology::SquareGrid;

#[test]
fn test_log_vertical_diffusion_stride_4() {
    // Width 4. Height 4. -> max_dim 4 -> stride_y 4.
    // Block size 64.
    // But total nodes = 4*4 = 16.
    // So only bits 0..15 are valid in block 0.

    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 4>::new(&mut arena, 4, 4, 1);

    // Initial state: One defect at index 0.
    let mut syndrome_words = vec![0u64; decoder.blocks_state.len()];
    syndrome_words[0] = 1; // Bit 0 set
    decoder.load_dense_syndromes(&syndrome_words);

    // Verify initial state
    assert_eq!(decoder.find(0), 0);
    // Occupied should have bit 0 set
    unsafe {
        assert_eq!(decoder.blocks_state.get_unchecked(0).occupied & 1, 1);
        assert_eq!(decoder.blocks_state.get_unchecked(0).occupied & (1 << 4), 0); // Not spread yet
    }

    // Force run process_block on block 0
    unsafe {
        decoder.process_block(0);
    }

    // Check occupied mask.
    // With stride 4, bit 0 should spread to 0, 4, 8, 12.
    // Indices 16+ are invalid.
    let occupied = unsafe { decoder.blocks_state.get_unchecked(0).occupied };

    // Construct expected mask for column 0
    let mut expected = 0u64;
    for i in 0..4 {
        expected |= 1 << (i * 4);
    }

    let col0_mask = expected; // 0x1111
    assert_eq!(
        occupied & col0_mask,
        expected,
        "Vertical diffusion did not fill column 0"
    );
}
