use prav_core::arena::Arena;
use prav_core::decoder::state::DecodingState;
use prav_core::topology::SquareGrid;

#[test]
fn test_small_stride_vertical_boundary_crossing() {
    // Width 8, Height 16 -> max_dim 16 -> stride_y 16.
    // Block size 64.
    // Block 0: indices 0..63.
    // Block 1: indices 64..127.
    // Rows are 16-aligned.
    // y=0: 0..7
    // y=1: 16..23
    // y=2: 32..39
    // y=3: 48..55 (Top of connection)
    // y=4: 64..71 (Bottom of connection, in Block 1)

    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 8, 16, 1);

    // Defect at (0, 3) -> 48
    // Defect at (0, 4) -> 64
    // 64 is 4*16 + 0.

    let mut syndrome_words = vec![0u64; decoder.blocks_state.len()];
    // 48 is in block 0, bit 48.
    syndrome_words[0] |= 1 << 48;
    // 64 is in block 1, bit 0.
    syndrome_words[1] |= 1 << 0;

    decoder.load_dense_syndromes(&syndrome_words);

    // Before growth, they are separate roots.
    let root_a = decoder.find(48);
    let root_b = decoder.find(64);
    assert_ne!(root_a, root_b);

    // Grow
    decoder.grow_clusters();

    // After growth, they should be merged because they are vertical neighbors.
    let root_a_new = decoder.find(48);
    let root_b_new = decoder.find(64);
    assert_eq!(
        root_a_new, root_b_new,
        "Roots should be connected after growth across block boundary"
    );
}

#[test]
fn test_small_stride_intra_block_spread() {
    // Width 8. Stride 8.
    // Block 0: 0..63.
    // y=0: 0..7
    // y=1: 8..15
    // Defect at (0,0) -> 0.
    // Defect at (0,1) -> 8.
    // Adjacent in column 0.

    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

    let mut syndrome_words = vec![0u64; decoder.blocks_state.len()];
    syndrome_words[0] |= (1 << 0) | (1 << 8);

    decoder.load_dense_syndromes(&syndrome_words);

    let r1 = decoder.find(0);
    let r2 = decoder.find(8);
    assert_ne!(r1, r2);

    decoder.grow_clusters();

    let r1_new = decoder.find(0);
    let r2_new = decoder.find(8);
    assert_eq!(r1_new, r2_new, "Intra-block vertical spread failed");
}
