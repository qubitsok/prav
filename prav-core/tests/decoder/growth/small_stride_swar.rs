use prav_core::arena::Arena;
use prav_core::decoder::state::DecodingState;
use prav_core::topology::SquareGrid;

#[test]
fn test_complex_vertical_chain() {
    // 32x32 grid, stride 32.
    // Create a vertical chain 0 -> 32 -> 64 ...
    // Since we are testing intra-block, we care about 0 -> 32 within block 0 (if valid).
    // Block size 64. Stride 32.
    // Rows in block 0:
    // Row 0: 0..31
    // Row 1: 32..63
    // Row 2: 64.. (Block 1)

    // So intra-block vertical is only Row 0 <-> Row 1.

    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

    let mut syndrome_words = vec![0u64; decoder.blocks_state.len()];

    // Set bits 0, 32, 1, 33.
    // 0 and 32 are connected.
    // 1 and 33 are connected.
    // 0 and 1 are connected (horizontal).
    // So all 4 should be connected.

    let mask = (1 << 0) | (1 << 32) | (1 << 1) | (1 << 33);
    syndrome_words[0] = mask;

    decoder.load_dense_syndromes(&syndrome_words);
    decoder.grow_clusters();

    let r0 = decoder.find(0);
    let r32 = decoder.find(32);
    let r1 = decoder.find(1);
    let r33 = decoder.find(33);

    assert_eq!(r0, r32, "Vertical connection 0-32 failed");
    assert_eq!(r1, r33, "Vertical connection 1-33 failed");
    assert_eq!(r0, r1, "Horizontal connection 0-1 failed");
    assert_eq!(r32, r33, "Horizontal connection 32-33 failed");
}

#[test]
fn test_snake_pattern() {
    // Stride 8.
    // Row 0: 0..7
    // Row 1: 8..15
    // Row 2: 16..23
    // ...
    // Make a snake:
    // 0-1-2-3
    //       |
    // 11-10-9-8
    // |
    // 12-13-14-15

    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

    let mut mask = 0u64;
    // Row 0: 0,1,2,3
    mask |= (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3);
    // Vertical drop: 3 -> 11 (3 + 8 = 11)
    mask |= (1 << 3) | (1 << 11);
    // Row 1: 8,9,10,11
    mask |= (1 << 8) | (1 << 9) | (1 << 10) | (1 << 11);
    // Vertical drop: 8 -> 16 (8 + 8 = 16)
    mask |= (1 << 8) | (1 << 16);
    // Row 2: 16,17,18,19
    mask |= (1 << 16) | (1 << 17) | (1 << 18) | (1 << 19);

    let mut syndromes = vec![0u64; decoder.blocks_state.len()];
    syndromes[0] = mask;

    decoder.load_dense_syndromes(&syndromes);
    decoder.grow_clusters();

    let start = decoder.find(0);
    let end = decoder.find(19);

    assert_eq!(start, end, "Snake pattern should be fully connected");
}
