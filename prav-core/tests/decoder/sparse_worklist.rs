use prav_core::arena::Arena;
use prav_core::decoder::state::DecodingState;
use prav_core::topology::SquareGrid;

#[test]
fn test_sparse_initialization_counts() {
    let width = 64;
    let height = 64;
    let buffer_size = 1024 * 1024 * 32;
    let mut buffer = vec![0u8; buffer_size];
    let mut arena = Arena::new(&mut buffer);

    let mut decoder = DecodingState::<SquareGrid, 64>::new(&mut arena, width, height, 1);

    // Inject two defects far apart
    let mut syndromes = vec![0u64; decoder.blocks_state.len()];

    // Block 0, bit 0
    syndromes[0] = 1;
    // Block 10, bit 0
    syndromes[10] = 1;

    decoder.load_dense_syndromes(&syndromes);

    // 64x64 (4096 nodes) is treated as a "small grid" and uses `active_block_mask`.
    assert_ne!(decoder.active_block_mask & 1, 0, "Block 0 should be active");
    assert_ne!(
        decoder.active_block_mask & (1 << 10),
        0,
        "Block 10 should be active"
    );
}

#[test]
fn test_sparse_growth_cycle() {
    let width = 64;
    let height = 64;
    let buffer_size = 1024 * 1024 * 32;
    let mut buffer = vec![0u8; buffer_size];
    let mut arena = Arena::new(&mut buffer);

    let mut decoder = DecodingState::<SquareGrid, 64>::new(&mut arena, width, height, 1);

    // Two defects close to each other: (0,0) and (2,0) -> distance 2
    // Block 0 covers x=0..63.
    // They are in the same block (Block 0).

    let mut syndromes = vec![0u64; decoder.blocks_state.len()];
    // bit 0 and bit 2
    syndromes[0] = (1 << 0) | (1 << 2);

    decoder.load_dense_syndromes(&syndromes);

    assert_ne!(decoder.active_block_mask & 1, 0, "Block 0 should be active");

    // Run growth
    decoder.grow_clusters();

    // They should have merged.
    let root0 = decoder.find(0);
    let root2 = decoder.find(2);
    assert_eq!(root0, root2, "Clusters should merge");
}
