use prav_core::arena::Arena;
use prav_core::decoder::state::DecodingState;
use prav_core::topology::SquareGrid;

#[test]
fn test_active_block_mask_logic() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);

    // 32x32 grid -> 1024 nodes -> 16 blocks. Small grid (<= 64 blocks).
    let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, 32, 32, 1);

    assert!(decoder.is_small_grid());

    // Create syndromes: two distant errors to create two active blocks.
    // Block 0 (nodes 0..63) -> Error at node 0.
    // Block 10 (nodes 640..703) -> Error at node 640.
    let num_blocks = decoder.blocks_state.len();
    let mut syndromes = vec![0u64; num_blocks];

    syndromes[0] = 1;
    syndromes[10] = 1;

    decoder.load_dense_syndromes(&syndromes);

    // Verify active_block_mask
    // Should have bits 0 and 10 set.
    assert_eq!(decoder.active_block_mask, (1 << 0) | (1 << 10));
    assert_eq!(
        decoder.ingestion_count, 0,
        "Ingestion list should be skipped for small grids"
    );

    // Run growth
    decoder.grow_clusters();

    // Verify that growth happened
    // Verify boundary/occupied expansion.
    assert_ne!(
        decoder.blocks_state[0].boundary, 1,
        "Block 0 should have expanded"
    );
    assert_ne!(
        decoder.blocks_state[10].boundary, 1,
        "Block 10 should have expanded"
    );

    // Active mask should be 0 after growth finishes
    assert_eq!(decoder.active_block_mask, 0);
}

#[test]
fn test_active_block_mask_propagation() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);

    // 8x8 grid -> 64 nodes -> 1 block.
    // Not interesting for mask propagation.
    // 16x16 -> 256 nodes -> 4 blocks.
    // Block 0: 0..63 (Rows 0-3)
    // Block 1: 64..127 (Rows 4-7)
    // Block 2: 128..191 (Rows 8-11)
    // Block 3: 192..255 (Rows 12-15)

    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 16, 16, 1);

    // Error at bottom of Block 0 (node 63, which is x=15, y=3).
    // It should propagate to Block 1 (node 64+x) via vertical connection?
    // 16x16 grid. Stride = 16.
    // Node 63 (y=3, x=15). Neighbor South is 63 + 16 = 79.
    // 79 is in Block 1 (79 / 64 = 1).

    let num_blocks = decoder.blocks_state.len();
    let mut syndromes = vec![0u64; num_blocks];
    syndromes[0] = 1 << 63;

    decoder.load_dense_syndromes(&syndromes);

    assert_eq!(decoder.active_block_mask, 1);

    decoder.grow_clusters();

    // Check if Block 1 got activated and processed.
    // Node 79 should be occupied.
    let occ1 = decoder.blocks_state[1].occupied;
    let bit = 79 % 64;
    assert_eq!(
        occ1 & (1 << bit),
        1 << bit,
        "Growth should have propagated to Block 1"
    );
}
