use prav_core::arena::Arena;
use prav_core::decoder::{DecodingState, EdgeCorrection, Peeling};
use prav_core::topology::SquareGrid;

#[test]
fn test_trait_exports() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

    // Test UnionFind trait through re-exports
    let root = decoder.find(0);
    assert_eq!(root, 0);

    // Test ClusterGrowth trait through re-exports
    decoder.sparse_reset();
    let syndromes = [0u64; 1];
    decoder.load_dense_syndromes(&syndromes);
    decoder.grow_clusters();

    // Test Peeling trait through re-exports
    let mut corrections = [EdgeCorrection::default(); 10];
    let count = decoder.peel_forest(&mut corrections);
    assert_eq!(count, 0);
}

#[test]
fn test_block_processing_dispatch() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);

    // Small stride (< 64)
    let mut decoder_small = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);
    unsafe {
        decoder_small.process_block(0);
    }

    // Large stride (>= 64) - SquareGrid with width 64
    let mut arena2 = Arena::new(&mut memory);
    let mut decoder_large = DecodingState::<SquareGrid, 64>::new(&mut arena2, 64, 64, 1);
    unsafe {
        decoder_large.process_block(0);
    }
}

#[test]
fn test_coordinate_consistency() {
    let mut memory = vec![0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 10, 10, 1);

    let u = 0;
    let (x, y, z) = decoder.get_coord(u);
    assert_eq!(x, 0);
    assert_eq!(y, 0);
    assert_eq!(z, 0);

    let stride_y = 16; // next power of 2
    let u2 = (5 * stride_y + 3) as u32;
    let (x2, y2, z2) = decoder.get_coord(u2);
    assert_eq!(x2, 3);
    assert_eq!(y2, 5);
    assert_eq!(z2, 0);
}
