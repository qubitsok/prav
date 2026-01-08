//! Tests for growth stagnation detection and recovery.

use crate::common;

use prav_core::arena::Arena;
use prav_core::decoder::state::{DecodingState, EdgeCorrection};
use prav_core::topology::SquareGrid;

#[test]
fn test_stagnation_logic() {
    let mut memory = vec![0u8; 1024 * 1024 * 5]; // 5MB
    let mut arena = Arena::new(&mut memory);

    // 10x10 grid
    let mut decoder = DecodingState::<SquareGrid, 16>::new(&mut arena, 10, 10, 1);

    // Set up two defects that should merge
    // For 10x10, stride_y is 16.
    // Defect at (2, 1) -> index 1*16 + 2 = 18
    // Defect at (4, 1) -> index 1*16 + 4 = 20
    let defect1 = 18;
    let defect2 = 20;

    // Block size is 64. 18 and 20 are in block 0.
    let mut syndromes = vec![0u64; decoder.blocks_state.len()];
    syndromes[0] |= 1 << defect1;
    syndromes[0] |= 1 << defect2;

    decoder.load_dense_syndromes(&syndromes);

    // Use decode which calls grow_clusters internally
    let mut corrections = vec![EdgeCorrection::default(); 100];
    let count = decoder.decode(&mut corrections);

    let final_corrections = &corrections[0..count];
    let result = common::verify_matching(&syndromes, final_corrections);

    if let Err(remaining) = result {
        panic!("Failed to match defects. Remaining: {:?}", remaining);
    }
}
