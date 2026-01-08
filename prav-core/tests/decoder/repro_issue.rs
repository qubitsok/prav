//! Regression test for single defect handling.

use crate::common;

use prav_core::arena::Arena;
use prav_core::decoder::{DecodingState, EdgeCorrection};
use prav_core::topology::SquareGrid;

#[test]
fn test_single_defect_boundary_match() {
    let w = 20;
    let h = 20;
    let mut memory = vec![0u8; 1024 * 1024 * 5];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, w, h, 1);

    let max_dim = w.max(h);
    let pow2 = max_dim.next_power_of_two();
    let total_nodes = pow2 * pow2;
    let num_blocks = (total_nodes + 63) / 64;
    let mut dense_defects = vec![0u64; num_blocks];

    // Place a single defect at (10, 10)
    let stride = pow2;
    let idx = 10 * stride + 10;
    let blk = idx / 64;
    let bit = idx % 64;
    dense_defects[blk] ^= 1 << bit;

    decoder.load_dense_syndromes(&dense_defects);

    let mut corrections = vec![EdgeCorrection::default(); total_nodes * 4];
    let count = decoder.decode(&mut corrections);

    let result = common::verify_matching(&dense_defects, &corrections[0..count]);

    assert!(
        result.is_ok(),
        "Single defect should be resolved (matched to boundary), but got unmatched: {:?}",
        result.err()
    );
}
