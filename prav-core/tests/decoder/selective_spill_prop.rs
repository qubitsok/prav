use crate::common;

use prav_core::arena::Arena;
use prav_core::decoder::{DecodingState, EdgeCorrection};
use prav_core::topology::SquareGrid;
use proptest::prelude::*;

fn run_prop_test_scenario_32(w: usize, h: usize, defects: Vec<(usize, usize)>) -> bool {
    let mut memory = vec![0u8; 1024 * 1024 * 10];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 32>::new(&mut arena, w, h, 1);

    let stride = 32;
    let total_nodes = stride * stride;

    let num_blocks = (total_nodes + 63) / 64;
    let mut dense_defects = vec![0u64; num_blocks];

    for (x, y) in defects {
        let idx = y * stride + x;
        let blk = idx / 64;
        let bit = idx % 64;
        if blk < dense_defects.len() {
            dense_defects[blk] ^= 1 << bit;
        }
    }

    decoder.load_dense_syndromes(&dense_defects);

    let mut corrections = vec![EdgeCorrection::default(); total_nodes * 4];
    let count = decoder.decode(&mut corrections);

    common::verify_matching_bool(&dense_defects, &corrections[0..count])
}

proptest! {
    /// Property test for decoder on 32x32 grid.
    ///
    /// Tests decoder correctness with randomly generated defect positions
    /// on a 32x32 grid.
    #[test]
    #[allow(unused_variables)]
    fn prop_decoder_32x32_grid(
        defects_input in proptest::collection::vec((0usize..32, 0usize..32), 0..20)
    ) {
         let w = 32;
         let h = 32;
         let defects: Vec<(usize, usize)> = defects_input.iter()
             .map(|(x, y)| (*x % w, *y % h))
             .collect();

         let success = run_prop_test_scenario_32(w, h, defects);
         prop_assert!(success, "Decoder failed to resolve defects on 32x32 grid");
    }
}
