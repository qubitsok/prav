//! Property-based tests for decoder correctness with varying grid sizes.

use crate::common;

use prav_core::arena::Arena;
use prav_core::decoder::{DecodingState, EdgeCorrection};
use prav_core::topology::SquareGrid;
use proptest::prelude::*;

proptest! {
    /// Property test for decoder with variable grid sizes.
    ///
    /// Tests decoder correctness across a range of grid dimensions (3-50)
    /// and random defect configurations.
    #[test]
    fn prop_decoder_square_grid(
        w in 3usize..50,
        h in 3usize..50,
        defects_input in proptest::collection::vec((0usize..1000, 0usize..1000), 0..50)
    ) {
        // Map abstract defects to grid coordinates
        let defects: Vec<(usize, usize)> = defects_input.iter()
            .map(|(x, y)| (x % w, y % h))
            .collect();

        let success = run_prop_test_dispatch(w, h, defects);
        prop_assert!(success, "Decoder failed to resolve defects for {}x{} grid", w, h);
    }

    /// Property test for medium-sized sparse grid (60x60).
    ///
    /// Tests decoder correctness with randomly generated defect positions
    /// on a 60x60 grid with up to 20 defects.
    #[test]
    fn prop_decoder_medium_sparse_grid(
        defects in proptest::collection::vec((0usize..60, 0usize..60), 0..20)
    ) {
        prop_assert!(
            run_prop_test_scenario::<64>(60, 60, defects),
            "Decoder failed to resolve defects on 60x60 grid"
        );
    }

    /// Property test for odd-sized grid (33x33).
    ///
    /// Tests decoder correctness with randomly generated defect positions
    /// on a 33x33 grid (non-power-of-2) with up to 50 defects.
    #[test]
    fn prop_decoder_odd_sized_grid(
        defects in proptest::collection::vec((0usize..33, 0usize..33), 0..50)
    ) {
        prop_assert!(
            run_prop_test_scenario::<64>(33, 33, defects),
            "Decoder failed to resolve defects on 33x33 grid"
        );
    }

    /// Property test for small dense grid (20x20).
    ///
    /// Tests decoder correctness with randomly generated defect positions
    /// on a 20x20 grid with up to 40 defects (higher density).
    #[test]
    fn prop_decoder_small_dense_grid(
        defects in proptest::collection::vec((0usize..20, 0usize..20), 0..40)
    ) {
        prop_assert!(
            run_prop_test_scenario::<32>(20, 20, defects),
            "Decoder failed to resolve defects on 20x20 grid"
        );
    }
}

fn run_prop_test_dispatch(w: usize, h: usize, defects: Vec<(usize, usize)>) -> bool {
    let max_dim = w.max(h);
    let stride_y = max_dim.next_power_of_two();

    match stride_y {
        4 => run_prop_test_scenario::<4>(w, h, defects),
        8 => run_prop_test_scenario::<8>(w, h, defects),
        16 => run_prop_test_scenario::<16>(w, h, defects),
        32 => run_prop_test_scenario::<32>(w, h, defects),
        64 => run_prop_test_scenario::<64>(w, h, defects),
        _ => panic!("Unsupported stride: {}", stride_y),
    }
}

fn run_prop_test_scenario<const STRIDE_Y: usize>(
    w: usize,
    h: usize,
    defects: Vec<(usize, usize)>,
) -> bool {
    let mut memory = vec![0u8; 1024 * 1024 * 50]; // Increased memory for larger grids
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, STRIDE_Y>::new(&mut arena, w, h, 1);

    // Linear setup
    let stride = STRIDE_Y;
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

    decoder.sparse_reset(); // Ensure clean state
    decoder.load_dense_syndromes(&dense_defects);

    // corrections buffer
    let mut corrections = vec![EdgeCorrection::default(); total_nodes * 4];
    let count = decoder.decode(&mut corrections);

    common::verify_matching_bool(&dense_defects, &corrections[0..count])
}