//! Tests for small grid optimization path in the decoder.

use crate::common;

use prav_core::arena::Arena;
use prav_core::decoder::{DecodingState, EdgeCorrection};
use prav_core::topology::SquareGrid;

/// Runs a small-grid decoder test and verifies correctness.
fn run_test_scenario<const STRIDE_Y: usize>(w: usize, h: usize, defects: &[u32], test_name: &str) {
    let mut memory = vec![0u8; 1024 * 1024 * 10]; // 10MB
    let mut arena = Arena::new(&mut memory);

    let mut decoder = DecodingState::<SquareGrid, STRIDE_Y>::new(&mut arena, w, h, 1);

    // Ensure we are hitting the small grid path
    if !decoder.is_small_grid() {
        panic!(
            "Test {} intended for small grid but is_small_grid() returned false!",
            test_name
        );
    }

    // Convert defects to dense
    let max_dim = w.max(h);
    let pow2 = max_dim.next_power_of_two();
    let stride_y = pow2;
    let total_nodes = stride_y * stride_y;

    let num_blocks = (total_nodes + 63) / 64;
    let mut dense_defects = vec![0u64; num_blocks];

    for &d in defects {
        let blk = (d as usize) / 64;
        let bit = (d as usize) % 64;
        if blk < dense_defects.len() {
            dense_defects[blk] ^= 1 << bit;
        }
    }

    let mut corrections = vec![EdgeCorrection::default(); w * h * 8];

    decoder.sparse_reset();
    decoder.load_dense_syndromes(&dense_defects);
    let count = decoder.decode(&mut corrections);
    let final_corrections = &corrections[0..count];

    let result = common::verify_matching(&dense_defects, final_corrections);

    if let Err(remaining_syndromes) = result {
        let remaining_coords: Vec<(usize, usize)> = remaining_syndromes
            .iter()
            .map(|&idx| {
                let y = idx / stride_y;
                let x = idx % stride_y;
                (x, y)
            })
            .collect();

        let defect_coords: Vec<(usize, usize)> = defects
            .iter()
            .map(|&idx| {
                let y = (idx as usize) / stride_y;
                let x = (idx as usize) % stride_y;
                (x, y)
            })
            .collect();

        panic!(
            "\n\nFAILED: {}\n\
            Defects: {:?}\n\
            Unmatched: {:?}\n\
            ",
            test_name, defect_coords, remaining_coords
        );
    }
}

#[test]
fn test_small_grid_explicit_pair() {
    let w = 8;
    let h = 8;
    // Stride 8. 8x8=64 nodes. Fits in 2 blocks (u64)? No, 64 nodes fits in 1 block + sentinel = 2 blocks.
    // DecodingState::new(8,8) => alloc_size = 64. alloc_nodes = 65. num_blocks = 2.
    // boundary.len() = 2 <= 64. So it is small grid.

    let d1 = common::idx(2, 2, w, h);
    let d2 = common::idx(4, 2, w, h);
    run_test_scenario::<8>(w, h, &[d1, d2], "Small Grid Explicit Pair");
}

#[test]
fn test_tiny_grid_2x2() {
    // 2x2 grid smallest practical size
    let w = 2;
    let h = 2;
    let d1 = common::idx(0, 0, w, h);
    let d2 = common::idx(1, 0, w, h);
    run_test_scenario::<2>(w, h, &[d1, d2], "Tiny Grid 2x2");
}
