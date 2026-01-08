//! Unit tests for decoder correctness with various defect configurations.

use crate::common;

use prav_core::arena::Arena;
use prav_core::decoder::{DecodingState, EdgeCorrection};
use prav_core::topology::SquareGrid;

/// Runs a decoder test with given defects and verifies all defects are matched.
fn run_test_scenario<const STRIDE_Y: usize>(w: usize, h: usize, defects: &[u32], test_name: &str) {
    let mut memory = vec![0u8; 1024 * 1024 * 10]; // 10MB
    let mut arena = Arena::new(&mut memory);

    let mut decoder = DecodingState::<SquareGrid, STRIDE_Y>::new(&mut arena, w, h, 1);

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

    println!("Running Test: {}", test_name);

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
            --------------------------------------------------\n\
            Defects: {:?}\n\
            Unmatched: {:?}\n\
            --------------------------------------------------\n\n",
            test_name, defect_coords, remaining_coords
        );
    } else {
        println!("PASSED: {} ({} corrections)", test_name, count);
    }
}

#[test]
fn test_decoder_horizontal_pair() {
    let w = 10;
    let d1 = common::idx(2, 2, w, w);
    let d2 = common::idx(3, 2, w, w);
    run_test_scenario::<16>(w, w, &[d1, d2], "Horizontal Pair");
}

#[test]
fn test_decoder_vertical_pair_odd_distance() {
    let w = 10;
    let d1 = common::idx(2, 2, w, w);
    let d2 = common::idx(2, 4, w, w);
    run_test_scenario::<16>(w, w, &[d1, d2], "Vertical Pair Odd Distance");
}

#[test]
fn test_decoder_vertical_pair_even_distance() {
    let w = 10;
    let d1 = common::idx(2, 2, w, w);
    let d2 = common::idx(2, 5, w, w);
    run_test_scenario::<16>(w, w, &[d1, d2], "Vertical Pair Even Distance");
}

#[test]
fn test_decoder_boundary_adjacent_pairs() {
    let w = 10;
    let h = 10;
    let d1 = common::idx(0, 0, w, h);
    let d2 = common::idx(1, 0, w, h);
    run_test_scenario::<16>(w, h, &[d1, d2], "Boundary Huggers Top");

    let d3 = common::idx(w - 1, h - 1, w, h);
    let d4 = common::idx(w - 2, h - 1, w, h);
    run_test_scenario::<16>(w, h, &[d3, d4], "Boundary Huggers Bottom-Right");
}

#[test]
fn test_decoder_linear_chain_four_defects() {
    let w = 20;
    let d1 = common::idx(2, 2, w, w);
    let d2 = common::idx(4, 2, w, w);
    let d3 = common::idx(6, 2, w, w);
    let d4 = common::idx(8, 2, w, w);
    run_test_scenario::<32>(w, w, &[d1, d2, d3, d4], "Linear Chain 4-Defects");
}

#[test]
fn test_decoder_extreme_aspect_ratio_strip() {
    let w = 50;
    let h = 3;
    let d1 = common::idx(0, 1, w, h);
    let d2 = common::idx(49, 1, w, h);
    run_test_scenario::<64>(w, h, &[d1, d2], "Extreme Aspect Ratio Strip");
}


// This test verifies the end-to-end correctness of the refactored decoder.
// It creates a specific syndrome pattern and checks if the decoder produces
// the correct set of edge corrections. This validates both the implicit
// topology and the coarse block reset logic working together.
#[test]
fn test_refactored_decoder_correctness() {
    // 1. Setup
    // Use a small grid to make the test case manageable.
    let width = 8;
    let height = 8;
    let mut memory = vec![0u8; 1024 * 1024]; // 1MB should be plenty
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, width, height, 1);

    // 2. Create Syndromes
    // A simple pattern: two defects, separated by one empty node.
    // D - D (This comment says separated by one empty node? No, (3,3) and (4,3) are neighbors).
    // The previous code had (3,3) and (4,3). They are adjacent.
    // This should result in a single horizontal edge correction.
    let defect_1_x = 3;
    let defect_1_y = 3;
    let defect_2_x = 4;
    let defect_2_y = 3;

    // Linear encoding
    let stride_y = width.next_power_of_two(); // 8 -> 8
    let defect_1_idx = defect_1_y * stride_y + defect_1_x;
    let defect_2_idx = defect_2_y * stride_y + defect_2_x;

    let mut syndromes = vec![0u64; (width * height * 4 + 63) / 64]; // Allocation size approx
    // Note: decoder allocates based on pow2. 8*8 = 64 nodes.
    // syndromes length must match decoder's boundary length check.
    // Decoder uses (stride_y * height + 63)/64? No, dim_pow2 * dim_pow2.
    // 8*8 = 64. 1 block.
    // Just ensure syndromes vector is large enough. 2 blocks is safe.
    if defect_1_idx / 64 < syndromes.len() {
        syndromes[defect_1_idx / 64] |= 1 << (defect_1_idx % 64);
    }
    if defect_2_idx / 64 < syndromes.len() {
        syndromes[defect_2_idx / 64] |= 1 << (defect_2_idx % 64);
    }

    decoder.load_dense_syndromes(&syndromes);

    // 3. Decode
    let mut corrections = [EdgeCorrection::default(); 16];
    let count = decoder.decode(&mut corrections);

    // 4. Verify
    // The decoder might produce a path that is topologically valid but different from the
    // minimal direct edge (e.g., detour through vertical connections).
    // We verify that the produced corrections resolve the defects.

    let mut state = syndromes.clone();
    let final_corrections = &corrections[0..count];

    for c in final_corrections {
        let u = c.u as usize;
        let v = c.v as usize;

        let u_blk = u / 64;
        let u_bit = u % 64;
        if u_blk < state.len() {
            state[u_blk] ^= 1 << u_bit;
        }

        // v might be boundary
        if v != usize::MAX {
            let v_blk = v / 64;
            let v_bit = v % 64;
            if v_blk < state.len() {
                state[v_blk] ^= 1 << v_bit;
            }
        }
    }

    // Check if state is clean
    for (i, &word) in state.iter().enumerate() {
        assert_eq!(word, 0, "Block {} has remaining defects", i);
    }
}
