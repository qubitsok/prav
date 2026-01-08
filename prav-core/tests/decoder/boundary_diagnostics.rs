// tests/boundary_diagnostics.rs
use prav_core::arena::Arena;
use prav_core::decoder::{DecodingState, EdgeCorrection};
use prav_core::topology::SquareGrid;

fn run_test<const STRIDE_Y: usize>(w: usize, h: usize, defects: &[u32], test_name: &str) {
    let mut memory = vec![0u8; 1024 * 1024 * 10];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = DecodingState::<SquareGrid, STRIDE_Y>::new(&mut arena, w, h, 1);

    let mut corrections = vec![EdgeCorrection::default(); w * h * 20];

    // Pack defects
    let max_dim = w.max(h);
    let pow2 = max_dim.next_power_of_two();
    let total_nodes = pow2 * pow2;
    let num_blocks = (total_nodes + 63) / 64;
    let mut dense_defects = vec![0u64; num_blocks];

    for &d in defects {
        let blk = (d as usize) / 64;
        let bit = (d as usize) % 64;
        if blk < dense_defects.len() {
            dense_defects[blk] ^= 1 << bit;
        }
    }

    println!("EXPANSION TEST: {}", test_name);

    decoder.load_dense_syndromes(&dense_defects);
    let count = decoder.decode(&mut corrections);
    let final_corrections = &corrections[0..count];

    println!("  Corrections: {}", count);

    // Verify
    let mut state = dense_defects.clone();
    for c in final_corrections {
        let u = c.u as usize;
        let v = c.v as usize;
        let u_blk = u / 64;
        let u_bit = u % 64;
        if u_blk < state.len() {
            state[u_blk] ^= 1 << u_bit;
        }
        let v_blk = v / 64;
        let v_bit = v % 64;
        if v_blk < state.len() {
            state[v_blk] ^= 1 << v_bit;
        }
    }

    let mut remaining = Vec::new();
    for (blk_idx, &word) in state.iter().enumerate() {
        if word != 0 {
            let mut w = word;
            let base = blk_idx * 64;
            while w != 0 {
                let b = w.trailing_zeros();
                w &= w - 1;
                remaining.push(base + b as usize);
            }
        }
    }

    if !remaining.is_empty() {
        // Reverse map for diagnostics
        let stride_y = pow2;
        let remaining_coords: Vec<(usize, usize)> = remaining
            .iter()
            .map(|&idx| {
                let y = idx / stride_y;
                let x = idx % stride_y;
                (x, y)
            })
            .collect();

        panic!(
            "FAILED: {}\nUnmatched: {:?}\nTotal Corrections: {}",
            test_name, remaining_coords, count
        );
    } else {
        println!("  PASSED ({} steps)", count);
    }
}

fn idx(x: usize, y: usize, w: usize) -> u32 {
    let h = 3;
    let stride_y = w.max(h).next_power_of_two();
    (y * stride_y + x) as u32
}

// --- TILE BOUNDARY HOP TESTS (Horizontal) ---
#[test]
fn test_hop_tile_0_to_1() {
    let w = 50;
    let h = 3;
    run_test::<64>(w, h, &[idx(7, 1, w), idx(8, 1, w)], "Hop Tile 0->1 (7-8)");
}

#[test]
fn test_hop_tile_1_to_2() {
    let w = 50;
    let h = 3;
    run_test::<64>(
        w,
        h,
        &[idx(15, 1, w), idx(16, 1, w)],
        "Hop Tile 1->2 (15-16)",
    );
}

#[test]
fn test_hop_tile_2_to_3() {
    let w = 50;
    let h = 3;
    run_test::<64>(
        w,
        h,
        &[idx(23, 1, w), idx(24, 1, w)],
        "Hop Tile 2->3 (23-24)",
    );
}

#[test]
fn test_hop_tile_3_to_4() {
    let w = 50;
    let h = 3;
    run_test::<64>(
        w,
        h,
        &[idx(31, 1, w), idx(32, 1, w)],
        "Hop Tile 3->4 (31-32)",
    );
}

// --- SEGMENT TESTS ---

#[test]
fn test_segment_left_half() {
    let w = 50;
    let h = 3;
    run_test::<64>(
        w,
        h,
        &[idx(0, 1, w), idx(31, 1, w)],
        "Segment Left (0 to 31)",
    );
}

#[test]
fn test_segment_right_half() {
    let w = 50;
    let h = 3;
    run_test::<64>(
        w,
        h,
        &[idx(32, 1, w), idx(49, 1, w)],
        "Segment Right (32 to 49)",
    );
}

#[test]
fn test_segment_crossing_one_tile_gap() {
    let w = 50;
    let h = 3;
    run_test::<64>(w, h, &[idx(0, 1, w), idx(16, 1, w)], "Span 2 Tiles (0-16)");
}

#[test]
fn test_segment_crossing_two_tile_gap() {
    // Original name typo fix? No, was two_tile_gaps
    let w = 50;
    let h = 3;
    run_test::<64>(w, h, &[idx(0, 1, w), idx(24, 1, w)], "Span 3 Tiles (0-24)");
}

#[test]
fn test_vertical_propagation_within_strip() {
    let w = 50;
    let h = 3;
    run_test::<64>(w, h, &[idx(0, 0, w), idx(7, 2, w)], "Tile 0 Diagonal");
}

#[test]
fn test_boundary_stalling() {
    let w = 50;
    let h = 3;
    run_test::<64>(
        w,
        h,
        &[idx(30, 1, w), idx(33, 1, w)],
        "Stall Check (30 to 33)",
    );
}
