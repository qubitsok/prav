use prav_core::arena::Arena;
use prav_core::decoder::{DecodingState, EdgeCorrection};
use prav_core::topology::{Grid3D, HoneycombGrid, SquareGrid, Topology, TriangularGrid};

fn lin_idx(x: usize, y: usize, w: usize, h: usize) -> u32 {
    let max = w.max(h);
    let stride = max.next_power_of_two();
    (y * stride + x) as u32
}

fn lin_idx_3d(x: usize, y: usize, z: usize, w: usize, h: usize, d: usize) -> u32 {
    let max = w.max(h).max(d);
    let stride = max.next_power_of_two();
    let stride_z = stride * stride;
    (z * stride_z + y * stride + x) as u32
}

fn run_test_scenario<T: Topology, const STRIDE_Y: usize>(
    w: usize,
    h: usize,
    d: usize,
    defects: &[u32],
    test_name: &str,
) {
    let mut memory = vec![0u8; 1024 * 1024 * 10]; // 10MB
    let mut arena = Arena::new(&mut memory);

    let mut decoder = DecodingState::<T, STRIDE_Y>::new(&mut arena, w, h, d);

    // Convert defects to dense
    let is_3d = d > 1;
    let max_dim = w.max(h).max(if is_3d { d } else { 1 });
    let pow2 = max_dim.next_power_of_two();
    let total_nodes = if is_3d {
        pow2 * pow2 * pow2
    } else {
        pow2 * pow2
    };
    let num_blocks = (total_nodes + 63) / 64;
    let mut dense_defects = vec![0u64; num_blocks];

    for &def in defects {
        let blk = (def as usize) / 64;
        let bit = (def as usize) % 64;
        if blk < dense_defects.len() {
            dense_defects[blk] ^= 1 << bit;
        }
    }

    let mut corrections = vec![EdgeCorrection::default(); w * h * d * 8];

    println!("Running Test (Implicit Topology): {}", test_name);

    decoder.sparse_reset();
    decoder.load_dense_syndromes(&dense_defects);
    let count = decoder.decode(&mut corrections);

    // Minimal verification: We expect corrections to resolve defects.
    assert!(count > 0, "Expected corrections for {}", test_name);
}

#[test]
fn test_implicit_square_grid() {
    let w = 16;
    let h = 16;
    let d1 = lin_idx(2, 2, w, h);
    let d2 = lin_idx(3, 2, w, h);
    run_test_scenario::<SquareGrid, 16>(w, h, 1, &[d1, d2], "SquareGrid Pair");
}

#[test]
fn test_implicit_honeycomb() {
    let w = 16;
    let h = 16;
    let d1 = lin_idx(2, 2, w, h);
    // Honeycomb has different connectivity.
    let d2 = lin_idx(2, 3, w, h);
    run_test_scenario::<HoneycombGrid, 16>(w, h, 1, &[d1, d2], "HoneycombGrid Pair");
}

#[test]
fn test_implicit_triangular() {
    let w = 16;
    let h = 16;
    let d1 = lin_idx(2, 2, w, h);
    let d2 = lin_idx(3, 3, w, h);
    run_test_scenario::<TriangularGrid, 16>(w, h, 1, &[d1, d2], "TriangularGrid Pair");
}

#[test]
fn test_implicit_grid3d() {
    let w = 8;
    let h = 8;
    let d = 8;
    let d1 = lin_idx_3d(2, 2, 2, w, h, d);
    let d2 = lin_idx_3d(2, 2, 3, w, h, d);
    run_test_scenario::<Grid3D, 8>(w, h, d, &[d1, d2], "Grid3D Pair");
}
