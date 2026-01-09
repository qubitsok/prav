use alloc::format;
use alloc::vec::Vec;
use core::cmp::max;

use prav_core::{
    arena::Arena,
    decoder::{EdgeCorrection, TiledDecodingState},
    testing_grids::{GridConfig, TestGrids, ERROR_PROBS},
    topology::{HoneycombGrid, SquareGrid, Topology, TriangularGrid},
};

use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};

use crate::platforms::{BenchmarkHost, Platform};

// Number of cycles to run the benchmark for statistical significance.
#[cfg(target_os = "none")]
const CYCLES: usize = 100;
#[cfg(not(target_os = "none"))]
const CYCLES: usize = 10_000;

/// Represents the command-line argument for topology selection.
#[derive(Copy, Clone, PartialEq, Eq)]
#[cfg_attr(not(target_os = "none"), derive(clap::ValueEnum, Debug))]
pub enum TopologyArg {
    Square,
    Rectangle,
    Triangular,
    Honeycomb,
    All,
}

/// Helper function to generate a random boolean with probability `p`.
fn random_bool(rng: &mut ChaCha8Rng, p: f64) -> bool {
    let val = rng.next_u32();
    let float_val = (val as f64) / (u32::MAX as f64);
    float_val < p
}

/// Maps a "tiled" node index to a row-major index.
/// This is used because the decoder operates on a tiled memory layout for cache efficiency,
/// but verification is easier on a standard row-major grid.
fn map_tiled_to_row(u: u32, tiles_x: usize, stride_y: usize) -> usize {
    let u = u as usize;
    let tile_idx = u / 1024;
    let local_idx = u % 1024;
    let tx = tile_idx % tiles_x;
    let ty = tile_idx / tiles_x;
    let lx = local_idx % 32;
    let ly = local_idx / 32;
    let gx = tx * 32 + lx;
    let gy = ty * 32 + ly;
    gy * stride_y + gx
}

/// Verifies that the corrections produced by the decoder actually fix the defects.
/// It applies the corrections to the input syndrome and checks if all defects are cleared.
fn verify_matching(
    dense_input: &[u64],
    corrections: &[EdgeCorrection],
    buffer: &mut [u64],
    width: usize,
    stride_y: usize,
) -> usize {
    // Copy the input defects into a scratch buffer.
    buffer.copy_from_slice(dense_input);
    let tiles_x = width.div_ceil(32);

    // Apply each correction by flipping the bits at the endpoints of the matched edge.
    for c in corrections {
        let u = map_tiled_to_row(c.u, tiles_x, stride_y);
        let blk_u = u / 64;
        let bit_u = u % 64;
        if blk_u < buffer.len() {
            buffer[blk_u] ^= 1 << bit_u;
        }

        if c.v != u32::MAX {
            let v = map_tiled_to_row(c.v, tiles_x, stride_y);
            let blk_v = v / 64;
            let bit_v = v % 64;
            if blk_v < buffer.len() {
                buffer[blk_v] ^= 1 << bit_v;
            }
        }
    }
    // Count remaining defects (ones). Ideally, this should be 0.
    buffer.iter().map(|w| w.count_ones() as usize).sum()
}

/// Generates a random set of defects (errors) on the grid.
/// `p` is the probability of a defect occurring at any given node.
fn generate_defects(width: usize, height: usize, stride_y: usize, p: f64, seed: u64) -> Vec<u64> {
    let alloc_nodes = height * stride_y;
    let num_blocks = alloc_nodes.div_ceil(64);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut defects = alloc::vec![0u64; num_blocks];
    for y in 0..height {
        for x in 0..width {
            if random_bool(&mut rng, p) {
                let idx = y * stride_y + x;
                let blk = idx / 64;
                let bit = idx % 64;
                if blk < defects.len() {
                    defects[blk] |= 1 << bit;
                }
            }
        }
    }
    defects
}

/// Runs the benchmark for a specific topology and grid configuration.
fn run_benchmark<T: Topology>(
    name: &str,
    config: GridConfig,
    p: f64,
    defects_per_cycle: &[Vec<u64>],
    widths: (usize, usize, usize, usize),
) {
    // Allocation strategy:
    // On bare metal, we have very limited RAM (e.g., 256KB heap), so we use a small arena.
    // On host, we can afford a large arena to avoid re-allocations.
    #[cfg(target_os = "none")]
    let memory_size = 1024 * 128; // 128KB
    #[cfg(not(target_os = "none"))]
    let memory_size = 1024 * 1024 * 128; // 128MB

    let mut memory = alloc::vec![0u8; memory_size];
    let mut arena = Arena::new(&mut memory);

    // Initialize the decoder state.
    let mut decoder = TiledDecodingState::<T>::new(&mut arena, config.width, config.height);

    let stride_y = config.stride_y;
    let alloc_nodes = config.height * stride_y;
    let mut corrections = alloc::vec![EdgeCorrection::default(); alloc_nodes * 2];
    let num_blocks = alloc_nodes.div_ceil(64);
    let mut scratch_buffer = alloc::vec![0u64; num_blocks];

    let mut measurements = Vec::with_capacity(defects_per_cycle.len());
    let mut total_defects = 0;
    let mut total_remaining = 0;

    // Warmup phase: Run once without measuring to populate caches/TLB.
    if let Some(first_defects) = defects_per_cycle.first() {
        decoder.sparse_reset();
        decoder.load_dense_syndromes(first_defects);
        decoder.grow_clusters();
        decoder.peel_forest(&mut corrections);
    }

    // Main Benchmark Loop
    for defects in defects_per_cycle.iter() {
        let defects_count: usize = defects.iter().map(|x| x.count_ones() as usize).sum();
        total_defects += defects_count;

        let t0 = Platform::now();

        // 1. Reset: clear previous state (optimized for sparse graphs).
        decoder.sparse_reset();
        // 2. Load: inject the error syndromes into the graph.
        decoder.load_dense_syndromes(defects);
        // 3. Grow: run the Union-Find growth algorithm to find matching clusters.
        decoder.grow_clusters();
        // 4. Peel: extract the matching edges (corrections) from the clusters.
        let count = decoder.peel_forest(&mut corrections);

        let m = Platform::measure(t0);
        measurements.push(m);

        // Verify that the corrections are valid.
        let remaining = verify_matching(
            defects,
            &corrections[0..count],
            &mut scratch_buffer,
            config.width,
            stride_y,
        );
        total_remaining += remaining;
    }

    // Calculate statistics (Average, p50, p95, p99).
    measurements.sort_by(|a, b| a.micros.total_cmp(&b.micros));

    let len = measurements.len();
    let avg_us = if len > 0 {
        measurements.iter().map(|m| m.micros).sum::<f64>() / len as f64
    } else {
        0.0
    };
    let p50_us = if len > 0 {
        measurements[len * 50 / 100].micros
    } else {
        0.0
    };
    let p95_us = if len > 0 {
        measurements[len * 95 / 100].micros
    } else {
        0.0
    };
    let p99_us = if len > 0 {
        measurements[len * 99 / 100].micros
    } else {
        0.0
    };

    let solved_percent = if total_defects > 0 {
        (1.0 - (total_remaining as f64 / total_defects as f64)) * 100.0
    } else {
        100.0
    };

    let dims = format!("{}x{}", config.width, config.height);
    let nodes_str = format!("{} ({})", config.actual_nodes(), config.target_nodes);
    let clean_name = name.replace("Tiled-Growth-", "");
    let p_str = format!("{:.3}", p);

    let (w_shape, w_dims, w_nodes, w_p) = widths;

    // Fixed widths for data columns
    let w_avg = 8;
    let w_p50 = 8;
    let w_p95 = 8;
    let w_p99 = 8;
    let w_solve = 8;

    let msg = format!(
        "{:<w0$} | {:<w1$} | {:<w2$} | {:<w3$} | {:<w4$.2} | {:<w5$.2} | {:<w6$.2} | {:<w7$.2} | {:<w8$.2}",
        clean_name, dims, nodes_str, p_str, avg_us, p50_us, p95_us, p99_us, solved_percent,
        w0=w_shape, w1=w_dims, w2=w_nodes, w3=w_p, w4=w_avg, w5=w_p50, w6=w_p95, w7=w_p99, w8=w_solve
    );
    Platform::print(&msg);
}

/// Main entry point for the benchmark suite.
/// Iterates over configured grid sizes and probabilities, generating defects and running benchmarks.
pub fn run_suite(topo: TopologyArg, all_grids: bool) {
    // 1. Calculate formatting widths for the results table.
    let all_grids_data = TestGrids::all();
    let default_grids_data = TestGrids::defaults();

    let grids: &[GridConfig] = if all_grids {
        &all_grids_data
    } else {
        &default_grids_data
    };

    let mut w_shape = "Shape".len();
    let mut w_dims = "Dims".len();
    let mut w_nodes = "Nodes (Target)".len();
    let mut w_p = "p".len();

    for config in grids {
        if config.actual_nodes() > 4100 {
            #[cfg(target_os = "none")]
            continue;
        }

        if matches!(topo, TopologyArg::Square | TopologyArg::All) {
            w_shape = max(w_shape, "Square".len());
        }
        if matches!(topo, TopologyArg::Rectangle | TopologyArg::All) {
            w_shape = max(w_shape, "Rectangle".len());
            let r = config.to_rectangular(3.0);
            w_dims = max(w_dims, format!("{}x{}", r.width, r.height).len());
            w_nodes = max(
                w_nodes,
                format!("{} ({})", r.actual_nodes(), r.target_nodes).len(),
            );
        }
        if matches!(topo, TopologyArg::Triangular | TopologyArg::All) {
            w_shape = max(w_shape, "Triangular".len());
        }
        if matches!(topo, TopologyArg::Honeycomb | TopologyArg::All) {
            w_shape = max(w_shape, "Honeycomb".len());
        }

        w_dims = max(w_dims, format!("{}x{}", config.width, config.height).len());
        w_nodes = max(
            w_nodes,
            format!("{} ({})", config.actual_nodes(), config.target_nodes).len(),
        );
    }

    for &p in &ERROR_PROBS {
        w_p = max(w_p, format!("{:.3}", p).len());
    }

    let w_avg = 8;
    let w_p50 = 8;
    let w_p95 = 8;
    let w_p99 = 8;
    let w_solve = 8;

    // 2. Print Table Header
    let header = format!(
        "{:<w0$} | {:<w1$} | {:<w2$} | {:<w3$} | {:<w4$} | {:<w5$} | {:<w6$} | {:<w7$} | {:<w8$}",
        "Shape",
        "Dims",
        "Nodes (Target)",
        "p",
        "Avg(us)",
        "p50",
        "p95",
        "p99",
        "Solve%",
        w0 = w_shape,
        w1 = w_dims,
        w2 = w_nodes,
        w3 = w_p,
        w4 = w_avg,
        w5 = w_p50,
        w6 = w_p95,
        w7 = w_p99,
        w8 = w_solve
    );
    Platform::print(&header);
    Platform::print(&"-".repeat(header.len()));

    let widths = (w_shape, w_dims, w_nodes, w_p);

    // 3. Execute Benchmarks
    for &config in grids {
        // Skip large grids on bare metal to avoid Out of Memory (OOM) errors.
        #[cfg(target_os = "none")]
        if config.actual_nodes() > 4100 {
            continue;
        }

        for &p in &ERROR_PROBS {
            // Generate deterministic defects for consistency.
            let mut defects_list = Vec::with_capacity(CYCLES);
            let mut rng_seed = 0x12345678;
            for _ in 0..CYCLES {
                defects_list.push(generate_defects(
                    config.width,
                    config.height,
                    config.stride_y,
                    p,
                    rng_seed,
                ));
                rng_seed += 1;
            }

            if matches!(topo, TopologyArg::Square | TopologyArg::All) {
                run_benchmark::<SquareGrid>(
                    "Tiled-Growth-Square",
                    config,
                    p,
                    &defects_list,
                    widths,
                );
            }

            if matches!(topo, TopologyArg::Rectangle | TopologyArg::All) {
                let rect_config = config.to_rectangular(3.0);
                let mut rect_defects = Vec::with_capacity(CYCLES);
                let mut rect_seed = 0x12345678;
                for _ in 0..CYCLES {
                    rect_defects.push(generate_defects(
                        rect_config.width,
                        rect_config.height,
                        rect_config.stride_y,
                        p,
                        rect_seed,
                    ));
                    rect_seed += 1;
                }
                run_benchmark::<SquareGrid>(
                    "Tiled-Growth-Rectangle",
                    rect_config,
                    p,
                    &rect_defects,
                    widths,
                );
            }

            if matches!(topo, TopologyArg::Triangular | TopologyArg::All) {
                run_benchmark::<TriangularGrid>(
                    "Tiled-Growth-Triangular",
                    config,
                    p,
                    &defects_list,
                    widths,
                );
            }
            if matches!(topo, TopologyArg::Honeycomb | TopologyArg::All) {
                run_benchmark::<HoneycombGrid>(
                    "Tiled-Growth-Honeycomb",
                    config,
                    p,
                    &defects_list,
                    widths,
                );
            }
        }
    }
}
