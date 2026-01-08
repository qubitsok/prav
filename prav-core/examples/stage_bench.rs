use clap::{Parser, ValueEnum};
use prav_core::{
    arena::Arena, decoder::TiledDecodingState, decoder::EdgeCorrection,
    testing_grids::{GridConfig, TestGrids, ERROR_PROBS},
    topology::{SquareGrid, TriangularGrid, HoneycombGrid, Topology},
};
use rand::prelude::*;
use rand::rngs::StdRng;
use std::time::{Duration, Instant};

const CYCLES: usize = 10_000;

#[derive(Parser)]
struct Cli {
    #[arg(short, long, value_enum, default_value_t = TopologyArg::Square)]
    topology: TopologyArg,

    #[arg(long, default_value_t = false)]
    all_grids: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum TopologyArg {
    Square,
    Rectangle,
    Triangular,
    Honeycomb,
    All,
}

fn run_benchmark<T: Topology>(config: GridConfig, error_prob: f64, topo_name: &str) {
    // Use config dimensions
    let w = config.width;
    let h = config.height;
    let seed = 0xDEADBEEF;

    let stride_y = config.stride_y;
    let total_nodes = stride_y * stride_y; // stride_y is next power of two of max dim
    let num_blocks = (total_nodes + 63) / 64;

    let mut memory = vec![0u8; 1024 * 1024 * 512];
    let mut arena = Arena::new(&mut memory);

    // We use TiledDecodingState for scalable benchmarking
    let mut decoder = TiledDecodingState::<T>::new(&mut arena, w, h);

    let mut corrections = vec![EdgeCorrection::default(); total_nodes * 2];

    let mut rng = StdRng::seed_from_u64(seed);
    println!(
        "--- [{} - Grid {}x{} (~{} nodes, Stride {})] Target Nodes: {}, Error Probability: {} ---",
        topo_name, config.width, config.height, config.target_nodes, config.stride_y,
        config.target_nodes, error_prob
    );
    // Pre-generate defects to avoid benching RNG
    let mut scenarios = Vec::with_capacity(100);
    for _ in 0..100 {
        let mut def = vec![0u64; num_blocks];

        for y in 0..h {
            for x in 0..w {
                if rng.random_bool(error_prob) {
                    let idx = (y * stride_y) + x;
                    let blk = idx / 64;
                    let bit = idx % 64;
                    if blk < def.len() {
                        def[blk] |= 1 << bit;
                    }
                }
            }
        }
        scenarios.push(def);
    }

    let mut t_reset = Duration::new(0, 0);
    let mut t_load = Duration::new(0, 0);
    let mut t_grow = Duration::new(0, 0);
    let mut t_trace = Duration::new(0, 0);
    let mut t_compact = Duration::new(0, 0);

    // let _boundary_node = (decoder.parents.len() - 1) as u32;

    for i in 0..CYCLES {
        let defects = &scenarios[i % scenarios.len()];

        let t0 = Instant::now();
        decoder.sparse_reset();
        t_reset += t0.elapsed();

        let t1 = Instant::now();
        decoder.load_dense_syndromes(defects);
        t_load += t1.elapsed();

        let t2 = Instant::now();
        decoder.grow_clusters();
        t_grow += t2.elapsed();

        let t3 = Instant::now();
        // Manually implement peel_forest to split timing
        let mut _count = 0;
        // Trace Phase
        /*
        // TiledDecodingState has different internals, so manual tracing is commented out
        */
        _count = decoder.peel_forest(&mut corrections);
        t_trace += t3.elapsed();

        let t4 = Instant::now();
        // Compact Phase
        // decoder.reconstruct_corrections(&mut corrections);
        t_compact += t4.elapsed();
    }

    let avg_reset = t_reset.as_secs_f64() * 1e6 / CYCLES as f64;
    let avg_load = t_load.as_secs_f64() * 1e6 / CYCLES as f64;
    let avg_grow = t_grow.as_secs_f64() * 1e6 / CYCLES as f64;
    let avg_trace = t_trace.as_secs_f64() * 1e6 / CYCLES as f64;
    let avg_compact = t_compact.as_secs_f64() * 1e6 / CYCLES as f64;

    let total = avg_reset + avg_load + avg_grow + avg_trace + avg_compact;

    // Visual Bar Chart Logic
    let bar_width: usize = 50;
    let p_reset = avg_reset / total;
    let p_load = avg_load / total;
    let p_grow = avg_grow / total;
    let p_trace = avg_trace / total;
    let p_compact = avg_compact / total;

    let w_reset = (p_reset * bar_width as f64).round() as usize;
    let w_load = (p_load * bar_width as f64).round() as usize;
    let w_grow = (p_grow * bar_width as f64).round() as usize;
    let w_trace = (p_trace * bar_width as f64).round() as usize;
    // Fill the rest with compact to ensure exact width
    let w_compact = bar_width.saturating_sub(w_reset + w_load + w_grow + w_trace);

    let bar = format!(
        "{}{}{}{}{}",
        "█".repeat(w_grow),   // Grow - Heavy work
        "▓".repeat(w_trace),  // Trace
        "▒".repeat(w_load),   // Load
        "░".repeat(w_reset),  // Reset
        ":".repeat(w_compact) // Compact
    );

    println!(
        "Target: {:<6} | P(e): {:<6} | Total: {:.2} us",
        config.target_nodes, error_prob, total
    );
    println!("[{}]", bar);
    println!(
        " Grow ({:.0}%) · Trace ({:.0}%) · Load ({:.0}%) · Reset ({:.0}%) · Compact ({:.0}%)",
        p_grow * 100.0,
        p_trace * 100.0,
        p_load * 100.0,
        p_reset * 100.0,
        p_compact * 100.0
    );
    println!();
}

fn run_suite<const STRIDE_Y: usize>(config: GridConfig, topo: TopologyArg) {
    for &error_prob in &ERROR_PROBS {
        if matches!(topo, TopologyArg::Square | TopologyArg::All) {
            run_benchmark::<SquareGrid>(config, error_prob, "Square");
        }
        
        if matches!(topo, TopologyArg::Rectangle | TopologyArg::All) {
            let rect_config = config.to_rectangular(3.0);
            run_benchmark::<SquareGrid>(rect_config, error_prob, "Rectangle");
        }
        
        if matches!(topo, TopologyArg::Triangular | TopologyArg::All) {
            run_benchmark::<TriangularGrid>(config, error_prob, "Triangular");
        }
        if matches!(topo, TopologyArg::Honeycomb | TopologyArg::All) {
            run_benchmark::<HoneycombGrid>(config, error_prob, "Honeycomb");
        }
    }
}

fn main() {
    let args = Cli::parse();
    println!("[Micro-Benchmark] Staged Breakdown");

    // Small grids with stride 32
    run_suite::<32>(TestGrids::TINY, args.topology);
    run_suite::<32>(TestGrids::SMALL, args.topology);
    run_suite::<32>(TestGrids::MEDIUM, args.topology);

    // Large grid with stride 64
    run_suite::<64>(TestGrids::LARGE, args.topology);

    if args.all_grids {
        // Large+ grid with stride 128
        run_suite::<128>(TestGrids::LARGE_PLUS, args.topology);

        // Extra large grids with stride 512
        run_suite::<512>(TestGrids::XLARGE, args.topology);
        run_suite::<512>(TestGrids::XXLARGE, args.topology);
    }
}
