use clap::{Parser, ValueEnum};
use prav_core::arena::Arena;
use prav_core::decoder::EdgeCorrection;
use prav_core::testing_grids::{GridConfig, TestGrids, ERROR_PROBS};
use prav_core::topology::{SquareGrid, TriangularGrid, HoneycombGrid, Topology};
use rand::prelude::*;
use rand::rngs::StdRng;
use std::time::Instant;

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

fn generate_defects(width: usize, height: usize, stride_y: usize, p: f64, seed: u64) -> Vec<u64> {
    let alloc_size = height * stride_y;
    let alloc_nodes = alloc_size + 1;
    let num_blocks = (alloc_nodes + 63) / 64;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut defects = vec![0u64; num_blocks];

    for y in 0..height {
        for x in 0..width {
            if rng.random_bool(p) {
                let idx = y * stride_y + x;
                let blk = idx / 64;
                let bit = idx % 64;
                defects[blk] |= 1 << bit;
            }
        }
    }
    defects
}

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

fn verify_matching_tiled(
    dense_input: &[u64],
    corrections: &[EdgeCorrection],
    buffer: &mut [u64],
    width: usize,
    stride_y: usize,
) -> usize {
    buffer.copy_from_slice(dense_input);
    let tiles_x = (width + 31) / 32;

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
    buffer.iter().map(|w| w.count_ones() as usize).sum()
}

fn run_tiled_growth_bench<T: Topology>(
    name: &str,
    config: GridConfig,
    p: f64,
    defects_per_cycle: &[Vec<u64>],
    widths: (usize, usize, usize, usize),
) {
    let width = config.width;
    let height = config.height;
    let mut memory = vec![0u8; 1024 * 1024 * 512];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = prav_core::decoder::TiledDecodingState::<T>::new(&mut arena, width, height);

    // Correction buffer
    let stride_y = width.max(height).next_power_of_two();
    let alloc_size = stride_y * stride_y;
    let mut corrections = vec![EdgeCorrection::default(); alloc_size * 2];

    // Scratch buffer for verification
    let defects_alloc_nodes = height * stride_y + 1;
    let num_blocks = (defects_alloc_nodes + 63) / 64;
    let mut scratch_buffer = vec![0u64; num_blocks];

    let mut total_duration = std::time::Duration::default();
    let mut total_grow_duration = std::time::Duration::default();
    let mut total_peel_duration = std::time::Duration::default();
    let mut total_remaining = 0;

    for defects in defects_per_cycle.iter() {
        let t0 = Instant::now();
        decoder.sparse_reset();
        decoder.load_dense_syndromes(defects);

        let grow_start = Instant::now();
        decoder.grow_clusters();
        let grow_end = Instant::now();
        total_grow_duration += grow_end.duration_since(grow_start);

        let peel_start = Instant::now();
        let count = decoder.peel_forest(&mut corrections);
        let peel_end = Instant::now();
        total_peel_duration += peel_end.duration_since(peel_start);

        // Accumulate total decoding time (Reset + Load + Grow + Peel)
        total_duration += peel_end.duration_since(t0);

        let remaining = verify_matching_tiled(
            defects,
            &corrections[0..count],
            &mut scratch_buffer,
            width,
            stride_y,
        );
        total_remaining += remaining;

        if remaining > 0 && name == "Small-Grid-AVX" && stride_y == 32 {
            println!("FAILURE: Small-Grid-AVX 32x32 | Remaining: {}", remaining);
            println!("Defects: {:?}", defects);
        }
    }

    let avg = total_duration.as_secs_f64() * 1e6 / defects_per_cycle.len() as f64;
    let avg_grow = total_grow_duration.as_secs_f64() * 1e6 / defects_per_cycle.len() as f64;
    let avg_peel = total_peel_duration.as_secs_f64() * 1e6 / defects_per_cycle.len() as f64;
    let avg_remaining = total_remaining as f64 / defects_per_cycle.len() as f64;

    let clean_name = name.replace("Tiled-Growth-", "");
    let dims = format!("{}x{}", config.width, config.height);
    let nodes_str = format!("{} ({})", config.actual_nodes(), config.target_nodes);
    let p_str = format!("{:.3}", p);

    let (w_shape, w_dims, w_nodes, w_p) = widths;
    let w_avg = 8;
    let w_grow = 8;
    let w_peel = 8;
    let w_rem = 8;

    println!(
        "{:<w0$} | {:<w1$} | {:<w2$} | {:<w3$} | {:<w4$.2} | {:<w5$.2} | {:<w6$.2} | {:<w7$.4}",
        clean_name, dims, nodes_str, p_str, avg, avg_grow, avg_peel, avg_remaining,
        w0=w_shape, w1=w_dims, w2=w_nodes, w3=w_p, w4=w_avg, w5=w_grow, w6=w_peel, w7=w_rem
    );
}

fn run_topology_suite<T: Topology, const STRIDE_Y: usize>(
    config: GridConfig, 
    topo_name: &str,
    widths: (usize, usize, usize, usize)
) {
    for &p in &ERROR_PROBS {
        let mut defects_list = Vec::with_capacity(CYCLES);
        let mut rng_seed = 12345;
        for _ in 0..CYCLES {
            defects_list.push(generate_defects(
                config.width,
                config.height,
                STRIDE_Y,
                p,
                rng_seed,
            ));
            rng_seed += 1;
        }

        // Tiled Growth
        let bench_name = format!("Tiled-Growth-{}", topo_name);
        run_tiled_growth_bench::<T>(&bench_name, config, p, &defects_list, widths);
    }
}

fn run_suite<const STRIDE_Y: usize>(config: GridConfig, topo: TopologyArg) {
    use std::cmp::max;
    
    // Calculate column widths
    let mut w_shape = "Shape".len();
    let mut w_dims = "Dims".len();
    let mut w_nodes = "Nodes (Target)".len();
    let mut w_p = "p".len();

    // Check applicable topologies to adjust widths
    let mut check_config = |c: GridConfig, name: &str| {
        w_shape = max(w_shape, name.len());
        w_dims = max(w_dims, format!("{}x{}", c.width, c.height).len());
        w_nodes = max(w_nodes, format!("{} ({})", c.actual_nodes(), c.target_nodes).len());
    };

    if matches!(topo, TopologyArg::Square | TopologyArg::All) {
        check_config(config, "Square");
    }
    if matches!(topo, TopologyArg::Rectangle | TopologyArg::All) {
        check_config(config.to_rectangular(3.0), "Rectangle");
    }
    if matches!(topo, TopologyArg::Triangular | TopologyArg::All) {
        check_config(config, "Triangular");
    }
    if matches!(topo, TopologyArg::Honeycomb | TopologyArg::All) {
        check_config(config, "Honeycomb");
    }

    for &p in &ERROR_PROBS {
        w_p = max(w_p, format!("{:.3}", p).len());
    }

    let w_avg = 8;
    let w_grow = 8;
    let w_peel = 8;
    let w_rem = 8;

    // Print Header
    println!();
    println!("Benchmarking Suite: Grid {}x{} (~{} nodes, Stride {})",
        config.width, config.height, config.target_nodes, config.stride_y);
    println!("Cycles: {}", CYCLES);
    let header = format!(
        "{:<w0$} | {:<w1$} | {:<w2$} | {:<w3$} | {:<w4$} | {:<w5$} | {:<w6$} | {:<w7$}",
        "Shape", "Dims", "Nodes (Target)", "p", "Avg(us)", "Grow", "Peel", "Rem",
        w0=w_shape, w1=w_dims, w2=w_nodes, w3=w_p, w4=w_avg, w5=w_grow, w6=w_peel, w7=w_rem
    );
    println!("{}", header);
    println!("{}", "-".repeat(header.len()));

    let widths = (w_shape, w_dims, w_nodes, w_p);

    if matches!(topo, TopologyArg::Square | TopologyArg::All) {
        run_topology_suite::<SquareGrid, STRIDE_Y>(config, "Square", widths);
    }
    
    // Rectangle ~3:1
    if matches!(topo, TopologyArg::Rectangle | TopologyArg::All) {
        let rect_config = config.to_rectangular(3.0);
        
        let rect_stride = rect_config.stride_y;
        
        for &p in &ERROR_PROBS {
            let mut defects_list = Vec::with_capacity(CYCLES);
            let mut rng_seed = 12345;
            for _ in 0..CYCLES {
                 // Use dynamic stride here
                defects_list.push(generate_defects(
                    rect_config.width,
                    rect_config.height,
                    rect_stride,
                    p,
                    rng_seed,
                ));
                rng_seed += 1;
            }

            // Tiled Growth - Square Topology on Rectangle
            run_tiled_growth_bench::<SquareGrid>(
                "Tiled-Growth-Rectangle", 
                rect_config, 
                p, 
                &defects_list,
                widths
            );
        }
    }

    if matches!(topo, TopologyArg::Triangular | TopologyArg::All) {
        run_topology_suite::<TriangularGrid, STRIDE_Y>(config, "Triangular", widths);
    }

    if matches!(topo, TopologyArg::Honeycomb | TopologyArg::All) {
        run_topology_suite::<HoneycombGrid, STRIDE_Y>(config, "Honeycomb", widths);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_defects_structure() {
        let width = 4;
        let height = 4;
        let stride = 8;
        let defects = generate_defects(width, height, stride, 0.5, 123);
        // check size
        let alloc_size = stride * stride;
        let num_blocks = (alloc_size + 63) / 64;
        assert_eq!(defects.len(), num_blocks);
    }
}

fn main() {
    let args = Cli::parse();

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
