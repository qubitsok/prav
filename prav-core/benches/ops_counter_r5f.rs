use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use std::hint::black_box;

use prav_core::arena::Arena;
use prav_core::decoder::EdgeCorrection;
use prav_core::intrinsics::morton_encode_2d;
use prav_core::qec_engine::QecEngine;
use prav_core::topology::SquareGrid;
use prav_core::testing_grids::{GridConfig, TestGrids};

const ERROR_PROBABILITY: f64 = 0.001;

fn generate_defects(config: GridConfig, p: f64) -> Vec<u64> {
    let w = config.width;
    let h = config.height;
    let max_dim = w.max(h);
    let pow2 = max_dim.next_power_of_two();
    let total_morton = pow2 * pow2;
    let num_blocks = (total_morton + 1 + 63) / 64;
    let mut dense = vec![0u64; num_blocks];

    let mut seed = 123456789u64;
    let mut rng = |limit: usize| -> usize {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 33) as usize % limit
    };

    let count = ((w * h) as f64 * p).ceil() as usize;

    for _ in 0..count {
        let x = rng(w);
        let y = rng(h);
        let idx = morton_encode_2d(x as u32, y as u32) as usize;
        let blk = idx / 64;
        let bit = idx % 64;
        if blk < dense.len() {
            dense[blk] |= 1 << bit;
        }
    }
    dense
}

// Stride 32 Context
struct Context32<'a> {
    _arena: Arena<'a>,
    engine: QecEngine<'a, SquareGrid, 32>,
    dense_defects: Vec<u64>,
    corrections: Vec<EdgeCorrection>,
}

fn setup_grid_32(config: GridConfig) -> Context32<'static> {
    let buffer = Box::leak(vec![0u8; 1024 * 1024 * 32].into_boxed_slice());
    let mut arena = Arena::new(buffer);

    let engine = QecEngine::<SquareGrid, 32>::new(&mut arena, config.width, config.height, 1);
    let dense_defects = generate_defects(config, ERROR_PROBABILITY);
    let corrections = vec![EdgeCorrection::default(); config.actual_nodes() * 2];

    Context32 {
        _arena: arena,
        engine,
        dense_defects,
        corrections,
    }
}

fn setup_tiny() -> Context32<'static> { setup_grid_32(TestGrids::TINY) }
fn setup_medium() -> Context32<'static> { setup_grid_32(TestGrids::MEDIUM) }

// Stride 64 Context
struct Context64<'a> {
    _arena: Arena<'a>,
    engine: QecEngine<'a, SquareGrid, 64>,
    dense_defects: Vec<u64>,
    corrections: Vec<EdgeCorrection>,
}

fn setup_grid_64(config: GridConfig) -> Context64<'static> {
    let buffer = Box::leak(vec![0u8; 1024 * 1024 * 64].into_boxed_slice());
    let mut arena = Arena::new(buffer);

    let engine = QecEngine::<SquareGrid, 64>::new(&mut arena, config.width, config.height, 1);
    let dense_defects = generate_defects(config, ERROR_PROBABILITY);
    let corrections = vec![EdgeCorrection::default(); config.actual_nodes() * 2];

    Context64 {
        _arena: arena,
        engine,
        dense_defects,
        corrections,
    }
}

fn setup_large() -> Context64<'static> { setup_grid_64(TestGrids::LARGE) }

#[library_benchmark]
#[bench::tiny(setup_tiny())]
#[bench::medium(setup_medium())]
fn bench_process_cycle_32(mut ctx: Context32<'static>) {
    black_box(ctx.engine.process_cycle_dense(
        black_box(&ctx.dense_defects),
        black_box(&mut ctx.corrections),
    ));
}

#[library_benchmark]
#[bench::large(setup_large())]
fn bench_process_cycle_64(mut ctx: Context64<'static>) {
    black_box(ctx.engine.process_cycle_dense(
        black_box(&ctx.dense_defects),
        black_box(&mut ctx.corrections),
    ));
}

library_benchmark_group!(
    name = ops_counter_r5f_group;
    benchmarks = bench_process_cycle_32, bench_process_cycle_64
);

main!(
    config = iai_callgrind::LibraryBenchmarkConfig::default()
        .valgrind_args([
            "--I1=32768,4,32",
            "--D1=32768,4,32",
            "--LL=131072,4,32"
        ]);
    library_benchmark_groups = ops_counter_r5f_group
);
