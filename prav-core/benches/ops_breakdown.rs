use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use std::hint::black_box;

use prav_core::arena::Arena;
use prav_core::decoder::DecodingState;
use prav_core::decoder::EdgeCorrection;
use prav_core::decoder::{ClusterGrowth, Peeling};
use prav_core::intrinsics::morton_encode_2d;
use prav_core::topology::SquareGrid;

// --- Constants ---
const P: f64 = 0.003;
const L_SMALL: usize = 32; // ~1K nodes
const L_BIG: usize = 64; // 4096 nodes (Stride 64)

// --- Data Generation ---

fn generate_dense_defects(count: usize, w: usize, h: usize) -> Vec<u64> {
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

// --- Setup Helpers ---

struct BenchContext<'a, T: prav_core::topology::Topology, const STRIDE_Y: usize> {
    _arena: Arena<'a>,
    decoder: DecodingState<'a, T, STRIDE_Y>,
    dense_defects: Vec<u64>,
    corrections: Vec<EdgeCorrection>,
}

fn setup_base<const STRIDE_Y: usize>(
    l: usize,
    p: f64,
) -> BenchContext<'static, SquareGrid, STRIDE_Y> {
    // Allocate enough memory for big scenarios
    let buffer = Box::leak(vec![0u8; 1024 * 1024 * 128].into_boxed_slice());
    let mut arena = Arena::new(buffer);

    let decoder = DecodingState::new(&mut arena, l, l, 1);
    let count = ((l * l) as f64 * p).ceil() as usize;
    let dense_defects = generate_dense_defects(count.max(1), l, l);
    let corrections = vec![EdgeCorrection::default(); l * l * 2];

    BenchContext {
        _arena: arena,
        decoder,
        dense_defects,
        corrections,
    }
}

// Specialized setups for operations
fn setup_sparse_reset<const STRIDE_Y: usize>(
    l: usize,
    p: f64,
) -> BenchContext<'static, SquareGrid, STRIDE_Y> {
    let mut ctx = setup_base::<STRIDE_Y>(l, p);
    ctx.decoder.load_dense_syndromes(&ctx.dense_defects);
    ctx.decoder.grow_clusters();
    ctx.decoder.peel_forest(&mut ctx.corrections);
    ctx
}

fn setup_load<const STRIDE_Y: usize>(
    l: usize,
    p: f64,
) -> BenchContext<'static, SquareGrid, STRIDE_Y> {
    let mut ctx = setup_base::<STRIDE_Y>(l, p);
    ctx.decoder.sparse_reset();
    ctx
}

fn setup_grow<const STRIDE_Y: usize>(
    l: usize,
    p: f64,
) -> BenchContext<'static, SquareGrid, STRIDE_Y> {
    let mut ctx = setup_base::<STRIDE_Y>(l, p);
    ctx.decoder.sparse_reset();
    ctx.decoder.load_dense_syndromes(&ctx.dense_defects);
    ctx
}

fn setup_peel<const STRIDE_Y: usize>(
    l: usize,
    p: f64,
) -> BenchContext<'static, SquareGrid, STRIDE_Y> {
    let mut ctx = setup_base::<STRIDE_Y>(l, p);
    ctx.decoder.sparse_reset();
    ctx.decoder.load_dense_syndromes(&ctx.dense_defects);
    ctx.decoder.grow_clusters();
    ctx
}

fn setup_grow_iter<const STRIDE_Y: usize>(
    l: usize,
    p: f64,
) -> BenchContext<'static, SquareGrid, STRIDE_Y> {
    let mut ctx = setup_base::<STRIDE_Y>(l, p);
    ctx.decoder.sparse_reset();
    ctx.decoder.load_dense_syndromes(&ctx.dense_defects);
    ctx
}

fn setup_uf_find<const STRIDE_Y: usize>(
    l: usize,
    p: f64,
) -> (DecodingState<'static, SquareGrid, STRIDE_Y>, u32) {
    let ctx = setup_base::<STRIDE_Y>(l, p);
    // Create a chain of parents
    for i in 0..63 {
        ctx.decoder.parents[i] = (i + 1) as u32;
    }
    ctx.decoder.parents[63] = 63;
    (ctx.decoder, 0)
}

fn setup_uf_union<const STRIDE_Y: usize>(
    l: usize,
    p: f64,
) -> (DecodingState<'static, SquareGrid, STRIDE_Y>, u32, u32) {
    let ctx = setup_base::<STRIDE_Y>(l, p);
    (ctx.decoder, 10, 20)
}

fn setup_process_block<const STRIDE_Y: usize>(
    l: usize,
    p: f64,
) -> (DecodingState<'static, SquareGrid, STRIDE_Y>, usize) {
    let mut ctx = setup_base::<STRIDE_Y>(l, p);
    ctx.decoder.sparse_reset();
    ctx.decoder.load_dense_syndromes(&ctx.dense_defects);
    (ctx.decoder, 0)
}

fn setup_trace_detailed<const STRIDE_Y: usize>(
    l: usize,
    p: f64,
) -> (DecodingState<'static, SquareGrid, STRIDE_Y>, u32, u32) {
    let mut ctx = setup_base::<STRIDE_Y>(l, p);
    ctx.decoder.sparse_reset();
    ctx.decoder.load_dense_syndromes(&ctx.dense_defects);
    ctx.decoder.grow_clusters();

    let u = (ctx.decoder.graph.stride_y + 5) as u32;
    let v = (ctx.decoder.graph.stride_y * 2 + 10) as u32;
    (ctx.decoder, u, v)
}

// Helper wrappers for iai
fn setup_uf_find_small() -> (DecodingState<'static, SquareGrid, 32>, u32) {
    setup_uf_find::<32>(L_SMALL, P)
}
fn setup_uf_find_big() -> (DecodingState<'static, SquareGrid, 64>, u32) {
    setup_uf_find::<64>(L_BIG, P)
}

fn setup_uf_union_small() -> (DecodingState<'static, SquareGrid, 32>, u32, u32) {
    setup_uf_union::<32>(L_SMALL, P)
}
fn setup_uf_union_big() -> (DecodingState<'static, SquareGrid, 64>, u32, u32) {
    setup_uf_union::<64>(L_BIG, P)
}

fn setup_process_block_small() -> (DecodingState<'static, SquareGrid, 32>, usize) {
    setup_process_block::<32>(L_SMALL, P)
}
fn setup_process_block_big() -> (DecodingState<'static, SquareGrid, 64>, usize) {
    setup_process_block::<64>(L_BIG, P)
}

fn setup_sparse_reset_small() -> BenchContext<'static, SquareGrid, 32> {
    setup_sparse_reset::<32>(L_SMALL, P)
}
fn setup_sparse_reset_big() -> BenchContext<'static, SquareGrid, 64> {
    setup_sparse_reset::<64>(L_BIG, P)
}

fn setup_load_small() -> BenchContext<'static, SquareGrid, 32> {
    setup_load::<32>(L_SMALL, P)
}
fn setup_load_big() -> BenchContext<'static, SquareGrid, 64> {
    setup_load::<64>(L_BIG, P)
}

fn setup_grow_small() -> BenchContext<'static, SquareGrid, 32> {
    setup_grow::<32>(L_SMALL, P)
}
fn setup_grow_big() -> BenchContext<'static, SquareGrid, 64> {
    setup_grow::<64>(L_BIG, P)
}

fn setup_peel_small() -> BenchContext<'static, SquareGrid, 32> {
    setup_peel::<32>(L_SMALL, P)
}
fn setup_peel_big() -> BenchContext<'static, SquareGrid, 64> {
    setup_peel::<64>(L_BIG, P)
}

fn setup_grow_iter_small() -> BenchContext<'static, SquareGrid, 32> {
    setup_grow_iter::<32>(L_SMALL, P)
}
fn setup_grow_iter_big() -> BenchContext<'static, SquareGrid, 64> {
    setup_grow_iter::<64>(L_BIG, P)
}

fn setup_trace_detailed_small() -> (DecodingState<'static, SquareGrid, 32>, u32, u32) {
    setup_trace_detailed::<32>(L_SMALL, P)
}
fn setup_trace_detailed_big() -> (DecodingState<'static, SquareGrid, 64>, u32, u32) {
    setup_trace_detailed::<64>(L_BIG, P)
}

fn setup_base_small() -> BenchContext<'static, SquareGrid, 32> {
    setup_base::<32>(L_SMALL, P)
}
fn setup_base_big() -> BenchContext<'static, SquareGrid, 64> {
    setup_base::<64>(L_BIG, P)
}

// --- Benchmarks: Union-Find ---

#[library_benchmark]
#[bench::small(setup_uf_find_small())]
fn bench_uf_find_small(input: (DecodingState<'static, SquareGrid, 32>, u32)) {
    let (mut decoder, u) = input;
    black_box(decoder.find(u));
}

#[library_benchmark]
#[bench::big(setup_uf_find_big())]
fn bench_uf_find_big(input: (DecodingState<'static, SquareGrid, 64>, u32)) {
    let (mut decoder, u) = input;
    black_box(decoder.find(u));
}

#[library_benchmark]
#[bench::small(setup_uf_union_small())]
fn bench_uf_union_small(input: (DecodingState<'static, SquareGrid, 32>, u32, u32)) {
    let (mut decoder, u, v) = input;
    unsafe { black_box(decoder.union(u, v)) };
}

#[library_benchmark]
#[bench::big(setup_uf_union_big())]
fn bench_uf_union_big(input: (DecodingState<'static, SquareGrid, 64>, u32, u32)) {
    let (mut decoder, u, v) = input;
    unsafe { black_box(decoder.union(u, v)) };
}

// --- Benchmarks: Growth ---

#[library_benchmark]
#[bench::small(setup_process_block_small())]
fn bench_process_block_small(input: (DecodingState<'static, SquareGrid, 32>, usize)) {
    let (mut decoder, blk_idx) = input;
    unsafe { black_box(decoder.process_block(blk_idx)) };
}

#[library_benchmark]
#[bench::big(setup_process_block_big())]
fn bench_process_block_big(input: (DecodingState<'static, SquareGrid, 64>, usize)) {
    let (mut decoder, blk_idx) = input;
    unsafe { black_box(decoder.process_block(blk_idx)) };
}

#[library_benchmark]
#[bench::small(setup_sparse_reset_small())]
fn bench_sparse_reset_small(mut ctx: BenchContext<'static, SquareGrid, 32>) {
    black_box(ctx.decoder.sparse_reset());
}

#[library_benchmark]
#[bench::big(setup_sparse_reset_big())]
fn bench_sparse_reset_big(mut ctx: BenchContext<'static, SquareGrid, 64>) {
    black_box(ctx.decoder.sparse_reset());
}

#[library_benchmark]
#[bench::small(setup_load_small())]
fn bench_load_dense_syndromes_small(mut ctx: BenchContext<'static, SquareGrid, 32>) {
    black_box(
        ctx.decoder
            .load_dense_syndromes(black_box(&ctx.dense_defects)),
    );
}

#[library_benchmark]
#[bench::big(setup_load_big())]
fn bench_load_dense_syndromes_big(mut ctx: BenchContext<'static, SquareGrid, 64>) {
    black_box(
        ctx.decoder
            .load_dense_syndromes(black_box(&ctx.dense_defects)),
    );
}

#[library_benchmark]
#[bench::small(setup_grow_small())]
fn bench_grow_clusters_small(mut ctx: BenchContext<'static, SquareGrid, 32>) {
    black_box(ctx.decoder.grow_clusters());
}

#[library_benchmark]
#[bench::big(setup_grow_big())]
fn bench_grow_clusters_big(mut ctx: BenchContext<'static, SquareGrid, 64>) {
    black_box(ctx.decoder.grow_clusters());
}

#[library_benchmark]
#[bench::small(setup_grow_iter_small())]
fn bench_grow_iteration_small(mut ctx: BenchContext<'static, SquareGrid, 32>) {
    black_box(ctx.decoder.grow_iteration());
}

#[library_benchmark]
#[bench::big(setup_grow_iter_big())]
fn bench_grow_iteration_big(mut ctx: BenchContext<'static, SquareGrid, 64>) {
    black_box(ctx.decoder.grow_iteration());
}

// --- Benchmarks: Peeling ---

#[library_benchmark]
#[bench::small(setup_peel_small())]
fn bench_peel_forest_small(mut ctx: BenchContext<'static, SquareGrid, 32>) {
    black_box(ctx.decoder.peel_forest(black_box(&mut ctx.corrections)));
}

#[library_benchmark]
#[bench::big(setup_peel_big())]
fn bench_peel_forest_big(mut ctx: BenchContext<'static, SquareGrid, 64>) {
    black_box(ctx.decoder.peel_forest(black_box(&mut ctx.corrections)));
}

#[library_benchmark]
#[bench::small(setup_trace_detailed_small())]
fn bench_trace_manhattan_small(input: (DecodingState<'static, SquareGrid, 32>, u32, u32)) {
    let (mut decoder, u, v) = input;
    black_box(decoder.trace_manhattan(u, v));
}

#[library_benchmark]
#[bench::big(setup_trace_detailed_big())]
fn bench_trace_manhattan_big(input: (DecodingState<'static, SquareGrid, 64>, u32, u32)) {
    let (mut decoder, u, v) = input;
    black_box(decoder.trace_manhattan(u, v));
}

#[library_benchmark]
#[bench::small(setup_trace_detailed_small())]
fn bench_trace_bfs_small(input: (DecodingState<'static, SquareGrid, 32>, u32, u32)) {
    let (mut decoder, u, v) = input;
    black_box(decoder.trace_bfs(u, v, u64::MAX));
}

#[library_benchmark]
#[bench::big(setup_trace_detailed_big())]
fn bench_trace_bfs_big(input: (DecodingState<'static, SquareGrid, 64>, u32, u32)) {
    let (mut decoder, u, v) = input;
    black_box(decoder.trace_bfs(u, v, u64::MAX));
}

#[library_benchmark]
#[bench::small(setup_trace_detailed_small())]
fn bench_trace_path_small(input: (DecodingState<'static, SquareGrid, 32>, u32, u32)) {
    let (mut decoder, u, _) = input;
    let boundary = (decoder.parents.len() - 1) as u32;
    black_box(decoder.trace_path(u, boundary));
}

#[library_benchmark]
#[bench::big(setup_trace_detailed_big())]
fn bench_trace_path_big(input: (DecodingState<'static, SquareGrid, 64>, u32, u32)) {
    let (mut decoder, u, _) = input;
    let boundary = (decoder.parents.len() - 1) as u32;
    black_box(decoder.trace_path(u, boundary));
}

#[library_benchmark]
#[bench::small(setup_base_small())]
fn bench_emit_linear_small(mut ctx: BenchContext<'static, SquareGrid, 32>) {
    black_box(ctx.decoder.emit_linear(10, 11));
}

#[library_benchmark]
#[bench::big(setup_base_big())]
fn bench_emit_linear_big(mut ctx: BenchContext<'static, SquareGrid, 64>) {
    black_box(ctx.decoder.emit_linear(10, 11));
}

#[library_benchmark]
#[bench::small(setup_base_small())]
fn bench_reconstruct_corrections_small(mut ctx: BenchContext<'static, SquareGrid, 32>) {
    ctx.decoder.emit_linear(10, 11);
    ctx.decoder.emit_linear(20, 21);
    black_box(
        ctx.decoder
            .reconstruct_corrections(black_box(&mut ctx.corrections)),
    );
}

#[library_benchmark]
#[bench::big(setup_base_big())]
fn bench_reconstruct_corrections_big(mut ctx: BenchContext<'static, SquareGrid, 64>) {
    ctx.decoder.emit_linear(10, 11);
    ctx.decoder.emit_linear(20, 21);
    black_box(
        ctx.decoder
            .reconstruct_corrections(black_box(&mut ctx.corrections)),
    );
}

library_benchmark_group!(
    name = uf_ops;
    benchmarks = bench_uf_find_small, bench_uf_find_big, bench_uf_union_small, bench_uf_union_big
);

library_benchmark_group!(
    name = growth_ops;
    benchmarks = bench_sparse_reset_small, bench_sparse_reset_big,
                 bench_load_dense_syndromes_small, bench_load_dense_syndromes_big,
                 bench_grow_clusters_small, bench_grow_clusters_big,
                 bench_grow_iteration_small, bench_grow_iteration_big,
                 bench_process_block_small, bench_process_block_big
);

library_benchmark_group!(
    name = peeling_ops;
    benchmarks = bench_peel_forest_small, bench_peel_forest_big,
                 bench_trace_manhattan_small, bench_trace_manhattan_big,
                 bench_trace_bfs_small, bench_trace_bfs_big,
                 bench_trace_path_small, bench_trace_path_big,
                 bench_emit_linear_small, bench_emit_linear_big,
                 bench_reconstruct_corrections_small, bench_reconstruct_corrections_big
);

main!(library_benchmark_groups = uf_ops, growth_ops, peeling_ops);
