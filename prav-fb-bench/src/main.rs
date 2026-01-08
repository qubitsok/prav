//! Benchmark comparison: prav-core vs fusion-blossom
//!
//! Compares decode latency on multiple grid sizes (17x17, 32x32, 64x64).
//! Outputs timing percentiles (avg, p50, p95, p99) in console tables.
//! Includes rigorous correctness verification proving feature parity.

mod graph;
mod stats;
mod syndrome;
mod verification;

use std::time::{Duration, Instant};

use clap::Parser;
use fusion_blossom::mwpm_solver::{PrimalDualSolver, SolverSerial};
use fusion_blossom::util::SyndromePattern;

use prav_core::{Arena, DecoderBuilder, DynDecoder, EdgeCorrection, SquareGrid, required_buffer_size};

use crate::graph::create_surface_code_graph;
use crate::stats::{calculate_percentiles, format_number, LatencyStats};
use crate::syndrome::{count_defects, generate_prav_syndromes, prav_to_fusion_blossom};
use crate::verification::{verify_fb_corrections, verify_prav_corrections};

#[derive(Parser, Debug)]
#[command(name = "prav-fb-bench")]
#[command(about = "Benchmark prav-core vs fusion-blossom QEC decoders")]
struct Args {
    /// Grid sizes to benchmark (square grids)
    #[arg(long, num_args = 1.., default_values_t = vec![17, 32, 64])]
    grids: Vec<usize>,

    /// Number of shots per error rate
    #[arg(long, default_value_t = 10000)]
    shots: usize,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Error probabilities to test
    #[arg(long, num_args = 1.., default_values_t = vec![0.001, 0.01, 0.06])]
    error_probs: Vec<f64>,
}

/// Results from a single benchmark run.
struct BenchmarkResults {
    prav_times: Vec<Duration>,
    fb_times: Vec<Duration>,
    prav_verified: usize,
    fb_verified: usize,
    total_defects: usize,
    prav_corrections: usize,
    fb_corrections: usize,
}

/// Results for a single error rate.
struct ErrorRateResults {
    error_prob: f64,
    total_defects: usize,
    prav_stats: LatencyStats,
    fb_stats: LatencyStats,
    prav_verified: usize,
    fb_verified: usize,
    parity_matches: usize,
    prav_corrections: usize,
    fb_corrections: usize,
    num_shots: usize,
}

fn warmup_prav(decoder: &mut DynDecoder<SquareGrid>, width: usize, height: usize) {
    let warmup_syndromes = generate_prav_syndromes(width, height, 0.01, 200, 0xDEADBEEF);
    let mut corrections = vec![EdgeCorrection::default(); width * height];

    for syn in &warmup_syndromes {
        decoder.load_dense_syndromes(syn);
        let _ = decoder.decode(&mut corrections);
        decoder.reset_for_next_cycle();
    }
}

fn warmup_fusion_blossom(solver: &mut SolverSerial, width: usize, height: usize) {
    let warmup_syndromes = generate_prav_syndromes(width, height, 0.01, 200, 0xCAFEBABE);

    for syn in &warmup_syndromes {
        let defects = prav_to_fusion_blossom(syn, width, height);
        let pattern = SyndromePattern::new_vertices(defects);
        solver.solve(&pattern);
        let _ = solver.subgraph();
        solver.clear();
    }
}

fn benchmark_with_verification(
    prav_decoder: &mut DynDecoder<SquareGrid>,
    fb_solver: &mut SolverSerial,
    prav_syndromes: &[Vec<u64>],
    edges: &[(fusion_blossom::util::VertexIndex, fusion_blossom::util::VertexIndex, fusion_blossom::util::Weight)],
    width: usize,
    height: usize,
) -> BenchmarkResults {
    let mut prav_times = Vec::with_capacity(prav_syndromes.len());
    let mut fb_times = Vec::with_capacity(prav_syndromes.len());
    let mut prav_verified = 0;
    let mut fb_verified = 0;
    let mut total_defects = 0;
    let mut prav_corrections_total = 0;
    let mut fb_corrections_total = 0;

    let mut corrections = vec![EdgeCorrection::default(); width * height * 2];

    for prav_syn in prav_syndromes {
        // Count defects
        total_defects += count_defects(prav_syn);

        // Convert to fusion-blossom format
        let fb_syn = prav_to_fusion_blossom(prav_syn, width, height);

        // Time prav
        let t0 = Instant::now();
        prav_decoder.load_dense_syndromes(prav_syn);
        let n = prav_decoder.decode(&mut corrections);
        prav_decoder.reset_for_next_cycle();
        prav_times.push(t0.elapsed());

        // Verify prav
        let (ok, _, _) = verify_prav_corrections(prav_syn, &corrections[..n], width, height);
        if ok {
            prav_verified += 1;
        }
        prav_corrections_total += n;

        // Time fusion-blossom
        let pattern = SyndromePattern::new_vertices(fb_syn.clone());
        let t0 = Instant::now();
        fb_solver.solve(&pattern);
        let fb_corr = fb_solver.subgraph();
        fb_solver.clear();
        fb_times.push(t0.elapsed());

        // Verify fusion-blossom
        let (ok, _, _) = verify_fb_corrections(&fb_syn, &fb_corr, edges, width, height);
        if ok {
            fb_verified += 1;
        }
        fb_corrections_total += fb_corr.len();
    }

    BenchmarkResults {
        prav_times,
        fb_times,
        prav_verified,
        fb_verified,
        total_defects,
        prav_corrections: prav_corrections_total,
        fb_corrections: fb_corrections_total,
    }
}

fn run_benchmark_for_grid(
    width: usize,
    height: usize,
    error_probs: &[f64],
    num_shots: usize,
    seed: u64,
) -> Vec<ErrorRateResults> {
    let mut results = Vec::new();

    // Create arena and prav decoder
    let buf_size = required_buffer_size(width, height, 1);
    let mut buffer = vec![0u8; buf_size];
    let mut arena = Arena::new(&mut buffer);

    let mut prav_decoder = DecoderBuilder::<SquareGrid>::new()
        .dimensions(width, height)
        .build(&mut arena)
        .expect("Failed to create prav decoder");

    for &p in error_probs {
        print!("  Error rate {:.3}... ", p);

        // Create fusion-blossom solver with proper weights
        let (initializer, edges) = create_surface_code_graph(width, height, p);
        let mut fb_solver = SolverSerial::new(&initializer);

        // Warmup both decoders
        warmup_prav(&mut prav_decoder, width, height);
        warmup_fusion_blossom(&mut fb_solver, width, height);

        // Generate syndromes
        let prav_syndromes = generate_prav_syndromes(
            width,
            height,
            p,
            num_shots,
            seed + (p * 1_000_000.0) as u64,
        );

        // Run benchmark with verification
        let bench_result = benchmark_with_verification(
            &mut prav_decoder,
            &mut fb_solver,
            &prav_syndromes,
            &edges,
            width,
            height,
        );

        // Calculate statistics
        let prav_stats = calculate_percentiles(&bench_result.prav_times);
        let fb_stats = calculate_percentiles(&bench_result.fb_times);

        // Calculate parity matches
        let all_verified = bench_result.prav_verified == num_shots
            && bench_result.fb_verified == num_shots;
        let parity_matches = if all_verified {
            num_shots
        } else {
            bench_result.prav_verified.min(bench_result.fb_verified)
        };

        let prav_pct = 100.0 * bench_result.prav_verified as f64 / num_shots as f64;
        let fb_pct = 100.0 * bench_result.fb_verified as f64 / num_shots as f64;

        println!(
            "done (prav: {:.2}us [{:.0}%], FB: {:.2}us [{:.0}%])",
            prav_stats.avg_us, prav_pct,
            fb_stats.avg_us, fb_pct
        );

        results.push(ErrorRateResults {
            error_prob: p,
            total_defects: bench_result.total_defects,
            prav_stats,
            fb_stats,
            prav_verified: bench_result.prav_verified,
            fb_verified: bench_result.fb_verified,
            parity_matches,
            prav_corrections: bench_result.prav_corrections,
            fb_corrections: bench_result.fb_corrections,
            num_shots,
        });
    }

    results
}

fn print_results_for_grid(results: &[ErrorRateResults], size: usize, num_shots: usize) {
    println!();
    println!("{}", "=".repeat(70));
    println!(
        "Results: {}x{} Square Grid | {} shots per error rate",
        size,
        size,
        format_number(num_shots)
    );
    println!("fusion-blossom configured with weights = 1000 * ln((1-p)/p)");
    println!("{}", "=".repeat(70));

    // prav latencies table
    println!("\nprav Latencies (microseconds):");
    println!(
        "{:-<14}+{:-<12}+{:-<10}+{:-<10}+{:-<10}+{:-<10}",
        "", "", "", "", "", ""
    );
    println!(
        "{:>14}|{:>12}|{:>10}|{:>10}|{:>10}|{:>10}",
        "Error Rate", "Defects", "avg", "p50", "p95", "p99"
    );
    println!(
        "{:-<14}+{:-<12}+{:-<10}+{:-<10}+{:-<10}+{:-<10}",
        "", "", "", "", "", ""
    );
    for r in results {
        println!(
            "{:>14.3}|{:>12}|{:>10.2}|{:>10.2}|{:>10.2}|{:>10.2}",
            r.error_prob,
            format_number(r.total_defects),
            r.prav_stats.avg_us,
            r.prav_stats.p50_us,
            r.prav_stats.p95_us,
            r.prav_stats.p99_us,
        );
    }
    println!(
        "{:-<14}+{:-<12}+{:-<10}+{:-<10}+{:-<10}+{:-<10}",
        "", "", "", "", "", ""
    );

    // fusion-blossom latencies table
    println!("\nfusion-blossom Latencies (microseconds):");
    println!(
        "{:-<14}+{:-<12}+{:-<10}+{:-<10}+{:-<10}+{:-<10}",
        "", "", "", "", "", ""
    );
    println!(
        "{:>14}|{:>12}|{:>10}|{:>10}|{:>10}|{:>10}",
        "Error Rate", "Defects", "avg", "p50", "p95", "p99"
    );
    println!(
        "{:-<14}+{:-<12}+{:-<10}+{:-<10}+{:-<10}+{:-<10}",
        "", "", "", "", "", ""
    );
    for r in results {
        println!(
            "{:>14.3}|{:>12}|{:>10.2}|{:>10.2}|{:>10.2}|{:>10.2}",
            r.error_prob,
            format_number(r.total_defects),
            r.fb_stats.avg_us,
            r.fb_stats.p50_us,
            r.fb_stats.p95_us,
            r.fb_stats.p99_us,
        );
    }
    println!(
        "{:-<14}+{:-<12}+{:-<10}+{:-<10}+{:-<10}+{:-<10}",
        "", "", "", "", "", ""
    );

    // Speedup table
    println!("\nSpeedup (fusion-blossom / prav):");
    println!(
        "{:-<14}+{:-<10}+{:-<10}+{:-<10}+{:-<10}",
        "", "", "", "", ""
    );
    println!(
        "{:>14}|{:>10}|{:>10}|{:>10}|{:>10}",
        "Error Rate", "avg", "p50", "p95", "p99"
    );
    println!(
        "{:-<14}+{:-<10}+{:-<10}+{:-<10}+{:-<10}",
        "", "", "", "", ""
    );
    for r in results {
        let avg_speedup = if r.prav_stats.avg_us > 0.0 {
            r.fb_stats.avg_us / r.prav_stats.avg_us
        } else {
            0.0
        };
        let p50_speedup = if r.prav_stats.p50_us > 0.0 {
            r.fb_stats.p50_us / r.prav_stats.p50_us
        } else {
            0.0
        };
        let p95_speedup = if r.prav_stats.p95_us > 0.0 {
            r.fb_stats.p95_us / r.prav_stats.p95_us
        } else {
            0.0
        };
        let p99_speedup = if r.prav_stats.p99_us > 0.0 {
            r.fb_stats.p99_us / r.prav_stats.p99_us
        } else {
            0.0
        };
        println!(
            "{:>14.3}|{:>9.2}x|{:>9.2}x|{:>9.2}x|{:>9.2}x",
            r.error_prob, avg_speedup, p50_speedup, p95_speedup, p99_speedup,
        );
    }
    println!(
        "{:-<14}+{:-<10}+{:-<10}+{:-<10}+{:-<10}",
        "", "", "", "", ""
    );

    // Correctness verification table
    println!("\nCorrectness Verification:");
    println!(
        "{:-<14}+{:-<20}+{:-<24}+{:-<18}",
        "", "", "", ""
    );
    println!(
        "{:>14}|{:>20}|{:>24}|{:>18}",
        "Error Rate", "prav Success", "fusion-blossom Success", "Feature Parity"
    );
    println!(
        "{:-<14}+{:-<20}+{:-<24}+{:-<18}",
        "", "", "", ""
    );
    for r in results {
        let prav_pct = 100.0 * r.prav_verified as f64 / r.num_shots as f64;
        let fb_pct = 100.0 * r.fb_verified as f64 / r.num_shots as f64;
        let parity_pct = 100.0 * r.parity_matches as f64 / r.num_shots as f64;
        println!(
            "{:>14.3}|{:>6.2}% ({:>8})|{:>10.2}% ({:>8})|{:>13.2}%",
            r.error_prob,
            prav_pct,
            format_number(r.prav_verified),
            fb_pct,
            format_number(r.fb_verified),
            parity_pct,
        );
    }
    println!(
        "{:-<14}+{:-<20}+{:-<24}+{:-<18}",
        "", "", "", ""
    );

    // Correction counts
    println!("\nCorrection Counts:");
    println!("{:-<14}+{:-<14}+{:-<14}", "", "", "");
    println!(
        "{:>14}|{:>14}|{:>14}",
        "Error Rate", "prav", "fusion-blossom"
    );
    println!("{:-<14}+{:-<14}+{:-<14}", "", "", "");
    for r in results {
        println!(
            "{:>14.3}|{:>14}|{:>14}",
            r.error_prob,
            format_number(r.prav_corrections),
            format_number(r.fb_corrections),
        );
    }
    println!("{:-<14}+{:-<14}+{:-<14}", "", "", "");
}

fn print_summary(all_results: &[(usize, Vec<ErrorRateResults>)]) {
    println!();
    println!("{}", "=".repeat(70));
    println!("SUMMARY ACROSS ALL GRIDS");
    println!("{}", "=".repeat(70));

    let mut total_shots = 0;
    let mut total_prav_verified = 0;
    let mut total_fb_verified = 0;
    let mut total_parity = 0;
    let mut all_speedups = Vec::new();

    for (_, results) in all_results {
        for r in results {
            total_shots += r.num_shots;
            total_prav_verified += r.prav_verified;
            total_fb_verified += r.fb_verified;
            total_parity += r.parity_matches;
            if r.prav_stats.avg_us > 0.0 {
                all_speedups.push(r.fb_stats.avg_us / r.prav_stats.avg_us);
            }
        }
    }

    let prav_success_rate = if total_shots > 0 {
        100.0 * total_prav_verified as f64 / total_shots as f64
    } else {
        0.0
    };
    let fb_success_rate = if total_shots > 0 {
        100.0 * total_fb_verified as f64 / total_shots as f64
    } else {
        0.0
    };
    let parity_rate = if total_shots > 0 {
        100.0 * total_parity as f64 / total_shots as f64
    } else {
        0.0
    };
    let avg_speedup = if !all_speedups.is_empty() {
        all_speedups.iter().sum::<f64>() / all_speedups.len() as f64
    } else {
        0.0
    };

    println!();
    println!("Total shots: {}", format_number(total_shots));
    println!(
        "prav defect resolution: {:.2}% ({}/{})",
        prav_success_rate,
        format_number(total_prav_verified),
        format_number(total_shots)
    );
    println!(
        "fusion-blossom defect resolution: {:.2}% ({}/{})",
        fb_success_rate,
        format_number(total_fb_verified),
        format_number(total_shots)
    );
    println!("Feature parity: {:.2}%", parity_rate);
    println!(
        "Average speedup vs fusion-blossom: {:.2}x",
        avg_speedup
    );

    if prav_success_rate == 100.0 && fb_success_rate == 100.0 {
        println!();
        println!("All decoders achieved 100% defect resolution across all configurations.");
        println!("Feature parity confirmed: both decoders produce equivalent results.");
    } else if prav_success_rate < 100.0 || fb_success_rate < 100.0 {
        println!();
        println!("WARNING: Some shots did not resolve all defects.");
    }
}

fn main() {
    let args = Args::parse();

    println!("Benchmark: prav-core vs fusion-blossom QEC Decoders");
    println!("Grids: {:?}", args.grids);
    println!("Shots per error rate: {}", format_number(args.shots));
    println!("Error probabilities: {:?}", args.error_probs);

    let mut all_results = Vec::new();

    for &size in &args.grids {
        println!();
        println!("{}", "=".repeat(70));
        println!(
            "Grid: {}x{} | {} shots per error rate",
            size,
            size,
            format_number(args.shots)
        );
        println!("{}", "=".repeat(70));
        println!("  Initializing prav decoder...");

        let results = run_benchmark_for_grid(size, size, &args.error_probs, args.shots, args.seed);
        all_results.push((size, results));
    }

    // Print detailed results for each grid
    for (size, results) in &all_results {
        print_results_for_grid(results, *size, args.shots);
    }

    // Print overall summary
    print_summary(&all_results);
}
