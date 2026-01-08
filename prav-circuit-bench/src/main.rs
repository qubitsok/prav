//! 3D Circuit-Level Benchmark for prav Union-Find QEC Decoder
//!
//! This benchmark tests the prav decoder on 3D space-time decoding problems
//! generated from Stim Detector Error Models (DEM) or phenomenological noise.
//!
//! # Usage
//!
//! ```bash
//! # Run with default settings
//! cargo run --release -p prav-circuit-bench
//!
//! # Run with DEM file
//! cargo run --release -p prav-circuit-bench -- --dem path/to/model.dem
//!
//! # Run with specific distances
//! cargo run --release -p prav-circuit-bench -- --distances 3,5,7
//! ```

mod dem;
mod stats;
mod surface_code;
mod syndrome;
mod verification;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;

use prav_core::{
    Arena, DecoderBuilder, DynDecoder, EdgeCorrection, Grid3D,
    Grid3DConfig, TestGrids3D, CIRCUIT_ERROR_PROBS, required_buffer_size,
};

use crate::stats::{
    calculate_percentiles, SuppressionFactor, ThresholdPoint, CSV_HEADER,
};
use crate::syndrome::{generate_correlated_syndromes, CircuitSampler, SyndromeWithLogical};
use crate::surface_code::DetectorMapper;
use crate::verification::verify_with_logical;

#[derive(Parser, Debug)]
#[command(name = "prav-circuit-bench")]
#[command(about = "3D circuit-level benchmark for prav Union-Find QEC decoder")]
struct Args {
    /// Path to Stim DEM file (optional, uses phenomenological if not provided)
    #[arg(long)]
    dem: Option<PathBuf>,

    /// Number of shots per configuration
    #[arg(long, default_value_t = 10000)]
    shots: usize,

    /// Code distances to benchmark (comma-separated)
    #[arg(long, value_delimiter = ',', default_values_t = vec![3, 5, 7])]
    distances: Vec<usize>,

    /// Error probabilities to test (comma-separated)
    #[arg(long, value_delimiter = ',')]
    error_probs: Option<Vec<f64>>,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Skip verification (faster but no correctness check)
    #[arg(long)]
    no_verify: bool,

    /// Output results as CSV to stdout
    #[arg(long)]
    csv: bool,

    /// Run threshold study mode (denser error rate sweep, more distances)
    #[arg(long)]
    threshold_study: bool,

    /// Minimum logical errors before stopping (adaptive sampling)
    #[arg(long, default_value_t = 100)]
    min_errors: usize,

    /// Maximum shots per data point
    #[arg(long, default_value_t = 1_000_000)]
    max_shots: usize,
}

/// Error rates for threshold study (denser around expected threshold ~0.5%)
const THRESHOLD_STUDY_PROBS: &[f64] = &[
    0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010,
];

/// Default distances for threshold study
const THRESHOLD_STUDY_DISTANCES: &[usize] = &[3, 5, 7, 9];

/// Results from a single benchmark run.
struct BenchmarkResults {
    times: Vec<Duration>,
    verified: usize,
    total_shots: usize,
    total_defects: usize,
    total_corrections: usize,
    /// Number of logical errors (predicted != actual)
    logical_errors: usize,
}

fn warmup_decoder(decoder: &mut DynDecoder<Grid3D>, samples: &[SyndromeWithLogical], max_corrections: usize) {
    let mut corrections = vec![EdgeCorrection::default(); max_corrections];
    for sample in samples.iter().take(200) {
        decoder.load_dense_syndromes(&sample.syndrome);
        let _ = decoder.decode(&mut corrections);
        decoder.reset_for_next_cycle();
    }
}

fn benchmark_3d_circuit(
    config: &Grid3DConfig,
    samples: &[SyndromeWithLogical],
    verify: bool,
) -> BenchmarkResults {
    // Calculate buffer size for 3D grid
    let buf_size = required_buffer_size(config.width, config.height, config.depth);
    let mut buffer = vec![0u8; buf_size];
    let mut arena = Arena::new(&mut buffer);

    // Create Grid3D decoder
    let mut decoder = DecoderBuilder::<Grid3D>::new()
        .dimensions_3d(config.width, config.height, config.depth)
        .build(&mut arena)
        .expect("Failed to create 3D decoder");

    // Warmup
    let max_corrections = config.width * config.height * config.depth * 3;
    warmup_decoder(&mut decoder, samples, max_corrections);

    // Benchmark
    let mut times = Vec::with_capacity(samples.len());
    let mut corrections = vec![EdgeCorrection::default(); max_corrections];
    let mut verified = 0;
    let mut total_defects = 0;
    let mut total_corrections = 0;
    let mut logical_errors = 0;

    for sample in samples {
        // Count defects
        total_defects += sample.syndrome.iter().map(|w| w.count_ones() as usize).sum::<usize>();

        // Time decoding
        let t0 = Instant::now();
        decoder.load_dense_syndromes(&sample.syndrome);
        let n = decoder.decode(&mut corrections);
        decoder.reset_for_next_cycle();
        times.push(t0.elapsed());

        total_corrections += n;

        // Verify and track logical errors
        if verify {
            let result = verify_with_logical(&sample.syndrome, &corrections[..n], config);
            if result.defects_resolved {
                verified += 1;
            }
            // Logical error if predicted != actual
            if result.predicted_logical != sample.logical_flips {
                logical_errors += 1;
            }
        } else {
            verified += 1; // Count as verified if verification disabled
        }
    }

    BenchmarkResults {
        times,
        verified,
        total_shots: samples.len(),
        total_defects,
        total_corrections,
        logical_errors,
    }
}

fn get_config_for_distance(d: usize) -> Grid3DConfig {
    match d {
        3 => TestGrids3D::D3,
        5 => TestGrids3D::D5,
        7 => TestGrids3D::D7,
        11 => TestGrids3D::D11,
        17 => TestGrids3D::D17,
        21 => TestGrids3D::D21,
        _ => Grid3DConfig::for_rotated_surface(d),
    }
}

fn main() {
    let args = Args::parse();

    // Determine distances and error rates
    let distances: Vec<usize> = if args.threshold_study {
        THRESHOLD_STUDY_DISTANCES.to_vec()
    } else {
        args.distances.clone()
    };

    let error_probs: Vec<f64> = args.error_probs.clone().unwrap_or_else(|| {
        if args.threshold_study {
            THRESHOLD_STUDY_PROBS.to_vec()
        } else {
            CIRCUIT_ERROR_PROBS.to_vec()
        }
    });

    // Print header (to stderr if CSV mode)
    let print_info = |s: &str| {
        if args.csv {
            eprintln!("{}", s);
        } else {
            println!("{}", s);
        }
    };

    print_info("3D Circuit-Level Threshold Study: prav Union-Find Decoder");
    print_info(&"=".repeat(70));
    print_info("");

    // CSV header
    if args.csv {
        println!("{}", CSV_HEADER);
    }

    // Collect all threshold points for Lambda computation
    let mut all_points: Vec<ThresholdPoint> = Vec::new();

    for &d in &distances {
        let config = get_config_for_distance(d);

        print_info(&format!(
            "Distance d={}, {}x{}x{} grid, {} rounds",
            d, config.width, config.height, config.depth, config.num_rounds
        ));

        for &p in &error_probs {
            // Generate syndromes with logical tracking
            let samples: Vec<SyndromeWithLogical> = if let Some(ref dem_path) = args.dem {
                // Load from DEM file
                let dem_content = std::fs::read_to_string(dem_path)
                    .expect("Failed to read DEM file");
                let parsed = dem::parser::parse_dem(&dem_content)
                    .expect("Failed to parse DEM");
                let mapper = DetectorMapper::new(&config);
                let mut sampler = CircuitSampler::new(&parsed, args.seed);
                sampler
                    .sample_batch(args.shots)
                    .into_iter()
                    .map(|(syn, logical_flips)| {
                        let remapped = mapper.remap_syndrome(&syn, &parsed.detectors);
                        SyndromeWithLogical {
                            syndrome: remapped,
                            logical_flips,
                        }
                    })
                    .collect()
            } else {
                // Generate correlated phenomenological noise
                generate_correlated_syndromes(&config, p, p, args.shots, args.seed)
            };

            // Run benchmark
            let results = benchmark_3d_circuit(&config, &samples, !args.no_verify);

            // Calculate statistics
            let stats = calculate_percentiles(&results.times);

            // Create threshold point
            let point = ThresholdPoint::new(
                d,
                p,
                results.total_shots,
                config.num_rounds,
                results.logical_errors,
                stats.avg_us,
            );

            // Output
            if args.csv {
                println!("{}", point.to_csv());
            } else {
                print_info(&format!(
                    "  p={:.4}: LER={:.2e}/rnd [{:.2e},{:.2e}], n={}, t={:.2}µs",
                    p,
                    point.ler_per_round,
                    point.ler_ci_low,
                    point.ler_ci_high,
                    point.num_shots,
                    point.decode_time_us,
                ));
            }

            all_points.push(point);
        }

        print_info("");
    }

    // Compute and display Lambda suppression factors
    if !args.csv && distances.len() >= 2 {
        print_lambda_table(&all_points, &distances, &error_probs);
    }

    print_info("Benchmark complete.");
}

/// Print Lambda suppression factor table.
fn print_lambda_table(points: &[ThresholdPoint], distances: &[usize], error_probs: &[f64]) {
    println!("Error Suppression Factor Λ (= ε_d / ε_{{d+2}}):");
    println!();

    // Build header
    let mut header = format!("{:>8}", "p");
    for i in 0..distances.len().saturating_sub(1) {
        header.push_str(&format!(" {:>12}", format!("Λ({}→{})", distances[i], distances[i + 1])));
    }
    println!("{}", header);
    println!("{}", "-".repeat(header.len()));

    // For each error rate, compute Lambda between adjacent distances
    for &p in error_probs {
        let mut row = format!("{:>8.4}", p);

        for i in 0..distances.len().saturating_sub(1) {
            let d_low = distances[i];
            let d_high = distances[i + 1];

            // Find points for these distances at this error rate
            let point_low = points.iter().find(|pt| pt.distance == d_low && (pt.physical_error_rate - p).abs() < 1e-9);
            let point_high = points.iter().find(|pt| pt.distance == d_high && (pt.physical_error_rate - p).abs() < 1e-9);

            if let (Some(low), Some(high)) = (point_low, point_high) {
                if let Some(lambda) = SuppressionFactor::from_points(low, high) {
                    row.push_str(&format!(" {:>5.2}±{:<5.2}", lambda.lambda, lambda.lambda_err));
                } else {
                    row.push_str(&format!(" {:>12}", "N/A"));
                }
            } else {
                row.push_str(&format!(" {:>12}", "-"));
            }
        }

        println!("{}", row);
    }

    println!();
}
