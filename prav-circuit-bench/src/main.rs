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

// Work-in-progress: infrastructure for future benchmarking features
#![allow(dead_code, unused_imports)]

mod dem;
mod stats;
mod surface_code;
mod syndrome;
mod verification;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;

use prav_core::{
    Arena, CIRCUIT_ERROR_PROBS, DecoderBuilder, DynDecoder, EdgeCorrection, Grid3D, Grid3DConfig,
    TestGrids3D, required_buffer_size,
};

use crate::stats::{CSV_HEADER, SuppressionFactor, ThresholdPoint, calculate_percentiles};
use crate::surface_code::DetectorMapper;
use crate::syndrome::{CircuitSampler, SyndromeWithLogical, generate_correlated_syndromes};
use crate::verification::verify_with_logical;

#[derive(Parser, Debug)]
#[command(name = "prav-circuit-bench")]
#[command(about = "3D circuit-level benchmark for prav Union-Find QEC decoder")]
struct Args {
    /// Path to Stim DEM file (optional, uses phenomenological if not provided)
    #[arg(long)]
    dem: Option<PathBuf>,

    /// Directory containing Stim DEM files (runs batch threshold study)
    /// DEM filenames should match pattern: surface_d{distance}_r{rounds}_p{noise}.dem
    #[arg(long)]
    dem_dir: Option<PathBuf>,

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

fn warmup_decoder(
    decoder: &mut DynDecoder<Grid3D>,
    samples: &[SyndromeWithLogical],
    max_corrections: usize,
) {
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
        total_defects += sample
            .syndrome
            .iter()
            .map(|w| w.count_ones() as usize)
            .sum::<usize>();

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

/// Parsed DEM file info extracted from filename.
#[derive(Debug, Clone)]
struct DemFileInfo {
    path: PathBuf,
    distance: usize,
    rounds: usize,
    noise_level: f64,
}

/// Parse DEM filename to extract distance, rounds, and noise level.
/// Expected format: surface_d{distance}_r{rounds}_p{noise}.dem
fn parse_dem_filename(path: &PathBuf) -> Option<DemFileInfo> {
    let filename = path.file_stem()?.to_str()?;

    // Parse distance: d{N}
    let d_idx = filename.find("_d")?;
    let after_d = &filename[d_idx + 2..];
    let d_end = after_d.find('_').unwrap_or(after_d.len());
    let distance: usize = after_d[..d_end].parse().ok()?;

    // Parse rounds: r{N}
    let r_idx = filename.find("_r")?;
    let after_r = &filename[r_idx + 2..];
    let r_end = after_r.find('_').unwrap_or(after_r.len());
    let rounds: usize = after_r[..r_end].parse().ok()?;

    // Parse noise: p{N.NNNN}
    let p_idx = filename.find("_p")?;
    let after_p = &filename[p_idx + 2..];
    let noise_level: f64 = after_p.parse().ok()?;

    Some(DemFileInfo {
        path: path.clone(),
        distance,
        rounds,
        noise_level,
    })
}

/// Scan a directory for DEM files and return sorted info.
fn scan_dem_directory(dir: &PathBuf) -> Vec<DemFileInfo> {
    let mut files = Vec::new();

    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "dem") {
                if let Some(info) = parse_dem_filename(&path) {
                    files.push(info);
                }
            }
        }
    }

    // Sort by distance, then by noise level
    files.sort_by(|a, b| {
        a.distance
            .cmp(&b.distance)
            .then(a.noise_level.partial_cmp(&b.noise_level).unwrap())
    });

    files
}

/// Run threshold study using DEM files from a directory.
fn run_dem_directory_benchmark(
    dem_files: &[DemFileInfo],
    shots: usize,
    seed: u64,
    no_verify: bool,
    csv: bool,
) -> Vec<ThresholdPoint> {
    let print_info = |s: &str| {
        if csv {
            eprintln!("{}", s);
        } else {
            println!("{}", s);
        }
    };

    print_info("3D Circuit-Level Threshold Study: prav Union-Find Decoder (Stim DEMs)");
    print_info(&"=".repeat(70));
    print_info("");

    if csv {
        println!("{}", CSV_HEADER);
    }

    let mut all_points = Vec::new();
    let mut current_distance = 0;

    for dem_info in dem_files {
        // Print distance header when it changes
        if dem_info.distance != current_distance {
            current_distance = dem_info.distance;
            let config = get_config_for_distance(current_distance);
            print_info(&format!(
                "Distance d={}, {}x{}x{} grid, {} rounds (from Stim DEM)",
                current_distance,
                config.width,
                config.height,
                config.depth,
                dem_info.rounds
            ));
        }

        // Load and parse DEM
        let dem_content = match std::fs::read_to_string(&dem_info.path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Warning: Failed to read {:?}: {}", dem_info.path, e);
                continue;
            }
        };
        let parsed = match dem::parser::parse_dem(&dem_content) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Warning: Failed to parse {:?}: {}", dem_info.path, e);
                continue;
            }
        };

        // Create grid config from DEM info
        let config = Grid3DConfig::for_rotated_surface(dem_info.distance);
        let mapper = DetectorMapper::new(&config);

        // Sample syndromes from DEM
        let mut sampler = CircuitSampler::new(&parsed, seed);
        let samples: Vec<SyndromeWithLogical> = sampler
            .sample_batch(shots)
            .into_iter()
            .map(|(syn, logical_flips)| {
                let remapped = mapper.remap_syndrome(&syn, &parsed.detectors);
                SyndromeWithLogical {
                    syndrome: remapped,
                    logical_flips,
                }
            })
            .collect();

        // Run benchmark
        let results = benchmark_3d_circuit(&config, &samples, !no_verify);
        let stats = calculate_percentiles(&results.times);

        // Create threshold point
        let point = ThresholdPoint::new(
            dem_info.distance,
            dem_info.noise_level,
            results.total_shots,
            dem_info.rounds,
            results.logical_errors,
            stats.avg_us,
        );

        if csv {
            println!("{}", point.to_csv());
        } else {
            print_info(&format!(
                "  p={:.4}: LER={:.2e}/rnd [{:.2e},{:.2e}], n={}, t={:.2}µs",
                dem_info.noise_level,
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
    all_points
}

fn main() {
    let args = Args::parse();

    // Handle --dem-dir mode (batch DEM processing)
    if let Some(ref dem_dir) = args.dem_dir {
        let dem_files = scan_dem_directory(dem_dir);
        if dem_files.is_empty() {
            eprintln!("Error: No DEM files found in {:?}", dem_dir);
            eprintln!("Expected filename format: surface_d{{distance}}_r{{rounds}}_p{{noise}}.dem");
            std::process::exit(1);
        }

        eprintln!("Found {} DEM files in {:?}", dem_files.len(), dem_dir);

        let all_points =
            run_dem_directory_benchmark(&dem_files, args.shots, args.seed, args.no_verify, args.csv);

        // Extract unique distances and error rates for Lambda table
        let mut distances: Vec<usize> = all_points.iter().map(|p| p.distance).collect();
        distances.sort();
        distances.dedup();

        let mut error_probs: Vec<f64> = all_points.iter().map(|p| p.physical_error_rate).collect();
        error_probs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        error_probs.dedup_by(|a, b| (*a - *b).abs() < 1e-9);

        // Print Lambda table
        if !args.csv && distances.len() >= 2 {
            print_lambda_table(&all_points, &distances, &error_probs);
        }

        if !args.csv {
            println!("Benchmark complete.");
        }
        return;
    }

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
                let dem_content =
                    std::fs::read_to_string(dem_path).expect("Failed to read DEM file");
                let parsed = dem::parser::parse_dem(&dem_content).expect("Failed to parse DEM");
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
        header.push_str(&format!(
            " {:>12}",
            format!("Λ({}→{})", distances[i], distances[i + 1])
        ));
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
            let point_low = points
                .iter()
                .find(|pt| pt.distance == d_low && (pt.physical_error_rate - p).abs() < 1e-9);
            let point_high = points
                .iter()
                .find(|pt| pt.distance == d_high && (pt.physical_error_rate - p).abs() < 1e-9);

            if let (Some(low), Some(high)) = (point_low, point_high) {
                if let Some(lambda) = SuppressionFactor::from_points(low, high) {
                    row.push_str(&format!(
                        " {:>5.2}±{:<5.2}",
                        lambda.lambda, lambda.lambda_err
                    ));
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
