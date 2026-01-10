//! # 3D Circuit-Level Benchmark for prav Union-Find QEC Decoder
//!
//! This benchmark tests the **prav** decoder on 3D space-time decoding problems.
//! It measures how often the decoder makes logical errors and how fast it runs.
//!
//! ## What This Tool Does
//!
//! 1. **Generates syndromes**: Creates noisy measurement patterns that a real
//!    quantum computer would produce. Two sources are supported:
//!    - Phenomenological noise (built-in, simplified model)
//!    - Stim DEM files (realistic circuit-level noise)
//!
//! 2. **Runs the decoder**: The prav Union-Find decoder processes each syndrome
//!    and produces a set of corrections (edges to flip).
//!
//! 3. **Verifies correctness**: Applies the corrections and checks that all
//!    detector triggers are resolved.
//!
//! 4. **Tracks logical errors**: Compares the decoder's predicted logical frame
//!    with the ground truth to detect logical errors.
//!
//! 5. **Computes statistics**: Calculates logical error rate (LER), confidence
//!    intervals, error suppression factor (Lambda), and latency percentiles.
//!
//! ## Key Concepts
//!
//! - **Syndrome**: The pattern of triggered detectors. Each detector corresponds
//!   to a stabilizer measurement at a specific (x, y, t) coordinate.
//!
//! - **Logical error**: When the decoder's correction changes the logical qubit
//!   state. The syndrome is resolved, but the quantum information is corrupted.
//!
//! - **Threshold**: The physical error rate below which larger codes perform
//!   better. Measured by the error suppression factor Lambda (Λ).
//!
//! - **Lambda (Λ)**: Ratio of logical error rates between adjacent code distances.
//!   Λ > 1 means we're below threshold (good). Λ < 1 means above threshold (bad).
//!
//! ## Usage Examples
//!
//! ```bash
//! # Run with default settings (phenomenological noise)
//! cargo run --release -p prav-circuit-bench
//!
//! # Run with a single Stim DEM file
//! cargo run --release -p prav-circuit-bench -- --dem path/to/model.dem
//!
//! # Run batch threshold study with directory of DEM files
//! cargo run --release -p prav-circuit-bench -- --dem-dir dems/
//!
//! # Specify distances and shots
//! cargo run --release -p prav-circuit-bench -- --distances 3,5,7,9 --shots 50000
//!
//! # Output CSV for analysis
//! cargo run --release -p prav-circuit-bench -- --csv > results.csv
//! ```
//!
//! ## Output Interpretation
//!
//! The benchmark outputs for each (distance, error_rate) configuration:
//! - `LER`: Logical error rate per round with 95% confidence interval
//! - `n`: Number of shots sampled
//! - `t`: Average decode time in microseconds
//!
//! The Lambda table shows error suppression between adjacent distances.
//! Look for Λ > 1 to confirm you're below threshold.

// Work-in-progress: infrastructure for future benchmarking features
#![allow(dead_code, unused_imports)]

mod color_code_bench;
mod dem;
mod dual_decoder;
mod stats;
mod streaming_bench;
mod surface_code;
mod syndrome;
mod verification;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::Parser;

use prav_core::{
    Arena, CIRCUIT_ERROR_PROBS, DecoderBuilder, DynDecoder, EdgeCorrection, Grid3D, Grid3DConfig,
    ObservableMode, TestGrids3D, required_buffer_size,
};

use crate::color_code_bench::{ColorCodeBenchConfig, benchmark_color_code, generate_color_code_syndromes};
use crate::dual_decoder::{DualDecoderConfig, benchmark_dual_3d};
use crate::stats::{CSV_HEADER, DUAL_CSV_HEADER, DualThresholdPoint, LatencyStats, SuppressionFactor, ThresholdPoint, calculate_percentiles};
use crate::streaming_bench::{STREAMING_CSV_HEADER, StreamingBenchConfig, run_streaming_benchmark};
use crate::surface_code::DetectorMapper;
use crate::syndrome::{CircuitSampler, SyndromeWithLogical, generate_correlated_syndromes};
use crate::verification::verify_with_logical;

/// Command-line arguments for the benchmark.
///
/// The benchmark supports multiple modes of operation:
///
/// 1. **Phenomenological mode** (default): Uses a built-in simplified noise model.
///    Good for quick tests but not accurate for threshold estimation.
///
/// 2. **Single DEM mode** (`--dem`): Load a specific Stim DEM file.
///    Useful for testing a particular circuit configuration.
///
/// 3. **Batch DEM mode** (`--dem-dir`): Process all DEM files in a directory.
///    Best for comprehensive threshold studies.
///
/// # Examples
///
/// ```bash
/// # Default phenomenological mode
/// prav-circuit-bench
///
/// # Batch DEM mode with CSV output
/// prav-circuit-bench --dem-dir dems/ --csv > results.csv
/// ```
#[derive(Parser, Debug)]
#[command(name = "prav-circuit-bench")]
#[command(about = "3D circuit-level benchmark for prav Union-Find QEC decoder")]
struct Args {
    /// Path to a single Stim DEM file.
    ///
    /// When provided, syndromes are sampled from this DEM instead of
    /// using the phenomenological noise model. The DEM specifies the
    /// error mechanisms and their probabilities.
    #[arg(long)]
    dem: Option<PathBuf>,

    /// Directory containing Stim DEM files for batch processing.
    ///
    /// The tool will scan this directory for files matching the pattern:
    /// `surface_d{distance}_r{rounds}_p{noise}.dem`
    ///
    /// This is the recommended mode for threshold studies - it processes
    /// multiple distances and error rates in one run.
    #[arg(long)]
    dem_dir: Option<PathBuf>,

    /// Number of syndrome samples ("shots") per configuration.
    ///
    /// More shots = better statistics but longer runtime.
    /// For rough estimates: 1,000-10,000 shots.
    /// For publication: 100,000+ shots.
    #[arg(long, default_value_t = 10000)]
    shots: usize,

    /// Code distances to benchmark (comma-separated).
    ///
    /// The code distance d determines the grid size: (d-1) x (d-1) x d.
    /// Larger distances provide better error protection but require more
    /// qubits and are slower to decode.
    ///
    /// Common choices: 3, 5, 7, 9, 11, 13
    #[arg(long, value_delimiter = ',', default_values_t = vec![3, 5, 7])]
    distances: Vec<usize>,

    /// Physical error rates to test (comma-separated).
    ///
    /// These are the per-gate or per-timestep error probabilities.
    /// The threshold for surface codes is around 0.5-1%.
    ///
    /// For threshold studies, use values around 0.1% to 1%.
    #[arg(long, value_delimiter = ',')]
    error_probs: Option<Vec<f64>>,

    /// Random seed for reproducibility.
    ///
    /// The same seed produces the same syndrome samples, making
    /// results reproducible across runs.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Skip correction verification.
    ///
    /// When set, the benchmark does not verify that corrections
    /// actually resolve all defects. Faster, but you lose the
    /// correctness check.
    #[arg(long)]
    no_verify: bool,

    /// Output results in CSV format.
    ///
    /// CSV output goes to stdout. Progress messages go to stderr.
    /// Use shell redirection to save: `--csv > results.csv`
    #[arg(long)]
    csv: bool,

    /// Enable threshold study mode.
    ///
    /// Uses a denser grid of error rates around the expected
    /// threshold (0.2% to 1.0%) and more distances by default.
    #[arg(long)]
    threshold_study: bool,

    /// Minimum logical errors required per data point.
    ///
    /// Used for adaptive sampling: keep sampling until we observe
    /// at least this many logical errors. Ensures statistical
    /// significance for low-error-rate configurations.
    #[arg(long, default_value_t = 100)]
    min_errors: usize,

    /// Maximum shots per data point.
    ///
    /// Upper limit on sampling even if min_errors hasn't been reached.
    /// Prevents infinite loops for very low error rates.
    #[arg(long, default_value_t = 1_000_000)]
    max_shots: usize,

    /// Run Helios-compatible benchmark: d=13, p=0.1% phenomenological noise.
    ///
    /// This mode produces output directly comparable to published decoder
    /// benchmarks from the Helios paper (arXiv:2301.08419).
    /// Equivalent to: --distances 13 --error-probs 0.001 --shots 50000
    #[arg(long)]
    helios: bool,

    /// Run quick single-point benchmark at specified distance.
    ///
    /// Uses p=0.1% phenomenological noise with default shots.
    /// Example: --quick-bench 13
    #[arg(long, value_name = "DISTANCE")]
    quick_bench: Option<usize>,

    /// Enable separate X/Z basis decoding.
    ///
    /// When set, syndromes are split into X and Z components which are
    /// decoded independently. This matches real fault-tolerant QEC where
    /// both bases must be decoded. Outputs separate X and Z timing and
    /// logical error rates.
    #[arg(long)]
    dual_decode: bool,

    /// Use 2D mode (single measurement round).
    ///
    /// Runs with depth=1 instead of depth=distance. Useful for testing
    /// decoder performance on simpler problems or when only one round
    /// of syndrome extraction is available.
    #[arg(long)]
    mode_2d: bool,

    /// Run triangular color code benchmark.
    ///
    /// Uses the restriction decoder approach with three parallel Union-Find
    /// decoders (one per color class). Supports (6,6,6) triangular color codes.
    /// Default distances: 3, 5, 7. Default error rate: 1%.
    #[arg(long)]
    color_code: bool,

    /// Run streaming decoder benchmark with sliding window.
    ///
    /// Uses the StreamingDecoder for round-by-round processing instead of
    /// batch decoding. Measures per-round ingest/commit latency and memory usage.
    /// Window size defaults to min(3, distance).
    ///
    /// This mode is designed for real-time QEC where syndromes arrive
    /// round-by-round and corrections must be emitted with minimal latency.
    #[arg(long)]
    streaming: bool,
}

/// Error rates for threshold study mode.
///
/// These values are clustered around the expected threshold (~0.5-1%) for
/// surface codes with circuit-level noise. The denser sampling helps
/// pinpoint the exact threshold location.
const THRESHOLD_STUDY_PROBS: &[f64] = &[
    0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010,
];

/// Default distances for threshold study mode.
///
/// These provide good coverage for determining threshold behavior.
/// Distance 3 is the minimum meaningful surface code.
/// Distance 9 is large enough to show clear threshold crossing.
const THRESHOLD_STUDY_DISTANCES: &[usize] = &[3, 5, 7, 9];

/// Results from benchmarking a batch of syndrome samples.
///
/// This struct collects all the metrics we care about:
/// - **times**: How long each decode took (for latency statistics)
/// - **verified**: How many corrections passed verification
/// - **logical_errors**: How many times the decoder made a logical error
///
/// From these, we compute LER, Lambda, and latency percentiles.
struct BenchmarkResults {
    /// Decode time for each shot. Used to compute latency percentiles.
    times: Vec<Duration>,

    /// Number of shots where verification passed (all defects resolved).
    /// Should equal total_shots if decoder is working correctly.
    verified: usize,

    /// Total number of syndrome samples processed.
    total_shots: usize,

    /// Total number of defects across all samples.
    /// Useful for debugging and understanding error density.
    total_defects: usize,

    /// Total number of edge corrections returned by the decoder.
    total_corrections: usize,

    /// Number of logical errors: shots where predicted_logical != actual_logical.
    /// This is the key metric. LER = logical_errors / (total_shots * rounds).
    logical_errors: usize,
}

/// Warm up the decoder to get stable timing measurements.
///
/// Modern CPUs have caches, branch predictors, and other features that
/// make the first few operations slower than steady-state. We run 200
/// warmup decodes to ensure subsequent timing measurements are accurate.
///
/// # Parameters
///
/// - `decoder`: The prav decoder instance to warm up
/// - `samples`: Syndrome samples to use for warmup
/// - `max_corrections`: Size of the corrections buffer
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

/// Run the core benchmark loop for a batch of syndrome samples.
///
/// This is where the actual decoding and measurement happens:
///
/// 1. Create a prav decoder for the given grid configuration
/// 2. Enable observable tracking (phenomenological mode)
/// 3. Warm up the decoder (200 iterations)
/// 4. For each syndrome sample:
///    - Load the syndrome into the decoder
///    - Time the decode() call
///    - Use decoder's predicted_observables() for logical tracking
///    - Optionally verify corrections resolve all defects
///
/// # Parameters
///
/// - `config`: Grid dimensions (width, height, depth) and layout info
/// - `samples`: Syndrome samples with ground-truth logical flips
/// - `verify`: Whether to verify that corrections resolve all defects
///
/// # Returns
///
/// A `BenchmarkResults` struct with timing data, verification counts,
/// and logical error counts.
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

    // Enable phenomenological observable tracking for proper logical error measurement.
    // This accumulates observable flips as the decoder emits boundary corrections.
    decoder.set_observable_mode(ObservableMode::Phenomenological);

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
        // Get predicted observables before reset (reset clears the accumulator)
        let predicted = decoder.predicted_observables();
        decoder.reset_for_next_cycle();
        times.push(t0.elapsed());

        total_corrections += n;

        // Track logical errors using decoder's built-in observable tracking
        // This compares the decoder's accumulated observable flips with ground truth
        if predicted != sample.logical_flips {
            logical_errors += 1;
        }

        // Optionally verify that corrections resolve all defects
        if verify {
            let result = verify_with_logical(&sample.syndrome, &corrections[..n], config);
            if result.defects_resolved {
                verified += 1;
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

/// Get the grid configuration for a given code distance.
///
/// For common distances (3, 5, 7, 11, 17, 21), we use pre-defined
/// configurations from `TestGrids3D`. For other distances, we
/// compute the configuration dynamically.
///
/// # Grid Dimensions for Rotated Surface Codes
///
/// For a distance-d rotated surface code:
/// - Width = d - 1 (number of X/Z stabilizers per row)
/// - Height = d - 1 (number of rows)
/// - Depth = d (number of measurement rounds, typically equals distance)
///
/// For example, d=5 gives a 4x4x5 detector grid.
fn get_config_for_distance(d: usize) -> Grid3DConfig {
    match d {
        3 => TestGrids3D::D3,
        5 => TestGrids3D::D5,
        7 => TestGrids3D::D7,
        9 => TestGrids3D::D9,
        11 => TestGrids3D::D11,
        13 => TestGrids3D::D13,
        15 => TestGrids3D::D15,
        17 => TestGrids3D::D17,
        21 => TestGrids3D::D21,
        _ => Grid3DConfig::for_rotated_surface(d),
    }
}

/// Information extracted from a DEM filename.
///
/// Our DEM files follow the naming convention:
/// `surface_d{distance}_r{rounds}_p{noise}.dem`
///
/// For example: `surface_d5_r5_p0.0030.dem`
/// - distance = 5
/// - rounds = 5
/// - noise_level = 0.003 (0.3%)
///
/// This struct holds the parsed values plus the file path.
#[derive(Debug, Clone)]
struct DemFileInfo {
    /// Full path to the DEM file.
    path: PathBuf,

    /// Code distance (d). Determines grid size.
    distance: usize,

    /// Number of measurement rounds (r). Usually equals distance.
    rounds: usize,

    /// Physical error rate (p). Probability of each error mechanism.
    noise_level: f64,
}

/// Parse a DEM filename to extract configuration parameters.
///
/// Expected filename format: `surface_d{distance}_r{rounds}_p{noise}.dem`
///
/// # Examples
///
/// - `surface_d5_r5_p0.0030.dem` → distance=5, rounds=5, noise=0.003
/// - `surface_d7_r7_p0.0100.dem` → distance=7, rounds=7, noise=0.01
///
/// # Returns
///
/// `Some(DemFileInfo)` if parsing succeeded, `None` if the filename
/// doesn't match the expected pattern.
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

/// Scan a directory for DEM files and return sorted information.
///
/// Finds all `.dem` files in the directory, parses their filenames,
/// and returns them sorted by (distance, noise_level). This ordering
/// makes the output easier to read and analyze.
///
/// Files that don't match the expected naming pattern are silently skipped.
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

/// Run a batch threshold study using DEM files from a directory.
///
/// This is the main entry point for realistic threshold studies. It:
///
/// 1. Iterates through all DEM files, grouped by distance
/// 2. For each DEM file:
///    - Parses the DEM to extract error mechanisms
///    - Samples syndromes using Monte Carlo simulation
///    - Remaps detector coordinates to prav's layout
///    - Runs the decoder and measures performance
/// 3. Collects results into `ThresholdPoint` structures
///
/// # Parameters
///
/// - `dem_files`: List of DEM file info, pre-sorted by (distance, noise_level)
/// - `shots`: Number of syndrome samples per DEM file
/// - `seed`: Random seed for reproducibility
/// - `no_verify`: Skip verification if true
/// - `csv`: Output CSV format if true
///
/// # Returns
///
/// A vector of `ThresholdPoint` results, one per DEM file.
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

/// Print Helios-compatible benchmark summary with decoder comparison.
///
/// This output format is designed for easy comparison with results from the
/// Helios paper (arXiv:2301.08419) and other QEC decoder benchmarks.
fn print_helios_summary(point: &ThresholdPoint, stats: &stats::LatencyStats) {
    println!();
    println!("Helios-Compatible Benchmark Results");
    println!("====================================");
    println!();
    println!("Configuration:");
    println!("  Distance:        d = {}", point.distance);
    println!("  Error Rate:      p = {:.2}%", point.physical_error_rate * 100.0);
    println!("  Rounds:          {} measurement rounds", point.num_rounds);
    println!("  Shots:           {}", point.num_shots);
    println!("  Grid Size:       {}x{}x{}", point.distance - 1, point.distance - 1, point.num_rounds);
    println!();
    println!("Timing Results:");
    println!("  Total decode:    {:.2} us (avg)", point.decode_time_us);
    println!("  Per round:       {:.0} ns/round", point.time_per_round_ns());
    println!("  Median (p50):    {:.2} us", stats.p50_us);
    println!("  Tail (p99):      {:.2} us", stats.p99_us);
    println!();
    println!("Accuracy:");
    println!("  Logical errors:  {}/{} shots", point.logical_errors, point.num_shots);
    println!("  LER per round:   {:.2e}", point.ler_per_round);
    println!();
    if point.distance == 13 && (point.physical_error_rate - 0.001).abs() < 1e-6 {
        println!("Comparison (d=13, p=0.1%, phenomenological noise):");
        println!("  prav (this):     {:.0} ns/round", point.time_per_round_ns());
        println!("  Sparse Blossom:  160 ns/round (M1 Max, MWPM)");
        println!("  Fusion Blossom:  295 ns/round (M1 Max, MWPM)");
        println!("  Helios (FPGA):   15 ns/round (VCU129, UF)");
    } else {
        println!("Performance:");
        println!("  prav (this):     {:.0} ns/round", point.time_per_round_ns());
        println!();
        println!("Reference (d=13, p=0.1%, Helios paper):");
        println!("  Sparse Blossom:  160 ns/round (M1 Max)");
        println!("  Fusion Blossom:  295 ns/round (M1 Max)");
        println!("  Helios (FPGA):   15 ns/round (VCU129)");
    }
    println!();
    println!("Note: prav is a pure Rust implementation running on CPU.");
    println!("Reference: arXiv:2301.08419, arXiv:2406.08491");
}

/// Run dual decoder benchmark (separate X/Z basis decoding).
///
/// This function handles the `--dual-decode` mode where X and Z stabilizers
/// are decoded independently using separate decoders.
fn run_dual_decode_benchmark(args: &Args) {
    let print_info = |s: &str| {
        if args.csv {
            eprintln!("{}", s);
        } else {
            println!("{}", s);
        }
    };

    let mode_str = if args.mode_2d { "2D" } else { "3D" };
    print_info(&format!("Dual Decoder Benchmark ({} mode): prav Union-Find", mode_str));
    print_info(&"=".repeat(70));
    print_info("");

    if args.csv {
        println!("{}", DUAL_CSV_HEADER);
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

    let mut all_points: Vec<DualThresholdPoint> = Vec::new();

    for &d in &distances {
        // Determine depth based on mode
        let depth = if args.mode_2d { 1 } else { d };
        let config = DualDecoderConfig::new(d, depth);

        let grid_str = format!(
            "{}x{}x{} (X: {}x{}x{}, Z: {}x{}x{})",
            config.unified_config.width,
            config.unified_config.height,
            config.unified_config.depth,
            config.x_config.width,
            config.x_config.height,
            config.x_config.depth,
            config.z_config.width,
            config.z_config.height,
            config.z_config.depth,
        );
        print_info(&format!("Distance d={}, {} grid, {} rounds", d, grid_str, depth));

        for &p in &error_probs {
            // Generate syndromes with unified config
            let samples = generate_correlated_syndromes(
                &config.unified_config,
                p,
                p,
                args.shots,
                args.seed,
            );

            // Run dual benchmark
            let results = benchmark_dual_3d(&config, &samples, !args.no_verify);

            // Calculate average times
            let x_avg_us = results.x_times.iter().map(|d| d.as_secs_f64() * 1e6).sum::<f64>()
                / results.x_times.len().max(1) as f64;
            let z_avg_us = results.z_times.iter().map(|d| d.as_secs_f64() * 1e6).sum::<f64>()
                / results.z_times.len().max(1) as f64;

            // Create threshold point
            let point = DualThresholdPoint::new(
                d,
                p,
                results.total_shots,
                depth,
                results.x_logical_errors,
                results.z_logical_errors,
                results.combined_logical_errors,
                x_avg_us,
                z_avg_us,
            );

            if args.csv {
                println!("{}", point.to_csv());
            } else {
                print_info(&format!(
                    "  p={:.4}: X={:.2e}/rnd Z={:.2e}/rnd Comb={:.2e}/rnd | X:{:.2}µs Z:{:.2}µs Tot:{:.2}µs ({:.0}ns/rnd)",
                    p,
                    point.x_ler_per_round,
                    point.z_ler_per_round,
                    point.combined_ler_per_round,
                    point.x_decode_time_us,
                    point.z_decode_time_us,
                    point.total_decode_time_us,
                    point.time_per_round_ns(),
                ));
            }

            all_points.push(point);
        }

        print_info("");
    }

    if !args.csv {
        print_info("Dual decode benchmark complete.");
    }
}

/// Run triangular color code benchmark.
///
/// This function handles the `--color-code` mode using the restriction decoder
/// approach with three parallel Union-Find decoders (one per color class).
fn run_color_code_benchmark(args: &Args) {
    let print_info = |s: &str| {
        if args.csv {
            eprintln!("{}", s);
        } else {
            println!("{}", s);
        }
    };

    let mode_str = if args.mode_2d { "2D" } else { "3D" };
    print_info(&format!("Triangular Color Code Benchmark ({} mode): prav Restriction Decoder", mode_str));
    print_info(&"=".repeat(70));
    print_info("Architecture: Three parallel Union-Find decoders (Red, Green, Blue)");
    print_info("");

    // CSV header for color code results
    const COLOR_CODE_CSV_HEADER: &str = "distance,error_rate,shots,rounds,logical_errors,ler_per_round,decode_time_us,defects_red,defects_green,defects_blue";
    if args.csv {
        println!("{}", COLOR_CODE_CSV_HEADER);
    }

    // Determine distances and error rates
    let distances: Vec<usize> = if args.threshold_study {
        vec![3, 5, 7, 9]
    } else {
        args.distances.clone()
    };

    let error_probs: Vec<f64> = args.error_probs.clone().unwrap_or_else(|| {
        if args.threshold_study {
            vec![0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010]
        } else {
            vec![0.001, 0.003, 0.005, 0.01]
        }
    });

    for &d in &distances {
        // Create color code config
        let config = if args.mode_2d {
            ColorCodeBenchConfig::new_2d(d)
        } else {
            ColorCodeBenchConfig::new_3d(d)
        };

        print_info(&format!(
            "Distance d={}, {}x{}x{} triangular grid, {} rounds",
            d,
            config.grid_config.width,
            config.grid_config.height,
            config.depth,
            config.depth
        ));

        for &p in &error_probs {
            // Generate syndromes
            let samples = generate_color_code_syndromes(
                &config.grid_config,
                p,
                args.shots,
                args.seed,
            );

            // Run benchmark
            let results = benchmark_color_code(&config, &samples);

            // Calculate LER per round
            let ler_per_round = if config.depth > 0 {
                results.logical_error_rate() / config.depth as f64
            } else {
                results.logical_error_rate()
            };

            if args.csv {
                // CSV output: distance,error_rate,shots,rounds,logical_errors,ler_per_round,decode_time_us,defects_r,defects_g,defects_b
                println!(
                    "{},{:.6},{},{},{},{:.6e},{:.2},{},{},{}",
                    d,
                    p,
                    results.total_shots,
                    config.depth,
                    results.logical_errors,
                    ler_per_round,
                    results.avg_decode_time_us(),
                    results.defects_by_color[0],
                    results.defects_by_color[1],
                    results.defects_by_color[2],
                );
            } else {
                print_info(&format!(
                    "  p={:.4}: LER={:.2e}/rnd, n={}, t={:.2}µs, defects[R/G/B]={}/{}/{}",
                    p,
                    ler_per_round,
                    results.total_shots,
                    results.avg_decode_time_us(),
                    results.defects_by_color[0],
                    results.defects_by_color[1],
                    results.defects_by_color[2],
                ));
            }
        }

        print_info("");
    }

    if !args.csv {
        print_info("Color code benchmark complete.");
        print_info("");
        print_info("Note: The restriction decoder projects the color code onto three");
        print_info("surface-code-like subgraphs, one per color. Each subgraph is decoded");
        print_info("independently using Union-Find, and corrections are lifted back.");
        print_info("");
        print_info("Reference: https://quantum-journal.org/papers/q-2023-02-21-929/");
    }
}

/// Run streaming decoder benchmark.
///
/// This function handles the `--streaming` mode which uses the sliding window
/// StreamingDecoder for round-by-round processing. It measures:
///
/// - **Ingest latency**: Time to load one round's syndromes and grow clusters
/// - **Commit latency**: Time to extract corrections when a round exits the window
/// - **Flush latency**: Time to commit remaining rounds at stream end
/// - **Memory usage**: Bytes allocated for the streaming decoder
fn run_streaming_decoder_benchmark(args: &Args) {
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

    let results = run_streaming_benchmark(
        &distances,
        &error_probs,
        args.shots,
        args.seed,
        args.csv,
    );

    // Output is already printed by run_streaming_benchmark
    let _ = results;
}

fn main() {
    let mut args = Args::parse();

    // Handle --helios mode: d=13, p=0.1%, 50000 shots
    if args.helios {
        args.distances = vec![13];
        args.error_probs = Some(vec![0.001]);
        args.shots = 50000;
    }

    // Handle --quick-bench mode
    if let Some(d) = args.quick_bench {
        args.distances = vec![d];
        args.error_probs = Some(vec![0.001]);
    }

    // Handle --dual-decode mode
    if args.dual_decode {
        run_dual_decode_benchmark(&args);
        return;
    }

    // Handle --color-code mode (triangular color code)
    if args.color_code {
        run_color_code_benchmark(&args);
        return;
    }

    // Handle --streaming mode (sliding window decoder)
    if args.streaming {
        run_streaming_decoder_benchmark(&args);
        return;
    }

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
            } else if args.helios || args.quick_bench.is_some() {
                // Helios mode: show detailed summary at the end
                print_helios_summary(&point, &stats);
            } else {
                print_info(&format!(
                    "  p={:.4}: LER={:.2e}/rnd [{:.2e},{:.2e}], n={}, t={:.2}µs ({:.0} ns/rnd)",
                    p,
                    point.ler_per_round,
                    point.ler_ci_low,
                    point.ler_ci_high,
                    point.num_shots,
                    point.decode_time_us,
                    point.time_per_round_ns(),
                ));
            }

            all_points.push(point);
        }

        if !args.helios && args.quick_bench.is_none() {
            print_info("");
        }
    }

    // Compute and display Lambda suppression factors
    if !args.csv && distances.len() >= 2 && !args.helios && args.quick_bench.is_none() {
        print_lambda_table(&all_points, &distances, &error_probs);
    }

    if !args.csv && !args.helios && args.quick_bench.is_none() {
        print_info("Benchmark complete.");
    }
}

/// Print the Lambda (error suppression factor) table.
///
/// Lambda (Λ) is the key metric for threshold analysis. It shows how much
/// better a larger code performs compared to a smaller one:
///
/// ```text
/// Λ = LER(d) / LER(d+2)
/// ```
///
/// - **Λ > 1**: Larger code is better → below threshold ✓
/// - **Λ < 1**: Larger code is worse → above threshold ✗
/// - **Λ = 1**: At threshold
///
/// The table shows Λ for each pair of adjacent distances at each error rate.
/// The threshold is where Λ crosses 1.0.
///
/// # Example Output
///
/// ```text
/// Error Suppression Factor Λ (= ε_d / ε_{d+2}):
///
///        p      Λ(3→5)       Λ(5→7)
/// ---------------------------------
///   0.0010  5.20±1.10   4.80±0.95
///   0.0030  2.15±0.42   1.95±0.38
///   0.0050  1.32±0.25   1.15±0.22
/// ```
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
