//! # Streaming Decoder Benchmark Module
//!
//! Benchmarks for the sliding window streaming decoder that processes
//! syndrome measurements round-by-round for real-time QEC.
//!
//! ## Key Metrics
//!
//! - **Ingest latency**: Time to load one round's syndromes and grow clusters
//! - **Commit latency**: Time to extract corrections when a round exits the window
//! - **Per-round latency**: Total processing time per measurement round
//! - **Flush latency**: Time to commit all remaining rounds at stream end
//!
//! ## Architecture
//!
//! ```text
//! Round N arrives:
//! ┌─────────────────────────────────────────────────────┐
//! │ Sliding Window (size W)                             │
//! │  ┌───────┬───────┬───────┬───────┐                 │
//! │  │ R(N-W)│  ...  │ R(N-1)│  R(N) │  ← New round    │
//! │  │ EXIT  │       │       │ LOAD  │                  │
//! │  └───┬───┴───────┴───────┴───────┘                 │
//! │      │                                              │
//! │      ▼                                              │
//! │  Commit corrections for R(N-W)                      │
//! └─────────────────────────────────────────────────────┘
//! ```

use std::time::{Duration, Instant};

use prav_core::{
    Arena, Grid3D, StreamingConfig, StreamingDecoder, streaming_buffer_size,
};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::stats::{LatencyStats, ThresholdPoint, calculate_percentiles};

/// Configuration for streaming decoder benchmarks.
#[derive(Debug, Clone)]
pub struct StreamingBenchConfig {
    /// Code distance.
    pub distance: usize,
    /// Sliding window size (number of rounds kept in memory).
    pub window_size: usize,
    /// Physical error rate.
    pub error_prob: f64,
    /// Number of complete streams to process.
    pub num_streams: usize,
    /// Rounds per stream (typically equals distance).
    pub rounds_per_stream: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl StreamingBenchConfig {
    /// Create configuration for a rotated surface code.
    pub fn for_surface_code(distance: usize, error_prob: f64, num_streams: usize, seed: u64) -> Self {
        Self {
            distance,
            window_size: 3.min(distance), // Default window size
            error_prob,
            num_streams,
            rounds_per_stream: distance,
            seed,
        }
    }

    /// Set custom window size.
    pub fn with_window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }
}

/// Results from streaming decoder benchmark.
#[derive(Debug, Clone)]
pub struct StreamingThresholdPoint {
    /// Base threshold point (distance, error rate, LER, etc.).
    pub base: ThresholdPoint,

    /// Sliding window size used.
    pub window_size: usize,

    /// Per-round ingest latency statistics (time to load syndromes + grow).
    pub ingest_latency: LatencyStats,

    /// Commit latency statistics (time to extract corrections for exiting round).
    pub commit_latency: LatencyStats,

    /// Flush latency per round (time to commit remaining rounds at stream end).
    pub flush_latency_per_round_us: f64,

    /// Memory usage in bytes for the streaming decoder.
    pub memory_bytes: usize,

    /// Average corrections per round.
    pub corrections_per_round: f64,
}

impl StreamingThresholdPoint {
    /// Format as CSV row.
    pub fn to_csv(&self) -> String {
        format!(
            "{},{:.6},{},{},{},{},{:.6e},{:.6e},{:.6e},{:.3},{:.3},{:.3},{:.3},{:.3},{}",
            self.base.distance,
            self.base.physical_error_rate,
            self.window_size,
            self.base.num_rounds,
            self.base.num_shots,
            self.base.logical_errors,
            self.base.ler_per_round,
            self.base.ler_ci_low,
            self.base.ler_ci_high,
            self.ingest_latency.avg_us,
            self.commit_latency.avg_us,
            self.flush_latency_per_round_us,
            self.base.decode_time_us,
            self.base.time_per_round_us,
            self.memory_bytes,
        )
    }

    /// Per-round processing time in nanoseconds.
    pub fn per_round_ns(&self) -> f64 {
        self.base.time_per_round_us * 1000.0
    }
}

/// CSV header for streaming benchmark output.
pub const STREAMING_CSV_HEADER: &str = "distance,physical_p,window_size,rounds,shots,logical_errors,ler_per_round,ler_ci_low,ler_ci_high,ingest_avg_us,commit_avg_us,flush_per_round_us,total_decode_us,time_per_round_us,memory_bytes";

/// Run streaming decoder benchmark with the given configuration.
pub fn benchmark_streaming(config: &StreamingBenchConfig) -> StreamingThresholdPoint {
    // Calculate grid dimensions for rotated surface code
    let width = config.distance - 1;
    let height = config.distance - 1;

    // Create streaming configuration
    let streaming_config = StreamingConfig::for_rotated_surface(config.distance, config.window_size);

    // Allocate buffer
    let buf_size = streaming_buffer_size(width, height, config.window_size);
    let mut buffer = vec![0u8; buf_size];
    let mut arena = Arena::new(&mut buffer);

    // Create decoder
    // STRIDE_Y needs to be power of 2 >= max(width, height, window_size)
    let stride_y = width.max(height).max(config.window_size).next_power_of_two();

    // We need to handle different STRIDE_Y values at runtime
    // For simplicity, we'll use a macro or match on common values
    let result = match stride_y {
        1 => run_streaming_benchmark_inner::<1>(config, streaming_config, &mut arena),
        2 => run_streaming_benchmark_inner::<2>(config, streaming_config, &mut arena),
        4 => run_streaming_benchmark_inner::<4>(config, streaming_config, &mut arena),
        8 => run_streaming_benchmark_inner::<8>(config, streaming_config, &mut arena),
        16 => run_streaming_benchmark_inner::<16>(config, streaming_config, &mut arena),
        32 => run_streaming_benchmark_inner::<32>(config, streaming_config, &mut arena),
        64 => run_streaming_benchmark_inner::<64>(config, streaming_config, &mut arena),
        _ => panic!("Unsupported STRIDE_Y: {}", stride_y),
    };

    StreamingThresholdPoint {
        base: result.threshold_point,
        window_size: config.window_size,
        ingest_latency: result.ingest_latency,
        commit_latency: result.commit_latency,
        flush_latency_per_round_us: result.flush_latency_per_round_us,
        memory_bytes: buf_size,
        corrections_per_round: result.corrections_per_round,
    }
}

struct StreamingBenchResult {
    threshold_point: ThresholdPoint,
    ingest_latency: LatencyStats,
    commit_latency: LatencyStats,
    flush_latency_per_round_us: f64,
    corrections_per_round: f64,
}

/// Generate simple per-round syndromes for streaming decoder.
///
/// This generates syndromes directly in the per-round format expected by
/// the streaming decoder, with each round having `words_per_round` u64 words.
fn generate_streaming_syndromes(
    width: usize,
    height: usize,
    num_rounds: usize,
    error_prob: f64,
    seed: u64,
) -> (Vec<Vec<u64>>, u8) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let detectors_per_round = width * height;
    let words_per_round = (detectors_per_round + 63) / 64;

    let mut rounds = Vec::with_capacity(num_rounds);
    let mut logical_flips = 0u8;

    for _ in 0..num_rounds {
        let mut round_syndrome = vec![0u64; words_per_round];

        // Generate random errors for this round
        for y in 0..height {
            for x in 0..width {
                // Left/right boundary errors affect Z logical
                if x == 0 && rng.random::<f64>() < error_prob {
                    let bit = y * width + x;
                    round_syndrome[bit / 64] ^= 1u64 << (bit % 64);
                    logical_flips ^= 0b10; // Z logical
                }
                if x == width - 1 && rng.random::<f64>() < error_prob {
                    let bit = y * width + x;
                    round_syndrome[bit / 64] ^= 1u64 << (bit % 64);
                    logical_flips ^= 0b10; // Z logical
                }

                // Top/bottom boundary errors affect X logical
                if y == 0 && rng.random::<f64>() < error_prob {
                    let bit = y * width + x;
                    round_syndrome[bit / 64] ^= 1u64 << (bit % 64);
                    logical_flips ^= 0b01; // X logical
                }
                if y == height - 1 && rng.random::<f64>() < error_prob {
                    let bit = y * width + x;
                    round_syndrome[bit / 64] ^= 1u64 << (bit % 64);
                    logical_flips ^= 0b01; // X logical
                }

                // Interior errors (create pairs of defects)
                if x < width - 1 && rng.random::<f64>() < error_prob {
                    let bit1 = y * width + x;
                    let bit2 = y * width + x + 1;
                    round_syndrome[bit1 / 64] ^= 1u64 << (bit1 % 64);
                    round_syndrome[bit2 / 64] ^= 1u64 << (bit2 % 64);
                }
                if y < height - 1 && rng.random::<f64>() < error_prob {
                    let bit1 = y * width + x;
                    let bit2 = (y + 1) * width + x;
                    round_syndrome[bit1 / 64] ^= 1u64 << (bit1 % 64);
                    round_syndrome[bit2 / 64] ^= 1u64 << (bit2 % 64);
                }
            }
        }

        rounds.push(round_syndrome);
    }

    (rounds, logical_flips)
}

fn run_streaming_benchmark_inner<const STRIDE_Y: usize>(
    config: &StreamingBenchConfig,
    streaming_config: StreamingConfig,
    arena: &mut Arena<'_>,
) -> StreamingBenchResult {
    let mut decoder: StreamingDecoder<Grid3D, STRIDE_Y> =
        StreamingDecoder::new(arena, streaming_config);

    let width = config.distance - 1;
    let height = config.distance - 1;

    // Timing accumulators
    let mut ingest_times: Vec<Duration> = Vec::with_capacity(config.num_streams * config.rounds_per_stream);
    let mut commit_times: Vec<Duration> = Vec::with_capacity(config.num_streams * config.rounds_per_stream);
    let mut flush_times: Vec<Duration> = Vec::new();
    let mut total_decode_times: Vec<Duration> = Vec::with_capacity(config.num_streams);

    let mut logical_errors = 0usize;
    let mut total_corrections = 0usize;
    let mut total_rounds = 0usize;

    // Process each stream
    for stream_idx in 0..config.num_streams {
        let stream_seed = config.seed.wrapping_add(stream_idx as u64);

        // Generate per-round syndromes for this stream
        let (round_syndromes, actual_logical) = generate_streaming_syndromes(
            width,
            height,
            config.rounds_per_stream,
            config.error_prob,
            stream_seed,
        );

        decoder.reset();

        let stream_start = Instant::now();
        let mut predicted_logical = 0u8;
        let mut stream_corrections = 0usize;

        // Process each round
        for round_syndrome in &round_syndromes {
            let ingest_start = Instant::now();
            let committed = decoder.ingest_round(round_syndrome);
            let ingest_time = ingest_start.elapsed();
            ingest_times.push(ingest_time);

            if let Some(corrections) = committed {
                let commit_start = Instant::now();
                // Process committed corrections
                stream_corrections += corrections.corrections.len();
                predicted_logical ^= corrections.observable;
                commit_times.push(commit_start.elapsed());
            }

            total_rounds += 1;
        }

        // Flush remaining rounds
        let flush_start = Instant::now();
        let mut flush_round_count = 0;
        for corrections in decoder.flush() {
            stream_corrections += corrections.corrections.len();
            predicted_logical ^= corrections.observable;
            flush_round_count += 1;
        }
        let flush_time = flush_start.elapsed();
        if flush_round_count > 0 {
            flush_times.push(Duration::from_nanos(
                flush_time.as_nanos() as u64 / flush_round_count as u64
            ));
        }

        let stream_time = stream_start.elapsed();
        total_decode_times.push(stream_time);
        total_corrections += stream_corrections;

        // Check for logical error
        if predicted_logical != actual_logical {
            logical_errors += 1;
        }
    }

    // Calculate statistics
    let ingest_latency = calculate_percentiles(&ingest_times);
    let commit_latency = if commit_times.is_empty() {
        LatencyStats {
            avg_us: 0.0,
            min_us: 0.0,
            max_us: 0.0,
            p50_us: 0.0,
            p95_us: 0.0,
            p99_us: 0.0,
        }
    } else {
        calculate_percentiles(&commit_times)
    };

    let flush_latency_per_round_us = if flush_times.is_empty() {
        0.0
    } else {
        flush_times.iter().map(|d| d.as_secs_f64() * 1e6).sum::<f64>() / flush_times.len() as f64
    };

    let avg_decode_us = if total_decode_times.is_empty() {
        0.0
    } else {
        total_decode_times.iter().map(|d| d.as_secs_f64() * 1e6).sum::<f64>()
            / total_decode_times.len() as f64
    };

    let corrections_per_round = if total_rounds > 0 {
        total_corrections as f64 / total_rounds as f64
    } else {
        0.0
    };

    let threshold_point = ThresholdPoint::new(
        config.distance,
        config.error_prob,
        config.num_streams,
        config.rounds_per_stream,
        logical_errors,
        avg_decode_us,
    );

    StreamingBenchResult {
        threshold_point,
        ingest_latency,
        commit_latency,
        flush_latency_per_round_us,
        corrections_per_round,
    }
}

/// Run the full streaming benchmark suite and print results.
pub fn run_streaming_benchmark(
    distances: &[usize],
    error_probs: &[f64],
    shots: usize,
    seed: u64,
    csv_output: bool,
) -> Vec<StreamingThresholdPoint> {
    let mut results = Vec::new();

    if csv_output {
        println!("{}", STREAMING_CSV_HEADER);
    } else {
        println!();
        println!("Streaming Decoder Benchmark: prav Union-Find (Sliding Window)");
        println!("==============================================================");
        println!();
    }

    for &distance in distances {
        let window_size = 3.min(distance);

        if !csv_output {
            println!(
                "Distance d={}, {}x{}x{} grid, window={} rounds",
                distance,
                distance - 1,
                distance - 1,
                distance,
                window_size
            );
        }

        for &error_prob in error_probs {
            let config = StreamingBenchConfig::for_surface_code(distance, error_prob, shots, seed)
                .with_window_size(window_size);

            let point = benchmark_streaming(&config);

            if csv_output {
                println!("{}", point.to_csv());
            } else {
                println!(
                    "  p={:.4}: LER={:.2e}/rnd [{:.2e},{:.2e}], n={}, ingest={:.2}µs, commit={:.2}µs, total={:.2}µs ({:.0}ns/rnd)",
                    error_prob,
                    point.base.ler_per_round,
                    point.base.ler_ci_low,
                    point.base.ler_ci_high,
                    point.base.num_shots,
                    point.ingest_latency.avg_us,
                    point.commit_latency.avg_us,
                    point.base.decode_time_us,
                    point.per_round_ns(),
                );
            }

            results.push(point);
        }

        if !csv_output {
            println!();
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_benchmark_basic() {
        let config = StreamingBenchConfig::for_surface_code(3, 0.01, 100, 42);
        let result = benchmark_streaming(&config);

        assert_eq!(result.base.distance, 3);
        assert_eq!(result.base.num_shots, 100);
        assert!(result.base.decode_time_us > 0.0);
        assert!(result.ingest_latency.avg_us >= 0.0);
    }

    #[test]
    fn test_streaming_config() {
        let config = StreamingBenchConfig::for_surface_code(5, 0.001, 1000, 12345)
            .with_window_size(4);

        assert_eq!(config.distance, 5);
        assert_eq!(config.window_size, 4);
        assert_eq!(config.rounds_per_stream, 5);
    }
}
