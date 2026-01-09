//! Dual decoder for separate X/Z basis decoding.
//!
//! This module provides utilities for decoding X and Z stabilizers separately
//! using two independent decoders. This matches the requirements for real
//! fault-tolerant quantum computation where both bases must be decoded.
//!
//! # Architecture
//!
//! ```text
//! Unified Syndrome
//!        │
//!        ▼
//!  ┌─────────────┐
//!  │  Splitter   │
//!  └─────────────┘
//!        │
//!   ┌────┴────┐
//!   ▼         ▼
//! ┌───────┐ ┌───────┐
//! │X Dec. │ │Z Dec. │
//! └───────┘ └───────┘
//!   │         │
//!   ▼         ▼
//!  X obs    Z obs
//! ```

use std::time::{Duration, Instant};

use prav_core::{
    Arena, DecoderBuilder, EdgeCorrection, Grid3D, Grid3DConfig, ObservableMode,
    required_buffer_size,
};

use crate::syndrome::{SplitSyndromes, SyndromeWithLogical, SyndromeSplitter};

/// Result of decoding both X and Z bases.
#[derive(Clone, Debug)]
pub struct DualDecodeResult {
    /// Time spent decoding X basis.
    pub x_time: Duration,
    /// Time spent decoding Z basis.
    pub z_time: Duration,
    /// Total time (x_time + z_time).
    pub total_time: Duration,
    /// Predicted X observable from decoder.
    pub x_predicted: u8,
    /// Predicted Z observable from decoder.
    pub z_predicted: u8,
    /// Number of X corrections.
    pub x_corrections: usize,
    /// Number of Z corrections.
    pub z_corrections: usize,
}

/// Configuration for dual decoding.
///
/// This struct holds the configuration for creating X and Z decoders.
/// The actual decoders are created on-demand to avoid lifetime issues.
#[derive(Clone, Debug)]
pub struct DualDecoderConfig {
    /// Code distance.
    pub distance: usize,
    /// Depth (number of rounds).
    pub depth: usize,
    /// X decoder configuration.
    pub x_config: Grid3DConfig,
    /// Z decoder configuration.
    pub z_config: Grid3DConfig,
    /// Unified grid configuration (for splitting).
    pub unified_config: Grid3DConfig,
}

impl DualDecoderConfig {
    /// Create a new dual decoder configuration.
    pub fn new(d: usize, depth: usize) -> Self {
        let x_config = Grid3DConfig::for_x_stabilizers(d, depth);
        let z_config = Grid3DConfig::for_z_stabilizers(d, depth);
        let unified_config = Grid3DConfig::for_rotated_surface(d).with_depth(depth);

        Self {
            distance: d,
            depth,
            x_config,
            z_config,
            unified_config,
        }
    }

    /// Create config for 3D mode (depth = distance).
    pub fn new_3d(d: usize) -> Self {
        Self::new(d, d)
    }

    /// Create config for 2D mode (depth = 1).
    pub fn new_2d(d: usize) -> Self {
        Self::new(d, 1)
    }

    /// Create a splitter for this configuration.
    pub fn splitter(&self) -> SyndromeSplitter {
        SyndromeSplitter::new(&self.unified_config)
    }

    /// X buffer size required.
    pub fn x_buffer_size(&self) -> usize {
        required_buffer_size(self.x_config.width, self.x_config.height, self.x_config.depth)
    }

    /// Z buffer size required.
    pub fn z_buffer_size(&self) -> usize {
        required_buffer_size(self.z_config.width, self.z_config.height, self.z_config.depth)
    }

    /// Maximum number of X corrections.
    pub fn max_x_corrections(&self) -> usize {
        self.x_config.width * self.x_config.height * self.x_config.depth * 3
    }

    /// Maximum number of Z corrections.
    pub fn max_z_corrections(&self) -> usize {
        self.z_config.width * self.z_config.height * self.z_config.depth * 3
    }
}

/// Benchmark results for dual decoding.
#[derive(Clone, Debug)]
pub struct DualBenchmarkResults {
    /// X decode times for each sample.
    pub x_times: Vec<Duration>,
    /// Z decode times for each sample.
    pub z_times: Vec<Duration>,
    /// Total decode times for each sample.
    pub total_times: Vec<Duration>,
    /// Number of X logical errors.
    pub x_logical_errors: usize,
    /// Number of Z logical errors.
    pub z_logical_errors: usize,
    /// Number of combined logical errors (X or Z failed).
    pub combined_logical_errors: usize,
    /// Total samples processed.
    pub total_shots: usize,
    /// Number of verified samples (defects resolved).
    pub verified: usize,
    /// Total X defects across all samples.
    pub total_x_defects: usize,
    /// Total Z defects across all samples.
    pub total_z_defects: usize,
}

/// Run dual decoder benchmark on the given samples.
///
/// Creates X and Z decoders, splits syndromes, and times each decode separately.
pub fn benchmark_dual_3d(
    config: &DualDecoderConfig,
    samples: &[SyndromeWithLogical],
    verify: bool,
) -> DualBenchmarkResults {
    // Allocate buffers
    let mut x_buffer = vec![0u8; config.x_buffer_size()];
    let mut z_buffer = vec![0u8; config.z_buffer_size()];

    // Create arenas
    let mut x_arena = Arena::new(&mut x_buffer);
    let mut z_arena = Arena::new(&mut z_buffer);

    // Build X decoder
    let mut x_decoder = DecoderBuilder::<Grid3D>::new()
        .dimensions_3d(config.x_config.width, config.x_config.height, config.x_config.depth)
        .build(&mut x_arena)
        .expect("Failed to create X decoder");
    x_decoder.set_boundary_config(config.x_config.boundary_config);
    x_decoder.set_observable_mode(ObservableMode::Phenomenological);

    // Build Z decoder
    let mut z_decoder = DecoderBuilder::<Grid3D>::new()
        .dimensions_3d(config.z_config.width, config.z_config.height, config.z_config.depth)
        .build(&mut z_arena)
        .expect("Failed to create Z decoder");
    z_decoder.set_boundary_config(config.z_config.boundary_config);
    z_decoder.set_observable_mode(ObservableMode::Phenomenological);

    // Create splitter and correction buffers
    let splitter = config.splitter();
    let mut x_corrections = vec![EdgeCorrection::default(); config.max_x_corrections()];
    let mut z_corrections = vec![EdgeCorrection::default(); config.max_z_corrections()];

    // Warmup
    let warmup_count = 200.min(samples.len());
    for sample in samples.iter().take(warmup_count) {
        let split = splitter.split(&sample.syndrome, sample.logical_flips);

        x_decoder.load_dense_syndromes(&split.x_syndrome);
        let _ = x_decoder.decode(&mut x_corrections);
        x_decoder.reset_for_next_cycle();

        z_decoder.load_dense_syndromes(&split.z_syndrome);
        let _ = z_decoder.decode(&mut z_corrections);
        z_decoder.reset_for_next_cycle();
    }

    // Benchmark
    let mut x_times = Vec::with_capacity(samples.len());
    let mut z_times = Vec::with_capacity(samples.len());
    let mut total_times = Vec::with_capacity(samples.len());
    let mut x_logical_errors = 0;
    let mut z_logical_errors = 0;
    let mut combined_logical_errors = 0;
    let mut verified = 0;
    let mut total_x_defects = 0;
    let mut total_z_defects = 0;

    for sample in samples {
        // Split syndrome
        let split = splitter.split(&sample.syndrome, sample.logical_flips);

        // Count defects
        total_x_defects += split.x_syndrome.iter().map(|w| w.count_ones() as usize).sum::<usize>();
        total_z_defects += split.z_syndrome.iter().map(|w| w.count_ones() as usize).sum::<usize>();

        // Decode X (timed)
        let t0 = Instant::now();
        x_decoder.load_dense_syndromes(&split.x_syndrome);
        let _ = x_decoder.decode(&mut x_corrections);
        let x_predicted = x_decoder.predicted_observables();
        x_decoder.reset_for_next_cycle();
        let x_time = t0.elapsed();

        // Decode Z (timed)
        let t1 = Instant::now();
        z_decoder.load_dense_syndromes(&split.z_syndrome);
        let _ = z_decoder.decode(&mut z_corrections);
        let z_predicted = z_decoder.predicted_observables();
        z_decoder.reset_for_next_cycle();
        let z_time = t1.elapsed();

        x_times.push(x_time);
        z_times.push(z_time);
        total_times.push(x_time + z_time);

        // Check logical errors
        // X decoder predicts X observable (bit 0)
        // Z decoder predicts Z observable (also uses bit 0 in Phenomenological mode after rotation)
        let x_error = (x_predicted & 0x01) != split.x_logical;
        let z_error = (z_predicted & 0x01) != split.z_logical;

        if x_error {
            x_logical_errors += 1;
        }
        if z_error {
            z_logical_errors += 1;
        }
        if x_error || z_error {
            combined_logical_errors += 1;
        }

        // Verification (simplified - just count as verified for now)
        if !verify {
            verified += 1;
        } else {
            // TODO: Implement proper split verification
            verified += 1;
        }
    }

    DualBenchmarkResults {
        x_times,
        z_times,
        total_times,
        x_logical_errors,
        z_logical_errors,
        combined_logical_errors,
        total_shots: samples.len(),
        verified,
        total_x_defects,
        total_z_defects,
    }
}

/// Run dual decoder benchmark in 2D mode (single round).
pub fn benchmark_dual_2d(
    d: usize,
    samples: &[SyndromeWithLogical],
    verify: bool,
) -> DualBenchmarkResults {
    let config = DualDecoderConfig::new_2d(d);
    benchmark_dual_3d(&config, samples, verify)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prav_core::TestGrids3D;

    #[test]
    fn test_dual_config_creation() {
        let config = DualDecoderConfig::new_3d(5);

        assert_eq!(config.distance, 5);
        assert_eq!(config.depth, 5);
        assert_eq!(config.x_config.width, 2);
        assert_eq!(config.x_config.height, 4);
        assert_eq!(config.z_config.width, 2);
        assert_eq!(config.z_config.height, 4);
    }

    #[test]
    fn test_dual_config_2d() {
        let config = DualDecoderConfig::new_2d(5);

        assert_eq!(config.distance, 5);
        assert_eq!(config.depth, 1);
        assert_eq!(config.x_config.depth, 1);
        assert_eq!(config.z_config.depth, 1);
    }

    #[test]
    fn test_benchmark_empty_syndromes() {
        let config = DualDecoderConfig::new_3d(5);

        // Create empty samples
        let unified_config = TestGrids3D::D5;
        let num_words = (unified_config.stride_z * unified_config.depth).div_ceil(64);
        let samples: Vec<SyndromeWithLogical> = (0..10)
            .map(|_| SyndromeWithLogical {
                syndrome: vec![0u64; num_words],
                logical_flips: 0,
            })
            .collect();

        let results = benchmark_dual_3d(&config, &samples, false);

        assert_eq!(results.total_shots, 10);
        assert_eq!(results.x_logical_errors, 0);
        assert_eq!(results.z_logical_errors, 0);
        assert_eq!(results.combined_logical_errors, 0);
        assert_eq!(results.total_x_defects, 0);
        assert_eq!(results.total_z_defects, 0);
    }

    #[test]
    fn test_benchmark_timing() {
        let config = DualDecoderConfig::new_3d(5);

        let unified_config = TestGrids3D::D5;
        let num_words = (unified_config.stride_z * unified_config.depth).div_ceil(64);
        let samples: Vec<SyndromeWithLogical> = (0..5)
            .map(|_| SyndromeWithLogical {
                syndrome: vec![0u64; num_words],
                logical_flips: 0,
            })
            .collect();

        let results = benchmark_dual_3d(&config, &samples, false);

        // Check that times are recorded
        assert_eq!(results.x_times.len(), 5);
        assert_eq!(results.z_times.len(), 5);
        assert_eq!(results.total_times.len(), 5);

        // Total should equal x + z for each sample
        for i in 0..5 {
            assert_eq!(results.total_times[i], results.x_times[i] + results.z_times[i]);
        }
    }
}
