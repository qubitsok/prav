//! Color code benchmarking module.
//!
//! This module provides benchmarking support for triangular color codes
//! using the restriction decoder approach (three parallel decoders).
//!
//! # Architecture
//!
//! The color code decoder splits syndromes by color class and runs
//! three independent Union-Find decoders:
//!
//! ```text
//!         Full Color Code Syndrome
//!                  │
//!           ┌──────┼──────┐
//!           ▼      ▼      ▼
//!         Red    Green   Blue
//!        Decoder Decoder Decoder
//!           │      │      │
//!           └──────┼──────┘
//!                  ▼
//!         Combined Logical Frame
//! ```

use std::time::{Duration, Instant};

use prav_core::color_code::grid_3d::ColorCodeGrid3DConfig;
use prav_core::color_code::{ColorCodeResult, FaceColor};
use prav_core::{
    Arena, DecoderBuilder, EdgeCorrection, ObservableMode, SquareGrid,
    required_buffer_size,
};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// Configuration for color code benchmarking.
#[derive(Clone, Debug)]
pub struct ColorCodeBenchConfig {
    /// Code distance.
    pub distance: usize,
    /// Number of measurement rounds (depth).
    pub depth: usize,
    /// Grid configuration.
    pub grid_config: ColorCodeGrid3DConfig,
}

impl ColorCodeBenchConfig {
    /// Create a 3D color code config (depth = distance).
    pub fn new_3d(d: usize) -> Self {
        Self {
            distance: d,
            depth: d,
            grid_config: ColorCodeGrid3DConfig::for_triangular_6_6_6(d),
        }
    }

    /// Create a 2D color code config (single measurement round).
    pub fn new_2d(d: usize) -> Self {
        let grid_config = ColorCodeGrid3DConfig::for_triangular_6_6_6(d).with_depth(1);
        Self {
            distance: d,
            depth: 1,
            grid_config,
        }
    }

    /// Buffer size required for one color's decoder.
    pub fn buffer_size_per_color(&self) -> usize {
        // Each color class has approximately 1/3 of detectors
        let restricted_width = (self.grid_config.width + 2) / 3;
        let restricted_height = self.grid_config.height;
        required_buffer_size(restricted_width, restricted_height, self.depth)
    }

    /// Total buffer size for all three decoders.
    pub fn total_buffer_size(&self) -> usize {
        self.buffer_size_per_color() * 3 + 4096 // Extra for syndrome buffers
    }
}

/// Syndrome with logical observable ground truth for color codes.
#[derive(Clone)]
pub struct ColorCodeSyndrome {
    /// Defect indices in the full color code grid.
    pub defects: Vec<u32>,
    /// Ground truth logical observable frame.
    pub logical_flips: u8,
    /// Number of defects per color class.
    pub defects_by_color: [usize; 3],
}

/// Results from color code benchmarking.
#[derive(Clone, Debug)]
pub struct ColorCodeBenchResults {
    /// Decode times for each sample.
    pub times: Vec<Duration>,
    /// Number of logical errors detected.
    pub logical_errors: usize,
    /// Total number of samples processed.
    pub total_shots: usize,
    /// Total defects across all samples.
    pub total_defects: usize,
    /// Defect distribution by color.
    pub defects_by_color: [usize; 3],
}

impl ColorCodeBenchResults {
    /// Calculate logical error rate.
    pub fn logical_error_rate(&self) -> f64 {
        self.logical_errors as f64 / self.total_shots as f64
    }

    /// Calculate average decode time in microseconds.
    pub fn avg_decode_time_us(&self) -> f64 {
        if self.times.is_empty() {
            return 0.0;
        }
        let total_ns: u128 = self.times.iter().map(|d| d.as_nanos()).sum();
        (total_ns as f64 / self.times.len() as f64) / 1000.0
    }
}

/// Generate phenomenological syndromes for color codes.
///
/// In a color code, errors create defect pairs of the same color.
/// This generator simulates that by:
/// 1. Randomly selecting qubit positions
/// 2. Creating defect pairs on same-colored faces
///
/// # Arguments
/// * `config` - Color code grid configuration
/// * `error_rate` - Physical error probability per qubit
/// * `num_samples` - Number of syndrome samples to generate
/// * `seed` - Random seed for reproducibility
pub fn generate_color_code_syndromes(
    config: &ColorCodeGrid3DConfig,
    error_rate: f64,
    num_samples: usize,
    seed: u64,
) -> Vec<ColorCodeSyndrome> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut samples = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let mut defects = Vec::new();
        let mut logical_flips = 0u8;
        let mut defects_by_color = [0usize; 3];

        // Iterate over the grid and apply errors with given probability
        for t in 0..config.depth {
            for y in 0..config.height {
                for x in 0..config.width {
                    // Check if an error occurs at this position
                    let r: f64 = rand::Rng::random(&mut rng);
                    if r < error_rate {
                        let idx = config.coord_to_linear(x, y, t) as u32;
                        let color = config.detector_color(x, y);

                        // In a real color code, a single qubit error creates
                        // defects on multiple same-colored faces. For simplicity,
                        // we create a single defect at this position.
                        defects.push(idx);
                        defects_by_color[color.index()] += 1;

                        // Boundary errors can flip logical observables
                        if x == 0 || x == config.width - 1 {
                            // Edge boundary - may flip logical
                            if color == FaceColor::Red {
                                logical_flips ^= 0b01; // X observable
                            }
                        }
                        if y == 0 || y == config.height - 1 {
                            if color == FaceColor::Green {
                                logical_flips ^= 0b10; // Z observable
                            }
                        }
                    }
                }
            }
        }

        samples.push(ColorCodeSyndrome {
            defects,
            logical_flips,
            defects_by_color,
        });
    }

    samples
}

/// Run color code benchmark.
///
/// This creates three parallel decoders (one per color class) and
/// measures decoding performance.
///
/// # Arguments
/// * `config` - Benchmark configuration
/// * `samples` - Pre-generated syndrome samples
/// * `verify` - Whether to verify corrections (not yet implemented for color codes)
pub fn benchmark_color_code(
    config: &ColorCodeBenchConfig,
    samples: &[ColorCodeSyndrome],
) -> ColorCodeBenchResults {
    // For the color code decoder, we use three separate decoders
    // Each decodes approximately 1/3 of the defects

    // Calculate restricted grid dimensions (approximately 1/3 width)
    let restricted_width = (config.grid_config.width + 2) / 3;
    let restricted_height = config.grid_config.height;
    let depth = config.depth;

    // Allocate buffers for each color's decoder
    let buf_size = required_buffer_size(restricted_width, restricted_height, depth);
    let mut red_buffer = vec![0u8; buf_size];
    let mut green_buffer = vec![0u8; buf_size];
    let mut blue_buffer = vec![0u8; buf_size];

    let mut red_arena = Arena::new(&mut red_buffer);
    let mut green_arena = Arena::new(&mut green_buffer);
    let mut blue_arena = Arena::new(&mut blue_buffer);

    // Create decoders for each color class
    let mut red_decoder = DecoderBuilder::<SquareGrid>::new()
        .dimensions(restricted_width, restricted_height)
        .build(&mut red_arena)
        .expect("Failed to create red decoder");

    let mut green_decoder = DecoderBuilder::<SquareGrid>::new()
        .dimensions(restricted_width, restricted_height)
        .build(&mut green_arena)
        .expect("Failed to create green decoder");

    let mut blue_decoder = DecoderBuilder::<SquareGrid>::new()
        .dimensions(restricted_width, restricted_height)
        .build(&mut blue_arena)
        .expect("Failed to create blue decoder");

    red_decoder.set_observable_mode(ObservableMode::Phenomenological);
    green_decoder.set_observable_mode(ObservableMode::Phenomenological);
    blue_decoder.set_observable_mode(ObservableMode::Phenomenological);

    // Correction buffers
    let max_corrections = restricted_width * restricted_height * depth * 2;
    let mut red_corrections = vec![EdgeCorrection::default(); max_corrections];
    let mut green_corrections = vec![EdgeCorrection::default(); max_corrections];
    let mut blue_corrections = vec![EdgeCorrection::default(); max_corrections];

    // Warmup
    for sample in samples.iter().take(100) {
        let (red_syn, green_syn, blue_syn) = split_defects_by_color(
            &sample.defects,
            &config.grid_config,
            restricted_width,
        );

        if !red_syn.is_empty() {
            load_sparse_syndrome(&mut red_decoder, &red_syn, restricted_width, restricted_height);
            let _ = red_decoder.decode(&mut red_corrections);
            red_decoder.reset_for_next_cycle();
        }
        if !green_syn.is_empty() {
            load_sparse_syndrome(&mut green_decoder, &green_syn, restricted_width, restricted_height);
            let _ = green_decoder.decode(&mut green_corrections);
            green_decoder.reset_for_next_cycle();
        }
        if !blue_syn.is_empty() {
            load_sparse_syndrome(&mut blue_decoder, &blue_syn, restricted_width, restricted_height);
            let _ = blue_decoder.decode(&mut blue_corrections);
            blue_decoder.reset_for_next_cycle();
        }
    }

    // Benchmark loop
    let mut times = Vec::with_capacity(samples.len());
    let mut logical_errors = 0;
    let mut total_defects = 0;
    let mut defects_by_color = [0usize; 3];

    for sample in samples {
        total_defects += sample.defects.len();
        for i in 0..3 {
            defects_by_color[i] += sample.defects_by_color[i];
        }

        // Split defects by color
        let (red_syn, green_syn, blue_syn) = split_defects_by_color(
            &sample.defects,
            &config.grid_config,
            restricted_width,
        );

        let t0 = Instant::now();

        // Decode each color class
        let mut predicted_frame = 0u8;

        if !red_syn.is_empty() {
            load_sparse_syndrome(&mut red_decoder, &red_syn, restricted_width, restricted_height);
            let _ = red_decoder.decode(&mut red_corrections);
            predicted_frame ^= red_decoder.predicted_observables();
            red_decoder.reset_for_next_cycle();
        }

        if !green_syn.is_empty() {
            load_sparse_syndrome(&mut green_decoder, &green_syn, restricted_width, restricted_height);
            let _ = green_decoder.decode(&mut green_corrections);
            predicted_frame ^= green_decoder.predicted_observables();
            green_decoder.reset_for_next_cycle();
        }

        if !blue_syn.is_empty() {
            load_sparse_syndrome(&mut blue_decoder, &blue_syn, restricted_width, restricted_height);
            let _ = blue_decoder.decode(&mut blue_corrections);
            predicted_frame ^= blue_decoder.predicted_observables();
            blue_decoder.reset_for_next_cycle();
        }

        times.push(t0.elapsed());

        // Compare predicted vs actual logical flips
        if predicted_frame != sample.logical_flips {
            logical_errors += 1;
        }
    }

    ColorCodeBenchResults {
        times,
        logical_errors,
        total_shots: samples.len(),
        total_defects,
        defects_by_color,
    }
}

/// Split defects into three color classes with restricted indices.
fn split_defects_by_color(
    defects: &[u32],
    config: &ColorCodeGrid3DConfig,
    _restricted_width: usize,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut red = Vec::new();
    let mut green = Vec::new();
    let mut blue = Vec::new();

    // Track position within each color class for compact indexing
    let mut color_counters = [0u32; 3];

    for &idx in defects {
        let (x, y, _t) = config.linear_to_coord(idx as usize);
        let color = config.detector_color(x, y);

        // Compute restricted index within this color's subgraph
        // Simple linear mapping: use color counter
        let restricted_idx = color_counters[color.index()];
        color_counters[color.index()] += 1;

        match color {
            FaceColor::Red => red.push(restricted_idx),
            FaceColor::Green => green.push(restricted_idx),
            FaceColor::Blue => blue.push(restricted_idx),
        }
    }

    (red, green, blue)
}

/// Load sparse syndrome into decoder.
fn load_sparse_syndrome<T: prav_core::Topology>(
    decoder: &mut prav_core::DynDecoder<'_, T>,
    defects: &[u32],
    width: usize,
    height: usize,
) {
    // Convert sparse defects to dense syndrome
    let num_blocks = (width * height + 63) / 64;
    let mut dense = vec![0u64; num_blocks];

    for &idx in defects {
        let idx = idx as usize;
        if idx < width * height {
            let block = idx / 64;
            let bit = idx % 64;
            if block < dense.len() {
                dense[block] |= 1u64 << bit;
            }
        }
    }

    decoder.load_dense_syndromes(&dense);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config_3d = ColorCodeBenchConfig::new_3d(5);
        assert_eq!(config_3d.distance, 5);
        assert_eq!(config_3d.depth, 5);

        let config_2d = ColorCodeBenchConfig::new_2d(5);
        assert_eq!(config_2d.distance, 5);
        assert_eq!(config_2d.depth, 1);
    }

    #[test]
    fn test_syndrome_generation() {
        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(5);
        let samples = generate_color_code_syndromes(&config, 0.01, 100, 42);

        assert_eq!(samples.len(), 100);

        // With 0.01 error rate, most samples should have some defects
        let samples_with_defects = samples.iter().filter(|s| !s.defects.is_empty()).count();
        assert!(samples_with_defects > 0);
    }

    #[test]
    fn test_benchmark_empty_syndromes() {
        let config = ColorCodeBenchConfig::new_2d(3);
        let samples = vec![
            ColorCodeSyndrome {
                defects: vec![],
                logical_flips: 0,
                defects_by_color: [0, 0, 0],
            };
            10
        ];

        let results = benchmark_color_code(&config, &samples);
        assert_eq!(results.total_shots, 10);
        assert_eq!(results.logical_errors, 0);
    }

    #[test]
    fn test_split_defects_by_color() {
        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(5);

        // Create defects at known positions
        let defects = vec![
            config.coord_to_linear(0, 0, 0) as u32, // Red (0+0=0)
            config.coord_to_linear(1, 0, 0) as u32, // Green (1+0=1)
            config.coord_to_linear(2, 0, 0) as u32, // Blue (2+0=2)
            config.coord_to_linear(3, 0, 0) as u32, // Red (3+0=0)
        ];

        let (red, green, blue) = split_defects_by_color(&defects, &config, 2);

        assert_eq!(red.len(), 2);
        assert_eq!(green.len(), 1);
        assert_eq!(blue.len(), 1);
    }
}
