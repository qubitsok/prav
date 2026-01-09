//! # Phenomenological Noise Model for 3D Syndrome Generation
//!
//! This module provides a simplified noise model for testing the decoder
//! without needing a full Stim DEM file. It's useful for quick experiments
//! but not accurate for threshold estimation.
//!
//! ## What is Phenomenological Noise?
//!
//! Phenomenological noise is a simplified error model that captures the
//! essential structure of quantum errors without modeling the full circuit.
//! It assumes:
//!
//! - **Space-like errors**: Random errors on data qubits, each with probability `p_space`
//! - **Time-like errors**: Random measurement errors, each with probability `p_time`
//!
//! This is simpler than circuit-level noise (which models individual gate errors)
//! but captures the 3D structure of the decoding problem.
//!
//! ## Error Types
//!
//! ### Space-like Errors (Data Qubit Errors)
//!
//! A space-like error represents a bit-flip or phase-flip on a data qubit.
//! It flips the two adjacent detectors at the same time step.
//!
//! ```text
//!     Space-like error on data qubit between D0 and D1:
//!
//!     Before:  ○──○──○         After:   ●──●──○
//!              D0 D1 D2                 D0 D1 D2
//!
//!     Both adjacent detectors flip (XOR with 1).
//! ```
//!
//! At boundaries, only one detector exists, so only one flips. These
//! boundary errors can affect logical observables.
//!
//! ### Time-like Errors (Measurement Errors)
//!
//! A time-like error represents a faulty measurement. It flips the same
//! detector at two consecutive time steps.
//!
//! ```text
//!     Time-like error at detector D0:
//!
//!     t=0:  ●──○──○    D0 at t=0 flips
//!           │
//!     t=1:  ●──○──○    D0 at t=1 also flips
//!
//!     Same spatial position, two time steps.
//! ```
//!
//! Time boundaries (first/last round) only flip one detector.
//!
//! ## Logical Error Tracking
//!
//! Boundary errors can cause logical errors. The stabilizer type determines
//! which logical is affected:
//!
//! - **Even parity** `(x + y) % 2 == 0` → Z stabilizer → Z logical (bit 1)
//! - **Odd parity** `(x + y) % 2 == 1` → X stabilizer → X logical (bit 0)
//!
//! This convention matches Stim's rotated surface code.
//!
//! ## Usage
//!
//! ```ignore
//! use prav_core::TestGrids3D;
//!
//! let config = TestGrids3D::D5;  // Distance-5 code
//! let p_space = 0.001;           // 0.1% data error rate
//! let p_time = 0.001;            // 0.1% measurement error rate
//!
//! let samples = generate_correlated_syndromes(&config, p_space, p_time, 10000, 42);
//!
//! for sample in &samples {
//!     // sample.syndrome: bit vector of triggered detectors
//!     // sample.logical_flips: ground truth logical errors (bits 0 and 1)
//! }
//! ```
//!
//! ## Limitations
//!
//! Phenomenological noise is **not accurate** for threshold estimation because:
//!
//! 1. Real circuits have correlated errors from two-qubit gates
//! 2. Error probabilities vary by gate type and position
//! 3. Hook errors and other circuit-specific effects are missing
//!
//! For accurate threshold studies, use Stim DEM files instead.

use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use prav_core::Grid3DConfig;

/// A syndrome sample with ground-truth logical error information.
///
/// This struct pairs a syndrome (the pattern of triggered detectors) with
/// the actual logical errors that occurred. The decoder tries to correct
/// the syndrome without causing logical errors.
///
/// ## Logical Error Tracking
///
/// The `logical_flips` field is the "ground truth" - it tells us which
/// logical operators were flipped by the actual errors that occurred.
/// After decoding, we compare the decoder's predicted logical flips
/// with this ground truth. If they differ, it's a logical error.
#[derive(Clone, Debug)]
pub struct SyndromeWithLogical {
    /// Packed syndrome bits with one bit per detector.
    ///
    /// Bit layout: detector `i` is at bit `i % 64` of word `i / 64`.
    /// A bit is 1 if the detector triggered (measured -1), 0 otherwise.
    pub syndrome: Vec<u64>,

    /// Ground truth logical frame changes.
    ///
    /// - Bit 0 (0x01): X logical was flipped
    /// - Bit 1 (0x02): Z logical was flipped
    ///
    /// These are the actual logical errors from the sampled error pattern.
    /// The decoder's job is to predict these correctly.
    pub logical_flips: u8,
}

/// Generate independent (uncorrelated) phenomenological syndromes.
///
/// This is the simplest noise model: each detector independently has
/// probability `p` of being triggered. There's no correlation structure
/// and no logical error tracking.
///
/// **Warning**: This model is too simple for realistic benchmarking.
/// Use [`generate_correlated_syndromes`] instead.
///
/// # Parameters
///
/// - `config`: Grid dimensions (width, height, depth)
/// - `error_prob`: Probability each detector triggers (0.0 to 1.0)
/// - `num_shots`: Number of syndrome samples to generate
/// - `seed`: Random seed for reproducibility
///
/// # Returns
///
/// Vector of syndromes (no logical error tracking).
pub fn generate_phenomenological_syndromes(
    config: &Grid3DConfig,
    error_prob: f64,
    num_shots: usize,
    seed: u64,
) -> Vec<Vec<u64>> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let num_words = config.alloc_nodes().div_ceil(64);

    (0..num_shots)
        .map(|_| generate_single_syndrome(config, error_prob, num_words, &mut rng))
        .collect()
}

fn generate_single_syndrome(
    config: &Grid3DConfig,
    error_prob: f64,
    num_words: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Vec<u64> {
    let mut syndrome = vec![0u64; num_words];

    for t in 0..config.depth {
        for y in 0..config.height {
            for x in 0..config.width {
                if rng.random::<f64>() < error_prob {
                    let linear = config.coord_to_linear(x, y, t);
                    let word = linear / 64;
                    let bit = linear % 64;
                    if word < num_words {
                        syndrome[word] ^= 1 << bit;
                    }
                }
            }
        }
    }

    syndrome
}

/// Generate correlated phenomenological syndromes with logical error tracking.
///
/// This is the main function for phenomenological noise generation. It
/// creates realistic-ish syndromes by simulating:
///
/// 1. **Space-like errors**: Data qubit errors that flip adjacent detectors
/// 2. **Time-like errors**: Measurement errors that flip detectors at adjacent times
/// 3. **Boundary effects**: Boundary errors that affect logical observables
///
/// The returned syndromes include ground-truth logical flip information,
/// which is used to determine if the decoder made a logical error.
///
/// # Parameters
///
/// - `config`: Grid dimensions (width, height, depth)
/// - `p_space`: Probability of each space-like error (data qubit error rate)
/// - `p_time`: Probability of each time-like error (measurement error rate)
/// - `num_shots`: Number of syndrome samples to generate
/// - `seed`: Random seed for reproducibility
///
/// # Returns
///
/// Vector of [`SyndromeWithLogical`] containing both syndrome bits and
/// ground-truth logical flips.
///
/// # Error Model Details
///
/// ## Space-like Errors
///
/// For each potential error location (edge in the matching graph):
/// - Interior edges: flip both adjacent detectors
/// - Boundary edges: flip one detector + XOR into logical
///
/// ## Time-like Errors
///
/// For each detector at each time step (except boundaries):
/// - Interior: flip detector at t and t+1
/// - Time boundaries: flip only one time step (no logical effect)
///
/// # Example
///
/// ```ignore
/// let config = TestGrids3D::D5;
/// let samples = generate_correlated_syndromes(&config, 0.001, 0.001, 10000, 42);
///
/// let logical_error_count = samples.iter()
///     .filter(|s| s.logical_flips != 0)
///     .count();
/// ```
pub fn generate_correlated_syndromes(
    config: &Grid3DConfig,
    p_space: f64, // Probability of space-like (data qubit) error
    p_time: f64,  // Probability of time-like (measurement) error
    num_shots: usize,
    seed: u64,
) -> Vec<SyndromeWithLogical> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let num_words = config.alloc_nodes().div_ceil(64);

    (0..num_shots)
        .map(|_| generate_correlated_single(config, p_space, p_time, num_words, &mut rng))
        .collect()
}

fn generate_correlated_single(
    config: &Grid3DConfig,
    p_space: f64,
    p_time: f64,
    num_words: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> SyndromeWithLogical {
    let mut syndrome = vec![0u64; num_words];
    let mut logical_flips = 0u8;

    // Space-like errors (data qubit errors)
    // These create edges in the matching graph
    for t in 0..config.depth {
        // Horizontal edges (including left boundary)
        for y in 0..config.height {
            // Left boundary edge (x=0, only flips one detector)
            if rng.random::<f64>() < p_space {
                flip_detector(&mut syndrome, config, 0, y, t, num_words);
                // Use stabilizer parity to determine logical
                // (0 + y) % 2 determines if Z or X stabilizer
                if y % 2 == 0 {
                    logical_flips ^= 0b10; // Z logical
                } else {
                    logical_flips ^= 0b01; // X logical
                }
            }

            // Interior horizontal edges
            for x in 0..config.width.saturating_sub(1) {
                if rng.random::<f64>() < p_space {
                    flip_detector(&mut syndrome, config, x, y, t, num_words);
                    flip_detector(&mut syndrome, config, x + 1, y, t, num_words);
                }
            }

            // Right boundary edge (x=width-1, only flips one detector)
            if config.width > 0 && rng.random::<f64>() < p_space {
                let x = config.width - 1;
                flip_detector(&mut syndrome, config, x, y, t, num_words);
                // Use stabilizer parity: (x + y) % 2
                if (x + y) % 2 == 0 {
                    logical_flips ^= 0b10; // Z logical
                } else {
                    logical_flips ^= 0b01; // X logical
                }
            }
        }

        // Vertical edges (including top/bottom boundaries)
        for x in 0..config.width {
            // Bottom boundary edge (y=0, only flips one detector)
            if rng.random::<f64>() < p_space {
                flip_detector(&mut syndrome, config, x, 0, t, num_words);
                // Use stabilizer parity: (x + 0) % 2
                if x % 2 == 0 {
                    logical_flips ^= 0b10; // Z logical
                } else {
                    logical_flips ^= 0b01; // X logical
                }
            }

            // Interior vertical edges
            for y in 0..config.height.saturating_sub(1) {
                if rng.random::<f64>() < p_space {
                    flip_detector(&mut syndrome, config, x, y, t, num_words);
                    flip_detector(&mut syndrome, config, x, y + 1, t, num_words);
                }
            }

            // Top boundary edge (y=height-1, only flips one detector)
            if config.height > 0 && rng.random::<f64>() < p_space {
                let y = config.height - 1;
                flip_detector(&mut syndrome, config, x, y, t, num_words);
                // Use stabilizer parity: (x + y) % 2
                if (x + y) % 2 == 0 {
                    logical_flips ^= 0b10; // Z logical
                } else {
                    logical_flips ^= 0b01; // X logical
                }
            }
        }
    }

    // Time-like errors (measurement errors)
    // These flip the same detector at adjacent time steps
    for t in 0..config.depth.saturating_sub(1) {
        for y in 0..config.height {
            for x in 0..config.width {
                if rng.random::<f64>() < p_time {
                    flip_detector(&mut syndrome, config, x, y, t, num_words);
                    flip_detector(&mut syndrome, config, x, y, t + 1, num_words);
                }
            }
        }
    }

    // Initial time boundary (t=0) - measurement errors at first round
    for y in 0..config.height {
        for x in 0..config.width {
            if rng.random::<f64>() < p_time {
                flip_detector(&mut syndrome, config, x, y, 0, num_words);
                // Time boundary errors don't affect logical (they're like initialization)
            }
        }
    }

    // Final time boundary (t=depth-1) - measurement errors at last round
    if config.depth > 0 {
        for y in 0..config.height {
            for x in 0..config.width {
                if rng.random::<f64>() < p_time {
                    flip_detector(&mut syndrome, config, x, y, config.depth - 1, num_words);
                }
            }
        }
    }

    SyndromeWithLogical {
        syndrome,
        logical_flips,
    }
}

/// Flip (XOR) a single detector bit in the syndrome.
///
/// This is the fundamental operation for building syndromes. Each error
/// flips one or more detectors, and we use XOR so that two errors on
/// the same detector cancel out.
///
/// # Parameters
///
/// - `syndrome`: The packed syndrome bit vector to modify
/// - `config`: Grid configuration for coordinate conversion
/// - `x`, `y`, `t`: Detector coordinates
/// - `num_words`: Size of syndrome vector (bounds check)
fn flip_detector(
    syndrome: &mut [u64],
    config: &Grid3DConfig,
    x: usize,
    y: usize,
    t: usize,
    num_words: usize,
) {
    let linear = config.coord_to_linear(x, y, t);
    let word = linear / 64;
    let bit = linear % 64;
    if word < num_words {
        syndrome[word] ^= 1 << bit;
    }
}

/// Count the number of triggered detectors (defects) in a syndrome.
///
/// This counts the total number of 1 bits across all words. A defect
/// is a detector that measured -1 (triggered).
///
/// # Example
///
/// ```ignore
/// let syndrome = vec![0b1010u64, 0b1111u64];
/// assert_eq!(count_defects(&syndrome), 6);  // 2 + 4
/// ```
pub fn count_defects(syndrome: &[u64]) -> usize {
    syndrome.iter().map(|w| w.count_ones() as usize).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use prav_core::TestGrids3D;

    #[test]
    fn test_phenomenological_basic() {
        let config = TestGrids3D::D5;
        let syndromes = generate_phenomenological_syndromes(&config, 0.01, 100, 42);

        assert_eq!(syndromes.len(), 100);
        for syn in &syndromes {
            assert!(syn.len() >= config.num_detectors().div_ceil(64));
        }
    }

    #[test]
    fn test_phenomenological_zero_prob() {
        let config = TestGrids3D::D3;
        let syndromes = generate_phenomenological_syndromes(&config, 0.0, 100, 42);

        for syn in &syndromes {
            assert!(syn.iter().all(|&w| w == 0));
        }
    }

    #[test]
    fn test_correlated_basic() {
        let config = TestGrids3D::D5;
        let results = generate_correlated_syndromes(&config, 0.01, 0.01, 100, 42);

        assert_eq!(results.len(), 100);
        for r in &results {
            assert!(r.syndrome.len() >= config.num_detectors().div_ceil(64));
        }
    }

    #[test]
    fn test_correlated_zero_prob() {
        let config = TestGrids3D::D3;
        let results = generate_correlated_syndromes(&config, 0.0, 0.0, 100, 42);

        for r in &results {
            assert!(r.syndrome.iter().all(|&w| w == 0));
            assert_eq!(r.logical_flips, 0);
        }
    }

    #[test]
    fn test_correlated_parity() {
        // Interior errors always flip pairs, so parity should be even
        // (boundary errors flip single detectors, which can create odd parity)
        let config = TestGrids3D::D5;
        let results = generate_correlated_syndromes(&config, 0.001, 0.001, 1000, 42);

        // At low error rates, most syndromes should have even parity
        // (only boundary errors create odd parity)
        let mut even_count = 0;
        for r in &results {
            let total: u32 = r.syndrome.iter().map(|w| w.count_ones()).sum();
            if total.is_multiple_of(2) {
                even_count += 1;
            }
        }
        // At p=0.001, we expect ~90%+ even parity
        assert!(
            even_count > 800,
            "Expected mostly even parity, got {}/1000",
            even_count
        );
    }

    #[test]
    fn test_logical_flips_tracked() {
        let config = TestGrids3D::D5;
        // Higher error rate to ensure some logical flips
        let results = generate_correlated_syndromes(&config, 0.1, 0.1, 100, 42);

        // At p=0.1, we should see some logical flips
        let has_x_flip = results.iter().any(|r| r.logical_flips & 0b01 != 0);
        let has_z_flip = results.iter().any(|r| r.logical_flips & 0b10 != 0);

        assert!(has_x_flip, "Expected some X logical flips");
        assert!(has_z_flip, "Expected some Z logical flips");
    }
}
