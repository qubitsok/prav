//! # Circuit-Level Syndrome Sampling from DEM
//!
//! This module samples syndromes from a Stim Detector Error Model (DEM).
//! It simulates the error behavior of a quantum circuit by Monte Carlo sampling.
//!
//! ## How Sampling Works
//!
//! The DEM describes all possible error mechanisms and their probabilities.
//! For each "shot" (syndrome sample):
//!
//! 1. Start with an empty syndrome (all zeros)
//! 2. For each error mechanism, flip a coin with the mechanism's probability
//! 3. If the coin lands "error", XOR the mechanism's detectors into the syndrome
//! 4. Also XOR any logical observable effects
//!
//! Because we use XOR, multiple errors on the same detector cancel out.
//! This correctly models the Z₂ (mod 2) algebra of Pauli errors.
//!
//! ## Example
//!
//! ```ignore
//! // Parse a DEM file
//! let dem = parse_dem(&content)?;
//!
//! // Create sampler with seed for reproducibility
//! let mut sampler = CircuitSampler::new(&dem, 42);
//!
//! // Generate 10,000 syndrome samples
//! for _ in 0..10_000 {
//!     let (syndrome, logical_flips) = sampler.sample();
//!     // syndrome: bit vector of triggered detectors
//!     // logical_flips: which logical observables were flipped
//! }
//! ```
//!
//! ## Efficiency
//!
//! The sampler pre-extracts all mechanism data from the DEM into flat vectors.
//! This avoids pointer chasing during the hot sampling loop.

use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::dem::types::ParsedDem;

/// Monte Carlo sampler for circuit-level syndromes.
///
/// This struct holds pre-processed data from a DEM file and generates
/// syndrome samples by simulating error occurrence.
///
/// ## Algorithm
///
/// For each sample:
/// 1. Initialize syndrome to all zeros
/// 2. For each error mechanism with probability p:
///    - Generate random number r in [0, 1)
///    - If r < p, the error occurred:
///      - XOR all target detectors into syndrome
///      - XOR frame changes into logical flip accumulator
/// 3. Return (syndrome, logical_flips)
///
/// ## Random Number Generator
///
/// Uses Xoshiro256++ for fast, high-quality randomness with reproducible
/// results from a seed.
pub struct CircuitSampler {
    /// Error probabilities for each mechanism. Length = number of mechanisms.
    probabilities: Vec<f32>,

    /// Detector IDs affected by each mechanism. Outer length = number of mechanisms.
    detector_targets: Vec<Vec<u32>>,

    /// Logical observable effects for each mechanism. Bitmask where bit i = `L<i>`.
    frame_changes: Vec<u8>,

    /// Total number of detectors (for sizing the output syndrome).
    num_detectors: u32,

    /// Random number generator state.
    rng: Xoshiro256PlusPlus,
}

impl CircuitSampler {
    /// Create a new circuit sampler from a parsed DEM.
    pub fn new(dem: &ParsedDem, seed: u64) -> Self {
        let probabilities: Vec<f32> = dem.mechanisms.iter().map(|m| m.probability).collect();
        let detector_targets: Vec<Vec<u32>> =
            dem.mechanisms.iter().map(|m| m.detectors.clone()).collect();
        let frame_changes: Vec<u8> = dem.mechanisms.iter().map(|m| m.frame_changes).collect();

        Self {
            probabilities,
            detector_targets,
            frame_changes,
            num_detectors: dem.num_detectors,
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }

    /// Generate one syndrome shot.
    ///
    /// Returns (syndrome_bits, frame_changes) where:
    /// - syndrome_bits: packed u64 words with one bit per detector
    /// - frame_changes: bitmask of flipped logical observables
    pub fn sample(&mut self) -> (Vec<u64>, u8) {
        let num_words = (self.num_detectors as usize).div_ceil(64);
        let mut syndrome = vec![0u64; num_words];
        let mut logical_flips = 0u8;

        for (i, &prob) in self.probabilities.iter().enumerate() {
            if self.rng.random::<f32>() < prob {
                // Error occurred - flip all connected detectors
                for &det_id in &self.detector_targets[i] {
                    let word = det_id as usize / 64;
                    let bit = det_id as usize % 64;
                    if word < syndrome.len() {
                        syndrome[word] ^= 1 << bit;
                    }
                }
                logical_flips ^= self.frame_changes[i];
            }
        }

        (syndrome, logical_flips)
    }

    /// Generate multiple syndrome shots.
    pub fn sample_batch(&mut self, num_shots: usize) -> Vec<(Vec<u64>, u8)> {
        (0..num_shots).map(|_| self.sample()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dem::types::{OwnedErrorMechanism, ParsedDem};
    use prav_core::Detector;

    fn create_test_dem() -> ParsedDem {
        let detectors = vec![
            Detector::new(0, 0.0, 0.0, 0.0),
            Detector::new(1, 1.0, 0.0, 0.0),
            Detector::new(2, 0.0, 1.0, 0.0),
            Detector::new(3, 1.0, 1.0, 0.0),
        ];

        let mechanisms = vec![
            OwnedErrorMechanism::new(0.5, vec![0, 1], 0),
            OwnedErrorMechanism::new(0.5, vec![2, 3], 0),
        ];

        ParsedDem {
            num_detectors: 4,
            num_observables: 0,
            detectors,
            mechanisms,
        }
    }

    #[test]
    fn test_circuit_sampler_basic() {
        let dem = create_test_dem();
        let mut sampler = CircuitSampler::new(&dem, 42);

        let (syndrome, _) = sampler.sample();
        assert_eq!(syndrome.len(), 1); // 4 detectors fit in 1 word
    }

    #[test]
    fn test_circuit_sampler_batch() {
        let dem = create_test_dem();
        let mut sampler = CircuitSampler::new(&dem, 42);

        let batch = sampler.sample_batch(100);
        assert_eq!(batch.len(), 100);
    }

    #[test]
    fn test_syndrome_parity() {
        // With 50% error probability on 2-detector errors,
        // syndromes should have even parity (or match boundary)
        let dem = create_test_dem();
        let mut sampler = CircuitSampler::new(&dem, 12345);

        for _ in 0..100 {
            let (syndrome, _) = sampler.sample();
            let total_defects: u32 = syndrome.iter().map(|w| w.count_ones()).sum();
            // Each error flips exactly 2 detectors, so total should be even
            assert_eq!(total_defects % 2, 0);
        }
    }
}
