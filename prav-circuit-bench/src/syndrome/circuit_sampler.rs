//! Circuit-level syndrome sampling from DEM.

use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rand::SeedableRng;

use crate::dem::types::ParsedDem;

/// Sample syndromes from a parsed DEM.
pub struct CircuitSampler {
    /// Error mechanism probabilities.
    probabilities: Vec<f32>,
    /// Error mechanism detector targets.
    detector_targets: Vec<Vec<u32>>,
    /// Error mechanism frame changes.
    frame_changes: Vec<u8>,
    /// Number of detectors.
    num_detectors: u32,
    /// Random number generator.
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
        let num_words = (self.num_detectors as usize + 63) / 64;
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
