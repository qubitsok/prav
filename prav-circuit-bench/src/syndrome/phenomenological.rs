//! Phenomenological noise model for 3D syndrome generation.
//!
//! This module provides a correlated phenomenological noise model that
//! approximates circuit-level noise without requiring a full DEM file.
//!
//! # Error Types
//!
//! ```text
//! Space-like error (data qubit):     Time-like error (measurement):
//!
//!   t:  ○──●──●──○                    t:    ○──●──○
//!          └──┘                             │
//!       both flipped                   t+1:  ○──●──○
//!                                           same detector flipped twice
//! ```
//!
//! # Logical Errors
//!
//! Boundary errors contribute to logical frame changes:
//! - Left boundary (x=0): contributes to logical Z
//! - Bottom boundary (y=0): contributes to logical X

use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rand::SeedableRng;

use prav_core::Grid3DConfig;

/// Result of syndrome generation with logical error tracking.
#[derive(Clone, Debug)]
pub struct SyndromeWithLogical {
    /// Packed syndrome bits (one bit per detector).
    pub syndrome: Vec<u64>,
    /// Ground truth logical frame changes (bit 0 = X, bit 1 = Z).
    pub logical_flips: u8,
}

/// Generate independent (uncorrelated) phenomenological syndromes.
///
/// Each detector has independent probability `p` of being flipped.
/// This does NOT track logical errors (always returns 0).
pub fn generate_phenomenological_syndromes(
    config: &Grid3DConfig,
    error_prob: f64,
    num_shots: usize,
    seed: u64,
) -> Vec<Vec<u64>> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let num_words = (config.alloc_nodes() + 63) / 64;

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
/// This model has proper error structure:
/// - Space-like errors flip pairs of adjacent detectors
/// - Time-like errors flip the same detector at adjacent times
/// - Boundary errors contribute to logical frame changes
///
/// Returns syndromes with ground-truth logical flips for verification.
pub fn generate_correlated_syndromes(
    config: &Grid3DConfig,
    p_space: f64,  // Probability of space-like (data qubit) error
    p_time: f64,   // Probability of time-like (measurement) error
    num_shots: usize,
    seed: u64,
) -> Vec<SyndromeWithLogical> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let num_words = (config.alloc_nodes() + 63) / 64;

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
                // Left boundary contributes to logical Z
                logical_flips ^= 0b10; // bit 1 = Z
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
                flip_detector(&mut syndrome, config, config.width - 1, y, t, num_words);
                // Right boundary also contributes to logical Z
                logical_flips ^= 0b10; // bit 1 = Z
            }
        }

        // Vertical edges (including top/bottom boundaries)
        for x in 0..config.width {
            // Bottom boundary edge (y=0, only flips one detector)
            if rng.random::<f64>() < p_space {
                flip_detector(&mut syndrome, config, x, 0, t, num_words);
                // Bottom boundary contributes to logical X
                logical_flips ^= 0b01; // bit 0 = X
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
                flip_detector(&mut syndrome, config, x, config.height - 1, t, num_words);
                // Top boundary also contributes to logical X
                logical_flips ^= 0b01; // bit 0 = X
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

/// Count defects in a syndrome.
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
            assert!(syn.len() >= (config.num_detectors() + 63) / 64);
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
            assert!(r.syndrome.len() >= (config.num_detectors() + 63) / 64);
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
            if total % 2 == 0 {
                even_count += 1;
            }
        }
        // At p=0.001, we expect ~90%+ even parity
        assert!(even_count > 800, "Expected mostly even parity, got {}/1000", even_count);
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
