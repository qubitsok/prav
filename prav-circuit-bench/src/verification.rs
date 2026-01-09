//! Verification of decoder corrections.

use prav_core::{EdgeCorrection, Grid3DConfig};

/// Result of verification including logical error analysis.
#[derive(Debug, Clone, Copy)]
pub struct VerificationResult {
    /// Whether all defects were resolved.
    pub defects_resolved: bool,
    /// Predicted logical frame changes from boundary corrections.
    /// bit 0 = X, bit 1 = Z
    pub predicted_logical: u8,
}

/// Verify corrections and compute predicted logical flips.
///
/// Returns verification result with defect resolution and predicted logical.
///
/// # Logical Boundary Mapping
///
/// For rotated surface codes, stabilizer type is determined by coordinate parity:
/// - Even parity (x + y) % 2 == 0 → Z stabilizer → boundary corrections affect Z logical
/// - Odd parity (x + y) % 2 == 1 → X stabilizer → boundary corrections affect X logical
///
/// This fixes the corner boundary bug where corners would incorrectly flip both logicals.
pub fn verify_with_logical(
    syndrome: &[u64],
    corrections: &[EdgeCorrection],
    config: &Grid3DConfig,
) -> VerificationResult {
    let mut state = syndrome.to_vec();
    let mut predicted_logical = 0u8;

    for corr in corrections {
        let u = corr.u as usize;

        // Flip u
        let blk_u = u / 64;
        let bit_u = u % 64;
        if blk_u < state.len() {
            state[blk_u] ^= 1 << bit_u;
        }

        // Boundary correction
        if corr.v == u32::MAX {
            // Determine which logical based on stabilizer type (coordinate parity)
            let (x, y, _t) = config.linear_to_coord(u);

            // Check if this is actually a boundary position
            let on_x_boundary = x == 0 || x == config.width - 1;
            let on_y_boundary = y == 0 || y == config.height - 1;

            if on_x_boundary || on_y_boundary {
                // Use stabilizer parity to determine which logical is affected
                // Even parity → Z stabilizer → Z logical (bit 1)
                // Odd parity → X stabilizer → X logical (bit 0)
                let stabilizer_parity = (x + y) % 2;
                if stabilizer_parity == 0 {
                    predicted_logical ^= 0b10; // Z logical
                } else {
                    predicted_logical ^= 0b01; // X logical
                }
            }
        } else {
            // Interior edge - flip v
            let v = corr.v as usize;
            let blk_v = v / 64;
            let bit_v = v % 64;
            if blk_v < state.len() {
                state[blk_v] ^= 1 << bit_v;
            }
        }
    }

    // Check all defects resolved
    let defects_resolved = check_all_defects_resolved(&state, config);

    VerificationResult {
        defects_resolved,
        predicted_logical,
    }
}

fn check_all_defects_resolved(state: &[u64], config: &Grid3DConfig) -> bool {
    for t in 0..config.depth {
        for y in 0..config.height {
            for x in 0..config.width {
                let linear = config.coord_to_linear(x, y, t);
                let blk = linear / 64;
                let bit = linear % 64;
                if blk < state.len() && (state[blk] & (1 << bit)) != 0 {
                    return false;
                }
            }
        }
    }
    true
}

/// Verify that corrections resolve all syndromes.
///
/// Returns true if all defects are resolved after applying corrections.
pub fn verify_corrections(
    syndrome: &[u64],
    corrections: &[EdgeCorrection],
    config: &Grid3DConfig,
) -> bool {
    // Copy syndrome
    let mut state = syndrome.to_vec();

    // Apply corrections (XOR endpoints)
    for corr in corrections {
        let u = corr.u as usize;

        // Flip u
        let blk_u = u / 64;
        let bit_u = u % 64;
        if blk_u < state.len() {
            state[blk_u] ^= 1 << bit_u;
        }

        // Flip v (if not boundary)
        if corr.v != u32::MAX {
            let v = corr.v as usize;
            let blk_v = v / 64;
            let bit_v = v % 64;
            if blk_v < state.len() {
                state[blk_v] ^= 1 << bit_v;
            }
        }
    }

    // Check all defects resolved (only within valid region)
    for t in 0..config.depth {
        for y in 0..config.height {
            for x in 0..config.width {
                let linear = config.coord_to_linear(x, y, t);
                let blk = linear / 64;
                let bit = linear % 64;
                if blk < state.len() && (state[blk] & (1 << bit)) != 0 {
                    return false; // Unresolved defect
                }
            }
        }
    }

    true
}

/// Count defects in a syndrome.
pub fn count_defects(syndrome: &[u64]) -> usize {
    syndrome.iter().map(|w| w.count_ones() as usize).sum()
}

/// Count defects within valid region of a 3D grid.
pub fn count_valid_defects(syndrome: &[u64], config: &Grid3DConfig) -> usize {
    let mut count = 0;
    for t in 0..config.depth {
        for y in 0..config.height {
            for x in 0..config.width {
                let linear = config.coord_to_linear(x, y, t);
                let blk = linear / 64;
                let bit = linear % 64;
                if blk < syndrome.len() && (syndrome[blk] & (1 << bit)) != 0 {
                    count += 1;
                }
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use prav_core::TestGrids3D;

    #[test]
    fn test_verify_empty() {
        let config = TestGrids3D::D3;
        let syndrome = vec![0u64; 10];
        let corrections = vec![];

        assert!(verify_corrections(&syndrome, &corrections, &config));
    }

    #[test]
    fn test_count_defects() {
        let syndrome = vec![0b1010u64, 0b1111u64];
        assert_eq!(count_defects(&syndrome), 6); // 2 + 4
    }
}
