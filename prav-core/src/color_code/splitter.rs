//! Syndrome splitting for color code decoding.
//!
//! This module splits a unified color code syndrome into three separate
//! syndromes (one per color class) for the restriction decoder.
//!
//! # Syndrome Representation
//!
//! Syndromes can be represented in multiple formats:
//! - **Dense**: Bitmask where bit `i` indicates defect at position `i`
//! - **Sparse**: List of defect indices
//!
//! Both representations are supported with efficient conversion.

use crate::color_code::grid_3d::ColorCodeGrid3DConfig;
use crate::color_code::types::FaceColor;

/// Splits a dense syndrome bitmask by color.
///
/// Given a syndrome as a slice of u64 bitmasks (one per 64-node block),
/// produces three separate syndromes for each color class.
///
/// # Arguments
/// * `syndrome` - Dense syndrome (bit `i` in `syndrome[i/64]` indicates defect at node `i`)
/// * `config` - Grid configuration
/// * `red_out` - Output buffer for red defects (sparse list)
/// * `green_out` - Output buffer for green defects (sparse list)
/// * `blue_out` - Output buffer for blue defects (sparse list)
///
/// # Returns
/// `(red_count, green_count, blue_count)` - Number of defects in each color
pub fn split_dense_syndrome(
    syndrome: &[u64],
    config: &ColorCodeGrid3DConfig,
    red_out: &mut [u32],
    green_out: &mut [u32],
    blue_out: &mut [u32],
) -> (usize, usize, usize) {
    let mut counts = [0usize; 3];

    for t in 0..config.depth {
        for y in 0..config.height {
            for x in 0..config.width {
                let idx = config.coord_to_linear(x, y, t);
                let block = idx / 64;
                let bit = idx % 64;

                // Check if this position has a defect
                if block < syndrome.len() && (syndrome[block] & (1u64 << bit)) != 0 {
                    let color = config.detector_color(x, y);
                    let color_idx = color.index();

                    match color {
                        FaceColor::Red => {
                            if counts[0] < red_out.len() {
                                red_out[counts[0]] = idx as u32;
                            }
                        }
                        FaceColor::Green => {
                            if counts[1] < green_out.len() {
                                green_out[counts[1]] = idx as u32;
                            }
                        }
                        FaceColor::Blue => {
                            if counts[2] < blue_out.len() {
                                blue_out[counts[2]] = idx as u32;
                            }
                        }
                    }
                    counts[color_idx] += 1;
                }
            }
        }
    }

    (counts[0], counts[1], counts[2])
}

/// Splits a sparse syndrome (list of defect indices) by color.
///
/// # Arguments
/// * `defects` - List of defect indices in full grid
/// * `config` - Grid configuration
/// * `red_out` - Output buffer for red defects
/// * `green_out` - Output buffer for green defects
/// * `blue_out` - Output buffer for blue defects
///
/// # Returns
/// `(red_count, green_count, blue_count)` - Number of defects in each color
pub fn split_sparse_syndrome(
    defects: &[u32],
    config: &ColorCodeGrid3DConfig,
    red_out: &mut [u32],
    green_out: &mut [u32],
    blue_out: &mut [u32],
) -> (usize, usize, usize) {
    let mut counts = [0usize; 3];

    for &idx in defects {
        let color = config.detector_color_at(idx as usize);
        let color_idx = color.index();

        match color {
            FaceColor::Red => {
                if counts[0] < red_out.len() {
                    red_out[counts[0]] = idx;
                }
            }
            FaceColor::Green => {
                if counts[1] < green_out.len() {
                    green_out[counts[1]] = idx;
                }
            }
            FaceColor::Blue => {
                if counts[2] < blue_out.len() {
                    blue_out[counts[2]] = idx;
                }
            }
        }
        counts[color_idx] += 1;
    }

    (counts[0], counts[1], counts[2])
}

/// Convert sparse defect list to dense bitmask.
///
/// # Arguments
/// * `defects` - List of defect indices
/// * `defect_count` - Number of valid defects in the list
/// * `syndrome_out` - Output dense syndrome (must be zeroed)
pub fn sparse_to_dense(defects: &[u32], defect_count: usize, syndrome_out: &mut [u64]) {
    for &idx in defects.iter().take(defect_count) {
        let block = idx as usize / 64;
        let bit = idx as usize % 64;
        if block < syndrome_out.len() {
            syndrome_out[block] |= 1u64 << bit;
        }
    }
}

/// Convert dense syndrome to sparse defect list.
///
/// # Arguments
/// * `syndrome` - Dense syndrome bitmask
/// * `defects_out` - Output buffer for defect indices
///
/// # Returns
/// Number of defects found
pub fn dense_to_sparse(syndrome: &[u64], defects_out: &mut [u32]) -> usize {
    let mut count = 0;

    for (block_idx, &block) in syndrome.iter().enumerate() {
        if block == 0 {
            continue;
        }

        let mut remaining = block;
        while remaining != 0 {
            let bit = remaining.trailing_zeros() as usize;
            let idx = block_idx * 64 + bit;

            if count < defects_out.len() {
                defects_out[count] = idx as u32;
            }
            count += 1;

            remaining &= remaining - 1; // Clear lowest set bit
        }
    }

    count
}

/// In-place color-aware syndrome processing.
///
/// This struct provides efficient syndrome splitting without additional
/// allocations by using pre-sized buffers.
pub struct SyndromeSplitter {
    /// Maximum defects per color (buffer size).
    max_defects: usize,
}

impl SyndromeSplitter {
    /// Create a new splitter with given buffer capacity.
    #[must_use]
    pub const fn new(max_defects: usize) -> Self {
        Self { max_defects }
    }

    /// Required buffer size for each color's defect list.
    #[must_use]
    pub const fn buffer_size(&self) -> usize {
        self.max_defects
    }

    /// Split syndrome and return counts.
    ///
    /// # Safety
    ///
    /// Output buffers must have at least `buffer_size()` elements.
    pub fn split(
        &self,
        defects: &[u32],
        config: &ColorCodeGrid3DConfig,
        red_out: &mut [u32],
        green_out: &mut [u32],
        blue_out: &mut [u32],
    ) -> (usize, usize, usize) {
        debug_assert!(red_out.len() >= self.max_defects);
        debug_assert!(green_out.len() >= self.max_defects);
        debug_assert!(blue_out.len() >= self.max_defects);

        split_sparse_syndrome(defects, config, red_out, green_out, blue_out)
    }
}

/// Verify that defects satisfy the color code parity constraint.
///
/// In a valid color code syndrome (from a single error), defects should
/// come in pairs of the same color. This function checks if the total
/// number of defects per color is even.
///
/// # Arguments
/// * `red_count` - Number of red defects
/// * `green_count` - Number of green defects
/// * `blue_count` - Number of blue defects
///
/// # Returns
/// `true` if all color classes have even parity (valid syndrome)
#[must_use]
pub const fn check_color_parity(red_count: usize, green_count: usize, blue_count: usize) -> bool {
    (red_count % 2 == 0) && (green_count % 2 == 0) && (blue_count % 2 == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_sparse_empty() {
        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(5);
        let defects: [u32; 0] = [];
        let mut red = [0u32; 8];
        let mut green = [0u32; 8];
        let mut blue = [0u32; 8];

        let (r, g, b) = split_sparse_syndrome(&defects, &config, &mut red, &mut green, &mut blue);

        assert_eq!((r, g, b), (0, 0, 0));
    }

    #[test]
    fn test_split_sparse_single_color() {
        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(5);

        // Create defects at positions that are all Red (x+y = 0 mod 3)
        let defects = [
            config.coord_to_linear(0, 0, 0) as u32, // 0+0=0
            config.coord_to_linear(3, 0, 0) as u32, // 3+0=0
            config.coord_to_linear(0, 3, 0) as u32, // 0+3=0
        ];

        let mut red = [0u32; 8];
        let mut green = [0u32; 8];
        let mut blue = [0u32; 8];

        let (r, g, b) = split_sparse_syndrome(&defects, &config, &mut red, &mut green, &mut blue);

        assert_eq!(r, 3);
        assert_eq!(g, 0);
        assert_eq!(b, 0);
    }

    #[test]
    fn test_split_sparse_all_colors() {
        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(5);

        let defects = [
            config.coord_to_linear(0, 0, 0) as u32, // Red (0)
            config.coord_to_linear(1, 0, 0) as u32, // Green (1)
            config.coord_to_linear(2, 0, 0) as u32, // Blue (2)
        ];

        let mut red = [0u32; 8];
        let mut green = [0u32; 8];
        let mut blue = [0u32; 8];

        let (r, g, b) = split_sparse_syndrome(&defects, &config, &mut red, &mut green, &mut blue);

        assert_eq!(r, 1);
        assert_eq!(g, 1);
        assert_eq!(b, 1);
    }

    #[test]
    fn test_dense_sparse_roundtrip() {
        let mut dense = [0u64; 4];
        dense[0] = 0b101010;
        dense[1] = 0b110011;

        let mut sparse = [0u32; 64];
        let count = dense_to_sparse(&dense, &mut sparse);

        let mut dense2 = [0u64; 4];
        sparse_to_dense(&sparse, count, &mut dense2);

        assert_eq!(dense, dense2);
    }

    #[test]
    fn test_color_parity_check() {
        assert!(check_color_parity(0, 0, 0));
        assert!(check_color_parity(2, 2, 2));
        assert!(check_color_parity(4, 0, 2));

        assert!(!check_color_parity(1, 0, 0));
        assert!(!check_color_parity(0, 1, 0));
        assert!(!check_color_parity(0, 0, 1));
        assert!(!check_color_parity(1, 1, 1));
    }

    #[test]
    fn test_split_dense_syndrome() {
        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(3);

        // Create a dense syndrome with defects at known positions
        let mut syndrome = [0u64; 16];
        let idx0 = config.coord_to_linear(0, 0, 0); // Red
        let idx1 = config.coord_to_linear(1, 0, 0); // Green
        let idx2 = config.coord_to_linear(2, 0, 0); // Blue

        syndrome[idx0 / 64] |= 1u64 << (idx0 % 64);
        syndrome[idx1 / 64] |= 1u64 << (idx1 % 64);
        syndrome[idx2 / 64] |= 1u64 << (idx2 % 64);

        let mut red = [0u32; 16];
        let mut green = [0u32; 16];
        let mut blue = [0u32; 16];

        let (r, g, b) = split_dense_syndrome(&syndrome, &config, &mut red, &mut green, &mut blue);

        assert_eq!(r, 1);
        assert_eq!(g, 1);
        assert_eq!(b, 1);
    }

    #[test]
    fn test_splitter_struct() {
        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(5);
        let splitter = SyndromeSplitter::new(64);

        let defects = [
            config.coord_to_linear(0, 0, 0) as u32,
            config.coord_to_linear(1, 0, 0) as u32,
        ];

        let mut red = [0u32; 64];
        let mut green = [0u32; 64];
        let mut blue = [0u32; 64];

        let (r, g, b) = splitter.split(&defects, &config, &mut red, &mut green, &mut blue);

        assert_eq!(r + g + b, 2);
    }
}
