//! Syndrome splitter for X/Z basis decoding.
//!
//! This module provides utilities to split a unified syndrome (containing both
//! X and Z stabilizer measurements) into separate X-only and Z-only syndromes
//! suitable for independent decoding.
//!
//! # Coordinate System
//!
//! In a rotated surface code, stabilizers are arranged in a checkerboard pattern:
//! - X stabilizers: positions where `(x + y) % 2 == 0`
//! - Z stabilizers: positions where `(x + y) % 2 == 1`
//!
//! # Compact Reindexing
//!
//! The splitter compacts each basis into a smaller grid:
//! - Original grid: `(d-1) x (d-1) x depth`
//! - Compact grid: `(d-1)/2 x (d-1) x depth` for each basis
//!
//! For Z stabilizers, coordinates are rotated so that the left/right boundaries
//! (which affect the Z logical observable) become top/bottom boundaries after
//! compaction.

use prav_core::Grid3DConfig;

/// Result of splitting a unified syndrome.
#[derive(Clone, Debug)]
pub struct SplitSyndromes {
    /// Compact-indexed X stabilizer syndrome.
    pub x_syndrome: Vec<u64>,
    /// Compact-indexed Z stabilizer syndrome (coordinates rotated).
    pub z_syndrome: Vec<u64>,
    /// Ground truth X logical observable (bit 0 of original).
    pub x_logical: u8,
    /// Ground truth Z logical observable (bit 1 of original).
    pub z_logical: u8,
}

/// Splits unified syndromes into X-only and Z-only compact syndromes.
///
/// The splitter handles the mapping from the full (d-1)x(d-1)xD grid
/// to compact (d-1)/2 x (d-1) x D grids for each stabilizer type.
pub struct SyndromeSplitter {
    /// Original grid width (d-1 for distance d).
    width: usize,
    /// Original grid height (d-1 for distance d).
    height: usize,
    /// Depth (number of measurement rounds).
    depth: usize,
    /// Original stride_y.
    stride_y: usize,
    /// Original stride_z.
    stride_z: usize,
    /// Compact width (width/2).
    compact_width: usize,
    /// Compact stride_y for the compact grid.
    compact_stride_y: usize,
    /// Compact stride_z for the compact grid.
    compact_stride_z: usize,
    /// Number of detectors per round in the original grid.
    detectors_per_round: usize,
    /// Number of X detectors per round.
    x_detectors_per_round: usize,
    /// Number of Z detectors per round.
    z_detectors_per_round: usize,
}

impl SyndromeSplitter {
    /// Create a new splitter for the given grid configuration.
    pub fn new(config: &Grid3DConfig) -> Self {
        let width = config.width;
        let height = config.height;
        let depth = config.depth;

        // Compact dimensions
        let compact_width = width / 2;
        let compact_height = height;

        // Compute compact strides
        let max_compact_dim = compact_width.max(compact_height).max(depth);
        let compact_stride_y = max_compact_dim.next_power_of_two();
        let compact_stride_z = compact_stride_y * compact_stride_y;

        // Count X and Z detectors per round
        // For even width, X and Z counts are equal
        let total_per_round = width * height;
        let x_per_round = total_per_round / 2;
        let z_per_round = total_per_round - x_per_round;

        Self {
            width,
            height,
            depth,
            stride_y: config.stride_y,
            stride_z: config.stride_z,
            compact_width,
            compact_stride_y,
            compact_stride_z,
            detectors_per_round: total_per_round,
            x_detectors_per_round: x_per_round,
            z_detectors_per_round: z_per_round,
        }
    }

    /// Create a splitter for X-only decoding configuration.
    pub fn for_x_config(d: usize, depth: usize) -> Self {
        let config = Grid3DConfig::for_x_stabilizers(d, depth);
        // For X-only, the compact config already has the right dimensions
        Self {
            width: d - 1,
            height: d - 1,
            depth,
            stride_y: (d - 1).next_power_of_two(),
            stride_z: (d - 1).next_power_of_two().pow(2),
            compact_width: config.width,
            compact_stride_y: config.stride_y,
            compact_stride_z: config.stride_z,
            detectors_per_round: (d - 1) * (d - 1),
            x_detectors_per_round: (d - 1) * (d - 1) / 2,
            z_detectors_per_round: (d - 1) * (d - 1) / 2,
        }
    }

    /// Split a unified syndrome into X and Z components.
    ///
    /// # Arguments
    /// * `unified` - The unified syndrome in prav format (stride-based layout)
    /// * `logical_flips` - Ground truth logical flips (bit 0 = X, bit 1 = Z)
    ///
    /// # Returns
    /// Split syndromes with compact indexing for each basis.
    pub fn split(&self, unified: &[u64], logical_flips: u8) -> SplitSyndromes {
        // Calculate output sizes
        let x_total_nodes = self.compact_stride_z * self.depth;
        let z_total_nodes = self.compact_stride_z * self.depth;
        let x_num_words = x_total_nodes.div_ceil(64);
        let z_num_words = z_total_nodes.div_ceil(64);

        let mut x_syndrome = vec![0u64; x_num_words];
        let mut z_syndrome = vec![0u64; z_num_words];

        // Iterate through original grid positions
        for t in 0..self.depth {
            for y in 0..self.height {
                for x in 0..self.width {
                    let parity = (x + y) % 2;

                    // Compute original linear index
                    let orig_linear = t * self.stride_z + y * self.stride_y + x;
                    let orig_word = orig_linear / 64;
                    let orig_bit = orig_linear % 64;

                    // Check if bit is set in unified syndrome
                    if orig_word < unified.len() && (unified[orig_word] & (1 << orig_bit)) != 0 {
                        if parity == 0 {
                            // X stabilizer
                            if let Some((compact_linear, _)) =
                                self.unified_to_x_compact(x, y, t)
                            {
                                let word = compact_linear / 64;
                                let bit = compact_linear % 64;
                                if word < x_syndrome.len() {
                                    x_syndrome[word] |= 1 << bit;
                                }
                            }
                        } else {
                            // Z stabilizer
                            if let Some((compact_linear, _)) =
                                self.unified_to_z_compact(x, y, t)
                            {
                                let word = compact_linear / 64;
                                let bit = compact_linear % 64;
                                if word < z_syndrome.len() {
                                    z_syndrome[word] |= 1 << bit;
                                }
                            }
                        }
                    }
                }
            }
        }

        SplitSyndromes {
            x_syndrome,
            z_syndrome,
            x_logical: logical_flips & 0x01,
            z_logical: (logical_flips >> 1) & 0x01,
        }
    }

    /// Map unified (x, y, t) to compact X index.
    ///
    /// Returns `Some((compact_linear_idx, (cx, cy, ct)))` if this is an X stabilizer,
    /// or `None` if it's a Z stabilizer.
    fn unified_to_x_compact(&self, x: usize, y: usize, t: usize) -> Option<(usize, (usize, usize, usize))> {
        // X stabilizers have (x + y) % 2 == 0
        if (x + y) % 2 != 0 {
            return None;
        }

        // Compact indexing: within each row, X positions are every other column
        // Row y has X at columns: y%2, y%2+2, y%2+4, ...
        // So compact x = (x - y%2) / 2
        let cx = (x - (y % 2)) / 2;
        let cy = y;
        let ct = t;

        let compact_linear = ct * self.compact_stride_z + cy * self.compact_stride_y + cx;
        Some((compact_linear, (cx, cy, ct)))
    }

    /// Map unified (x, y, t) to compact Z index.
    ///
    /// For Z, we rotate coordinates so left/right boundaries become top/bottom.
    /// Returns `Some((compact_linear_idx, (cx, cy, ct)))` if this is a Z stabilizer,
    /// or `None` if it's an X stabilizer.
    fn unified_to_z_compact(&self, x: usize, y: usize, t: usize) -> Option<(usize, (usize, usize, usize))> {
        // Z stabilizers have (x + y) % 2 == 1
        if (x + y) % 2 != 1 {
            return None;
        }

        // For Z, we rotate: swap x and y so left/right (x boundaries) become top/bottom (y boundaries)
        // After rotation: original (x, y) -> (y, x)
        // Then compact: within each row (now indexed by original x), the Z positions are at every other column
        //
        // After rotation:
        // - New row index cy = original x
        // - New column positions in row cy: these are the original y values where (x + y) % 2 == 1
        //   - If x is even: y must be odd -> y = 1, 3, 5, ...
        //   - If x is odd: y must be even -> y = 0, 2, 4, ...
        //
        // Compact x = y / 2 (integer division)
        let cy = x;  // Original x becomes new row
        let cx = y / 2;  // Original y (0,2,4... or 1,3,5...) compacts to 0,1,2...
        let ct = t;

        let compact_linear = ct * self.compact_stride_z + cy * self.compact_stride_y + cx;
        Some((compact_linear, (cx, cy, ct)))
    }

    /// Get the X compact grid configuration.
    pub fn x_config(&self, d: usize) -> Grid3DConfig {
        Grid3DConfig::for_x_stabilizers(d, self.depth)
    }

    /// Get the Z compact grid configuration.
    pub fn z_config(&self, d: usize) -> Grid3DConfig {
        Grid3DConfig::for_z_stabilizers(d, self.depth)
    }

    /// Number of X detectors per measurement round.
    pub fn x_detectors_per_round(&self) -> usize {
        self.x_detectors_per_round
    }

    /// Number of Z detectors per measurement round.
    pub fn z_detectors_per_round(&self) -> usize {
        self.z_detectors_per_round
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prav_core::TestGrids3D;

    #[test]
    fn test_splitter_creation() {
        let config = TestGrids3D::D5;
        let splitter = SyndromeSplitter::new(&config);

        assert_eq!(splitter.width, 4);
        assert_eq!(splitter.height, 4);
        assert_eq!(splitter.depth, 5);
        assert_eq!(splitter.compact_width, 2);
        // For 4x4 grid, X and Z each have 8 detectors per round
        assert_eq!(splitter.x_detectors_per_round, 8);
        assert_eq!(splitter.z_detectors_per_round, 8);
    }

    #[test]
    fn test_x_compact_mapping() {
        let config = TestGrids3D::D5;
        let splitter = SyndromeSplitter::new(&config);

        // X stabilizers at (x+y) % 2 == 0
        // (0,0): X, cx = (0 - 0) / 2 = 0, cy = 0
        assert!(splitter.unified_to_x_compact(0, 0, 0).is_some());
        let (_, (cx, cy, _)) = splitter.unified_to_x_compact(0, 0, 0).unwrap();
        assert_eq!((cx, cy), (0, 0));

        // (2,0): X, cx = (2 - 0) / 2 = 1, cy = 0
        assert!(splitter.unified_to_x_compact(2, 0, 0).is_some());
        let (_, (cx, cy, _)) = splitter.unified_to_x_compact(2, 0, 0).unwrap();
        assert_eq!((cx, cy), (1, 0));

        // (1,1): X, cx = (1 - 1) / 2 = 0, cy = 1
        assert!(splitter.unified_to_x_compact(1, 1, 0).is_some());
        let (_, (cx, cy, _)) = splitter.unified_to_x_compact(1, 1, 0).unwrap();
        assert_eq!((cx, cy), (0, 1));

        // (1,0): Z, should return None
        assert!(splitter.unified_to_x_compact(1, 0, 0).is_none());
    }

    #[test]
    fn test_z_compact_mapping() {
        let config = TestGrids3D::D5;
        let splitter = SyndromeSplitter::new(&config);

        // Z stabilizers at (x+y) % 2 == 1
        // (1,0): Z
        assert!(splitter.unified_to_z_compact(1, 0, 0).is_some());

        // (0,1): Z
        assert!(splitter.unified_to_z_compact(0, 1, 0).is_some());

        // (0,0): X, should return None
        assert!(splitter.unified_to_z_compact(0, 0, 0).is_none());
    }

    #[test]
    fn test_split_empty_syndrome() {
        let config = TestGrids3D::D5;
        let splitter = SyndromeSplitter::new(&config);

        let unified = vec![0u64; 16]; // Empty syndrome
        let split = splitter.split(&unified, 0);

        // All should be empty
        assert!(split.x_syndrome.iter().all(|&w| w == 0));
        assert!(split.z_syndrome.iter().all(|&w| w == 0));
        assert_eq!(split.x_logical, 0);
        assert_eq!(split.z_logical, 0);
    }

    #[test]
    fn test_split_single_x_defect() {
        let config = TestGrids3D::D5;
        let splitter = SyndromeSplitter::new(&config);

        // Set bit at (0, 0, 0) which is an X stabilizer
        let linear = config.coord_to_linear(0, 0, 0);
        let word = linear / 64;
        let bit = linear % 64;

        let mut unified = vec![0u64; 16];
        unified[word] |= 1 << bit;

        let split = splitter.split(&unified, 0);

        // X syndrome should have exactly one bit set
        let x_defect_count: u32 = split.x_syndrome.iter().map(|w| w.count_ones()).sum();
        assert_eq!(x_defect_count, 1);

        // Z syndrome should be empty
        let z_defect_count: u32 = split.z_syndrome.iter().map(|w| w.count_ones()).sum();
        assert_eq!(z_defect_count, 0);
    }

    #[test]
    fn test_split_single_z_defect() {
        let config = TestGrids3D::D5;
        let splitter = SyndromeSplitter::new(&config);

        // Set bit at (1, 0, 0) which is a Z stabilizer
        let linear = config.coord_to_linear(1, 0, 0);
        let word = linear / 64;
        let bit = linear % 64;

        let mut unified = vec![0u64; 16];
        unified[word] |= 1 << bit;

        let split = splitter.split(&unified, 0);

        // X syndrome should be empty
        let x_defect_count: u32 = split.x_syndrome.iter().map(|w| w.count_ones()).sum();
        assert_eq!(x_defect_count, 0);

        // Z syndrome should have exactly one bit set
        let z_defect_count: u32 = split.z_syndrome.iter().map(|w| w.count_ones()).sum();
        assert_eq!(z_defect_count, 1);
    }

    #[test]
    fn test_logical_split() {
        let config = TestGrids3D::D5;
        let splitter = SyndromeSplitter::new(&config);

        let unified = vec![0u64; 16];

        // Test X logical only
        let split = splitter.split(&unified, 0x01);
        assert_eq!(split.x_logical, 1);
        assert_eq!(split.z_logical, 0);

        // Test Z logical only
        let split = splitter.split(&unified, 0x02);
        assert_eq!(split.x_logical, 0);
        assert_eq!(split.z_logical, 1);

        // Test both
        let split = splitter.split(&unified, 0x03);
        assert_eq!(split.x_logical, 1);
        assert_eq!(split.z_logical, 1);
    }

    #[test]
    fn test_defect_count_conservation() {
        let config = TestGrids3D::D5;
        let splitter = SyndromeSplitter::new(&config);

        // Create unified syndrome with known number of X and Z defects
        let mut unified = vec![0u64; 16];

        // Set 4 X defects: (0,0), (2,0), (1,1), (3,1)
        for &(x, y) in &[(0, 0), (2, 0), (1, 1), (3, 1)] {
            let linear = config.coord_to_linear(x, y, 0);
            unified[linear / 64] |= 1 << (linear % 64);
        }

        // Set 3 Z defects: (1,0), (0,1), (2,1)
        for &(x, y) in &[(1, 0), (0, 1), (2, 1)] {
            let linear = config.coord_to_linear(x, y, 0);
            unified[linear / 64] |= 1 << (linear % 64);
        }

        let split = splitter.split(&unified, 0);

        let x_defect_count: u32 = split.x_syndrome.iter().map(|w| w.count_ones()).sum();
        let z_defect_count: u32 = split.z_syndrome.iter().map(|w| w.count_ones()).sum();

        assert_eq!(x_defect_count, 4);
        assert_eq!(z_defect_count, 3);
    }
}
