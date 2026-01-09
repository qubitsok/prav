//! 3D grid configurations for color code decoding.
//!
//! This module provides configurations for 3D space-time decoding grids
//! for triangular color codes. The third dimension represents measurement
//! rounds (time).
//!
//! # Triangular Color Code Layout
//!
//! For a triangular (6,6,6) color code of distance `d`:
//! - Spatial dimensions depend on the lattice structure
//! - Time dimension: `d` measurement rounds
//! - Three-colorable faces with colored boundaries

use crate::color_code::types::{ColorCodeBoundaryConfig, FaceColor};
use crate::decoder::types::BoundaryConfig;

/// Type of triangular color code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorCodeType {
    /// (6,6,6) triangular color code - hexagonal tiling.
    /// Each face is a hexagon with 6 neighbors.
    Triangular666,
    /// (4,8,8) color code - square-octagon tiling.
    Triangular488,
}

/// Configuration for 3D color code decoding grids.
///
/// Represents a 3D decoding volume where:
/// - X, Y are spatial dimensions (face/detector positions)
/// - Z is the time dimension (measurement rounds)
#[derive(Debug, Clone, Copy)]
pub struct ColorCodeGrid3DConfig {
    /// Code distance.
    pub code_distance: usize,
    /// Number of measurement rounds.
    pub num_rounds: usize,
    /// Grid width (X dimension).
    pub width: usize,
    /// Grid height (Y dimension).
    pub height: usize,
    /// Grid depth (time dimension).
    pub depth: usize,
    /// Y stride (next power of two from max dimension).
    pub stride_y: usize,
    /// Z stride (stride_y^2).
    pub stride_z: usize,
    /// Color code type.
    pub code_type: ColorCodeType,
    /// Boundary configuration for each color.
    pub boundary_config: ColorCodeBoundaryConfig,
}

impl ColorCodeGrid3DConfig {
    /// Create a 3D config for a (6,6,6) triangular color code.
    ///
    /// The (6,6,6) code uses a hexagonal tiling where each hexagon has 6 neighbors.
    /// For distance `d`:
    /// - Approximately `d^2` detectors per time slice
    /// - `d` measurement rounds
    ///
    /// # Arguments
    /// * `d` - Code distance (must be odd for color codes)
    #[must_use]
    pub const fn for_triangular_6_6_6(d: usize) -> Self {
        // For triangular color codes, the lattice is roughly d x d
        // The exact number depends on the boundary structure
        let detector_dim = if d > 1 { d } else { 1 };
        let num_rounds = d;

        let max_dim = max3(detector_dim, detector_dim, num_rounds);
        let stride_y = next_power_of_two(max_dim);
        let stride_z = stride_y * stride_y;

        Self {
            code_distance: d,
            num_rounds,
            width: detector_dim,
            height: detector_dim,
            depth: num_rounds,
            stride_y,
            stride_z,
            code_type: ColorCodeType::Triangular666,
            boundary_config: ColorCodeBoundaryConfig::all_boundaries(),
        }
    }

    /// Create a 3D config for a (4,8,8) color code.
    ///
    /// The (4,8,8) code uses a square-octagon tiling.
    #[must_use]
    pub const fn for_triangular_4_8_8(d: usize) -> Self {
        let detector_dim = if d > 1 { d } else { 1 };
        let num_rounds = d;

        let max_dim = max3(detector_dim, detector_dim, num_rounds);
        let stride_y = next_power_of_two(max_dim);
        let stride_z = stride_y * stride_y;

        Self {
            code_distance: d,
            num_rounds,
            width: detector_dim,
            height: detector_dim,
            depth: num_rounds,
            stride_y,
            stride_z,
            code_type: ColorCodeType::Triangular488,
            boundary_config: ColorCodeBoundaryConfig::all_boundaries(),
        }
    }

    /// Create a custom 3D config with explicit dimensions.
    #[must_use]
    pub const fn custom(width: usize, height: usize, depth: usize) -> Self {
        let max_dim = max3(width, height, depth);
        let stride_y = next_power_of_two(max_dim);
        let stride_z = stride_y * stride_y;

        Self {
            code_distance: 0, // Unknown for custom
            num_rounds: depth,
            width,
            height,
            depth,
            stride_y,
            stride_z,
            code_type: ColorCodeType::Triangular666,
            boundary_config: ColorCodeBoundaryConfig::all_boundaries(),
        }
    }

    /// Create a 2D config (single time slice) for a color code.
    #[must_use]
    pub const fn with_depth(self, new_depth: usize) -> Self {
        let max_dim = max3(self.width, self.height, new_depth);
        let stride_y = next_power_of_two(max_dim);
        let stride_z = stride_y * stride_y;

        Self {
            depth: new_depth,
            num_rounds: new_depth,
            stride_y,
            stride_z,
            ..self
        }
    }

    /// Total number of detectors in the 3D volume.
    #[inline(always)]
    #[must_use]
    pub const fn num_detectors(&self) -> usize {
        self.width * self.height * self.depth
    }

    /// Number of detectors per time slice.
    #[inline(always)]
    #[must_use]
    pub const fn detectors_per_round(&self) -> usize {
        self.width * self.height
    }

    /// Allocated nodes (power-of-two padded for Morton encoding).
    #[inline(always)]
    #[must_use]
    pub const fn alloc_nodes(&self) -> usize {
        if self.depth > 1 {
            self.stride_y * self.stride_y * self.stride_y
        } else {
            self.stride_y * self.stride_y
        }
    }

    /// Convert 3D coordinates to linear index.
    #[inline(always)]
    #[must_use]
    pub const fn coord_to_linear(&self, x: usize, y: usize, t: usize) -> usize {
        t * self.stride_z + y * self.stride_y + x
    }

    /// Convert linear index to 3D coordinates.
    #[inline(always)]
    #[must_use]
    pub const fn linear_to_coord(&self, idx: usize) -> (usize, usize, usize) {
        let t = idx / self.stride_z;
        let remainder = idx % self.stride_z;
        let y = remainder / self.stride_y;
        let x = remainder % self.stride_y;
        (x, y, t)
    }

    /// Get the color of a detector at given spatial coordinates.
    ///
    /// Color is determined by `(x + y) % 3` and is time-invariant.
    #[inline(always)]
    #[must_use]
    pub const fn detector_color(&self, x: usize, y: usize) -> FaceColor {
        FaceColor::from_coords(x as u32, y as u32)
    }

    /// Get the color of a detector at a linear index.
    #[inline(always)]
    #[must_use]
    pub const fn detector_color_at(&self, idx: usize) -> FaceColor {
        let (x, y, _t) = self.linear_to_coord(idx);
        self.detector_color(x, y)
    }

    /// Count detectors of each color in the grid.
    ///
    /// Returns `[red_count, green_count, blue_count]`.
    #[must_use]
    pub const fn count_by_color(&self) -> [usize; 3] {
        let mut counts = [0usize; 3];
        let mut y = 0;
        while y < self.height {
            let mut x = 0;
            while x < self.width {
                let color = self.detector_color(x, y);
                counts[color.index()] += self.depth;
                x += 1;
            }
            y += 1;
        }
        counts
    }

    /// Get the BoundaryConfig for a specific color.
    ///
    /// In color codes, each color has its own boundary for defect matching.
    /// This returns a standard BoundaryConfig suitable for the restricted
    /// subgraph decoder.
    #[must_use]
    pub const fn boundary_for_color(&self, color: FaceColor) -> BoundaryConfig {
        // For the restriction decoder, each color's restricted subgraph
        // uses horizontal_only boundaries (like X-stabilizers in surface codes)
        // This is because in the projection, boundaries become linear
        if self.boundary_config.is_active(color) {
            BoundaryConfig::horizontal_only()
        } else {
            BoundaryConfig::no_boundaries()
        }
    }

    /// Get the restricted subgraph dimensions for a given color.
    ///
    /// Each color class contains approximately 1/3 of the detectors.
    /// The restricted grid is compacted to remove gaps.
    ///
    /// Returns `(width, height, depth)` for the restricted subgraph.
    #[must_use]
    pub const fn restricted_dimensions(&self, _color: FaceColor) -> (usize, usize, usize) {
        // In a (x + y) % 3 coloring pattern:
        // - Each row has approximately width/3 detectors of each color
        // - Compact representation: ceiling division
        let restricted_width = (self.width + 2) / 3;
        let restricted_height = self.height;
        (restricted_width, restricted_height, self.depth)
    }
}

/// Maximum of three values (const fn).
const fn max3(a: usize, b: usize, c: usize) -> usize {
    let max_ab = if a > b { a } else { b };
    if max_ab > c {
        max_ab
    } else {
        c
    }
}

/// Next power of two (const fn).
const fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut v = n - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    #[cfg(target_pointer_width = "64")]
    {
        v |= v >> 32;
    }
    v + 1
}

/// Predefined color code configurations for benchmarking.
pub struct TestColorCodeGrids;

impl TestColorCodeGrids {
    /// Distance-3 triangular color code.
    pub const D3: ColorCodeGrid3DConfig = ColorCodeGrid3DConfig::for_triangular_6_6_6(3);
    /// Distance-5 triangular color code.
    pub const D5: ColorCodeGrid3DConfig = ColorCodeGrid3DConfig::for_triangular_6_6_6(5);
    /// Distance-7 triangular color code.
    pub const D7: ColorCodeGrid3DConfig = ColorCodeGrid3DConfig::for_triangular_6_6_6(7);
    /// Distance-9 triangular color code.
    pub const D9: ColorCodeGrid3DConfig = ColorCodeGrid3DConfig::for_triangular_6_6_6(9);
    /// Distance-11 triangular color code.
    pub const D11: ColorCodeGrid3DConfig = ColorCodeGrid3DConfig::for_triangular_6_6_6(11);
    /// Distance-13 triangular color code.
    pub const D13: ColorCodeGrid3DConfig = ColorCodeGrid3DConfig::for_triangular_6_6_6(13);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangular_666_d5() {
        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(5);
        assert_eq!(config.code_distance, 5);
        assert_eq!(config.num_rounds, 5);
        assert_eq!(config.width, 5);
        assert_eq!(config.height, 5);
        assert_eq!(config.depth, 5);
    }

    #[test]
    fn test_coord_roundtrip() {
        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(5);
        for t in 0..config.depth {
            for y in 0..config.height {
                for x in 0..config.width {
                    let idx = config.coord_to_linear(x, y, t);
                    let (x2, y2, t2) = config.linear_to_coord(idx);
                    assert_eq!((x, y, t), (x2, y2, t2));
                }
            }
        }
    }

    #[test]
    fn test_detector_color_consistency() {
        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(5);

        // Color should be consistent across time
        for t in 0..config.depth {
            for y in 0..config.height {
                for x in 0..config.width {
                    let idx = config.coord_to_linear(x, y, t);
                    let color1 = config.detector_color(x, y);
                    let color2 = config.detector_color_at(idx);
                    assert_eq!(color1, color2);
                }
            }
        }
    }

    #[test]
    fn test_color_count_distribution() {
        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(6);
        let counts = config.count_by_color();

        // Total should equal num_detectors
        let total: usize = counts.iter().sum();
        assert_eq!(total, config.num_detectors());

        // Distribution should be roughly equal (within 1 per row)
        let max_diff = counts.iter().max().unwrap() - counts.iter().min().unwrap();
        let max_allowed_diff = config.height * config.depth; // At most 1 diff per row per time
        assert!(max_diff <= max_allowed_diff);
    }

    #[test]
    fn test_stride_power_of_two() {
        for d in 3..=15 {
            let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(d);
            assert!(config.stride_y.is_power_of_two());
            assert!(config.stride_y >= config.width);
            assert!(config.stride_y >= config.height);
            assert!(config.stride_y >= config.depth);
        }
    }

    #[test]
    fn test_with_depth() {
        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(5);
        let config_2d = config.with_depth(1);

        assert_eq!(config_2d.width, config.width);
        assert_eq!(config_2d.height, config.height);
        assert_eq!(config_2d.depth, 1);
        assert_eq!(config_2d.num_rounds, 1);
    }

    #[test]
    fn test_predefined_configs() {
        // Verify predefined configs are valid
        let configs = [
            TestColorCodeGrids::D3,
            TestColorCodeGrids::D5,
            TestColorCodeGrids::D7,
            TestColorCodeGrids::D9,
            TestColorCodeGrids::D11,
            TestColorCodeGrids::D13,
        ];

        for config in configs {
            assert!(config.stride_y.is_power_of_two());
            assert!(config.num_detectors() > 0);
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    #[kani::unwind(10)]
    fn verify_coord_roundtrip() {
        let d: usize = kani::any();
        kani::assume(d >= 3 && d <= 10);

        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(d);

        let x: usize = kani::any();
        let y: usize = kani::any();
        let t: usize = kani::any();
        kani::assume(x < config.width && y < config.height && t < config.depth);

        let idx = config.coord_to_linear(x, y, t);
        let (x2, y2, t2) = config.linear_to_coord(idx);

        assert!(x == x2);
        assert!(y == y2);
        assert!(t == t2);
    }

    #[kani::proof]
    #[kani::unwind(5)]
    fn verify_stride_power_of_two() {
        let d: usize = kani::any();
        kani::assume(d >= 3 && d <= 20);

        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(d);
        assert!(config.stride_y.is_power_of_two());
    }

    #[kani::proof]
    #[kani::unwind(5)]
    fn verify_stride_covers_dimensions() {
        let d: usize = kani::any();
        kani::assume(d >= 3 && d <= 20);

        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(d);
        assert!(config.stride_y >= config.width);
        assert!(config.stride_y >= config.height);
        assert!(config.stride_y >= config.depth);
    }

    #[kani::proof]
    #[kani::unwind(5)]
    fn verify_linear_index_bounds() {
        let d: usize = kani::any();
        kani::assume(d >= 3 && d <= 10);

        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(d);

        let x: usize = kani::any();
        let y: usize = kani::any();
        let t: usize = kani::any();
        kani::assume(x < config.width && y < config.height && t < config.depth);

        let idx = config.coord_to_linear(x, y, t);
        assert!(idx < config.alloc_nodes());
    }
}
