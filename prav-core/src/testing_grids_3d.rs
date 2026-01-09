//! 3D grid configurations for circuit-level QEC benchmarking.
//!
//! This module provides configurations for 3D space-time decoding grids
//! used in circuit-level noise simulation. The third dimension represents
//! measurement rounds (time).
//!
//! # Surface Code Detector Layout
//!
//! For a rotated surface code of distance `d`:
//! - Spatial dimensions: `(d-1) x (d-1)` detectors per stabilizer type
//! - Time dimension: `d` measurement rounds
//! - Total detectors per type: `(d-1)^2 * d`
//!
//! # Usage
//!
//! ```ignore
//! use prav_core::testing_grids_3d::{Grid3DConfig, TestGrids3D, SurfaceCodeType};
//!
//! // Get config for distance-5 rotated surface code
//! let config = TestGrids3D::D5;
//! assert_eq!(config.code_distance, 5);
//! assert_eq!(config.num_rounds, 5);
//! ```

use crate::decoder::types::BoundaryConfig;

/// Type of surface code layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceCodeType {
    /// Rotated surface code (most common in practice).
    /// Detector grid is (d-1) x (d-1) per stabilizer type.
    RotatedSurface,
    /// Unrotated (CSS) planar surface code.
    /// Detector grid is d x (d-1) per stabilizer type.
    UnrotatedPlanar,
}

/// Configuration for 3D circuit-level testing grids.
///
/// Represents a 3D decoding volume where:
/// - X, Y are spatial dimensions (detector positions)
/// - Z is the time dimension (measurement rounds)
#[derive(Debug, Clone, Copy)]
pub struct Grid3DConfig {
    /// Surface code distance.
    pub code_distance: usize,
    /// Number of measurement rounds (typically equals code_distance).
    pub num_rounds: usize,
    /// Grid width in detectors.
    pub width: usize,
    /// Grid height in detectors.
    pub height: usize,
    /// Grid depth (time slices).
    pub depth: usize,
    /// Y stride (next power of two from max(width, height, depth)).
    pub stride_y: usize,
    /// Z stride (stride_y^2).
    pub stride_z: usize,
    /// Surface code type.
    pub code_type: SurfaceCodeType,
    /// Boundary configuration for matching.
    /// Defaults to `All` for unified X+Z decoding.
    pub boundary_config: BoundaryConfig,
}

impl Grid3DConfig {
    /// Create a 3D config for a rotated surface code of given distance.
    ///
    /// Uses `d` measurement rounds (standard for surface codes).
    #[must_use]
    pub const fn for_rotated_surface(d: usize) -> Self {
        // Rotated surface code: (d-1) x (d-1) detectors per stabilizer type
        let detector_dim = if d > 1 { d - 1 } else { 1 };
        let num_rounds = d;

        // Stride must be power of 2 and >= all dimensions
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
            code_type: SurfaceCodeType::RotatedSurface,
            boundary_config: BoundaryConfig::all_boundaries(),
        }
    }

    /// Create a 3D config for an unrotated planar surface code of given distance.
    #[must_use]
    pub const fn for_unrotated_planar(d: usize) -> Self {
        // Unrotated surface code: d x (d-1) detectors per stabilizer type
        let width = d;
        let height = if d > 1 { d - 1 } else { 1 };
        let num_rounds = d;

        let max_dim = max3(width, height, num_rounds);
        let stride_y = next_power_of_two(max_dim);
        let stride_z = stride_y * stride_y;

        Self {
            code_distance: d,
            num_rounds,
            width,
            height,
            depth: num_rounds,
            stride_y,
            stride_z,
            code_type: SurfaceCodeType::UnrotatedPlanar,
            boundary_config: BoundaryConfig::all_boundaries(),
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
            code_type: SurfaceCodeType::RotatedSurface,
            boundary_config: BoundaryConfig::all_boundaries(),
        }
    }

    /// Create a 3D config for X-type stabilizer decoding (compact reindexed).
    ///
    /// For a rotated surface code of distance `d`:
    /// - X stabilizers are at positions where `(x + y) % 2 == 0`
    /// - Grid is compacted to `(d-1)/2 × (d-1) × depth`
    /// - Only top/bottom boundaries are active (for X logical observable)
    ///
    /// # Arguments
    /// * `d` - Code distance
    /// * `depth` - Number of measurement rounds (use 1 for 2D mode)
    #[must_use]
    pub const fn for_x_stabilizers(d: usize, depth: usize) -> Self {
        let grid_size = if d > 1 { d - 1 } else { 1 };
        // Compact width: half the columns (X detectors per row)
        let compact_width = if grid_size > 1 { grid_size / 2 } else { 1 };
        let compact_height = grid_size;

        let max_dim = max3(compact_width, compact_height, depth);
        let stride_y = next_power_of_two(max_dim);
        let stride_z = stride_y * stride_y;

        Self {
            code_distance: d,
            num_rounds: depth,
            width: compact_width,
            height: compact_height,
            depth,
            stride_y,
            stride_z,
            code_type: SurfaceCodeType::RotatedSurface,
            boundary_config: BoundaryConfig::horizontal_only(),
        }
    }

    /// Create a 3D config for Z-type stabilizer decoding (compact reindexed).
    ///
    /// For a rotated surface code of distance `d`:
    /// - Z stabilizers are at positions where `(x + y) % 2 == 1`
    /// - Grid is compacted to `(d-1)/2 × (d-1) × depth`
    /// - Coordinates are rotated so left/right boundaries become top/bottom
    /// - Uses horizontal boundary config after rotation
    ///
    /// # Arguments
    /// * `d` - Code distance
    /// * `depth` - Number of measurement rounds (use 1 for 2D mode)
    #[must_use]
    pub const fn for_z_stabilizers(d: usize, depth: usize) -> Self {
        let grid_size = if d > 1 { d - 1 } else { 1 };
        // Compact width: half the columns (Z detectors per row)
        // After coordinate rotation, the grid dimensions are swapped
        let compact_width = if grid_size > 1 { grid_size / 2 } else { 1 };
        let compact_height = grid_size;

        let max_dim = max3(compact_width, compact_height, depth);
        let stride_y = next_power_of_two(max_dim);
        let stride_z = stride_y * stride_y;

        Self {
            code_distance: d,
            num_rounds: depth,
            width: compact_width,
            height: compact_height,
            depth,
            stride_y,
            stride_z,
            code_type: SurfaceCodeType::RotatedSurface,
            // After coordinate rotation, left/right becomes top/bottom
            boundary_config: BoundaryConfig::horizontal_only(),
        }
    }

    /// Returns a copy with a different boundary configuration.
    #[must_use]
    pub const fn with_boundary_config(self, boundary_config: BoundaryConfig) -> Self {
        Self {
            boundary_config,
            ..self
        }
    }

    /// Returns a copy with a different depth (for 2D mode).
    #[must_use]
    pub const fn with_depth(self, depth: usize) -> Self {
        let max_dim = max3(self.width, self.height, depth);
        let stride_y = next_power_of_two(max_dim);
        let stride_z = stride_y * stride_y;

        Self {
            depth,
            num_rounds: depth,
            stride_y,
            stride_z,
            ..self
        }
    }

    /// Total number of detector nodes in the 3D grid.
    #[must_use]
    pub const fn num_detectors(&self) -> usize {
        self.width * self.height * self.depth
    }

    /// Check if a position is on the boundary according to boundary_config.
    ///
    /// For split X/Z decoding, this determines which edges are valid
    /// matching targets for clusters.
    #[must_use]
    #[inline(always)]
    pub const fn is_boundary(&self, x: usize, y: usize) -> bool {
        self.boundary_config
            .is_boundary(x, y, self.width, self.height)
    }

    /// Number of detectors per measurement round in this configuration.
    #[must_use]
    pub const fn detectors_per_round(&self) -> usize {
        self.width * self.height
    }

    /// Total allocated nodes (including padding for Morton alignment).
    #[must_use]
    pub const fn alloc_nodes(&self) -> usize {
        self.stride_z * self.depth
    }

    /// Convert (x, y, t) coordinates to linear index.
    ///
    /// Uses stride-based layout compatible with prav-core's DecodingState.
    #[must_use]
    pub const fn coord_to_linear(&self, x: usize, y: usize, t: usize) -> usize {
        t * self.stride_z + y * self.stride_y + x
    }

    /// Convert linear index back to (x, y, t) coordinates.
    #[must_use]
    pub const fn linear_to_coord(&self, idx: usize) -> (usize, usize, usize) {
        let t = idx / self.stride_z;
        let rem = idx % self.stride_z;
        let y = rem / self.stride_y;
        let x = rem % self.stride_y;
        (x, y, t)
    }
}

/// Common error probabilities for circuit-level benchmarking.
///
/// Range spans from well below threshold (~0.1%) to above threshold (~1%).
pub const CIRCUIT_ERROR_PROBS: [f64; 5] = [0.001, 0.003, 0.005, 0.007, 0.01];

/// Predefined 3D grid configurations for surface code testing.
pub struct TestGrids3D;

impl TestGrids3D {
    /// Distance-3 rotated surface code.
    /// 2x2 detectors per round, 3 rounds = 12 detectors total.
    pub const D3: Grid3DConfig = Grid3DConfig::for_rotated_surface(3);

    /// Distance-5 rotated surface code.
    /// 4x4 detectors per round, 5 rounds = 80 detectors total.
    pub const D5: Grid3DConfig = Grid3DConfig::for_rotated_surface(5);

    /// Distance-7 rotated surface code.
    /// 6x6 detectors per round, 7 rounds = 252 detectors total.
    pub const D7: Grid3DConfig = Grid3DConfig::for_rotated_surface(7);

    /// Distance-9 rotated surface code.
    /// 8x8 detectors per round, 9 rounds = 576 detectors total.
    pub const D9: Grid3DConfig = Grid3DConfig::for_rotated_surface(9);

    /// Distance-11 rotated surface code.
    /// 10x10 detectors per round, 11 rounds = 1100 detectors total.
    pub const D11: Grid3DConfig = Grid3DConfig::for_rotated_surface(11);

    /// Distance-13 rotated surface code.
    /// 12x12 detectors per round, 13 rounds = 1872 detectors total.
    /// Key comparison point for Helios paper benchmarks.
    pub const D13: Grid3DConfig = Grid3DConfig::for_rotated_surface(13);

    /// Distance-15 rotated surface code.
    /// 14x14 detectors per round, 15 rounds = 2940 detectors total.
    pub const D15: Grid3DConfig = Grid3DConfig::for_rotated_surface(15);

    /// Distance-17 rotated surface code.
    /// 16x16 detectors per round, 17 rounds = 4352 detectors total.
    pub const D17: Grid3DConfig = Grid3DConfig::for_rotated_surface(17);

    /// Distance-21 rotated surface code.
    /// 20x20 detectors per round, 21 rounds = 8400 detectors total.
    pub const D21: Grid3DConfig = Grid3DConfig::for_rotated_surface(21);

    /// Returns all predefined rotated surface code configurations.
    pub const fn all_rotated() -> [Grid3DConfig; 9] {
        [
            Self::D3,
            Self::D5,
            Self::D7,
            Self::D9,
            Self::D11,
            Self::D13,
            Self::D15,
            Self::D17,
            Self::D21,
        ]
    }

    /// Returns default configurations for quick testing (D3, D5, D7).
    pub const fn defaults() -> [Grid3DConfig; 3] {
        [Self::D3, Self::D5, Self::D7]
    }

    /// Distance-3 unrotated planar surface code.
    pub const D3_UNROTATED: Grid3DConfig = Grid3DConfig::for_unrotated_planar(3);

    /// Distance-5 unrotated planar surface code.
    pub const D5_UNROTATED: Grid3DConfig = Grid3DConfig::for_unrotated_planar(5);

    /// Distance-7 unrotated planar surface code.
    pub const D7_UNROTATED: Grid3DConfig = Grid3DConfig::for_unrotated_planar(7);

    /// Returns unrotated surface code configurations.
    pub const fn all_unrotated() -> [Grid3DConfig; 3] {
        [Self::D3_UNROTATED, Self::D5_UNROTATED, Self::D7_UNROTATED]
    }
}

/// Const-compatible max of three values.
const fn max3(a: usize, b: usize, c: usize) -> usize {
    let ab = if a > b { a } else { b };
    if ab > c { ab } else { c }
}

/// Const-compatible next power of two.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotated_surface_d3() {
        let config = Grid3DConfig::for_rotated_surface(3);
        assert_eq!(config.code_distance, 3);
        assert_eq!(config.width, 2);
        assert_eq!(config.height, 2);
        assert_eq!(config.depth, 3);
        assert_eq!(config.num_detectors(), 2 * 2 * 3);
        assert!(config.stride_y.is_power_of_two());
        assert_eq!(config.stride_z, config.stride_y * config.stride_y);
    }

    #[test]
    fn test_rotated_surface_d5() {
        let config = Grid3DConfig::for_rotated_surface(5);
        assert_eq!(config.code_distance, 5);
        assert_eq!(config.width, 4);
        assert_eq!(config.height, 4);
        assert_eq!(config.depth, 5);
        assert_eq!(config.num_detectors(), 4 * 4 * 5);
    }

    #[test]
    fn test_unrotated_planar_d5() {
        let config = Grid3DConfig::for_unrotated_planar(5);
        assert_eq!(config.width, 5);
        assert_eq!(config.height, 4);
        assert_eq!(config.depth, 5);
        assert_eq!(config.code_type, SurfaceCodeType::UnrotatedPlanar);
    }

    #[test]
    fn test_coord_conversion_roundtrip() {
        let config = TestGrids3D::D7;
        for x in 0..config.width {
            for y in 0..config.height {
                for t in 0..config.depth {
                    let linear = config.coord_to_linear(x, y, t);
                    let (x2, y2, t2) = config.linear_to_coord(linear);
                    assert_eq!((x, y, t), (x2, y2, t2));
                }
            }
        }
    }

    #[test]
    fn test_predefined_configs() {
        for config in TestGrids3D::all_rotated() {
            assert!(config.stride_y.is_power_of_two());
            assert!(config.stride_y >= config.width);
            assert!(config.stride_y >= config.height);
            assert!(config.stride_y >= config.depth);
            assert_eq!(config.stride_z, config.stride_y * config.stride_y);
            assert_eq!(config.code_type, SurfaceCodeType::RotatedSurface);
        }
    }

    #[test]
    fn test_custom_config() {
        let config = Grid3DConfig::custom(10, 10, 8);
        assert_eq!(config.width, 10);
        assert_eq!(config.height, 10);
        assert_eq!(config.depth, 8);
        assert!(config.stride_y >= 10);
        assert!(config.stride_y.is_power_of_two());
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(0), 1);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(2), 2);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(16), 16);
        assert_eq!(next_power_of_two(17), 32);
    }

    #[test]
    fn test_all_rotated() {
        let configs = TestGrids3D::all_rotated();
        assert_eq!(configs.len(), 9);
        assert_eq!(configs[0].code_distance, 3);
        assert_eq!(configs[5].code_distance, 13); // Helios benchmark point
        assert_eq!(configs[8].code_distance, 21);
    }

    #[test]
    fn test_d13_helios_benchmark() {
        let config = TestGrids3D::D13;
        assert_eq!(config.code_distance, 13);
        assert_eq!(config.width, 12);
        assert_eq!(config.height, 12);
        assert_eq!(config.depth, 13);
        assert_eq!(config.num_rounds, 13);
        // 12x12x13 = 1872 detectors
        assert_eq!(config.num_detectors(), 1872);
    }

    #[test]
    fn test_defaults() {
        let defaults = TestGrids3D::defaults();
        assert_eq!(defaults.len(), 3);
        assert_eq!(defaults[0].code_distance, 3);
        assert_eq!(defaults[2].code_distance, 7);
    }

    #[test]
    fn test_x_stabilizer_config_d5() {
        let config = Grid3DConfig::for_x_stabilizers(5, 5);
        assert_eq!(config.code_distance, 5);
        // d=5: grid_size=4, compact_width=2, height=4
        assert_eq!(config.width, 2);
        assert_eq!(config.height, 4);
        assert_eq!(config.depth, 5);
        // X uses horizontal boundaries (top/bottom only)
        assert!(config.boundary_config.check_top);
        assert!(config.boundary_config.check_bottom);
        assert!(!config.boundary_config.check_left);
        assert!(!config.boundary_config.check_right);
        // 2x4x5 = 40 detectors (half of original 4x4x5=80)
        assert_eq!(config.num_detectors(), 40);
    }

    #[test]
    fn test_z_stabilizer_config_d5() {
        let config = Grid3DConfig::for_z_stabilizers(5, 5);
        assert_eq!(config.code_distance, 5);
        // Same dimensions as X (symmetric for even grid_size)
        assert_eq!(config.width, 2);
        assert_eq!(config.height, 4);
        assert_eq!(config.depth, 5);
        // Z uses horizontal after rotation (left/right -> top/bottom)
        assert!(config.boundary_config.check_top);
        assert!(config.boundary_config.check_bottom);
        assert_eq!(config.num_detectors(), 40);
    }

    #[test]
    fn test_x_stabilizer_2d_mode() {
        // 2D mode: depth=1
        let config = Grid3DConfig::for_x_stabilizers(5, 1);
        assert_eq!(config.depth, 1);
        assert_eq!(config.num_rounds, 1);
        // 2x4x1 = 8 detectors
        assert_eq!(config.num_detectors(), 8);
    }

    #[test]
    fn test_boundary_config_all() {
        let config = Grid3DConfig::for_rotated_surface(5);
        // All boundaries enabled
        assert!(config.boundary_config.check_top);
        assert!(config.boundary_config.check_bottom);
        assert!(config.boundary_config.check_left);
        assert!(config.boundary_config.check_right);

        // 4x4 grid: corners and edges are boundaries
        assert!(config.is_boundary(0, 0)); // corner
        assert!(config.is_boundary(3, 0)); // bottom-right
        assert!(config.is_boundary(0, 3)); // top-left
        assert!(config.is_boundary(3, 3)); // top-right
        assert!(config.is_boundary(1, 0)); // bottom edge
        assert!(config.is_boundary(0, 2)); // left edge
        assert!(!config.is_boundary(1, 1)); // interior
        assert!(!config.is_boundary(2, 2)); // interior
    }

    #[test]
    fn test_boundary_config_horizontal() {
        let config = Grid3DConfig::for_x_stabilizers(5, 5);
        // Horizontal boundaries only
        assert!(config.boundary_config.check_top);
        assert!(config.boundary_config.check_bottom);
        assert!(!config.boundary_config.check_left);
        assert!(!config.boundary_config.check_right);

        // 2x4 compact grid: only y=0 and y=3 are boundaries
        assert!(config.is_boundary(0, 0)); // bottom edge
        assert!(config.is_boundary(1, 0)); // bottom edge
        assert!(config.is_boundary(0, 3)); // top edge
        assert!(config.is_boundary(1, 3)); // top edge
        assert!(!config.is_boundary(0, 1)); // left edge but NOT boundary for Horizontal
        assert!(!config.is_boundary(0, 2)); // left edge but NOT boundary
        assert!(!config.is_boundary(1, 1)); // interior
    }

    #[test]
    fn test_boundary_config_vertical() {
        let config =
            Grid3DConfig::custom(4, 4, 1).with_boundary_config(BoundaryConfig::vertical_only());

        // Only x=0 and x=3 are boundaries
        assert!(config.is_boundary(0, 0)); // left edge
        assert!(config.is_boundary(0, 2)); // left edge
        assert!(config.is_boundary(3, 1)); // right edge
        assert!(config.is_boundary(3, 3)); // right edge
        assert!(!config.is_boundary(1, 0)); // bottom edge but NOT boundary for Vertical
        assert!(!config.is_boundary(2, 3)); // top edge but NOT boundary
        assert!(!config.is_boundary(1, 1)); // interior
    }

    #[test]
    fn test_with_depth() {
        let config_3d = Grid3DConfig::for_rotated_surface(5);
        assert_eq!(config_3d.depth, 5);

        let config_2d = config_3d.with_depth(1);
        assert_eq!(config_2d.depth, 1);
        assert_eq!(config_2d.num_rounds, 1);
        assert_eq!(config_2d.width, 4); // spatial dims unchanged
        assert_eq!(config_2d.height, 4);
        assert!(config_2d.stride_y.is_power_of_two());
    }

    #[test]
    fn test_x_z_stabilizer_detector_counts() {
        // For each distance, X + Z detectors should sum to full grid
        for d in [3, 5, 7, 9, 11] {
            let full = Grid3DConfig::for_rotated_surface(d);
            let x_config = Grid3DConfig::for_x_stabilizers(d, d);
            let z_config = Grid3DConfig::for_z_stabilizers(d, d);

            // For even grid_size, X and Z have exactly equal counts
            let grid_size = d - 1;
            if grid_size % 2 == 0 {
                assert_eq!(x_config.num_detectors(), full.num_detectors() / 2);
                assert_eq!(z_config.num_detectors(), full.num_detectors() / 2);
            }
        }
    }
}

// ============================================================================
// Kani Formal Verification Proofs
// ============================================================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify stride is always a power of two for rotated surface codes.
    #[kani::proof]
    fn verify_rotated_stride_power_of_two() {
        let d: usize = kani::any();
        kani::assume(d >= 3 && d <= 100);

        let config = Grid3DConfig::for_rotated_surface(d);

        kani::assert(
            config.stride_y.is_power_of_two(),
            "stride_y must be power of two",
        );
    }

    /// Verify stride covers all dimensions for rotated surface codes.
    #[kani::proof]
    fn verify_rotated_stride_covers_dimensions() {
        let d: usize = kani::any();
        kani::assume(d >= 3 && d <= 100);

        let config = Grid3DConfig::for_rotated_surface(d);

        kani::assert(config.stride_y >= config.width, "stride_y >= width");
        kani::assert(config.stride_y >= config.height, "stride_y >= height");
        kani::assert(config.stride_y >= config.depth, "stride_y >= depth");
    }

    /// Verify coordinate conversion is bijective.
    #[kani::proof]
    fn verify_coord_roundtrip() {
        let d: usize = kani::any();
        kani::assume(d >= 3 && d <= 20);

        let config = Grid3DConfig::for_rotated_surface(d);

        let x: usize = kani::any();
        let y: usize = kani::any();
        let t: usize = kani::any();
        kani::assume(x < config.width);
        kani::assume(y < config.height);
        kani::assume(t < config.depth);

        let linear = config.coord_to_linear(x, y, t);
        let (x2, y2, t2) = config.linear_to_coord(linear);

        kani::assert(x == x2, "x roundtrip");
        kani::assert(y == y2, "y roundtrip");
        kani::assert(t == t2, "t roundtrip");
    }

    /// Verify linear index is within allocation bounds.
    #[kani::proof]
    fn verify_linear_index_bounds() {
        let d: usize = kani::any();
        kani::assume(d >= 3 && d <= 50);

        let config = Grid3DConfig::for_rotated_surface(d);

        let x: usize = kani::any();
        let y: usize = kani::any();
        let t: usize = kani::any();
        kani::assume(x < config.width);
        kani::assume(y < config.height);
        kani::assume(t < config.depth);

        let linear = config.coord_to_linear(x, y, t);

        kani::assert(
            linear < config.alloc_nodes(),
            "linear index must be within allocation",
        );
    }

    /// Verify num_detectors matches width * height * depth.
    #[kani::proof]
    fn verify_num_detectors() {
        let d: usize = kani::any();
        kani::assume(d >= 3 && d <= 100);

        let config = Grid3DConfig::for_rotated_surface(d);

        kani::assert(
            config.num_detectors() == config.width * config.height * config.depth,
            "num_detectors must equal w*h*d",
        );
    }
}
