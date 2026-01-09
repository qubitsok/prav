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
        }
    }

    /// Total number of detector nodes in the 3D grid.
    #[must_use]
    pub const fn num_detectors(&self) -> usize {
        self.width * self.height * self.depth
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

    /// Distance-11 rotated surface code.
    /// 10x10 detectors per round, 11 rounds = 1100 detectors total.
    pub const D11: Grid3DConfig = Grid3DConfig::for_rotated_surface(11);

    /// Distance-17 rotated surface code.
    /// 16x16 detectors per round, 17 rounds = 4352 detectors total.
    pub const D17: Grid3DConfig = Grid3DConfig::for_rotated_surface(17);

    /// Distance-21 rotated surface code.
    /// 20x20 detectors per round, 21 rounds = 8400 detectors total.
    pub const D21: Grid3DConfig = Grid3DConfig::for_rotated_surface(21);

    /// Returns all predefined rotated surface code configurations.
    pub const fn all_rotated() -> [Grid3DConfig; 6] {
        [
            Self::D3,
            Self::D5,
            Self::D7,
            Self::D11,
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
        assert_eq!(configs.len(), 6);
        assert_eq!(configs[0].code_distance, 3);
        assert_eq!(configs[5].code_distance, 21);
    }

    #[test]
    fn test_defaults() {
        let defaults = TestGrids3D::defaults();
        assert_eq!(defaults.len(), 3);
        assert_eq!(defaults[0].code_distance, 3);
        assert_eq!(defaults[2].code_distance, 7);
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
