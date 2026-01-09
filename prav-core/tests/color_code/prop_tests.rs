//! Property-based tests for color code module.
//!
//! These tests verify mathematical invariants of color code structures
//! using randomized inputs.

use prav_core::color_code::{ColorCodeGrid3DConfig, FaceColor};
use proptest::prelude::*;

proptest! {
    /// Verify the 3-coloring property: adjacent faces have different colors.
    /// For any (x, y), all cardinal neighbors must have different colors.
    #[test]
    fn prop_valid_3_coloring(x in 1u32..1000, y in 1u32..1000) {
        let color = FaceColor::from_coords(x, y);

        // All cardinal neighbors must have different colors
        prop_assert_ne!(
            color,
            FaceColor::from_coords(x - 1, y),
            "Left neighbor at ({}, {}) has same color as ({}, {})",
            x - 1, y, x, y
        );
        prop_assert_ne!(
            color,
            FaceColor::from_coords(x + 1, y),
            "Right neighbor at ({}, {}) has same color as ({}, {})",
            x + 1, y, x, y
        );
        prop_assert_ne!(
            color,
            FaceColor::from_coords(x, y - 1),
            "Top neighbor at ({}, {}) has same color as ({}, {})",
            x, y - 1, x, y
        );
        prop_assert_ne!(
            color,
            FaceColor::from_coords(x, y + 1),
            "Bottom neighbor at ({}, {}) has same color as ({}, {})",
            x, y + 1, x, y
        );
    }

    /// Verify color assignment is deterministic.
    #[test]
    fn prop_color_deterministic(x in 0u32..10000, y in 0u32..10000) {
        let c1 = FaceColor::from_coords(x, y);
        let c2 = FaceColor::from_coords(x, y);
        prop_assert_eq!(c1, c2);
    }

    /// Verify color index is always in range [0, 2].
    #[test]
    fn prop_color_index_in_range(x in 0u32..10000, y in 0u32..10000) {
        let color = FaceColor::from_coords(x, y);
        let idx = color.index();
        prop_assert!(idx < 3, "Color index {} must be < 3", idx);
    }

    /// Verify stride is always a power of two for any distance.
    #[test]
    fn prop_grid_stride_power_of_two(d in 3usize..50) {
        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(d);
        prop_assert!(
            config.stride_y.is_power_of_two(),
            "stride_y {} must be power of two for distance {}",
            config.stride_y, d
        );
    }

    /// Verify stride covers all dimensions.
    #[test]
    fn prop_grid_stride_covers_dimensions(d in 3usize..50) {
        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(d);
        prop_assert!(
            config.stride_y >= config.width,
            "stride_y {} must be >= width {} for distance {}",
            config.stride_y, config.width, d
        );
        prop_assert!(
            config.stride_y >= config.height,
            "stride_y {} must be >= height {} for distance {}",
            config.stride_y, config.height, d
        );
        prop_assert!(
            config.stride_y >= config.depth,
            "stride_y {} must be >= depth {} for distance {}",
            config.stride_y, config.depth, d
        );
    }

    /// Verify coordinate conversion roundtrip.
    #[test]
    fn prop_coord_roundtrip(
        d in 3usize..15,
        x in 0usize..10,
        y in 0usize..10,
        t in 0usize..10
    ) {
        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(d);

        // Only test valid coordinates
        prop_assume!(x < config.width);
        prop_assume!(y < config.height);
        prop_assume!(t < config.depth);

        let idx = config.coord_to_linear(x, y, t);
        let (x2, y2, t2) = config.linear_to_coord(idx);

        prop_assert_eq!(
            (x, y, t), (x2, y2, t2),
            "Roundtrip failed for d={}: ({},{},{}) -> {} -> ({},{},{})",
            d, x, y, t, idx, x2, y2, t2
        );
    }

    /// Verify linear index is within allocated bounds.
    #[test]
    fn prop_linear_index_bounds(
        d in 3usize..15,
        x in 0usize..10,
        y in 0usize..10,
        t in 0usize..10
    ) {
        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(d);

        prop_assume!(x < config.width);
        prop_assume!(y < config.height);
        prop_assume!(t < config.depth);

        let idx = config.coord_to_linear(x, y, t);
        prop_assert!(
            idx < config.alloc_nodes(),
            "Index {} must be < alloc_nodes {} for d={}",
            idx, config.alloc_nodes(), d
        );
    }

    /// Verify color count sums to total detectors.
    #[test]
    fn prop_color_count_sum(d in 3usize..20) {
        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(d);
        let counts = config.count_by_color();
        let total: usize = counts.iter().sum();

        prop_assert_eq!(
            total, config.num_detectors(),
            "Color counts {:?} sum to {} but num_detectors is {}",
            counts, total, config.num_detectors()
        );
    }

    /// Verify color distribution is roughly equal.
    #[test]
    fn prop_color_distribution_balanced(d in 3usize..20) {
        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(d);
        let counts = config.count_by_color();

        let max_count = *counts.iter().max().unwrap();
        let min_count = *counts.iter().min().unwrap();

        // Allow some imbalance due to grid boundaries
        // The difference should be at most O(d^2) for boundary effects
        let max_allowed_diff = config.width * config.depth + config.height * config.depth;

        prop_assert!(
            max_count - min_count <= max_allowed_diff,
            "Color distribution imbalanced: max={}, min={}, diff={}, allowed={}",
            max_count, min_count, max_count - min_count, max_allowed_diff
        );
    }

    /// Verify 2D mode conversion preserves spatial dimensions.
    #[test]
    fn prop_2d_mode_preserves_spatial(d in 3usize..20) {
        let config_3d = ColorCodeGrid3DConfig::for_triangular_6_6_6(d);
        let config_2d = config_3d.with_depth(1);

        prop_assert_eq!(config_2d.width, config_3d.width);
        prop_assert_eq!(config_2d.height, config_3d.height);
        prop_assert_eq!(config_2d.depth, 1);
        prop_assert_eq!(config_2d.num_rounds, 1);
    }

    /// Verify color cyclic properties.
    #[test]
    fn prop_color_cyclic(x in 0u32..1000, y in 0u32..1000) {
        let color = FaceColor::from_coords(x, y);

        // next(next(next(c))) == c
        prop_assert_eq!(
            color.next().next().next(),
            color,
            "Triple next should return original color"
        );

        // prev(prev(prev(c))) == c
        prop_assert_eq!(
            color.prev().prev().prev(),
            color,
            "Triple prev should return original color"
        );

        // next(prev(c)) == c
        prop_assert_eq!(
            color.prev().next(),
            color,
            "prev followed by next should return original color"
        );
    }
}

/// Additional targeted tests that don't fit the proptest pattern
#[cfg(test)]
mod targeted_tests {
    use super::*;

    #[test]
    fn test_color_pattern_at_origin() {
        // Verify the specific pattern at the origin region
        // Row y=0: R G B R G B ...
        // Row y=1: G B R G B R ...
        // Row y=2: B R G B R G ...
        assert_eq!(FaceColor::from_coords(0, 0), FaceColor::Red);
        assert_eq!(FaceColor::from_coords(1, 0), FaceColor::Green);
        assert_eq!(FaceColor::from_coords(2, 0), FaceColor::Blue);

        assert_eq!(FaceColor::from_coords(0, 1), FaceColor::Green);
        assert_eq!(FaceColor::from_coords(1, 1), FaceColor::Blue);
        assert_eq!(FaceColor::from_coords(2, 1), FaceColor::Red);

        assert_eq!(FaceColor::from_coords(0, 2), FaceColor::Blue);
        assert_eq!(FaceColor::from_coords(1, 2), FaceColor::Red);
        assert_eq!(FaceColor::from_coords(2, 2), FaceColor::Green);
    }

    #[test]
    fn test_color_diagonal_invariant() {
        // Along diagonals where x + y is constant, color should be constant
        for sum in 0..10u32 {
            let expected_color = FaceColor::from_coords(sum, 0);
            for x in 0..=sum {
                let y = sum - x;
                assert_eq!(
                    FaceColor::from_coords(x, y),
                    expected_color,
                    "Diagonal x+y={} should have consistent color at ({}, {})",
                    sum,
                    x,
                    y
                );
            }
        }
    }
}
