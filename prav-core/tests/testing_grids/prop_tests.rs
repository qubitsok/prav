//! Property-based tests for testing_grids module.
//!
//! These tests verify mathematical invariants of grid configuration functions
//! using randomized inputs.

use proptest::prelude::*;
use prav_core::testing_grids::{isqrt, GridConfig};

proptest! {
    /// Verify isqrt maintains the integer square root invariant:
    /// For all n: isqrt(n)^2 <= n < (isqrt(n)+1)^2
    #[test]
    fn prop_isqrt_bounds(n in 0usize..1_000_000) {
        let s = isqrt(n);

        // isqrt(n)^2 must be <= n
        prop_assert!(
            s * s <= n,
            "isqrt({})^2 = {} should be <= {}", n, s * s, n
        );

        // (isqrt(n)+1)^2 must be > n
        prop_assert!(
            (s + 1) * (s + 1) > n,
            "(isqrt({})+1)^2 = {} should be > {}", n, (s + 1) * (s + 1), n
        );
    }

    /// Verify from_target_nodes produces reasonable approximations.
    /// The actual node count should be within 2x of the target.
    #[test]
    fn prop_from_target_nodes_reasonable(target in 4usize..100_000) {
        let config = GridConfig::from_target_nodes(target);
        let actual = config.actual_nodes();

        // Should be within 2x of target (sqrt approximation)
        prop_assert!(
            actual >= target / 2,
            "actual_nodes {} should be >= target/2 = {}", actual, target / 2
        );
        prop_assert!(
            actual <= target * 2,
            "actual_nodes {} should be <= target*2 = {}", actual, target * 2
        );
    }

    /// Verify stride_y is always a power of two.
    /// This is critical for Morton encoding correctness.
    #[test]
    fn prop_stride_is_power_of_two(target in 4usize..100_000) {
        let config = GridConfig::from_target_nodes(target);
        prop_assert!(
            config.stride_y.is_power_of_two(),
            "stride_y {} must be power of two", config.stride_y
        );
    }

    /// Verify stride_y is large enough to cover the grid dimensions.
    #[test]
    fn prop_stride_covers_dimensions(target in 4usize..100_000) {
        let config = GridConfig::from_target_nodes(target);

        prop_assert!(
            config.stride_y >= config.width,
            "stride_y {} must be >= width {}", config.stride_y, config.width
        );
        prop_assert!(
            config.stride_y >= config.height,
            "stride_y {} must be >= height {}", config.stride_y, config.height
        );
    }

    /// Verify to_rectangular preserves approximate node count.
    /// The actual nodes should be within 3x of target for various aspect ratios.
    #[test]
    fn prop_to_rectangular_preserves_approximate_nodes(
        target in 100usize..10_000,
        ratio in 0.5f64..2.0
    ) {
        let config = GridConfig::from_target_nodes(target);
        let rect = config.to_rectangular(ratio);
        let actual = rect.width * rect.height;

        // Should be within 3x of target
        prop_assert!(
            actual >= target / 3,
            "rectangular actual {} should be >= target/3 = {} (ratio={})",
            actual, target / 3, ratio
        );
        prop_assert!(
            actual <= target * 3,
            "rectangular actual {} should be <= target*3 = {} (ratio={})",
            actual, target * 3, ratio
        );
    }

    /// Verify to_rectangular maintains power-of-two stride.
    #[test]
    fn prop_to_rectangular_stride_power_of_two(
        target in 100usize..10_000,
        ratio in 0.5f64..2.0
    ) {
        let config = GridConfig::from_target_nodes(target);
        let rect = config.to_rectangular(ratio);

        prop_assert!(
            rect.stride_y.is_power_of_two(),
            "rectangular stride_y {} must be power of two", rect.stride_y
        );
    }

    /// Verify to_rectangular produces positive dimensions.
    #[test]
    fn prop_to_rectangular_positive_dimensions(
        target in 1usize..10_000,
        ratio in 0.1f64..10.0
    ) {
        let config = GridConfig::from_target_nodes(target);
        let rect = config.to_rectangular(ratio);

        prop_assert!(rect.width >= 1, "width must be >= 1");
        prop_assert!(rect.height >= 1, "height must be >= 1");
        prop_assert!(rect.stride_y >= 1, "stride_y must be >= 1");
    }
}
