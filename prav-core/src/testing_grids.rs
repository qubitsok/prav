/// Configuration for square grid testing
#[derive(Debug, Clone, Copy)]
pub struct GridConfig {
    /// Target number of nodes (approximately width * height)
    pub target_nodes: usize,
    /// Grid width
    pub width: usize,
    /// Grid height (equals width for square grids)
    pub height: usize,
    /// Stride Y (next power of two from max dimension)
    pub stride_y: usize,
}

impl GridConfig {
    /// Create a new grid configuration from a target node count.
    /// Finds the nearest integer square root to create a square grid.
    pub fn from_target_nodes(target_nodes: usize) -> Self {
        // Integer sqrt to avoid std/libm dependency
        let dim = isqrt(target_nodes);
        let stride_y = dim.next_power_of_two();

        Self {
            target_nodes,
            width: dim,
            height: dim,
            stride_y,
        }
    }

    /// Get actual node count (width * height)
    pub fn actual_nodes(&self) -> usize {
        self.width * self.height
    }

    /// Create a rectangular version of this config with approximately the same number of nodes
    /// and the given aspect ratio (width / height).
    pub fn to_rectangular(&self, aspect_ratio: f64) -> Self {
        let nodes = self.target_nodes;
        
        // h^2 = N / ratio
        let val = (nodes as f64 / aspect_ratio) as usize;
        let h = isqrt(val).max(1);

        // w = N / h approx
        let w = nodes.div_ceil(h);

        // Ensure non-zero
        let h = h.max(1);
        let w = w.max(1);

        let stride_y = w.max(h).next_power_of_two();

        Self {
            target_nodes: nodes,
            width: w,
            height: h,
            stride_y,
        }
    }
}

/// Common error probabilities for benchmarking
pub const ERROR_PROBS: [f64; 6] = [0.001, 0.003, 0.006, 0.01, 0.03, 0.06];

/// Integer square root using Newton's method.
/// Returns the largest integer s such that s*s <= n.
#[must_use]
pub fn isqrt(n: usize) -> usize {
    if n < 2 {
        return n;
    }
    let mut x = n;
    let mut y = x.div_ceil(2);
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

/// Predefined grid configurations for testing
pub struct TestGrids;

impl TestGrids {
    /// Small grid: ~289 nodes (17x17)
    pub const TINY: GridConfig = GridConfig {
        target_nodes: 289,
        width: 17,
        height: 17,
        stride_y: 32,
    };

    /// Small grid: ~500 nodes (22x22)
    pub const SMALL: GridConfig = GridConfig {
        target_nodes: 500,
        width: 22,
        height: 22,
        stride_y: 32,
    };

    /// Medium grid: ~1024 nodes (32x32)
    pub const MEDIUM: GridConfig = GridConfig {
        target_nodes: 1024,
        width: 32,
        height: 32,
        stride_y: 32,
    };

    /// Large grid: ~4096 nodes (64x64)
    pub const LARGE: GridConfig = GridConfig {
        target_nodes: 4096,
        width: 64,
        height: 64,
        stride_y: 64,
    };

    /// Large grid: ~5000 nodes (71x71)
    pub const LARGE_PLUS: GridConfig = GridConfig {
        target_nodes: 5000,
        width: 71,
        height: 71,
        stride_y: 128,
    };

    /// Extra large grid: ~100_000 nodes (316x316)
    pub const XLARGE: GridConfig = GridConfig {
        target_nodes: 100_000,
        width: 316,
        height: 316,
        stride_y: 512,
    };

    /// Extra large grid: ~131_072 nodes (362x362)
    pub const XXLARGE: GridConfig = GridConfig {
        target_nodes: 131_072,
        width: 362,
        height: 362,
        stride_y: 512,
    };

    /// Returns an array of all predefined grid configurations.
    pub const fn all() -> [GridConfig; 7] {
        [
            Self::TINY,
            Self::SMALL,
            Self::MEDIUM,
            Self::LARGE,
            Self::LARGE_PLUS,
            Self::XLARGE,
            Self::XXLARGE,
        ]
    }

    /// Returns an array of default grid configurations (up to LARGE).
    pub const fn defaults() -> [GridConfig; 4] {
        [
            Self::TINY,
            Self::SMALL,
            Self::MEDIUM,
            Self::LARGE,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_config_from_target() {
        let config = GridConfig::from_target_nodes(1024);
        assert_eq!(config.width, 32);
        assert_eq!(config.height, 32);
        assert_eq!(config.stride_y, 32);
    }

    #[test]
    fn test_predefined_configs() {
        let tiny = TestGrids::TINY;
        assert_eq!(tiny.actual_nodes(), 17 * 17);
        assert_eq!(tiny.actual_nodes(), 289);

        let medium = TestGrids::MEDIUM;
        assert_eq!(medium.actual_nodes(), 32 * 32);
        assert_eq!(medium.actual_nodes(), 1024);
    }

    #[test]
    fn test_isqrt_edge_cases() {
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(1), 1);
        assert_eq!(isqrt(4), 2);
        assert_eq!(isqrt(5), 2);
        assert_eq!(isqrt(9), 3);
        assert_eq!(isqrt(10), 3);
        assert_eq!(isqrt(100), 10);
    }

    #[test]
    fn test_to_rectangular() {
        let config = GridConfig::from_target_nodes(1000);

        // Wide rectangle (2:1 aspect ratio)
        let rect = config.to_rectangular(2.0);
        assert!(rect.width > rect.height, "width should be greater for ratio > 1");
        assert!(rect.stride_y.is_power_of_two());

        // Tall rectangle (0.5:1 aspect ratio)
        let rect_tall = config.to_rectangular(0.5);
        assert!(rect_tall.height > rect_tall.width, "height should be greater for ratio < 1");
        assert!(rect_tall.stride_y.is_power_of_two());

        // Square (1:1 aspect ratio)
        let rect_square = config.to_rectangular(1.0);
        // Should be approximately square
        let diff = (rect_square.width as i64 - rect_square.height as i64).abs();
        assert!(diff <= 2, "should be approximately square");
    }

    #[test]
    fn test_all_grids() {
        let all = TestGrids::all();
        assert_eq!(all.len(), 7);
        assert_eq!(all[0].actual_nodes(), TestGrids::TINY.actual_nodes());
        assert_eq!(all[1].actual_nodes(), TestGrids::SMALL.actual_nodes());
        assert_eq!(all[2].actual_nodes(), TestGrids::MEDIUM.actual_nodes());
        assert_eq!(all[3].actual_nodes(), TestGrids::LARGE.actual_nodes());
        assert_eq!(all[4].actual_nodes(), TestGrids::LARGE_PLUS.actual_nodes());
        assert_eq!(all[5].actual_nodes(), TestGrids::XLARGE.actual_nodes());
        assert_eq!(all[6].actual_nodes(), TestGrids::XXLARGE.actual_nodes());
    }

    #[test]
    fn test_defaults_grids() {
        let defaults = TestGrids::defaults();
        assert_eq!(defaults.len(), 4);
        assert_eq!(defaults[0].actual_nodes(), TestGrids::TINY.actual_nodes());
        assert_eq!(defaults[3].actual_nodes(), TestGrids::LARGE.actual_nodes());
    }

    #[test]
    fn test_all_predefined_grids() {
        // Exercise all predefined grids to ensure coverage
        assert_eq!(TestGrids::SMALL.actual_nodes(), 22 * 22);
        assert_eq!(TestGrids::LARGE.actual_nodes(), 64 * 64);
        assert_eq!(TestGrids::LARGE_PLUS.actual_nodes(), 71 * 71);
        assert_eq!(TestGrids::XLARGE.actual_nodes(), 316 * 316);
        assert_eq!(TestGrids::XXLARGE.actual_nodes(), 362 * 362);

        // Verify stride_y is always power of two
        for config in TestGrids::all() {
            assert!(config.stride_y.is_power_of_two());
            assert!(config.stride_y >= config.width.max(config.height));
        }
    }
}

// ============================================================================
// Kani Formal Verification Proofs
// ============================================================================
//
// These proofs verify critical safety invariants in grid configuration.
//
// Run with: `cargo kani --package prav-core`

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // ============================================================================
    // Proof 1: isqrt maintains mathematical invariant
    // ============================================================================
    // File: testing_grids.rs:65-76
    // What: Prove isqrt(n)^2 <= n < (isqrt(n)+1)^2 for all n
    // Why: Incorrect sqrt → wrong grid dimensions → buffer overflows

    /// Verify isqrt returns correct integer square root.
    ///
    /// Setup: Symbolic input n bounded to realistic grid sizes (up to 1M nodes).
    /// Invariant: For all n, isqrt(n)^2 <= n and (isqrt(n)+1)^2 > n
    #[kani::proof]
    #[kani::unwind(33)] // Newton's method converges in O(log n) iterations
    fn verify_isqrt_invariant() {
        let n: usize = kani::any();
        kani::assume(n <= 1_000_000);

        let s = isqrt(n);

        // isqrt(n)^2 must be <= n
        let s_squared = s.checked_mul(s);
        kani::assert(
            s_squared.map(|sq| sq <= n).unwrap_or(false),
            "isqrt(n)^2 must be <= n",
        );

        // (isqrt(n)+1)^2 must be > n
        let s_plus_1 = s + 1;
        let s_plus_1_squared = s_plus_1.checked_mul(s_plus_1);
        kani::assert(
            s_plus_1_squared.map(|sq| sq > n).unwrap_or(true),
            "(isqrt(n)+1)^2 must be > n",
        );
    }

    // ============================================================================
    // Proof 2: Grid dimensions are always positive
    // ============================================================================
    // File: testing_grids.rs:17-28
    // What: Prove width, height, stride_y >= 1 for any valid target
    // Why: Zero dimensions → division by zero, empty allocations

    /// Verify GridConfig always produces positive dimensions.
    ///
    /// Setup: Symbolic target_nodes from 1 to 1M.
    /// Invariant: All dimensions are strictly positive.
    #[kani::proof]
    fn verify_grid_dimensions_positive() {
        let target: usize = kani::any();
        kani::assume(target >= 1 && target <= 1_000_000);

        let config = GridConfig::from_target_nodes(target);

        kani::assert(config.width >= 1, "width must be positive");
        kani::assert(config.height >= 1, "height must be positive");
        kani::assert(config.stride_y >= 1, "stride_y must be positive");
    }

    // ============================================================================
    // Proof 3: Stride is always a power of two
    // ============================================================================
    // File: testing_grids.rs:20
    // What: Prove stride_y is always a power of two
    // Why: Non-power-of-two stride → incorrect Morton encoding → wrong indices

    /// Verify stride_y is always a power of two.
    ///
    /// The decoder relies on stride being a power of two for Morton encoding.
    #[kani::proof]
    fn verify_stride_power_of_two() {
        let target: usize = kani::any();
        kani::assume(target >= 1 && target <= 1_000_000);

        let config = GridConfig::from_target_nodes(target);

        kani::assert(
            config.stride_y.is_power_of_two(),
            "stride_y must be power of two",
        );
    }

    // ============================================================================
    // Proof 4: Stride covers grid dimensions
    // ============================================================================
    // File: testing_grids.rs:20
    // What: Prove stride_y >= max(width, height)
    // Why: Insufficient stride → array index out of bounds

    /// Verify stride_y is large enough to cover the grid.
    #[kani::proof]
    fn verify_stride_covers_dimensions() {
        let target: usize = kani::any();
        kani::assume(target >= 1 && target <= 1_000_000);

        let config = GridConfig::from_target_nodes(target);

        kani::assert(
            config.stride_y >= config.width,
            "stride_y must be >= width",
        );
        kani::assert(
            config.stride_y >= config.height,
            "stride_y must be >= height",
        );
    }
}
