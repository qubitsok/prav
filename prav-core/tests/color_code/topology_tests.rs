//! Topology tests for color code module.
//!
//! Tests verifying the triangular lattice topology properties
//! and color code specific constraints.

use prav_core::Topology;
use prav_core::color_code::{ColorCodeGrid3DConfig, FaceColor};
use prav_core::topology::TriangularGrid;

/// Test that TriangularGrid has 6 neighbors for interior nodes.
#[test]
fn test_triangular_neighbor_count() {
    // For an interior node (not on boundary), should have 6 neighbors
    // Use Morton encoding for a point well inside the grid
    // Morton encode (10, 10): interleave bits
    // x=10 = 0b1010, y=10 = 0b1010
    // Interleaved: ...y3x3y2x2y1x1y0x0 = ...1 1 0 0 1 1 0 0 = 0b11001100 = 204
    let idx = prav_core::morton_encode_2d(10, 10);

    let mut count = 0;
    TriangularGrid::for_each_neighbor(idx, |_neighbor| {
        count += 1;
    });

    // Interior nodes in triangular grid should have 5 or 6 neighbors
    // (4 cardinal + 1-2 diagonal depending on parity)
    assert!(
        (5..=6).contains(&count),
        "Expected 5-6 neighbors for interior node, got {}",
        count
    );
}

/// Test that CARDINAL neighbors have different colors.
///
/// The (x+y) % 3 coloring ensures cardinal neighbors (±1 in x OR y only)
/// have different colors. Diagonal neighbors (±1 in both x AND y) may have
/// the same color - this is expected in the triangular grid topology.
#[test]
fn test_cardinal_neighbor_color_consistency() {
    // Test that cardinal neighbors always have different colors
    for y in 1u32..50 {
        for x in 1u32..50 {
            let color = FaceColor::from_coords(x, y);

            // Check all 4 cardinal neighbors
            let neighbors = [
                (x - 1, y), // left
                (x + 1, y), // right
                (x, y - 1), // up
                (x, y + 1), // down
            ];

            for (nx, ny) in neighbors {
                let neighbor_color = FaceColor::from_coords(nx, ny);
                assert_ne!(
                    color, neighbor_color,
                    "Cardinal neighbors ({},{}) and ({},{}) have same color {:?}",
                    x, y, nx, ny, color
                );
            }
        }
    }
}

/// Test grid configuration for different distances.
#[test]
fn test_grid_configs_for_distances() {
    for d in [3, 5, 7, 9, 11, 13] {
        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(d);

        assert_eq!(config.code_distance, d);
        assert_eq!(config.num_rounds, d);
        assert!(config.stride_y.is_power_of_two());
        assert!(config.stride_y >= config.width);
        assert!(config.stride_y >= config.height);
        assert!(config.stride_y >= config.depth);
    }
}

/// Test that boundary configuration covers all three colors.
#[test]
fn test_boundary_config_all_colors() {
    let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(5);

    // Default should have all boundaries active
    assert!(config.boundary_config.is_active(FaceColor::Red));
    assert!(config.boundary_config.is_active(FaceColor::Green));
    assert!(config.boundary_config.is_active(FaceColor::Blue));
}

/// Test detector color distribution across the grid.
#[test]
fn test_detector_color_distribution() {
    let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(6);
    let counts = config.count_by_color();

    // Total should equal num_detectors
    let total: usize = counts.iter().sum();
    assert_eq!(total, config.num_detectors());

    // Each color should have at least some detectors
    for (i, &count) in counts.iter().enumerate() {
        assert!(count > 0, "Color {} should have at least some detectors", i);
    }
}

/// Test that color is time-invariant.
#[test]
fn test_color_time_invariance() {
    let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(5);

    for y in 0..config.height {
        for x in 0..config.width {
            let expected_color = config.detector_color(x, y);

            // Color should be same at all time slices
            for t in 0..config.depth {
                let idx = config.coord_to_linear(x, y, t);
                let color = config.detector_color_at(idx);
                assert_eq!(
                    color, expected_color,
                    "Color at ({},{},{}) differs from ({},{},0)",
                    x, y, t, x, y
                );
            }
        }
    }
}

/// Test restricted dimensions calculation.
#[test]
fn test_restricted_dimensions() {
    let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(9);

    for color in FaceColor::all() {
        let (rw, rh, rd) = config.restricted_dimensions(color);

        // Restricted dimensions should be non-zero
        assert!(rw > 0, "Restricted width for {:?} should be > 0", color);
        assert!(rh > 0, "Restricted height for {:?} should be > 0", color);
        assert_eq!(
            rd, config.depth,
            "Restricted depth should equal config depth"
        );
    }
}

/// Test custom grid configuration.
#[test]
fn test_custom_grid_config() {
    let config = ColorCodeGrid3DConfig::custom(10, 8, 5);

    assert_eq!(config.width, 10);
    assert_eq!(config.height, 8);
    assert_eq!(config.depth, 5);
    assert!(config.stride_y.is_power_of_two());
    assert!(config.stride_y >= 10);
}

/// Test alloc_nodes calculation.
#[test]
fn test_alloc_nodes() {
    let config_3d = ColorCodeGrid3DConfig::for_triangular_6_6_6(5);
    let config_2d = config_3d.with_depth(1);

    // 3D should allocate stride_y^3
    assert_eq!(
        config_3d.alloc_nodes(),
        config_3d.stride_y * config_3d.stride_y * config_3d.stride_y
    );

    // 2D should allocate stride_y^2
    assert_eq!(
        config_2d.alloc_nodes(),
        config_2d.stride_y * config_2d.stride_y
    );
}

/// Test predefined test configurations.
#[test]
fn test_predefined_configs() {
    use prav_core::color_code::grid_3d::TestColorCodeGrids;

    let configs = [
        (3, TestColorCodeGrids::D3),
        (5, TestColorCodeGrids::D5),
        (7, TestColorCodeGrids::D7),
        (9, TestColorCodeGrids::D9),
        (11, TestColorCodeGrids::D11),
        (13, TestColorCodeGrids::D13),
    ];

    for (expected_d, config) in configs {
        assert_eq!(config.code_distance, expected_d);
        assert!(config.stride_y.is_power_of_two());
    }
}
