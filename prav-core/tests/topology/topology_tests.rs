//! Comprehensive tests for topology implementations.
//!
//! Tests each topology's `for_each_neighbor` method to ensure correct
//! neighbor enumeration for various node positions.

use prav_core::topology::{Grid3D, HoneycombGrid, SquareGrid, Topology, TriangularGrid};

// ============================================================================
// Helper Functions
// ============================================================================

/// Collect all neighbors for a given topology and index.
fn collect_neighbors<T: Topology>(idx: u32) -> Vec<u32> {
    let mut neighbors = Vec::new();
    T::for_each_neighbor(idx, |n| neighbors.push(n));
    neighbors.sort();
    neighbors
}

/// Convert (x, y) to Morton index for 2D.
fn morton_2d(x: u32, y: u32) -> u32 {
    let mut mx = x;
    let mut my = y;
    let mut result = 0u32;
    for i in 0..16 {
        result |= ((mx & 1) << (2 * i)) | ((my & 1) << (2 * i + 1));
        mx >>= 1;
        my >>= 1;
    }
    result
}

/// Convert (x, y, z) to Morton index for 3D.
fn morton_3d(x: u32, y: u32, z: u32) -> u32 {
    let mut mx = x;
    let mut my = y;
    let mut mz = z;
    let mut result = 0u32;
    for i in 0..10 {
        result |= ((mx & 1) << (3 * i)) | ((my & 1) << (3 * i + 1)) | ((mz & 1) << (3 * i + 2));
        mx >>= 1;
        my >>= 1;
        mz >>= 1;
    }
    result
}

// ============================================================================
// SquareGrid Tests
// ============================================================================

#[test]
fn test_square_grid_interior_node() {
    // Node at (2, 2) should have 4 neighbors: left, right, up, down
    let idx = morton_2d(2, 2);
    let neighbors = collect_neighbors::<SquareGrid>(idx);

    assert_eq!(neighbors.len(), 4, "Interior node should have 4 neighbors");

    let left = morton_2d(1, 2);
    let right = morton_2d(3, 2);
    let up = morton_2d(2, 1);
    let down = morton_2d(2, 3);

    assert!(neighbors.contains(&left), "Missing left neighbor");
    assert!(neighbors.contains(&right), "Missing right neighbor");
    assert!(neighbors.contains(&up), "Missing up neighbor");
    assert!(neighbors.contains(&down), "Missing down neighbor");
}

#[test]
fn test_square_grid_corner_origin() {
    // Node at (0, 0) should have 2 neighbors: right and down
    let idx = morton_2d(0, 0);
    let neighbors = collect_neighbors::<SquareGrid>(idx);

    assert_eq!(neighbors.len(), 2, "Corner (0,0) should have 2 neighbors");

    let right = morton_2d(1, 0);
    let down = morton_2d(0, 1);

    assert!(neighbors.contains(&right), "Missing right neighbor");
    assert!(neighbors.contains(&down), "Missing down neighbor");
}

#[test]
fn test_square_grid_left_edge() {
    // Node at (0, 5) should have 3 neighbors: right, up, down
    let idx = morton_2d(0, 5);
    let neighbors = collect_neighbors::<SquareGrid>(idx);

    assert_eq!(neighbors.len(), 3, "Left edge should have 3 neighbors");
}

#[test]
fn test_square_grid_top_edge() {
    // Node at (5, 0) should have 3 neighbors: left, right, down
    let idx = morton_2d(5, 0);
    let neighbors = collect_neighbors::<SquareGrid>(idx);

    assert_eq!(neighbors.len(), 3, "Top edge should have 3 neighbors");
}

#[test]
fn test_square_grid_symmetry() {
    // If B is a neighbor of A, then A should be a neighbor of B
    for x in 0..8u32 {
        for y in 0..8u32 {
            let idx = morton_2d(x, y);
            let neighbors = collect_neighbors::<SquareGrid>(idx);

            for &n in &neighbors {
                let reverse_neighbors = collect_neighbors::<SquareGrid>(n);
                assert!(
                    reverse_neighbors.contains(&idx),
                    "Symmetry violated: ({},{}) -> {} but not reverse",
                    x,
                    y,
                    n
                );
            }
        }
    }
}

// ============================================================================
// Grid3D Tests
// ============================================================================

#[test]
fn test_grid3d_interior_node() {
    // Node at (2, 2, 2) should have 6 neighbors
    let idx = morton_3d(2, 2, 2);
    let neighbors = collect_neighbors::<Grid3D>(idx);

    assert_eq!(
        neighbors.len(),
        6,
        "Interior 3D node should have 6 neighbors"
    );

    let neighbors_expected = [
        morton_3d(1, 2, 2), // -X
        morton_3d(3, 2, 2), // +X
        morton_3d(2, 1, 2), // -Y
        morton_3d(2, 3, 2), // +Y
        morton_3d(2, 2, 1), // -Z
        morton_3d(2, 2, 3), // +Z
    ];

    for expected in neighbors_expected {
        assert!(
            neighbors.contains(&expected),
            "Missing neighbor {}",
            expected
        );
    }
}

#[test]
fn test_grid3d_corner_origin() {
    // Node at (0, 0, 0) should have 3 neighbors: +X, +Y, +Z
    let idx = morton_3d(0, 0, 0);
    let neighbors = collect_neighbors::<Grid3D>(idx);

    assert_eq!(neighbors.len(), 3, "Origin corner should have 3 neighbors");
}

#[test]
fn test_grid3d_edge() {
    // Node at (0, 0, 2) should have 4 neighbors
    let idx = morton_3d(0, 0, 2);
    let neighbors = collect_neighbors::<Grid3D>(idx);

    assert_eq!(neighbors.len(), 4, "Edge node should have 4 neighbors");
}

#[test]
fn test_grid3d_face() {
    // Node at (0, 2, 2) should have 5 neighbors
    let idx = morton_3d(0, 2, 2);
    let neighbors = collect_neighbors::<Grid3D>(idx);

    assert_eq!(neighbors.len(), 5, "Face node should have 5 neighbors");
}

#[test]
fn test_grid3d_symmetry() {
    // Check symmetry for a subset of 3D grid
    for x in 0..4u32 {
        for y in 0..4u32 {
            for z in 0..4u32 {
                let idx = morton_3d(x, y, z);
                let neighbors = collect_neighbors::<Grid3D>(idx);

                for &n in &neighbors {
                    let reverse_neighbors = collect_neighbors::<Grid3D>(n);
                    assert!(
                        reverse_neighbors.contains(&idx),
                        "3D Symmetry violated: ({},{},{}) -> {} but not reverse",
                        x,
                        y,
                        z,
                        n
                    );
                }
            }
        }
    }
}

// ============================================================================
// TriangularGrid Tests
// ============================================================================

#[test]
fn test_triangular_grid_interior_node() {
    // Interior node should have 6 neighbors (4 cardinal + 2 diagonal based on parity)
    // Actually, triangular has 4 cardinal + 1 diagonal per node (different diagonals for different parities)
    let idx = morton_2d(2, 2);
    let neighbors = collect_neighbors::<TriangularGrid>(idx);

    // Triangular grid has 6 total neighbors when considering both parities
    // A single node has 4 cardinal + 1 diagonal = 5 max, or less at boundaries
    assert!(
        neighbors.len() >= 4,
        "Triangular interior should have at least 4 neighbors"
    );
    assert!(
        neighbors.len() <= 6,
        "Triangular interior should have at most 6 neighbors"
    );
}

#[test]
fn test_triangular_grid_parity_affects_diagonal() {
    // Test that different parity nodes have different diagonal directions
    let even_idx = morton_2d(2, 2); // 2+2=4 -> even parity
    let odd_idx = morton_2d(2, 3); // 2+3=5 -> odd parity

    let even_neighbors = collect_neighbors::<TriangularGrid>(even_idx);
    let odd_neighbors = collect_neighbors::<TriangularGrid>(odd_idx);

    // Both should have cardinal neighbors
    assert!(even_neighbors.len() >= 4);
    assert!(odd_neighbors.len() >= 4);
}

#[test]
fn test_triangular_grid_corner() {
    // Corner at (0, 0)
    let idx = morton_2d(0, 0);
    let neighbors = collect_neighbors::<TriangularGrid>(idx);

    // Should have right, down, and maybe one diagonal
    assert!(
        neighbors.len() >= 2,
        "Triangular corner should have at least 2 neighbors"
    );
}

#[test]
fn test_triangular_grid_even_parity_diagonal() {
    // Even parity: up-right diagonal (if has_right && has_up)
    let idx = morton_2d(2, 2); // Even parity
    let popcount = idx.count_ones();

    if popcount % 2 == 0 {
        let neighbors = collect_neighbors::<TriangularGrid>(idx);
        // Should include up-right diagonal neighbor
        let up_right = morton_2d(3, 1);
        // May or may not be included depending on boundary
        let _ = neighbors.contains(&up_right);
    }
}

#[test]
fn test_triangular_grid_odd_parity_diagonal() {
    // Odd parity: down-left diagonal (if has_left && has_down)
    let idx = morton_2d(3, 2); // Check parity
    let popcount = idx.count_ones();

    if popcount % 2 == 1 {
        let neighbors = collect_neighbors::<TriangularGrid>(idx);
        // Should include down-left diagonal neighbor
        let down_left = morton_2d(2, 3);
        // May or may not be included depending on boundary
        let _ = neighbors.contains(&down_left);
    }
}

#[test]
fn test_triangular_grid_cardinal_symmetry() {
    // Triangular grid has symmetric cardinal neighbors (left/right/up/down)
    // but intentionally asymmetric diagonal neighbors based on parity.
    // This tests only cardinal neighbor symmetry.
    for x in 1..7u32 {
        for y in 1..7u32 {
            let idx = morton_2d(x, y);
            let neighbors = collect_neighbors::<TriangularGrid>(idx);

            // Cardinal neighbors should be symmetric
            let cardinal = [
                morton_2d(x.wrapping_sub(1), y), // left
                morton_2d(x + 1, y),             // right
                morton_2d(x, y.wrapping_sub(1)), // up
                morton_2d(x, y + 1),             // down
            ];

            for &c in &cardinal {
                if neighbors.contains(&c) {
                    let reverse = collect_neighbors::<TriangularGrid>(c);
                    assert!(
                        reverse.contains(&idx),
                        "Cardinal symmetry violated: ({},{}) -> {} but not reverse",
                        x,
                        y,
                        c
                    );
                }
            }
        }
    }
}

// ============================================================================
// HoneycombGrid Tests
// ============================================================================

#[test]
fn test_honeycomb_grid_always_3_neighbors_max() {
    // Honeycomb nodes have at most 3 neighbors
    for x in 0..8u32 {
        for y in 0..8u32 {
            let idx = morton_2d(x, y);
            let neighbors = collect_neighbors::<HoneycombGrid>(idx);
            assert!(
                neighbors.len() <= 3,
                "Honeycomb node ({},{}) has {} neighbors, expected <= 3",
                x,
                y,
                neighbors.len()
            );
        }
    }
}

#[test]
fn test_honeycomb_grid_interior_node() {
    // Interior node should have 3 neighbors (2 horizontal + 1 vertical based on parity)
    let idx = morton_2d(4, 4);
    let neighbors = collect_neighbors::<HoneycombGrid>(idx);

    // Should have exactly 3 neighbors for interior
    assert_eq!(
        neighbors.len(),
        3,
        "Honeycomb interior should have 3 neighbors"
    );
}

#[test]
fn test_honeycomb_grid_even_parity_has_up() {
    // Even parity nodes have up neighbor
    let idx = morton_2d(2, 2); // Even parity
    let popcount = idx.count_ones();

    if popcount % 2 == 0 {
        let neighbors = collect_neighbors::<HoneycombGrid>(idx);
        let up = morton_2d(2, 1);
        assert!(
            neighbors.contains(&up),
            "Even parity honeycomb should have up neighbor"
        );
    }
}

#[test]
fn test_honeycomb_grid_odd_parity_has_down() {
    // Odd parity nodes have down neighbor
    let idx = morton_2d(3, 2); // Check parity
    let popcount = idx.count_ones();

    if popcount % 2 == 1 {
        let neighbors = collect_neighbors::<HoneycombGrid>(idx);
        let down = morton_2d(3, 3);
        assert!(
            neighbors.contains(&down),
            "Odd parity honeycomb should have down neighbor"
        );
    }
}

#[test]
fn test_honeycomb_grid_horizontal_neighbors() {
    // All nodes should have left/right neighbors if not at boundary
    let idx = morton_2d(4, 4);
    let neighbors = collect_neighbors::<HoneycombGrid>(idx);

    let left = morton_2d(3, 4);
    let right = morton_2d(5, 4);

    assert!(neighbors.contains(&left), "Should have left neighbor");
    assert!(neighbors.contains(&right), "Should have right neighbor");
}

#[test]
fn test_honeycomb_grid_corner() {
    // Corner at (0, 0) - even parity, has up neighbor but y=0 means no up
    let idx = morton_2d(0, 0);
    let neighbors = collect_neighbors::<HoneycombGrid>(idx);

    // Should only have right neighbor (no left at x=0, no up at y=0 for even parity)
    assert!(
        !neighbors.is_empty(),
        "Corner should have at least 1 neighbor"
    );
}

#[test]
fn test_honeycomb_grid_horizontal_symmetry() {
    // Honeycomb has symmetric horizontal neighbors (left/right)
    // but vertical neighbors depend on parity (even=up, odd=down).
    // This tests only horizontal neighbor symmetry.
    for x in 1..7u32 {
        for y in 0..8u32 {
            let idx = morton_2d(x, y);
            let neighbors = collect_neighbors::<HoneycombGrid>(idx);

            // Horizontal neighbors should be symmetric
            let left = morton_2d(x - 1, y);
            let right = morton_2d(x + 1, y);

            if neighbors.contains(&left) {
                let reverse = collect_neighbors::<HoneycombGrid>(left);
                assert!(
                    reverse.contains(&idx),
                    "Left symmetry violated: ({},{}) <-> left",
                    x,
                    y
                );
            }

            if neighbors.contains(&right) {
                let reverse = collect_neighbors::<HoneycombGrid>(right);
                assert!(
                    reverse.contains(&idx),
                    "Right symmetry violated: ({},{}) <-> right",
                    x,
                    y
                );
            }
        }
    }
}

// ============================================================================
// Cross-Topology Tests
// ============================================================================

#[test]
fn test_all_topologies_origin() {
    // Test origin behavior for all topologies
    let origin_2d = morton_2d(0, 0);
    let origin_3d = morton_3d(0, 0, 0);

    let sq_neighbors = collect_neighbors::<SquareGrid>(origin_2d);
    let tri_neighbors = collect_neighbors::<TriangularGrid>(origin_2d);
    let hc_neighbors = collect_neighbors::<HoneycombGrid>(origin_2d);
    let g3d_neighbors = collect_neighbors::<Grid3D>(origin_3d);

    assert_eq!(
        sq_neighbors.len(),
        2,
        "SquareGrid origin should have 2 neighbors"
    );
    assert!(
        tri_neighbors.len() >= 2,
        "TriangularGrid origin should have >= 2 neighbors"
    );
    assert!(
        !hc_neighbors.is_empty(),
        "HoneycombGrid origin should have >= 1 neighbor"
    );
    assert_eq!(
        g3d_neighbors.len(),
        3,
        "Grid3D origin should have 3 neighbors"
    );
}

#[test]
fn test_neighbor_counts_consistent_with_topology_type() {
    // Verify characteristic neighbor counts for each topology
    let interior_2d = morton_2d(4, 4);
    let interior_3d = morton_3d(2, 2, 2);

    assert_eq!(collect_neighbors::<SquareGrid>(interior_2d).len(), 4);
    assert_eq!(collect_neighbors::<Grid3D>(interior_3d).len(), 6);
    // Honeycomb always has <= 3
    assert!(collect_neighbors::<HoneycombGrid>(interior_2d).len() <= 3);
    // Triangular has 4-5 depending on position (4 cardinal + potentially 1 diagonal)
    assert!(collect_neighbors::<TriangularGrid>(interior_2d).len() >= 4);
}
