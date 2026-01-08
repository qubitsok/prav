//! Surface code graph construction for fusion-blossom.
//!
//! Creates a matching graph equivalent to prav's SquareGrid topology.

use fusion_blossom::util::{SolverInitializer, VertexIndex, Weight};

/// Create a surface code matching graph for fusion-blossom.
///
/// Constructs a square grid with:
/// - Internal edges between adjacent nodes (horizontal and vertical)
/// - A single virtual boundary vertex connected to all boundary nodes
/// - Weights computed as `ln((1-p)/p)` scaled to integers
pub fn create_surface_code_graph(
    width: usize,
    height: usize,
    error_prob: f64,
) -> (SolverInitializer, Vec<(VertexIndex, VertexIndex, Weight)>) {
    let vertex_num = width * height;
    let mut weighted_edges = Vec::new();

    // Weight formula: weight = ln((1-p)/p) scaled to integer
    // Clamp probability to avoid log singularities
    // IMPORTANT: fusion-blossom requires EVEN weights
    let p = error_prob.clamp(1e-10, 1.0 - 1e-10);
    let weight = ((1.0 - p) / p).ln();
    let scaled_weight = ((weight * 500.0) as Weight) * 2;  // Ensure even weight

    // Horizontal edges (between horizontally adjacent nodes)
    for y in 0..height {
        for x in 0..width - 1 {
            let u = y * width + x;
            let v = y * width + x + 1;
            weighted_edges.push((u as VertexIndex, v as VertexIndex, scaled_weight));
        }
    }

    // Vertical edges (between vertically adjacent nodes)
    for y in 0..height - 1 {
        for x in 0..width {
            let u = y * width + x;
            let v = (y + 1) * width + x;
            weighted_edges.push((u as VertexIndex, v as VertexIndex, scaled_weight));
        }
    }

    // Single virtual boundary vertex
    let boundary_vertex = vertex_num as VertexIndex;
    let virtual_vertices = vec![boundary_vertex];

    // Connect boundary nodes to virtual vertex
    // Left boundary
    for y in 0..height {
        let node = y * width;
        weighted_edges.push((node as VertexIndex, boundary_vertex, scaled_weight));
    }
    // Right boundary
    for y in 0..height {
        let node = y * width + (width - 1);
        weighted_edges.push((node as VertexIndex, boundary_vertex, scaled_weight));
    }
    // Top boundary
    for x in 0..width {
        weighted_edges.push((x as VertexIndex, boundary_vertex, scaled_weight));
    }
    // Bottom boundary
    for x in 0..width {
        let node = (height - 1) * width + x;
        weighted_edges.push((node as VertexIndex, boundary_vertex, scaled_weight));
    }

    let initializer = SolverInitializer::new(
        vertex_num + 1,
        weighted_edges.clone(),
        virtual_vertices,
    );

    (initializer, weighted_edges)
}
