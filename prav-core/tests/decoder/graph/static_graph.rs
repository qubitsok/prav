use prav_core::arena::Arena;
use prav_core::decoder::DecodingState;
use prav_core::topology::SquareGrid;

/// Test that StaticGraph fields are correctly initialized from decoder dimensions.
#[test]
fn test_static_graph_dimensions() {
    let mut buffer = [0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut buffer);

    let width = 8;
    let height = 8;
    let depth = 1;

    let decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, width, height, depth);
    let graph = decoder.graph;

    // Verify dimensions
    assert_eq!(graph.width, width);
    assert_eq!(graph.height, height);
    assert_eq!(graph.depth, depth);

    // Verify stride calculations
    // For 8x8 grid, stride_y should be 8 (next_power_of_two(8) = 8)
    assert_eq!(graph.stride_y, 8);
    assert_eq!(graph.stride_x, 1);
    assert_eq!(graph.stride_z, 64); // 8 * 8

    // Verify shift values
    assert_eq!(graph.shift_y, 3); // log2(8)
    assert_eq!(graph.shift_z, 6); // log2(64)
}

/// Test that StaticGraph correctly handles non-power-of-2 dimensions.
#[test]
fn test_static_graph_non_pow2_dimensions() {
    let mut buffer = [0u8; 1024 * 1024];
    let mut arena = Arena::new(&mut buffer);

    let width = 6;
    let height = 5;
    let depth = 1;

    let decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, width, height, depth);
    let graph = decoder.graph;

    // Verify dimensions preserved
    assert_eq!(graph.width, width);
    assert_eq!(graph.height, height);

    // For 6x5 grid, max_dim=6, stride_y should be 8 (next_power_of_two(6) = 8)
    assert_eq!(graph.stride_y, 8);
    assert_eq!(graph.stride_z, 64);
}
