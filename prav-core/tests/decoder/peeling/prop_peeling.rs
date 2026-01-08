//! Property-based tests for peeling module correctness.
//!
//! These tests verify algorithm invariants using random inputs.

use prav_core::arena::Arena;
use prav_core::decoder::DecodingState;
use prav_core::decoder::peeling::Peeling;
use prav_core::topology::SquareGrid;
use proptest::prelude::*;

proptest! {
    /// Property: Manhattan path length equals |dx| + |dy| + |dz|
    ///
    /// For any two nodes u and v, the number of edges emitted by
    /// trace_manhattan equals the Manhattan distance between them.
    #[test]
    fn prop_trace_manhattan_path_length(
        ux in 0usize..7,
        uy in 0usize..7,
        vx in 0usize..7,
        vy in 0usize..7,
    ) {
        let mut memory = vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

        let stride_y = 8;
        let u = (uy * stride_y + ux) as u32;
        let v = (vy * stride_y + vx) as u32;

        // Skip if same node
        if u == v {
            return Ok(());
        }

        decoder.trace_manhattan(u, v);

        // Count emitted edges by checking edge_dirty_count
        // Each emit_linear call for a valid edge increments dirty count (first time per word)
        // But we need to count bits in edge_bitmap
        let expected_distance = (ux as isize - vx as isize).unsigned_abs()
            + (uy as isize - vy as isize).unsigned_abs();

        // Count total set bits in edge_bitmap
        let mut total_edges = 0usize;
        for word in decoder.edge_bitmap.iter() {
            total_edges += word.count_ones() as usize;
        }

        prop_assert_eq!(
            total_edges,
            expected_distance,
            "Manhattan path length should equal coordinate distance"
        );
    }

    /// Property: emit_linear is XOR-idempotent
    ///
    /// Calling emit_linear twice with the same edge should cancel out,
    /// leaving the bitmap in its original state.
    #[test]
    fn prop_emit_linear_xor_idempotent(
        ux in 0usize..6,
        uy in 0usize..6,
        direction in 0usize..2, // 0 = horizontal (+1), 1 = vertical (+stride_y)
    ) {
        let mut memory = vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

        let stride_y = 8;
        let u = (uy * stride_y + ux) as u32;
        let v = if direction == 0 {
            u + 1 // horizontal
        } else {
            u + stride_y as u32 // vertical
        };

        // Save initial state
        let initial_bitmap: Vec<u64> = decoder.edge_bitmap.to_vec();

        // Emit edge twice
        decoder.emit_linear(u, v);
        decoder.emit_linear(u, v);

        // Bitmap should return to initial state (XOR property)
        for (i, &word) in decoder.edge_bitmap.iter().enumerate() {
            prop_assert_eq!(
                word,
                initial_bitmap[i],
                "Bitmap word {} should return to initial state after double emit",
                i
            );
        }
    }

    /// Property: get_coord is a bijection (roundtrip identity)
    ///
    /// For any valid index, get_coord extracts coordinates that
    /// reconstruct back to the original index.
    #[test]
    fn prop_get_coord_roundtrip(
        x in 0usize..8,
        y in 0usize..8,
    ) {
        let mut memory = vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut memory);
        let decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

        let stride_y = 8;

        // Construct index from coordinates
        let idx = y * stride_y + x;

        // Extract coordinates
        let (ex, ey, ez) = decoder.get_coord(idx as u32);

        // Verify roundtrip
        prop_assert_eq!(ex, x, "X coordinate mismatch");
        prop_assert_eq!(ey, y, "Y coordinate mismatch");
        prop_assert_eq!(ez, 0, "Z coordinate should be 0 for 2D grid");
    }

    /// Property: get_coord roundtrip for 3D grids
    ///
    /// For 3D grids, coordinate extraction should also roundtrip correctly.
    #[test]
    fn prop_get_coord_roundtrip_3d(
        x in 0usize..8,
        y in 0usize..8,
        z in 0usize..4,
    ) {
        let mut memory = vec![0u8; 1024 * 1024 * 16];
        let mut arena = Arena::new(&mut memory);
        let decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 4);

        let stride_y = 8;
        let stride_z = decoder.graph.stride_z;

        // Construct index from coordinates
        let idx = z * stride_z + y * stride_y + x;

        // Extract coordinates
        let (ex, ey, ez) = decoder.get_coord(idx as u32);

        // Verify roundtrip
        prop_assert_eq!(ex, x, "X coordinate mismatch in 3D");
        prop_assert_eq!(ey, y, "Y coordinate mismatch in 3D");
        prop_assert_eq!(ez, z, "Z coordinate mismatch in 3D");
    }
}
