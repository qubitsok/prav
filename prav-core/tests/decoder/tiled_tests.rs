//! Unit tests for TiledDecodingState - large grid optimization.
//!
//! Tests the tiled decoder which breaks large grids into 32x32 tiles:
//! - Tile configuration and initialization
//! - Coordinate mapping between global and tiled layouts
//! - Sparse reset behavior
//! - Load syndrome mapping from dense to tiled format
//! - Union-Find operations across tile boundaries

use prav_core::arena::Arena;
use prav_core::decoder::TiledDecodingState;
use prav_core::topology::SquareGrid;

/// Helper to create a tiled decoder for testing
fn create_tiled_decoder<'a>(
    arena: &mut Arena<'a>,
    width: usize,
    height: usize,
) -> TiledDecodingState<'a, SquareGrid> {
    TiledDecodingState::<SquareGrid>::new(arena, width, height)
}

/// Compute node index from global (x, y) coordinates using tiled layout formula.
/// Returns None if coordinates are out of bounds.
fn get_tiled_node(width: usize, height: usize, tiles_x: usize, x: usize, y: usize) -> Option<u32> {
    if x >= width || y >= height {
        return None;
    }
    let tx = x / 32;
    let ty = y / 32;
    let lx = x % 32;
    let ly = y % 32;
    let tile_idx = ty * tiles_x + tx;
    let local_idx = ly * 32 + lx;
    Some((tile_idx * 1024 + local_idx) as u32)
}

/// Compute global (x, y) coordinates from tiled node index.
fn get_global_from_tiled(tiles_x: usize, node: u32) -> (usize, usize) {
    let tile_idx = (node as usize) / 1024;
    let local_idx = (node as usize) % 1024;

    let tx = tile_idx % tiles_x;
    let ty = tile_idx / tiles_x;

    let lx = local_idx % 32;
    let ly = local_idx / 32;

    (tx * 32 + lx, ty * 32 + ly)
}

// =============================================================================
// Tile Configuration Tests
// =============================================================================

#[test]
fn test_tiled_new_single_tile() {
    // 32x32 grid should use exactly 1 tile
    let mut memory = vec![0u8; 16 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = create_tiled_decoder(&mut arena, 32, 32);

    assert_eq!(decoder.tiles_x, 1, "32x32 grid should have 1 tile in x");
    assert_eq!(decoder.tiles_y, 1, "32x32 grid should have 1 tile in y");
    assert_eq!(decoder.width, 32);
    assert_eq!(decoder.height, 32);
}

#[test]
fn test_tiled_new_two_tiles_horizontal() {
    // 64x32 grid should use 2 tiles horizontally
    let mut memory = vec![0u8; 32 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = create_tiled_decoder(&mut arena, 64, 32);

    assert_eq!(decoder.tiles_x, 2, "64x32 grid should have 2 tiles in x");
    assert_eq!(decoder.tiles_y, 1, "64x32 grid should have 1 tile in y");
}

#[test]
fn test_tiled_new_two_tiles_vertical() {
    // 32x64 grid should use 2 tiles vertically
    let mut memory = vec![0u8; 32 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = create_tiled_decoder(&mut arena, 32, 64);

    assert_eq!(decoder.tiles_x, 1, "32x64 grid should have 1 tile in x");
    assert_eq!(decoder.tiles_y, 2, "32x64 grid should have 2 tiles in y");
}

#[test]
fn test_tiled_new_four_tiles() {
    // 64x64 grid should use 4 tiles (2x2)
    let mut memory = vec![0u8; 64 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = create_tiled_decoder(&mut arena, 64, 64);

    assert_eq!(decoder.tiles_x, 2, "64x64 grid should have 2 tiles in x");
    assert_eq!(decoder.tiles_y, 2, "64x64 grid should have 2 tiles in y");
}

#[test]
fn test_tiled_new_partial_tiles() {
    // 48x48 grid: (48 + 31) / 32 = 2 tiles per dimension
    let mut memory = vec![0u8; 32 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = create_tiled_decoder(&mut arena, 48, 48);

    assert_eq!(decoder.tiles_x, 2, "48x48 grid should have 2 tiles in x");
    assert_eq!(decoder.tiles_y, 2, "48x48 grid should have 2 tiles in y");
    assert_eq!(decoder.width, 48);
    assert_eq!(decoder.height, 48);
}

#[test]
fn test_tiled_new_asymmetric() {
    // 96x48 grid: 3 tiles x 2 tiles
    let mut memory = vec![0u8; 64 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = create_tiled_decoder(&mut arena, 96, 48);

    assert_eq!(decoder.tiles_x, 3, "96x48 grid should have 3 tiles in x");
    assert_eq!(decoder.tiles_y, 2, "96x48 grid should have 2 tiles in y");
}

// =============================================================================
// Valid Mask Initialization Tests
// =============================================================================

#[test]
fn test_tiled_initialize_full_tile() {
    let mut memory = vec![0u8; 16 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = create_tiled_decoder(&mut arena, 32, 32);

    // Single full tile should have 1024 valid nodes = 16 blocks all full
    let mut total_valid = 0u64;
    for block in decoder.blocks_state.iter() {
        total_valid += block.valid_mask.count_ones() as u64;
    }

    assert_eq!(total_valid, 1024, "32x32 grid should have 1024 valid nodes");
}

#[test]
fn test_tiled_initialize_partial_tile() {
    // 48x32: First tile full (32x32), second tile partial (16x32)
    let mut memory = vec![0u8; 32 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = create_tiled_decoder(&mut arena, 48, 32);

    let mut total_valid = 0u64;
    for block in decoder.blocks_state.iter() {
        total_valid += block.valid_mask.count_ones() as u64;
    }

    // 48 * 32 = 1536 valid nodes
    assert_eq!(total_valid, 1536, "48x32 grid should have 1536 valid nodes");
}

#[test]
fn test_tiled_initialize_corner_partial() {
    // 48x48: 4 tiles, only top-left is full (32x32 = 1024)
    // Total: 48 * 48 = 2304
    let mut memory = vec![0u8; 64 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = create_tiled_decoder(&mut arena, 48, 48);

    let mut total_valid = 0u64;
    for block in decoder.blocks_state.iter() {
        total_valid += block.valid_mask.count_ones() as u64;
    }

    assert_eq!(total_valid, 48 * 48, "48x48 grid should have 2304 valid nodes");
}

// =============================================================================
// Coordinate Mapping Tests (using helper functions)
// =============================================================================

#[test]
fn test_tiled_node_mapping_basic() {
    // 64x64 grid, tiles_x = 2
    let tiles_x = 2;

    // Node at (0, 0) should be node 0 in tile 0
    let node = get_tiled_node(64, 64, tiles_x, 0, 0).unwrap();
    assert_eq!(node, 0, "(0,0) should map to node 0");

    // Node at (31, 0) should be node 31 in tile 0
    let node = get_tiled_node(64, 64, tiles_x, 31, 0).unwrap();
    assert_eq!(node, 31, "(31,0) should map to node 31");

    // Node at (0, 1) should be node 32 in tile 0 (row 1)
    let node = get_tiled_node(64, 64, tiles_x, 0, 1).unwrap();
    assert_eq!(node, 32, "(0,1) should map to node 32");
}

#[test]
fn test_tiled_node_mapping_second_tile() {
    let tiles_x = 2;

    // Node at (32, 0) should be in tile 1 (right of tile 0)
    // Tile 1 starts at node 1024
    let node = get_tiled_node(64, 64, tiles_x, 32, 0).unwrap();
    assert_eq!(node, 1024, "(32,0) should be first node of tile 1");

    // Node at (33, 0) should be node 1025
    let node = get_tiled_node(64, 64, tiles_x, 33, 0).unwrap();
    assert_eq!(node, 1025, "(33,0) should be node 1025");
}

#[test]
fn test_tiled_node_mapping_bottom_tile() {
    let tiles_x = 2;

    // Node at (0, 32) should be in tile 2 (below tile 0)
    // Tile 2 starts at node 2 * 1024 = 2048
    let node = get_tiled_node(64, 64, tiles_x, 0, 32).unwrap();
    assert_eq!(node, 2048, "(0,32) should be first node of tile 2");
}

#[test]
fn test_tiled_node_mapping_out_of_bounds() {
    let tiles_x = 1;

    // Out of bounds should return None
    assert!(get_tiled_node(32, 32, tiles_x, 32, 0).is_none(), "x=32 should be out of bounds");
    assert!(get_tiled_node(32, 32, tiles_x, 0, 32).is_none(), "y=32 should be out of bounds");
    assert!(get_tiled_node(32, 32, tiles_x, 100, 100).is_none(), "Both out of bounds");
}

#[test]
fn test_tiled_global_coord_basic() {
    let tiles_x = 2;

    // Node 0 should be at (0, 0)
    let (x, y) = get_global_from_tiled(tiles_x, 0);
    assert_eq!((x, y), (0, 0), "Node 0 should be at (0, 0)");

    // Node 31 should be at (31, 0)
    let (x, y) = get_global_from_tiled(tiles_x, 31);
    assert_eq!((x, y), (31, 0), "Node 31 should be at (31, 0)");

    // Node 32 should be at (0, 1)
    let (x, y) = get_global_from_tiled(tiles_x, 32);
    assert_eq!((x, y), (0, 1), "Node 32 should be at (0, 1)");
}

#[test]
fn test_tiled_global_coord_second_tile() {
    let tiles_x = 2;

    // Node 1024 (first node of tile 1) should be at (32, 0)
    let (x, y) = get_global_from_tiled(tiles_x, 1024);
    assert_eq!((x, y), (32, 0), "Node 1024 should be at (32, 0)");
}

#[test]
fn test_tiled_coordinate_round_trip() {
    let width = 64;
    let height = 64;
    let tiles_x = 2;

    // Test round trip for various coordinates
    for y in [0usize, 15, 31, 32, 48, 63] {
        for x in [0usize, 15, 31, 32, 48, 63] {
            if let Some(node) = get_tiled_node(width, height, tiles_x, x, y) {
                let (rx, ry) = get_global_from_tiled(tiles_x, node);
                assert_eq!(
                    (rx, ry), (x, y),
                    "Round trip failed for ({}, {}): got ({}, {})",
                    x, y, rx, ry
                );
            }
        }
    }
}

// =============================================================================
// Union-Find Tests
// =============================================================================

#[test]
fn test_tiled_find_self_rooted() {
    let mut memory = vec![0u8; 32 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_tiled_decoder(&mut arena, 64, 64);

    // Initially all nodes should be self-rooted
    for i in 0..100 {
        let root = decoder.find(i);
        assert_eq!(root, i, "Node {} should be self-rooted", i);
    }
}

#[test]
fn test_tiled_find_path_compression() {
    let mut memory = vec![0u8; 32 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_tiled_decoder(&mut arena, 64, 64);

    // Create a chain: 0 -> 1 -> 2 -> 3 (root)
    decoder.parents[0] = 1;
    decoder.parents[1] = 2;
    decoder.parents[2] = 3;

    let root = decoder.find(0);
    assert_eq!(root, 3, "Should find root 3");
}

#[test]
fn test_tiled_union_same_tile() {
    let mut memory = vec![0u8; 32 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_tiled_decoder(&mut arena, 64, 64);

    // Union two nodes in the same tile
    let result = decoder.union(10, 20);
    assert!(result, "Union should return true");

    // Both should have same root
    let root10 = decoder.find(10);
    let root20 = decoder.find(20);
    assert_eq!(root10, root20, "Nodes should have same root after union");
}

#[test]
fn test_tiled_union_cross_tile() {
    let mut memory = vec![0u8; 64 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_tiled_decoder(&mut arena, 64, 64);

    // Union nodes from different tiles
    // Node 31 is in tile 0, node 1024 is in tile 1
    let result = decoder.union(31, 1024);
    assert!(result, "Cross-tile union should return true");

    let root31 = decoder.find(31);
    let root1024 = decoder.find(1024);
    assert_eq!(root31, root1024, "Cross-tile nodes should have same root");
}

#[test]
fn test_tiled_union_marks_dirty() {
    let mut memory = vec![0u8; 32 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_tiled_decoder(&mut arena, 64, 64);

    // Clear dirty masks
    decoder.block_dirty_mask.fill(0);

    // Union should mark blocks as dirty
    decoder.union(0, 100);

    // At least one block should be dirty
    let dirty_count: u32 = decoder.block_dirty_mask.iter().map(|w| w.count_ones()).sum();
    assert!(dirty_count > 0, "Union should mark blocks dirty");
}

// =============================================================================
// Sparse Reset Tests
// =============================================================================

#[test]
fn test_tiled_sparse_reset_clears_dirty() {
    let mut memory = vec![0u8; 32 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_tiled_decoder(&mut arena, 64, 64);

    // Modify block 0 state
    decoder.blocks_state[0].boundary = 0xDEAD;
    decoder.blocks_state[0].occupied = 0xBEEF;
    decoder.defect_mask[0] = 0xCAFE;

    // Mark block 0 dirty
    decoder.block_dirty_mask[0] = 1;

    // Reset
    decoder.sparse_reset();

    assert_eq!(decoder.blocks_state[0].boundary, 0);
    assert_eq!(decoder.blocks_state[0].occupied, 0);
    assert_eq!(decoder.defect_mask[0], 0);
}

#[test]
fn test_tiled_sparse_reset_resets_parents() {
    let mut memory = vec![0u8; 32 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_tiled_decoder(&mut arena, 64, 64);

    // Modify parents in block 0
    decoder.parents[0] = 100;
    decoder.parents[1] = 100;

    // Mark block 0 dirty
    decoder.block_dirty_mask[0] = 1;

    // Reset
    decoder.sparse_reset();

    assert_eq!(decoder.parents[0], 0, "Parent 0 should be reset");
    assert_eq!(decoder.parents[1], 1, "Parent 1 should be reset");
}

#[test]
fn test_tiled_sparse_reset_preserves_boundary() {
    let mut memory = vec![0u8; 32 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_tiled_decoder(&mut arena, 64, 64);

    let boundary_idx = decoder.parents.len() - 1;

    // Mark some blocks dirty
    decoder.block_dirty_mask[0] = 0xFF;

    decoder.sparse_reset();

    assert_eq!(
        decoder.parents[boundary_idx], boundary_idx as u32,
        "Boundary node should be preserved"
    );
}

// =============================================================================
// Load Dense Syndromes Tests
// =============================================================================

#[test]
fn test_load_dense_syndromes_single_defect() {
    let mut memory = vec![0u8; 32 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_tiled_decoder(&mut arena, 64, 64);

    // Create syndrome with single defect at (0, 0)
    // For 64x64 grid, stride = 64
    let mut syndromes = vec![0u64; 64]; // 64*64/64 = 64 words
    syndromes[0] = 1; // Bit 0 = node (0, 0)

    decoder.load_dense_syndromes(&syndromes);

    // Block 0 should have defect at bit 0
    assert_ne!(decoder.blocks_state[0].boundary & 1, 0, "Defect should be loaded");
    assert_ne!(decoder.defect_mask[0] & 1, 0, "Defect mask should be set");
}

#[test]
fn test_load_dense_syndromes_cross_tile() {
    let mut memory = vec![0u8; 64 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_tiled_decoder(&mut arena, 64, 64);

    // Create syndromes with defects in different tiles
    let mut syndromes = vec![0u64; 64];

    // Defect at (0, 0) - tile 0
    syndromes[0] |= 1;

    // Defect at (32, 0) - tile 1 (at global index 32)
    syndromes[0] |= 1 << 32;

    decoder.load_dense_syndromes(&syndromes);

    // Both tiles should have defects
    // Tile 0, block 0, bit 0
    assert_ne!(decoder.defect_mask[0] & 1, 0, "Tile 0 should have defect");

    // Tile 1 starts at block 16 (1 tile = 16 blocks)
    // Local coordinate (0, 0) in tile 1
    assert_ne!(decoder.defect_mask[16] & 1, 0, "Tile 1 should have defect");
}

#[test]
fn test_load_dense_syndromes_ignores_out_of_bounds() {
    let mut memory = vec![0u8; 16 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_tiled_decoder(&mut arena, 32, 32);

    // Create syndromes with bits beyond grid bounds
    // 32x32 grid with stride 32, only 32 words needed
    let mut syndromes = vec![0u64; 32];

    // Set bit at index 32 (which would be x=0, y=1 with stride 32)
    syndromes[0] = 1 << 32;

    decoder.load_dense_syndromes(&syndromes);

    // Should not panic and should load valid defects
}

// =============================================================================
// Integration Tests
// =============================================================================

#[test]
fn test_tiled_grow_simple_pair() {
    let mut memory = vec![0u8; 32 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let mut decoder = create_tiled_decoder(&mut arena, 64, 64);

    // Set up two adjacent defects at (0, 0) and (1, 0)
    // Using computed node indices
    let node_a = get_tiled_node(64, 64, decoder.tiles_x, 0, 0).unwrap();
    let node_b = get_tiled_node(64, 64, decoder.tiles_x, 1, 0).unwrap();

    let blk_a = node_a as usize / 64;
    let bit_a = node_a as usize % 64;
    let blk_b = node_b as usize / 64;
    let bit_b = node_b as usize % 64;

    decoder.blocks_state[blk_a].boundary |= 1 << bit_a;
    decoder.blocks_state[blk_a].occupied |= 1 << bit_a;
    decoder.defect_mask[blk_a] |= 1 << bit_a;

    decoder.blocks_state[blk_b].boundary |= 1 << bit_b;
    decoder.blocks_state[blk_b].occupied |= 1 << bit_b;
    decoder.defect_mask[blk_b] |= 1 << bit_b;

    // Mark blocks active
    decoder.active_mask[blk_a / 64] |= 1 << (blk_a % 64);
    decoder.active_mask[blk_b / 64] |= 1 << (blk_b % 64);

    // Grow clusters
    decoder.grow_clusters();

    // Both nodes should be in the same cluster
    let root_a = decoder.find(node_a);
    let root_b = decoder.find(node_b);
    assert_eq!(root_a, root_b, "Adjacent defects should be in same cluster");
}

#[test]
fn test_tiled_blocks_per_tile() {
    let mut memory = vec![0u8; 32 * 1024 * 1024];
    let mut arena = Arena::new(&mut memory);
    let decoder = create_tiled_decoder(&mut arena, 64, 64);

    // Each tile has 16 blocks (1024 nodes / 64 nodes per block)
    let num_tiles = decoder.tiles_x * decoder.tiles_y;
    let expected_blocks = num_tiles * 16;

    assert_eq!(decoder.blocks_state.len(), expected_blocks, "Should have correct number of blocks");
}

#[test]
fn test_tiled_union_deterministic() {
    // Union should be deterministic regardless of order
    let mut memory1 = vec![0u8; 32 * 1024 * 1024];
    let mut arena1 = Arena::new(&mut memory1);
    let mut decoder1 = create_tiled_decoder(&mut arena1, 64, 64);

    let mut memory2 = vec![0u8; 32 * 1024 * 1024];
    let mut arena2 = Arena::new(&mut memory2);
    let mut decoder2 = create_tiled_decoder(&mut arena2, 64, 64);

    // Same unions, different order
    decoder1.union(10, 20);
    decoder1.union(30, 20);

    decoder2.union(30, 20);
    decoder2.union(10, 20);

    // All three nodes should have same root in both decoders
    let root1_10 = decoder1.find(10);
    let root1_20 = decoder1.find(20);
    let root1_30 = decoder1.find(30);

    let root2_10 = decoder2.find(10);
    let root2_20 = decoder2.find(20);
    let root2_30 = decoder2.find(30);

    // All should be equal within each decoder
    assert_eq!(root1_10, root1_20);
    assert_eq!(root1_20, root1_30);
    assert_eq!(root2_10, root2_20);
    assert_eq!(root2_20, root2_30);

    // And the roots should match between decoders (deterministic)
    assert_eq!(root1_10, root2_10, "Union should be deterministic");
}
