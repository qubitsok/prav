//! Tests for tiled decoder peeling operations.

use crate::common;

use prav_core::arena::Arena;
use prav_core::decoder::EdgeCorrection;
use prav_core::decoder::tiled::TiledDecodingState;
use prav_core::topology::SquareGrid;

#[test]
fn test_tiled_peeling_intra_tile() {
    let mut memory = vec![0u8; 1024 * 1024 * 16];
    let mut arena = Arena::new(&mut memory);

    // 32x32 grid (1 tile)
    let width = 32;
    let height = 32;
    let mut decoder = TiledDecodingState::<SquareGrid>::new(&mut arena, width, height);

    let mut defects = vec![0u64; decoder.defect_mask.len()];

    // Pair 1: (5, 5) and (5, 10). Vertical.
    let n1 = 5 * 32 + 5;
    let n2 = 10 * 32 + 5;

    // Pair 2: (10, 10) and (15, 10). Horizontal.
    let n3 = 10 * 32 + 10;
    let n4 = 10 * 32 + 15;

    let nodes = [n1, n2, n3, n4];

    for &n in &nodes {
        let blk = n / 64;
        let bit = n % 64;
        defects[blk] |= 1 << bit;
    }

    decoder.load_dense_syndromes(&defects);
    decoder.grow_clusters();

    let mut corrections = vec![EdgeCorrection::default(); 1000];
    let count = decoder.peel_forest(&mut corrections);

    assert!(
        common::verify_matching_bool(&defects, &corrections[0..count]),
        "Intra-tile peeling failed: defects remain unmatched"
    );
}

#[test]
fn test_tiled_peeling_cross_tile_horizontal() {
    let mut memory = vec![0u8; 1024 * 1024 * 16];
    let mut arena = Arena::new(&mut memory);

    // 64x32 grid (2 tiles wide)
    let width = 64;
    let height = 32;
    let mut decoder = TiledDecodingState::<SquareGrid>::new(&mut arena, width, height);

    let mut defects = vec![0u64; decoder.defect_mask.len()];

    // Pair: (31, 0) in Tile 0 and (32, 0) in Tile 1
    let n1 = 31;
    let n2 = 1024;

    let blk1 = n1 / 64;
    let bit1 = n1 % 64;
    defects[blk1] |= 1 << bit1;

    let blk2 = n2 / 64;
    let bit2 = n2 % 64;
    defects[blk2] |= 1 << bit2;

    // Inject manually into decoder state
    decoder.defect_mask[blk1] |= 1 << bit1;
    decoder.blocks_state[blk1].occupied |= 1 << bit1;
    decoder.blocks_state[blk1].boundary |= 1 << bit1;
    decoder.active_mask[blk1 >> 6] |= 1 << (blk1 & 63);

    decoder.defect_mask[blk2] |= 1 << bit2;
    decoder.blocks_state[blk2].occupied |= 1 << bit2;
    decoder.blocks_state[blk2].boundary |= 1 << bit2;
    decoder.active_mask[blk2 >> 6] |= 1 << (blk2 & 63);

    decoder.grow_clusters();

    let mut corrections = vec![EdgeCorrection::default(); 1000];
    let count = decoder.peel_forest(&mut corrections);

    assert!(
        common::verify_matching_bool(&defects, &corrections[0..count]),
        "Cross-tile horizontal peeling failed"
    );
}

#[test]
fn test_tiled_peeling_cross_tile_vertical() {
    let mut memory = vec![0u8; 1024 * 1024 * 16];
    let mut arena = Arena::new(&mut memory);

    // 32x64 grid (2 tiles high)
    let width = 32;
    let height = 64;
    let mut decoder = TiledDecodingState::<SquareGrid>::new(&mut arena, width, height);

    let mut defects = vec![0u64; decoder.defect_mask.len()];

    // Pair: (0, 31) in Tile 0 and (0, 32) in Tile 1
    let n1 = 992;
    let n2 = 1024;

    let blk1 = n1 / 64;
    let bit1 = n1 % 64;
    defects[blk1] |= 1 << bit1;

    let blk2 = n2 / 64;
    let bit2 = n2 % 64;
    defects[blk2] |= 1 << bit2;

    // Inject manually
    decoder.defect_mask[blk1] |= 1 << bit1;
    decoder.blocks_state[blk1].occupied |= 1 << bit1;
    decoder.blocks_state[blk1].boundary |= 1 << bit1;
    decoder.active_mask[blk1 >> 6] |= 1 << (blk1 & 63);

    decoder.defect_mask[blk2] |= 1 << bit2;
    decoder.blocks_state[blk2].occupied |= 1 << bit2;
    decoder.blocks_state[blk2].boundary |= 1 << bit2;
    decoder.active_mask[blk2 >> 6] |= 1 << (blk2 & 63);

    decoder.grow_clusters();

    let mut corrections = vec![EdgeCorrection::default(); 1000];
    let count = decoder.peel_forest(&mut corrections);

    assert!(
        common::verify_matching_bool(&defects, &corrections[0..count]),
        "Cross-tile vertical peeling failed"
    );
}

#[test]
fn test_tiled_peeling_to_boundary() {
    let mut memory = vec![0u8; 1024 * 1024 * 16];
    let mut arena = Arena::new(&mut memory);

    // 32x32 grid
    let width = 32;
    let height = 32;
    let mut decoder = TiledDecodingState::<SquareGrid>::new(&mut arena, width, height);

    let mut defects = vec![0u64; decoder.defect_mask.len()];

    // Node close to boundary: (0, 0).
    let n1 = 0;

    let blk1 = n1 / 64;
    let bit1 = n1 % 64;
    defects[blk1] |= 1 << bit1;

    // Inject
    decoder.defect_mask[blk1] |= 1 << bit1;
    decoder.blocks_state[blk1].occupied |= 1 << bit1;
    decoder.blocks_state[blk1].boundary |= 1 << bit1;
    decoder.active_mask[blk1 >> 6] |= 1 << (blk1 & 63);

    decoder.grow_clusters();

    let mut corrections = vec![EdgeCorrection::default(); 1000];
    let count = decoder.peel_forest(&mut corrections);

    // Verify
    assert!(
        common::verify_matching_bool(&defects, &corrections[0..count]),
        "Boundary peeling failed"
    );
    assert!(count > 0);

    // Check if correction targets u32::MAX (boundary)
    let has_boundary = corrections[0..count].iter().any(|c| c.v == u32::MAX);
    assert!(has_boundary, "Expected correction to boundary");
}
