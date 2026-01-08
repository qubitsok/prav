use prav_core::topology::INTRA_BLOCK_NEIGHBORS;

#[test]
fn test_bloom_center() {
    // 8x8 grid. Center is roughly around (3,3).
    // Row-Major: index = y * 8 + x = 3*8 + 3 = 27.

    // Neighbors of 27:
    // Left: 26
    // Right: 28
    // Up: 19 (2*8+3)
    // Down: 35 (4*8+3)

    let center = 27;
    let neighbors = INTRA_BLOCK_NEIGHBORS[center];

    let expected = (1 << 26) | (1 << 28) | (1 << 19) | (1 << 35);
    assert_eq!(
        neighbors, expected,
        "Bloom test failed for center (3,3) -> index 27"
    );
    assert_eq!(neighbors.count_ones(), 4);
}

#[test]
fn test_wall_edge() {
    // Right edge of 8x8 tile. x=7, y=0.
    // Row-Major: index = 0*8 + 7 = 7.

    // Neighbors:
    // Left: 6
    // Right: None
    // Up: None
    // Down: 15 (1*8+7)

    let edge = 7;
    let neighbors = INTRA_BLOCK_NEIGHBORS[edge];

    let expected = (1 << 6) | (1 << 15);
    assert_eq!(
        neighbors, expected,
        "Wall test failed for edge (7,0) -> index 7"
    );
    // Explicitly check it doesn't wrap to anything unexpected
    assert_eq!(neighbors.count_ones(), 2);
}
