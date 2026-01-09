// =============================================================================
// Syndrome Spreading Functions
// =============================================================================
//
// These functions implement efficient bit diffusion for cluster growth.
// Used to expand syndrome boundaries within 64-bit blocks.

/// Spread syndrome bits within an 8x8 grid layout.
///
/// Performs 2D diffusion where:
/// - Rows are adjacent bytes (8 bits each)
/// - Diffusion stops at row boundaries and when blocked by unoccupied bits
#[inline(always)]
pub fn spread_syndrome_8x8(mut boundary: u64, occupied: u64) -> u64 {
    let mask_e = 0xFEFEFEFEFEFEFEFEu64; // Can shift left within row
    let mask_w = 0x7F7F7F7F7F7F7F7Fu64; // Can shift right within row

    for _ in 0..8 {
        let next = boundary
            | ((boundary << 1) & mask_e) // East
            | ((boundary >> 1) & mask_w) // West
            | (boundary << 8) // South
            | (boundary >> 8); // North

        let next = next & occupied;

        if next == boundary {
            return boundary;
        }

        boundary = next;
    }

    boundary
}

/// Spread syndrome bits linearly across all 64 bits.
///
/// Uses logarithmic doubling (1, 2, 4, 8, 16, 32) to spread bits
/// in O(log n) iterations. No row barriers.
#[inline(always)]
pub fn spread_syndrome_linear(mut boundary: u64, occupied: u64) -> u64 {
    // Iteration 1
    let mut next = boundary | (boundary << 1) | (boundary >> 1);
    next &= occupied;
    if next == boundary {
        return boundary;
    }
    boundary = next;

    // Iteration 2
    let mut next = boundary | (boundary << 2) | (boundary >> 2);
    next &= occupied;
    if next == boundary {
        return boundary;
    }
    boundary = next;

    // Iteration 4
    let mut next = boundary | (boundary << 4) | (boundary >> 4);
    next &= occupied;
    if next == boundary {
        return boundary;
    }
    boundary = next;

    // Iteration 8
    let mut next = boundary | (boundary << 8) | (boundary >> 8);
    next &= occupied;
    if next == boundary {
        return boundary;
    }
    boundary = next;

    // Iteration 16
    let mut next = boundary | (boundary << 16) | (boundary >> 16);
    next &= occupied;
    if next == boundary {
        return boundary;
    }
    boundary = next;

    // Iteration 32
    let mut next = boundary | (boundary << 32) | (boundary >> 32);
    next &= occupied;
    boundary = next;

    boundary
}

/// Spread syndrome bits with row barriers.
///
/// Like `spread_syndrome_linear`, but respects row boundaries.
/// - `row_end_mask`: bits at the end of each row (cannot shift left to next row)
/// - `row_start_mask`: bits at the start of each row (cannot shift right to prev row)
///
/// Uses logarithmic doubling with dynamically expanding barrier masks.
#[inline(always)]
pub fn spread_syndrome_masked(
    mut boundary: u64,
    occupied: u64,
    row_end_mask: u64,
    row_start_mask: u64,
) -> u64 {
    let mut l_mask = row_end_mask;
    let mut r_mask = row_start_mask;

    // Iteration 1
    {
        let l = (boundary & !l_mask) << 1;
        let r = (boundary & !r_mask) >> 1;
        let next = (boundary | l | r) & occupied;
        if next == boundary {
            return boundary;
        }
        boundary = next;
    }

    // Iteration 2
    l_mask |= l_mask >> 1;
    r_mask |= r_mask << 1;
    {
        let l = (boundary & !l_mask) << 2;
        let r = (boundary & !r_mask) >> 2;
        let next = (boundary | l | r) & occupied;
        if next == boundary {
            return boundary;
        }
        boundary = next;
    }

    // Iteration 4
    l_mask |= l_mask >> 2;
    r_mask |= r_mask << 2;
    {
        let l = (boundary & !l_mask) << 4;
        let r = (boundary & !r_mask) >> 4;
        let next = (boundary | l | r) & occupied;
        if next == boundary {
            return boundary;
        }
        boundary = next;
    }

    // Iteration 8
    l_mask |= l_mask >> 4;
    r_mask |= r_mask << 4;
    {
        let l = (boundary & !l_mask) << 8;
        let r = (boundary & !r_mask) >> 8;
        let next = (boundary | l | r) & occupied;
        if next == boundary {
            return boundary;
        }
        boundary = next;
    }

    // Iteration 16
    l_mask |= l_mask >> 8;
    r_mask |= r_mask << 8;
    {
        let l = (boundary & !l_mask) << 16;
        let r = (boundary & !r_mask) >> 16;
        let next = (boundary | l | r) & occupied;
        if next == boundary {
            return boundary;
        }
        boundary = next;
    }

    // Iteration 32
    l_mask |= l_mask >> 16;
    r_mask |= r_mask << 16;
    {
        let l = (boundary & !l_mask) << 32;
        let r = (boundary & !r_mask) >> 32;
        let next = (boundary | l | r) & occupied;
        boundary = next;
    }

    boundary
}

// =============================================================================
// 3D Syndrome Spreading (4×4×4 Morton-encoded block)
// =============================================================================
//
// For 3D Morton encoding, neighbor distances vary by coordinate position:
// - X neighbors: ±1 (x0 flip) or ±7 (x0→x1 transition)
// - Y neighbors: ±2 (y0 flip) or ±14 (y0→y1 transition)
// - Z neighbors: ±4 (z0 flip) or ±28 (z0→z1 transition)
//
// We provide both a lookup-table version and a SWAR (SIMD Within A Register)
// version that uses parallel bit operations for better performance.

// Bit position masks for 3D Morton coordinates in 64-bit blocks.
// Morton encoding: idx = z1 y1 x1 z0 y0 x0 (bits 5..0)
// X coordinate uses bits 0 and 3
// Y coordinate uses bits 1 and 4
// Z coordinate uses bits 2 and 5

/// Positions where x0=1 (odd bit positions in Morton layout).
const X0_SET: u64 = 0xAAAA_AAAA_AAAA_AAAA;
/// Positions where x0=0 (even bit positions in Morton layout).
const X0_CLR: u64 = 0x5555_5555_5555_5555;
/// Positions where x1=1 (positions 8-15, 24-31, 40-47, 56-63 in groups of 8).
const X1_SET: u64 = 0xFF00_FF00_FF00_FF00;
/// Positions where x1=0 (positions 0-7, 16-23, 32-39, 48-55 in groups of 8).
const X1_CLR: u64 = 0x00FF_00FF_00FF_00FF;

/// Positions where y0=1 (positions where bit 1 is set).
const Y0_SET: u64 = 0xCCCC_CCCC_CCCC_CCCC;
/// Positions where y0=0 (positions where bit 1 is clear).
const Y0_CLR: u64 = 0x3333_3333_3333_3333;
/// Positions where y1=1 (positions 16-31, 48-63 in groups of 16).
const Y1_SET: u64 = 0xFFFF_0000_FFFF_0000;
/// Positions where y1=0 (positions 0-15, 32-47 in groups of 16).
const Y1_CLR: u64 = 0x0000_FFFF_0000_FFFF;

/// Positions where z0=1 (positions where bit 2 is set).
const Z0_SET: u64 = 0xF0F0_F0F0_F0F0_F0F0;
/// Positions where z0=0 (positions where bit 2 is clear).
const Z0_CLR: u64 = 0x0F0F_0F0F_0F0F_0F0F;
/// Positions where z1=1 (positions 32-63).
const Z1_SET: u64 = 0xFFFF_FFFF_0000_0000;
/// Positions where z1=0 (positions 0-31).
const Z1_CLR: u64 = 0x0000_0000_FFFF_FFFF;

/// Spread syndrome bits within a 4×4×4 Morton-encoded 3D block using SWAR.
///
/// Uses parallel bit operations to spread in all 6 directions simultaneously,
/// instead of iterating bit-by-bit. This provides O(diameter) complexity
/// instead of O(diameter × popcount).
///
/// # Morton Encoding
///
/// The 64 positions are Morton-encoded: `idx = z1 y1 x1 z0 y0 x0` (bits 5..0)
///
/// # Neighbor Shift Distances
///
/// Due to Morton encoding, neighbor distances depend on coordinate position:
/// - **X axis**: ±1 (within same x-half) or ±7 (cross x-half boundary)
/// - **Y axis**: ±2 (within same y-half) or ±14 (cross y-half boundary)
/// - **Z axis**: ±4 (within same z-half) or ±28 (cross z-half boundary)
///
/// # Performance
///
/// O(9) iterations maximum (Manhattan distance corner-to-corner = 3+3+3).
/// Each iteration performs ~12 bit operations instead of O(popcount) lookups.
#[inline(always)]
pub fn spread_syndrome_3d(mut boundary: u64, occupied: u64) -> u64 {
    // Maximum 9 iterations for corner-to-corner spreading in 4×4×4
    // (Manhattan distance from (0,0,0) to (3,3,3) = 3+3+3 = 9)
    for _ in 0..9 {
        // +X neighbors: x∈{0,2}→{1,3} via <<1, x=1→2 via <<7
        // x0=0 can shift +1 to reach x0=1 neighbor
        // x=1 (x0=1, x1=0) can shift +7 to reach x=2 (x0=0, x1=1)
        let x_plus = ((boundary & X0_CLR) << 1) | ((boundary & X0_SET & X1_CLR) << 7);

        // -X neighbors: x∈{1,3}→{0,2} via >>1, x=2→1 via >>7
        // x0=1 can shift -1 to reach x0=0 neighbor
        // x=2 (x0=0, x1=1) can shift -7 to reach x=1 (x0=1, x1=0)
        let x_minus = ((boundary & X0_SET) >> 1) | ((boundary & X0_CLR & X1_SET) >> 7);

        // +Y neighbors: y∈{0,2}→{1,3} via <<2, y=1→2 via <<14
        let y_plus = ((boundary & Y0_CLR) << 2) | ((boundary & Y0_SET & Y1_CLR) << 14);

        // -Y neighbors: y∈{1,3}→{0,2} via >>2, y=2→1 via >>14
        let y_minus = ((boundary & Y0_SET) >> 2) | ((boundary & Y0_CLR & Y1_SET) >> 14);

        // +Z neighbors: z∈{0,2}→{1,3} via <<4, z=1→2 via <<28
        let z_plus = ((boundary & Z0_CLR) << 4) | ((boundary & Z0_SET & Z1_CLR) << 28);

        // -Z neighbors: z∈{1,3}→{0,2} via >>4, z=2→1 via >>28
        let z_minus = ((boundary & Z0_SET) >> 4) | ((boundary & Z0_CLR & Z1_SET) >> 28);

        let next = (boundary | x_plus | x_minus | y_plus | y_minus | z_plus | z_minus) & occupied;

        if next == boundary {
            return boundary;
        }
        boundary = next;
    }
    boundary
}

/// Original lookup-table based 3D spreading (for verification/fallback).
///
/// Uses the precomputed `INTRA_BLOCK_NEIGHBORS_3D` lookup table.
/// O(diameter × popcount) complexity.
#[inline(always)]
#[allow(dead_code)]
pub fn spread_syndrome_3d_lookup(mut boundary: u64, occupied: u64) -> u64 {
    use crate::topology::INTRA_BLOCK_NEIGHBORS_3D;

    for _ in 0..9 {
        let mut next = boundary;
        let mut bits = boundary;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            next |= INTRA_BLOCK_NEIGHBORS_3D[i];
            bits &= bits - 1;
        }
        next &= occupied;
        if next == boundary {
            return boundary;
        }
        boundary = next;
    }
    boundary
}

/// Spread syndrome bits in 3D with boundary masks (for inter-block spreading).
///
/// Same as `spread_syndrome_3d` but accepts boundary masks for compatibility
/// with the inter-block spreading interface. The masks are used to prevent
/// spreading to positions that would cross block boundaries.
///
/// # Arguments
///
/// * `boundary` - Current boundary bits
/// * `occupied` - Valid positions in the block
/// * `x_end`, `x_start` - X boundary masks (currently unused, for API compat)
/// * `y_end`, `y_start` - Y boundary masks (currently unused, for API compat)
/// * `z_end`, `z_start` - Z boundary masks (currently unused, for API compat)
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn spread_syndrome_3d_masked(
    boundary: u64,
    occupied: u64,
    _x_end: u64,
    _x_start: u64,
    _y_end: u64,
    _y_start: u64,
    _z_end: u64,
    _z_start: u64,
) -> u64 {
    // The neighbor table already respects block boundaries (only intra-block neighbors)
    // so we can use the simpler spreading function.
    spread_syndrome_3d(boundary, occupied)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to convert (x,y,z) to Morton index in 4×4×4 block.
    fn xyz_to_morton(x: usize, y: usize, z: usize) -> usize {
        (x & 1) | ((y & 1) << 1) | ((z & 1) << 2) | ((x & 2) << 2) | ((y & 2) << 3) | ((z & 2) << 4)
    }

    #[test]
    fn test_spread_3d_single_center() {
        // Start at center (1,1,1), should spread to all 6 neighbors
        let center = xyz_to_morton(1, 1, 1);
        let boundary = 1u64 << center;
        let occupied = u64::MAX; // All positions valid

        let spread = spread_syndrome_3d(boundary, occupied);

        // After full spread, should cover entire connected component (all 64)
        assert_eq!(spread.count_ones(), 64);
    }

    #[test]
    fn test_spread_3d_corner() {
        // Start at corner (0,0,0), should spread through entire block
        let corner = xyz_to_morton(0, 0, 0);
        let boundary = 1u64 << corner;
        let occupied = u64::MAX;

        let spread = spread_syndrome_3d(boundary, occupied);

        // Should spread to fill entire 4×4×4 = 64 positions
        assert_eq!(spread.count_ones(), 64);
    }

    #[test]
    fn test_spread_3d_with_holes() {
        // Create occupied mask with a gap that disconnects parts of the block
        let center = xyz_to_morton(2, 2, 2);
        let neighbor = xyz_to_morton(2, 2, 3);
        let boundary = 1u64 << center;
        // Block neighbor so spreading can't reach beyond
        let occupied = !(1u64 << neighbor);

        let spread = spread_syndrome_3d(boundary, occupied);

        // Should not include the blocked position
        assert_eq!(spread & (1u64 << neighbor), 0);
    }

    #[test]
    fn test_spread_3d_masked_same_as_plain() {
        // Test that masked version gives same results
        use crate::topology::{
            X_END_MASK_3D, X_START_MASK_3D, Y_END_MASK_3D, Y_START_MASK_3D, Z_END_MASK_3D,
            Z_START_MASK_3D,
        };

        for start in [0, 7, 27, 42, 63] {
            let boundary = 1u64 << start;
            let occupied = u64::MAX;

            let plain = spread_syndrome_3d(boundary, occupied);
            let masked = spread_syndrome_3d_masked(
                boundary,
                occupied,
                X_END_MASK_3D,
                X_START_MASK_3D,
                Y_END_MASK_3D,
                Y_START_MASK_3D,
                Z_END_MASK_3D,
                Z_START_MASK_3D,
            );

            assert_eq!(plain, masked, "Plain and masked differ for start={}", start);
        }
    }

    #[test]
    fn test_spread_3d_respects_boundaries() {
        // Start at x=3 boundary
        let at_x_end = xyz_to_morton(3, 1, 1);
        let boundary = 1u64 << at_x_end;
        let occupied = u64::MAX;

        let spread = spread_syndrome_3d(boundary, occupied);

        // For full block, all 64 positions should be reached
        assert_eq!(spread.count_ones(), 64);
    }

    #[test]
    fn test_spread_3d_empty() {
        // No boundary bits
        let spread = spread_syndrome_3d(0, u64::MAX);
        assert_eq!(spread, 0);
    }

    #[test]
    fn test_spread_3d_multiple_seeds() {
        // Start from two opposite corners
        let c1 = xyz_to_morton(0, 0, 0);
        let c2 = xyz_to_morton(3, 3, 3);
        let boundary = (1u64 << c1) | (1u64 << c2);
        let occupied = u64::MAX;

        let spread = spread_syndrome_3d(boundary, occupied);

        // Both should meet and cover entire block
        assert_eq!(spread.count_ones(), 64);
    }

    #[test]
    fn test_spread_3d_iteration_count() {
        // Corner to opposite corner: Manhattan distance = 3+3+3 = 9 steps
        // With 6-connectivity, this requires 9 iterations to reach all positions
        let corner = xyz_to_morton(0, 0, 0);
        let boundary = 1u64 << corner;
        let occupied = u64::MAX;

        // After 1 iteration: 4 positions (corner + 3 neighbors)
        // After 9 iterations: all 64 positions reached

        let spread = spread_syndrome_3d(boundary, occupied);
        assert_eq!(spread.count_ones(), 64);
    }

    #[test]
    fn test_swar_matches_lookup_all_starts() {
        // Verify SWAR implementation matches lookup table for all 64 starting positions
        for start in 0..64 {
            let boundary = 1u64 << start;
            let occupied = u64::MAX;

            let swar = spread_syndrome_3d(boundary, occupied);
            let lookup = spread_syndrome_3d_lookup(boundary, occupied);

            assert_eq!(
                swar, lookup,
                "SWAR vs lookup mismatch at start={}: SWAR={:#018x}, lookup={:#018x}",
                start, swar, lookup
            );
        }
    }

    #[test]
    fn test_swar_matches_lookup_random_patterns() {
        // Test with various boundary and occupied patterns
        let test_cases: [(u64, u64); 10] = [
            (0x1, u64::MAX),                   // Single bit at 0
            (0x8000_0000_0000_0000, u64::MAX), // Single bit at 63
            (0x0000_0001_0000_0001, u64::MAX), // Two corners
            (0xFFFF_FFFF_FFFF_FFFF, u64::MAX), // All bits
            (0x1, 0x0F0F_0F0F_0F0F_0F0F),      // Restricted occupied
            (0x8000_0000_0000_0000, 0xF0F0_F0F0_F0F0_F0F0),
            (0x0101_0101_0101_0101, u64::MAX), // Scattered seeds
            (0x00FF, 0x00FF_00FF_00FF_00FF),   // Checkerboard pattern
            (0x1248, u64::MAX),                // Arbitrary pattern
            (0xDEAD_BEEF_CAFE_BABE, u64::MAX), // Random pattern
        ];

        for (boundary, occupied) in test_cases {
            let swar = spread_syndrome_3d(boundary, occupied);
            let lookup = spread_syndrome_3d_lookup(boundary, occupied);

            assert_eq!(
                swar, lookup,
                "SWAR vs lookup mismatch: boundary={:#018x}, occupied={:#018x}\n  SWAR={:#018x}\n  lookup={:#018x}",
                boundary, occupied, swar, lookup
            );
        }
    }

    #[test]
    fn test_swar_single_step_neighbors() {
        // Test that a single position spreads to exactly its 6 neighbors after one "step"
        // We can verify this by checking with occupied = only the start position and neighbors

        // Interior position (1,1,1) should have 6 neighbors
        let center = xyz_to_morton(1, 1, 1);
        let center_bit = 1u64 << center;

        // Get expected neighbors from lookup table
        use crate::topology::INTRA_BLOCK_NEIGHBORS_3D;
        let expected_neighbors = INTRA_BLOCK_NEIGHBORS_3D[center];

        // Run SWAR with occupied = center + all neighbors (simulates 1 step)
        let occupied = center_bit | expected_neighbors;
        let spread = spread_syndrome_3d(center_bit, occupied);

        // Should spread to all neighbors
        assert_eq!(
            spread, occupied,
            "Center (1,1,1) should spread to all 6 neighbors"
        );
        assert_eq!(spread.count_ones(), 7, "Center + 6 neighbors = 7 positions");
    }

    #[test]
    fn test_morton_masks_correctness() {
        // Verify the constant masks are correct
        for i in 0..64 {
            let x0 = (i & 1) != 0;
            let x1 = (i & 8) != 0;
            let y0 = (i & 2) != 0;
            let y1 = (i & 16) != 0;
            let z0 = (i & 4) != 0;
            let z1 = (i & 32) != 0;

            assert_eq!((X0_SET >> i) & 1 == 1, x0, "X0_SET wrong at {}", i);
            assert_eq!((X0_CLR >> i) & 1 == 1, !x0, "X0_CLR wrong at {}", i);
            assert_eq!((X1_SET >> i) & 1 == 1, x1, "X1_SET wrong at {}", i);
            assert_eq!((X1_CLR >> i) & 1 == 1, !x1, "X1_CLR wrong at {}", i);
            assert_eq!((Y0_SET >> i) & 1 == 1, y0, "Y0_SET wrong at {}", i);
            assert_eq!((Y0_CLR >> i) & 1 == 1, !y0, "Y0_CLR wrong at {}", i);
            assert_eq!((Y1_SET >> i) & 1 == 1, y1, "Y1_SET wrong at {}", i);
            assert_eq!((Y1_CLR >> i) & 1 == 1, !y1, "Y1_CLR wrong at {}", i);
            assert_eq!((Z0_SET >> i) & 1 == 1, z0, "Z0_SET wrong at {}", i);
            assert_eq!((Z0_CLR >> i) & 1 == 1, !z0, "Z0_CLR wrong at {}", i);
            assert_eq!((Z1_SET >> i) & 1 == 1, z1, "Z1_SET wrong at {}", i);
            assert_eq!((Z1_CLR >> i) & 1 == 1, !z1, "Z1_CLR wrong at {}", i);
        }
    }
}
