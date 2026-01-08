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
// We use the neighbor lookup table for accurate 6-connected spreading.

/// Spread syndrome bits within a 4×4×4 Morton-encoded 3D block.
///
/// Uses the precomputed `INTRA_BLOCK_NEIGHBORS_3D` lookup table for
/// accurate 6-connected spreading. Iterates until convergence or
/// maximum diameter (9 steps for 4×4×4, Manhattan distance 3+3+3).
///
/// # Arguments
///
/// * `boundary` - Current boundary bits (defects to spread from)
/// * `occupied` - Valid positions in the block
///
/// # Performance
///
/// O(diameter × popcount) where diameter ≤ 9 for 4×4×4 block.
/// Early termination when no new positions are reached.
#[inline(always)]
pub fn spread_syndrome_3d(mut boundary: u64, occupied: u64) -> u64 {
    use crate::topology::INTRA_BLOCK_NEIGHBORS_3D;

    // Maximum 9 iterations for corner-to-corner spreading in 4×4×4
    // (Manhattan distance from (0,0,0) to (3,3,3) = 3+3+3 = 9)
    for _ in 0..9 {
        let mut next = boundary;
        let mut bits = boundary;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            next |= INTRA_BLOCK_NEIGHBORS_3D[i];
            bits &= bits - 1; // Clear lowest set bit
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
        let occupied = u64::MAX & !(1u64 << neighbor);

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
}
