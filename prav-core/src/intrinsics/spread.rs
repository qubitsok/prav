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
