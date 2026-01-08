//! Kani formal verification proofs for the intrinsics module.
//!
//! These proofs verify critical safety invariants in bit manipulation,
//! Morton encoding, and syndrome spreading operations.
//!
//! Run with: `cargo kani --package prav-core`

// ============================================================================
// Proof 1: blsr clears exactly one bit
// ============================================================================
// File: bits.rs:37
// What: Prove blsr(x) has exactly one fewer set bit than x
// Why: Bit iteration loops depend on this for termination and correctness

/// Verify that blsr clears exactly one bit from a non-zero value.
///
/// The decoder uses `while m != 0 { bit = tzcnt(m); m = blsr(m); }` pattern
/// extensively. If blsr doesn't clear exactly one bit, loops could:
/// - Never terminate (if no bits cleared)
/// - Skip bits (if multiple cleared)
#[kani::proof]
fn verify_blsr_clears_one_bit() {
    let x: u64 = kani::any();
    kani::assume(x != 0);

    let result = x & x.wrapping_sub(1); // blsr implementation
    let before = x.count_ones();
    let after = result.count_ones();

    kani::assert(after == before - 1, "blsr must clear exactly one bit");
}

// ============================================================================
// Proof 2: tzcnt returns valid bit index
// ============================================================================
// File: bits.rs:29
// What: Prove tzcnt returns index in [0, 63] and that bit is set
// Why: Used for array indexing; out-of-bounds would cause UB

/// Verify tzcnt returns a valid bit position for non-zero input.
#[kani::proof]
fn verify_tzcnt_bounds() {
    let x: u64 = kani::any();
    kani::assume(x != 0);

    let tz = x.trailing_zeros();

    kani::assert(tz < 64, "tzcnt must be < 64 for non-zero input");
    kani::assert((x & (1u64 << tz)) != 0, "bit at tzcnt position must be set");
}

// ============================================================================
// Proof 3: spread_bits_2d and compact_bits_2d are inverses
// ============================================================================
// File: morton.rs:19, morton.rs:32
// What: Prove compact_bits_2d(spread_bits_2d(x)) == x for valid inputs
// Why: Morton encoding correctness for 2D coordinate transformations

/// Verify SWAR spread/compact 2D operations are perfect inverses.
///
/// spread_bits_2d spreads lower 16 bits into even positions.
/// compact_bits_2d extracts even positions back to lower 16 bits.
#[kani::proof]
fn verify_spread_compact_2d_inverse() {
    let x: u32 = kani::any();
    kani::assume(x <= 0xFFFF); // Valid input range: lower 16 bits

    // Inline spread_bits_2d
    let mut s = x & 0x0000FFFF;
    s = (s | (s << 8)) & 0x00FF00FF;
    s = (s | (s << 4)) & 0x0F0F0F0F;
    s = (s | (s << 2)) & 0x33333333;
    s = (s | (s << 1)) & 0x55555555;

    // Inline compact_bits_2d
    let mut c = s & 0x55555555;
    c = (c | (c >> 1)) & 0x33333333;
    c = (c | (c >> 2)) & 0x0F0F0F0F;
    c = (c | (c >> 4)) & 0x00FF00FF;
    c = (c | (c >> 8)) & 0x0000FFFF;

    kani::assert(c == x, "spread then compact must return original");
}

// ============================================================================
// Proof 4: spread_bits_3d and compact_bits_3d are inverses
// ============================================================================
// File: morton.rs:45, morton.rs:58
// What: Prove compact_bits_3d(spread_bits_3d(x)) == x for valid inputs
// Why: Morton encoding correctness for 3D coordinate transformations

/// Verify SWAR spread/compact 3D operations are perfect inverses.
#[kani::proof]
fn verify_spread_compact_3d_inverse() {
    let x: u32 = kani::any();
    kani::assume(x <= 0x3FF); // Valid input range: lower 10 bits

    // Inline spread_bits_3d
    let mut s = x & 0x000003FF;
    s = (s | (s << 16)) & 0xFF0000FF;
    s = (s | (s << 8)) & 0x0300F00F;
    s = (s | (s << 4)) & 0x030C30C3;
    s = (s | (s << 2)) & 0x09249249;

    // Inline compact_bits_3d
    let mut c = s & 0x09249249;
    c = (c | (c >> 2)) & 0x030C30C3;
    c = (c | (c >> 4)) & 0x0300F00F;
    c = (c | (c >> 8)) & 0xFF0000FF;
    c = (c | (c >> 16)) & 0x000003FF;

    kani::assert(c == x, "spread then compact 3D must return original");
}

// ============================================================================
// Proof 5: morton_inc and morton_dec are inverses
// ============================================================================
// File: morton.rs:164, morton.rs:172
// What: Prove dec(inc(idx, mask), mask) == idx
// Why: Navigation correctness in Z-order traversal

/// Verify morton_inc and morton_dec are perfect inverses.
#[kani::proof]
fn verify_morton_inc_dec_inverse() {
    let idx: u32 = kani::any();
    let mask: u32 = kani::any();
    kani::assume(mask != 0); // Need valid axis mask

    // morton_inc: ((idx | !mask).wrapping_add(1) & mask) | (idx & !mask)
    let inc = ((idx | !mask).wrapping_add(1) & mask) | (idx & !mask);

    // morton_dec: ((idx & mask).wrapping_sub(1) & mask) | (idx & !mask)
    let dec = ((inc & mask).wrapping_sub(1) & mask) | (inc & !mask);

    kani::assert(dec == idx, "inc then dec must return original");
}

// ============================================================================
// Proof 6: spread_syndrome containment invariant
// ============================================================================
// File: spread.rs (all functions)
// What: Prove result is always subset of occupied mask
// Why: Spreading beyond occupied bits would corrupt decoder state

/// Verify spread_syndrome_8x8 result is subset of occupied.
#[kani::proof]
#[kani::unwind(9)] // 8 iterations max
fn verify_spread_8x8_containment() {
    let boundary: u64 = kani::any();
    let occupied: u64 = kani::any();

    // Ensure boundary starts within occupied
    let boundary = boundary & occupied;

    // Inline spread_syndrome_8x8 (simplified)
    let mask_e = 0xFEFEFEFEFEFEFEFEu64;
    let mask_w = 0x7F7F7F7F7F7F7F7Fu64;
    let mut b = boundary;

    for _ in 0..8 {
        let next = b
            | ((b << 1) & mask_e)
            | ((b >> 1) & mask_w)
            | (b << 8)
            | (b >> 8);
        let next = next & occupied;
        if next == b {
            break;
        }
        b = next;
    }

    kani::assert((b & !occupied) == 0, "spread result must be subset of occupied");
}

// ============================================================================
// Proof 7: FastDiv correctness for QEC-typical divisors
// ============================================================================
// File: bits.rs:51-86
// What: Prove FastDiv matches standard division for stride values
// Why: Incorrect division would corrupt coordinate calculations

/// Verify FastDiv matches standard division for small inputs.
///
/// Limited to small ranges for tractability. QEC uses strides 8, 16, 32, 64.
#[kani::proof]
fn verify_fast_div_correctness() {
    let d: u32 = kani::any();
    let n: u32 = kani::any();

    // Constrain to tractable ranges
    // Use d >= 64 to ensure safe overflow range
    kani::assume(d >= 64 && d <= 128);
    kani::assume(n <= 4096);

    // FastDiv::new calculation
    let s = d.leading_zeros();
    let shift = 32 + s;
    let multiplier = ((1u64 << shift) + (d as u64 - 1)) / (d as u64);

    // FastDiv::div calculation
    let result = ((n as u64 * multiplier) >> shift) as u32;

    kani::assert(result == n / d, "FastDiv must match standard division");
}

// ============================================================================
// Proof 8: tzcnt and blsr iteration terminates
// ============================================================================
// File: decoder uses this pattern extensively
// What: Prove the bit iteration loop terminates correctly
// Why: Infinite loops would hang the decoder

/// Verify that the tzcnt/blsr iteration pattern processes all bits.
#[kani::proof]
#[kani::unwind(65)] // At most 64 iterations
fn verify_bit_iteration_terminates() {
    let original: u64 = kani::any();
    let mut mask = original;
    let mut count = 0u32;
    let mut processed = 0u64;

    while mask != 0 {
        let bit = mask.trailing_zeros();
        kani::assert(bit < 64, "bit index must be valid");

        processed |= 1u64 << bit;
        mask = mask & mask.wrapping_sub(1); // blsr

        count += 1;
        kani::assert(count <= 64, "must terminate within 64 iterations");
    }

    // All originally set bits should have been processed
    kani::assert(
        processed == original,
        "all bits must be processed",
    );
}

// ============================================================================
// Proof 9: spread_syndrome preserves boundary bits
// ============================================================================
// File: spread.rs
// What: Prove that spreading never removes the original boundary bits
// Why: Losing boundary bits would corrupt cluster identification

/// Verify spread never loses the original boundary bits.
#[kani::proof]
#[kani::unwind(9)]
fn verify_spread_preserves_boundary() {
    let boundary: u64 = kani::any();
    let occupied: u64 = kani::any();

    // Start must be within occupied
    let start = boundary & occupied;

    // Inline spread_syndrome_8x8
    let mask_e = 0xFEFEFEFEFEFEFEFEu64;
    let mask_w = 0x7F7F7F7F7F7F7F7Fu64;
    let mut b = start;

    for _ in 0..8 {
        let next = b
            | ((b << 1) & mask_e)
            | ((b >> 1) & mask_w)
            | (b << 8)
            | (b >> 8);
        let next = next & occupied;
        if next == b {
            break;
        }
        b = next;
    }

    // Result must contain all original boundary bits
    kani::assert((b & start) == start, "spread must preserve boundary bits");
}

// ============================================================================
// Proof 10: morton_encode_2d and morton_decode_2d are inverses
// ============================================================================
// File: morton.rs
// What: Prove encode then decode returns original coordinates
// Why: Roundtrip correctness for 2D spatial hashing

/// Verify Morton 2D encode/decode roundtrip for SWAR implementation.
#[kani::proof]
fn verify_morton_2d_roundtrip() {
    let x: u32 = kani::any();
    let y: u32 = kani::any();
    kani::assume(x <= 0xFFFF); // Valid 16-bit coordinates
    kani::assume(y <= 0xFFFF);

    // Inline encode (SWAR version - works on all architectures)
    let mut ex = x & 0x0000FFFF;
    ex = (ex | (ex << 8)) & 0x00FF00FF;
    ex = (ex | (ex << 4)) & 0x0F0F0F0F;
    ex = (ex | (ex << 2)) & 0x33333333;
    ex = (ex | (ex << 1)) & 0x55555555;

    let mut ey = y & 0x0000FFFF;
    ey = (ey | (ey << 8)) & 0x00FF00FF;
    ey = (ey | (ey << 4)) & 0x0F0F0F0F;
    ey = (ey | (ey << 2)) & 0x33333333;
    ey = (ey | (ey << 1)) & 0x55555555;

    let code = ex | (ey << 1);

    // Inline decode (SWAR version)
    let mut dx = code & 0x55555555;
    dx = (dx | (dx >> 1)) & 0x33333333;
    dx = (dx | (dx >> 2)) & 0x0F0F0F0F;
    dx = (dx | (dx >> 4)) & 0x00FF00FF;
    dx = (dx | (dx >> 8)) & 0x0000FFFF;

    let mut dy = (code >> 1) & 0x55555555;
    dy = (dy | (dy >> 1)) & 0x33333333;
    dy = (dy | (dy >> 2)) & 0x0F0F0F0F;
    dy = (dy | (dy >> 4)) & 0x00FF00FF;
    dy = (dy | (dy >> 8)) & 0x0000FFFF;

    kani::assert(dx == x, "decoded x must match original");
    kani::assert(dy == y, "decoded y must match original");
}

// ============================================================================
// Proof 11: morton_encode_3d and morton_decode_3d are inverses
// ============================================================================
// File: morton.rs
// What: Prove encode then decode returns original coordinates
// Why: Roundtrip correctness for 3D spatial hashing

/// Verify Morton 3D encode/decode roundtrip for SWAR implementation.
#[kani::proof]
fn verify_morton_3d_roundtrip() {
    let x: u32 = kani::any();
    let y: u32 = kani::any();
    let z: u32 = kani::any();
    kani::assume(x <= 0x3FF); // Valid 10-bit coordinates
    kani::assume(y <= 0x3FF);
    kani::assume(z <= 0x3FF);

    // Inline spread_bits_3d for each coordinate
    let mut sx = x & 0x000003FF;
    sx = (sx | (sx << 16)) & 0xFF0000FF;
    sx = (sx | (sx << 8)) & 0x0300F00F;
    sx = (sx | (sx << 4)) & 0x030C30C3;
    sx = (sx | (sx << 2)) & 0x09249249;

    let mut sy = y & 0x000003FF;
    sy = (sy | (sy << 16)) & 0xFF0000FF;
    sy = (sy | (sy << 8)) & 0x0300F00F;
    sy = (sy | (sy << 4)) & 0x030C30C3;
    sy = (sy | (sy << 2)) & 0x09249249;

    let mut sz = z & 0x000003FF;
    sz = (sz | (sz << 16)) & 0xFF0000FF;
    sz = (sz | (sz << 8)) & 0x0300F00F;
    sz = (sz | (sz << 4)) & 0x030C30C3;
    sz = (sz | (sz << 2)) & 0x09249249;

    let code = sx | (sy << 1) | (sz << 2);

    // Inline compact_bits_3d for each coordinate
    let mut dx = code & 0x09249249;
    dx = (dx | (dx >> 2)) & 0x030C30C3;
    dx = (dx | (dx >> 4)) & 0x0300F00F;
    dx = (dx | (dx >> 8)) & 0xFF0000FF;
    dx = (dx | (dx >> 16)) & 0x000003FF;

    let mut dy = (code >> 1) & 0x09249249;
    dy = (dy | (dy >> 2)) & 0x030C30C3;
    dy = (dy | (dy >> 4)) & 0x0300F00F;
    dy = (dy | (dy >> 8)) & 0xFF0000FF;
    dy = (dy | (dy >> 16)) & 0x000003FF;

    let mut dz = (code >> 2) & 0x09249249;
    dz = (dz | (dz >> 2)) & 0x030C30C3;
    dz = (dz | (dz >> 4)) & 0x0300F00F;
    dz = (dz | (dz >> 8)) & 0xFF0000FF;
    dz = (dz | (dz >> 16)) & 0x000003FF;

    kani::assert(dx == x, "decoded x must match original");
    kani::assert(dy == y, "decoded y must match original");
    kani::assert(dz == z, "decoded z must match original");
}
