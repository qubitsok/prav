// =============================================================================
// Morton Encoding/Decoding (Z-order curve)
// =============================================================================
//
// Morton codes interleave coordinate bits for cache-efficient spatial locality.
// Uses BMI2 hardware instructions (PDEP/PEXT) on x86_64, with SWAR fallback.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::_pdep_u32;

// =============================================================================
// Bit Spreading/Compacting (SWAR operations)
// =============================================================================

/// Spread bits for 2D Morton encoding (SWAR fallback).
///
/// Spreads the lower 16 bits of x into even bit positions.
#[inline(always)]
pub fn spread_bits_2d(x: u32) -> u32 {
    let mut x = x & 0x0000FFFF;
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    x
}

/// Compact bits for 2D Morton decoding (SWAR fallback).
///
/// Compacts even bit positions into the lower 16 bits.
#[inline(always)]
pub fn compact_bits_2d(x: u32) -> u32 {
    let mut x = x & 0x55555555;
    x = (x | (x >> 1)) & 0x33333333;
    x = (x | (x >> 2)) & 0x0F0F0F0F;
    x = (x | (x >> 4)) & 0x00FF00FF;
    x = (x | (x >> 8)) & 0x0000FFFF;
    x
}

/// Spread bits for 3D Morton encoding.
///
/// Spreads the lower 10 bits into every third bit position.
#[inline(always)]
pub fn spread_bits_3d(x: u32) -> u32 {
    let mut x = x & 0x000003FF;
    x = (x | (x << 16)) & 0xFF0000FF;
    x = (x | (x << 8)) & 0x0300F00F;
    x = (x | (x << 4)) & 0x030C30C3;
    x = (x | (x << 2)) & 0x09249249;
    x
}

/// Compact bits for 3D Morton decoding.
///
/// Compacts every third bit position into the lower 10 bits.
#[inline(always)]
pub fn compact_bits_3d(x: u32) -> u32 {
    let mut x = x & 0x09249249;
    x = (x | (x >> 2)) & 0x030C30C3;
    x = (x | (x >> 4)) & 0x0300F00F;
    x = (x | (x >> 8)) & 0xFF0000FF;
    x = (x | (x >> 16)) & 0x000003FF;
    x
}

// =============================================================================
// Morton Encoding
// =============================================================================

/// Encode 2D coordinates (x, y) into a Morton code (Z-order).
///
/// Uses BMI2 PDEP instruction on x86_64, SWAR fallback on other architectures.
#[inline(always)]
pub fn morton_encode_2d(x: u32, y: u32) -> u32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        // 0x55555555 = 0b0101... (X positions)
        // 0xAAAAAAAA = 0b1010... (Y positions)
        _pdep_u32(x, 0x55555555) | _pdep_u32(y, 0xAAAAAAAA)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        // Fallback (SWAR)
        let mut x = x & 0x0000FFFF;
        x = (x | (x << 8)) & 0x00FF00FF;
        x = (x | (x << 4)) & 0x0F0F0F0F;
        x = (x | (x << 2)) & 0x33333333;
        x = (x | (x << 1)) & 0x55555555;

        let mut y = y & 0x0000FFFF;
        y = (y | (y << 8)) & 0x00FF00FF;
        y = (y | (y << 4)) & 0x0F0F0F0F;
        y = (y | (y << 2)) & 0x33333333;
        y = (y | (y << 1)) & 0x55555555;
        x | (y << 1)
    }
}

/// Encode 3D coordinates (x, y, z) into a Morton code.
///
/// Uses BMI2 PDEP instruction on x86_64.
#[inline(always)]
pub fn morton_encode_3d(_x: u32, _y: u32, _z: u32) -> u32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        _pdep_u32(_x, 0x09249249) | _pdep_u32(_y, 0x12492492) | _pdep_u32(_z, 0x24924924)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        spread_bits_3d(_x) | (spread_bits_3d(_y) << 1) | (spread_bits_3d(_z) << 2)
    }
}

// =============================================================================
// Morton Decoding
// =============================================================================

/// Decode a 2D Morton code back to (x, y) coordinates.
///
/// Uses BMI2 PEXT instruction on x86_64, SWAR fallback on other architectures.
#[inline(always)]
pub fn morton_decode_2d(_code: u32) -> (u32, u32) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::_pext_u32;
        let x = _pext_u32(_code, 0x55555555);
        let y = _pext_u32(_code, 0xAAAAAAAA);
        (x, y)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        (compact_bits_2d(_code), compact_bits_2d(_code >> 1))
    }
}

/// Decode a 3D Morton code back to (x, y, z) coordinates.
///
/// Uses BMI2 PEXT instruction on x86_64, SWAR fallback on other architectures.
#[inline(always)]
pub fn morton_decode_3d(_code: u32) -> (u32, u32, u32) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::_pext_u32;
        let x = _pext_u32(_code, 0x09249249);
        let y = _pext_u32(_code, 0x12492492);
        let z = _pext_u32(_code, 0x24924924);
        (x, y, z)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        (
            compact_bits_3d(_code),
            compact_bits_3d(_code >> 1),
            compact_bits_3d(_code >> 2),
        )
    }
}

// =============================================================================
// Morton Navigation
// =============================================================================

/// Increment a Morton-encoded coordinate along a specific axis.
///
/// `mask` specifies which bits belong to the axis (e.g., 0x55555555 for X in 2D).
#[inline(always)]
pub fn morton_inc(idx: u32, mask: u32) -> u32 {
    ((idx | !mask).wrapping_add(1) & mask) | (idx & !mask)
}

/// Decrement a Morton-encoded coordinate along a specific axis.
///
/// `mask` specifies which bits belong to the axis (e.g., 0x55555555 for X in 2D).
#[inline(always)]
pub fn morton_dec(idx: u32, mask: u32) -> u32 {
    ((idx & mask).wrapping_sub(1) & mask) | (idx & !mask)
}
