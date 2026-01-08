//! Property-based tests for prav-core intrinsics module.
//!
//! Uses proptest to verify mathematical properties and invariants
//! across random inputs.

use proptest::prelude::*;

proptest! {
    // =========================================================================
    // bits.rs properties
    // =========================================================================

    #[test]
    fn prop_blsr_reduces_popcount(x in 1u64..=u64::MAX) {
        use prav_core::intrinsics::blsr;

        // Critical for bit iteration: blsr always removes exactly one bit
        prop_assert_eq!(blsr(x).count_ones(), x.count_ones() - 1);
    }

    #[test]
    fn prop_tzcnt_finds_lowest_set_bit(x in 1u64..=u64::MAX) {
        use prav_core::intrinsics::tzcnt;

        let tz = tzcnt(x);
        prop_assert!(tz < 64);
        prop_assert!((x & (1u64 << tz)) != 0, "Bit at tz position must be set");
        if tz > 0 {
            prop_assert_eq!(x & ((1u64 << tz) - 1), 0, "All bits below tz must be zero");
        }
    }

    #[test]
    fn prop_blsr_clears_tzcnt_bit(x in 1u64..=u64::MAX) {
        use prav_core::intrinsics::{blsr, tzcnt};

        // The bit at tzcnt position should be cleared by blsr
        let tz = tzcnt(x);
        let cleared = blsr(x);
        prop_assert_eq!(cleared & (1u64 << tz), 0, "blsr must clear the lowest bit");
    }

    #[test]
    fn prop_fast_div_correctness(d in 64u32..256u32, n in 0u32..4096u32) {
        use prav_core::intrinsics::FastDiv;

        // QEC typical: stride values and node indices
        // Use d >= 64 to have safe range for n up to ~8000
        let fd = FastDiv::new(d);
        prop_assert_eq!(fd.div(n), n / d, "div must match standard division");
        prop_assert_eq!(fd.rem(n), n % d, "rem must match standard modulo");
    }

    #[test]
    fn prop_fast_div_rem_relation(d in 64u32..256u32, n in 0u32..4096u32) {
        use prav_core::intrinsics::FastDiv;

        let fd = FastDiv::new(d);
        let (q, r) = fd.div_rem(n);
        prop_assert_eq!(q * d + r, n, "q*d + r must equal n");
        prop_assert!(r < d, "remainder must be less than divisor");
    }

    // =========================================================================
    // morton.rs properties
    // =========================================================================

    #[test]
    fn prop_spread_compact_2d_inverse(x in 0u32..0xFFFFu32) {
        use prav_core::intrinsics::{compact_bits_2d, spread_bits_2d};

        // spread then compact must be identity for lower 16 bits
        prop_assert_eq!(compact_bits_2d(spread_bits_2d(x)), x);
    }

    #[test]
    fn prop_spread_compact_3d_inverse(x in 0u32..0x3FFu32) {
        use prav_core::intrinsics::{compact_bits_3d, spread_bits_3d};

        // spread then compact must be identity for lower 10 bits
        prop_assert_eq!(compact_bits_3d(spread_bits_3d(x)), x);
    }

    #[test]
    fn prop_spread_2d_preserves_bit_count(x in 0u32..0xFFFFu32) {
        use prav_core::intrinsics::spread_bits_2d;

        // Spreading should preserve the number of set bits
        prop_assert_eq!(spread_bits_2d(x).count_ones(), x.count_ones());
    }

    #[test]
    fn prop_spread_3d_preserves_bit_count(x in 0u32..0x3FFu32) {
        use prav_core::intrinsics::spread_bits_3d;

        // Spreading should preserve the number of set bits
        prop_assert_eq!(spread_bits_3d(x).count_ones(), x.count_ones());
    }

    #[test]
    fn prop_morton_2d_roundtrip(x in 0u32..0xFFFFu32, y in 0u32..0xFFFFu32) {
        use prav_core::intrinsics::{morton_decode_2d, morton_encode_2d};

        // Works on all architectures: BMI2 on x86_64, SWAR fallback elsewhere
        let code = morton_encode_2d(x, y);
        let (dx, dy) = morton_decode_2d(code);
        prop_assert_eq!((x, y), (dx, dy));
    }

    #[test]
    fn prop_morton_3d_roundtrip(x in 0u32..0x3FFu32, y in 0u32..0x3FFu32, z in 0u32..0x3FFu32) {
        use prav_core::intrinsics::{morton_decode_3d, morton_encode_3d};

        // Works on all architectures: BMI2 on x86_64, SWAR fallback elsewhere
        let code = morton_encode_3d(x, y, z);
        let (dx, dy, dz) = morton_decode_3d(code);
        prop_assert_eq!((x, y, z), (dx, dy, dz));
    }

    #[test]
    fn prop_morton_inc_dec_inverse(idx in 0u32..10000u32) {
        use prav_core::intrinsics::{morton_dec, morton_inc};

        let x_mask = 0x55555555u32; // X axis in 2D Morton code
        let inc = morton_inc(idx, x_mask);
        let dec = morton_dec(inc, x_mask);
        prop_assert_eq!(dec, idx);
    }

    #[test]
    fn prop_morton_inc_preserves_other_bits(idx in 0u32..10000u32) {
        use prav_core::intrinsics::morton_inc;

        let x_mask = 0x55555555u32;
        let inc = morton_inc(idx, x_mask);

        // Bits outside the mask should be preserved
        prop_assert_eq!(inc & !x_mask, idx & !x_mask);
    }

    // =========================================================================
    // spread.rs properties
    // =========================================================================

    #[test]
    fn prop_spread_linear_subset_of_occupied(boundary in any::<u64>(), occupied in any::<u64>()) {
        use prav_core::intrinsics::spread_syndrome_linear;

        // Critical safety: spreading cannot exceed occupied mask
        let result = spread_syndrome_linear(boundary & occupied, occupied);
        prop_assert_eq!(result & !occupied, 0, "Result must be subset of occupied");
    }

    #[test]
    fn prop_spread_8x8_subset_of_occupied(boundary in any::<u64>(), occupied in any::<u64>()) {
        use prav_core::intrinsics::spread_syndrome_8x8;

        let result = spread_syndrome_8x8(boundary & occupied, occupied);
        prop_assert_eq!(result & !occupied, 0, "Result must be subset of occupied");
    }

    #[test]
    fn prop_spread_masked_subset_of_occupied(
        boundary in any::<u64>(),
        occupied in any::<u64>(),
        row_end in any::<u64>(),
        row_start in any::<u64>()
    ) {
        use prav_core::intrinsics::spread_syndrome_masked;

        let result = spread_syndrome_masked(boundary & occupied, occupied, row_end, row_start);
        prop_assert_eq!(result & !occupied, 0, "Result must be subset of occupied");
    }

    #[test]
    fn prop_spread_linear_includes_boundary(boundary in any::<u64>(), occupied in any::<u64>()) {
        use prav_core::intrinsics::spread_syndrome_linear;

        let start = boundary & occupied;
        let result = spread_syndrome_linear(start, occupied);

        // Result must include all original boundary bits
        prop_assert_eq!(result & start, start, "Result must include boundary");
    }

    #[test]
    fn prop_spread_8x8_includes_boundary(boundary in any::<u64>(), occupied in any::<u64>()) {
        use prav_core::intrinsics::spread_syndrome_8x8;

        let start = boundary & occupied;
        let result = spread_syndrome_8x8(start, occupied);

        // Result must include all original boundary bits
        prop_assert_eq!(result & start, start, "Result must include boundary");
    }

    // NOTE: Both spread functions may not be idempotent in a single call.
    // They use bounded iterations (8 or 6) and may not reach fixed point.

    #[test]
    fn prop_spread_8x8_monotonic(boundary in any::<u64>(), occupied in any::<u64>()) {
        use prav_core::intrinsics::spread_syndrome_8x8;

        let start = boundary & occupied;
        let result1 = spread_syndrome_8x8(start, occupied);
        let result2 = spread_syndrome_8x8(result1, occupied);

        // Applying spread should never remove bits (monotonic)
        prop_assert_eq!(result2 & result1, result1, "8x8 spread should be monotonic");
        // Result should grow or stay the same
        prop_assert!(result2.count_ones() >= result1.count_ones(), "Spread should grow");
    }

    #[test]
    fn prop_spread_linear_monotonic(boundary in any::<u64>(), occupied in any::<u64>()) {
        use prav_core::intrinsics::spread_syndrome_linear;

        let start = boundary & occupied;
        let result1 = spread_syndrome_linear(start, occupied);
        let result2 = spread_syndrome_linear(result1, occupied);

        // Applying spread should never remove bits (monotonic)
        prop_assert_eq!(result2 & result1, result1, "Linear spread should be monotonic");
        // Result should grow or stay the same
        prop_assert!(result2.count_ones() >= result1.count_ones(), "Spread should grow");
    }
}
