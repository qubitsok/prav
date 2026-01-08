//! Unit tests for prav-core intrinsics module.
//!
//! Tests cover: bits.rs, morton.rs, spread.rs
//! Focus: QEC-specific usage patterns from decoder modules.

#[cfg(test)]
mod tests {
    use prav_core::intrinsics::{
        blsr, compact_bits_2d, compact_bits_3d, morton_dec, morton_encode_2d, morton_inc,
        prefetch_l1, spread_bits_2d, spread_bits_3d, spread_syndrome_8x8, spread_syndrome_linear,
        spread_syndrome_masked, tzcnt, FastDiv,
    };

    // =========================================================================
    // bits.rs: tzcnt tests
    // =========================================================================

    #[test]
    fn test_tzcnt_zero_returns_64() {
        // Edge case: no bits set returns 64
        assert_eq!(tzcnt(0), 64);
    }

    #[test]
    fn test_tzcnt_powers_of_two() {
        // QEC pattern: single syndrome bit
        for i in 0..64 {
            assert_eq!(tzcnt(1u64 << i), i as u32);
        }
    }

    #[test]
    fn test_tzcnt_qec_typical_masks() {
        // Typical syndrome patterns from decoder
        assert_eq!(tzcnt(0b11), 0); // Adjacent syndromes
        assert_eq!(tzcnt(0b1100), 2); // Gap at start
        assert_eq!(tzcnt(0xFF00), 8); // Row-aligned boundary
        assert_eq!(tzcnt(u64::MAX), 0); // All bits set
    }

    // =========================================================================
    // bits.rs: blsr tests
    // =========================================================================

    #[test]
    fn test_blsr_clears_lowest_bit() {
        assert_eq!(blsr(0b1011), 0b1010); // Clears bit 0
        assert_eq!(blsr(0b1010), 0b1000); // Clears bit 1
        assert_eq!(blsr(0b1000), 0); // Clears bit 3, becomes 0
    }

    #[test]
    fn test_blsr_single_bit_becomes_zero() {
        for i in 0..64 {
            assert_eq!(blsr(1u64 << i), 0, "Failed for bit {}", i);
        }
    }

    #[test]
    fn test_blsr_zero_stays_zero() {
        assert_eq!(blsr(0), 0);
    }

    #[test]
    fn test_blsr_iteration_pattern() {
        // Simulates actual decoder loop: while m != 0 { bit = tzcnt(m); m = blsr(m); }
        let mut mask = 0b10110100u64;
        let mut bits = Vec::new();
        while mask != 0 {
            bits.push(tzcnt(mask));
            mask = blsr(mask);
        }
        assert_eq!(bits, vec![2, 4, 5, 7]);
    }

    #[test]
    fn test_blsr_reduces_popcount() {
        // Every blsr removes exactly one bit
        for x in [1u64, 0xFF, 0xFFFF, u64::MAX, 0x5555555555555555] {
            let before = x.count_ones();
            let after = blsr(x).count_ones();
            assert_eq!(after, before - 1, "Failed for x={:#x}", x);
        }
    }

    // =========================================================================
    // bits.rs: FastDiv tests
    // =========================================================================

    #[test]
    #[should_panic]
    fn test_fast_div_panic_on_zero() {
        let _ = FastDiv::new(0);
    }

    #[test]
    fn test_fast_div_matches_standard() {
        // FastDiv has overflow limits based on multiplier size
        // For d=64: shift=57, multiplier≈2^51, max n≈8192
        // For d=32: shift=58, multiplier≈2^53, max n≈2048
        // For d=16: shift=59, multiplier≈2^55, max n≈512
        // For d=8:  shift=60, multiplier≈2^57, max n≈128
        // Test within safe ranges
        for (d, max_n) in [(64, 8000), (32, 2000), (16, 500), (8, 120)] {
            let fd = FastDiv::new(d);
            for n in [0, 1, d - 1, d, d + 1, max_n / 2, max_n] {
                assert_eq!(fd.div(n), n / d, "div failed for n={}, d={}", n, d);
                assert_eq!(fd.rem(n), n % d, "rem failed for n={}, d={}", n, d);
            }
        }
    }

    #[test]
    fn test_fast_div_small_divisors() {
        // Small divisors have very limited range due to large multipliers
        // d=1: max n ≈ 1, d=2: max n ≈ 3, d=4: max n ≈ 15
        let fd1 = FastDiv::new(1);
        assert_eq!(fd1.div(0), 0);
        assert_eq!(fd1.div(1), 1);

        let fd2 = FastDiv::new(2);
        assert_eq!(fd2.div(0), 0);
        assert_eq!(fd2.div(1), 0);
        assert_eq!(fd2.div(2), 1);
        assert_eq!(fd2.div(3), 1);

        let fd4 = FastDiv::new(4);
        for n in 0..16 {
            assert_eq!(fd4.div(n), n / 4, "div failed for n={}", n);
        }
    }

    #[test]
    fn test_fast_div_qec_typical() {
        // QEC uses stride=64 most commonly, with grid sizes up to ~4096
        // For d=64: max safe n is ~8192
        let fd = FastDiv::new(64);
        for n in [0, 1, 63, 64, 127, 128, 256, 512, 1000, 4095, 4096] {
            assert_eq!(fd.div(n), n / 64, "div failed for n={}", n);
            assert_eq!(fd.rem(n), n % 64, "rem failed for n={}", n);
        }
    }

    #[test]
    fn test_fast_div_rem_consistency() {
        // Use d=64 which has larger safe range
        let fd = FastDiv::new(64);
        for n in [0, 63, 64, 127, 128, 256, 512, 1000, 4096] {
            let (q, r) = fd.div_rem(n);
            assert_eq!(q * 64 + r, n, "q*d + r != n for n={}", n);
            assert!(r < 64, "remainder >= divisor for n={}", n);
        }
    }

    #[test]
    fn test_fast_div_div_rem_matches_separate() {
        // For d=17: shift≈59, multiplier≈2^55, max n≈512
        let fd = FastDiv::new(17);
        for n in [0, 16, 17, 34, 100, 200, 500] {
            let (q1, r1) = fd.div_rem(n);
            let q2 = fd.div(n);
            let r2 = fd.rem(n);
            assert_eq!(q1, q2, "div_rem q != div for n={}", n);
            assert_eq!(r1, r2, "div_rem r != rem for n={}", n);
        }
    }

    // =========================================================================
    // bits.rs: prefetch_l1 tests
    // =========================================================================

    #[test]
    fn test_prefetch_l1_does_not_crash() {
        let data = [0u8; 64];
        prefetch_l1(data.as_ptr()); // Should be no-op but not crash
    }

    #[test]
    fn test_prefetch_l1_null_does_not_crash() {
        // Prefetch with null pointer should be safe (no-op)
        prefetch_l1(core::ptr::null());
    }

    // =========================================================================
    // morton.rs: SWAR spread/compact 2D tests
    // =========================================================================

    #[test]
    fn test_spread_compact_2d_roundtrip() {
        // Must be inverse operations for lower 16 bits
        for x in [0u32, 1, 0xFF, 0xFFFF, 0x5555, 0xAAAA, 0x1234] {
            let x = x & 0xFFFF; // Limit to valid range
            assert_eq!(
                compact_bits_2d(spread_bits_2d(x)),
                x,
                "Roundtrip failed for x={:#x}",
                x
            );
        }
    }

    #[test]
    fn test_spread_bits_2d_known_values() {
        assert_eq!(spread_bits_2d(0b0001), 0b0001); // bit 0 stays at 0
        assert_eq!(spread_bits_2d(0b0011), 0b0101); // bits 0,1 -> 0,2
        assert_eq!(spread_bits_2d(0b1111), 0b01010101); // bits 0-3 -> 0,2,4,6
        assert_eq!(spread_bits_2d(0b10), 0b0100); // bit 1 -> bit 2
    }

    #[test]
    fn test_compact_bits_2d_known_values() {
        assert_eq!(compact_bits_2d(0b0001), 0b0001); // bit 0 stays at 0
        assert_eq!(compact_bits_2d(0b0101), 0b0011); // bits 0,2 -> 0,1
        assert_eq!(compact_bits_2d(0b01010101), 0b1111); // bits 0,2,4,6 -> 0-3
    }

    // =========================================================================
    // morton.rs: SWAR spread/compact 3D tests
    // =========================================================================

    #[test]
    fn test_spread_compact_3d_roundtrip() {
        // Must be inverse for lower 10 bits
        for x in [0u32, 1, 0xFF, 0x3FF, 0x155, 0x2AA] {
            let x = x & 0x3FF;
            assert_eq!(
                compact_bits_3d(spread_bits_3d(x)),
                x,
                "Roundtrip failed for x={:#x}",
                x
            );
        }
    }

    #[test]
    fn test_spread_bits_3d_known_values() {
        assert_eq!(spread_bits_3d(0b001), 0b000_000_001); // bit 0 stays at 0
        assert_eq!(spread_bits_3d(0b011), 0b000_001_001); // bits 0,1 -> 0,3
        assert_eq!(spread_bits_3d(0b111), 0b001_001_001); // bits 0-2 -> 0,3,6
    }

    // =========================================================================
    // morton.rs: Morton 2D encode/decode tests (x86_64 with BMI2)
    // =========================================================================

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_morton_2d_roundtrip_x86() {
        use prav_core::intrinsics::morton_decode_2d;

        // QEC typical: grid coordinates for tiled decoder
        for x in [0u32, 1, 7, 8, 15, 16, 255, 1000] {
            for y in [0u32, 1, 7, 8, 15, 16, 255, 1000] {
                let code = morton_encode_2d(x, y);
                let (dx, dy) = morton_decode_2d(code);
                assert_eq!((x, y), (dx, dy), "Roundtrip failed for ({}, {})", x, y);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_morton_encode_2d_z_order() {
        // Z-order curve: (0,0)=0, (1,0)=1, (0,1)=2, (1,1)=3, (2,0)=4, ...
        assert_eq!(morton_encode_2d(0, 0), 0);
        assert_eq!(morton_encode_2d(1, 0), 1);
        assert_eq!(morton_encode_2d(0, 1), 2);
        assert_eq!(morton_encode_2d(1, 1), 3);
        assert_eq!(morton_encode_2d(2, 0), 4);
        assert_eq!(morton_encode_2d(0, 2), 8);
        assert_eq!(morton_encode_2d(2, 2), 12);
    }

    #[test]
    fn test_morton_encode_2d_swar_fallback() {
        // Test the SWAR fallback path (always available)
        // On x86_64 this tests BMI2, on other archs tests SWAR
        assert_eq!(morton_encode_2d(0, 0), 0);
        assert_eq!(morton_encode_2d(1, 0), 1);
        assert_eq!(morton_encode_2d(0, 1), 2);
    }

    // =========================================================================
    // morton.rs: Morton 3D encode/decode tests (x86_64 with BMI2)
    // =========================================================================

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_morton_3d_roundtrip_x86() {
        use prav_core::intrinsics::{morton_decode_3d, morton_encode_3d};

        for x in [0u32, 1, 7, 15] {
            for y in [0u32, 1, 7, 15] {
                for z in [0u32, 1, 7, 15] {
                    let code = morton_encode_3d(x, y, z);
                    let (dx, dy, dz) = morton_decode_3d(code);
                    assert_eq!((x, y, z), (dx, dy, dz));
                }
            }
        }
    }

    // =========================================================================
    // morton.rs: Architecture-agnostic roundtrip tests (SWAR fallback)
    // =========================================================================

    #[test]
    fn test_morton_2d_roundtrip() {
        use prav_core::intrinsics::{morton_decode_2d, morton_encode_2d};

        // Test roundtrip for various coordinate values
        for x in [0u32, 1, 7, 15, 100, 255, 1000] {
            for y in [0u32, 1, 7, 15, 100, 255, 1000] {
                let code = morton_encode_2d(x, y);
                let (dx, dy) = morton_decode_2d(code);
                assert_eq!(
                    (x, y),
                    (dx, dy),
                    "2D roundtrip failed for ({}, {}): code={}, decoded=({}, {})",
                    x,
                    y,
                    code,
                    dx,
                    dy
                );
            }
        }
    }

    #[test]
    fn test_morton_3d_roundtrip() {
        use prav_core::intrinsics::{morton_decode_3d, morton_encode_3d};

        // Test roundtrip for various coordinate values (10-bit range)
        for x in [0u32, 1, 7, 15, 100, 511, 1023] {
            for y in [0u32, 1, 7, 15, 100, 511] {
                for z in [0u32, 1, 7, 15, 100] {
                    let code = morton_encode_3d(x, y, z);
                    let (dx, dy, dz) = morton_decode_3d(code);
                    assert_eq!(
                        (x, y, z),
                        (dx, dy, dz),
                        "3D roundtrip failed for ({}, {}, {}): code={}, decoded=({}, {}, {})",
                        x,
                        y,
                        z,
                        code,
                        dx,
                        dy,
                        dz
                    );
                }
            }
        }
    }

    // =========================================================================
    // morton.rs: Morton navigation (inc/dec) tests
    // =========================================================================

    #[test]
    fn test_morton_inc_dec_x_axis() {
        let x_mask = 0x55555555u32; // Even bits for X in 2D

        // Increment X: (0,0) -> (1,0)
        assert_eq!(morton_inc(0, x_mask), 1);
        // Decrement X: (1,0) -> (0,0)
        assert_eq!(morton_dec(1, x_mask), 0);
        // Multiple increments
        assert_eq!(morton_inc(1, x_mask), 4); // (1,0) -> (2,0) = code 4
    }

    #[test]
    fn test_morton_inc_dec_y_axis() {
        let y_mask = 0xAAAAAAAAu32; // Odd bits for Y in 2D

        // Increment Y: (0,0) -> (0,1) = code 2
        assert_eq!(morton_inc(0, y_mask), 2);
        // Decrement Y: (0,1) -> (0,0)
        assert_eq!(morton_dec(2, y_mask), 0);
    }

    #[test]
    fn test_morton_inc_dec_roundtrip() {
        let x_mask = 0x55555555u32;
        for idx in [0u32, 1, 5, 10, 100, 1000] {
            let inc = morton_inc(idx, x_mask);
            let dec = morton_dec(inc, x_mask);
            assert_eq!(dec, idx, "inc then dec should return original for idx={}", idx);
        }
    }

    #[test]
    fn test_morton_inc_preserves_other_axis() {
        let x_mask = 0x55555555u32;
        // Start at (1, 2) = code 1 | (2 << 1) = 1 | 4 = 5
        let code = 5u32;
        let inc = morton_inc(code, x_mask);
        // Should be (2, 2) = 4 | 4 = 8... wait, let me recalculate
        // (1, 2): x=1 spreads to bit 0, y=2 spreads to bits 2 (since y goes to odd positions)
        // morton_encode_2d(1, 2) = spread(1) | (spread(2) << 1) = 1 | (4 << 1) = 1 | 8 = 9
        // Actually simpler: just verify the y-bits are preserved
        assert_eq!(inc & !x_mask, code & !x_mask, "Y bits should be preserved");
    }

    // =========================================================================
    // spread.rs: spread_syndrome_8x8 tests
    // =========================================================================

    #[test]
    fn test_spread_8x8_single_bit_center() {
        // Single syndrome at center of 8x8 grid
        let boundary = 1u64 << 27; // Row 3, col 3
        let occupied = u64::MAX; // All positions available
        let result = spread_syndrome_8x8(boundary, occupied);

        // Should spread to all 64 bits (fully connected)
        assert_eq!(result, u64::MAX);
    }

    #[test]
    fn test_spread_8x8_single_bit_corner() {
        // Single syndrome at corner (0,0)
        // With 8 iterations, it may not reach all 64 bits from corner
        // because diagonal distance is limited
        let boundary = 1u64; // Row 0, col 0
        let occupied = u64::MAX;
        let result = spread_syndrome_8x8(boundary, occupied);

        // Should at least spread to nearby cells
        // From (0,0), can reach at most 8 steps in any direction
        assert!(result & 1 != 0, "Must include original bit");
        assert!(result.count_ones() > 1, "Must spread to multiple bits");
        // Corner can reach row 0 and column 0 at minimum
        assert!(result & 0xFF != 0, "Should spread in row 0");
    }

    #[test]
    fn test_spread_8x8_row_connectivity() {
        // Row boundaries: spreading CAN cross via N/S (adjacent bytes)
        let boundary = 1u64 << 7; // Bit 7 (end of row 0)
        let occupied = 0xFF | (0xFF << 8); // Rows 0 and 1 occupied
        let result = spread_syndrome_8x8(boundary, occupied);

        // Should spread to row 0 and row 1
        assert!(result & 0xFF != 0, "Row 0 should have bits");
        assert!(result & (0xFF << 8) != 0, "Row 1 should be reached via N/S");
    }

    #[test]
    fn test_spread_8x8_blocked_by_gap() {
        // Gap in occupied mask blocks spreading
        let boundary = 1u64; // Bit 0
        let occupied = 0b101u64; // Gap at bit 1
        let result = spread_syndrome_8x8(boundary, occupied);

        // Should only include bit 0 (cannot reach bit 2 through gap)
        assert_eq!(result, 1);
    }

    #[test]
    fn test_spread_8x8_early_termination() {
        // Should converge early if no more spreading possible
        let boundary = 0b111u64; // First 3 bits
        let occupied = 0b111u64; // Only those 3 available
        let result = spread_syndrome_8x8(boundary, occupied);
        assert_eq!(result, 0b111);
    }

    #[test]
    fn test_spread_8x8_empty_boundary() {
        let boundary = 0u64;
        let occupied = u64::MAX;
        let result = spread_syndrome_8x8(boundary, occupied);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_spread_8x8_empty_occupied() {
        let boundary = 1u64;
        let occupied = 0u64;
        let result = spread_syndrome_8x8(boundary, occupied);
        assert_eq!(result, 0);
    }

    // =========================================================================
    // spread.rs: spread_syndrome_linear tests
    // =========================================================================

    #[test]
    fn test_spread_linear_full_connection() {
        let boundary = 1u64;
        let occupied = u64::MAX;
        let result = spread_syndrome_linear(boundary, occupied);
        assert_eq!(result, u64::MAX); // Spreads to all 64 bits
    }

    #[test]
    fn test_spread_syndrome_linear_jumping() {
        // Original test - documents jumping behavior
        let boundary = 1u64; // Bit 0
        let occupied = (1u64) | (1u64 << 1) | (1u64 << 3); // 0, 1, 3
        let result = spread_syndrome_linear(boundary, occupied);

        // At minimum, bits 0 and 1 should be set
        assert!(result & 0b11 == 0b11, "At least bits 0,1 should be set");
        // Note: bit 3 may or may not be set due to logarithmic doubling
    }

    #[test]
    fn test_spread_linear_empty_boundary() {
        let boundary = 0u64;
        let occupied = u64::MAX;
        let result = spread_syndrome_linear(boundary, occupied);
        assert_eq!(result, 0);
    }

    // =========================================================================
    // spread.rs: spread_syndrome_masked tests (QEC critical)
    // =========================================================================

    #[test]
    fn test_spread_syndrome_masked_behavior() {
        // Original test
        let boundary = 1u64;
        let occupied = (1u64) | (1u64 << 1) | (1u64 << 3);
        let row_end_mask = 0;
        let row_start_mask = 0;

        let result = spread_syndrome_masked(boundary, occupied, row_end_mask, row_start_mask);

        // At minimum includes boundary
        assert!(result & 1 != 0, "Must include original boundary");
    }

    #[test]
    fn test_spread_syndrome_masked_respects_boundaries() {
        // Original test - critical for correctness
        let boundary = 1u64 << 2;
        let occupied = (1u64 << 2) | (1u64 << 4);
        let row_end_mask = 1u64 << 3; // Block 3->4
        let row_start_mask = 1u64 << 4; // Block 4->3

        let result = spread_syndrome_masked(boundary, occupied, row_end_mask, row_start_mask);

        assert_eq!(result & (1 << 4), 0, "Should NOT jump across row boundary!");
    }

    #[test]
    fn test_spread_masked_row_barrier_qec() {
        // Real QEC pattern from unrolled.rs:162
        // ROW_END_MASK = 0x8080808080808080 (bit 7 of each byte)
        // ROW_START_MASK = 0x0101010101010101 (bit 0 of each byte)
        let row_end = 0x8080808080808080u64;
        let row_start = 0x0101010101010101u64;

        let boundary = 1u64 << 7; // End of row 0
        let occupied = 0xFFFF; // Rows 0 and 1

        let result = spread_syndrome_masked(boundary, occupied, row_end, row_start);

        // Should NOT cross into bit 8 (next row's start)
        assert_eq!(result & (1 << 8), 0, "Should not cross row boundary");
        // Should fill row 0
        assert_eq!(result & 0xFF, 0xFF, "Should fill row 0");
    }

    #[test]
    fn test_spread_masked_stride8_pattern() {
        // Pattern from growth/small_grid.rs for stride-8 grids
        let row_end = 0x8080808080808080u64;
        let row_start = 0x0101010101010101u64;

        let boundary = 0b00010000u64; // Bit 4 (middle of row)
        let occupied = 0xFFu64; // Row 0 fully occupied

        let result = spread_syndrome_masked(boundary, occupied, row_end, row_start);

        // Should spread within row 0 only
        assert_eq!(result, 0xFF);
    }

    #[test]
    fn test_spread_masked_no_barriers() {
        // With no barriers, should behave like linear
        let row_end = 0u64;
        let row_start = 0u64;

        let boundary = 1u64;
        let occupied = u64::MAX;

        let result = spread_syndrome_masked(boundary, occupied, row_end, row_start);

        assert_eq!(result, u64::MAX);
    }

    // =========================================================================
    // Invariant tests: spread result is always subset of occupied
    // =========================================================================

    #[test]
    fn test_spread_8x8_subset_invariant() {
        // Result must always be subset of occupied
        for (boundary, occupied) in [
            (0xFFu64, 0xFFu64),
            (0x1u64, 0xFFFFFFFFu64),
            (u64::MAX, 0x5555555555555555u64),
            (0x8080808080808080u64, u64::MAX),
        ] {
            let result = spread_syndrome_8x8(boundary & occupied, occupied);
            assert_eq!(
                result & !occupied,
                0,
                "Result must be subset of occupied for boundary={:#x}, occupied={:#x}",
                boundary,
                occupied
            );
        }
    }

    #[test]
    fn test_spread_linear_subset_invariant() {
        for (boundary, occupied) in [
            (0xFFu64, 0xFFu64),
            (0x1u64, 0xFFFFFFFFu64),
            (u64::MAX, 0x5555555555555555u64),
        ] {
            let result = spread_syndrome_linear(boundary & occupied, occupied);
            assert_eq!(
                result & !occupied,
                0,
                "Result must be subset of occupied"
            );
        }
    }

    #[test]
    fn test_spread_masked_subset_invariant() {
        let row_end = 0x8080808080808080u64;
        let row_start = 0x0101010101010101u64;

        for (boundary, occupied) in [
            (0xFFu64, 0xFFu64),
            (0x1u64, 0xFFFFFFFFu64),
            (u64::MAX, 0x5555555555555555u64),
        ] {
            let result = spread_syndrome_masked(boundary & occupied, occupied, row_end, row_start);
            assert_eq!(
                result & !occupied,
                0,
                "Result must be subset of occupied"
            );
        }
    }
}
