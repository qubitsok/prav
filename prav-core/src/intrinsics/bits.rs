// =============================================================================
// Bit Manipulation Intrinsics
// =============================================================================
//
// Low-level bit operations optimized for the target architecture.

#[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

/// Prefetch data into L1 cache.
///
/// This is a hint to the CPU that the data at `ptr` will be accessed soon.
/// Supported architectures:
/// - x86_64: Uses SSE prefetch intrinsic
/// - aarch64: Uses ARM prefetch intrinsic
/// - arm (armv7r): Uses PLD instruction via inline assembly
/// - wasm32 and others: No-op
#[inline(always)]
pub fn prefetch_l1(ptr: *const u8) {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    unsafe {
        _mm_prefetch::<_MM_HINT_T0>(ptr as *const i8);
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        // ARM64 prefetch: PRFM PLDL1KEEP
        // _prefetch(ptr, READ, LOCALITY3) maps to PLDL1KEEP
        core::arch::aarch64::_prefetch(
            ptr as *const i8,
            core::arch::aarch64::_PREFETCH_READ,
            core::arch::aarch64::_PREFETCH_LOCALITY3,
        );
    }

    #[cfg(all(target_arch = "arm", not(target_arch = "aarch64")))]
    unsafe {
        // ARMv7 PLD instruction for data prefetch into L1 cache
        core::arch::asm!("pld [{0}]", in(reg) ptr, options(readonly, nostack, preserves_flags));
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "sse"),
        target_arch = "aarch64",
        target_arch = "arm"
    )))]
    {
        // No-op for wasm32 and other architectures
        let _ = ptr;
    }
}

/// Count trailing zeros in a 64-bit value.
#[inline(always)]
pub fn tzcnt(x: u64) -> u32 {
    x.trailing_zeros()
}

/// Bit scan and reset: clears the lowest set bit.
///
/// Equivalent to `x & (x - 1)`.
#[inline(always)]
pub fn blsr(x: u64) -> u64 {
    x & (x.wrapping_sub(1))
}

/// Fast division using multiplication and shift.
///
/// Pre-computes constants for efficient repeated division by the same divisor.
#[derive(Clone, Copy, Debug)]
pub struct FastDiv {
    multiplier: u64,
    shift: u8,
    divisor: u32,
}

impl FastDiv {
    /// Create a new FastDiv for the given divisor.
    ///
    /// # Panics
    /// Panics if `d == 0`.
    pub fn new(d: u32) -> Self {
        assert!(d > 0);
        let s = d.leading_zeros();
        let shift = 32 + s;
        let multiplier = (1u64 << shift).div_ceil(d as u64);
        Self {
            multiplier,
            shift: shift as u8,
            divisor: d,
        }
    }

    /// Compute `n / divisor`.
    #[inline(always)]
    pub fn div(&self, n: u32) -> u32 {
        ((n as u64 * self.multiplier) >> self.shift) as u32
    }

    /// Compute `n % divisor`.
    #[inline(always)]
    pub fn rem(&self, n: u32) -> u32 {
        let q = self.div(n);
        n - q * self.divisor
    }

    /// Compute `(n / divisor, n % divisor)`.
    #[inline(always)]
    pub fn div_rem(&self, n: u32) -> (u32, u32) {
        let q = self.div(n);
        (q, n - q * self.divisor)
    }
}
