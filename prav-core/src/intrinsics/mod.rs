// =============================================================================
// Intrinsics Module
// =============================================================================
//
// Low-level bit manipulation, Morton encoding, and syndrome spreading operations
// optimized for the target architecture.

/// Bit manipulation: tzcnt, blsr, prefetch, FastDiv.
pub mod bits;

/// Morton encoding/decoding (Z-order curve).
pub mod morton;

/// Syndrome spreading for cluster growth.
pub mod spread;

/// Kani formal verification proofs.
#[cfg(kani)]
mod kani_proofs;

// =============================================================================
// Public Re-exports
// =============================================================================

// Bit operations
pub use bits::{blsr, prefetch_l1, tzcnt, FastDiv};

// Morton encoding (commonly needed for defect generation)
pub use morton::{
    compact_bits_2d, compact_bits_3d, morton_dec, morton_decode_2d, morton_decode_3d,
    morton_encode_2d, morton_encode_3d, morton_inc, spread_bits_2d, spread_bits_3d,
};

// Syndrome spreading
pub use spread::{spread_syndrome_8x8, spread_syndrome_linear, spread_syndrome_masked};
