//! Unit tests for arena allocation edge cases.
//!
//! These tests cover error paths and functionality not exercised by other tests.

use prav_core::arena::Arena;

/// Test that arena returns OOM error when allocation exceeds buffer size.
#[test]
fn test_arena_oom_error() {
    let mut memory = [0u8; 64];
    let mut arena = Arena::new(&mut memory);

    // Try to allocate more than buffer size
    let result = arena.alloc_slice::<u64>(100);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "OOM: Arena too small");
}

/// Test that arena returns OOM error when buffer is exhausted after multiple allocations.
#[test]
fn test_arena_oom_exhaustion() {
    let mut memory = [0u8; 128];
    let mut arena = Arena::new(&mut memory);

    // First allocation should succeed
    let first = arena.alloc_slice::<u64>(8);
    assert!(first.is_ok());

    // Second allocation should fail (64 bytes used, 64 remaining, but need 800)
    let second = arena.alloc_slice::<u64>(100);
    assert!(second.is_err());
    assert_eq!(second.unwrap_err(), "OOM: Arena too small");
}

/// Test that arena reset allows reallocation.
#[test]
fn test_arena_reset() {
    let mut memory = [0u8; 256];
    let mut arena = Arena::new(&mut memory);

    // Allocate some memory
    let _ = arena.alloc_slice::<u64>(10).unwrap();

    // This should fail (not enough space for another 200 u64s)
    let result = arena.alloc_slice::<u64>(200);
    assert!(result.is_err());

    // Reset the arena
    arena.reset();

    // Should succeed after reset (same allocation that worked before)
    let result = arena.alloc_slice::<u64>(10);
    assert!(result.is_ok());
}

/// Test that reset allows full reuse of buffer.
#[test]
fn test_arena_reset_full_reuse() {
    let mut memory = [0u8; 256];
    let mut arena = Arena::new(&mut memory);

    // Fill the arena
    let _ = arena.alloc_slice::<u64>(30).unwrap(); // 240 bytes

    // Should fail now
    let result = arena.alloc_slice::<u64>(10);
    assert!(result.is_err());

    // Reset
    arena.reset();

    // Should be able to allocate the same amount again
    let slice = arena.alloc_slice::<u64>(30).unwrap();
    assert_eq!(slice.len(), 30);
}

/// Test alloc_value functionality.
#[test]
fn test_arena_alloc_value() {
    let mut memory = [0u8; 256];
    let mut arena = Arena::new(&mut memory);

    let val = arena.alloc_value(42u64).unwrap();
    assert_eq!(*val, 42);

    // Modify the value
    *val = 100;
    assert_eq!(*val, 100);
}

/// Test alloc_value with different types.
#[test]
fn test_arena_alloc_value_types() {
    let mut memory = [0u8; 256];
    let mut arena = Arena::new(&mut memory);

    let u8_val = arena.alloc_value(255u8).unwrap();
    assert_eq!(*u8_val, 255);

    let u16_val = arena.alloc_value(65535u16).unwrap();
    assert_eq!(*u16_val, 65535);

    let u32_val = arena.alloc_value(0xDEADBEEFu32).unwrap();
    assert_eq!(*u32_val, 0xDEADBEEF);

    let u64_val = arena.alloc_value(0xCAFEBABE_DEADBEEF_u64).unwrap();
    assert_eq!(*u64_val, 0xCAFEBABE_DEADBEEF);
}

/// Test alignment edge cases when allocating after misaligned data.
#[test]
fn test_arena_alignment_edge_cases() {
    let mut memory = [0u8; 512];
    let mut arena = Arena::new(&mut memory);

    // Allocate u8 first (no alignment requirement)
    let _ = arena.alloc_slice::<u8>(1).unwrap();

    // Then u64 (should auto-align to 8 bytes)
    let slice = arena.alloc_slice::<u64>(1).unwrap();
    assert_eq!(slice.as_ptr() as usize % 8, 0);
}

/// Test custom alignment with alloc_slice_aligned.
#[test]
fn test_arena_custom_alignment() {
    let mut memory = [0u8; 512];
    let mut arena = Arena::new(&mut memory);

    // Allocate with 64-byte alignment (cache line)
    let slice = arena.alloc_slice_aligned::<u64>(4, 64).unwrap();
    assert_eq!(slice.as_ptr() as usize % 64, 0);
}

/// Test OOM due to alignment padding.
#[test]
fn test_arena_oom_due_to_alignment() {
    // Small buffer where alignment padding matters
    let mut memory = [0u8; 32];
    let mut arena = Arena::new(&mut memory);

    // Allocate 1 byte to create misalignment
    let _ = arena.alloc_slice::<u8>(1).unwrap();

    // Try to allocate with high alignment - may fail due to padding
    // Even though we have ~31 bytes left, alignment to 64 would require
    // significant padding that exceeds our buffer
    let result = arena.alloc_slice_aligned::<u64>(4, 64);
    // This should fail because 64-byte alignment + 32 bytes data > 31 bytes remaining
    assert!(result.is_err());
}

/// Test zero-length allocation.
#[test]
fn test_arena_zero_length_alloc() {
    let mut memory = [0u8; 64];
    let mut arena = Arena::new(&mut memory);

    // Zero-length allocation should succeed
    let slice = arena.alloc_slice::<u64>(0).unwrap();
    assert_eq!(slice.len(), 0);
}
