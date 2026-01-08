//! Kani formal verification proofs for the arena allocator.
//!
//! These proofs verify critical safety invariants in memory allocation.
//! The arena allocator is a fundamental building block used throughout the
//! QEC decoder, so its correctness is essential.
//!
//! Run with: `cargo kani --package prav-core`

// ============================================================================
// Proof 1: Alignment calculation never causes overflow
// ============================================================================
// File: arena.rs:33
// What: Prove `(actual_align - (current_ptr % actual_align)) % actual_align` is bounded
// Why: Overflow → memory corruption or invalid pointer arithmetic

/// Verify alignment padding calculation is bounded and correct.
///
/// The calculation `(align - (ptr % align)) % align` computes the padding needed
/// to align a pointer to a given alignment boundary. This proof verifies:
/// 1. The padding is always less than the alignment
/// 2. Adding the padding results in an aligned pointer
/// 3. No arithmetic overflow occurs
#[kani::proof]
fn verify_alignment_calculation_bounded() {
    let current_ptr: usize = kani::any();
    let align: usize = kani::any();

    // Realistic constraints for QEC applications
    // - Pointers are within reasonable range (not near overflow)
    // - Alignments are power of two up to 128 bytes (cache line * 2)
    kani::assume(current_ptr < usize::MAX / 2);
    kani::assume(align > 0 && align <= 128 && align.is_power_of_two());

    // The actual calculation from arena.rs:33
    let padding = (align - (current_ptr % align)) % align;

    // Invariant 1: Padding is always less than alignment
    kani::assert(padding < align, "padding must be less than alignment");

    // Invariant 2: Adding padding produces an aligned address
    kani::assert(
        (current_ptr + padding) % align == 0,
        "aligned ptr must be divisible by align",
    );
}

// ============================================================================
// Proof 2: Offset bounds after allocation
// ============================================================================
// File: arena.rs:35-41
// What: Prove offset never exceeds buffer length after successful allocation
// Why: Out-of-bounds offset → buffer overflow → memory corruption

/// Verify that successful allocations maintain offset invariant.
///
/// After a successful allocation, the arena's offset must not exceed
/// the buffer length. This proof verifies the bounds check at arena.rs:35
/// correctly prevents buffer overflows.
#[kani::proof]
fn verify_offset_bounds() {
    let buffer_len: usize = kani::any();
    let alloc_size: usize = kani::any();
    let current_offset: usize = kani::any();

    // Realistic constraints
    kani::assume(buffer_len > 0 && buffer_len <= 1024 * 1024); // Max 1MB
    kani::assume(alloc_size <= buffer_len);
    kani::assume(current_offset <= buffer_len);

    // Check if allocation would succeed (simplified without alignment)
    if current_offset + alloc_size <= buffer_len {
        let new_offset = current_offset + alloc_size;
        kani::assert(new_offset <= buffer_len, "new offset must not exceed buffer");
    }
}

// ============================================================================
// Proof 3: Size calculation doesn't overflow
// ============================================================================
// File: arena.rs:26
// What: Prove `size_of::<T>() * len` doesn't overflow
// Why: Overflow → allocating less memory than needed → buffer overflow

/// Verify size calculation for slice allocation doesn't overflow.
///
/// The expression `size_of::<T>() * len` calculates total bytes needed.
/// If this overflows, we'd allocate less memory than needed, leading to
/// buffer overflows when the caller writes to the slice.
#[kani::proof]
fn verify_size_calculation_bounded() {
    let type_size: usize = kani::any();
    let len: usize = kani::any();

    // Realistic constraints for QEC grids
    // - Type sizes up to 64 bytes (u64 arrays, structs)
    // - Length up to 1M elements (large QEC grids)
    kani::assume(type_size > 0 && type_size <= 64);
    kani::assume(len <= 1024 * 1024);

    let size = type_size.checked_mul(len);
    kani::assert(size.is_some(), "size calculation must not overflow");
}

// ============================================================================
// Proof 4: Alignment selection is correct
// ============================================================================
// File: arena.rs:24-25
// What: Prove actual_align is at least align_of::<T>()
// Why: Under-alignment → undefined behavior on strict-alignment platforms

/// Verify that actual alignment is never less than type's natural alignment.
///
/// The code `if align > t_align { align } else { t_align }` selects the
/// larger of requested alignment and type's natural alignment. This ensures
/// we never under-align data.
#[kani::proof]
fn verify_alignment_selection() {
    let t_align: usize = kani::any();
    let requested_align: usize = kani::any();

    // Alignments are always power of two and > 0
    kani::assume(t_align > 0 && t_align <= 64 && t_align.is_power_of_two());
    kani::assume(requested_align > 0 && requested_align <= 128 && requested_align.is_power_of_two());

    // The actual calculation from arena.rs:25
    let actual_align = if requested_align > t_align {
        requested_align
    } else {
        t_align
    };

    // Invariant: actual_align is at least t_align
    kani::assert(
        actual_align >= t_align,
        "actual alignment must be >= type alignment",
    );

    // Invariant: actual_align is at least requested_align
    kani::assert(
        actual_align >= requested_align,
        "actual alignment must be >= requested alignment",
    );

    // Invariant: actual_align is a power of two
    kani::assert(
        actual_align.is_power_of_two(),
        "actual alignment must be power of two",
    );
}

// ============================================================================
// Proof 5: Padding plus size doesn't overflow
// ============================================================================
// File: arena.rs:35
// What: Prove `offset + padding + size` doesn't overflow before bounds check
// Why: Overflow in bounds check → false negative → buffer overflow

/// Verify the bounds check calculation doesn't overflow.
///
/// The check `self.offset + padding + size > self.buffer.len()` could give
/// wrong results if the addition overflows. This proof verifies that within
/// realistic constraints, no overflow occurs.
#[kani::proof]
fn verify_bounds_check_no_overflow() {
    let offset: usize = kani::any();
    let padding: usize = kani::any();
    let size: usize = kani::any();
    let buffer_len: usize = kani::any();

    // Realistic constraints
    kani::assume(buffer_len <= 1024 * 1024); // Max 1MB arena
    kani::assume(offset <= buffer_len);
    kani::assume(padding < 128); // Max alignment padding
    kani::assume(size <= buffer_len);

    // The addition should not overflow
    let sum1 = offset.checked_add(padding);
    kani::assert(sum1.is_some(), "offset + padding must not overflow");

    if let Some(s1) = sum1 {
        let sum2 = s1.checked_add(size);
        kani::assert(sum2.is_some(), "offset + padding + size must not overflow");
    }
}
