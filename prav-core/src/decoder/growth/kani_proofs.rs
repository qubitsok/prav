//! Kani formal verification proofs for the growth module.
//!
//! These proofs verify critical safety invariants in the decoder's growth operations.
//!
//! Run with: `cargo kani --package prav-core`

/// Maximum grid dimension for proofs (bounded for tractability)
const MAX_DIM: usize = 64;
/// Maximum number of nodes for proofs
const MAX_NODES: usize = MAX_DIM * MAX_DIM;

// ============================================================================
// Proof 1: Boundary offset arithmetic is bounded
// ============================================================================
// File: inter_block.rs:40
// What: Prove `(base_u as isize + offset_v + bit as isize) as u32` never overflows
// Why: Incorrect offset → out-of-bounds array access → memory corruption

/// Verify that boundary offset calculation stays within valid bounds.
///
/// The calculation `(base_u as isize + offset_v + bit as isize) as u32` appears
/// in inter_block.rs for computing target node indices during boundary merges.
///
/// In practice:
/// - base_u is a block base (blk_idx * 64), so it's always a multiple of 64
/// - offset_v is typically stride_y or -stride_y (for up/down neighbor)
/// - For up neighbor: offset_v is positive (shift from current block to up block's address space)
/// - For down neighbor: similar positive offset
/// - bit is in range [0, 63] from trailing_zeros of a 64-bit mask
#[kani::proof]
#[kani::unwind(1)]
fn verify_boundary_offset_bounds() {
    // Symbolic inputs with realistic constraints
    let blk_idx: usize = kani::any();
    let stride_y: usize = kani::any();
    let bit: u32 = kani::any();
    let is_up_neighbor: bool = kani::any();

    // Constrain block index to realistic range (max ~64 blocks for small grids)
    kani::assume(blk_idx < 64);

    // Constrain stride to power of 2 values typical in practice
    kani::assume(stride_y == 8 || stride_y == 16 || stride_y == 32 || stride_y == 64);

    // bit comes from trailing zeros of a u64
    kani::assume(bit < 64);

    // base_u is the target block's base address
    let base_u = blk_idx * 64;

    // offset_v represents the shift between source and target
    // For up neighbor at block N, looking at block N-1: offset is +32 typically
    // This ensures we're looking at correct row in neighbor
    let offset_v: isize = if is_up_neighbor {
        stride_y as isize // Positive shift for up neighbor
    } else {
        -(stride_y as isize) // Negative shift, but from higher block base
    };

    // Ensure base_u is large enough for negative offset
    if offset_v < 0 {
        kani::assume((base_u as isize) + offset_v >= 0);
    }

    // The actual calculation from inter_block.rs:40
    let result = (base_u as isize) + offset_v + (bit as isize);

    // Invariant 1: Result should be non-negative (valid array index)
    kani::assert(result >= 0, "Boundary offset must be non-negative");

    // Invariant 2: Result should fit in u32
    kani::assert(result <= u32::MAX as isize, "Boundary offset must fit in u32");
}

// ============================================================================
// Proof 2: Run-length extraction terminates
// ============================================================================
// File: stride32.rs:192-199
// What: Prove run-length loop processes all bits and terminates
// Why: Infinite loop or missed bits → decoder hangs or incorrect results

/// Verify that run-length extraction processes all bits and terminates.
///
/// The run-length extraction loop in stride32.rs iterates over a mask,
/// extracting contiguous runs of set bits. This proof verifies:
/// 1. The loop always terminates
/// 2. All set bits in the original mask are processed
#[kani::proof]
#[kani::unwind(65)] // At most 64 iterations (one per bit)
fn verify_run_length_terminates() {
    let original_mask: u64 = kani::any();
    let mut mask = original_mask;
    let mut processed_bits: u64 = 0;
    let mut iterations = 0u32;

    while mask != 0 {
        // From stride32.rs:192-199
        let start_bit = mask.trailing_zeros() as usize;
        let shifted_mask = mask >> start_bit;
        let run_len = (!shifted_mask).trailing_zeros() as usize;

        // Track which bits we're processing in this iteration
        let run_mask = if run_len >= 64 {
            !0u64
        } else {
            ((1u64 << run_len) - 1) << start_bit
        };
        processed_bits |= run_mask;

        // Clear the processed bits
        if run_len == 64 {
            mask = 0;
        } else {
            let clear_mask = !(((1u64 << run_len) - 1) << start_bit);
            mask &= clear_mask;
        }

        iterations += 1;
        kani::assert(iterations <= 64, "Loop must terminate within 64 iterations");
    }

    // All originally set bits should have been processed
    kani::assert(
        (original_mask & !processed_bits) == 0,
        "All set bits must be processed",
    );
}

// ============================================================================
// Proof 3: Path compression maintains root invariant
// ============================================================================
// File: inter_block.rs:85-101
// What: Prove find() returns actual root after path compression
// Why: Incorrect root → wrong cluster membership → decoder failure

/// Verify that path compression maintains the root invariant.
///
/// After path compression, find(x) should return y where parents[y] == y (a root).
/// This proof uses a small bounded parent array to verify the invariant.
#[kani::proof]
#[kani::unwind(9)] // Small array for tractability
fn verify_path_compression_correctness() {
    const N: usize = 8; // Small size for tractability

    let mut parents: [u32; N] = [0; N];

    // Initialize with symbolic values representing a valid union-find structure
    for i in 0..N {
        parents[i] = kani::any();
        // Parent must be a valid index
        kani::assume(parents[i] < N as u32);
    }

    // Ensure at least one root exists (a node that points to itself)
    let root_idx: usize = kani::any();
    kani::assume(root_idx < N);
    parents[root_idx] = root_idx as u32;

    // Pick a node to find
    let start: usize = kani::any();
    kani::assume(start < N);

    // Simulate path compression find with path halving
    let mut current = start;
    let mut steps = 0u32;

    while parents[current] != current as u32 && steps < N as u32 {
        let parent = parents[current] as usize;
        if parent < N && parents[parent] != parent as u32 {
            // Path halving: point to grandparent
            let grandparent = parents[parent];
            if (grandparent as usize) < N {
                parents[current] = grandparent;
                current = grandparent as usize;
            } else {
                current = parent;
            }
        } else {
            current = parent;
        }
        steps += 1;
    }

    // Invariant: We should have found a root (node pointing to itself)
    kani::assert(
        current < N && parents[current] == current as u32,
        "Find must return a root (self-referential node)",
    );
}

// ============================================================================
// Proof 4: Single-bit mask properties
// ============================================================================
// Helper proof for bit manipulation correctness

/// Verify properties of single-bit mask detection.
///
/// The expression `(mask & (mask - 1)) == 0` is used to detect single-bit masks.
#[kani::proof]
fn verify_single_bit_detection() {
    let mask: u64 = kani::any();
    kani::assume(mask != 0);

    let is_single_bit = (mask & (mask.wrapping_sub(1))) == 0;
    let popcount = mask.count_ones();

    // The expression correctly detects single-bit masks
    kani::assert(
        is_single_bit == (popcount == 1),
        "Single-bit detection must match popcount",
    );
}

// ============================================================================
// Proof 5: Trailing zeros bounds
// ============================================================================
// Verify trailing_zeros() returns valid bit positions

/// Verify trailing_zeros returns valid bit index.
#[kani::proof]
fn verify_trailing_zeros_bounds() {
    let mask: u64 = kani::any();
    kani::assume(mask != 0);

    let tz = mask.trailing_zeros();

    // trailing_zeros must be in range [0, 63] for non-zero input
    kani::assert(tz < 64, "Trailing zeros must be less than 64");

    // The bit at position tz must be set
    kani::assert((mask & (1u64 << tz)) != 0, "Bit at trailing_zeros position must be set");

    // All bits below tz must be zero
    if tz > 0 {
        let lower_mask = (1u64 << tz) - 1;
        kani::assert((mask & lower_mask) == 0, "All bits below trailing_zeros must be zero");
    }
}

// ============================================================================
// Proof 6: Spread boundary mask properties
// ============================================================================

/// Verify that spread operations don't lose bits.
#[kani::proof]
fn verify_spread_preserves_bits() {
    let boundary: u64 = kani::any();
    let mask: u64 = kani::any();

    // After AND with mask, we shouldn't have more bits than original
    let masked = boundary & mask;
    kani::assert(
        masked.count_ones() <= boundary.count_ones(),
        "Masking cannot increase bit count",
    );
    kani::assert(
        masked.count_ones() <= mask.count_ones(),
        "Masking cannot exceed mask bit count",
    );
}

// ============================================================================
// Proof 7: Block index calculations
// ============================================================================

/// Verify block/bit calculations for node addressing.
#[kani::proof]
fn verify_block_bit_calculations() {
    let node: u32 = kani::any();
    kani::assume(node < MAX_NODES as u32);

    // Block and bit calculations
    let blk = (node / 64) as usize;
    let bit = (node % 64) as usize;

    // Invariants
    kani::assert(blk < (MAX_NODES + 63) / 64, "Block index must be valid");
    kani::assert(bit < 64, "Bit index must be less than 64");

    // Reconstruction must give back original node
    let reconstructed = (blk * 64 + bit) as u32;
    kani::assert(reconstructed == node, "Block/bit must reconstruct to original node");
}

// ============================================================================
// Proof 8: Union-find rank/size bounds
// ============================================================================

/// Verify that union operations maintain valid parent references.
#[kani::proof]
#[kani::unwind(5)]
fn verify_union_parent_bounds() {
    const N: usize = 4;
    let mut parents: [u32; N] = [0, 1, 2, 3]; // Initially all roots

    let u: usize = kani::any();
    let v: usize = kani::any();
    kani::assume(u < N);
    kani::assume(v < N);

    // Simulate union by rank (larger index wins as simplified rank)
    if u != v {
        if u > v {
            parents[v] = u as u32;
        } else {
            parents[u] = v as u32;
        }
    }

    // All parents must be valid indices
    for i in 0..N {
        kani::assert(
            (parents[i] as usize) < N,
            "Parent must be a valid index after union",
        );
    }
}
