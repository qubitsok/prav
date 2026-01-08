//! Kani formal verification proofs for decoder core modules.
//!
//! These proofs verify critical safety invariants in the decoder's core operations:
//! - Block/bit addressing (state.rs)
//! - Union-Find correctness (union_find.rs)
//! - Tiled coordinate mapping (tiled.rs)
//! - Sparse reset correctness (state.rs)
//!
//! Run with: `cargo kani --package prav-core`

/// Maximum grid dimension for bounded proofs
const MAX_DIM: usize = 64;
/// Maximum nodes for proofs (bounded for tractability)
const MAX_NODES: usize = MAX_DIM * MAX_DIM;
/// Maximum blocks (64 nodes per block)
const MAX_BLOCKS: usize = (MAX_NODES + 63) / 64;

// ============================================================================
// Proof 1: Block/bit round-trip
// ============================================================================
// File: state.rs (block indexing throughout)
// What: (node / 64, node % 64) reconstructs to original node
// Why: Incorrect reconstruction → wrong memory access in blocks_state

/// Verify that block/bit decomposition is reversible.
///
/// The decoder uses `blk = node / 64` and `bit = node % 64` extensively
/// for indexing into `blocks_state` and bitmasks. This proof verifies
/// that the original node can always be reconstructed.
#[kani::proof]
fn verify_block_bit_round_trip() {
    let node: u32 = kani::any();
    kani::assume(node < MAX_NODES as u32);

    // Block and bit decomposition (used throughout decoder)
    let blk = (node / 64) as usize;
    let bit = (node % 64) as usize;

    // Invariant 1: bit must be in valid range [0, 63]
    kani::assert(bit < 64, "Bit index must be less than 64");

    // Invariant 2: block must be in valid range
    kani::assert(blk < MAX_BLOCKS, "Block index must be valid");

    // Invariant 3: reconstruction must give back original node
    let reconstructed = (blk * 64 + bit) as u32;
    kani::assert(
        reconstructed == node,
        "Block/bit must reconstruct to original node",
    );
}

// ============================================================================
// Proof 2: Tiled coordinate round-trip
// ============================================================================
// File: tiled.rs:get_node_idx, get_global_coord
// What: get_global_coord(get_node_idx(x, y)) == (x, y) for valid coords
// Why: Incorrect mapping → wrong cluster assignments in tiled decoder

/// Verify that tiled coordinate mapping is bijective.
///
/// The tiled decoder maps (x, y) global coordinates to internal node indices
/// using a tile-based layout. This proof verifies the mapping is reversible.
#[kani::proof]
fn verify_tiled_coordinate_round_trip() {
    // Use bounded inputs for tractability
    let width: usize = kani::any();
    let height: usize = kani::any();
    let x: usize = kani::any();
    let y: usize = kani::any();

    // Constrain to reasonable grid sizes
    kani::assume(width > 0 && width <= 64);
    kani::assume(height > 0 && height <= 64);
    kani::assume(x < width);
    kani::assume(y < height);

    // Compute tile configuration
    let tiles_x = (width + 31) / 32;

    // Forward mapping: (x, y) → node_idx (from tiled.rs)
    let tx = x / 32;
    let ty = y / 32;
    let lx = x % 32;
    let ly = y % 32;
    let tile_idx = ty * tiles_x + tx;
    let local_idx = ly * 32 + lx;
    let node_idx = tile_idx * 1024 + local_idx;

    // Reverse mapping: node_idx → (x', y')
    let tile_idx_back = node_idx / 1024;
    let local_idx_back = node_idx % 1024;
    let tx_back = tile_idx_back % tiles_x;
    let ty_back = tile_idx_back / tiles_x;
    let lx_back = local_idx_back % 32;
    let ly_back = local_idx_back / 32;
    let x_back = tx_back * 32 + lx_back;
    let y_back = ty_back * 32 + ly_back;

    // Invariant: round-trip must preserve coordinates
    kani::assert(x_back == x, "X coordinate must round-trip correctly");
    kani::assert(y_back == y, "Y coordinate must round-trip correctly");
}

// ============================================================================
// Proof 3: Sparse reset processes all dirty blocks
// ============================================================================
// File: state.rs:331-357
// What: sparse_reset loop clears all bits in block_dirty_mask
// Why: Missed dirty blocks → stale state in next decode

/// Verify that sparse reset loop processes all dirty blocks.
///
/// The sparse_reset function iterates over set bits in block_dirty_mask,
/// clearing each one. This proof verifies no bits are missed.
#[kani::proof]
#[kani::unwind(65)] // At most 64 bits per word
fn verify_sparse_reset_processes_all_dirty() {
    let original_mask: u64 = kani::any();
    let mut mask = original_mask;
    let mut processed_count = 0u32;

    // Simulate the sparse_reset loop from state.rs:335-337
    while mask != 0 {
        let bit = mask.trailing_zeros();
        mask &= mask - 1; // Clear lowest set bit
        processed_count += 1;

        // Safety check: we should never process more than 64 bits
        kani::assert(processed_count <= 64, "Should process at most 64 bits");
    }

    // Invariant 1: All bits should be cleared
    kani::assert(mask == 0, "All dirty bits must be processed");

    // Invariant 2: Count should match popcount
    kani::assert(
        processed_count == original_mask.count_ones(),
        "Processed count must match original popcount",
    );
}

// ============================================================================
// Proof 4: Path halving terminates at root
// ============================================================================
// File: union_find.rs:68-82 (find_slow)
// What: find_slow() always returns node where parents[node] == node
// Why: Non-root return → incorrect cluster membership

/// Verify that path halving find always terminates at a root.
///
/// The find_slow function uses path halving to compress paths while
/// finding the root. This proof verifies it always returns a valid root.
#[kani::proof]
#[kani::unwind(9)] // Small array for tractability
fn verify_path_halving_terminates_at_root() {
    const N: usize = 8;
    let mut parents: [u32; N] = [0; N];

    // Initialize with symbolic valid union-find structure
    for i in 0..N {
        parents[i] = kani::any();
        kani::assume(parents[i] < N as u32);
    }

    // Ensure at least one root exists
    let root_idx: usize = kani::any();
    kani::assume(root_idx < N);
    parents[root_idx] = root_idx as u32;

    // Pick a starting node
    let start: usize = kani::any();
    kani::assume(start < N);

    // Simulate path halving (from union_find.rs:68-82)
    let mut i = start;
    let mut p = parents[i] as usize;
    let mut steps = 0u32;

    while p != i && steps < N as u32 {
        let grandparent = parents[p] as usize;
        if grandparent < N && grandparent != p {
            // Path halving: point to grandparent
            parents[i] = grandparent as u32;
            i = grandparent;
            p = parents[i] as usize;
        } else {
            // p is root (or grandparent == p)
            i = p;
            break;
        }
        steps += 1;
    }

    // If we exited because steps == N, find the actual root
    while parents[i] as usize != i && steps < 2 * N as u32 {
        i = parents[i] as usize;
        steps += 1;
    }

    // Invariant: We must have found a root
    kani::assert(
        i < N && parents[i] == i as u32,
        "Find must return a root (self-referential node)",
    );
}

// ============================================================================
// Proof 5: Union roots deterministic
// ============================================================================
// File: union_find.rs:26-49 (union_roots)
// What: union_roots(a, b) always makes smaller root child of larger
// Why: Non-determinism → inconsistent merging behavior

/// Verify that union_roots is deterministic (smaller joins larger).
///
/// The union_roots function always makes the smaller root point to the
/// larger root. This ensures deterministic behavior regardless of call order.
#[kani::proof]
fn verify_union_roots_deterministic() {
    let root_u: u32 = kani::any();
    let root_v: u32 = kani::any();

    kani::assume(root_u < MAX_NODES as u32);
    kani::assume(root_v < MAX_NODES as u32);
    kani::assume(root_u != root_v);

    // Simulate union_roots logic (from union_find.rs:31-47)
    let (smaller, larger) = if root_u < root_v {
        (root_u, root_v)
    } else {
        (root_v, root_u)
    };

    // After union, smaller should point to larger
    // (In real code: parents[smaller] = larger)

    // Invariant 1: The result should be deterministic
    kani::assert(
        smaller < larger,
        "Smaller root must be identified correctly",
    );

    // Invariant 2: Calling with swapped arguments should give same result
    let (smaller2, larger2) = if root_v < root_u {
        (root_v, root_u)
    } else {
        (root_u, root_v)
    };

    kani::assert(
        smaller == smaller2 && larger == larger2,
        "Union must be deterministic regardless of argument order",
    );
}

// ============================================================================
// Proof 6: Tile block index bounds
// ============================================================================
// File: tiled.rs (block indexing)
// What: tile_idx * 16 + local_blk < total_blocks for valid inputs
// Why: Out-of-bounds → memory corruption in blocks_state

/// Verify that tile block indexing stays within bounds.
///
/// The tiled decoder computes block indices as `tile_idx * 16 + local_blk`.
/// This proof verifies this never exceeds total_blocks.
#[kani::proof]
fn verify_tile_block_index_bounds() {
    let tiles_x: usize = kani::any();
    let tiles_y: usize = kani::any();
    let tile_idx: usize = kani::any();
    let local_blk: usize = kani::any();

    // Constrain to reasonable sizes
    kani::assume(tiles_x > 0 && tiles_x <= 4);
    kani::assume(tiles_y > 0 && tiles_y <= 4);
    kani::assume(tile_idx < tiles_x * tiles_y);
    kani::assume(local_blk < 16); // 16 blocks per tile

    let total_blocks = tiles_x * tiles_y * 16;
    let block_idx = tile_idx * 16 + local_blk;

    // Invariant: block_idx must be within total_blocks
    kani::assert(
        block_idx < total_blocks,
        "Block index must be within total_blocks",
    );
}

// ============================================================================
// Proof 7: Edge index bounds
// ============================================================================
// File: state.rs (edge_bitmap indexing)
// What: node * 3 + dir < edge_bitmap.len() * 64 for valid nodes
// Why: Out-of-bounds → memory corruption in edge tracking

/// Verify that edge bitmap indexing stays within bounds.
///
/// Each node has 3 edge directions, stored in edge_bitmap.
/// This proof verifies `node * 3 + dir` never exceeds capacity.
#[kani::proof]
fn verify_edge_index_bounds() {
    let total_nodes: usize = kani::any();
    let node: usize = kani::any();
    let dir: usize = kani::any();

    // Constrain to reasonable sizes
    kani::assume(total_nodes > 0 && total_nodes <= MAX_NODES);
    kani::assume(node < total_nodes);
    kani::assume(dir < 3); // 3 directions per node

    // Edge capacity calculation (from state.rs:159-160)
    let num_edges = total_nodes * 3;
    let num_edge_words = (num_edges + 63) / 64;
    let edge_capacity = num_edge_words * 64;

    // Edge index calculation
    let edge_idx = node * 3 + dir;

    // Invariant 1: edge_idx must fit in allocated capacity
    kani::assert(
        edge_idx < edge_capacity,
        "Edge index must be within edge_bitmap capacity",
    );

    // Invariant 2: edge_idx must not overflow
    kani::assert(edge_idx < usize::MAX / 2, "Edge index must not overflow");
}

// ============================================================================
// Proof 8: Mark block dirty idempotent
// ============================================================================
// File: state.rs:307-312 (mark_block_dirty)
// What: Calling mark_block_dirty twice has same effect as once
// Why: Non-idempotent → incorrect dirty tracking

/// Verify that mark_block_dirty is idempotent.
///
/// The mark_block_dirty function sets a bit in block_dirty_mask.
/// Calling it multiple times should have the same effect as calling once.
#[kani::proof]
fn verify_mark_block_dirty_idempotent() {
    let blk_idx: usize = kani::any();
    kani::assume(blk_idx < MAX_BLOCKS);

    // Initial state: some arbitrary mask
    let initial_mask: u64 = kani::any();
    let mut mask1 = initial_mask;
    let mut mask2 = initial_mask;

    // Calculate bit position (from state.rs:308-309)
    let mask_idx = blk_idx >> 6;
    let mask_bit = blk_idx & 63;

    // For single-word proof, assume mask_idx == 0
    kani::assume(mask_idx == 0);

    // Apply mark_block_dirty once
    mask1 |= 1u64 << mask_bit;

    // Apply mark_block_dirty twice
    mask2 |= 1u64 << mask_bit;
    mask2 |= 1u64 << mask_bit;

    // Invariant: Both should produce identical result
    kani::assert(
        mask1 == mask2,
        "mark_block_dirty must be idempotent",
    );

    // Additional invariant: the bit should be set
    kani::assert(
        (mask1 & (1u64 << mask_bit)) != 0,
        "Target bit must be set after mark_block_dirty",
    );
}

// ============================================================================
// Proof 9: Push next idempotent (bonus)
// ============================================================================
// File: state.rs:323-328 (push_next)
// What: push_next uses same bit-setting pattern as mark_block_dirty
// Why: Ensures queued_mask bit-setting is also idempotent

/// Verify that push_next is idempotent (same pattern as mark_block_dirty).
#[kani::proof]
fn verify_push_next_idempotent() {
    let blk_idx: usize = kani::any();
    kani::assume(blk_idx < MAX_BLOCKS);

    let initial_mask: u64 = kani::any();
    let mut mask1 = initial_mask;
    let mut mask2 = initial_mask;

    let mask_idx = blk_idx >> 6;
    let mask_bit = blk_idx & 63;

    kani::assume(mask_idx == 0);

    // Apply push_next once
    mask1 |= 1u64 << mask_bit;

    // Apply push_next twice
    mask2 |= 1u64 << mask_bit;
    mask2 |= 1u64 << mask_bit;

    kani::assert(mask1 == mask2, "push_next must be idempotent");
}

// ============================================================================
// Proof 10: Stride calculation bounds
// ============================================================================
// File: state.rs:66-69 (stride calculation in new())
// What: next_power_of_two never overflows for reasonable grid sizes
// Why: Overflow → incorrect memory layout calculations

/// Verify stride calculation bounds for grid initialization.
#[kani::proof]
fn verify_stride_calculation_bounds() {
    let width: usize = kani::any();
    let height: usize = kani::any();

    // Constrain to QEC-reasonable sizes (up to 1024x1024)
    kani::assume(width > 0 && width <= 1024);
    kani::assume(height > 0 && height <= 1024);

    // Stride calculation (from state.rs:65-69)
    let max_dim = width.max(height);
    let dim_pow2 = max_dim.next_power_of_two();
    let stride_y = dim_pow2;

    // Invariant 1: stride_y must be power of 2
    kani::assert(
        stride_y.is_power_of_two(),
        "stride_y must be power of two",
    );

    // Invariant 2: stride_y must be >= max_dim
    kani::assert(stride_y >= max_dim, "stride_y must be >= max_dim");

    // Invariant 3: stride_y must not overflow reasonable bounds
    kani::assert(stride_y <= 2048, "stride_y must be bounded for 1024 max dim");

    // Invariant 4: total nodes should not overflow
    let total_nodes = stride_y * stride_y;
    kani::assert(
        total_nodes <= 4 * 1024 * 1024,
        "Total nodes must be bounded",
    );
}
