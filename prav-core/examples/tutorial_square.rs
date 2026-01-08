//! # Tutorial: Surface Code Decoding with a Square Lattice
//!
//! This example demonstrates quantum error correction using Union-Find
//! on a 4×4 square lattice - the foundation of surface codes.
//!
//! ## What You'll Learn
//!
//! 1. **Lattice Structure**: How syndrome nodes and data qubits are arranged
//! 2. **Error Propagation**: How a single bit-flip creates two syndromes
//! 3. **Union-Find Decoding**: How clusters grow and merge to find corrections
//! 4. **Boundary Matching**: How syndromes near edges can match to the boundary
//!
//! ## The Physical Picture
//!
//! In a surface code, we have two types of elements:
//!
//! - **Syndrome nodes** (stabilizers): Measure parity of neighboring data qubits
//! - **Data qubits**: Live on the edges between syndrome nodes
//!
//! When a data qubit experiences an error (bit-flip), the two adjacent syndrome
//! nodes detect a parity change. The decoder's job is to find which data qubits
//! to flip to restore the correct state.
//!
//! ## Running This Example
//!
//! ```bash
//! cargo run --example tutorial_square
//! ```

use prav_core::{Arena, DecodingState, EdgeCorrection, SquareGrid};

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Grid width in syndrome nodes
const WIDTH: usize = 4;
/// Grid height in syndrome nodes
const HEIGHT: usize = 4;
/// Stride for Morton encoding: next_power_of_two(max(WIDTH, HEIGHT))
const STRIDE_Y: usize = 4;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Converts (x, y) coordinates to a Morton-encoded index.
///
/// Morton encoding interleaves the bits of x and y coordinates to create
/// a single index that preserves spatial locality - nearby points in 2D
/// are also nearby in the 1D index, improving cache efficiency.
///
/// For our simple row-major layout with power-of-two stride:
/// `index = y * STRIDE_Y + x`
#[inline]
fn idx(x: usize, y: usize) -> u32 {
    (y * STRIDE_Y + x) as u32
}

/// Converts a Morton index back to (x, y) coordinates.
#[inline]
fn coords(index: u32) -> (usize, usize) {
    let i = index as usize;
    (i % STRIDE_Y, i / STRIDE_Y)
}

/// Sets a syndrome bit in the dense syndrome array.
///
/// Syndromes are packed into u64 words, with 64 syndrome nodes per word.
/// This is efficient for both storage and SIMD operations during decoding.
fn set_syndrome(syndromes: &mut [u64], index: u32) {
    let i = index as usize;
    let block = i / 64;
    let bit = i % 64;
    if block < syndromes.len() {
        syndromes[block] |= 1 << bit;
    }
}

/// Checks if a syndrome is set at the given index.
fn has_syndrome(syndromes: &[u64], index: u32) -> bool {
    let i = index as usize;
    let block = i / 64;
    let bit = i % 64;
    if block < syndromes.len() {
        (syndromes[block] >> bit) & 1 == 1
    } else {
        false
    }
}

// =============================================================================
// VISUALIZATION
// =============================================================================

/// Prints the lattice structure with syndrome node positions.
fn print_lattice() {
    println!("
    The 4×4 Square Lattice
    ═══════════════════════

    Each 'o' is a syndrome node (stabilizer measurement).
    Data qubits sit on the edges connecting syndrome nodes.

           x=0   x=1   x=2   x=3
            │     │     │     │
    y=0 ─── o ─── o ─── o ─── o ───│ (boundary)
            │     │     │     │
    y=1 ─── o ─── o ─── o ─── o ───│
            │     │     │     │
    y=2 ─── o ─── o ─── o ─── o ───│
            │     │     │     │
    y=3 ─── o ─── o ─── o ─── o ───│
            │     │     │     │
          (boundary)

    Interior nodes have 4 neighbors (up, down, left, right).
    Boundary nodes have 2-3 neighbors and can match to the boundary.

    Morton indices for this grid:
");
    for y in 0..HEIGHT {
        print!    ("    y={}: ", y);
        for x in 0..WIDTH {
            print!("{:3} ", idx(x, y));
        }
        println!();
    }
    println!();
}

/// Prints the grid showing which syndromes are active.
fn print_syndromes_grid(syndromes: &[u64]) {
    println!("    Syndrome pattern (* = active syndrome):");
    println!();
    print!("       ");
    for x in 0..WIDTH {
        print!(" x={} ", x);
    }
    println!();

    for y in 0..HEIGHT {
        print!("    y={} ", y);
        for x in 0..WIDTH {
            let i = idx(x, y);
            if has_syndrome(syndromes, i) {
                print!("  *  ");
            } else {
                print!("  .  ");
            }
        }
        println!();
    }
    println!();
}

/// Prints edge corrections in a human-readable format.
fn print_corrections(corrections: &[EdgeCorrection], count: usize) {
    println!("    Corrections to apply ({} edges):", count);
    for c in &corrections[..count] {
        let (ux, uy) = coords(c.u);
        if c.v == u32::MAX {
            println!("      - Edge from ({},{}) to BOUNDARY", ux, uy);
        } else {
            let (vx, vy) = coords(c.v);
            println!("      - Edge between ({},{}) and ({},{})", ux, uy, vx, vy);
        }
    }
    println!();
}

// =============================================================================
// MAIN DEMONSTRATION
// =============================================================================

fn main() {
    println!("
╔═══════════════════════════════════════════════════════════════════════════╗
║         QUANTUM ERROR CORRECTION TUTORIAL: Square Lattice                 ║
╚═══════════════════════════════════════════════════════════════════════════╝
");

    // =========================================================================
    // STEP 1: Understand the Lattice
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  STEP 1: THE LATTICE STRUCTURE");
    println!("═══════════════════════════════════════════════════════════════════════════");

    print_lattice();

    println!("    Key insight: In a square lattice, each syndrome node measures the");
    println!("    parity of its 4 neighboring data qubits. When a data qubit flips,");
    println!("    exactly TWO adjacent syndrome nodes detect the error.");
    println!();

    // =========================================================================
    // STEP 2: Setup Memory and Decoder
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  STEP 2: INITIALIZE THE DECODER");
    println!("═══════════════════════════════════════════════════════════════════════════");

    // Allocate memory for the decoder using an arena allocator.
    // This allows no_std operation without heap allocation.
    let mut memory = [0u8; 256 * 1024]; // 256 KB buffer
    let mut arena = Arena::new(&mut memory);

    // Create the decoder. The type parameters are:
    // - SquareGrid: The topology (4-neighbor connectivity)
    // - STRIDE_Y: Must be next_power_of_two(max(width, height))
    let mut decoder: DecodingState<SquareGrid, STRIDE_Y> =
        DecodingState::new(&mut arena, WIDTH, HEIGHT, 1);

    println!("    Created decoder for {}x{} grid with STRIDE_Y={}", WIDTH, HEIGHT, STRIDE_Y);
    println!("    Using SquareGrid topology (4-neighbor connectivity)");
    println!();

    // Calculate number of u64 blocks needed for syndrome storage
    let num_blocks = (STRIDE_Y * STRIDE_Y + 63) / 64;
    let mut syndromes = vec![0u64; num_blocks];
    let mut corrections = vec![EdgeCorrection::default(); WIDTH * HEIGHT * 2];

    // =========================================================================
    // SCENARIO 1: Single Error (Two Adjacent Syndromes)
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  SCENARIO 1: Single Data Qubit Error");
    println!("═══════════════════════════════════════════════════════════════════════════");

    println!("
    Physical situation: A single data qubit on the edge between
    syndrome nodes (1,1) and (2,1) experiences a bit-flip error.

    This activates exactly two syndrome nodes - the ones adjacent to
    the errored data qubit.

           x=0   x=1   x=2   x=3
    y=0 ─── o ─── o ─── o ─── o ───
            │     │     │     │
    y=1 ─── o ─── * ═X═ * ─── o ───  ← Error on edge between (1,1) and (2,1)
            │     │     │     │
    y=2 ─── o ─── o ─── o ─── o ───
            │     │     │     │
    y=3 ─── o ─── o ─── o ─── o ───

    * = syndrome triggered, X = data qubit error
");

    // Clear and set syndromes
    syndromes.fill(0);
    set_syndrome(&mut syndromes, idx(1, 1)); // Node at (1,1)
    set_syndrome(&mut syndromes, idx(2, 1)); // Node at (2,1)

    print_syndromes_grid(&syndromes);

    // Decode
    decoder.sparse_reset();
    decoder.load_dense_syndromes(&syndromes);
    decoder.grow_clusters();
    let count = decoder.peel_forest(&mut corrections);

    println!("    Union-Find Decoding Process:");
    println!("    1. Two syndrome clusters start at (1,1) and (2,1)");
    println!("    2. Clusters grow toward each other");
    println!("    3. When they meet, they merge into one cluster");
    println!("    4. Peeling extracts the correction edge");
    println!();

    print_corrections(&corrections, count);

    println!("    Result: The decoder found the direct edge connecting the two");
    println!("    syndromes - this is exactly the errored data qubit!");
    println!();

    // =========================================================================
    // SCENARIO 2: Two Errors Creating a Chain
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  SCENARIO 2: Two Adjacent Errors (Chain)");
    println!("═══════════════════════════════════════════════════════════════════════════");

    println!("
    Physical situation: Two adjacent data qubits both have errors.
    The syndrome at the middle node cancels out (XOR of two 1s = 0).

           x=0   x=1   x=2   x=3
    y=0 ─── o ─── o ─── o ─── o ───
            │     │     │     │
    y=1 ─── * ═X═ o ═X═ * ─── o ───  ← Errors on edges (0,1)-(1,1) and (1,1)-(2,1)
            │     │     │     │
    y=2 ─── o ─── o ─── o ─── o ───
            │     │     │     │
    y=3 ─── o ─── o ─── o ─── o ───

    The node at (1,1) sees TWO adjacent errors, so its parity is 0+1+1=0 (even).
    Only the endpoints of the error chain have odd parity.
");

    syndromes.fill(0);
    set_syndrome(&mut syndromes, idx(0, 1)); // Left end of chain
    set_syndrome(&mut syndromes, idx(2, 1)); // Right end of chain
    // Note: (1,1) has NO syndrome - the errors cancel there!

    print_syndromes_grid(&syndromes);

    decoder.sparse_reset();
    decoder.load_dense_syndromes(&syndromes);
    decoder.grow_clusters();
    let count = decoder.peel_forest(&mut corrections);

    println!("    Key insight: The decoder doesn't need to know there were two errors.");
    println!("    It just needs to find ANY path connecting the syndromes that, when");
    println!("    applied, neutralizes them. XOR ensures equivalent corrections work!");
    println!();

    print_corrections(&corrections, count);

    // =========================================================================
    // SCENARIO 3: Boundary Matching
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  SCENARIO 3: Boundary Matching");
    println!("═══════════════════════════════════════════════════════════════════════════");

    println!("
    Physical situation: An error occurs on a data qubit at the edge
    of the lattice. One syndrome node detects it, but the other
    'syndrome' is effectively the boundary itself.

           x=0   x=1   x=2   x=3
    y=0 ─X─ * ─── o ─── o ─── o ───  ← Error on boundary edge at (0,0)
            │     │     │     │
    y=1 ─── o ─── o ─── o ─── o ───
            │     │     │     │
    ...

    The syndrome at (0,0) has odd parity, but there's no second syndrome
    inside the lattice. The decoder matches it to the boundary.
");

    syndromes.fill(0);
    set_syndrome(&mut syndromes, idx(0, 0)); // Single syndrome near boundary

    print_syndromes_grid(&syndromes);

    decoder.sparse_reset();
    decoder.load_dense_syndromes(&syndromes);
    decoder.grow_clusters();
    let count = decoder.peel_forest(&mut corrections);

    println!("    Union-Find handles boundaries naturally:");
    println!("    - The cluster at (0,0) grows and hits the boundary");
    println!("    - The boundary acts as a 'virtual syndrome' that absorbs odd-parity clusters");
    println!("    - Correction is applied to the boundary edge");
    println!();

    print_corrections(&corrections, count);

    // =========================================================================
    // SUMMARY
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  SUMMARY: What We Learned");
    println!("═══════════════════════════════════════════════════════════════════════════");

    println!("
    1. SYNDROME GENERATION
       - Each data qubit error activates exactly 2 adjacent syndrome nodes
       - Multiple errors on adjacent qubits can cancel syndromes (XOR property)

    2. UNION-FIND DECODING
       - Start with each syndrome as its own cluster
       - Grow clusters outward to neighbors
       - Merge clusters when they meet (union operation)
       - Use path compression for efficient root-finding

    3. PEELING
       - After clustering, extract correction edges
       - Find paths between paired syndromes
       - Corrections are XOR'd, so overlapping paths cancel

    4. BOUNDARY HANDLING
       - Syndromes near boundaries can match to the boundary
       - This handles cases where errors occur on edge data qubits

    The decoder doesn't need to find the EXACT errors - it only needs to find
    a correction that, when XOR'd with the actual errors, gives the identity.
    This is why Union-Find is so powerful: it finds efficient pairings that
    neutralize all syndromes.
");
}
