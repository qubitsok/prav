//! # Tutorial: Rectangular Surface Code Decoding
//!
//! This example demonstrates quantum error correction on a **5×3 rectangular**
//! lattice using the same SquareGrid topology as the square case.
//!
//! ## What's Different from Square?
//!
//! A rectangular lattice has **asymmetric boundaries**:
//! - The short edges (top/bottom, 5 nodes wide) are closer together
//! - The long edges (left/right, 3 nodes tall) are farther apart
//!
//! This asymmetry affects how syndromes are matched to boundaries:
//! - Syndromes near short edges have a shorter path to the boundary
//! - This can create different correction patterns than a square grid
//!
//! ## Running This Example
//!
//! ```bash
//! cargo run --example tutorial_rectangular
//! ```

use prav_core::{Arena, DecodingState, EdgeCorrection, SquareGrid};

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Grid width in syndrome nodes (longer dimension)
const WIDTH: usize = 5;
/// Grid height in syndrome nodes (shorter dimension)
const HEIGHT: usize = 3;
/// Stride for Morton encoding: next_power_of_two(max(WIDTH, HEIGHT))
const STRIDE_Y: usize = 8; // next_power_of_two(5) = 8

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Converts (x, y) coordinates to a Morton-encoded index.
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

fn print_lattice() {
    println!("
    The 5×3 Rectangular Lattice
    ═══════════════════════════

    A rectangular surface code with asymmetric boundaries.
    Width = 5 nodes, Height = 3 nodes.

           x=0   x=1   x=2   x=3   x=4
            │     │     │     │     │
    y=0 ─── o ─── o ─── o ─── o ─── o ───│  ← Top boundary (5 nodes)
            │     │     │     │     │
    y=1 ─── o ─── o ─── o ─── o ─── o ───│  ← Middle row
            │     │     │     │     │
    y=2 ─── o ─── o ─── o ─── o ─── o ───│  ← Bottom boundary (5 nodes)
            │     │     │     │     │
          ════════════════════════════
          Left boundary    Right boundary
          (3 nodes)        (3 nodes)

    Key observation: Top/bottom boundaries are 5 nodes wide,
    but left/right boundaries are only 3 nodes tall.

    Morton indices:
");
    for y in 0..HEIGHT {
        print!("    y={}: ", y);
        for x in 0..WIDTH {
            print!("{:3} ", idx(x, y));
        }
        println!();
    }
    println!();
}

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
║       QUANTUM ERROR CORRECTION TUTORIAL: Rectangular Lattice              ║
╚═══════════════════════════════════════════════════════════════════════════╝
");

    // =========================================================================
    // STEP 1: Understand the Rectangular Lattice
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  STEP 1: THE RECTANGULAR LATTICE");
    println!("═══════════════════════════════════════════════════════════════════════════");

    print_lattice();

    println!("    Why rectangular codes matter:");
    println!("    - Different X and Z error distances can be optimized separately");
    println!("    - Hardware constraints may favor non-square layouts");
    println!("    - Asymmetric noise models benefit from asymmetric codes");
    println!();

    // =========================================================================
    // STEP 2: Setup
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  STEP 2: INITIALIZE THE DECODER");
    println!("═══════════════════════════════════════════════════════════════════════════");

    let mut memory = [0u8; 256 * 1024];
    let mut arena = Arena::new(&mut memory);

    // Note: We use SquareGrid topology - the 4-neighbor connectivity is the same,
    // only the dimensions differ.
    let mut decoder: DecodingState<SquareGrid, STRIDE_Y> =
        DecodingState::new(&mut arena, WIDTH, HEIGHT, 1);

    println!("    Created decoder for {}x{} grid", WIDTH, HEIGHT);
    println!("    STRIDE_Y = {} (next_power_of_two of max dimension)", STRIDE_Y);
    println!("    Same SquareGrid topology - only dimensions change!");
    println!();

    let num_blocks = (STRIDE_Y * STRIDE_Y + 63) / 64;
    let mut syndromes = vec![0u64; num_blocks];
    let mut corrections = vec![EdgeCorrection::default(); WIDTH * HEIGHT * 2];

    // =========================================================================
    // SCENARIO 1: Error Near Short Boundary (Top/Bottom)
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  SCENARIO 1: Error Near Short Boundary (Top Edge)");
    println!("═══════════════════════════════════════════════════════════════════════════");

    println!("
    A single syndrome near the top boundary (y=0).
    Distance to top boundary: 0 edges (adjacent)
    Distance to bottom boundary: 2 edges (far)

           x=0   x=1   x=2   x=3   x=4
    ═══════════════════════════════════════  ← Top boundary
    y=0 ─── * ─── o ─── o ─── o ─── o ───   ← Syndrome at (0,0)
            │     │     │     │     │
    y=1 ─── o ─── o ─── o ─── o ─── o ───
            │     │     │     │     │
    y=2 ─── o ─── o ─── o ─── o ─── o ───
    ═══════════════════════════════════════  ← Bottom boundary

    The syndrome will match to the NEAREST boundary - the top edge.
");

    syndromes.fill(0);
    set_syndrome(&mut syndromes, idx(0, 0));

    print_syndromes_grid(&syndromes);

    decoder.sparse_reset();
    decoder.load_dense_syndromes(&syndromes);
    decoder.grow_clusters();
    let count = decoder.peel_forest(&mut corrections);

    print_corrections(&corrections, count);

    println!("    The cluster grows and immediately reaches the top boundary.");
    println!("    Correction matches syndrome to boundary - minimal weight path!");
    println!();

    // =========================================================================
    // SCENARIO 2: Error Near Long Boundary (Left/Right)
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  SCENARIO 2: Error Near Long Boundary (Left Edge)");
    println!("═══════════════════════════════════════════════════════════════════════════");

    println!("
    A single syndrome on the left boundary (x=0), middle row.
    This syndrome is equidistant from top and bottom boundaries,
    so it matches to the left boundary.

           x=0   x=1   x=2   x=3   x=4
            │     │     │     │     │
    y=0 ─── o ─── o ─── o ─── o ─── o ───
        ║   │     │     │     │     │
    y=1 ═X═ * ─── o ─── o ─── o ─── o ───  ← Syndrome at (0,1)
        ║   │     │     │     │     │
    y=2 ─── o ─── o ─── o ─── o ─── o ───
            │     │     │     │     │

    The syndrome matches to the left boundary.
");

    syndromes.fill(0);
    set_syndrome(&mut syndromes, idx(0, 1));

    print_syndromes_grid(&syndromes);

    decoder.sparse_reset();
    decoder.load_dense_syndromes(&syndromes);
    decoder.grow_clusters();
    let count = decoder.peel_forest(&mut corrections);

    print_corrections(&corrections, count);

    // =========================================================================
    // SCENARIO 3: Two Syndromes - Asymmetric Matching
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  SCENARIO 3: Two Syndromes at Opposite Corners");
    println!("═══════════════════════════════════════════════════════════════════════════");

    println!("
    Two syndromes at opposite corners of the rectangle.
    Which is the shortest path: through the lattice, or via boundaries?

           x=0   x=1   x=2   x=3   x=4
            │     │     │     │     │
    y=0 ─── * ─── o ─── o ─── o ─── o ───  ← Syndrome at (0,0)
            │     │     │     │     │
    y=1 ─── o ─── o ─── o ─── o ─── o ───
            │     │     │     │     │
    y=2 ─── o ─── o ─── o ─── o ─── * ───  ← Syndrome at (4,2)
            │     │     │     │     │

    Path through lattice: 4 + 2 = 6 edges (Manhattan distance)
    Path (0,0)→boundary + boundary→(4,2): could be shorter!
");

    syndromes.fill(0);
    set_syndrome(&mut syndromes, idx(0, 0)); // Top-left
    set_syndrome(&mut syndromes, idx(4, 2)); // Bottom-right

    print_syndromes_grid(&syndromes);

    decoder.sparse_reset();
    decoder.load_dense_syndromes(&syndromes);
    decoder.grow_clusters();
    let count = decoder.peel_forest(&mut corrections);

    print_corrections(&corrections, count);

    println!("    The decoder finds the minimum-weight matching automatically.");
    println!("    In a rectangular code, boundary distances are asymmetric!");
    println!();

    // =========================================================================
    // SCENARIO 4: Multiple Errors Creating a Horizontal Chain
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  SCENARIO 4: Error Chain Across the Long Axis");
    println!("═══════════════════════════════════════════════════════════════════════════");

    println!("
    Two syndromes on a horizontal line (along the long axis).
    The 5-node width allows for longer error chains.

           x=0   x=1   x=2   x=3   x=4
            │     │     │     │     │
    y=0 ─── o ─── o ─── o ─── o ─── o ───
            │     │     │     │     │
    y=1 ─── * ═══════════════════ * ───  ← Syndromes at (0,1) and (4,1)
            │     │     │     │     │
    y=2 ─── o ─── o ─── o ─── o ─── o ───
            │     │     │     │     │

    Distance between syndromes: 4 edges (the full width minus 1)
    Could match through lattice, or each to nearest boundary.
");

    syndromes.fill(0);
    set_syndrome(&mut syndromes, idx(0, 1)); // Left
    set_syndrome(&mut syndromes, idx(4, 1)); // Right

    print_syndromes_grid(&syndromes);

    decoder.sparse_reset();
    decoder.load_dense_syndromes(&syndromes);
    decoder.grow_clusters();
    let count = decoder.peel_forest(&mut corrections);

    print_corrections(&corrections, count);

    println!("    With 4 edges between them, the decoder weighs:");
    println!("    - Direct path: 4 edges");
    println!("    - Via boundaries: 1 + 1 = 2 edges (if boundary matching is cheaper)");
    println!();

    // =========================================================================
    // SUMMARY
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  SUMMARY: Rectangular vs Square Codes");
    println!("═══════════════════════════════════════════════════════════════════════════");

    println!("
    Key differences in rectangular codes:

    1. ASYMMETRIC BOUNDARY DISTANCES
       - Short edges (top/bottom) are closer for y-direction errors
       - Long edges (left/right) are closer for x-direction errors
       - Decoder automatically finds optimal boundary matching

    2. DIFFERENT ERROR DISTANCES
       - Code distance differs along x and y axes
       - Can be tuned to match asymmetric noise models
       - Width=5 gives distance-5 in x, height=3 gives distance-3 in y

    3. STRIDE CALCULATION
       - STRIDE_Y = next_power_of_two(max(WIDTH, HEIGHT))
       - For 5×3: STRIDE_Y = 8 (from max(5,3)=5, next_power_of_two(5)=8)
       - This ensures correct Morton indexing

    4. SAME TOPOLOGY
       - SquareGrid works for any rectangle
       - The 4-neighbor connectivity is unchanged
       - Only dimensions and boundary behavior differ

    Rectangular codes offer flexibility in trading off X vs Z error protection,
    which is valuable when physical error rates are asymmetric.
");
}
