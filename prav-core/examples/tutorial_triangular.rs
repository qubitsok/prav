//! # Tutorial: Color Code Decoding with a Triangular Lattice
//!
//! This example demonstrates quantum error correction on a **triangular lattice**
//! with 6-neighbor connectivity - the foundation for color codes.
//!
//! ## What Makes Triangular Special?
//!
//! Unlike the square lattice (4 neighbors), the triangular lattice has **6 neighbors**:
//! - 4 cardinal neighbors (up, down, left, right) - same as square
//! - 2 diagonal neighbors that depend on the node's **parity**
//!
//! ## Parity-Dependent Connectivity
//!
//! The parity of a node is determined by counting the 1-bits in its Morton index:
//! - **Even parity** (popcount even): diagonal neighbor is UP-RIGHT
//! - **Odd parity** (popcount odd): diagonal neighbor is DOWN-LEFT
//!
//! This creates the characteristic triangular tiling pattern.
//!
//! ## Running This Example
//!
//! ```bash
//! cargo run --example tutorial_triangular
//! ```

use prav_core::{Arena, DecodingState, EdgeCorrection, TriangularGrid};

// =============================================================================
// CONFIGURATION
// =============================================================================

const WIDTH: usize = 4;
const HEIGHT: usize = 4;
const STRIDE_Y: usize = 4;

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

#[inline]
fn idx(x: usize, y: usize) -> u32 {
    (y * STRIDE_Y + x) as u32
}

#[inline]
fn coords(index: u32) -> (usize, usize) {
    let i = index as usize;
    (i % STRIDE_Y, i / STRIDE_Y)
}

/// Determines the parity of a node based on its Morton index.
/// Even parity = even number of 1-bits in the index.
fn parity(index: u32) -> &'static str {
    if index.count_ones() % 2 == 0 {
        "even"
    } else {
        "odd"
    }
}

fn set_syndrome(syndromes: &mut [u64], index: u32) {
    let i = index as usize;
    let block = i / 64;
    let bit = i % 64;
    if block < syndromes.len() {
        syndromes[block] |= 1 << bit;
    }
}

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
    The 4×4 Triangular Lattice
    ═══════════════════════════

    Each node has 6 neighbors: 4 cardinal + 2 diagonal (parity-dependent).

    PARITY DETERMINES DIAGONAL CONNECTIVITY:

    Even parity node:          Odd parity node:
         \\  /                       |
        - o -                     - o -
          |                        / \\

    This creates a triangular tiling pattern:

           x=0   x=1   x=2   x=3
            │\\    │/    │\\    │/
    y=0 ─── E ─── O ─── E ─── O ───
            │/    │\\    │/    │\\
    y=1 ─── O ─── E ─── O ─── E ───
            │\\    │/    │\\    │/
    y=2 ─── E ─── O ─── E ─── O ───
            │/    │\\    │/    │\\
    y=3 ─── O ─── E ─── O ─── E ───

    E = Even parity (diagonal to up-right)
    O = Odd parity (diagonal to down-left)

    Node parities (index.count_ones() % 2):
");
    for y in 0..HEIGHT {
        print!("    y={}: ", y);
        for x in 0..WIDTH {
            let i = idx(x, y);
            print!("{:2}({}) ", i, if i.count_ones() % 2 == 0 { "E" } else { "O" });
        }
        println!();
    }
    println!();
}

fn print_syndromes_grid(syndromes: &[u64]) {
    println!("    Syndrome pattern (* = active, E/O = parity):");
    println!();
    print!("       ");
    for x in 0..WIDTH {
        print!(" x={}  ", x);
    }
    println!();

    for y in 0..HEIGHT {
        print!("    y={} ", y);
        for x in 0..WIDTH {
            let i = idx(x, y);
            let p = if i.count_ones() % 2 == 0 { "E" } else { "O" };
            if has_syndrome(syndromes, i) {
                print!(" *{}  ", p);
            } else {
                print!("  {}  ", p);
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
            println!("      - Edge from ({},{}) [{}] to BOUNDARY",
                     ux, uy, parity(c.u));
        } else {
            let (vx, vy) = coords(c.v);
            // Determine if this is a diagonal edge
            let dx = (vx as i32 - ux as i32).abs();
            let dy = (vy as i32 - uy as i32).abs();
            let edge_type = if dx == 1 && dy == 1 { "DIAGONAL" } else { "cardinal" };
            println!("      - Edge between ({},{}) and ({},{}) [{}]",
                     ux, uy, vx, vy, edge_type);
        }
    }
    println!();
}

fn print_neighbors(x: usize, y: usize) {
    let i = idx(x, y);
    let p = i.count_ones() % 2 == 0;
    println!("    Neighbors of ({},{}) [index={}, parity={}]:",
             x, y, i, if p { "even" } else { "odd" });

    // Cardinal neighbors
    if x > 0 {
        println!("      - LEFT:  ({},{}) [cardinal]", x - 1, y);
    }
    if x < WIDTH - 1 {
        println!("      - RIGHT: ({},{}) [cardinal]", x + 1, y);
    }
    if y > 0 {
        println!("      - UP:    ({},{}) [cardinal]", x, y - 1);
    }
    if y < HEIGHT - 1 {
        println!("      - DOWN:  ({},{}) [cardinal]", x, y + 1);
    }

    // Diagonal neighbors (parity-dependent)
    if p {
        // Even parity: up-right diagonal
        if x < WIDTH - 1 && y > 0 {
            println!("      - UP-RIGHT: ({},{}) [DIAGONAL - even parity]", x + 1, y - 1);
        }
    } else {
        // Odd parity: down-left diagonal
        if x > 0 && y < HEIGHT - 1 {
            println!("      - DOWN-LEFT: ({},{}) [DIAGONAL - odd parity]", x - 1, y + 1);
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
║       QUANTUM ERROR CORRECTION TUTORIAL: Triangular Lattice               ║
╚═══════════════════════════════════════════════════════════════════════════╝
");

    // =========================================================================
    // STEP 1: Understand the Triangular Lattice
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  STEP 1: THE TRIANGULAR LATTICE STRUCTURE");
    println!("═══════════════════════════════════════════════════════════════════════════");

    print_lattice();

    println!("    Why triangular codes matter:");
    println!("    - Color codes require triangular connectivity");
    println!("    - 6 neighbors allows more error patterns to be distinguished");
    println!("    - Can achieve better code distance per qubit than surface codes");
    println!();

    // Show specific neighbor examples
    println!("    Example: Neighbors of specific nodes");
    println!();
    print_neighbors(1, 1);
    print_neighbors(2, 1);

    // =========================================================================
    // STEP 2: Setup
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  STEP 2: INITIALIZE THE DECODER");
    println!("═══════════════════════════════════════════════════════════════════════════");

    let mut memory = [0u8; 256 * 1024];
    let mut arena = Arena::new(&mut memory);

    let mut decoder: DecodingState<TriangularGrid, STRIDE_Y> =
        DecodingState::new(&mut arena, WIDTH, HEIGHT, 1);

    println!("    Created decoder for {}x{} grid", WIDTH, HEIGHT);
    println!("    Using TriangularGrid topology (6-neighbor connectivity)");
    println!();

    let num_blocks = (STRIDE_Y * STRIDE_Y + 63) / 64;
    let mut syndromes = vec![0u64; num_blocks];
    let mut corrections = vec![EdgeCorrection::default(); WIDTH * HEIGHT * 3];

    // =========================================================================
    // SCENARIO 1: Cardinal Neighbors (Same as Square)
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  SCENARIO 1: Cardinal Error (Same as Square Grid)");
    println!("═══════════════════════════════════════════════════════════════════════════");

    println!("
    An error between two horizontally adjacent nodes.
    This uses cardinal connectivity (shared with square grid).

           x=0   x=1   x=2   x=3
    y=0 ─── E ─── O ─── E ─── O ───

    y=1 ─── O ─── * ═X═ * ─── E ───  ← Error between (1,1) and (2,1)

    y=2 ─── E ─── O ─── E ─── O ───
");

    syndromes.fill(0);
    set_syndrome(&mut syndromes, idx(1, 1)); // Odd parity
    set_syndrome(&mut syndromes, idx(2, 1)); // Even parity

    print_syndromes_grid(&syndromes);

    decoder.sparse_reset();
    decoder.load_dense_syndromes(&syndromes);
    decoder.grow_clusters();
    let count = decoder.peel_forest(&mut corrections);

    print_corrections(&corrections, count);

    println!("    This behaves exactly like the square grid - cardinal neighbors");
    println!("    are connected regardless of parity.");
    println!();

    // =========================================================================
    // SCENARIO 2: Diagonal Error (Parity-Dependent)
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  SCENARIO 2: Diagonal Error (Even Parity → Up-Right)");
    println!("═══════════════════════════════════════════════════════════════════════════");

    println!("
    An error on the diagonal edge from an EVEN parity node.
    Even nodes connect diagonally to UP-RIGHT.

           x=0   x=1   x=2   x=3
                        ╱
    y=0 ─── E ─── O ═══*═══ O ───  ← Syndrome at (2,0)
                      ╱
    y=1 ─── O ─── * ─── O ─── E ───  ← Syndrome at (1,1), connects diagonally UP-RIGHT

    Node (1,1) has index 5, popcount=2, EVEN parity → diagonal to (2,0)
");

    syndromes.fill(0);
    set_syndrome(&mut syndromes, idx(1, 1)); // Even parity node (index 5 = 0b101)
    set_syndrome(&mut syndromes, idx(2, 0)); // Its up-right diagonal neighbor

    print_syndromes_grid(&syndromes);

    // Verify parity
    println!("    Parity check:");
    println!("      - Node (1,1): index={}, popcount={}, parity={}",
             idx(1, 1), idx(1, 1).count_ones(), parity(idx(1, 1)));
    println!("      - Node (2,0): index={}, popcount={}, parity={}",
             idx(2, 0), idx(2, 0).count_ones(), parity(idx(2, 0)));
    println!();

    decoder.sparse_reset();
    decoder.load_dense_syndromes(&syndromes);
    decoder.grow_clusters();
    let count = decoder.peel_forest(&mut corrections);

    print_corrections(&corrections, count);

    // =========================================================================
    // SCENARIO 3: Diagonal Error (Odd Parity → Down-Left)
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  SCENARIO 3: Diagonal Error (Odd Parity → Down-Left)");
    println!("═══════════════════════════════════════════════════════════════════════════");

    println!("
    An error on the diagonal edge from an ODD parity node.
    Odd nodes connect diagonally to DOWN-LEFT.

           x=0   x=1   x=2   x=3

    y=0 ─── E ─── * ─── E ─── O ───  ← Syndrome at (1,0), ODD parity
                  ╲
    y=1 ─── * ═════════ O ─── E ───  ← Syndrome at (0,1), connects from (1,0) DOWN-LEFT
            ╲
    Node (1,0) has index 1, popcount=1, ODD parity → diagonal to (0,1)
");

    syndromes.fill(0);
    set_syndrome(&mut syndromes, idx(1, 0)); // Odd parity (index 1 = 0b001)
    set_syndrome(&mut syndromes, idx(0, 1)); // Its down-left diagonal neighbor

    print_syndromes_grid(&syndromes);

    println!("    Parity check:");
    println!("      - Node (1,0): index={}, popcount={}, parity={}",
             idx(1, 0), idx(1, 0).count_ones(), parity(idx(1, 0)));
    println!("      - Node (0,1): index={}, popcount={}, parity={}",
             idx(0, 1), idx(0, 1).count_ones(), parity(idx(0, 1)));
    println!();

    decoder.sparse_reset();
    decoder.load_dense_syndromes(&syndromes);
    decoder.grow_clusters();
    let count = decoder.peel_forest(&mut corrections);

    print_corrections(&corrections, count);

    // =========================================================================
    // SCENARIO 4: Multiple Errors with Mixed Connectivity
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  SCENARIO 4: Mixed Cardinal and Diagonal Errors");
    println!("═══════════════════════════════════════════════════════════════════════════");

    println!("
    A more complex error pattern spanning both cardinal and diagonal edges.
    The triangular connectivity provides more paths between syndromes.

           x=0   x=1   x=2   x=3
                    ╱
    y=0 ─── E ─── * ─── E ─── O ───  ← Syndrome at (1,0)
                ╲ │
    y=1 ─── O ─── * ─── O ─── E ───  ← Syndrome at (1,1)
                  │
    y=2 ─── E ─── O ─── E ─── O ───

    Both syndromes are at x=1. They share:
    - A cardinal edge (vertical)
    - Possibly diagonal connections depending on parity
");

    syndromes.fill(0);
    set_syndrome(&mut syndromes, idx(1, 0));
    set_syndrome(&mut syndromes, idx(1, 1));

    print_syndromes_grid(&syndromes);

    decoder.sparse_reset();
    decoder.load_dense_syndromes(&syndromes);
    decoder.grow_clusters();
    let count = decoder.peel_forest(&mut corrections);

    print_corrections(&corrections, count);

    // =========================================================================
    // SUMMARY
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  SUMMARY: Triangular Lattice Key Points");
    println!("═══════════════════════════════════════════════════════════════════════════");

    println!("
    1. CONNECTIVITY
       - 4 cardinal neighbors (like square grid)
       - 2 additional diagonal neighbors (parity-dependent)
       - Total: 6 neighbors per interior node

    2. PARITY RULE
       - Even parity (popcount even): diagonal is UP-RIGHT
       - Odd parity (popcount odd): diagonal is DOWN-LEFT
       - Creates alternating checkerboard-like diagonal pattern

    3. COLOR CODES
       - Triangular connectivity is required for color codes
       - Color codes have symmetric X/Z error correction
       - Higher connectivity enables more error detection

    4. CLUSTER GROWTH
       - More neighbors = faster cluster growth
       - Diagonal connections create shorter paths
       - Decoder automatically uses optimal connectivity

    The triangular lattice provides richer connectivity than the square grid,
    enabling quantum codes with different properties and potentially better
    error thresholds for certain noise models.
");
}
