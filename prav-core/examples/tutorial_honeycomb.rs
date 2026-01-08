//! # Tutorial: Honeycomb Code Decoding
//!
//! This example demonstrates quantum error correction on a **honeycomb lattice**
//! with **3-neighbor connectivity** - the sparsest of our 2D topologies.
//!
//! ## What Makes Honeycomb Special?
//!
//! The honeycomb lattice has the **minimum connectivity** possible for a 2D QEC code:
//! - 2 horizontal neighbors (left and right) - always present
//! - 1 vertical neighbor that depends on the node's **parity**
//!
//! ## Parity-Dependent Vertical Connection
//!
//! - **Even parity** (popcount even): vertical neighbor is UP
//! - **Odd parity** (popcount odd): vertical neighbor is DOWN
//!
//! This creates the characteristic hexagonal/honeycomb tiling where nodes
//! alternate between connecting upward and downward.
//!
//! ## Running This Example
//!
//! ```bash
//! cargo run --example tutorial_honeycomb
//! ```

use prav_core::{Arena, DecodingState, EdgeCorrection, HoneycombGrid};

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

fn parity(index: u32) -> &'static str {
    if index.count_ones() % 2 == 0 {
        "even"
    } else {
        "odd"
    }
}

fn vertical_direction(index: u32) -> &'static str {
    if index.count_ones() % 2 == 0 {
        "UP"
    } else {
        "DOWN"
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
    The 4×4 Honeycomb Lattice
    ═══════════════════════════

    Each node has exactly 3 neighbors: 2 horizontal + 1 vertical.
    The vertical connection alternates based on parity.

    PARITY DETERMINES VERTICAL CONNECTION:

    Even parity (connects UP):    Odd parity (connects DOWN):
           |
         - o -                         - o -
                                         |

    The resulting pattern forms hexagonal cells:

           x=0   x=1   x=2   x=3
            |           |
    y=0 ─── E ─── O ─── E ─── O ───
                  |           |
    y=1 ─── O ─── E ─── O ─── E ───
            |           |
    y=2 ─── E ─── O ─── E ─── O ───
                  |           |
    y=3 ─── O ─── E ─── O ─── E ───
            |           |

    E = Even parity (vertical to UP)
    O = Odd parity (vertical to DOWN)

    Notice: Vertical lines only connect every other pair!
    This creates the sparse honeycomb connectivity.

    Node parities and vertical directions:
");
    for y in 0..HEIGHT {
        print!("    y={}: ", y);
        for x in 0..WIDTH {
            let i = idx(x, y);
            let dir = if i.count_ones() % 2 == 0 { "^" } else { "v" };
            print!("{:2}({}) ", i, dir);
        }
        println!();
    }
    println!("    ^ = connects UP, v = connects DOWN");
    println!();
}

fn print_syndromes_grid(syndromes: &[u64]) {
    println!("    Syndrome pattern (* = active, ^/v = vertical direction):");
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
            let dir = if i.count_ones() % 2 == 0 { "^" } else { "v" };
            if has_syndrome(syndromes, i) {
                print!(" *{}  ", dir);
            } else {
                print!("  {}  ", dir);
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
                     ux, uy, vertical_direction(c.u));
        } else {
            let (vx, vy) = coords(c.v);
            let edge_type = if ux == vx { "VERTICAL" } else { "horizontal" };
            println!("      - Edge between ({},{}) and ({},{}) [{}]",
                     ux, uy, vx, vy, edge_type);
        }
    }
    println!();
}

fn print_neighbors(x: usize, y: usize) {
    let i = idx(x, y);
    let is_even = i.count_ones() % 2 == 0;

    println!("    Neighbors of ({},{}) [index={}, parity={}, vertical={}]:",
             x, y, i, if is_even { "even" } else { "odd" },
             if is_even { "UP" } else { "DOWN" });

    // Horizontal neighbors (always present if in bounds)
    if x > 0 {
        println!("      - LEFT:  ({},{}) [horizontal - always present]", x - 1, y);
    } else {
        println!("      - LEFT:  (boundary)");
    }
    if x < WIDTH - 1 {
        println!("      - RIGHT: ({},{}) [horizontal - always present]", x + 1, y);
    } else {
        println!("      - RIGHT: (boundary)");
    }

    // Vertical neighbor (parity-dependent)
    if is_even {
        if y > 0 {
            println!("      - UP:    ({},{}) [VERTICAL - even parity]", x, y - 1);
        } else {
            println!("      - UP:    (boundary - even parity but at top)");
        }
        println!("      - DOWN:  (none - even parity connects UP only)");
    } else {
        println!("      - UP:    (none - odd parity connects DOWN only)");
        if y < HEIGHT - 1 {
            println!("      - DOWN:  ({},{}) [VERTICAL - odd parity]", x, y + 1);
        } else {
            println!("      - DOWN:  (boundary - odd parity but at bottom)");
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
║         QUANTUM ERROR CORRECTION TUTORIAL: Honeycomb Lattice              ║
╚═══════════════════════════════════════════════════════════════════════════╝
");

    // =========================================================================
    // STEP 1: Understand the Honeycomb Lattice
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  STEP 1: THE HONEYCOMB LATTICE STRUCTURE");
    println!("═══════════════════════════════════════════════════════════════════════════");

    print_lattice();

    println!("    Why honeycomb codes matter:");
    println!("    - Minimum connectivity (3 neighbors) for 2D codes");
    println!("    - Lower connectivity = simpler error propagation");
    println!("    - Used in Kitaev honeycomb model for topological quantum computing");
    println!("    - Good for theoretical analysis and certain hardware layouts");
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

    let mut decoder: DecodingState<HoneycombGrid, STRIDE_Y> =
        DecodingState::new(&mut arena, WIDTH, HEIGHT, 1);

    println!("    Created decoder for {}x{} grid", WIDTH, HEIGHT);
    println!("    Using HoneycombGrid topology (3-neighbor connectivity)");
    println!("    This is the SPARSEST 2D topology in prav-core!");
    println!();

    let num_blocks = (STRIDE_Y * STRIDE_Y + 63) / 64;
    let mut syndromes = vec![0u64; num_blocks];
    let mut corrections = vec![EdgeCorrection::default(); WIDTH * HEIGHT * 2];

    // =========================================================================
    // SCENARIO 1: Horizontal Error (Always Connected)
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  SCENARIO 1: Horizontal Error (Universal Connection)");
    println!("═══════════════════════════════════════════════════════════════════════════");

    println!("
    An error between horizontally adjacent nodes.
    Horizontal connections exist regardless of parity.

           x=0   x=1   x=2   x=3
            |           |
    y=0 ─── ^ ─── v ─── ^ ─── v ───
                  |           |
    y=1 ─── v ─── * ═X═ * ─── ^ ───  ← Error between (1,1) and (2,1)
            |           |
    y=2 ─── ^ ─── v ─── ^ ─── v ───

    Both nodes have horizontal neighbors regardless of parity.
");

    syndromes.fill(0);
    set_syndrome(&mut syndromes, idx(1, 1)); // Even parity
    set_syndrome(&mut syndromes, idx(2, 1)); // Odd parity

    print_syndromes_grid(&syndromes);

    decoder.sparse_reset();
    decoder.load_dense_syndromes(&syndromes);
    decoder.grow_clusters();
    let count = decoder.peel_forest(&mut corrections);

    print_corrections(&corrections, count);

    println!("    Horizontal edges work just like in square/triangular grids.");
    println!("    All nodes have left and right neighbors (if not at boundary).");
    println!();

    // =========================================================================
    // SCENARIO 2: Vertical Error - Even Parity (Connects UP)
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  SCENARIO 2: Vertical Error - Even Parity Node (Connects UP)");
    println!("═══════════════════════════════════════════════════════════════════════════");

    println!("
    An error on the vertical edge from an EVEN parity node.
    Even nodes connect UP only.

           x=0   x=1   x=2   x=3
            |           |
    y=0 ─── ^ ─── * ─── ^ ─── v ───  ← Syndrome at (1,0), connected from below
                  ║           |
    y=1 ─── v ─── * ─── v ─── ^ ───  ← Syndrome at (1,1), EVEN parity → UP
            |           |
    y=2 ─── ^ ─── v ─── ^ ─── v ───

    Node (1,1) has index 5, popcount=2, EVEN → connects UP to (1,0)
");

    syndromes.fill(0);
    set_syndrome(&mut syndromes, idx(1, 1)); // Even parity - connects UP
    set_syndrome(&mut syndromes, idx(1, 0)); // Its upward neighbor

    print_syndromes_grid(&syndromes);

    println!("    Parity check:");
    println!("      - Node (1,1): index={}, popcount={}, parity={}",
             idx(1, 1), idx(1, 1).count_ones(), parity(idx(1, 1)));
    println!("      - Even parity → vertical neighbor is UP at (1,0)");
    println!();

    decoder.sparse_reset();
    decoder.load_dense_syndromes(&syndromes);
    decoder.grow_clusters();
    let count = decoder.peel_forest(&mut corrections);

    print_corrections(&corrections, count);

    // =========================================================================
    // SCENARIO 3: Vertical Error - Odd Parity (Connects DOWN)
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  SCENARIO 3: Vertical Error - Odd Parity Node (Connects DOWN)");
    println!("═══════════════════════════════════════════════════════════════════════════");

    println!("
    An error on the vertical edge from an ODD parity node.
    Odd nodes connect DOWN only.

           x=0   x=1   x=2   x=3
            |           |
    y=0 ─── ^ ─── * ─── ^ ─── v ───  ← Syndrome at (1,0), ODD parity → DOWN
                  ║           |
    y=1 ─── v ─── * ─── v ─── ^ ───  ← Syndrome at (1,1), connected from above
            |           |
    y=2 ─── ^ ─── v ─── ^ ─── v ───

    Node (1,0) has index 1, popcount=1, ODD → connects DOWN to (1,1)
");

    syndromes.fill(0);
    set_syndrome(&mut syndromes, idx(1, 0)); // Odd parity - connects DOWN
    set_syndrome(&mut syndromes, idx(1, 1)); // Its downward neighbor

    print_syndromes_grid(&syndromes);

    println!("    Parity check:");
    println!("      - Node (1,0): index={}, popcount={}, parity={}",
             idx(1, 0), idx(1, 0).count_ones(), parity(idx(1, 0)));
    println!("      - Odd parity → vertical neighbor is DOWN at (1,1)");
    println!();

    decoder.sparse_reset();
    decoder.load_dense_syndromes(&syndromes);
    decoder.grow_clusters();
    let count = decoder.peel_forest(&mut corrections);

    print_corrections(&corrections, count);

    // =========================================================================
    // SCENARIO 4: Disconnected Vertical Pair (No Direct Connection!)
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  SCENARIO 4: Vertically Aligned but NOT Directly Connected");
    println!("═══════════════════════════════════════════════════════════════════════════");

    println!("
    IMPORTANT: In honeycomb, not all vertical neighbors are connected!

           x=0   x=1   x=2   x=3
            |           |
    y=0 ─── * ─── v ─── ^ ─── v ───  ← Syndrome at (0,0), EVEN → UP (boundary!)
            |     |           |
    y=1 ─── * ─── ^ ─── v ─── ^ ───  ← Syndrome at (0,1), ODD → DOWN
            |           |

    Node (0,0): index=0, popcount=0, EVEN → connects UP (but that's boundary)
    Node (0,1): index=4, popcount=1, ODD → connects DOWN (to y=2, not y=0!)

    These two nodes are NOT directly connected vertically!
    The decoder must find an alternative path (horizontal then vertical).
");

    syndromes.fill(0);
    set_syndrome(&mut syndromes, idx(0, 0)); // Even - connects UP (boundary)
    set_syndrome(&mut syndromes, idx(0, 1)); // Odd - connects DOWN (to y=2)

    print_syndromes_grid(&syndromes);

    println!("    Parity analysis:");
    println!("      - (0,0): index={}, parity={} → connects UP (boundary)",
             idx(0, 0), parity(idx(0, 0)));
    println!("      - (0,1): index={}, parity={} → connects DOWN (to y=2)",
             idx(0, 1), parity(idx(0, 1)));
    println!("      - NO direct vertical edge between them!");
    println!();

    decoder.sparse_reset();
    decoder.load_dense_syndromes(&syndromes);
    decoder.grow_clusters();
    let count = decoder.peel_forest(&mut corrections);

    print_corrections(&corrections, count);

    println!("    The decoder finds the minimum path despite no direct connection.");
    println!("    This might involve boundary matching or horizontal detours.");
    println!();

    // =========================================================================
    // SCENARIO 5: Boundary Matching with Sparse Connectivity
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  SCENARIO 5: Single Syndrome - Boundary Matching");
    println!("═══════════════════════════════════════════════════════════════════════════");

    println!("
    A single syndrome must match to a boundary.
    With only 3 neighbors, there are fewer paths to boundaries.

           x=0   x=1   x=2   x=3
            |           |
    y=0 ─── ^ ─── v ─── ^ ─── v ───
                  |           |
    y=1 ─── v ─── * ─── v ─── ^ ───  ← Single syndrome at (1,1)
            |           |

    Node (1,1) has 3 neighbors: left (0,1), right (2,1), up (1,0)
    Shortest boundary path depends on position.
");

    syndromes.fill(0);
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
    println!("  SUMMARY: Honeycomb Lattice Key Points");
    println!("═══════════════════════════════════════════════════════════════════════════");

    println!("
    1. MINIMAL CONNECTIVITY
       - Only 3 neighbors per node (minimum for 2D QEC)
       - 2 horizontal (left/right) - always present
       - 1 vertical (up OR down) - parity-dependent

    2. PARITY RULE
       - Even parity (popcount even): connects UP
       - Odd parity (popcount odd): connects DOWN
       - Adjacent rows have OPPOSITE connectivity!

    3. HEXAGONAL PATTERN
       - Vertical connections alternate creating hexagons
       - Not all vertically adjacent nodes are connected
       - Must sometimes go horizontal to reach vertical neighbors

    4. DECODER BEHAVIOR
       - Clusters grow more slowly (fewer edges)
       - Some intuitive paths don't exist
       - Decoder automatically finds optimal routes

    5. COMPARISON TO OTHER TOPOLOGIES
       - Square: 4 neighbors (symmetric)
       - Triangular: 6 neighbors (richest connectivity)
       - Honeycomb: 3 neighbors (sparsest connectivity)

    The honeycomb lattice demonstrates that QEC works even with minimal
    connectivity. This is valuable for hardware-constrained implementations
    and provides theoretical insights into the minimum requirements for
    topological error correction.
");
}
