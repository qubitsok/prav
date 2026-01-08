//! Grid topology definitions for QEC lattices.
//!
//! This module defines the connectivity patterns for different quantum error correction
//! code lattices. The topology determines which nodes are neighbors and thus which
//! errors can propagate between them.
//!
//! # Supported Topologies
//!
//! | Topology | Neighbors | Use Case |
//! |----------|-----------|----------|
//! | [`SquareGrid`] | 4 | Surface codes, toric codes |
//! | [`Grid3D`] | 6 | 3D topological codes |
//! | [`TriangularGrid`] | 6 | Color codes, triangular lattices |
//! | [`HoneycombGrid`] | 3 | Honeycomb/hexagonal codes |
//!
//! # Morton Encoding
//!
//! All topologies use Morton (Z-order) encoding for node indices. This interleaves
//! coordinate bits for cache-efficient access:
//!
//! ```text
//! 2D: idx = ...y2x2y1x1y0x0  (x in odd bits, y in even bits)
//! 3D: idx = ...z2y2x2z1y1x1z0y0x0
//! ```
//!
//! Morton encoding ensures spatially close nodes are also close in memory,
//! improving cache utilization during cluster growth.

use crate::intrinsics::{morton_dec, morton_inc};

/// Precomputed neighbor masks for an 8x8 Morton-encoded block.
///
/// For each position `i` in a 64-node block (0..64), `INTRA_BLOCK_NEIGHBORS[i]`
/// is a bitmask where bit `j` is set if node `j` is a neighbor of node `i`
/// AND both nodes are within the same block.
///
/// # Usage
///
/// This table accelerates intra-block cluster growth. Instead of computing
/// neighbors dynamically, we can use a simple lookup and bitwise operations:
///
/// ```ignore
/// // Get neighbors of node i within the block
/// let neighbors = INTRA_BLOCK_NEIGHBORS[i] & occupied_mask;
///
/// // Spread syndrome to all connected neighbors
/// while boundary != 0 {
///     let new_boundary = /* spread using SWAR operations */;
///     // ...
/// }
/// ```
///
/// # Memory Layout
///
/// The table assumes a 2D 8x8 grid laid out in Morton order within each
/// 64-bit block. Neighbors that would cross block boundaries are NOT included
/// in these masks (they are handled separately during inter-block merging).
///
/// # Performance
///
/// Using precomputed masks eliminates the need for coordinate calculations
/// during the hot path of cluster growth, providing significant speedup.
pub static INTRA_BLOCK_NEIGHBORS: [u64; 64] = [
    258,
    517,
    1034,
    2068,
    4136,
    8272,
    16544,
    32832,
    66049,
    132354,
    264708,
    529416,
    1058832,
    2117664,
    4235328,
    8405120,
    16908544,
    33882624,
    67765248,
    135530496,
    271060992,
    542121984,
    1084243968,
    2151710720,
    4328587264,
    8673951744,
    17347903488,
    34695806976,
    69391613952,
    138783227904,
    277566455808,
    550837944320,
    1108118339584,
    2220531646464,
    4441063292928,
    8882126585856,
    17764253171712,
    35528506343424,
    71057012686848,
    141014513745920,
    283678294933504,
    568456101494784,
    1136912202989568,
    2273824405979136,
    4547648811958272,
    9095297623916544,
    18190595247833088,
    36099715518955520,
    72621643502977024,
    145524761982664704,
    291049523965329408,
    582099047930658816,
    1164198095861317632,
    2328396191722635264,
    4656792383445270528,
    9241527172852613120,
    144396663052566528,
    360850920143060992,
    721701840286121984,
    1443403680572243968,
    2886807361144487936,
    5773614722288975872,
    11547229444577951744,
    4647714815446351872,
];

/// Defines the neighbor connectivity pattern for a QEC lattice.
///
/// Each QEC code has a specific lattice structure that determines how qubits
/// are connected. This trait abstracts over different connectivity patterns,
/// allowing the decoder to work with any supported topology.
///
/// # How It Works
///
/// The decoder calls `for_each_neighbor` during cluster growth to find which
/// nodes should be merged when a cluster expands. The topology determines:
///
/// - How many neighbors each node has
/// - Which directions are valid (respecting grid boundaries)
/// - How coordinates map to neighbor indices
///
/// # Morton Coordinates
///
/// Node indices are Morton-encoded, meaning coordinates are interleaved into
/// the index bits. The topology uses bit masks to extract/modify coordinates:
///
/// ```text
/// 2D Morton: idx = ...y₂x₂y₁x₁y₀x₀
///   MASK_X_2D = 0x55555555 (odd bits)
///   MASK_Y_2D = 0xAAAAAAAA (even bits)
///
/// To move right:  idx' = morton_inc(idx, MASK_X_2D)
/// To move down:   idx' = morton_inc(idx, MASK_Y_2D)
/// ```
///
/// # Implementing Custom Topologies
///
/// To support a new lattice type, implement this trait:
///
/// ```ignore
/// #[derive(Clone, Copy)]
/// pub struct MyCustomGrid;
///
/// impl Topology for MyCustomGrid {
///     fn for_each_neighbor<F>(idx: u32, mut f: F)
///     where
///         F: FnMut(u32),
///     {
///         // Call f(neighbor_idx) for each valid neighbor
///     }
/// }
/// ```
pub trait Topology: Copy + 'static {
    /// Calls the provided closure for each valid neighbor of the given node.
    ///
    /// This method iterates over all neighbors of node `idx` that exist within
    /// the grid bounds. Invalid neighbors (those that would be outside the grid)
    /// are automatically skipped.
    ///
    /// # Arguments
    ///
    /// * `idx` - Morton-encoded index of the node whose neighbors to visit.
    /// * `f` - Closure called once per valid neighbor, receiving the neighbor's
    ///   Morton-encoded index.
    ///
    /// # Boundary Handling
    ///
    /// The implementation checks boundary conditions before calling `f`:
    ///
    /// - For left/up neighbors: checks if coordinate bits are non-zero
    /// - For right/down neighbors: checks if increment didn't wrap around
    ///
    /// # Performance
    ///
    /// This method is called frequently during cluster growth. Implementations
    /// should be as fast as possible, using bit manipulation rather than
    /// coordinate arithmetic where feasible.
    fn for_each_neighbor<F>(idx: u32, f: F)
    where
        F: FnMut(u32);
}

/// Mask for X coordinate bits in 2D Morton encoding.
const MASK_X_2D: u32 = 0x55555555;
/// Mask for Y coordinate bits in 2D Morton encoding.
const MASK_Y_2D: u32 = 0xAAAAAAAA;

/// Mask for X coordinate bits in 3D Morton encoding.
const MASK_X_3D: u32 = 0x09249249;
/// Mask for Y coordinate bits in 3D Morton encoding.
const MASK_Y_3D: u32 = 0x12492492;
/// Mask for Z coordinate bits in 3D Morton encoding.
const MASK_Z_3D: u32 = 0x24924924;

/// 2D square lattice topology with 4-neighbor connectivity.
///
/// This is the most common topology for surface codes. Each interior node
/// has exactly 4 neighbors (up, down, left, right). Boundary nodes have
/// fewer neighbors.
///
/// ```text
///     N
///     |
/// W - o - E
///     |
///     S
/// ```
///
/// # Use Cases
///
/// - **Surface codes**: The standard topology for planar surface codes
/// - **Toric codes**: Surface codes with periodic boundary conditions
/// - **Rotated surface codes**: Same connectivity, different physical layout
///
/// # Example
///
/// ```ignore
/// use prav_core::{DecodingState, SquareGrid};
///
/// // Create decoder for 32x32 surface code
/// let mut state: DecodingState<SquareGrid, 32> = DecodingState::new(
///     &mut arena, 32, 32, 1
/// );
/// ```
#[derive(Clone, Copy)]
pub struct SquareGrid;

impl Topology for SquareGrid {
    #[inline(always)]
    fn for_each_neighbor<F>(idx: u32, mut f: F)
    where
        F: FnMut(u32),
    {
        // Left neighbor (decrement X)
        if (idx & MASK_X_2D) != 0 {
            f(morton_dec(idx, MASK_X_2D));
        }
        // Right neighbor (increment X)
        let right = morton_inc(idx, MASK_X_2D);
        if (right & MASK_X_2D) > (idx & MASK_X_2D) {
            f(right);
        }

        // Up neighbor (decrement Y)
        if (idx & MASK_Y_2D) != 0 {
            f(morton_dec(idx, MASK_Y_2D));
        }
        // Down neighbor (increment Y)
        let down = morton_inc(idx, MASK_Y_2D);
        if (down & MASK_Y_2D) > (idx & MASK_Y_2D) {
            f(down);
        }
    }
}

/// 3D cubic lattice topology with 6-neighbor connectivity.
///
/// Extends the square grid to three dimensions. Each interior node has
/// 6 neighbors along the three coordinate axes.
///
/// ```text
///       N
///       |
///   W - o - E
///       |\
///       S B (back)
///         \
///          F (front)
/// ```
///
/// # Use Cases
///
/// - **3D topological codes**: Codes with genuine 3D structure
/// - **Space-time decoding**: Treating time as a third spatial dimension
/// - **Measurement-based QEC**: Where syndrome history forms a 3D structure
///
/// # Coordinate Layout
///
/// Uses 3D Morton encoding where each coordinate gets every third bit:
///
/// ```text
/// idx = ...z₂y₂x₂z₁y₁x₁z₀y₀x₀
/// ```
#[derive(Clone, Copy)]
pub struct Grid3D;

impl Topology for Grid3D {
    #[inline(always)]
    fn for_each_neighbor<F>(idx: u32, mut f: F)
    where
        F: FnMut(u32),
    {
        // X axis neighbors
        if (idx & MASK_X_3D) != 0 {
            f(morton_dec(idx, MASK_X_3D));
        }
        let nx = morton_inc(idx, MASK_X_3D);
        if (nx & MASK_X_3D) > (idx & MASK_X_3D) {
            f(nx);
        }

        // Y axis neighbors
        if (idx & MASK_Y_3D) != 0 {
            f(morton_dec(idx, MASK_Y_3D));
        }
        let ny = morton_inc(idx, MASK_Y_3D);
        if (ny & MASK_Y_3D) > (idx & MASK_Y_3D) {
            f(ny);
        }

        // Z axis neighbors
        if (idx & MASK_Z_3D) != 0 {
            f(morton_dec(idx, MASK_Z_3D));
        }
        let nz = morton_inc(idx, MASK_Z_3D);
        if (nz & MASK_Z_3D) > (idx & MASK_Z_3D) {
            f(nz);
        }
    }
}

/// Triangular lattice topology with 6-neighbor connectivity.
///
/// A 2D lattice where each node has 6 neighbors arranged in a triangular pattern.
/// The connectivity depends on the node's position (even vs odd parity).
///
/// ```text
/// Even parity:     Odd parity:
///   \ /              |
///  - o -           - o -
///    |              / \
/// ```
///
/// # Use Cases
///
/// - **Color codes**: Quantum codes defined on triangular lattices
/// - **Triangular surface codes**: Alternative surface code geometries
///
/// # Parity Determination
///
/// The parity of a node is determined by `idx.count_ones() & 1`:
/// - Even popcount: diagonal neighbors are up-right
/// - Odd popcount: diagonal neighbors are down-left
///
/// This creates the characteristic triangular tiling pattern.
#[derive(Clone, Copy)]
pub struct TriangularGrid;

impl Topology for TriangularGrid {
    #[inline(always)]
    fn for_each_neighbor<F>(idx: u32, mut f: F)
    where
        F: FnMut(u32),
    {
        let has_left = (idx & MASK_X_2D) != 0;
        let left = if has_left {
            morton_dec(idx, MASK_X_2D)
        } else {
            0
        };
        let right = morton_inc(idx, MASK_X_2D);
        let has_right = (right & MASK_X_2D) > (idx & MASK_X_2D);
        let has_up = (idx & MASK_Y_2D) != 0;
        let up = if has_up {
            morton_dec(idx, MASK_Y_2D)
        } else {
            0
        };
        let down = morton_inc(idx, MASK_Y_2D);
        let has_down = (down & MASK_Y_2D) > (idx & MASK_Y_2D);

        // Cardinal neighbors (shared with square grid)
        if has_left {
            f(left);
        }
        if has_right {
            f(right);
        }
        if has_up {
            f(up);
        }
        if has_down {
            f(down);
        }

        // Diagonal neighbors (parity-dependent)
        if (idx.count_ones() & 1) == 0 {
            // Even parity: up-right diagonal
            if has_right && has_up {
                f(morton_dec(right, MASK_Y_2D));
            }
        } else if has_left && has_down {
            // Odd parity: down-left diagonal
            f(morton_inc(left, MASK_Y_2D));
        }
    }
}

/// Honeycomb (hexagonal) lattice topology with 3-neighbor connectivity.
///
/// A 2D lattice where each node has exactly 3 neighbors, forming a honeycomb
/// pattern. The vertical neighbor depends on the node's parity.
///
/// ```text
/// Even parity:     Odd parity:
///     |
///   - o -           - o -
///                     |
/// ```
///
/// # Use Cases
///
/// - **Honeycomb codes**: QEC codes with honeycomb connectivity
/// - **Kitaev honeycomb model**: Certain topological quantum computing schemes
///
/// # Connectivity Pattern
///
/// Each node has:
/// - Two horizontal neighbors (left and right)
/// - One vertical neighbor (up if even parity, down if odd parity)
///
/// This creates the characteristic honeycomb/hexagonal tiling.
#[derive(Clone, Copy)]
pub struct HoneycombGrid;

impl Topology for HoneycombGrid {
    #[inline(always)]
    fn for_each_neighbor<F>(idx: u32, mut f: F)
    where
        F: FnMut(u32),
    {
        // Horizontal neighbors (always present if valid)
        if (idx & MASK_X_2D) != 0 {
            f(morton_dec(idx, MASK_X_2D));
        }
        let right = morton_inc(idx, MASK_X_2D);
        if (right & MASK_X_2D) > (idx & MASK_X_2D) {
            f(right);
        }

        // Vertical neighbor (parity-dependent)
        if (idx.count_ones() & 1) == 0 {
            // Even parity: up neighbor
            if (idx & MASK_Y_2D) != 0 {
                f(morton_dec(idx, MASK_Y_2D));
            }
        } else {
            // Odd parity: down neighbor
            let down = morton_inc(idx, MASK_Y_2D);
            if (down & MASK_Y_2D) > (idx & MASK_Y_2D) {
                f(down);
            }
        }
    }
}
