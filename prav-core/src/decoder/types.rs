// =============================================================================
// Core Types for Quantum Error Correction Decoding
// =============================================================================

/// Represents an edge correction in the decoded output.
///
/// In quantum error correction, corrections are applied along edges of the lattice
/// to restore the code state. Each `EdgeCorrection` identifies either:
///
/// - **Internal edge**: An edge between two lattice nodes `u` and `v`
/// - **Boundary edge**: An edge from node `u` to the boundary (when `v == u32::MAX`)
///
/// # Interpretation
///
/// The decoder outputs corrections as a list of edges. To correct the physical qubits:
///
/// 1. For internal edges `(u, v)`: Apply a correction operator to the data qubit
///    located on the edge between stabilizer nodes `u` and `v`.
/// 2. For boundary edges `(u, MAX)`: Apply a correction to the boundary data qubit
///    adjacent to stabilizer node `u`.
///
/// # Example
///
/// ```ignore
/// let corrections = [
///     EdgeCorrection { u: 5, v: 6 },     // Internal edge between nodes 5 and 6
///     EdgeCorrection { u: 12, v: u32::MAX }, // Boundary correction at node 12
/// ];
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
#[repr(C)]
pub struct EdgeCorrection {
    /// First endpoint of the edge (always a valid node index).
    pub u: u32,
    /// Second endpoint: either a node index or `u32::MAX` for boundary corrections.
    pub v: u32,
}

/// Flag indicating all 64 nodes in a block are valid (valid_mask == !0).
pub const FLAG_VALID_FULL: u32 = 1;

/// Cache-line aligned (64 bytes) block state for hot-path operations.
///
/// Each block represents 64 nodes in Morton order. This structure is carefully
/// laid out to fit in a single cache line for optimal memory access patterns.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(64))]
pub struct BlockStateHot {
    /// Active syndrome boundary - nodes at the frontier of cluster growth.
    pub boundary: u64,
    /// Nodes that have been visited/occupied during growth.
    pub occupied: u64,
    /// Valid and non-erased nodes (valid_mask & !erasure_mask).
    pub effective_mask: u64,
    /// Topology validity bitmap - which nodes exist in the physical grid.
    pub valid_mask: u64,
    /// Erased (lost) qubits that cannot be measured.
    pub erasure_mask: u64,
    /// Cached Union-Find root for this block (u32::MAX if invalid).
    pub root: u32,
    /// State flags (bit 0: FLAG_VALID_FULL).
    pub flags: u32,
    /// Union-Find rank of the cached root (for union-by-rank optimization).
    /// Only valid when `root != u32::MAX`.
    pub root_rank: u8,
    /// Reserved for future use.
    pub _reserved: [u8; 7],
    /// Padding to ensure 64-byte alignment.
    pub _padding: [u8; 8],
}

impl Default for BlockStateHot {
    fn default() -> Self {
        Self {
            boundary: 0,
            occupied: 0,
            effective_mask: 0,
            valid_mask: 0,
            erasure_mask: 0,
            root: u32::MAX,
            flags: 0,
            root_rank: 0,
            _reserved: [0; 7],
            _padding: [0; 8],
        }
    }
}

/// Configuration for boundary checking during cluster growth.
///
/// Controls which edges of the grid are treated as physical boundaries
/// (where defects can be matched to the boundary node). This enables
/// simulation of different boundary conditions:
///
/// - **Open boundaries** (default): All edges are boundaries, defects can match to any edge
/// - **Periodic boundaries**: Some edges are not boundaries (wrap around)
/// - **Mixed**: Custom combinations for specific code geometries
///
/// # Surface Code Example
///
/// In a standard planar surface code, all four edges are typically boundaries.
/// In a toric code (periodic), no edges are boundaries.
///
/// # Default
///
/// By default, all four edges are treated as boundaries (`check_* = true`).
#[derive(Debug, Clone, Copy)]
pub struct BoundaryConfig {
    /// Whether the top edge (y = 0) is a physical boundary.
    pub check_top: bool,
    /// Whether the bottom edge (y = height-1) is a physical boundary.
    pub check_bottom: bool,
    /// Whether the left edge (x = 0) is a physical boundary.
    pub check_left: bool,
    /// Whether the right edge (x = width-1) is a physical boundary.
    pub check_right: bool,
}

impl Default for BoundaryConfig {
    fn default() -> Self {
        Self {
            check_top: true,
            check_bottom: true,
            check_left: true,
            check_right: true,
        }
    }
}
