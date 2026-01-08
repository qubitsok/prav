/// Static graph structure containing grid topology metadata.
///
/// This structure holds precomputed information about the lattice dimensions
/// and memory layout. It enables efficient coordinate calculations during
/// decoding without runtime division or modulo operations.
///
/// # Design Philosophy
///
/// Traditional graph representations store explicit adjacency lists. This decoder
/// instead uses SWAR (SIMD Within A Register) bit operations for neighbor traversal,
/// achieving 19-427x faster performance than lookup tables. The `StaticGraph` stores
/// only the minimal metadata needed to support these operations.
///
/// # Memory Layout
///
/// Nodes are organized in 64-node blocks using Morton (Z-order) encoding.
/// Within each block, nodes are arranged as an 8x8 grid. Blocks themselves
/// are arranged in row-major order.
///
/// ```text
/// Grid Layout (4x4 blocks = 32x32 nodes):
/// +--------+--------+--------+--------+
/// | Blk 0  | Blk 1  | Blk 2  | Blk 3  |  Row 0
/// +--------+--------+--------+--------+
/// | Blk 4  | Blk 5  | Blk 6  | Blk 7  |  Row 1
/// +--------+--------+--------+--------+
/// | Blk 8  | Blk 9  | Blk 10 | Blk 11 |  Row 2
/// +--------+--------+--------+--------+
/// | Blk 12 | Blk 13 | Blk 14 | Blk 15 |  Row 3
/// +--------+--------+--------+--------+
/// ```
#[derive(Debug, Clone, Copy)]
pub struct StaticGraph {
    /// Width of the grid in nodes (not blocks).
    pub width: usize,
    /// Height of the grid in nodes (not blocks).
    pub height: usize,
    /// Depth of the grid for 3D codes (1 for 2D codes).
    pub depth: usize,
    /// Stride to move one position in X direction (always 1).
    pub stride_x: usize,
    /// Stride to move one position in Y direction (typically equals width).
    pub stride_y: usize,
    /// Stride to move one position in Z direction (typically equals width * height).
    pub stride_z: usize,
    /// Number of blocks per row of blocks.
    pub blk_stride_y: usize,
    /// Log2 of stride_y for fast division via bit shift.
    pub shift_y: u32,
    /// Log2 of stride_z for fast division via bit shift.
    pub shift_z: u32,
    /// Bitmask identifying nodes at the right edge of their 8-wide block row.
    pub row_end_mask: u64,
    /// Bitmask identifying nodes at the left edge of their 8-wide block row.
    pub row_start_mask: u64,
}
