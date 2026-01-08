//! Core decoder state for Union Find QEC decoding.
//!
//! This module contains the main decoder state structure that holds all data
//! needed for the decoding process: Union Find parent pointers, block states,
//! active masks, and correction tracking.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::arena::Arena;
use crate::decoder::graph::StaticGraph;
use crate::topology::Topology;

// Re-export types for backward compatibility with imports from decoder::state::*
pub use crate::decoder::types::{BlockStateHot, BoundaryConfig, EdgeCorrection, FLAG_VALID_FULL};

/// Main decoder state structure for Union Find QEC decoding.
///
/// This structure contains all the state needed for a single decoding instance.
/// It is parameterized by:
///
/// - `'a` - Lifetime of the backing arena memory
/// - `T: Topology` - The lattice topology (e.g., [`SquareGrid`](crate::SquareGrid))
/// - `STRIDE_Y` - Compile-time Y stride for performance optimization
///
/// # Memory Layout
///
/// All slices are allocated from an [`Arena`] and are contiguous.
/// The decoder uses Morton (Z-order) encoding to organize nodes into 64-node
/// blocks for cache efficiency.
///
/// # Usage
///
/// ```ignore
/// use prav_core::{Arena, DecodingState, SquareGrid, EdgeCorrection};
///
/// let mut buffer = [0u8; 1024 * 1024];
/// let mut arena = Arena::new(&mut buffer);
///
/// // Create decoder with STRIDE_Y matching grid dimensions
/// let mut state: DecodingState<SquareGrid, 32> = DecodingState::new(&mut arena, 32, 32, 1);
///
/// // Load syndromes and decode
/// state.load_dense_syndromes(&syndromes);
/// state.grow_clusters();
///
/// let mut corrections = [EdgeCorrection::default(); 1024];
/// let count = state.peel_forest(&mut corrections);
/// ```
///
/// # Thread Safety
///
/// `DecodingState` is not thread-safe. For parallel decoding, create one
/// instance per thread with separate arenas.
pub struct DecodingState<'a, T: Topology, const STRIDE_Y: usize> {
    /// Reference to static graph metadata (dimensions, strides).
    pub graph: &'a StaticGraph,
    /// Grid width in nodes.
    pub width: usize,
    /// Grid height in nodes.
    pub height: usize,
    /// Y stride for coordinate calculations (power of 2 for fast division).
    pub stride_y: usize,
    /// Bitmask identifying nodes at the start of their row within a block.
    pub row_start_mask: u64,
    /// Bitmask identifying nodes at the end of their row within a block.
    pub row_end_mask: u64,

    // Morton Layout State
    /// Block state for each 64-node block (boundary, occupied, masks, cached root).
    pub blocks_state: &'a mut [BlockStateHot],

    // Node State
    /// Union Find parent pointers. `parents[i] == i` means node i is a root.
    pub parents: &'a mut [u32],
    /// Bitmask of defect (syndrome) nodes per block.
    pub defect_mask: &'a mut [u64],
    /// Path marking bitmask used during peeling.
    pub path_mark: &'a mut [u64],

    // Sparse Reset Tracking
    /// Bitmask tracking which blocks have been modified (for sparse reset).
    pub block_dirty_mask: &'a mut [u64],

    // Active Set
    /// Bitmask of currently active blocks for this growth iteration.
    pub active_mask: &'a mut [u64],
    /// Bitmask of blocks queued for the next growth iteration.
    pub queued_mask: &'a mut [u64],
    /// Fast-path active mask for small grids (<=64 blocks).
    pub active_block_mask: u64,

    // Ingestion Worklist
    /// List of block indices with syndromes to process.
    pub ingestion_list: &'a mut [u32],
    /// Number of valid entries in `ingestion_list`.
    pub ingestion_count: usize,

    // O(1) Edge Compaction State
    /// Bitmask of edges to include in corrections (XOR-accumulated).
    pub edge_bitmap: &'a mut [u64],
    /// List of edge bitmap word indices that have been modified.
    pub edge_dirty_list: &'a mut [u32],
    /// Number of valid entries in `edge_dirty_list`.
    pub edge_dirty_count: usize,
    /// Bitmask tracking which edge_bitmap words are dirty.
    pub edge_dirty_mask: &'a mut [u64],

    /// Bitmask of boundary corrections per block.
    pub boundary_bitmap: &'a mut [u64],
    /// List of block indices with boundary corrections.
    pub boundary_dirty_list: &'a mut [u32],
    /// Number of valid entries in `boundary_dirty_list`.
    pub boundary_dirty_count: usize,
    /// Bitmask tracking which boundary_bitmap entries are dirty.
    pub boundary_dirty_mask: &'a mut [u64],

    /// BFS predecessor array for path tracing.
    pub bfs_pred: &'a mut [u16],
    /// BFS queue for path tracing.
    pub bfs_queue: &'a mut [u16],

    // AVX/Scalar coordination
    /// Flag indicating scalar fallback is needed for some blocks.
    pub needs_scalar_fallback: bool,
    /// Bitmask of blocks requiring scalar processing.
    pub scalar_fallback_mask: u64,

    /// Configuration for boundary matching behavior.
    pub boundary_config: BoundaryConfig,
    /// Offset for parent array (used in some optimizations).
    pub parent_offset: usize,

    /// Phantom marker for the topology type parameter.
    pub _marker: core::marker::PhantomData<T>,
}

impl<'a, T: Topology, const STRIDE_Y: usize> DecodingState<'a, T, STRIDE_Y> {
    /// Creates a new decoder state for the given grid dimensions.
    ///
    /// Allocates all necessary data structures from the provided arena.
    /// The decoder is initialized and ready for use after construction.
    ///
    /// # Arguments
    ///
    /// * `arena` - Arena allocator to use for all internal allocations.
    /// * `width` - Grid width in nodes.
    /// * `height` - Grid height in nodes.
    /// * `depth` - Grid depth (1 for 2D codes, >1 for 3D codes).
    ///
    /// # Panics
    ///
    /// Panics if `STRIDE_Y` doesn't match the calculated stride for the given
    /// dimensions. The stride is `max(width, height, depth).next_power_of_two()`.
    ///
    /// # Memory Requirements
    ///
    /// The arena must have sufficient space for:
    /// - `num_nodes * 4` bytes for parent array
    /// - `num_blocks * 64` bytes for block states
    /// - Additional space for masks, bitmaps, and queues
    ///
    /// A safe estimate is `num_nodes * 20` bytes. Use [`required_buffer_size`](crate::required_buffer_size)
    /// for exact calculation.
    #[must_use]
    pub fn new(arena: &mut Arena<'a>, width: usize, height: usize, depth: usize) -> Self {
        let is_3d = depth > 1;
        let max_dim = width.max(height).max(if is_3d { depth } else { 1 });
        let dim_pow2 = max_dim.next_power_of_two();

        let stride_x = 1;
        let stride_y = dim_pow2;

        // Runtime check to ensure the const generic matches the physical dimensions
        assert_eq!(
            stride_y, STRIDE_Y,
            "STRIDE_Y const generic ({}) must match calculated stride ({})",
            STRIDE_Y, stride_y
        );

        let stride_z = dim_pow2 * dim_pow2;
        let blk_stride_y = stride_y / 64;
        let shift_y = stride_y.trailing_zeros();
        let shift_z = stride_z.trailing_zeros();

        let mut row_end_mask = 0u64;
        let mut row_start_mask = 0u64;

        if stride_y < 64 {
            let mut i = 0;
            while i < 64 {
                row_start_mask |= 1 << i;
                row_end_mask |= 1 << (i + stride_y - 1);
                i += stride_y;
            }
        }

        let alloc_size = if is_3d {
            dim_pow2 * dim_pow2 * dim_pow2
        } else {
            dim_pow2 * dim_pow2
        };
        let alloc_nodes = alloc_size + 1;
        let num_blocks = alloc_nodes.div_ceil(64);
        let num_bitmask_words = num_blocks.div_ceil(64);

        let graph = StaticGraph {
            width,
            height,
            depth,
            stride_x,
            stride_y,
            stride_z,
            blk_stride_y,
            shift_y,
            shift_z,
            row_end_mask,
            row_start_mask,
        };
        let graph_ref = arena.alloc_value(graph).unwrap();

        let blocks_state = arena
            .alloc_slice_aligned::<BlockStateHot>(num_blocks, 64)
            .unwrap();

        let parents = arena.alloc_slice_aligned::<u32>(alloc_nodes, 64).unwrap();
        let defect_mask = arena.alloc_slice_aligned::<u64>(num_blocks, 64).unwrap();
        let path_mark = arena.alloc_slice_aligned::<u64>(num_blocks, 64).unwrap();

        let block_dirty_mask = arena
            .alloc_slice_aligned::<u64>(num_blocks.div_ceil(64), 64)
            .unwrap();

        let active_mask = arena
            .alloc_slice_aligned::<u64>(num_bitmask_words, 64)
            .unwrap();
        let queued_mask = arena
            .alloc_slice_aligned::<u64>(num_bitmask_words, 64)
            .unwrap();

        let ingestion_list = arena.alloc_slice::<u32>(num_blocks).unwrap();

        let num_edges = alloc_nodes * 3;
        let num_edge_words = num_edges.div_ceil(64);
        let edge_bitmap = arena
            .alloc_slice_aligned::<u64>(num_edge_words, 64)
            .unwrap();
        // Allocate extra space for dirty lists to handle XOR cancellations causing re-insertion
        let edge_dirty_list = arena.alloc_slice::<u32>(num_edge_words * 8).unwrap();

        let boundary_bitmap = arena.alloc_slice_aligned::<u64>(num_blocks, 64).unwrap();
        let boundary_dirty_list = arena.alloc_slice::<u32>(num_blocks * 8).unwrap();

        let edge_dirty_mask = arena
            .alloc_slice_aligned::<u64>(num_edge_words.div_ceil(64), 64)
            .unwrap();
        let boundary_dirty_mask = arena
            .alloc_slice_aligned::<u64>(num_blocks.div_ceil(64), 64)
            .unwrap();

        let bfs_pred = arena.alloc_slice::<u16>(alloc_nodes).unwrap();
        let bfs_queue = arena.alloc_slice::<u16>(alloc_nodes).unwrap();

        let mut decoder = Self {
            graph: graph_ref,
            width,
            height,
            stride_y,
            row_start_mask,
            row_end_mask,
            blocks_state,
            parents,
            defect_mask,
            path_mark,
            block_dirty_mask,
            active_mask,
            queued_mask,
            active_block_mask: 0,
            ingestion_list,
            ingestion_count: 0,
            edge_bitmap,
            edge_dirty_list,
            edge_dirty_count: 0,
            edge_dirty_mask,
            boundary_bitmap,
            boundary_dirty_list,
            boundary_dirty_count: 0,
            boundary_dirty_mask,
            bfs_pred,
            bfs_queue,
            needs_scalar_fallback: false,
            scalar_fallback_mask: 0,
            boundary_config: BoundaryConfig::default(),
            parent_offset: 0,
            _marker: core::marker::PhantomData,
        };

        decoder.initialize_internal();

        decoder.parents[alloc_size] = alloc_size as u32;

        for block in decoder.blocks_state.iter_mut() {
            *block = BlockStateHot::default();
        }

        if is_3d {
            for z in 0..depth {
                for y in 0..height {
                    for x in 0..width {
                        let idx = (z * stride_z) + (y * stride_y) + (x * stride_x);
                        let blk = idx / 64;
                        let bit = idx % 64;
                        if blk < num_blocks {
                            decoder.blocks_state[blk].valid_mask |= 1 << bit;
                        }
                    }
                }
            }
        } else {
            for y in 0..height {
                for x in 0..width {
                    let idx = (y * stride_y) + (x * stride_x);
                    let blk = idx / 64;
                    let bit = idx % 64;
                    if blk < num_blocks {
                        decoder.blocks_state[blk].valid_mask |= 1 << bit;
                    }
                }
            }
        }

        for block in decoder.blocks_state.iter_mut() {
            let valid = block.valid_mask;
            block.effective_mask = valid;
            if valid == !0 {
                block.flags |= FLAG_VALID_FULL;
            } else {
                block.flags &= !FLAG_VALID_FULL;
            }
        }

        decoder
    }

    /// Resets all internal state for a new decoding cycle.
    ///
    /// This is a full reset that clears all dynamic state while preserving
    /// the grid topology (valid_mask). Call this before loading new syndromes
    /// when you want to completely reset the decoder.
    ///
    /// # Performance Note
    ///
    /// For repeated decoding with sparse syndromes, prefer [`sparse_reset`](Self::sparse_reset)
    /// which only resets modified blocks (O(modified) vs O(n)).
    pub fn initialize_internal(&mut self) {
        for block in self.blocks_state.iter_mut() {
            block.boundary = 0;
            block.occupied = 0;
            block.root = u32::MAX;
            block.root_rank = 0;
            // effective_mask and flags are persistent across resets (topology doesn't change)
            // But if we wanted to fully reset everything we would need to recompute effective_mask.
            // Assuming topology is static, we keep effective_mask and flags.
        }
        self.defect_mask.fill(0);
        self.path_mark.fill(0);
        self.block_dirty_mask.fill(0);

        self.active_mask.fill(0);
        self.queued_mask.fill(0);
        self.active_block_mask = 0;

        self.edge_bitmap.fill(0);
        self.edge_dirty_count = 0;
        self.edge_dirty_mask.fill(0);
        self.boundary_bitmap.fill(0);
        self.boundary_dirty_count = 0;
        self.boundary_dirty_mask.fill(0);

        self.ingestion_count = 0;

        for (i, p) in self.parents.iter_mut().enumerate() {
            *p = i as u32;
        }
    }

    /// Loads erasure information indicating which qubits were lost.
    ///
    /// Erasures represent qubits that could not be measured (e.g., photon loss).
    /// The decoder excludes erased qubits from cluster growth.
    ///
    /// # Arguments
    ///
    /// * `erasures` - Dense bitarray where bit `i` in `erasures[blk]` indicates
    ///   node `(blk * 64 + i)` is erased.
    ///
    /// # Effect
    ///
    /// Updates `effective_mask = valid_mask & !erasure_mask` for each block,
    /// which controls which nodes participate in cluster growth.
    pub fn load_erasures(&mut self, erasures: &[u64]) {
        let len = erasures.len().min(self.blocks_state.len());
        for (i, &val) in erasures.iter().take(len).enumerate() {
            self.blocks_state[i].erasure_mask = val;
            let valid = self.blocks_state[i].valid_mask;
            self.blocks_state[i].effective_mask = valid & !val;
        }
        for block in self.blocks_state[len..].iter_mut() {
            block.erasure_mask = 0;
            let valid = block.valid_mask;
            block.effective_mask = valid;
        }
    }

    /// Marks a block as modified for sparse reset tracking.
    ///
    /// Called internally when a block's state changes. Marked blocks will be
    /// reset during the next [`sparse_reset`](Self::sparse_reset) call.
    #[inline(always)]
    pub fn mark_block_dirty(&mut self, blk_idx: usize) {
        let mask_idx = blk_idx >> 6;
        let mask_bit = blk_idx & 63;
        unsafe {
            *self.block_dirty_mask.get_unchecked_mut(mask_idx) |= 1 << mask_bit;
        }
    }

    /// Checks if this grid qualifies for small-grid optimizations.
    ///
    /// Small grids (<=64 blocks, i.e., <=4096 nodes) use a single `u64` bitmask
    /// for active block tracking, enabling faster iteration.
    ///
    /// # Returns
    ///
    /// `true` if the grid has at most 65 blocks (64 data + 1 boundary sentinel).
    #[inline(always)]
    pub fn is_small_grid(&self) -> bool {
        // `active_block_mask` is a single `u64`, so we can only track up to 64 data blocks.
        // `blocks_state` may include one extra sentinel block for the boundary node.
        self.blocks_state.len() <= 65
    }

    /// Queues a block for processing in the next growth iteration.
    ///
    /// Sets the corresponding bit in `queued_mask`. The block will be processed
    /// when `active_mask` and `queued_mask` are swapped at the end of the current
    /// iteration.
    #[inline(always)]
    pub fn push_next(&mut self, blk_idx: usize) {
        let mask_idx = blk_idx >> 6;
        let mask_bit = blk_idx & 63;
        unsafe {
            *self.queued_mask.get_unchecked_mut(mask_idx) |= 1 << mask_bit;
        }
    }

    /// Resets only the blocks that were modified, enabling efficient reuse.
    ///
    /// At typical error rates (p=0.001-0.06), only a small fraction of blocks
    /// are modified during decoding. This method resets only those blocks,
    /// achieving O(modified) complexity instead of O(total).
    ///
    /// # When to Use
    ///
    /// Call this between decoding cycles when:
    /// - Error rate is low (most blocks unmodified)
    /// - You want to minimize reset overhead
    ///
    /// For high error rates or when topology changes, use
    /// [`initialize_internal`](Self::initialize_internal) instead.
    ///
    /// # What Gets Reset
    ///
    /// For each dirty block:
    /// - `boundary`, `occupied`, `root`, `root_rank` in `BlockStateHot`
    /// - `defect_mask` entry
    /// - Parent pointers for all 64 nodes in the block
    pub fn sparse_reset(&mut self) {
        for (word_idx, word_ref) in self.block_dirty_mask.iter_mut().enumerate() {
            let mut w = *word_ref;
            *word_ref = 0;
            while w != 0 {
                let bit = w.trailing_zeros();
                w &= w - 1;
                let blk_idx = word_idx * 64 + bit as usize;

                // SAFETY: blk_idx is derived from bits set in block_dirty_mask,
                // which is only modified by mark_block_dirty() with valid block indices.
                // Therefore blk_idx < blocks_state.len() and blk_idx < defect_mask.len().
                unsafe {
                    let block = self.blocks_state.get_unchecked_mut(blk_idx);
                    block.boundary = 0;
                    block.occupied = 0;
                    block.root = u32::MAX;
                    block.root_rank = 0;
                    *self.defect_mask.get_unchecked_mut(blk_idx) = 0;
                }

                let start_node = blk_idx * 64;
                let end_node = (start_node + 64).min(self.parents.len());
                for node in start_node..end_node {
                    // SAFETY: node is bounded by min(blk_idx*64+64, parents.len()),
                    // so node < parents.len() is guaranteed.
                    unsafe {
                        *self.parents.get_unchecked_mut(node) = node as u32;
                    }
                }
            }
        }

        self.queued_mask.fill(0);
        self.active_mask.fill(0);
        self.active_block_mask = 0;

        let boundary_idx = self.parents.len() - 1;
        self.parents[boundary_idx] = boundary_idx as u32;
    }

    /// Resets state for the next decoding cycle (sparse reset).
    ///
    /// This efficiently resets only the blocks that were modified during
    /// the previous decoding cycle. At typical error rates (p < 0.1),
    /// this is significantly faster than [`full_reset`](Self::full_reset).
    ///
    /// Use this method between consecutive decoding cycles when the
    /// grid topology remains unchanged.
    ///
    /// # Complexity
    ///
    /// O(modified_blocks) where modified_blocks << total_blocks for typical error rates.
    ///
    /// # Example
    ///
    /// ```ignore
    /// for cycle in 0..1000 {
    ///     decoder.reset_for_next_cycle();
    ///     decoder.load_dense_syndromes(&syndromes[cycle]);
    ///     let count = decoder.decode(&mut corrections);
    /// }
    /// ```
    #[inline]
    pub fn reset_for_next_cycle(&mut self) {
        self.sparse_reset();
    }

    /// Fully resets all decoder state.
    ///
    /// This performs a complete reset of all internal data structures,
    /// suitable for when:
    /// - Starting fresh with a completely new problem
    /// - The grid topology has changed
    /// - You want guaranteed clean state
    ///
    /// For repeated decoding with similar syndrome patterns, prefer
    /// [`reset_for_next_cycle`](Self::reset_for_next_cycle) which is faster.
    ///
    /// # Complexity
    ///
    /// O(n) where n is the total number of nodes.
    #[inline]
    pub fn full_reset(&mut self) {
        self.initialize_internal();
    }
}

// Forwarding methods to traits - these provide convenient direct access
// without needing to import the traits.
impl<'a, T: Topology, const STRIDE_Y: usize> DecodingState<'a, T, STRIDE_Y> {
    /// Finds the cluster root for node `i`. See [`UnionFind::find`](crate::UnionFind::find).
    #[inline(always)]
    pub fn find(&mut self, i: u32) -> u32 {
        use super::union_find::UnionFind;
        UnionFind::find(self, i)
    }

    /// Merges clusters containing nodes `u` and `v`.
    ///
    /// See [`UnionFind::union`](crate::UnionFind::union) for details.
    ///
    /// # Safety
    ///
    /// Caller must ensure `u` and `v` are valid node indices within the parents array.
    #[inline(always)]
    pub unsafe fn union(&mut self, u: u32, v: u32) -> bool {
        use super::union_find::UnionFind;
        UnionFind::union(self, u, v)
    }

    /// Loads syndrome measurements. See [`ClusterGrowth::load_dense_syndromes`](crate::ClusterGrowth::load_dense_syndromes).
    #[inline(always)]
    pub fn load_dense_syndromes(&mut self, syndromes: &[u64]) {
        use super::growth::ClusterGrowth;
        ClusterGrowth::load_dense_syndromes(self, syndromes)
    }

    /// Expands clusters until convergence. See [`ClusterGrowth::grow_clusters`](crate::ClusterGrowth::grow_clusters).
    #[inline(always)]
    pub fn grow_clusters(&mut self) {
        use super::growth::ClusterGrowth;
        ClusterGrowth::grow_clusters(self)
    }

    /// Processes a single block during growth.
    ///
    /// See [`ClusterGrowth::process_block`](crate::ClusterGrowth::process_block) for details.
    ///
    /// # Safety
    ///
    /// Caller must ensure `blk_idx` is within bounds of the internal blocks state.
    #[inline(always)]
    pub unsafe fn process_block(&mut self, blk_idx: usize) -> bool {
        use super::growth::ClusterGrowth;
        ClusterGrowth::process_block(self, blk_idx)
    }

    /// Full decode: growth + peeling. See [`Peeling::decode`](crate::Peeling::decode).
    #[inline(always)]
    pub fn decode(&mut self, corrections: &mut [EdgeCorrection]) -> usize {
        use super::peeling::Peeling;
        Peeling::decode(self, corrections)
    }

    /// Extracts corrections from grown clusters. See [`Peeling::peel_forest`](crate::Peeling::peel_forest).
    #[inline(always)]
    pub fn peel_forest(&mut self, corrections: &mut [EdgeCorrection]) -> usize {
        use super::peeling::Peeling;
        Peeling::peel_forest(self, corrections)
    }
}
