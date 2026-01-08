//! Tiled decoder for large grids.
//!
//! This module provides [`TiledDecodingState`], which breaks large grids into
//! 32x32 tiles for improved cache locality and SWAR efficiency. Each tile is
//! processed using the optimized Stride-32 path, then tiles are stitched together.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::arena::Arena;
use crate::decoder::state::DecodingState;
use crate::decoder::types::{BlockStateHot, BoundaryConfig, EdgeCorrection, FLAG_VALID_FULL};
use crate::intrinsics::{morton_decode_2d, morton_encode_2d, tzcnt};
use crate::topology::Topology;

/// A tiled decoder that manages large grids by breaking them into 32x32 tiles.
///
/// For grids larger than ~4096 nodes, the standard decoder's cache utilization
/// degrades. `TiledDecodingState` addresses this by:
///
/// 1. Dividing the grid into 32x32 tiles (1024 nodes each)
/// 2. Running the optimized Stride-32 decoder on each tile
/// 3. Stitching tiles together at boundaries using Union Find
///
/// # When to Use
///
/// Use `TiledDecodingState` for grids larger than 64x64 nodes. For smaller grids,
/// the standard [`DecodingState`] is more efficient.
///
/// # Tile Layout
///
/// ```text
/// Global grid (96x64):
/// +--------+--------+--------+
/// | Tile 0 | Tile 1 | Tile 2 |  (each 32x32)
/// +--------+--------+--------+
/// | Tile 3 | Tile 4 | Tile 5 |
/// +--------+--------+--------+
/// ```
///
/// Each tile contains 16 blocks of 64 nodes, arranged in Stride-32 format.
pub struct TiledDecodingState<'a, T: Topology> {
    /// Total grid width in nodes.
    pub width: usize,
    /// Total grid height in nodes.
    pub height: usize,

    // Tiling configuration
    /// Number of tiles in X direction.
    pub tiles_x: usize,
    /// Number of tiles in Y direction.
    pub tiles_y: usize,

    // We hold the raw memory, but organized in tiles.
    // Each tile has 16 blocks (1024 nodes).
    // Total blocks = tiles_x * tiles_y * 16.
    /// Block state for all tiles (16 blocks per tile).
    pub blocks_state: &'a mut [BlockStateHot],
    /// Union Find parent pointers for all nodes.
    pub parents: &'a mut [u32],

    // Auxiliary state needed for DecodingState
    /// Defect mask per block.
    pub defect_mask: &'a mut [u64],
    /// Path marking for peeling.
    pub path_mark: &'a mut [u64],
    /// Dirty block tracking for sparse reset.
    pub block_dirty_mask: &'a mut [u64],
    /// Currently active blocks.
    pub active_mask: &'a mut [u64],
    /// Blocks queued for next iteration.
    pub queued_mask: &'a mut [u64],
    /// Syndrome ingestion worklist.
    pub ingestion_list: &'a mut [u32],
    /// Edge correction bitmap.
    pub edge_bitmap: &'a mut [u64],
    /// Dirty edge word list.
    pub edge_dirty_list: &'a mut [u32],
    /// Dirty edge mask.
    pub edge_dirty_mask: &'a mut [u64],
    /// Boundary correction bitmap.
    pub boundary_bitmap: &'a mut [u64],
    /// Dirty boundary block list.
    pub boundary_dirty_list: &'a mut [u32],
    /// Dirty boundary mask.
    pub boundary_dirty_mask: &'a mut [u64],
    /// BFS predecessor array for path tracing.
    pub bfs_pred: &'a mut [u16],
    /// BFS queue for path tracing.
    pub bfs_queue: &'a mut [u16],

    // Static graph for tiles (shared because all tiles are 32x32 Stride 32)
    /// Static graph metadata for 32x32 tiles (shared by all tiles).
    pub tile_graph: &'a crate::decoder::graph::StaticGraph,

    /// Syndrome ingestion count.
    pub ingestion_count: usize,
    /// Active block mask (for small-grid compatibility).
    pub active_block_mask: u64,
    /// Edge dirty count.
    pub edge_dirty_count: usize,
    /// Boundary dirty count.
    pub boundary_dirty_count: usize,

    /// Phantom marker for topology type.
    pub _marker: core::marker::PhantomData<T>,
}

impl<'a, T: Topology> TiledDecodingState<'a, T> {
    /// Creates a new tiled decoder for the given grid dimensions.
    ///
    /// # Arguments
    ///
    /// * `arena` - Arena allocator for all internal allocations.
    /// * `width` - Total grid width in nodes.
    /// * `height` - Total grid height in nodes.
    ///
    /// # Tile Calculation
    ///
    /// The grid is divided into ceil(width/32) x ceil(height/32) tiles.
    /// Each tile contains 1024 nodes (32x32) in 16 blocks.
    pub fn new(arena: &mut Arena<'a>, width: usize, height: usize) -> Self {
        let tiles_x = width.div_ceil(32);
        let tiles_y = height.div_ceil(32);
        let num_tiles = tiles_x * tiles_y;

        let blocks_per_tile = 16;
        let nodes_per_tile = 1024;

        let total_blocks = num_tiles * blocks_per_tile;
        let total_nodes = num_tiles * nodes_per_tile + 1; // +1 for boundary

        // Allocate Memory
        let blocks_state = arena
            .alloc_slice_aligned::<BlockStateHot>(total_blocks, 64)
            .unwrap();
        let parents = arena.alloc_slice_aligned::<u32>(total_nodes, 64).unwrap();

        let defect_mask = arena.alloc_slice_aligned::<u64>(total_blocks, 64).unwrap();
        let path_mark = arena.alloc_slice_aligned::<u64>(total_blocks, 64).unwrap();

        let block_dirty_mask = arena
            .alloc_slice_aligned::<u64>(total_blocks.div_ceil(64), 64)
            .unwrap();

        let num_bitmask_words = total_blocks.div_ceil(64);
        let active_mask = arena
            .alloc_slice_aligned::<u64>(num_bitmask_words, 64)
            .unwrap();
        let queued_mask = arena
            .alloc_slice_aligned::<u64>(num_bitmask_words, 64)
            .unwrap();

        let ingestion_list = arena.alloc_slice::<u32>(total_blocks).unwrap();

        // Edge/Boundary tracking (sized for total capacity)
        let num_edges = total_nodes * 3;
        let num_edge_words = num_edges.div_ceil(64);
        let edge_bitmap = arena
            .alloc_slice_aligned::<u64>(num_edge_words, 64)
            .unwrap();
        let edge_dirty_list = arena.alloc_slice::<u32>(num_edge_words * 8).unwrap();
        let edge_dirty_mask = arena
            .alloc_slice_aligned::<u64>(num_edge_words.div_ceil(64), 64)
            .unwrap();

        let boundary_bitmap = arena.alloc_slice_aligned::<u64>(total_blocks, 64).unwrap();
        let boundary_dirty_list = arena.alloc_slice::<u32>(total_blocks * 8).unwrap();
        let boundary_dirty_mask = arena
            .alloc_slice_aligned::<u64>(total_blocks.div_ceil(64), 64)
            .unwrap();

        let bfs_pred = arena.alloc_slice::<u16>(total_nodes).unwrap();
        let bfs_queue = arena.alloc_slice::<u16>(total_nodes).unwrap();

        // Create a StaticGraph for a 32x32 tile (Stride 32)
        // Neighbor traversal uses SWAR bit operations (spread_syndrome_*) rather than lookup tables
        let tile_graph = crate::decoder::graph::StaticGraph {
            width: 32,
            height: 32,
            depth: 1,
            stride_x: 1,
            stride_y: 32,
            stride_z: 1024,
            blk_stride_y: 0, // Not used for Stride 32
            shift_y: 5,
            shift_z: 10,
            row_end_mask: 0x8000000080000000,
            row_start_mask: 0x0000000100000001,
        };
        let tile_graph_ref = arena.alloc_value(tile_graph).unwrap();

        let mut state = Self {
            width,
            height,
            tiles_x,
            tiles_y,
            blocks_state,
            parents,
            defect_mask,
            path_mark,
            block_dirty_mask,
            active_mask,
            queued_mask,
            ingestion_list,
            edge_bitmap,
            edge_dirty_list,
            edge_dirty_mask,
            boundary_bitmap,
            boundary_dirty_list,
            boundary_dirty_mask,
            bfs_pred,
            bfs_queue,
            tile_graph: tile_graph_ref,
            ingestion_count: 0,
            active_block_mask: 0,
            edge_dirty_count: 0,
            boundary_dirty_count: 0,
            _marker: core::marker::PhantomData,
        };

        state.initialize();
        state
    }

    /// Initializes or reinitializes the decoder state.
    ///
    /// Sets up valid masks for all tiles based on actual grid dimensions
    /// and resets all dynamic state.
    pub fn initialize(&mut self) {
        for block in self.blocks_state.iter_mut() {
            *block = BlockStateHot::default();
        }

        // Initialize parents
        for (i, p) in self.parents.iter_mut().enumerate() {
            *p = i as u32;
        }

        // Initialize valid masks based on actual width/height
        let _total_blocks = self.tiles_x * self.tiles_y * 16;

        for ty in 0..self.tiles_y {
            for tx in 0..self.tiles_x {
                let tile_idx = ty * self.tiles_x + tx;
                let block_offset = tile_idx * 16;

                // Determine valid region for this tile
                let tile_base_x = tx * 32;
                let tile_base_y = ty * 32;

                for i in 0..1024 {
                    let lx = i % 32;
                    let ly = i / 32;

                    let gx = tile_base_x + lx;
                    let gy = tile_base_y + ly;

                    if gx < self.width && gy < self.height {
                        let blk = block_offset + (i / 64);
                        let bit = i % 64;
                        self.blocks_state[blk].valid_mask |= 1 << bit;
                    }
                }
            }
        }

        for block in self.blocks_state.iter_mut() {
            let valid = block.valid_mask;
            block.effective_mask = valid;
            if valid == !0 {
                block.flags |= FLAG_VALID_FULL;
            }
        }
    }

    /// Resets only modified blocks for efficient reuse.
    ///
    /// Similar to [`DecodingState::sparse_reset`], this only resets blocks
    /// that were touched during the previous decoding cycle.
    pub fn sparse_reset(&mut self) {
        for (word_idx, word_ref) in self.block_dirty_mask.iter_mut().enumerate() {
            let mut w = *word_ref;
            *word_ref = 0;
            while w != 0 {
                let bit = w.trailing_zeros();
                w &= w - 1;
                let blk_idx = word_idx * 64 + bit as usize;

                unsafe {
                    let block = self.blocks_state.get_unchecked_mut(blk_idx);
                    block.boundary = 0;
                    block.occupied = 0;
                    block.root = u32::MAX;
                    *self.defect_mask.get_unchecked_mut(blk_idx) = 0;
                }

                // Reset parents for this block
                // But parents are indexed globally?
                // In TiledDecodingState, parents are laid out in Tile Major order.
                // So Block 0 corresponds to parents 0..63.
                // It matches simply.
                let start_node = blk_idx * 64;
                let end_node = (start_node + 64).min(self.parents.len());
                for node in start_node..end_node {
                    unsafe {
                        *self.parents.get_unchecked_mut(node) = node as u32;
                    }
                }
            }
        }

        self.queued_mask.fill(0);
        self.active_mask.fill(0);

        let boundary_idx = self.parents.len() - 1;
        self.parents[boundary_idx] = boundary_idx as u32;
    }

    /// Loads syndrome measurements from a dense row-major bitarray.
    ///
    /// Converts from row-major input format to the internal tiled format.
    ///
    /// # Arguments
    ///
    /// * `syndromes` - Dense bitarray in row-major order with power-of-2 stride.
    pub fn load_dense_syndromes(&mut self, syndromes: &[u64]) {
        // Input `syndromes` is Row-Major (Stride = Width.next_power_of_two()).
        // We need to map to Tiled Layout.

        let max_dim = self.width.max(self.height);
        let stride_y = max_dim.next_power_of_two();
        let stride_shift = stride_y.trailing_zeros();
        let stride_mask = stride_y - 1;
        
        let _blk_stride = stride_y / 64; // Blocks per row in input

        // This is slow (scalar), but it's just loading.
        // Iterate over set bits in syndromes and map to tiled.

        for (blk_idx, &word) in syndromes.iter().enumerate() {
            if word == 0 {
                continue;
            }

            // Input Block covers 64 bits. Stride is `stride_y`.
            // Wait, input format depends on `generate_defects`.
            // In growth_bench, stride is power of 2.
            // Let's assume input matches that.

            // Map each bit
            let mut w = word;
            while w != 0 {
                let bit = w.trailing_zeros();
                w &= w - 1;

                let global_idx = blk_idx * 64 + bit as usize;
                let gy = global_idx >> stride_shift;
                let gx = global_idx & stride_mask;

                if gx >= self.width || gy >= self.height {
                    continue;
                }

                let tx = gx / 32;
                let ty = gy / 32;
                let lx = gx % 32;
                let ly = gy % 32;

                let tile_idx = ty * self.tiles_x + tx;
                let local_idx = ly * 32 + lx;

                let target_blk = tile_idx * 16 + (local_idx / 64);
                let target_bit = local_idx % 64;

                // Mark dirty
                let mask_idx = target_blk >> 6;
                let mask_bit = target_blk & 63;
                unsafe {
                    *self.block_dirty_mask.get_unchecked_mut(mask_idx) |= 1 << mask_bit;
                    let block = self.blocks_state.get_unchecked_mut(target_blk);
                    block.boundary |= 1 << target_bit;
                    block.occupied |= 1 << target_bit;
                    *self.defect_mask.get_unchecked_mut(target_blk) |= 1 << target_bit;

                    // Mark active
                    let active_word = target_blk >> 6;
                    let active_bit = target_blk & 63;
                    *self.active_mask.get_unchecked_mut(active_word) |= 1 << active_bit;
                }
            }
        }
    }

    /// Expands cluster boundaries until convergence.
    ///
    /// Processes tiles in two phases:
    /// 1. **Intra-tile growth**: Run Stride-32 decoder within each tile
    /// 2. **Inter-tile stitching**: Connect clusters across tile boundaries
    pub fn grow_clusters(&mut self) {
        let max_cycles = self.width.max(self.height) * 16;

        // Extract raw pointers to circumvent borrow checker in the loop.
        // Safety: We manage disjoint access patterns manually.
        let active_mask_ptr = self.active_mask.as_mut_ptr();

        for _ in 0..max_cycles {
            if self.active_mask.iter().all(|&w| w == 0) {
                break;
            }

            // 1. Intra-Tile Growth (Phase 1)
            // Iterate over all tiles
            for ty in 0..self.tiles_y {
                for tx in 0..self.tiles_x {
                    let tile_idx = ty * self.tiles_x + tx;

                    // Check if tile has active blocks
                    let start_blk = tile_idx * 16;
                    let end_blk = start_blk + 16;

                    // Quick check if tile is active using raw pointer to avoid full borrow
                    let mut tile_active = false;
                    for blk in start_blk..end_blk {
                        let word_idx = blk >> 6;
                        let bit = blk & 63;
                        unsafe {
                            if (*active_mask_ptr.add(word_idx) & (1 << bit)) != 0 {
                                tile_active = true;
                                break;
                            }
                        }
                    }

                    if !tile_active {
                        continue;
                    }

                    unsafe {
                        self.process_tile_unsafe(tile_idx, tx, ty);
                    }
                }
            }

            // 2. Inter-Tile Stitching (Phase 2)
            unsafe {
                self.stitch_tiles();
            }

            // Swap queues
            core::mem::swap(&mut self.active_mask, &mut self.queued_mask);
            self.queued_mask.fill(0);
        }
    }

    unsafe fn process_tile_unsafe(&mut self, tile_idx: usize, tx: usize, ty: usize) {
        let block_offset = tile_idx * 16;
        let parent_offset = tile_idx * 1024; // 32x32 nodes

        // Prepare Pointers for DecodingState reconstruction
        // We cast to lifetime 'a effectively.
        let blocks_ptr = self.blocks_state.as_mut_ptr().add(block_offset);
        let blocks_slice = core::slice::from_raw_parts_mut(blocks_ptr, 16);

        // Pass full parents slice.
        // We calculate parents_len to be safe, but decoder uses len() - 1 for boundary.
        // We want decoder.parents[decoder.parents.len()-1] to be the Global Boundary.
        // So we pass the full slice from index 0.
        // Wait, self.parents is contiguous.
        let parents_ptr = self.parents.as_mut_ptr();
        let parents_len = self.parents.len();
        let parents_slice = core::slice::from_raw_parts_mut(parents_ptr, parents_len);

        let queued_mask_slice = core::slice::from_raw_parts_mut(self.queued_mask.as_mut_ptr(), self.queued_mask.len());
        let dirty_mask_slice = core::slice::from_raw_parts_mut(self.block_dirty_mask.as_mut_ptr(), self.block_dirty_mask.len());

        // Disjoint slices for auxiliary
        // These are not actually used by process_block logic except dirty_masks.
        // We pass empty slices for unused ones.

        // Config
        let config = BoundaryConfig {
            check_left: tx == 0,
            check_right: tx == self.tiles_x - 1,
            check_top: ty == 0,
            check_bottom: ty == self.tiles_y - 1,
        };

        // Create manual DecodingState
        // Note: active_mask/queued_mask are passed but we will override them inside the loop
        // with stack buffers to avoid global indexing confusion.
        let mut decoder = DecodingState::<T, 32> {
            graph: self.tile_graph,
            width: 32,
            height: 32,
            stride_y: 32,
            row_start_mask: 0x0000000100000001,
            row_end_mask: 0x8000000080000000,

            blocks_state: blocks_slice,
            parents: parents_slice,

            defect_mask: &mut [],
            path_mark: &mut [],
            block_dirty_mask: dirty_mask_slice, // Global

            active_mask: &mut [],
            queued_mask: queued_mask_slice, // Global
            active_block_mask: 0,

            ingestion_list: &mut [],
            ingestion_count: 0,

            edge_bitmap: &mut [],
            edge_dirty_list: &mut [],
            edge_dirty_count: 0,
            edge_dirty_mask: &mut [],

            boundary_bitmap: &mut [],
            boundary_dirty_list: &mut [],
            boundary_dirty_count: 0,
            boundary_dirty_mask: &mut [],

            bfs_pred: &mut [],
            bfs_queue: &mut [],

            needs_scalar_fallback: false,
            scalar_fallback_mask: 0,
            boundary_config: config,
            parent_offset,
            _marker: core::marker::PhantomData,
        };

        // Pointers for global masks to merge back results
        // let global_queued_ptr = self.queued_mask.as_mut_ptr();
        // let global_dirty_ptr = self.block_dirty_mask.as_mut_ptr();
        let global_active_ptr = self.active_mask.as_ptr(); // Read-only access to verify activation

        // Loop over tile blocks
        let start_blk = tile_idx * 16;
        for i in 0..16 {
            let global_blk = start_blk + i;
            let word_idx = global_blk >> 6;
            let bit = global_blk & 63;

            let is_active = (*global_active_ptr.add(word_idx) & (1 << bit)) != 0;
            if is_active {
                // decoder.process_block writes directly to global masks using offset logic in portable_32.rs
                decoder.process_block(i);
            }
        }
    }

    unsafe fn stitch_tiles(&mut self) {
        // Generic Stitching using T::for_each_neighbor
        let blocks_ptr = self.blocks_state.as_mut_ptr();

        for ty in 0..self.tiles_y {
            for tx in 0..self.tiles_x {
                let tile_idx = ty * self.tiles_x + tx;
                let base_gx = tx * 32;
                let base_gy = ty * 32;

                // Iterate boundary pixels of this tile
                // Boundaries: Row 0, Row 31, Col 0, Col 31.
                // Duplicate checks are fine (UnionFind handles it).
                
                // Helper closure to process a pixel
                let mut process_pixel = |lx: usize, ly: usize| {
                    let gx = base_gx + lx;
                    let gy = base_gy + ly;
                    
                    if gx >= self.width || gy >= self.height {
                        return;
                    }

                    // My Global Morton Index
                    let m_idx = morton_encode_2d(gx as u32, gy as u32);
                    
                    // My Tiled Node Index
                    let my_blk_offset = tile_idx * 16;
                    let my_local = ly * 32 + lx;
                    let my_node = (tile_idx * 1024) + my_local;
                    
                    // Check if I am active
                    let my_blk = my_blk_offset + (my_local / 64);
                    let my_bit = my_local % 64;
                    let my_block = &*blocks_ptr.add(my_blk);
                    let my_active = (my_block.occupied & (1 << my_bit)) != 0;

                    // Iterate Neighbors
                    T::for_each_neighbor(m_idx, |n_m| {
                         let (nx, ny) = morton_decode_2d(n_m);
                         // Check if neighbor is out of bounds or in DIFFERENT tile
                         if nx as usize >= self.width || ny as usize >= self.height {
                             return;
                         }

                         let n_tx = (nx / 32) as usize;
                         let n_ty = (ny / 32) as usize;
                         
                         // We only stitch inter-tile edges
                         if n_tx == tx && n_ty == ty {
                             return;
                         }
                         
                         // Enforce order to avoid double locking/processing if possible
                         // (u, v) vs (v, u).
                         // Simple check: process only if neighbor is "larger" coordinate?
                         // But n_m vs m_idx works.
                         if n_m < m_idx {
                             return;
                         }

                         let n_tile_idx = n_ty * self.tiles_x + n_tx;
                         let n_lx = (nx % 32) as usize;
                         let n_ly = (ny % 32) as usize;
                         
                         let n_local = n_ly * 32 + n_lx;
                         let n_blk = n_tile_idx * 16 + (n_local / 64);
                         let n_bit = n_local % 64;
                         
                         let n_block = &*blocks_ptr.add(n_blk);
                         let n_active = (n_block.occupied & (1 << n_bit)) != 0;
                         
                         if my_active || n_active {
                             let n_node = (n_tile_idx * 1024) + n_local;
                             
                             if self.union(my_node as u32, n_node as u32) {
                                 if !my_active {
                                     (*blocks_ptr.add(my_blk)).occupied |= 1 << my_bit;
                                     self.mark_global_dirty(my_blk);
                                 }
                                 if !n_active {
                                     (*blocks_ptr.add(n_blk)).occupied |= 1 << n_bit;
                                     self.mark_global_dirty(n_blk);
                                 }
                             }
                         }
                    });
                };

                // Top & Bottom Rows
                for x in 0..32 {
                    process_pixel(x, 0);
                    process_pixel(x, 31);
                }
                // Left & Right Cols (excluding corners processed above? No, process all)
                for y in 1..31 {
                    process_pixel(0, y);
                    process_pixel(31, y);
                }
            }
        }
    }

    #[inline(always)]
    fn mark_global_dirty(&mut self, blk_idx: usize) {
        let mask_idx = blk_idx >> 6;
        let mask_bit = blk_idx & 63;
        self.queued_mask[mask_idx] |= 1 << mask_bit;
        self.block_dirty_mask[mask_idx] |= 1 << mask_bit;
    }

    // Union Find Helpers (Copied from state.rs/union_find.rs or forwarded)

    /// Finds the cluster root for node `i` with path compression.
    ///
    /// Uses path halving compression for O(α(n)) amortized complexity.
    pub fn find(&mut self, mut i: u32) -> u32 {
        unsafe {
            let mut p = *self.parents.get_unchecked(i as usize);
            if p != i {
                let mut gp = *self.parents.get_unchecked(p as usize);
                if gp != p {
                    loop {
                        *self.parents.get_unchecked_mut(i as usize) = gp;
                        i = gp;
                        p = *self.parents.get_unchecked(gp as usize);
                        if p == gp {
                            break;
                        }
                        gp = *self.parents.get_unchecked(p as usize);
                        if gp == p {
                            *self.parents.get_unchecked_mut(i as usize) = p;
                            break;
                        }
                    }
                }
                return gp;
            }
            i
        }
    }

    /// Merges clusters containing nodes `u` and `v`.
    ///
    /// Returns `true` if clusters were different and got merged.
    pub fn union(&mut self, u: u32, v: u32) -> bool {
        let root_u = self.find(u);
        let root_v = self.find(v);
        if root_u != root_v {
            let (small, big) = if root_u < root_v {
                (root_u, root_v)
            } else {
                (root_v, root_u)
            };
            unsafe {
                *self.parents.get_unchecked_mut(small as usize) = big;
                let tile = small as usize / 1024;
                if tile < self.tiles_x * self.tiles_y {
                    let local = small as usize % 1024;
                    let blk = tile * 16 + (local / 64);
                    self.mark_global_dirty(blk);
                }
            }
            return true;
        }
        false
    }

    /// Extracts edge corrections from grown clusters.
    ///
    /// Traces paths from syndrome nodes through the Union Find forest,
    /// converting logical edges to physical corrections.
    ///
    /// # Arguments
    ///
    /// * `corrections` - Output buffer for edge corrections.
    ///
    /// # Returns
    ///
    /// Number of corrections written to the buffer.
    pub fn peel_forest(&mut self, corrections: &mut [EdgeCorrection]) -> usize {
        // Clear auxiliary buffers
        self.path_mark.fill(0);

        // 1. Identify Syndromes and Trace Paths to Root/Boundary
        // Iterate over defect mask to find syndromes
        for blk_idx in 0..self.defect_mask.len() {
            let mut word = unsafe { *self.defect_mask.get_unchecked(blk_idx) };
            if word == 0 {
                continue;
            }

            let base_node = blk_idx * 64;
            while word != 0 {
                let bit = tzcnt(word) as usize;
                word &= word - 1;
                let u = (base_node + bit) as u32;

                // Trace path from syndrome u to root/boundary
                self.trace_path(u);
            }
        }

        // 2. Process Marked Paths (Peeling)
        // Iterate over path_mark. If a node u is marked, it means the edge (u, parent[u]) is part of an odd number of paths.
        // We need to 'realize' this logical edge as physical edges (corrections).
        for blk_idx in 0..self.path_mark.len() {
            let mut word = unsafe { *self.path_mark.get_unchecked(blk_idx) };
            if word == 0 {
                continue;
            }

            let base_node = blk_idx * 64;
            while word != 0 {
                let bit = tzcnt(word) as usize;
                word &= word - 1;
                let u = (base_node + bit) as u32;

                let v = unsafe { *self.parents.get_unchecked(u as usize) };

                if u != v {
                    self.trace_manhattan_tiled(u, v);
                }
            }
        }

        // 3. Collect Corrections from dirty edges
        self.reconstruct_tiled_corrections(corrections)
    }

    fn trace_path(&mut self, u: u32) {
        let mut curr = u;
        loop {
            let next = unsafe { *self.parents.get_unchecked(curr as usize) };
            if curr == next {
                break;
            }

            let blk = (curr as usize) / 64;
            let bit = (curr as usize) % 64;

            unsafe {
                *self.path_mark.get_unchecked_mut(blk) ^= 1 << bit;
            }
            curr = next;
        }
    }

    fn trace_manhattan_tiled(&mut self, u: u32, v: u32) {
        let boundary_node = (self.parents.len() - 1) as u32;

        if u == boundary_node {
            self.emit_tiled_edge(v, u32::MAX);
            return;
        }
        if v == boundary_node {
            self.emit_tiled_edge(u, u32::MAX);
            return;
        }

        let (ux, uy) = self.get_global_coord(u);
        let (vx, vy) = self.get_global_coord(v);

        // Simple Manhattan routing: Move X, then Move Y.
        // We assume the path is valid in the topology or at least sufficient for correction.
        // For Square/Rect: Always valid.
        // For Triangular/Honeycomb: Manhattan path exists (using only cardinal moves).
        
        let dx = (vx as isize) - (ux as isize);
        let dy = (vy as isize) - (uy as isize);

        let mut curr = u;
        let mut curr_x = ux as isize;
        let mut curr_y = uy as isize;

        // Move in X
        if dx != 0 {
            let step = if dx > 0 { 1 } else { -1 };
            let steps = dx.abs();
            for _ in 0..steps {
                let next_x = curr_x + step;
                if let Some(next) = self.get_node_idx(next_x as usize, curr_y as usize) {
                    self.emit_tiled_edge(curr, next);
                    curr = next;
                    curr_x = next_x;
                } else {
                    break; // Should not happen if bounds correct
                }
            }
        }

        // Move in Y
        if dy != 0 {
            let step = if dy > 0 { 1 } else { -1 };
            let steps = dy.abs();
            for _ in 0..steps {
                let next_y = curr_y + step;
                if let Some(next) = self.get_node_idx(curr_x as usize, next_y as usize) {
                    self.emit_tiled_edge(curr, next);
                    curr = next;
                    curr_y = next_y;
                } else {
                    break;
                }
            }
        }
    }

    fn get_node_idx(&self, x: usize, y: usize) -> Option<u32> {
        if x < self.width && y < self.height {
            let tx = x / 32;
            let ty = y / 32;
            let lx = x % 32;
            let ly = y % 32;
            let node = (ty * self.tiles_x + tx) * 1024 + (ly * 32 + lx);
            Some(node as u32)
        } else {
            None
        }
    }

    fn get_global_coord(&self, u: u32) -> (usize, usize) {
        let tile_idx = (u as usize) / 1024;
        let local_idx = (u as usize) % 1024;

        let tx = tile_idx % self.tiles_x;
        let ty = tile_idx / self.tiles_x;

        let lx = local_idx % 32;
        let ly = local_idx / 32;

        (tx * 32 + lx, ty * 32 + ly)
    }

    fn emit_tiled_edge(&mut self, u: u32, v: u32) {
        if v == u32::MAX {
            // Boundary edge
            let blk_idx = (u as usize) / 64;
            let bit_idx = (u as usize) % 64;

            let mask_idx = blk_idx >> 6;
            let mask_bit = blk_idx & 63;

            unsafe {
                let m_ptr = self.boundary_dirty_mask.get_unchecked_mut(mask_idx);
                if (*m_ptr & (1 << mask_bit)) == 0 {
                    *m_ptr |= 1 << mask_bit;
                    *self
                        .boundary_dirty_list
                        .get_unchecked_mut(self.boundary_dirty_count) = blk_idx as u32;
                    self.boundary_dirty_count += 1;
                }
                *self.boundary_bitmap.get_unchecked_mut(blk_idx) ^= 1 << bit_idx;
            }
            return;
        }

        // Regular edge
        // We need to determine an edge index for (u, v).
        // Canonical order: u < v.
        let (src, dst) = if u < v { (u, v) } else { (v, u) };

        // Determine 'dir' (0..2) relative to src.
        // We need to find dst in src's neighbors and get its index?
        // Or calculate relative position.
        let (ux, uy) = self.get_global_coord(src);
        let (vx, vy) = self.get_global_coord(dst);
        
        // Differences (dst - src)
        // Since src < dst (mostly), and layout is Row-Major...
        // vy >= uy.
        // If vy == uy, vx > ux (Right).
        // If vy > uy, vx could be anything.
        
        let dx = (vx as isize) - (ux as isize);
        let dy = (vy as isize) - (uy as isize);

        // TriangularGrid only has ONE diagonal type per node, so both diagonal
        // directions (Down-Right and Down-Left) map to slot 2.
        let dir = if dy == 0 && dx == 1 {
            0 // Right
        } else if dy == 1 && dx == 0 {
            1 // Down
        } else if dy == 1 && (dx == 1 || dx == -1) {
            2 // Diagonal (slot 2 for either direction)
        } else {
            // Should not happen for handled topologies
            0
        };

        // Edge Index
        let edge_idx = (src as usize) * 3 + dir;
        let word_idx = edge_idx / 64;
        let bit_idx = edge_idx % 64;

        let mask_idx = word_idx >> 6;
        let mask_bit = word_idx & 63;

        unsafe {
            let m_ptr = self.edge_dirty_mask.get_unchecked_mut(mask_idx);
            if (*m_ptr & (1 << mask_bit)) == 0 {
                *m_ptr |= 1 << mask_bit;
                *self
                    .edge_dirty_list
                    .get_unchecked_mut(self.edge_dirty_count) = word_idx as u32;
                self.edge_dirty_count += 1;
            }
            *self.edge_bitmap.get_unchecked_mut(word_idx) ^= 1 << bit_idx;
        }
    }

    fn reconstruct_tiled_corrections(&mut self, corrections: &mut [EdgeCorrection]) -> usize {
        let mut count = 0;

        // Process Dirty Edges
        for i in 0..self.edge_dirty_count {
            let word_idx = unsafe { *self.edge_dirty_list.get_unchecked(i) } as usize;

            // Clear mask
            let mask_idx = word_idx >> 6;
            let mask_bit = word_idx & 63;
            unsafe {
                *self.edge_dirty_mask.get_unchecked_mut(mask_idx) &= !(1 << mask_bit);
            }

            let word_ptr = unsafe { self.edge_bitmap.get_unchecked_mut(word_idx) };
            let mut word = *word_ptr;
            *word_ptr = 0;

            let base_idx = word_idx * 64;
            while word != 0 {
                let bit = tzcnt(word) as usize;
                word &= word - 1;

                let edge_idx = base_idx + bit;
                let u = (edge_idx / 3) as u32;
                let dir = edge_idx % 3;
                
                let (ux, uy) = self.get_global_coord(u);
                
                // Recover v from u and dir
                let (vx, vy) = match dir {
                    0 => (ux + 1, uy),
                    1 => (ux, uy + 1),
                    2 => {
                         // Determine diagonal direction based on Topology?
                         // Or assume fixed diagonal for now?
                         // TriangularGrid logic:
                         // if (idx.count_ones() & 1) == 0 -> Up-Right.
                         // But we are at 'u' looking for neighbor 'v' such that u < v.
                         // Neighbors of u: Right, Down, maybe Diag.
                         // If u has Down-Right diagonal: (ux+1, uy+1).
                         // If u has Down-Left diagonal: (ux-1, uy+1).
                         // We need T to know which one.
                         // BUT we are generic.
                         // We can assume for TriangularGrid, Diag is always "the diagonal".
                         // However, T::for_each_neighbor is the truth.
                         // If we assume TriangularGrid matches `prav-core/src/topology.rs`:
                         // Parity check uses Morton index.
                         let m_idx = morton_encode_2d(ux as u32, uy as u32);
                         if (m_idx.count_ones() & 1) == 0 {
                             // "Has Right and Up". Diag is Up-Right. (x+1, y-1).
                             // But v must be > u. (y-1) < y. So u > v.
                             // This edge would be stored at v (the smaller node).
                             // So if we are at u, we store edges to v > u.
                             // For this node u, valid v > u neighbors are:
                             // Right (x+1, y). Down (x, y+1).
                             // Does it have Down-Right or Down-Left?
                             // If neighbor w (Down-Left) exists, w < u? No, w.y > u.y. So w > u.
                             // So Down-Left is a valid forward edge.
                             // If neighbor z (Down-Right) exists, z > u.
                             
                             // We need to know which one 'dir=2' represents.
                             // In emit_tiled_edge:
                             // dy=1, dx=1 -> Down-Right.
                             // dy=1, dx=-1 -> Down-Left.
                             // Both mapped to dir=2.
                             // This implies a node has AT MOST one "forward diagonal" neighbor?
                             // TriangularGrid: 6 neighbors.
                             // Left, Right, Up, Down.
                             // Plus ONE diagonal pair.
                             // If Parity 0: Diag is / (Up-Right, Down-Left).
                             // Forward neighbors (>u): Right, Down, Down-Left.
                             // If Parity 1: Diag is \ (Up-Left, Down-Right).
                             // Forward neighbors (>u): Right, Down, Down-Right.
                             
                             // So yes! Only one "Down" diagonal per node.
                             // Parity 0 -> Down-Left.
                             // Parity 1 -> Down-Right.
                             if (m_idx.count_ones() & 1) == 0 {
                                 (ux.wrapping_sub(1), uy + 1)
                             } else {
                                 (ux + 1, uy + 1)
                             }
                         } else {
                             // Copy-paste error in reasoning above?
                             // Re-read Topology.rs:
                             // if parity 0: if has_right && has_up { dec(right, Y) -> (x+1, y-1) }
                             //    This is Up-Right.
                             //    Does it have Down-Left?
                             //    No, "else if parity 1: if has_left && has_down { inc(left, Y) -> (x-1, y+1) }"
                             //    This is Down-Left.
                             
                             // So Node Parity 0 has Up-Right.
                             // Node Parity 1 has Down-Left.
                             
                             // Let's verify reciprocal.
                             // Edge (u, v). u < v.
                             // v is "Down" relative to u (mostly).
                             // If Edge is Up-Right from u? v = (x+1, y-1).
                             // v.y < u.y. So v < u (mostly).
                             // So Up-Right is a BACKWARD edge.
                             // We don't store it at u. We store it at v.
                             
                             // If Edge is Down-Left from u? v = (x-1, y+1).
                             // v.y > u.y. So v > u.
                             // So Down-Left is a FORWARD edge.
                             // Does Parity 0 have Down-Left? No.
                             // Does Parity 1 have Down-Left? Yes.
                             
                             // So Parity 1 nodes have a generic "Diagonal Forward" (Down-Left).
                             // What about Parity 0 nodes?
                             // They have Up-Right (Backward).
                             // Do they have Down-Right? No.
                             // So Parity 0 nodes have NO Forward Diagonal?
                             // Wait.
                             // Triangular grid is connected.
                             // Edges are undirected.
                             // Edge between A(0,0) and B(1,1)?
                             // A=0 (Parity 0). B=3 (Parity 0).
                             // A has Up-Right? No (y=0).
                             // B has Up-Right? (2, 0).
                             // This assumes specific layout.
                             
                             // Let's rely on coordinates from emit_tiled_edge logic.
                             // We map dir=2 to "The Diagonal".
                             // But reconstructing requires knowing WHICH diagonal.
                             // We can use the Parity logic again.
                             
                             if (m_idx.count_ones() & 1) != 0 {
                                 // Parity 1 has Down-Left (Forward).
                                 (ux.wrapping_sub(1), uy + 1)
                             } else {
                                 // Parity 0 has... NO forward diagonal?
                                 // Check neighbors of Parity 0.
                                 // Right (x+1, y) -> >u.
                                 // Down (x, y+1) -> >u.
                                 // Up-Right (x+1, y-1) -> <u.
                                 // Left, Up -> <u.
                                 // So Parity 0 only has Right and Down as forward edges?
                                 // If so, dir=2 should never happen for Parity 0!
                                 // UNLESS I messed up u < v logic.
                                 // If u < v, and v is Up-Right of u?
                                 // v.y < u.y. v < u. Contradiction.
                                 
                                 // So, Parity 0 nodes ONLY have dir=0 and dir=1.
                                 // Parity 1 nodes have dir=0, dir=1, dir=2 (Down-Left).
                                 
                                 // Wait, what about Down-Right?
                                 // Topology.rs doesn't seem to implement Down-Right for anyone?
                                 // TriangularGrid:
                                 // Parity 0: Up-Right.
                                 // Parity 1: Down-Left.
                                 // This forms diagonals like ///.
                                 // So only one type of diagonal exists in the whole grid ( ///// ).
                                 // (x, y) connected to (x+1, y-1).
                                 // Equivalent to (x, y) connected to (x-1, y+1).
                                 // So yes, all diagonals are "Up-Right / Down-Left" type.
                                 // "Down-Right" (\) does not exist.
                                 
                                 // So dir=2 ALWAYS means Down-Left (x-1, y+1).
                                 (ux.wrapping_sub(1), uy + 1)
                             }
                         }
                    }
                    _ => (ux, uy),
                };
                
                // Map global v back to tiled u32
                if vx < self.width && vy < self.height {
                    let v_tx = vx / 32;
                    let v_ty = vy / 32;
                    let v_lx = vx % 32;
                    let v_ly = vy % 32;
                    let v_node = (v_ty * self.tiles_x + v_tx) * 1024 + (v_ly * 32 + v_lx);
                    
                    if count < corrections.len() {
                        unsafe {
                            *corrections.get_unchecked_mut(count) = EdgeCorrection { u, v: v_node as u32 };
                        }
                        count += 1;
                    }
                }
            }
        }
        self.edge_dirty_count = 0;

        // Process Dirty Boundaries
        for i in 0..self.boundary_dirty_count {
            let blk_idx = unsafe { *self.boundary_dirty_list.get_unchecked(i) } as usize;

            // Clear mask
            let mask_idx = blk_idx >> 6;
            let mask_bit = blk_idx & 63;
            unsafe {
                *self.boundary_dirty_mask.get_unchecked_mut(mask_idx) &= !(1 << mask_bit);
            }

            let word_ptr = unsafe { self.boundary_bitmap.get_unchecked_mut(blk_idx) };
            let mut word = *word_ptr;
            *word_ptr = 0;

            let base_u = blk_idx * 64;
            while word != 0 {
                let bit = tzcnt(word) as usize;
                word &= word - 1;
                let u = (base_u + bit) as u32;

                if count < corrections.len() {
                    unsafe {
                        *corrections.get_unchecked_mut(count) = EdgeCorrection { u, v: u32::MAX };
                    }
                    count += 1;
                }
            }
        }
        self.boundary_dirty_count = 0;

        count
    }
}

#[cfg(test)]

mod tests {

    use super::*;

    use crate::arena::Arena;
    use crate::topology::SquareGrid;

    extern crate std;

    #[test]

    fn test_tiled_horizontal_stitching() {
        let mut memory = std::vec![0u8; 1024 * 1024 * 16];

        let mut arena = Arena::new(&mut memory);

        // 64x32 grid (2 tiles wide, 1 tile high)

        let width = 64;

        let height = 32;

        let mut decoder = TiledDecodingState::<SquareGrid>::new(&mut arena, width, height);

        // Node A: (31, 0) -> Right edge of Tile 0. Global index 31.

        // Node B: (32, 0) -> Left edge of Tile 1. Global index 1024 (Tile 1 start).

        let node_a = 31;

        let node_b = 1024;

        // Manually inject defects/active state to simulate growth

        // We need to find the blocks corresponding to these nodes.

        // Node A: Tile 0. Local 31. Block 0. Bit 31.

        // Node B: Tile 1. Local 0. Block 16. Bit 0.

        let blk_a = 0;

        let bit_a = 31;

        let blk_b = 16;

        let bit_b = 0;

        decoder.blocks_state[blk_a].occupied |= 1 << bit_a;

        decoder.blocks_state[blk_a].boundary |= 1 << bit_a; // Ensure it spreads

        decoder.active_mask[blk_a >> 6] |= 1 << (blk_a & 63);

        decoder.blocks_state[blk_b].occupied |= 1 << bit_b;

        decoder.blocks_state[blk_b].boundary |= 1 << bit_b;

        decoder.active_mask[blk_b >> 6] |= 1 << (blk_b & 63);

        // Run growth

        decoder.grow_clusters();

        let root_a = decoder.find(node_a);

        let root_b = decoder.find(node_b);

        assert_eq!(
            root_a, root_b,
            "Horizontal stitching failed between Tile 0 and Tile 1"
        );
    }

    #[test]

    fn test_tiled_vertical_stitching() {
        let mut memory = std::vec![0u8; 1024 * 1024 * 16];

        let mut arena = Arena::new(&mut memory);

        // 32x64 grid (1 tile wide, 2 tiles high)

        let width = 32;

        let height = 64;

        let mut decoder = TiledDecodingState::<SquareGrid>::new(&mut arena, width, height);

        // Node A: (0, 31) -> Bottom edge of Tile 0. Global index 31*32 + 0 = 992.

        // Node B: (0, 32) -> Top edge of Tile 1. Global index 1024 + 0 = 1024.

        let node_a = 992;

        let node_b = 1024;

        // Node A: Tile 0. Local 992. Block 992/64 = 15. Bit 992%64 = 32.

        // Node B: Tile 1. Local 0. Block 16. Bit 0.

        let blk_a = 15;

        let bit_a = 32;

        let blk_b = 16;

        let bit_b = 0;

        decoder.blocks_state[blk_a].occupied |= 1 << bit_a;

        decoder.blocks_state[blk_a].boundary |= 1 << bit_a;

        decoder.active_mask[blk_a >> 6] |= 1 << (blk_a & 63);

        decoder.blocks_state[blk_b].occupied |= 1 << bit_b;

        decoder.blocks_state[blk_b].boundary |= 1 << bit_b;

        decoder.active_mask[blk_b >> 6] |= 1 << (blk_b & 63);

        decoder.grow_clusters();

        let root_a = decoder.find(node_a);

        let root_b = decoder.find(node_b);

        assert_eq!(
            root_a, root_b,
            "Vertical stitching failed between Tile 0 and Tile 1"
        );
    }
}
