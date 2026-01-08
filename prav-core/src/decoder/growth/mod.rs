//! Cluster growth algorithm for Union Find QEC decoding.
//!
//! This module implements the iterative boundary expansion that groups syndrome
//! nodes into connected clusters. The algorithm:
//!
//! 1. Loads syndrome measurements as seed points
//! 2. Iteratively expands cluster boundaries using SWAR bit operations
//! 3. Merges clusters when they meet via Union Find
//! 4. Terminates when no more expansion is possible
//!
//! # Algorithm Overview
//!
//! ```text
//! Initial:     After growth:
//!   .   .        . . .
//!   . X .   =>   . X .   (X = defect, . = occupied by cluster)
//!   .   .        . . .
//! ```
//!
//! # Performance Optimizations
//!
//! - **SWAR spreading**: Syndrome spreading uses SIMD-Within-A-Register operations,
//!   achieving 19-427x speedup over lookup tables.
//! - **Monochromatic fast-path**: When all 64 nodes in a block share the same root,
//!   skip Union Find operations entirely (covers ~95% of blocks).
//! - **Block-level parallelism**: Active blocks tracked via bitmasks for efficient
//!   iteration.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::decoder::state::DecodingState;
use crate::topology::Topology;

// =============================================================================
// Submodules
// =============================================================================

/// Inter-block neighbor processing and merging utilities.
pub mod inter_block;

/// Small grid fast-path (single u64 active mask, <=64 blocks).
pub mod small_grid;

/// Stride-32 specific implementation.
pub mod stride32;

/// Unrolled optimizations for stride-32/64.
pub mod unrolled;

/// Kani formal verification proofs.
#[cfg(kani)]
mod kani_proofs;

/// Cluster boundary expansion operations for QEC decoding.
///
/// This trait defines the interface for the cluster growth phase of Union Find
/// decoding. Starting from syndrome measurements (defects), clusters expand
/// outward until they either meet other clusters or reach the boundary.
///
/// # Decoding Flow
///
/// ```text
/// load_dense_syndromes() -> grow_clusters() -> peel_forest()
///        |                       |                 |
///        v                       v                 v
///   Initialize seeds      Expand boundaries   Extract corrections
/// ```
///
/// # Implementors
///
/// This trait is implemented by [`DecodingState`] for all topologies.
pub trait ClusterGrowth {
    /// Loads syndrome measurements from a dense bitarray.
    ///
    /// Syndromes indicate which stabilizer measurements detected errors.
    /// Each u64 represents 64 consecutive nodes, where bit `i` being set
    /// means node `(blk_idx * 64 + i)` has a syndrome (defect).
    ///
    /// # Arguments
    ///
    /// * `syndromes` - Dense bitarray where `syndromes[blk_idx]` contains
    ///   syndrome bits for block `blk_idx`. Length should match the number
    ///   of blocks in the decoder.
    ///
    /// # Implementation Details
    ///
    /// Uses a two-stage approach for large grids:
    /// 1. **Scanner stage**: Burst-mode collection of non-zero blocks
    /// 2. **Processor stage**: Initialize block state for active blocks
    ///
    /// For small grids (<=64 blocks), uses direct bitmask manipulation.
    fn load_dense_syndromes(&mut self, syndromes: &[u64]);

    /// Expands cluster boundaries until convergence.
    ///
    /// Iteratively calls [`grow_iteration`](Self::grow_iteration) until no
    /// more expansion is possible. The algorithm terminates when:
    ///
    /// - All defects have been connected to the boundary, OR
    /// - All defects have been paired with other defects in the same cluster
    ///
    /// # Termination Guarantee
    ///
    /// The algorithm is guaranteed to terminate within O(max_dimension) iterations,
    /// where `max_dimension` is the largest grid dimension. A safety limit of
    /// `max_dim * 16 + 128` iterations is enforced.
    fn grow_clusters(&mut self);

    /// Performs a single iteration of cluster growth.
    ///
    /// Processes all currently active blocks, expanding their boundaries
    /// and merging clusters as needed.
    ///
    /// # Returns
    ///
    /// * `true` if any expansion occurred (more iterations may be needed).
    /// * `false` if no expansion occurred (algorithm has converged).
    ///
    /// # Active Block Tracking
    ///
    /// Blocks are tracked in an active set. After processing:
    /// - Blocks that expanded are added to the next iteration's set
    /// - Blocks that can't expand further are removed
    fn grow_iteration(&mut self) -> bool;

    /// Processes a single block during cluster growth.
    ///
    /// Expands the boundary within the block and handles connections to
    /// neighboring blocks. This is the core operation called for each
    /// active block during growth.
    ///
    /// # Arguments
    ///
    /// * `blk_idx` - Index of the block to process.
    ///
    /// # Returns
    ///
    /// * `true` if the block's boundary expanded (neighbor blocks may activate).
    /// * `false` if no expansion occurred.
    ///
    /// # Safety
    ///
    /// Caller must ensure `blk_idx` is within bounds of the internal blocks state.
    unsafe fn process_block(&mut self, blk_idx: usize) -> bool;
}

impl<'a, T: Topology, const STRIDE_Y: usize> ClusterGrowth for DecodingState<'a, T, STRIDE_Y> {
    fn load_dense_syndromes(&mut self, syndromes: &[u64]) {
        self.ingestion_count = 0;
        self.active_block_mask = 0; // Reset for small grids
        let limit = syndromes.len().min(self.blocks_state.len());

        if self.is_small_grid() {
            for blk_idx in 0..limit {
                // Eagerly sync all blocks
                let word = unsafe { *syndromes.get_unchecked(blk_idx) };
                if word != 0 {
                    unsafe {
                        self.mark_block_dirty(blk_idx);
                        let block = self.blocks_state.get_unchecked_mut(blk_idx);
                        block.boundary |= word;
                        block.occupied |= word;
                        *self.defect_mask.get_unchecked_mut(blk_idx) |= word;

                        // Directly set bit in active_block_mask
                        self.active_block_mask |= 1 << blk_idx;
                    }
                }
            }
            // Ensure queued_mask is clear before growth starts
            if !self.queued_mask.is_empty() {
                self.queued_mask[0] = 0;
            }
            return;
        }

        let mut blk_idx = 0;

        // Stage 1: Scanner (Burst-Mode Ingestion)



        while blk_idx < limit {
            let word = unsafe { *syndromes.get_unchecked(blk_idx) };
            if word != 0 {
                unsafe {
                    *self.ingestion_list.get_unchecked_mut(self.ingestion_count) = blk_idx as u32;
                    self.ingestion_count += 1;
                }
            }
            blk_idx += 1;
        }

        // Stage 2: Processor
        for i in 0..self.ingestion_count {
            let blk_idx = unsafe { *self.ingestion_list.get_unchecked(i) } as usize;
            let word = unsafe { *syndromes.get_unchecked(blk_idx) };

            // Lazy Reset: Ensure block is ready for this epoch - REMOVED, assumed clean via sparse_reset
            unsafe {
                self.mark_block_dirty(blk_idx);
                let block = self.blocks_state.get_unchecked_mut(blk_idx);
                block.boundary |= word;
                block.occupied |= word;
                *self.defect_mask.get_unchecked_mut(blk_idx) |= word;

                self.push_next(blk_idx);
            }
        }

        // Prepare for first growth iteration
        if !self.is_small_grid() {
            // Swap active and queued
            core::mem::swap(&mut self.active_mask, &mut self.queued_mask);
            self.queued_mask.fill(0);
        }
    }

    #[inline(never)]
    fn grow_clusters(&mut self) {
        if self.is_small_grid() {
            let max_dim = self.width.max(self.height).max(self.graph.depth);
            let limit = max_dim * 16 + 128;

            for _ in 0..limit {
                self.grow_bitmask_iteration();

                if self.active_block_mask == 0 {
                    break;
                }
            }
            return;
        }

        let max_dim = self.width.max(self.height).max(self.graph.depth);
        let limit = max_dim * 16 + 128;

        for _ in 0..limit {
            // Check if active_mask is empty
            // Use iterator to check all words (SIMD optimized typically)
            if self.active_mask.iter().all(|&w| w == 0) {
                break;
            }

            self.grow_iteration();
        }
    }

    #[inline(always)]
    fn grow_iteration(&mut self) -> bool {
        if self.is_small_grid() {
            let mut any_expansion = false;
            let num_blocks = self.blocks_state.len();

            // Clear the mask as we are switching to flat scan
            if num_blocks > 0 {
                unsafe { *self.queued_mask.get_unchecked_mut(0) = 0 };
            }

            for blk_idx in 0..num_blocks {
                unsafe {
                    if self.process_block_silent(blk_idx) {
                        any_expansion = true;
                    }
                }
            }
            return any_expansion;
        }

        let mut any_expansion = false;

        // Queued mask is already cleared at the end of previous iteration (or init)

        let active_mask_ptr = self.active_mask.as_ptr();
        let active_mask_len = self.active_mask.len();

        for chunk_idx in 0..active_mask_len {
            let mut w = unsafe { *active_mask_ptr.add(chunk_idx) };
            if w == 0 {
                continue;
            }

            let base_idx = chunk_idx * 64;
            while w != 0 {
                let bit = w.trailing_zeros();
                w &= w - 1;
                let blk_idx = base_idx + bit as usize;

                unsafe {
                    if self.process_block(blk_idx) {
                        any_expansion = true;
                    }
                }
            }
        }

        core::mem::swap(&mut self.active_mask, &mut self.queued_mask);
        self.queued_mask.fill(0);

        any_expansion
    }

    #[inline(always)]
    unsafe fn process_block(&mut self, blk_idx: usize) -> bool {
        self.process_block_small_stride::<false>(blk_idx)
    }
}

impl<'a, T: Topology, const STRIDE_Y: usize> DecodingState<'a, T, STRIDE_Y> {
    fn grow_bitmask_iteration(&mut self) {
        // optimized_32 unrolled path for 16 blocks (1024 nodes)
        if STRIDE_Y == 32 && self.blocks_state.len() == 16 {
            unsafe {
                let (_expanded, next_mask) = self.process_all_blocks_stride_32_unrolled_16();
                self.active_block_mask = next_mask;
            }
            return;
        }

        // Reset queued mask for next iteration
        self.queued_mask[0] = 0;

        let mut current_mask = self.active_block_mask;
        while current_mask != 0 {
            let blk_idx = crate::intrinsics::tzcnt(current_mask) as usize;
            current_mask &= current_mask - 1;

            unsafe {
                self.process_block(blk_idx);
            }
        }

        // Update active mask from the queued mask populated by process_block
        self.active_block_mask = self.queued_mask[0];
    }

    #[inline(always)]
    unsafe fn process_block_silent(&mut self, blk_idx: usize) -> bool {
        self.process_block_small_stride::<true>(blk_idx)
    }
}
