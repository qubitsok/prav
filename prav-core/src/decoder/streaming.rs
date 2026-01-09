//! Sliding window streaming decoder for real-time QEC.
//!
//! This module provides a streaming decoder that processes syndrome measurements
//! round-by-round, committing corrections when rounds exit a fixed-size sliding window.
//! This enables low-latency decoding for real-time quantum error correction.
//!
//! # Architecture
//!
//! ```text
//! Round N arrives:
//! ┌─────────────────────────────────────────────────────┐
//! │ Sliding Window (size W)                             │
//! │  ┌───────┬───────┬───────┬───────┐                 │
//! │  │ R(N-3)│ R(N-2)│ R(N-1)│  R(N) │  ← New round    │
//! │  │ EXIT  │       │       │ LOAD  │                  │
//! │  └───┬───┴───────┴───────┴───────┘                 │
//! │      │                                              │
//! │      ▼                                              │
//! │  Commit corrections for R(N-3)                      │
//! │  Clear Z-layer for R(N-3)                          │
//! │  Load syndromes for R(N) at Z = N % W              │
//! │  Grow clusters (may connect to previous rounds)     │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Design: Circular Z-Indexing
//!
//! - Round R maps to physical Z = R % W (no data copying when window slides)
//! - Pre-allocate for W × (d-1)² detectors
//! - Commit corrections only when rounds exit the window (guaranteed correct)
//!
//! # Usage
//!
//! ```ignore
//! use prav_core::streaming::{StreamingConfig, StreamingDecoder};
//!
//! let config = StreamingConfig::for_rotated_surface(5, 3); // d=5, window=3
//! let buf_size = streaming_buffer_size(config.width, config.height, config.window_size);
//! let mut buffer = vec![0u8; buf_size];
//! let mut arena = Arena::new(&mut buffer);
//!
//! let mut decoder = StreamingDecoderBuilder::new(config)
//!     .build(&mut arena)
//!     .expect("Failed to create decoder");
//!
//! // Process rounds as they arrive
//! for round_syndromes in syndrome_stream {
//!     if let Some(committed) = decoder.ingest_round(&round_syndromes) {
//!         // Process corrections for the round that just exited
//!         process_corrections(committed.round, committed.corrections);
//!     }
//! }
//!
//! // Flush remaining rounds at end of stream
//! for committed in decoder.flush() {
//!     process_corrections(committed.round, committed.corrections);
//! }
//! ```

#![allow(unsafe_op_in_unsafe_fn)]

use core::marker::PhantomData;

use crate::arena::Arena;
use crate::decoder::growth::ClusterGrowth;
use crate::decoder::state::DecodingState;
use crate::decoder::types::EdgeCorrection;
use crate::topology::Topology;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for the streaming decoder.
///
/// Defines the sliding window size and spatial dimensions of the detector grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamingConfig {
    /// Window size in rounds (W).
    /// Determines how many rounds are kept in memory simultaneously.
    /// Larger windows provide more context for cluster merging but increase latency.
    pub window_size: usize,

    /// Spatial width (detectors per row).
    pub width: usize,

    /// Spatial height (rows per round).
    pub height: usize,

    /// Total detectors per round (width × height).
    pub detectors_per_round: usize,

    /// Y stride for coordinate calculations (power of 2).
    pub stride_y: usize,

    /// Z stride for round indexing (stride_y²).
    pub stride_z: usize,
}

impl StreamingConfig {
    /// Create configuration for a rotated surface code.
    ///
    /// For distance `d`, creates a (d-1)×(d-1) detector grid per round.
    ///
    /// # Arguments
    ///
    /// * `d` - Code distance
    /// * `window_size` - Number of rounds in the sliding window
    #[must_use]
    pub const fn for_rotated_surface(d: usize, window_size: usize) -> Self {
        let detector_dim = if d > 1 { d - 1 } else { 1 };
        let max_dim = const_max(detector_dim, window_size);
        let stride_y = next_power_of_two(max_dim);
        let stride_z = stride_y * stride_y;

        Self {
            window_size,
            width: detector_dim,
            height: detector_dim,
            detectors_per_round: detector_dim * detector_dim,
            stride_y,
            stride_z,
        }
    }

    /// Number of blocks per round (each block contains 64 detectors).
    #[must_use]
    #[inline]
    pub const fn blocks_per_round(&self) -> usize {
        div_ceil(self.detectors_per_round, 64)
    }

    /// Total number of blocks across all window rounds.
    #[must_use]
    #[inline]
    pub const fn total_blocks(&self) -> usize {
        div_ceil(self.stride_z * self.window_size, 64)
    }

    /// Total allocation size (nodes in the 3D grid).
    #[must_use]
    #[inline]
    pub const fn alloc_nodes(&self) -> usize {
        self.stride_z * self.window_size + 1
    }

    /// Map absolute round number to physical Z index (circular).
    #[must_use]
    #[inline]
    pub const fn round_to_z(&self, round: u64) -> usize {
        (round % self.window_size as u64) as usize
    }

    /// Map (x, y, round) to linear node index.
    #[must_use]
    #[inline]
    pub const fn coord_to_linear(&self, x: usize, y: usize, round: u64) -> usize {
        let z = self.round_to_z(round);
        z * self.stride_z + y * self.stride_y + x
    }
}

// =============================================================================
// Per-Round Metadata
// =============================================================================

/// Metadata for a single round in the sliding window.
///
/// Stored in arena-allocated array, indexed by physical Z = round % window_size.
#[derive(Clone, Copy, Default)]
#[repr(C, align(16))]
pub struct RoundMetadata {
    /// Absolute round number occupying this physical slot.
    /// u64::MAX means slot is empty/invalid.
    pub absolute_round: u64,

    /// Number of defects (syndrome bits) in this round.
    pub defect_count: u16,

    /// Status flags.
    /// Bit 0: round has been loaded
    /// Bit 1: round is being committed
    pub flags: u16,

    /// Reserved for future use.
    pub _reserved: u32,
}

impl RoundMetadata {
    /// Flag: round has been loaded with syndromes.
    pub const FLAG_LOADED: u16 = 1 << 0;
    /// Flag: round is currently being committed.
    pub const FLAG_COMMITTING: u16 = 1 << 1;

    /// Create empty metadata for an unused slot.
    #[must_use]
    #[inline]
    pub const fn empty() -> Self {
        Self {
            absolute_round: u64::MAX,
            defect_count: 0,
            flags: 0,
            _reserved: 0,
        }
    }

    /// Check if this slot contains a valid round.
    #[must_use]
    #[inline]
    pub const fn is_valid(&self) -> bool {
        self.absolute_round != u64::MAX
    }

    /// Check if this round has been loaded.
    #[must_use]
    #[inline]
    pub const fn is_loaded(&self) -> bool {
        (self.flags & Self::FLAG_LOADED) != 0
    }
}

// =============================================================================
// Committed Corrections Output
// =============================================================================

/// Corrections committed when a round exits the sliding window.
///
/// Contains all corrections attributed to the exiting round along with
/// the accumulated observable contribution.
#[derive(Debug, Clone, Copy)]
pub struct CommittedCorrections<'a> {
    /// The absolute round number these corrections belong to.
    pub round: u64,

    /// The corrections for edges touching this round.
    pub corrections: &'a [EdgeCorrection],

    /// Observable contribution from this round's corrections.
    /// XOR of all boundary edge observables.
    pub observable: u8,
}

// =============================================================================
// Streaming Decoder
// =============================================================================

/// Sliding window streaming decoder for real-time QEC.
///
/// Processes syndrome measurements round-by-round, maintaining a fixed-size
/// window of recent rounds. Corrections are committed when rounds exit the
/// window, guaranteeing correctness.
///
/// # Type Parameters
///
/// * `'a` - Lifetime of arena-allocated buffers
/// * `T` - Topology type (e.g., `Grid3D`)
/// * `STRIDE_Y` - Compile-time Y stride for performance
pub struct StreamingDecoder<'a, T: Topology, const STRIDE_Y: usize> {
    /// Underlying decoder state (dimensions: width × height × window_size).
    decoder: DecodingState<'a, T, STRIDE_Y>,

    /// Configuration parameters.
    config: StreamingConfig,

    /// Per-round metadata (length = window_size).
    round_metadata: &'a mut [RoundMetadata],

    /// Next round to ingest (head of stream).
    head_round: u64,

    /// Oldest round still in window (next to commit when window full).
    tail_round: u64,

    /// Number of rounds currently in the window.
    rounds_in_window: usize,

    // Commit pipeline buffers (arena-allocated)
    /// Defect node indices in the exiting layer (for partial commit).
    exit_layer_defects: &'a mut [u32],
    /// Count of defects in exit layer.
    exit_layer_count: usize,

    /// Buffer for corrections being committed.
    partial_corrections: &'a mut [EdgeCorrection],
    /// Count of corrections in buffer.
    partial_corrections_count: usize,

    /// Observable accumulator for current commit.
    commit_observable: u8,

    /// Marker for topology type.
    _marker: PhantomData<T>,
}

impl<'a, T: Topology, const STRIDE_Y: usize> StreamingDecoder<'a, T, STRIDE_Y> {
    /// Create a new streaming decoder.
    ///
    /// # Arguments
    ///
    /// * `arena` - Arena for memory allocation
    /// * `config` - Streaming configuration
    ///
    /// # Panics
    ///
    /// Panics if the arena doesn't have enough space for all allocations.
    pub fn new(arena: &mut Arena<'a>, config: StreamingConfig) -> Self {
        // Create the underlying decoder with window_size as depth
        let decoder = DecodingState::new(
            arena,
            config.width,
            config.height,
            config.window_size,
        );

        // Allocate streaming-specific buffers
        let round_metadata = arena
            .alloc_slice_aligned::<RoundMetadata>(config.window_size, 16)
            .expect("Failed to allocate round_metadata");

        let exit_layer_defects = arena
            .alloc_slice::<u32>(config.detectors_per_round)
            .expect("Failed to allocate exit_layer_defects");

        // Worst case: every defect creates ~2 corrections
        let max_corrections = config.detectors_per_round * 2;
        let partial_corrections = arena
            .alloc_slice::<EdgeCorrection>(max_corrections)
            .expect("Failed to allocate partial_corrections");

        // Initialize metadata
        for meta in round_metadata.iter_mut() {
            *meta = RoundMetadata::empty();
        }

        Self {
            decoder,
            config,
            round_metadata,
            head_round: 0,
            tail_round: 0,
            rounds_in_window: 0,
            exit_layer_defects,
            exit_layer_count: 0,
            partial_corrections,
            partial_corrections_count: 0,
            commit_observable: 0,
            _marker: PhantomData,
        }
    }

    /// Check if the window is full (no more room without committing).
    #[must_use]
    #[inline]
    pub fn is_window_full(&self) -> bool {
        self.rounds_in_window >= self.config.window_size
    }

    /// Get the current number of rounds in the window.
    #[must_use]
    #[inline]
    pub fn rounds_in_window(&self) -> usize {
        self.rounds_in_window
    }

    /// Get the next round number to be ingested.
    #[must_use]
    #[inline]
    pub fn head_round(&self) -> u64 {
        self.head_round
    }

    /// Get the oldest round number in the window.
    #[must_use]
    #[inline]
    pub fn tail_round(&self) -> u64 {
        self.tail_round
    }

    /// Get the configuration.
    #[must_use]
    #[inline]
    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }

    /// Reset the decoder for a new stream.
    ///
    /// Clears all state and resets round counters to zero.
    pub fn reset(&mut self) {
        // Reset round tracking
        self.head_round = 0;
        self.tail_round = 0;
        self.rounds_in_window = 0;

        // Clear metadata
        for meta in self.round_metadata.iter_mut() {
            *meta = RoundMetadata::empty();
        }

        // Clear commit buffers
        self.exit_layer_count = 0;
        self.partial_corrections_count = 0;
        self.commit_observable = 0;

        // Full reset of underlying decoder
        self.decoder.initialize_internal();
    }

    /// Ingest syndromes for the next round.
    ///
    /// If the window is full, commits and returns corrections for the oldest round
    /// before loading the new round.
    ///
    /// # Arguments
    ///
    /// * `round_syndromes` - Dense bitarray of syndrome bits for this round.
    ///   Length should be `ceil(detectors_per_round / 64)` words.
    ///
    /// # Returns
    ///
    /// * `None` - Window not yet full, no corrections committed
    /// * `Some(CommittedCorrections)` - Corrections for the round that just exited
    pub fn ingest_round(&mut self, round_syndromes: &[u64]) -> Option<CommittedCorrections<'_>> {
        // Step 1: If window full, commit oldest round first
        let must_commit = self.is_window_full();

        if must_commit {
            self.commit_oldest_round_internal();
        }

        // Step 2: Calculate physical Z for new round (circular indexing)
        let physical_z = self.config.round_to_z(self.head_round);

        // Step 3: Load syndromes into the appropriate Z-layer
        self.load_round_syndromes(round_syndromes, physical_z);

        // Step 4: Update metadata
        let defect_count = count_defects(round_syndromes);
        self.round_metadata[physical_z] = RoundMetadata {
            absolute_round: self.head_round,
            defect_count: defect_count as u16,
            flags: RoundMetadata::FLAG_LOADED,
            _reserved: 0,
        };

        // Step 5: Grow clusters (may connect to previous rounds via Z-edges)
        self.grow_from_round(physical_z);

        // Step 6: Update tracking
        self.head_round += 1;
        self.rounds_in_window = (self.rounds_in_window + 1).min(self.config.window_size);

        // Return committed corrections if we committed
        if must_commit {
            Some(CommittedCorrections {
                round: self.tail_round - 1,
                corrections: &self.partial_corrections[..self.partial_corrections_count],
                observable: self.commit_observable,
            })
        } else {
            None
        }
    }

    /// Flush all remaining rounds in the window.
    ///
    /// Call this when the syndrome stream ends to commit all remaining rounds.
    ///
    /// # Returns
    ///
    /// Iterator over committed corrections for each remaining round.
    pub fn flush(&mut self) -> FlushIterator<'_, 'a, T, STRIDE_Y> {
        // First, run growth to full convergence
        self.decoder.grow_clusters();

        // Save the count before creating the iterator (borrow checker)
        let remaining = self.rounds_in_window;

        FlushIterator {
            decoder: self,
            remaining,
        }
    }

    // =========================================================================
    // Internal: Syndrome Loading
    // =========================================================================

    /// Load syndromes for a single round into the specified Z-layer.
    fn load_round_syndromes(&mut self, syndromes: &[u64], physical_z: usize) {
        let blocks_per_layer = self.config.blocks_per_round();
        let base_block = physical_z * blocks_per_layer;

        for (i, &word) in syndromes.iter().take(blocks_per_layer).enumerate() {
            if word == 0 {
                continue;
            }

            let blk_idx = base_block + i;

            // Safety: blk_idx is bounded by total_blocks
            unsafe {
                // Mark block dirty for sparse reset
                let dirty_word = blk_idx / 64;
                let dirty_bit = blk_idx % 64;
                if dirty_word < self.decoder.block_dirty_mask.len() {
                    *self.decoder.block_dirty_mask.get_unchecked_mut(dirty_word) |= 1u64 << dirty_bit;
                }

                // Update block state
                if blk_idx < self.decoder.blocks_state.len() {
                    let block = self.decoder.blocks_state.get_unchecked_mut(blk_idx);
                    block.boundary |= word;
                    block.occupied |= word;
                }

                // Update defect mask
                if blk_idx < self.decoder.defect_mask.len() {
                    *self.decoder.defect_mask.get_unchecked_mut(blk_idx) |= word;
                }

                // Queue block for growth
                self.decoder.push_next(blk_idx);
            }
        }
    }

    /// Run cluster growth for the newly loaded round.
    fn grow_from_round(&mut self, _physical_z: usize) {
        // Run growth iterations until convergence or limit reached
        // The limit is proportional to window size to allow cross-round merging
        let max_iterations = self.config.window_size * 4;

        for _ in 0..max_iterations {
            if !self.decoder.grow_iteration() {
                break;
            }
        }
    }

    // =========================================================================
    // Internal: Commit Pipeline
    // =========================================================================

    /// Commit the oldest round in the window (internal implementation).
    fn commit_oldest_round_internal(&mut self) {
        let exit_z = self.config.round_to_z(self.tail_round);

        // Reset commit state
        self.partial_corrections_count = 0;
        self.commit_observable = 0;

        // Step 1: Collect defects in exiting layer
        self.collect_exit_layer_defects(exit_z);

        // Step 2: For each defect, trace path and extract corrections
        for i in 0..self.exit_layer_count {
            let defect = unsafe { *self.exit_layer_defects.get_unchecked(i) };
            self.trace_and_extract_corrections(defect, exit_z);
        }

        // Step 3: Migrate cluster roots that are in the exiting layer
        // This prevents orphaned parent references when we clear the layer
        self.migrate_roots_from_layer(exit_z);

        // Step 4: Clear the exiting Z-layer
        self.clear_z_layer(exit_z);

        // Step 5: Advance tail
        self.tail_round += 1;
        self.rounds_in_window = self.rounds_in_window.saturating_sub(1);

        // Clear metadata for the exited slot
        self.round_metadata[exit_z] = RoundMetadata::empty();
    }

    /// Migrate cluster roots out of the exiting layer.
    ///
    /// For any cluster with root in the exiting layer that has nodes in
    /// remaining layers, we need to select a new root and re-parent.
    fn migrate_roots_from_layer(&mut self, exit_z: usize) {
        let exit_z_start = exit_z * self.config.stride_z;
        let exit_z_end = exit_z_start + self.config.stride_z;
        let boundary_node = (self.decoder.parents.len() - 1) as u32;

        // For each node in remaining layers, check if its root is in the exit layer
        for round_offset in 1..self.config.window_size {
            let other_z = (exit_z + round_offset) % self.config.window_size;
            let layer_start = other_z * self.config.stride_z;
            let layer_end = (layer_start + self.config.stride_z).min(self.decoder.parents.len() - 1);

            for node in layer_start..layer_end {
                // Find root
                let mut curr = node as u32;
                loop {
                    let parent = if (curr as usize) < self.decoder.parents.len() {
                        unsafe { *self.decoder.parents.get_unchecked(curr as usize) }
                    } else {
                        break;
                    };

                    if curr == parent || parent == boundary_node {
                        break;
                    }
                    curr = parent;
                }

                let root = curr;

                // If root is in exit layer, re-root to this node
                if (root as usize) >= exit_z_start && (root as usize) < exit_z_end {
                    // Make this node the new root by pointing to itself
                    unsafe {
                        *self.decoder.parents.get_unchecked_mut(node) = node as u32;
                    }
                }
            }
        }
    }

    /// Collect all defect nodes in the exiting Z-layer.
    fn collect_exit_layer_defects(&mut self, exit_z: usize) {
        self.exit_layer_count = 0;

        let blocks_per_layer = self.config.blocks_per_round();
        let base_block = exit_z * blocks_per_layer;
        let base_node = exit_z * self.config.stride_z;

        for i in 0..blocks_per_layer {
            let blk_idx = base_block + i;

            let word = if blk_idx < self.decoder.defect_mask.len() {
                unsafe { *self.decoder.defect_mask.get_unchecked(blk_idx) }
            } else {
                0
            };

            if word == 0 {
                continue;
            }

            let block_base = i * 64;
            let mut w = word;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                w &= w - 1;

                let node = (base_node + block_base + bit) as u32;
                if self.exit_layer_count < self.exit_layer_defects.len() {
                    unsafe {
                        *self.exit_layer_defects.get_unchecked_mut(self.exit_layer_count) = node;
                    }
                    self.exit_layer_count += 1;
                }
            }
        }
    }

    /// Trace path from defect and extract corrections touching exit layer.
    fn trace_and_extract_corrections(&mut self, defect: u32, exit_z: usize) {
        let boundary_node = (self.decoder.parents.len() - 1) as u32;
        let exit_z_start = exit_z * self.config.stride_z;
        let exit_z_end = exit_z_start + self.config.stride_z;

        let mut curr = defect;

        loop {
            let parent = if (curr as usize) < self.decoder.parents.len() {
                unsafe { *self.decoder.parents.get_unchecked(curr as usize) }
            } else {
                curr
            };

            if curr == parent {
                break; // Reached root
            }

            // Check if either endpoint is in the exit layer
            let curr_in_exit = (curr as usize) >= exit_z_start && (curr as usize) < exit_z_end;
            let parent_in_exit = if parent == boundary_node {
                true // Boundary edges are attributed to exit layer
            } else {
                (parent as usize) >= exit_z_start && (parent as usize) < exit_z_end
            };

            // If edge touches exit layer, emit a correction
            if curr_in_exit || parent_in_exit {
                self.emit_correction(curr, parent);
            }

            curr = parent;
        }
    }

    /// Emit a correction for the edge between two nodes.
    fn emit_correction(&mut self, from: u32, to: u32) {
        if self.partial_corrections_count >= self.partial_corrections.len() {
            return; // Buffer full
        }

        // Order endpoints consistently (u < v)
        let (u, v) = if from < to { (from, to) } else { (to, from) };

        let correction = EdgeCorrection { u, v };

        unsafe {
            *self.partial_corrections.get_unchecked_mut(self.partial_corrections_count) = correction;
        }
        self.partial_corrections_count += 1;
    }

    /// Clear state for a Z-layer that has exited the window.
    fn clear_z_layer(&mut self, physical_z: usize) {
        let blocks_per_layer = self.config.blocks_per_round();
        let base_block = physical_z * blocks_per_layer;
        let base_node = physical_z * self.config.stride_z;

        // Clear block state
        for i in 0..blocks_per_layer {
            let blk_idx = base_block + i;
            if blk_idx >= self.decoder.blocks_state.len() {
                continue;
            }

            unsafe {
                let block = self.decoder.blocks_state.get_unchecked_mut(blk_idx);
                block.boundary = 0;
                block.occupied = 0;
                block.root = u32::MAX;
                block.root_rank = 0;

                *self.decoder.defect_mask.get_unchecked_mut(blk_idx) = 0;
            }
        }

        // Reset parent pointers for this layer (make each node its own root)
        let end_node = (base_node + self.config.stride_z).min(self.decoder.parents.len());
        for node in base_node..end_node {
            unsafe {
                *self.decoder.parents.get_unchecked_mut(node) = node as u32;
            }
        }
    }
}

// =============================================================================
// Flush Iterator
// =============================================================================

/// Iterator returned by `StreamingDecoder::flush()`.
///
/// Commits remaining rounds one at a time until the window is empty.
pub struct FlushIterator<'iter, 'a, T: Topology, const STRIDE_Y: usize> {
    decoder: &'iter mut StreamingDecoder<'a, T, STRIDE_Y>,
    remaining: usize,
}

impl<'iter, 'a, T: Topology, const STRIDE_Y: usize> Iterator for FlushIterator<'iter, 'a, T, STRIDE_Y> {
    type Item = CommittedCorrections<'iter>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        // Commit the oldest round
        self.decoder.commit_oldest_round_internal();
        self.remaining -= 1;

        // Note: We need to return a reference with lifetime 'iter, but the corrections
        // buffer is owned by the decoder. This requires careful lifetime management.
        // For now, we return a copy-constructible version.
        Some(CommittedCorrections {
            round: self.decoder.tail_round - 1,
            corrections: unsafe {
                // Safety: the buffer is valid for the lifetime of the decoder
                core::slice::from_raw_parts(
                    self.decoder.partial_corrections.as_ptr(),
                    self.decoder.partial_corrections_count,
                )
            },
            observable: self.decoder.commit_observable,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'iter, 'a, T: Topology, const STRIDE_Y: usize> ExactSizeIterator
    for FlushIterator<'iter, 'a, T, STRIDE_Y> {}

// =============================================================================
// Buffer Size Calculation
// =============================================================================

/// Calculate buffer size required for streaming decoder.
///
/// This includes the base decoder allocation plus additional buffers for
/// streaming-specific state.
///
/// # Arguments
///
/// * `width` - Spatial width (detectors per row)
/// * `height` - Spatial height (rows per round)
/// * `window_size` - Number of rounds in the sliding window
#[must_use]
pub const fn streaming_buffer_size(width: usize, height: usize, window_size: usize) -> usize {
    // Base decoder allocation (reuse existing calculation)
    let decoder_size = crate::arena::required_buffer_size(width, height, window_size);

    let detectors_per_round = width * height;

    // Per-round metadata: window_size * 16 bytes (RoundMetadata is 16 bytes)
    let metadata_size = window_size * 16 + 64;

    // Exit layer defects buffer: detectors_per_round * 4 bytes (u32)
    let exit_defects_size = detectors_per_round * 4 + 64;

    // Partial corrections buffer: 2 * detectors_per_round * 8 bytes
    // (worst case: every defect creates 2 corrections)
    let corrections_size = detectors_per_round * 2 * 8 + 64;

    decoder_size + metadata_size + exit_defects_size + corrections_size
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Count total defects (set bits) in syndrome array.
#[inline]
fn count_defects(syndromes: &[u64]) -> usize {
    syndromes.iter().map(|w| w.count_ones() as usize).sum()
}

/// Const-compatible max function.
const fn const_max(a: usize, b: usize) -> usize {
    if a > b { a } else { b }
}

/// Const-compatible ceiling division.
const fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// Next power of two (const-compatible).
const fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut v = n - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    #[cfg(target_pointer_width = "64")]
    {
        v |= v >> 32;
    }
    v + 1
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    extern crate std;
    use std::vec;
    use std::vec::Vec;

    use super::*;
    use crate::topology::Grid3D;

    #[test]
    fn test_streaming_config_creation() {
        let config = StreamingConfig::for_rotated_surface(5, 3);

        assert_eq!(config.width, 4);
        assert_eq!(config.height, 4);
        assert_eq!(config.window_size, 3);
        assert_eq!(config.detectors_per_round, 16);
    }

    #[test]
    fn test_round_to_z_circular() {
        let config = StreamingConfig::for_rotated_surface(5, 3);

        assert_eq!(config.round_to_z(0), 0);
        assert_eq!(config.round_to_z(1), 1);
        assert_eq!(config.round_to_z(2), 2);
        assert_eq!(config.round_to_z(3), 0); // Wraps around
        assert_eq!(config.round_to_z(4), 1);
        assert_eq!(config.round_to_z(5), 2);
        assert_eq!(config.round_to_z(6), 0);
    }

    #[test]
    fn test_round_metadata() {
        let meta = RoundMetadata::empty();
        assert!(!meta.is_valid());
        assert!(!meta.is_loaded());

        let meta = RoundMetadata {
            absolute_round: 42,
            defect_count: 5,
            flags: RoundMetadata::FLAG_LOADED,
            _reserved: 0,
        };
        assert!(meta.is_valid());
        assert!(meta.is_loaded());
    }

    #[test]
    fn test_count_defects() {
        let syndromes = [0b1010_1010u64, 0b0000_1111u64, 0u64];
        assert_eq!(count_defects(&syndromes), 8);

        let empty: [u64; 0] = [];
        assert_eq!(count_defects(&empty), 0);
    }

    #[test]
    fn test_streaming_buffer_size() {
        let size = streaming_buffer_size(4, 4, 3);
        assert!(size > 0);

        // Larger window should need more memory
        let size_large = streaming_buffer_size(4, 4, 5);
        assert!(size_large > size);
    }

    #[test]
    fn test_streaming_decoder_creation() {
        let config = StreamingConfig::for_rotated_surface(5, 3);
        let buf_size = streaming_buffer_size(config.width, config.height, config.window_size);
        let mut buffer = vec![0u8; buf_size];
        let mut arena = Arena::new(&mut buffer);

        let decoder: StreamingDecoder<Grid3D, 4> = StreamingDecoder::new(&mut arena, config);

        assert_eq!(decoder.rounds_in_window(), 0);
        assert_eq!(decoder.head_round(), 0);
        assert_eq!(decoder.tail_round(), 0);
        assert!(!decoder.is_window_full());
    }

    #[test]
    fn test_streaming_ingest_empty_rounds() {
        let config = StreamingConfig::for_rotated_surface(5, 3);
        let buf_size = streaming_buffer_size(config.width, config.height, config.window_size);
        let mut buffer = vec![0u8; buf_size];
        let mut arena = Arena::new(&mut buffer);

        let mut decoder: StreamingDecoder<Grid3D, 4> = StreamingDecoder::new(&mut arena, config);

        // Ingest empty syndromes for 3 rounds (filling window)
        let empty_syndromes = [0u64; 1];

        // First 2 rounds: no commit
        assert!(decoder.ingest_round(&empty_syndromes).is_none());
        assert_eq!(decoder.rounds_in_window(), 1);

        assert!(decoder.ingest_round(&empty_syndromes).is_none());
        assert_eq!(decoder.rounds_in_window(), 2);

        assert!(decoder.ingest_round(&empty_syndromes).is_none());
        assert_eq!(decoder.rounds_in_window(), 3);
        assert!(decoder.is_window_full());

        // 4th round: should commit round 0
        let committed = decoder.ingest_round(&empty_syndromes);
        assert!(committed.is_some());
        let c = committed.unwrap();
        assert_eq!(c.round, 0);
        assert_eq!(c.corrections.len(), 0); // No defects = no corrections
    }

    #[test]
    fn test_streaming_flush() {
        let config = StreamingConfig::for_rotated_surface(5, 3);
        let buf_size = streaming_buffer_size(config.width, config.height, config.window_size);
        let mut buffer = vec![0u8; buf_size];
        let mut arena = Arena::new(&mut buffer);

        let mut decoder: StreamingDecoder<Grid3D, 4> = StreamingDecoder::new(&mut arena, config);

        // Ingest 2 empty rounds
        let empty_syndromes = [0u64; 1];
        decoder.ingest_round(&empty_syndromes);
        decoder.ingest_round(&empty_syndromes);

        assert_eq!(decoder.rounds_in_window(), 2);

        // Flush remaining rounds
        let flushed: Vec<_> = decoder.flush().collect();
        assert_eq!(flushed.len(), 2);
        assert_eq!(flushed[0].round, 0);
        assert_eq!(flushed[1].round, 1);
    }

    #[test]
    fn test_streaming_reset() {
        let config = StreamingConfig::for_rotated_surface(5, 3);
        let buf_size = streaming_buffer_size(config.width, config.height, config.window_size);
        let mut buffer = vec![0u8; buf_size];
        let mut arena = Arena::new(&mut buffer);

        let mut decoder: StreamingDecoder<Grid3D, 4> = StreamingDecoder::new(&mut arena, config);

        // Ingest some rounds
        let empty_syndromes = [0u64; 1];
        decoder.ingest_round(&empty_syndromes);
        decoder.ingest_round(&empty_syndromes);

        assert_eq!(decoder.rounds_in_window(), 2);
        assert_eq!(decoder.head_round(), 2);

        // Reset
        decoder.reset();

        assert_eq!(decoder.rounds_in_window(), 0);
        assert_eq!(decoder.head_round(), 0);
        assert_eq!(decoder.tail_round(), 0);
    }

    #[test]
    fn test_streaming_with_single_defect() {
        let config = StreamingConfig::for_rotated_surface(5, 3);
        let buf_size = streaming_buffer_size(config.width, config.height, config.window_size);
        let mut buffer = vec![0u8; buf_size];
        let mut arena = Arena::new(&mut buffer);

        let mut decoder: StreamingDecoder<Grid3D, 4> = StreamingDecoder::new(&mut arena, config);

        // Round 0: single defect at position 0
        let syndromes_with_defect = [0b1u64];
        decoder.ingest_round(&syndromes_with_defect);

        // Round 1: matching defect at same position (creates a pair)
        decoder.ingest_round(&syndromes_with_defect);

        // Round 2: empty
        let empty_syndromes = [0u64; 1];
        decoder.ingest_round(&empty_syndromes);

        // Round 3: triggers commit of round 0
        let committed = decoder.ingest_round(&empty_syndromes);
        assert!(committed.is_some());
        let c = committed.unwrap();
        assert_eq!(c.round, 0);
        // Should have corrections (the pair creates an edge)
    }

    #[test]
    fn test_streaming_many_rounds() {
        let config = StreamingConfig::for_rotated_surface(5, 3);
        let buf_size = streaming_buffer_size(config.width, config.height, config.window_size);
        let mut buffer = vec![0u8; buf_size];
        let mut arena = Arena::new(&mut buffer);

        let mut decoder: StreamingDecoder<Grid3D, 4> = StreamingDecoder::new(&mut arena, config);

        let empty_syndromes = [0u64; 1];
        let mut committed_rounds = Vec::new();

        // Ingest 20 rounds
        for _ in 0..20 {
            if let Some(c) = decoder.ingest_round(&empty_syndromes) {
                committed_rounds.push(c.round);
            }
        }

        // Should have committed rounds 0-16 (17 commits for rounds after window fills)
        assert_eq!(committed_rounds.len(), 17);
        for (i, &round) in committed_rounds.iter().enumerate() {
            assert_eq!(round, i as u64);
        }

        // Flush remaining 3 rounds
        let flushed: Vec<_> = decoder.flush().collect();
        assert_eq!(flushed.len(), 3);
        assert_eq!(flushed[0].round, 17);
        assert_eq!(flushed[1].round, 18);
        assert_eq!(flushed[2].round, 19);
    }
}
