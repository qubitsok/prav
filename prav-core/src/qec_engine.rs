//! High-level QEC decoder API.
//!
//! This module provides [`QecEngine`], a convenient wrapper around [`DecodingState`]
//! that handles the common decode cycle: reset, load syndromes, decode.

use crate::arena::Arena;
use crate::decoder::{DecodingState, EdgeCorrection};
use crate::topology::Topology;

/// High-level quantum error correction decoder engine.
///
/// `QecEngine` wraps a [`DecodingState`] and provides a simple interface for
/// repeated decoding cycles. It automatically handles the sparse reset between
/// cycles for optimal performance.
///
/// # Type Parameters
///
/// - `'a` - Lifetime of the backing arena memory
/// - `T: Topology` - The lattice topology (e.g., [`SquareGrid`](crate::SquareGrid))
/// - `STRIDE_Y` - Compile-time Y stride (must match grid dimensions)
///
/// # Example
///
/// ```ignore
/// use prav_core::{Arena, QecEngine, SquareGrid, EdgeCorrection};
///
/// // Setup
/// let mut buffer = [0u8; 1024 * 1024];
/// let mut arena = Arena::new(&mut buffer);
/// let mut engine: QecEngine<SquareGrid, 32> = QecEngine::new(&mut arena, 32, 32, 1);
///
/// // Decode cycle
/// let mut corrections = [EdgeCorrection::default(); 512];
/// let syndromes: &[u64] = &[/* syndrome data */];
///
/// let num_corrections = engine.process_cycle_dense(syndromes, &mut corrections);
///
/// // Apply corrections[0..num_corrections] to your physical qubits
/// ```
///
/// # Performance
///
/// For repeated decoding, `QecEngine` uses sparse reset internally, which only
/// resets modified blocks. This is much faster than full reinitialization at
/// typical error rates.
pub struct QecEngine<'a, T: Topology, const STRIDE_Y: usize> {
    /// Internal decoder state.
    decoder: DecodingState<'a, T, STRIDE_Y>,
}

impl<'a, T: Topology + Sync, const STRIDE_Y: usize> QecEngine<'a, T, STRIDE_Y> {
    /// Creates a new QEC engine for the given grid dimensions.
    ///
    /// # Arguments
    ///
    /// * `arena` - Arena allocator for all internal allocations.
    /// * `width` - Grid width in nodes.
    /// * `height` - Grid height in nodes.
    /// * `depth` - Grid depth (1 for 2D codes, >1 for 3D codes).
    ///
    /// # Panics
    ///
    /// Panics if `STRIDE_Y` doesn't match the calculated stride for the dimensions.
    /// The stride is `max(width, height, depth).next_power_of_two()`.
    pub fn new(arena: &mut Arena<'a>, width: usize, height: usize, depth: usize) -> Self {
        Self {
            decoder: DecodingState::new(arena, width, height, depth),
        }
    }

    /// Processes a complete decoding cycle with dense syndrome input.
    ///
    /// This is the main entry point for decoding. It performs:
    /// 1. Sparse reset of modified state from previous cycle
    /// 2. Syndrome loading from dense bitarray
    /// 3. Cluster growth
    /// 4. Correction extraction via peeling
    ///
    /// # Arguments
    ///
    /// * `dense_defects` - Syndrome measurements as dense bitarray. Each u64
    ///   represents 64 nodes, where bit `i` indicates node `(blk * 64 + i)` has
    ///   a syndrome.
    /// * `out_corrections` - Output buffer for edge corrections. Should be sized
    ///   to hold the maximum expected corrections (typically `num_nodes / 2`).
    ///
    /// # Returns
    ///
    /// Number of corrections written to `out_corrections`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let syndromes = generate_syndromes(error_rate);
    /// let mut corrections = [EdgeCorrection::default(); 1024];
    ///
    /// let count = engine.process_cycle_dense(&syndromes, &mut corrections);
    /// for i in 0..count {
    ///     apply_correction(corrections[i]);
    /// }
    /// ```
    pub fn process_cycle_dense(
        &mut self,
        dense_defects: &[u64],
        out_corrections: &mut [EdgeCorrection],
    ) -> usize {
        self.decoder.sparse_reset();

        self.decoder.load_dense_syndromes(dense_defects);

        self.decoder.decode(out_corrections)
    }

    /// Loads erasure information for the next decoding cycle.
    ///
    /// Erasures indicate qubits that were lost (e.g., photon loss in optical
    /// systems). Erased qubits are excluded from cluster growth.
    ///
    /// # Arguments
    ///
    /// * `erasures` - Dense bitarray where bit `i` in `erasures[blk]` indicates
    ///   node `(blk * 64 + i)` is erased.
    ///
    /// # Note
    ///
    /// Call this before `process_cycle_dense` if your system has erasures.
    /// The erasure mask persists until changed.
    pub fn load_erasures(&mut self, erasures: &[u64]) {
        self.decoder.load_erasures(erasures);
    }
}
