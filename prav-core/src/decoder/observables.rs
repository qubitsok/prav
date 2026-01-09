//! Observable tracking for logical error measurement.
//!
//! This module provides types and utilities for tracking which logical observables
//! are affected by error corrections. The decoder accumulates observable frame changes
//! as it emits corrections, enabling proper comparison with ground truth.
//!
//! # Background
//!
//! In quantum error correction, each error mechanism may flip one or more logical
//! observables. The decoder must track these flips to determine if the correction
//! results in a logical error. This is done by XOR-accumulating frame changes as
//! corrections are emitted.
//!
//! # Observable Modes
//!
//! - [`ObservableMode::Disabled`]: No observable tracking (zero overhead)
//! - [`ObservableMode::Phenomenological`]: Boundary-based tracking for rotated surface codes
//! - [`ObservableMode::CircuitLevel`]: DEM-based tracking with per-edge frame changes

/// Observable tracking configuration.
///
/// Controls how the decoder tracks logical observable flips during correction.
/// Each mode has different performance characteristics and use cases.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum ObservableMode {
    /// No observable tracking. Zero runtime overhead.
    ///
    /// Use this mode when you don't need logical error rate measurement,
    /// or when you're implementing external observable tracking.
    #[default]
    Disabled = 0,

    /// Phenomenological observable tracking for rotated surface codes.
    ///
    /// Boundary corrections determine observable flips:
    /// - Left/right boundary (x = 0 or x = width-1) → Z logical observable
    /// - Top/bottom boundary (y = 0 or y = height-1) → X logical observable
    ///
    /// This mode is suitable for phenomenological noise models where
    /// interior edges do not affect logical observables.
    ///
    /// Runtime overhead: ~5 cycles per boundary correction (coordinate check).
    Phenomenological = 1,

    /// Circuit-level observable tracking using DEM frame changes.
    ///
    /// Each edge has an associated frame_changes bitmask from the Detector
    /// Error Model. The decoder looks up and XOR-accumulates these values
    /// as corrections are emitted.
    ///
    /// This mode supports arbitrary noise models and hyperedge decompositions.
    ///
    /// Runtime overhead: ~3 cycles per edge (indexed lookup + XOR).
    CircuitLevel = 2,
}

/// Arena-allocated edge→observable lookup table for circuit-level tracking.
///
/// Maps edge indices and boundary nodes to their frame_changes bitmasks.
/// The lookup table is built from a parsed Detector Error Model (DEM).
///
/// # Memory Layout
///
/// - `edge_lut`: `[u8; num_nodes * 4]` - indexed by `node * 4 + direction`
/// - `boundary_lut`: `[u8; num_nodes]` - indexed by boundary node
///
/// The `u8` bitmask supports up to 8 logical observables, which is sufficient
/// for most practical codes (rotated surface codes have 1 or 2 observables).
///
/// # Usage
///
/// ```ignore
/// // Build LUT from parsed DEM
/// let lut = EdgeObservableLut::from_dem(&mut arena, &dem, &config)?;
///
/// // Load into decoder
/// decoder.set_observable_lut(&lut);
///
/// // After decoding, get predicted observables
/// let predicted = decoder.predicted_observables();
/// ```
#[derive(Clone, Copy, Debug)]
pub struct EdgeObservableLut<'a> {
    /// Edge→frame_changes lookup.
    ///
    /// Indexed by `node_idx * 4 + direction` where direction is:
    /// - 0: +X edge
    /// - 1: +Y edge
    /// - 2: +Z edge (temporal)
    /// - 3: unused padding
    pub edge_lut: &'a [u8],

    /// Boundary node→frame_changes lookup.
    ///
    /// Indexed by node index. Contains the frame_changes for boundary
    /// corrections (when a defect is matched to the boundary).
    pub boundary_lut: &'a [u8],
}

impl<'a> EdgeObservableLut<'a> {
    /// Creates a new observable lookup table with the given slices.
    ///
    /// # Arguments
    ///
    /// * `edge_lut` - Edge frame_changes lookup, size = num_nodes * 4
    /// * `boundary_lut` - Boundary frame_changes lookup, size = num_nodes
    #[inline]
    pub const fn new(edge_lut: &'a [u8], boundary_lut: &'a [u8]) -> Self {
        Self {
            edge_lut,
            boundary_lut,
        }
    }

    /// Looks up the frame_changes for an interior edge.
    ///
    /// # Arguments
    ///
    /// * `edge_idx` - Edge index computed as `node * 4 + direction`
    ///
    /// # Returns
    ///
    /// The frame_changes bitmask, or 0 if the index is out of bounds.
    #[inline(always)]
    pub fn get_edge_observable(&self, edge_idx: usize) -> u8 {
        self.edge_lut.get(edge_idx).copied().unwrap_or(0)
    }

    /// Looks up the frame_changes for a boundary correction.
    ///
    /// # Arguments
    ///
    /// * `node` - The boundary node index
    ///
    /// # Returns
    ///
    /// The frame_changes bitmask, or 0 if the index is out of bounds.
    #[inline(always)]
    pub fn get_boundary_observable(&self, node: usize) -> u8 {
        self.boundary_lut.get(node).copied().unwrap_or(0)
    }
}

/// Computes the phenomenological boundary observable for a node.
///
/// For rotated surface codes:
/// - Left boundary (x = 0) → Z observable (bit 1)
/// - Right boundary (x = width-1) → Z observable (bit 1)
/// - Bottom boundary (y = 0) → X observable (bit 0)
/// - Top boundary (y = height-1) → X observable (bit 0)
///
/// Corner nodes may affect both observables.
///
/// # Arguments
///
/// * `x` - X coordinate of the boundary node
/// * `y` - Y coordinate of the boundary node
/// * `width` - Grid width
/// * `height` - Grid height
///
/// # Returns
///
/// Bitmask of affected observables (bit 0 = X, bit 1 = Z).
#[inline(always)]
pub fn phenomenological_boundary_observable(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> u8 {
    let mut obs = 0u8;

    // Left/right boundaries affect Z observable
    if x == 0 || x == width.saturating_sub(1) {
        obs |= 0b10; // Z observable
    }

    // Top/bottom boundaries affect X observable
    if y == 0 || y == height.saturating_sub(1) {
        obs |= 0b01; // X observable
    }

    obs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observable_mode_default() {
        assert_eq!(ObservableMode::default(), ObservableMode::Disabled);
    }

    #[test]
    fn test_phenomenological_boundary_observable() {
        // 5x5 grid (indices 0-4)
        let width = 5;
        let height = 5;

        // Interior node - no observable
        assert_eq!(phenomenological_boundary_observable(2, 2, width, height), 0);

        // Left boundary - Z observable
        assert_eq!(phenomenological_boundary_observable(0, 2, width, height), 0b10);

        // Right boundary - Z observable
        assert_eq!(phenomenological_boundary_observable(4, 2, width, height), 0b10);

        // Bottom boundary - X observable
        assert_eq!(phenomenological_boundary_observable(2, 0, width, height), 0b01);

        // Top boundary - X observable
        assert_eq!(phenomenological_boundary_observable(2, 4, width, height), 0b01);

        // Bottom-left corner - both observables
        assert_eq!(phenomenological_boundary_observable(0, 0, width, height), 0b11);

        // Top-right corner - both observables
        assert_eq!(phenomenological_boundary_observable(4, 4, width, height), 0b11);
    }

    #[test]
    fn test_edge_observable_lut() {
        let edge_data = [0u8, 1, 2, 0, 3, 0, 0, 0];
        let boundary_data = [0u8, 1, 2];

        let lut = EdgeObservableLut::new(&edge_data, &boundary_data);

        assert_eq!(lut.get_edge_observable(0), 0);
        assert_eq!(lut.get_edge_observable(1), 1);
        assert_eq!(lut.get_edge_observable(4), 3);
        assert_eq!(lut.get_edge_observable(100), 0); // Out of bounds

        assert_eq!(lut.get_boundary_observable(0), 0);
        assert_eq!(lut.get_boundary_observable(1), 1);
        assert_eq!(lut.get_boundary_observable(100), 0); // Out of bounds
    }
}
