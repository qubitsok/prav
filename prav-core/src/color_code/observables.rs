//! Observable tracking for color code decoding.
//!
//! Color codes have different logical observable structures than surface codes.
//! This module provides tracking for both phenomenological and circuit-level
//! noise models.
//!
//! # Color Code Observables
//!
//! In a triangular color code:
//! - Logical X and Z operators are string-like, connecting boundaries
//! - Each color boundary corresponds to a different logical operator path
//! - Corrections crossing certain boundaries flip the logical frame

use crate::color_code::types::FaceColor;

/// Mode for tracking logical observables in color codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorObservableMode {
    /// No observable tracking (maximum performance).
    Disabled,
    /// Phenomenological model: track boundary crossings.
    /// Each color's boundary crossing flips the corresponding observable bit.
    Phenomenological,
    /// Circuit-level model: use per-edge observable lookup table.
    /// Required for accurate tracking with realistic noise.
    CircuitLevel,
}

impl Default for ColorObservableMode {
    fn default() -> Self {
        Self::Phenomenological
    }
}

/// Per-color observable frame tracking.
///
/// Tracks logical frame flips for each color class independently,
/// then combines them for the overall logical observable.
#[derive(Debug, Clone, Copy, Default)]
pub struct ColorObservableFrame {
    /// Observable frame for red restricted subgraph.
    pub red: u8,
    /// Observable frame for green restricted subgraph.
    pub green: u8,
    /// Observable frame for blue restricted subgraph.
    pub blue: u8,
}

impl ColorObservableFrame {
    /// Create a new frame with all zeros.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            red: 0,
            green: 0,
            blue: 0,
        }
    }

    /// Reset all frames to zero.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.red = 0;
        self.green = 0;
        self.blue = 0;
    }

    /// Accumulate a frame flip for a specific color.
    #[inline(always)]
    pub fn accumulate(&mut self, color: FaceColor, flip: u8) {
        match color {
            FaceColor::Red => self.red ^= flip,
            FaceColor::Green => self.green ^= flip,
            FaceColor::Blue => self.blue ^= flip,
        }
    }

    /// Get the frame for a specific color.
    #[inline(always)]
    #[must_use]
    pub const fn get(&self, color: FaceColor) -> u8 {
        match color {
            FaceColor::Red => self.red,
            FaceColor::Green => self.green,
            FaceColor::Blue => self.blue,
        }
    }

    /// Get the combined logical frame (XOR of all colors).
    ///
    /// The interpretation depends on the color code structure:
    /// - Bit 0: X logical observable
    /// - Bit 1: Z logical observable (if applicable)
    #[inline(always)]
    #[must_use]
    pub const fn combined(&self) -> u8 {
        self.red ^ self.green ^ self.blue
    }

    /// Check if any logical error occurred.
    #[inline(always)]
    #[must_use]
    pub const fn has_error(&self) -> bool {
        self.combined() != 0
    }

    /// Get per-color frames as an array `[red, green, blue]`.
    #[inline(always)]
    #[must_use]
    pub const fn as_array(&self) -> [u8; 3] {
        [self.red, self.green, self.blue]
    }
}

/// Lookup table for circuit-level observable tracking.
///
/// In circuit-level noise, each edge in the decoding graph may contribute
/// to the logical observable. This table maps (edge_id) → frame_change.
pub struct ColorEdgeObservableLut<'a> {
    /// Per-color lookup tables.
    /// `tables[color][edge_id] = frame_change`
    tables: [&'a [u8]; 3],
}

impl<'a> ColorEdgeObservableLut<'a> {
    /// Create a new lookup table from per-color tables.
    #[must_use]
    pub const fn new(red: &'a [u8], green: &'a [u8], blue: &'a [u8]) -> Self {
        Self {
            tables: [red, green, blue],
        }
    }

    /// Get the frame change for an edge in a specific color's subgraph.
    #[inline(always)]
    #[must_use]
    pub fn lookup(&self, color: FaceColor, edge_id: usize) -> u8 {
        let table = self.tables[color.index()];
        if edge_id < table.len() {
            table[edge_id]
        } else {
            0
        }
    }
}

/// Compute phenomenological boundary observable for a color code.
///
/// In the phenomenological model, corrections that cross certain boundaries
/// flip the logical observable. For color codes:
/// - Red boundary crossing: may flip X observable
/// - Green boundary crossing: may flip Z observable
/// - Blue boundary crossing: may flip both (XZ)
///
/// The exact mapping depends on the boundary structure.
///
/// # Arguments
/// * `color` - Which color's restricted subgraph
/// * `is_boundary_correction` - Whether this correction reaches a boundary
/// * `boundary_side` - Which side of the boundary (0 = top/left, 1 = bottom/right)
///
/// # Returns
/// Observable flip bitmask (bit 0 = X, bit 1 = Z)
#[inline(always)]
#[must_use]
pub const fn phenomenological_color_observable(
    color: FaceColor,
    is_boundary_correction: bool,
    boundary_side: u8,
) -> u8 {
    if !is_boundary_correction {
        return 0;
    }

    // In standard color code:
    // - Red boundary → X observable (bit 0)
    // - Green boundary → Z observable (bit 1)
    // - Blue boundary → combination
    match color {
        FaceColor::Red => {
            if boundary_side == 0 {
                0b01 // X flip
            } else {
                0
            }
        }
        FaceColor::Green => {
            if boundary_side == 0 {
                0b10 // Z flip
            } else {
                0
            }
        }
        FaceColor::Blue => {
            // Blue boundary typically doesn't directly flip observables
            // in standard triangular color codes
            0
        }
    }
}

/// Observable accumulator for a complete decoding round.
///
/// Collects observable flips from all three color decoders and
/// produces the final logical frame.
#[derive(Debug, Default)]
pub struct ObservableAccumulator {
    /// Current frame.
    frame: ColorObservableFrame,
    /// Observable mode.
    mode: ColorObservableMode,
}

impl ObservableAccumulator {
    /// Create a new accumulator with given mode.
    #[must_use]
    pub const fn new(mode: ColorObservableMode) -> Self {
        Self {
            frame: ColorObservableFrame::new(),
            mode,
        }
    }

    /// Reset for a new decoding round.
    #[inline(always)]
    pub fn reset(&mut self) {
        self.frame.reset();
    }

    /// Record an observable flip from a color's decoder.
    #[inline(always)]
    pub fn record(&mut self, color: FaceColor, flip: u8) {
        if self.mode != ColorObservableMode::Disabled {
            self.frame.accumulate(color, flip);
        }
    }

    /// Get the final logical frame.
    #[inline(always)]
    #[must_use]
    pub const fn logical_frame(&self) -> u8 {
        self.frame.combined()
    }

    /// Get the per-color frames.
    #[inline(always)]
    #[must_use]
    pub const fn color_frames(&self) -> ColorObservableFrame {
        self.frame
    }

    /// Check if observable tracking is enabled.
    #[inline(always)]
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        !matches!(self.mode, ColorObservableMode::Disabled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_accumulation() {
        let mut frame = ColorObservableFrame::new();

        assert_eq!(frame.combined(), 0);

        frame.accumulate(FaceColor::Red, 0b01);
        assert_eq!(frame.red, 0b01);
        assert_eq!(frame.combined(), 0b01);

        frame.accumulate(FaceColor::Green, 0b10);
        assert_eq!(frame.green, 0b10);
        assert_eq!(frame.combined(), 0b11);

        // XOR cancellation
        frame.accumulate(FaceColor::Red, 0b01);
        assert_eq!(frame.red, 0);
        assert_eq!(frame.combined(), 0b10);
    }

    #[test]
    fn test_frame_reset() {
        let mut frame = ColorObservableFrame::new();
        frame.accumulate(FaceColor::Red, 0xFF);
        frame.accumulate(FaceColor::Green, 0xFF);
        frame.accumulate(FaceColor::Blue, 0xFF);

        frame.reset();
        assert_eq!(frame.combined(), 0);
    }

    #[test]
    fn test_accumulator_disabled() {
        let mut acc = ObservableAccumulator::new(ColorObservableMode::Disabled);
        acc.record(FaceColor::Red, 0xFF);
        acc.record(FaceColor::Green, 0xFF);

        // Should not accumulate when disabled
        assert_eq!(acc.logical_frame(), 0);
    }

    #[test]
    fn test_accumulator_phenomenological() {
        let mut acc = ObservableAccumulator::new(ColorObservableMode::Phenomenological);

        acc.record(FaceColor::Red, 0b01);
        acc.record(FaceColor::Green, 0b10);

        assert_eq!(acc.logical_frame(), 0b11);

        acc.reset();
        assert_eq!(acc.logical_frame(), 0);
    }

    #[test]
    fn test_phenomenological_observable() {
        // Red boundary correction should flip X
        let flip = phenomenological_color_observable(FaceColor::Red, true, 0);
        assert_eq!(flip, 0b01);

        // Green boundary correction should flip Z
        let flip = phenomenological_color_observable(FaceColor::Green, true, 0);
        assert_eq!(flip, 0b10);

        // Non-boundary correction should not flip
        let flip = phenomenological_color_observable(FaceColor::Red, false, 0);
        assert_eq!(flip, 0);
    }

    #[test]
    fn test_edge_lut() {
        let red_table = [0u8, 1, 0, 1];
        let green_table = [0u8, 0, 2, 2];
        let blue_table = [0u8; 4];

        let lut = ColorEdgeObservableLut::new(&red_table, &green_table, &blue_table);

        assert_eq!(lut.lookup(FaceColor::Red, 1), 1);
        assert_eq!(lut.lookup(FaceColor::Green, 2), 2);
        assert_eq!(lut.lookup(FaceColor::Blue, 0), 0);

        // Out of bounds should return 0
        assert_eq!(lut.lookup(FaceColor::Red, 100), 0);
    }
}
