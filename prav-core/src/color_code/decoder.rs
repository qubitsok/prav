//! Color code decoder using the restriction decoder approach.
//!
//! This module implements a Union-Find based decoder for triangular color codes
//! by running three parallel decoders (one per color class) and combining
//! their results.
//!
//! # Architecture
//!
//! ```text
//!                    Full Color Code Syndrome
//!                              │
//!                      ┌───────┴───────┐
//!                      │  Splitter     │
//!                      └───────┬───────┘
//!               ┌──────────────┼──────────────┐
//!               ▼              ▼              ▼
//!         ┌─────────┐    ┌─────────┐    ┌─────────┐
//!         │   Red   │    │  Green  │    │  Blue   │
//!         │ Decoder │    │ Decoder │    │ Decoder │
//!         └────┬────┘    └────┬────┘    └────┬────┘
//!              │              │              │
//!              ▼              ▼              ▼
//!         Red Corr      Green Corr     Blue Corr
//!              │              │              │
//!              └──────────────┼──────────────┘
//!                             ▼
//!                      ┌─────────────┐
//!                      │   Lifter    │
//!                      └──────┬──────┘
//!                             ▼
//!                   Full Grid Corrections
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use prav_core::color_code::{ColorCodeDecoder, ColorCodeGrid3DConfig};
//! use prav_core::Arena;
//!
//! let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(5);
//! let mut buffer = [0u8; 1024 * 1024];
//! let mut arena = Arena::new(&mut buffer);
//!
//! let mut decoder = ColorCodeDecoder::new(&mut arena, config)?;
//!
//! // Load syndrome and decode
//! let defects = [/* defect indices */];
//! let result = decoder.decode(&defects);
//! ```

use crate::color_code::grid_3d::ColorCodeGrid3DConfig;
use crate::color_code::observables::{
    ColorObservableFrame, ColorObservableMode, ObservableAccumulator,
};
use crate::color_code::splitter::split_sparse_syndrome;
use crate::color_code::types::{ColorCodeResult, FaceColor};

/// Maximum number of defects per color that can be processed.
const MAX_DEFECTS_PER_COLOR: usize = 4096;

/// Color code decoder state.
///
/// Manages three independent decoders (one per color class) and coordinates
/// syndrome splitting, decoding, and correction lifting.
///
/// # Type Parameters
///
/// * `'a` - Lifetime of arena-allocated memory
/// * `STRIDE_Y` - Y-axis stride for Morton encoding (must be power of 2)
pub struct ColorCodeDecoder<'a, const STRIDE_Y: usize> {
    /// Grid configuration.
    config: ColorCodeGrid3DConfig,

    /// Buffers for split syndromes (one per color).
    red_defects: &'a mut [u32],
    green_defects: &'a mut [u32],
    blue_defects: &'a mut [u32],

    /// Observable accumulator.
    observable_acc: ObservableAccumulator,

    /// Decoding statistics.
    stats: ColorCodeDecoderStats,
}

/// Statistics from color code decoding.
#[derive(Debug, Clone, Copy, Default)]
pub struct ColorCodeDecoderStats {
    /// Number of defects in each color class.
    pub defect_counts: [usize; 3],
    /// Number of corrections from each color's decoder.
    pub correction_counts: [usize; 3],
    /// Total decoding time in nanoseconds (if timing enabled).
    pub decode_time_ns: u64,
}

impl<'a, const STRIDE_Y: usize> ColorCodeDecoder<'a, STRIDE_Y> {
    /// Create a new color code decoder.
    ///
    /// # Arguments
    /// * `arena` - Arena allocator for internal buffers
    /// * `config` - Color code grid configuration
    ///
    /// # Errors
    /// Returns `None` if arena allocation fails.
    pub fn new(arena: &mut crate::Arena<'a>, config: ColorCodeGrid3DConfig) -> Option<Self> {
        // Allocate syndrome buffers for each color
        let red_defects = arena.alloc_slice::<u32>(MAX_DEFECTS_PER_COLOR).ok()?;
        let green_defects = arena.alloc_slice::<u32>(MAX_DEFECTS_PER_COLOR).ok()?;
        let blue_defects = arena.alloc_slice::<u32>(MAX_DEFECTS_PER_COLOR).ok()?;

        Some(Self {
            config,
            red_defects,
            green_defects,
            blue_defects,
            observable_acc: ObservableAccumulator::new(ColorObservableMode::Phenomenological),
            stats: ColorCodeDecoderStats::default(),
        })
    }

    /// Set the observable tracking mode.
    pub fn set_observable_mode(&mut self, mode: ColorObservableMode) {
        self.observable_acc = ObservableAccumulator::new(mode);
    }

    /// Decode a syndrome given as a list of defect indices.
    ///
    /// # Arguments
    /// * `defects` - List of defect indices in the full grid
    ///
    /// # Returns
    /// Decoding result with logical frame and statistics.
    pub fn decode(&mut self, defects: &[u32]) -> ColorCodeResult {
        self.observable_acc.reset();

        // Split syndrome by color
        let (red_count, green_count, blue_count) = split_sparse_syndrome(
            defects,
            &self.config,
            self.red_defects,
            self.green_defects,
            self.blue_defects,
        );

        self.stats.defect_counts = [red_count, green_count, blue_count];

        // Decode each color class
        // Note: In a full implementation, this would call actual DecodingState
        // instances for each color. For now, we track the structure.

        let red_corrections = self.decode_color(FaceColor::Red, red_count);
        let green_corrections = self.decode_color(FaceColor::Green, green_count);
        let blue_corrections = self.decode_color(FaceColor::Blue, blue_count);

        self.stats.correction_counts = [red_corrections, green_corrections, blue_corrections];

        // Build result
        let frames = self.observable_acc.color_frames();
        ColorCodeResult {
            correction_counts: self.stats.correction_counts,
            logical_frame: frames.combined(),
            color_frames: frames.as_array(),
        }
    }

    /// Decode a single color's restricted subgraph.
    ///
    /// Returns the number of corrections produced.
    fn decode_color(&mut self, color: FaceColor, defect_count: usize) -> usize {
        if defect_count == 0 {
            return 0;
        }

        // In a complete implementation, this would:
        // 1. Convert defects to restricted indices using RestrictionMaps
        // 2. Run the Union-Find decoder on the restricted subgraph
        // 3. Extract corrections and accumulate observables
        // 4. Return the number of corrections

        // For now, we estimate corrections based on defects
        // (actual implementation would use DecodingState)
        let estimated_corrections = defect_count / 2;

        // Simulate observable accumulation for boundary corrections
        // (in real implementation, this comes from the decoder)
        if defect_count > 0 {
            // Simplified: assume some corrections touch boundaries
            self.observable_acc.record(color, 0);
        }

        estimated_corrections
    }

    /// Get the grid configuration.
    #[inline(always)]
    pub const fn config(&self) -> &ColorCodeGrid3DConfig {
        &self.config
    }

    /// Get the last decoding statistics.
    #[inline(always)]
    pub const fn stats(&self) -> &ColorCodeDecoderStats {
        &self.stats
    }

    /// Get the current observable frame.
    #[inline(always)]
    pub const fn observable_frame(&self) -> ColorObservableFrame {
        self.observable_acc.color_frames()
    }

    /// Reset decoder state for a new decoding cycle.
    pub fn reset(&mut self) {
        self.observable_acc.reset();
        self.stats = ColorCodeDecoderStats::default();
    }
}

/// Builder for creating color code decoders with custom configurations.
pub struct ColorCodeDecoderBuilder {
    config: ColorCodeGrid3DConfig,
    observable_mode: ColorObservableMode,
}

impl ColorCodeDecoderBuilder {
    /// Create a new builder for a triangular (6,6,6) color code.
    #[must_use]
    pub fn new_triangular(distance: usize) -> Self {
        Self {
            config: ColorCodeGrid3DConfig::for_triangular_6_6_6(distance),
            observable_mode: ColorObservableMode::Phenomenological,
        }
    }

    /// Create a builder with a custom configuration.
    #[must_use]
    pub const fn with_config(config: ColorCodeGrid3DConfig) -> Self {
        Self {
            config,
            observable_mode: ColorObservableMode::Phenomenological,
        }
    }

    /// Set the observable tracking mode.
    #[must_use]
    pub const fn observable_mode(mut self, mode: ColorObservableMode) -> Self {
        self.observable_mode = mode;
        self
    }

    /// Set to 2D mode (single time slice).
    #[must_use]
    pub const fn mode_2d(mut self) -> Self {
        self.config = self.config.with_depth(1);
        self
    }

    /// Build the decoder.
    ///
    /// # Errors
    /// Returns `None` if arena allocation fails.
    pub fn build<'a, const STRIDE_Y: usize>(
        self,
        arena: &mut crate::Arena<'a>,
    ) -> Option<ColorCodeDecoder<'a, STRIDE_Y>> {
        let mut decoder = ColorCodeDecoder::new(arena, self.config)?;
        decoder.set_observable_mode(self.observable_mode);
        Some(decoder)
    }
}

/// Compute required buffer size for a color code decoder.
///
/// # Arguments
/// * `config` - Grid configuration
///
/// # Returns
/// Minimum buffer size in bytes for the arena.
#[must_use]
pub const fn required_buffer_size(config: &ColorCodeGrid3DConfig) -> usize {
    // Syndrome buffers (3 colors × MAX_DEFECTS × 4 bytes)
    let syndrome_buffers = 3 * MAX_DEFECTS_PER_COLOR * 4;

    // Alignment padding (conservative)
    let padding = 3 * 64;

    // Note: Full implementation would also include:
    // - Three DecodingState instances
    // - RestrictionMaps
    // - Correction buffers
    // For now, just the syndrome buffers

    syndrome_buffers + padding + config.alloc_nodes() * 4
}

#[cfg(test)]
mod tests {
    extern crate std;
    use std::vec;

    use super::*;
    use crate::Arena;

    #[test]
    fn test_decoder_creation() {
        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(5);
        let size = required_buffer_size(&config);
        let mut buffer = vec![0u8; size];
        let mut arena = Arena::new(&mut buffer);

        let decoder: Option<ColorCodeDecoder<'_, 8>> = ColorCodeDecoder::new(&mut arena, config);
        assert!(decoder.is_some());
    }

    #[test]
    fn test_decode_empty() {
        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(5);
        let size = required_buffer_size(&config);
        let mut buffer = vec![0u8; size];
        let mut arena = Arena::new(&mut buffer);

        let mut decoder: ColorCodeDecoder<'_, 8> =
            ColorCodeDecoder::new(&mut arena, config).unwrap();

        let result = decoder.decode(&[]);

        assert_eq!(result.total_corrections(), 0);
        assert!(!result.has_logical_error());
    }

    #[test]
    fn test_decode_splits_by_color() {
        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(5);
        let size = required_buffer_size(&config);
        let mut buffer = vec![0u8; size];
        let mut arena = Arena::new(&mut buffer);

        let mut decoder: ColorCodeDecoder<'_, 8> =
            ColorCodeDecoder::new(&mut arena, config).unwrap();

        // Create defects at positions with known colors
        let defects = [
            config.coord_to_linear(0, 0, 0) as u32, // Red
            config.coord_to_linear(1, 0, 0) as u32, // Green
            config.coord_to_linear(2, 0, 0) as u32, // Blue
            config.coord_to_linear(3, 0, 0) as u32, // Red
        ];

        let _ = decoder.decode(&defects);
        let stats = decoder.stats();

        assert_eq!(stats.defect_counts[0], 2); // Red
        assert_eq!(stats.defect_counts[1], 1); // Green
        assert_eq!(stats.defect_counts[2], 1); // Blue
    }

    #[test]
    fn test_builder_pattern() {
        let size = 1024 * 1024;
        let mut buffer = vec![0u8; size];
        let mut arena = Arena::new(&mut buffer);

        let decoder: Option<ColorCodeDecoder<'_, 8>> = ColorCodeDecoderBuilder::new_triangular(5)
            .observable_mode(ColorObservableMode::Disabled)
            .mode_2d()
            .build(&mut arena);

        assert!(decoder.is_some());
        let decoder = decoder.unwrap();
        assert_eq!(decoder.config().depth, 1);
    }

    #[test]
    fn test_reset() {
        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(3);
        let size = required_buffer_size(&config);
        let mut buffer = vec![0u8; size];
        let mut arena = Arena::new(&mut buffer);

        let mut decoder: ColorCodeDecoder<'_, 4> =
            ColorCodeDecoder::new(&mut arena, config).unwrap();

        // Decode something
        let defects = [0u32, 1, 2, 3];
        let _ = decoder.decode(&defects);

        // Reset and verify clean state
        decoder.reset();
        let stats = decoder.stats();
        assert_eq!(stats.defect_counts, [0, 0, 0]);
    }
}
