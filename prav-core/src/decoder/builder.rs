//! Builder pattern for [`DecodingState`] construction.
//!
//! This module provides a type-safe way to construct decoders without
//! manually calculating the `STRIDE_Y` const generic.
//!
//! # Motivation
//!
//! The `DecodingState` struct requires a `STRIDE_Y` const generic that must
//! equal `max(width, height, depth).next_power_of_two()`. Getting this wrong
//! causes a runtime panic. The builder pattern eliminates this error-prone
//! manual calculation.
//!
//! # Example
//!
//! ```ignore
//! use prav_core::{Arena, DecoderBuilder, SquareGrid, EdgeCorrection, required_buffer_size};
//!
//! let size = required_buffer_size(32, 32, 1);
//! let mut buffer = [0u8; size];
//! let mut arena = Arena::new(&mut buffer);
//!
//! // Builder automatically selects correct STRIDE_Y
//! let mut decoder = DecoderBuilder::<SquareGrid>::new()
//!     .dimensions(32, 32)
//!     .build(&mut arena)
//!     .unwrap();
//!
//! let syndromes = [0u64; 16];
//! decoder.load_dense_syndromes(&syndromes);
//! ```

use crate::arena::Arena;
use crate::decoder::state::DecodingState;
use crate::decoder::types::EdgeCorrection;
use crate::decoder::growth::ClusterGrowth;
use crate::topology::Topology;
use core::marker::PhantomData;

/// Builder for constructing [`DecodingState`] instances.
///
/// The builder pattern eliminates the need to manually calculate `STRIDE_Y`,
/// preventing the common pitfall of mismatched const generics.
///
/// # Type Parameter
///
/// * `T` - The topology type (e.g., [`SquareGrid`](crate::SquareGrid)).
///
/// # Example
///
/// ```ignore
/// use prav_core::{Arena, DecoderBuilder, SquareGrid, required_buffer_size};
///
/// let size = required_buffer_size(64, 64, 1);
/// let mut buffer = vec![0u8; size];
/// let mut arena = Arena::new(&mut buffer);
///
/// let decoder = DecoderBuilder::<SquareGrid>::new()
///     .dimensions(64, 64)
///     .build(&mut arena)
///     .expect("Failed to build decoder");
/// ```
pub struct DecoderBuilder<T: Topology> {
    width: usize,
    height: usize,
    depth: usize,
    _marker: PhantomData<T>,
}

impl<T: Topology> DecoderBuilder<T> {
    /// Creates a new decoder builder with default dimensions.
    ///
    /// You must call [`dimensions`](Self::dimensions) or
    /// [`dimensions_3d`](Self::dimensions_3d) before [`build`](Self::build).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            depth: 1,
            _marker: PhantomData,
        }
    }

    /// Sets the grid dimensions for a 2D code.
    ///
    /// # Arguments
    ///
    /// * `width` - Grid width in nodes.
    /// * `height` - Grid height in nodes.
    #[must_use]
    pub const fn dimensions(mut self, width: usize, height: usize) -> Self {
        self.width = width;
        self.height = height;
        self.depth = 1;
        self
    }

    /// Sets the grid dimensions for a 3D code.
    ///
    /// # Arguments
    ///
    /// * `width` - Grid width in nodes.
    /// * `height` - Grid height in nodes.
    /// * `depth` - Grid depth in nodes.
    #[must_use]
    pub const fn dimensions_3d(mut self, width: usize, height: usize, depth: usize) -> Self {
        self.width = width;
        self.height = height;
        self.depth = depth;
        self
    }

    /// Calculates the required `STRIDE_Y` for the configured dimensions.
    ///
    /// This is the value that would need to be specified as the const generic
    /// when using [`DecodingState`] directly.
    #[must_use]
    pub const fn stride_y(&self) -> usize {
        let is_3d = self.depth > 1;
        let max_dim = const_max(self.width, const_max(self.height, if is_3d { self.depth } else { 1 }));
        max_dim.next_power_of_two()
    }

    /// Builds the decoder with the appropriate `STRIDE_Y`.
    ///
    /// This method uses a dispatch table to select the correct const generic
    /// at runtime, then constructs the decoder.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Dimensions are not set (width or height is 0).
    /// - The grid is too large (max dimension > 512).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let decoder = DecoderBuilder::<SquareGrid>::new()
    ///     .dimensions(32, 32)
    ///     .build(&mut arena)?;
    /// ```
    pub fn build<'a>(self, arena: &mut Arena<'a>) -> Result<DynDecoder<'a, T>, &'static str> {
        if self.width == 0 || self.height == 0 {
            return Err("Dimensions not set: call dimensions() or dimensions_3d() first");
        }

        let stride = self.stride_y();

        match stride {
            1 => Ok(DynDecoder::S1(DecodingState::<T, 1>::new(
                arena, self.width, self.height, self.depth
            ))),
            2 => Ok(DynDecoder::S2(DecodingState::<T, 2>::new(
                arena, self.width, self.height, self.depth
            ))),
            4 => Ok(DynDecoder::S4(DecodingState::<T, 4>::new(
                arena, self.width, self.height, self.depth
            ))),
            8 => Ok(DynDecoder::S8(DecodingState::<T, 8>::new(
                arena, self.width, self.height, self.depth
            ))),
            16 => Ok(DynDecoder::S16(DecodingState::<T, 16>::new(
                arena, self.width, self.height, self.depth
            ))),
            32 => Ok(DynDecoder::S32(DecodingState::<T, 32>::new(
                arena, self.width, self.height, self.depth
            ))),
            64 => Ok(DynDecoder::S64(DecodingState::<T, 64>::new(
                arena, self.width, self.height, self.depth
            ))),
            128 => Ok(DynDecoder::S128(DecodingState::<T, 128>::new(
                arena, self.width, self.height, self.depth
            ))),
            256 => Ok(DynDecoder::S256(DecodingState::<T, 256>::new(
                arena, self.width, self.height, self.depth
            ))),
            512 => Ok(DynDecoder::S512(DecodingState::<T, 512>::new(
                arena, self.width, self.height, self.depth
            ))),
            _ => Err("Grid too large: max dimension exceeds 512"),
        }
    }
}

impl<T: Topology> Default for DecoderBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Const-compatible max function.
const fn const_max(a: usize, b: usize) -> usize {
    if a > b { a } else { b }
}

/// Dynamic decoder wrapper that hides the `STRIDE_Y` const generic.
///
/// This enum provides a unified interface regardless of the underlying
/// stride, at the cost of a small dispatch overhead per method call.
///
/// # Performance Note
///
/// For maximum performance in tight loops, prefer using [`DecodingState`]
/// directly with the correct const generic. The dynamic dispatch overhead
/// is typically negligible for most use cases.
///
/// # Example
///
/// ```ignore
/// let mut decoder = DecoderBuilder::<SquareGrid>::new()
///     .dimensions(32, 32)
///     .build(&mut arena)?;
///
/// // Use unified interface regardless of stride
/// decoder.load_dense_syndromes(&syndromes);
/// decoder.grow_clusters();
/// let count = decoder.peel_forest(&mut corrections);
/// decoder.reset_for_next_cycle();
/// ```
pub enum DynDecoder<'a, T: Topology> {
    /// Stride 1 (1x1 grids).
    S1(DecodingState<'a, T, 1>),
    /// Stride 2 (up to 2x2 grids).
    S2(DecodingState<'a, T, 2>),
    /// Stride 4 (up to 4x4 grids).
    S4(DecodingState<'a, T, 4>),
    /// Stride 8 (up to 8x8 grids).
    S8(DecodingState<'a, T, 8>),
    /// Stride 16 (up to 16x16 grids).
    S16(DecodingState<'a, T, 16>),
    /// Stride 32 (up to 32x32 grids).
    S32(DecodingState<'a, T, 32>),
    /// Stride 64 (up to 64x64 grids).
    S64(DecodingState<'a, T, 64>),
    /// Stride 128 (up to 128x128 grids).
    S128(DecodingState<'a, T, 128>),
    /// Stride 256 (up to 256x256 grids).
    S256(DecodingState<'a, T, 256>),
    /// Stride 512 (up to 512x512 grids).
    S512(DecodingState<'a, T, 512>),
}

/// Helper macro to dispatch method calls to the inner decoder.
macro_rules! dispatch {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            DynDecoder::S1(d) => d.$method($($arg),*),
            DynDecoder::S2(d) => d.$method($($arg),*),
            DynDecoder::S4(d) => d.$method($($arg),*),
            DynDecoder::S8(d) => d.$method($($arg),*),
            DynDecoder::S16(d) => d.$method($($arg),*),
            DynDecoder::S32(d) => d.$method($($arg),*),
            DynDecoder::S64(d) => d.$method($($arg),*),
            DynDecoder::S128(d) => d.$method($($arg),*),
            DynDecoder::S256(d) => d.$method($($arg),*),
            DynDecoder::S512(d) => d.$method($($arg),*),
        }
    };
}

impl<'a, T: Topology> DynDecoder<'a, T> {
    /// Loads syndrome measurements from a dense bitarray.
    ///
    /// Each `u64` in the slice represents 64 consecutive nodes, where bit `i`
    /// being set indicates a syndrome at node `(block_index * 64 + i)`.
    ///
    /// # Arguments
    ///
    /// * `syndromes` - Dense syndrome bitarray with one `u64` per 64-node block.
    #[inline]
    pub fn load_dense_syndromes(&mut self, syndromes: &[u64]) {
        dispatch!(self, load_dense_syndromes, syndromes);
    }

    /// Performs full cluster growth until convergence.
    ///
    /// This iteratively expands syndrome clusters until all defects are paired
    /// or reach boundaries.
    #[inline]
    pub fn grow_clusters(&mut self) {
        dispatch!(self, grow_clusters);
    }

    /// Performs a single growth iteration.
    ///
    /// Returns `true` if more iterations are needed, `false` if converged.
    #[inline]
    pub fn grow_iteration(&mut self) -> bool {
        dispatch!(self, grow_iteration)
    }

    /// Extracts corrections by peeling the cluster forest.
    ///
    /// This traces paths from defects and accumulates edge corrections.
    ///
    /// # Arguments
    ///
    /// * `corrections` - Output buffer for edge corrections.
    ///
    /// # Returns
    ///
    /// The number of corrections written to the buffer.
    #[inline]
    pub fn peel_forest(&mut self, corrections: &mut [EdgeCorrection]) -> usize {
        dispatch!(self, peel_forest, corrections)
    }

    /// Performs full decode cycle (grow + peel).
    ///
    /// This is equivalent to calling [`grow_clusters`](Self::grow_clusters)
    /// followed by [`peel_forest`](Self::peel_forest).
    ///
    /// # Arguments
    ///
    /// * `corrections` - Output buffer for edge corrections.
    ///
    /// # Returns
    ///
    /// The number of corrections written to the buffer.
    #[inline]
    pub fn decode(&mut self, corrections: &mut [EdgeCorrection]) -> usize {
        dispatch!(self, decode, corrections)
    }

    /// Resets state for the next decoding cycle (sparse reset).
    ///
    /// This efficiently resets only the blocks that were modified during
    /// the previous decoding cycle.
    #[inline]
    pub fn reset_for_next_cycle(&mut self) {
        dispatch!(self, sparse_reset);
    }

    /// Fully resets all decoder state.
    ///
    /// This performs a complete reset of all internal data structures.
    /// For repeated decoding, prefer [`reset_for_next_cycle`](Self::reset_for_next_cycle).
    #[inline]
    pub fn full_reset(&mut self) {
        dispatch!(self, initialize_internal);
    }

    /// Returns the grid width.
    #[inline]
    #[must_use]
    pub fn width(&self) -> usize {
        match self {
            DynDecoder::S1(d) => d.width,
            DynDecoder::S2(d) => d.width,
            DynDecoder::S4(d) => d.width,
            DynDecoder::S8(d) => d.width,
            DynDecoder::S16(d) => d.width,
            DynDecoder::S32(d) => d.width,
            DynDecoder::S64(d) => d.width,
            DynDecoder::S128(d) => d.width,
            DynDecoder::S256(d) => d.width,
            DynDecoder::S512(d) => d.width,
        }
    }

    /// Returns the grid height.
    #[inline]
    #[must_use]
    pub fn height(&self) -> usize {
        match self {
            DynDecoder::S1(d) => d.height,
            DynDecoder::S2(d) => d.height,
            DynDecoder::S4(d) => d.height,
            DynDecoder::S8(d) => d.height,
            DynDecoder::S16(d) => d.height,
            DynDecoder::S32(d) => d.height,
            DynDecoder::S64(d) => d.height,
            DynDecoder::S128(d) => d.height,
            DynDecoder::S256(d) => d.height,
            DynDecoder::S512(d) => d.height,
        }
    }

    /// Returns the stride Y value.
    #[inline]
    #[must_use]
    pub fn stride_y(&self) -> usize {
        match self {
            DynDecoder::S1(d) => d.stride_y,
            DynDecoder::S2(d) => d.stride_y,
            DynDecoder::S4(d) => d.stride_y,
            DynDecoder::S8(d) => d.stride_y,
            DynDecoder::S16(d) => d.stride_y,
            DynDecoder::S32(d) => d.stride_y,
            DynDecoder::S64(d) => d.stride_y,
            DynDecoder::S128(d) => d.stride_y,
            DynDecoder::S256(d) => d.stride_y,
            DynDecoder::S512(d) => d.stride_y,
        }
    }
}
