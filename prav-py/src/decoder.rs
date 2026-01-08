//! Main Decoder class for Python.

use numpy::{PyArray1, PyReadonlyArray1};
use prav_core::{
    Arena, DecoderBuilder, DynDecoder, EdgeCorrection, Grid3D, HoneycombGrid, SquareGrid, Topology,
    TriangularGrid,
};
use pyo3::prelude::*;

use crate::topology::TopologyType;

/// Union Find decoder for quantum error correction.
///
/// Supports 2D surface codes, triangular codes, honeycomb codes, and 3D codes.
///
/// Parameters
/// ----------
/// width : int
///     Grid width in nodes.
/// height : int
///     Grid height in nodes.
/// topology : str, optional
///     Grid topology: 'square' (default), 'triangular', 'honeycomb', or '3d'.
/// depth : int, optional
///     Grid depth for 3D codes (default: 1).
///
/// Examples
/// --------
/// >>> import prav
/// >>> import numpy as np
/// >>> decoder = prav.Decoder(17, 17)
/// >>> syndromes = np.zeros(8, dtype=np.uint64)
/// >>> syndromes[0] = 0b11  # Two adjacent defects
/// >>> corrections = decoder.decode(syndromes)
#[pyclass]
pub struct Decoder {
    inner: DecoderInner,
    width: usize,
    height: usize,
    depth: usize,
    topology: TopologyType,
    corrections_buffer: Vec<EdgeCorrection>,
}

/// Internal enum to hold different topology decoders with their owned buffers.
enum DecoderInner {
    Square(DecoderState<SquareGrid>),
    Triangular(DecoderState<TriangularGrid>),
    Honeycomb(DecoderState<HoneycombGrid>),
    Grid3D(DecoderState<Grid3D>),
}

/// Holds the decoder state with owned memory buffer.
///
/// # Safety
///
/// This struct uses unsafe code to manage the lifetime of the arena and decoder.
/// The buffer is owned by this struct, and the decoder borrows from it.
/// We ensure safety by:
/// 1. Never moving the buffer after decoder creation
/// 2. Dropping the decoder before the buffer (via Option)
/// 3. Not exposing references to internal state
struct DecoderState<T: Topology> {
    /// Owned memory buffer for the arena allocator.
    _buffer: Vec<u8>,
    /// The decoder instance (borrows from buffer via transmuted lifetime).
    decoder: Option<DynDecoder<'static, T>>,
}

impl<T: Topology> DecoderState<T> {
    fn new(width: usize, height: usize, depth: usize) -> Self {
        let buf_size = prav_core::required_buffer_size(width, height, depth);
        let mut buffer = vec![0u8; buf_size];

        // SAFETY: We ensure the buffer outlives the decoder by:
        // 1. Storing buffer in the same struct (not moving it after this)
        // 2. Using Option<> for decoder so it drops first
        // 3. Never exposing the buffer reference externally
        let decoder = unsafe {
            let buffer_ptr = buffer.as_mut_ptr();
            let buffer_len = buffer.len();

            // Create a 'static reference to the buffer.
            // This is safe because:
            // - The buffer lives in this struct and won't be moved
            // - The decoder will be dropped before the buffer
            let buffer_ref: &'static mut [u8] =
                core::slice::from_raw_parts_mut(buffer_ptr, buffer_len);

            let mut arena = Arena::new(buffer_ref);

            if depth > 1 {
                DecoderBuilder::<T>::new()
                    .dimensions_3d(width, height, depth)
                    .build(&mut arena)
                    .expect("Failed to build decoder")
            } else {
                DecoderBuilder::<T>::new()
                    .dimensions(width, height)
                    .build(&mut arena)
                    .expect("Failed to build decoder")
            }
        };

        Self {
            _buffer: buffer,
            decoder: Some(decoder),
        }
    }

    fn decoder_mut(&mut self) -> &mut DynDecoder<'static, T> {
        self.decoder.as_mut().expect("Decoder not initialized")
    }
}

impl<T: Topology> Drop for DecoderState<T> {
    fn drop(&mut self) {
        // Drop decoder first by setting to None.
        // The buffer (_buffer) drops automatically after.
        self.decoder = None;
    }
}

#[pymethods]
impl Decoder {
    /// Create a new QEC decoder.
    ///
    /// Parameters
    /// ----------
    /// width : int
    ///     Grid width in nodes. Must be greater than 0.
    /// height : int
    ///     Grid height in nodes. Must be greater than 0.
    /// topology : str, optional
    ///     Grid topology: 'square' (default), 'triangular', 'honeycomb', or '3d'.
    /// depth : int, optional
    ///     Grid depth for 3D codes (default: 1). Must be greater than 0.
    ///
    /// Returns
    /// -------
    /// Decoder
    ///     A new decoder instance.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If width, height, or depth is 0, or if topology is invalid.
    #[new]
    #[pyo3(signature = (width, height, topology="square", depth=1))]
    fn new(width: usize, height: usize, topology: &str, depth: usize) -> PyResult<Self> {
        // Input validation
        if width == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "width must be greater than 0",
            ));
        }
        if height == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "height must be greater than 0",
            ));
        }
        if depth == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "depth must be greater than 0",
            ));
        }

        let topo = TopologyType::new(topology)?;

        let inner = match topo {
            TopologyType::Square => DecoderInner::Square(DecoderState::new(width, height, depth)),
            TopologyType::Triangular => {
                DecoderInner::Triangular(DecoderState::new(width, height, depth))
            }
            TopologyType::Honeycomb => {
                DecoderInner::Honeycomb(DecoderState::new(width, height, depth))
            }
            TopologyType::Grid3D => DecoderInner::Grid3D(DecoderState::new(width, height, depth)),
        };

        // Preallocate correction buffer (generous size for worst case).
        let max_corrections = width * height * depth * 2;
        let corrections_buffer = vec![EdgeCorrection { u: 0, v: 0 }; max_corrections];

        Ok(Self {
            inner,
            width,
            height,
            depth,
            topology: topo,
            corrections_buffer,
        })
    }

    /// Decode syndromes and return edge corrections.
    ///
    /// Parameters
    /// ----------
    /// syndromes : np.ndarray[np.uint64]
    ///     Dense bitpacked syndrome array. Each u64 represents 64 nodes.
    ///     Bit i set means node (block_index * 64 + i) has a syndrome.
    ///
    /// Returns
    /// -------
    /// np.ndarray[np.uint32]
    ///     Correction edges as flat array [u0, v0, u1, v1, ...].
    ///     v=0xFFFFFFFF indicates a boundary correction.
    fn decode<'py>(
        &mut self,
        py: Python<'py>,
        syndromes: PyReadonlyArray1<'py, u64>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let syndrome_slice = syndromes.as_slice()?;

        // Dispatch to appropriate topology decoder.
        let num_corrections = match &mut self.inner {
            DecoderInner::Square(state) => {
                let decoder = state.decoder_mut();
                decoder.load_dense_syndromes(syndrome_slice);
                let count = decoder.decode(&mut self.corrections_buffer);
                decoder.reset_for_next_cycle();
                count
            }
            DecoderInner::Triangular(state) => {
                let decoder = state.decoder_mut();
                decoder.load_dense_syndromes(syndrome_slice);
                let count = decoder.decode(&mut self.corrections_buffer);
                decoder.reset_for_next_cycle();
                count
            }
            DecoderInner::Honeycomb(state) => {
                let decoder = state.decoder_mut();
                decoder.load_dense_syndromes(syndrome_slice);
                let count = decoder.decode(&mut self.corrections_buffer);
                decoder.reset_for_next_cycle();
                count
            }
            DecoderInner::Grid3D(state) => {
                let decoder = state.decoder_mut();
                decoder.load_dense_syndromes(syndrome_slice);
                let count = decoder.decode(&mut self.corrections_buffer);
                decoder.reset_for_next_cycle();
                count
            }
        };

        // Convert corrections to flat numpy array [u0, v0, u1, v1, ...].
        let mut result = Vec::with_capacity(num_corrections * 2);
        for correction in self.corrections_buffer.iter().take(num_corrections) {
            result.push(correction.u);
            result.push(correction.v);
        }

        Ok(PyArray1::from_vec(py, result))
    }

    /// Reset decoder state for next decoding cycle.
    ///
    /// This is called automatically after decode(), but can be called
    /// manually if needed.
    fn reset(&mut self) {
        match &mut self.inner {
            DecoderInner::Square(state) => state.decoder_mut().reset_for_next_cycle(),
            DecoderInner::Triangular(state) => state.decoder_mut().reset_for_next_cycle(),
            DecoderInner::Honeycomb(state) => state.decoder_mut().reset_for_next_cycle(),
            DecoderInner::Grid3D(state) => state.decoder_mut().reset_for_next_cycle(),
        }
    }

    /// Get decoder width.
    #[getter]
    fn width(&self) -> usize {
        self.width
    }

    /// Get decoder height.
    #[getter]
    fn height(&self) -> usize {
        self.height
    }

    /// Get decoder depth (1 for 2D codes).
    #[getter]
    fn depth(&self) -> usize {
        self.depth
    }

    /// Get topology type.
    #[getter]
    fn topology(&self) -> TopologyType {
        self.topology
    }

    fn __repr__(&self) -> String {
        format!(
            "Decoder(width={}, height={}, topology={}, depth={})",
            self.width, self.height, self.topology, self.depth
        )
    }
}
