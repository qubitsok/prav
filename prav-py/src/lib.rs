//! Python bindings for prav-core QEC decoder.
//!
//! This module provides PyO3 bindings for the high-performance Union Find
//! decoder implemented in `prav-core`.

use pyo3::prelude::*;

mod decoder;
mod topology;

use decoder::Decoder;
use topology::TopologyType;

/// High-performance Union Find decoder for quantum error correction.
///
/// This module provides Python bindings for the prav-core QEC decoder,
/// supporting surface codes, color codes, and other topological codes.
#[pymodule]
fn _prav(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Decoder>()?;
    m.add_class::<TopologyType>()?;
    m.add_function(wrap_pyfunction!(required_buffer_size, m)?)?;
    Ok(())
}

/// Calculate required buffer size for a decoder.
///
/// Parameters
/// ----------
/// width : int
///     Grid width in nodes.
/// height : int
///     Grid height in nodes.
/// depth : int, optional
///     Grid depth for 3D codes (default: 1).
///
/// Returns
/// -------
/// int
///     Required buffer size in bytes.
#[pyfunction]
#[pyo3(signature = (width, height, depth=1))]
fn required_buffer_size(width: usize, height: usize, depth: usize) -> usize {
    prav_core::required_buffer_size(width, height, depth)
}
