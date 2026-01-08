//! Topology type wrapper for Python.

use core::fmt;
use pyo3::prelude::*;

/// Supported grid topologies for QEC decoding.
///
/// Available topologies:
/// - `Square`: 4-neighbor square lattice (surface codes)
/// - `Triangular`: 6-neighbor triangular lattice (color codes)
/// - `Honeycomb`: 3-neighbor honeycomb lattice
/// - `Grid3D`: 6-neighbor 3D cubic lattice
#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TopologyType {
    /// 4-neighbor square lattice (surface codes).
    Square,
    /// 6-neighbor triangular lattice (color codes).
    Triangular,
    /// 3-neighbor honeycomb lattice.
    Honeycomb,
    /// 6-neighbor 3D cubic lattice.
    Grid3D,
}

#[pymethods]
impl TopologyType {
    /// Create a topology type from a string name.
    ///
    /// Parameters
    /// ----------
    /// name : str
    ///     Topology name: 'square', 'triangular', 'honeycomb', or '3d'/'grid3d'.
    ///
    /// Returns
    /// -------
    /// TopologyType
    ///     The corresponding topology type.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the topology name is not recognized.
    #[new]
    #[pyo3(signature = (name="square"))]
    pub fn new(name: &str) -> PyResult<Self> {
        match name.to_lowercase().as_str() {
            "square" => Ok(TopologyType::Square),
            "triangular" => Ok(TopologyType::Triangular),
            "honeycomb" => Ok(TopologyType::Honeycomb),
            "3d" | "grid3d" => Ok(TopologyType::Grid3D),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown topology: '{}'. Use 'square', 'triangular', 'honeycomb', or '3d'",
                name
            ))),
        }
    }

    fn __repr__(&self) -> String {
        match self {
            TopologyType::Square => "TopologyType.Square".to_string(),
            TopologyType::Triangular => "TopologyType.Triangular".to_string(),
            TopologyType::Honeycomb => "TopologyType.Honeycomb".to_string(),
            TopologyType::Grid3D => "TopologyType.Grid3D".to_string(),
        }
    }

    fn __str__(&self) -> String {
        self.to_string()
    }
}

impl fmt::Display for TopologyType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TopologyType::Square => write!(f, "square"),
            TopologyType::Triangular => write!(f, "triangular"),
            TopologyType::Honeycomb => write!(f, "honeycomb"),
            TopologyType::Grid3D => write!(f, "3d"),
        }
    }
}
