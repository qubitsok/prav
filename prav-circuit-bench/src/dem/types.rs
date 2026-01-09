//! Types for parsed DEM data.

use prav_core::Detector;

/// A parsed Detector Error Model.
///
/// This is the owned version of `CompiledDem` from prav-core,
/// suitable for use in std environments.
#[derive(Clone, Debug)]
pub struct ParsedDem {
    /// Total number of detectors.
    pub num_detectors: u32,
    /// Number of logical observables.
    pub num_observables: u8,
    /// Detector metadata with coordinates.
    pub detectors: Vec<Detector>,
    /// Error mechanisms with detector IDs and probabilities.
    pub mechanisms: Vec<OwnedErrorMechanism>,
}

impl ParsedDem {
    /// Create a new empty parsed DEM.
    pub fn new() -> Self {
        Self {
            num_detectors: 0,
            num_observables: 0,
            detectors: Vec::new(),
            mechanisms: Vec::new(),
        }
    }

    /// Get number of 64-bit words needed for syndrome storage.
    pub fn syndrome_words(&self) -> usize {
        (self.num_detectors as usize).div_ceil(64)
    }
}

impl Default for ParsedDem {
    fn default() -> Self {
        Self::new()
    }
}

/// An owned error mechanism (heap-allocated detector list).
#[derive(Clone, Debug)]
pub struct OwnedErrorMechanism {
    /// Probability of this error.
    pub probability: f32,
    /// Detector IDs affected by this error.
    pub detectors: Vec<u32>,
    /// Bitmask of affected logical observables.
    pub frame_changes: u8,
}

impl OwnedErrorMechanism {
    /// Create a new error mechanism.
    pub fn new(probability: f32, detectors: Vec<u32>, frame_changes: u8) -> Self {
        Self {
            probability,
            detectors,
            frame_changes,
        }
    }

    /// Check if this is a boundary error (affects only 1 detector).
    pub fn is_boundary(&self) -> bool {
        self.detectors.len() == 1
    }

    /// Check if this is a standard edge error (affects exactly 2 detectors).
    pub fn is_edge(&self) -> bool {
        self.detectors.len() == 2
    }

    /// Check if this is a hyperedge error (affects >2 detectors).
    pub fn is_hyperedge(&self) -> bool {
        self.detectors.len() > 2
    }
}
