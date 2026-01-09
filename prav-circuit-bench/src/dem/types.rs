//! # Types for Parsed Detector Error Model (DEM) Data
//!
//! This module defines the data structures used to represent a parsed DEM file.
//!
//! ## What is a DEM?
//!
//! A **Detector Error Model** (DEM) is a description of all the ways errors can
//! occur in a quantum error correction circuit. It comes from Stim, a fast
//! quantum circuit simulator.
//!
//! The DEM contains:
//! - **Detectors**: Measurement locations in space-time. Each detector has an ID
//!   and (x, y, t) coordinates.
//! - **Error mechanisms**: Ways errors can occur. Each mechanism has a probability
//!   and lists which detectors it affects.
//! - **Logical observables**: The encoded quantum information we're protecting.
//!   Some errors can flip logical observables without triggering any detectors.
//!
//! ## DEM File Format
//!
//! ```text
//! # Declare detectors with coordinates
//! detector(0, 0, 0) D0
//! detector(1, 0, 0) D1
//! detector(0, 1, 0) D2
//!
//! # Declare error mechanisms
//! error(0.001) D0 D1           # Edge error: flips D0 and D1
//! error(0.001) D0              # Boundary error: flips only D0
//! error(0.002) D0 ^ L0         # Boundary with logical: flips D0 and L0
//! ```
//!
//! ## Key Types
//!
//! - [`ParsedDem`]: The complete parsed model with detectors and error mechanisms
//! - [`OwnedErrorMechanism`]: A single error mechanism with probability and targets

use prav_core::Detector;

/// A parsed Detector Error Model (DEM).
///
/// This structure holds all the information extracted from a Stim DEM file:
///
/// - **Detectors**: Where measurements happen in space-time
/// - **Error mechanisms**: What errors can occur and their probabilities
/// - **Counts**: How many detectors and logical observables exist
///
/// # Usage
///
/// ```ignore
/// // Parse a DEM file
/// let content = std::fs::read_to_string("model.dem")?;
/// let dem = parse_dem(&content)?;
///
/// // Sample syndromes from the DEM
/// let mut sampler = CircuitSampler::new(&dem, 42);
/// let (syndrome, logical) = sampler.sample();
/// ```
///
/// # Owned vs. Borrowed
///
/// This is the "owned" version suitable for std environments. It uses `Vec`
/// for storage. The core library has a "borrowed" version that uses slices
/// for no_std/embedded environments.
#[derive(Clone, Debug)]
pub struct ParsedDem {
    /// Total number of detectors in the model.
    ///
    /// Detectors are numbered 0 to num_detectors-1. The syndrome is stored
    /// as a bit vector with one bit per detector.
    pub num_detectors: u32,

    /// Number of logical observables (usually 1 or 2).
    ///
    /// For a surface code storing one logical qubit:
    /// - L0 = X logical (bit-flip)
    /// - L1 = Z logical (phase-flip)
    ///
    /// Stored as a bitmask in error mechanisms.
    pub num_observables: u8,

    /// Detector metadata including coordinates.
    ///
    /// Each detector has:
    /// - `id`: Unique identifier (0 to num_detectors-1)
    /// - `x`, `y`: Spatial coordinates on the 2D grid
    /// - `t`: Time coordinate (measurement round)
    ///
    /// Coordinates are used to map between DEM detector IDs and
    /// prav's stride-based linear indices.
    pub detectors: Vec<Detector>,

    /// All error mechanisms in the model.
    ///
    /// Each mechanism describes one way errors can occur:
    /// - Which detectors it triggers (usually 1 or 2)
    /// - What probability it has
    /// - Which logical observables it affects (if any)
    pub mechanisms: Vec<OwnedErrorMechanism>,
}

impl ParsedDem {
    /// Create a new empty DEM.
    ///
    /// Used internally by the parser. Start with an empty model and
    /// add detectors/mechanisms as they are parsed.
    pub fn new() -> Self {
        Self {
            num_detectors: 0,
            num_observables: 0,
            detectors: Vec::new(),
            mechanisms: Vec::new(),
        }
    }

    /// Calculate how many 64-bit words are needed to store a syndrome.
    ///
    /// Syndromes are packed bit vectors with one bit per detector.
    /// This returns ceiling(num_detectors / 64).
    ///
    /// # Example
    ///
    /// - 100 detectors → 2 words (128 bits, 28 unused)
    /// - 64 detectors → 1 word (64 bits, 0 unused)
    pub fn syndrome_words(&self) -> usize {
        (self.num_detectors as usize).div_ceil(64)
    }
}

impl Default for ParsedDem {
    fn default() -> Self {
        Self::new()
    }
}

/// A single error mechanism in the DEM.
///
/// An error mechanism describes one way that errors can occur in the circuit.
/// When this error happens (with the given probability), it flips the listed
/// detectors and potentially affects logical observables.
///
/// ## Error Types
///
/// Error mechanisms come in three flavors based on how many detectors they affect:
///
/// ### Edge Errors (2 detectors)
///
/// The most common type. A data qubit error that flips two adjacent detectors.
///
/// ```text
/// DEM:    error(0.001) D0 D1
///
/// Before: ○──○──○      After:  ●──●──○
///         D0 D1 D2             D0 D1 D2
/// ```
///
/// ### Boundary Errors (1 detector)
///
/// Errors at the edge of the code. Only one detector is triggered because
/// the error is "half outside" the code.
///
/// ```text
/// DEM:    error(0.001) D0 ^ L0
///
/// Before: │──○──○      After:  │──●──○
///         B  D0 D1            B  D0 D1
///
/// This boundary error also flips logical observable L0.
/// ```
///
/// ### Hyperedge Errors (>2 detectors)
///
/// Rare. Correlated errors that affect more than 2 detectors. Most decoders
/// approximate these as multiple independent errors.
///
/// ```text
/// DEM:    error(0.0001) D0 D1 D2
/// ```
#[derive(Clone, Debug)]
pub struct OwnedErrorMechanism {
    /// Probability that this error occurs per shot.
    ///
    /// Typical values: 0.001 (0.1%) to 0.01 (1%) for circuit-level noise.
    pub probability: f32,

    /// List of detector IDs that this error flips.
    ///
    /// - Length 1: Boundary error
    /// - Length 2: Edge error (most common)
    /// - Length >2: Hyperedge error (rare)
    pub detectors: Vec<u32>,

    /// Bitmask of logical observables affected by this error.
    ///
    /// - Bit 0 (0x01): L0 (typically X logical)
    /// - Bit 1 (0x02): L1 (typically Z logical)
    /// - Bit 2 (0x04): L2, etc.
    ///
    /// Most errors have frame_changes = 0 (no logical effect).
    /// Boundary errors often have frame_changes != 0.
    pub frame_changes: u8,
}

impl OwnedErrorMechanism {
    /// Create a new error mechanism.
    ///
    /// # Parameters
    ///
    /// - `probability`: Chance this error occurs (0.0 to 1.0)
    /// - `detectors`: Which detectors this error triggers
    /// - `frame_changes`: Bitmask of logical observables affected
    pub fn new(probability: f32, detectors: Vec<u32>, frame_changes: u8) -> Self {
        Self {
            probability,
            detectors,
            frame_changes,
        }
    }

    /// Check if this is a boundary error.
    ///
    /// Boundary errors affect only one detector. They occur when an error
    /// happens at the edge of the code, where there's no second detector
    /// to pair with.
    ///
    /// Boundary errors are important for logical error tracking because
    /// they often affect logical observables.
    pub fn is_boundary(&self) -> bool {
        self.detectors.len() == 1
    }

    /// Check if this is a standard edge error.
    ///
    /// Edge errors affect exactly two detectors. They're the most common
    /// error type, representing a single data qubit error that triggers
    /// the two adjacent stabilizer measurements.
    pub fn is_edge(&self) -> bool {
        self.detectors.len() == 2
    }

    /// Check if this is a hyperedge error.
    ///
    /// Hyperedge errors affect more than two detectors. They're rare and
    /// represent correlated multi-qubit errors. Most decoders (including
    /// prav) handle these by decomposing them into simpler errors.
    pub fn is_hyperedge(&self) -> bool {
        self.detectors.len() > 2
    }
}
