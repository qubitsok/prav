//! Detector Error Model (DEM) types for circuit-level QEC.
//!
//! This module defines `no_std` compatible types for representing Stim's
//! Detector Error Model format. These types describe:
//!
//! - **Detectors**: Parity checks with 3D coordinates (x, y, time)
//! - **Error mechanisms**: Probabilistic errors affecting detectors
//! - **Logical observables**: Frame changes tracked for logical error rates
//!
//! # Stim DEM Format
//!
//! A DEM file contains lines like:
//! ```text
//! detector(1.5, 2.5, 0) D0
//! detector(2.5, 2.5, 0) D1
//! error(0.001) D0 D1
//! error(0.002) D0 ^ L0
//! ```
//!
//! Where:
//! - `detector(x, y, t) D<id>` declares a detector with coordinates
//! - `error(p) D<ids>... [^ L<ids>...]` declares an error mechanism
//! - `^` separates detector targets from logical observable targets
//!
//! # Memory Layout
//!
//! All types are designed for arena allocation with borrowed slices:
//!
//! ```ignore
//! let mut arena = Arena::new(&mut buffer);
//! let detectors: &[Detector] = arena.alloc_slice(num_detectors);
//! let mechanisms: &[ErrorMechanism] = arena.alloc_slice(num_mechanisms);
//! ```

/// A detector (parity check) with optional 3D coordinates.
///
/// In circuit-level QEC, detectors are syndrome changes between consecutive
/// measurement rounds. The coordinates (x, y, t) describe their position
/// in the space-time decoding graph.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Detector {
    /// Detector ID (0-indexed, matches Stim's D<id>).
    pub id: u32,
    /// X coordinate in the detector grid.
    pub x: f32,
    /// Y coordinate in the detector grid.
    pub y: f32,
    /// Time coordinate (measurement round).
    pub t: f32,
}

impl Detector {
    /// Create a new detector with given ID and coordinates.
    #[must_use]
    pub const fn new(id: u32, x: f32, y: f32, t: f32) -> Self {
        Self { id, x, y, t }
    }

    /// Create a detector with only an ID (no coordinates).
    #[must_use]
    pub const fn with_id(id: u32) -> Self {
        Self {
            id,
            x: 0.0,
            y: 0.0,
            t: 0.0,
        }
    }

    /// Get integer coordinates by rounding.
    ///
    /// Uses simple rounding: `(x + 0.5) as usize` for non-negative values.
    #[must_use]
    pub fn int_coords(&self) -> (usize, usize, usize) {
        (
            round_to_usize(self.x),
            round_to_usize(self.y),
            round_to_usize(self.t),
        )
    }
}

/// A logical observable (frame change tracker).
///
/// When an error occurs, it may flip one or more logical observables.
/// Tracking these flips is essential for computing logical error rates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LogicalObservable {
    /// Observable ID (0-indexed, matches Stim's L<id>).
    pub id: u8,
}

impl LogicalObservable {
    /// Create a new logical observable.
    #[must_use]
    pub const fn new(id: u8) -> Self {
        Self { id }
    }

    /// Convert to bitmask position (1 << id).
    #[must_use]
    pub const fn to_mask(&self) -> u8 {
        1u8 << self.id
    }
}

/// Target of an error mechanism (detector or logical observable).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ErrorTarget {
    /// A detector that gets flipped by this error.
    Detector(u32),
    /// A logical observable that gets flipped by this error.
    Observable(u8),
}

impl ErrorTarget {
    /// Check if this target is a detector.
    #[must_use]
    pub const fn is_detector(&self) -> bool {
        matches!(self, ErrorTarget::Detector(_))
    }

    /// Check if this target is a logical observable.
    #[must_use]
    pub const fn is_observable(&self) -> bool {
        matches!(self, ErrorTarget::Observable(_))
    }

    /// Get detector ID if this is a detector target.
    #[must_use]
    pub const fn detector_id(&self) -> Option<u32> {
        match self {
            ErrorTarget::Detector(id) => Some(*id),
            ErrorTarget::Observable(_) => None,
        }
    }

    /// Get observable ID if this is an observable target.
    #[must_use]
    pub const fn observable_id(&self) -> Option<u8> {
        match self {
            ErrorTarget::Detector(_) => None,
            ErrorTarget::Observable(id) => Some(*id),
        }
    }
}

/// An error mechanism from the DEM.
///
/// Each mechanism has a probability and a set of targets (detectors and
/// logical observables) that get flipped when the error occurs.
///
/// # Hyperedges
///
/// In graph-theoretic terms, an error mechanism is a hyperedge:
/// - Weight: `-log(p / (1-p))` for matching
/// - Endpoints: The affected detectors
/// - Frame change: The affected logical observables
#[derive(Clone, Copy, Debug)]
pub struct ErrorMechanism<'a> {
    /// Probability of this error occurring.
    pub probability: f32,
    /// Detector IDs affected by this error.
    ///
    /// When this error occurs, all listed detectors are XOR-flipped.
    /// Most errors affect exactly 2 detectors (edges in the matching graph).
    pub detectors: &'a [u32],
    /// Bitmask of affected logical observables.
    ///
    /// Bit `i` is set if observable `L<i>` is flipped by this error.
    /// Most surface codes have only 1 logical qubit (bits 0 for X, 1 for Z).
    pub frame_changes: u8,
}

impl<'a> ErrorMechanism<'a> {
    /// Create a new error mechanism.
    #[must_use]
    pub const fn new(probability: f32, detectors: &'a [u32], frame_changes: u8) -> Self {
        Self {
            probability,
            detectors,
            frame_changes,
        }
    }

    /// Check if this error affects any logical observables.
    #[must_use]
    pub const fn has_frame_changes(&self) -> bool {
        self.frame_changes != 0
    }

    /// Get the number of detectors affected by this error.
    #[must_use]
    pub const fn num_detectors(&self) -> usize {
        self.detectors.len()
    }

    /// Check if this is a boundary error (affects only 1 detector).
    #[must_use]
    pub const fn is_boundary(&self) -> bool {
        self.detectors.len() == 1
    }

    /// Check if this is a standard edge error (affects exactly 2 detectors).
    #[must_use]
    pub const fn is_edge(&self) -> bool {
        self.detectors.len() == 2
    }

    /// Check if this is a hyperedge error (affects >2 detectors).
    #[must_use]
    pub const fn is_hyperedge(&self) -> bool {
        self.detectors.len() > 2
    }

    /// Compute matching weight as integer: `-1000 * log2(p / (1-p))`.
    ///
    /// Uses integer approximation suitable for no_std. Higher weight means
    /// less likely error. Returns `i32::MAX` for p=0, `i32::MIN` for p=1.
    ///
    /// The factor of 1000 provides ~3 decimal digits of precision.
    #[must_use]
    pub fn matching_weight_int(&self) -> i32 {
        if self.probability <= 0.0 {
            return i32::MAX;
        }
        if self.probability >= 1.0 {
            return i32::MIN;
        }
        // Use log-odds approximation: -log(p/(1-p)) ≈ -log2(p/(1-p)) / log2(e)
        // For small p: ≈ -log2(p)
        // Integer approximation using bit manipulation
        let odds = self.probability / (1.0 - self.probability);
        // Approximate -log2(odds) * 1000
        let bits = odds.to_bits();
        let exponent = ((bits >> 23) & 0xFF) as i32 - 127;
        // Scale by 1000 for precision
        -exponent * 1000
    }
}

/// A compiled Detector Error Model ready for syndrome generation.
///
/// This structure holds all information needed to:
/// 1. Sample syndromes according to the DEM
/// 2. Map detector IDs to grid positions
/// 3. Track logical frame changes for verification
#[derive(Clone, Debug)]
pub struct CompiledDem<'a> {
    /// Total number of detectors.
    pub num_detectors: u32,
    /// Number of logical observables (usually 1-2 for surface codes).
    pub num_observables: u8,
    /// Detector metadata (coordinates, etc.).
    pub detectors: &'a [Detector],
    /// Error mechanisms (probabilities and targets).
    pub mechanisms: &'a [ErrorMechanism<'a>],
}

impl<'a> CompiledDem<'a> {
    /// Create a new compiled DEM.
    #[must_use]
    pub const fn new(
        num_detectors: u32,
        num_observables: u8,
        detectors: &'a [Detector],
        mechanisms: &'a [ErrorMechanism<'a>],
    ) -> Self {
        Self {
            num_detectors,
            num_observables,
            detectors,
            mechanisms,
        }
    }

    /// Get detector by ID.
    #[must_use]
    pub fn get_detector(&self, id: u32) -> Option<&Detector> {
        self.detectors.iter().find(|d| d.id == id)
    }

    /// Get number of 64-bit words needed for syndrome storage.
    #[must_use]
    pub const fn syndrome_words(&self) -> usize {
        (self.num_detectors as usize).div_ceil(64)
    }

    /// Check if the DEM has any hyperedges (errors affecting >2 detectors).
    #[must_use]
    pub fn has_hyperedges(&self) -> bool {
        self.mechanisms.iter().any(|m| m.is_hyperedge())
    }

    /// Count boundary errors (errors affecting only 1 detector).
    #[must_use]
    pub fn count_boundary_errors(&self) -> usize {
        self.mechanisms.iter().filter(|m| m.is_boundary()).count()
    }

    /// Count standard edge errors (errors affecting exactly 2 detectors).
    #[must_use]
    pub fn count_edge_errors(&self) -> usize {
        self.mechanisms.iter().filter(|m| m.is_edge()).count()
    }
}

/// Round a non-negative f32 to usize.
///
/// Uses `(x + 0.5) as usize` which works correctly for non-negative values.
/// For negative values, returns 0.
#[inline]
fn round_to_usize(x: f32) -> usize {
    if x < 0.0 { 0 } else { (x + 0.5) as usize }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detector_creation() {
        let det = Detector::new(0, 1.5, 2.5, 0.0);
        assert_eq!(det.id, 0);
        assert_eq!(det.x, 1.5);
        assert_eq!(det.y, 2.5);
        assert_eq!(det.t, 0.0);

        let (x, y, t) = det.int_coords();
        assert_eq!((x, y, t), (2, 3, 0)); // rounded
    }

    #[test]
    fn test_detector_with_id() {
        let det = Detector::with_id(42);
        assert_eq!(det.id, 42);
        assert_eq!(det.x, 0.0);
        assert_eq!(det.y, 0.0);
        assert_eq!(det.t, 0.0);
    }

    #[test]
    fn test_logical_observable() {
        let obs = LogicalObservable::new(0);
        assert_eq!(obs.to_mask(), 0b0001);

        let obs1 = LogicalObservable::new(1);
        assert_eq!(obs1.to_mask(), 0b0010);

        let obs3 = LogicalObservable::new(3);
        assert_eq!(obs3.to_mask(), 0b1000);
    }

    #[test]
    fn test_error_target() {
        let det_target = ErrorTarget::Detector(5);
        assert!(det_target.is_detector());
        assert!(!det_target.is_observable());
        assert_eq!(det_target.detector_id(), Some(5));
        assert_eq!(det_target.observable_id(), None);

        let obs_target = ErrorTarget::Observable(1);
        assert!(!obs_target.is_detector());
        assert!(obs_target.is_observable());
        assert_eq!(obs_target.detector_id(), None);
        assert_eq!(obs_target.observable_id(), Some(1));
    }

    #[test]
    fn test_error_mechanism_types() {
        let boundary = ErrorMechanism::new(0.01, &[0], 0b01);
        assert!(boundary.is_boundary());
        assert!(!boundary.is_edge());
        assert!(!boundary.is_hyperedge());
        assert!(boundary.has_frame_changes());

        let edge = ErrorMechanism::new(0.001, &[0, 1], 0);
        assert!(!edge.is_boundary());
        assert!(edge.is_edge());
        assert!(!edge.is_hyperedge());
        assert!(!edge.has_frame_changes());

        let hyper = ErrorMechanism::new(0.0001, &[0, 1, 2], 0);
        assert!(!hyper.is_boundary());
        assert!(!hyper.is_edge());
        assert!(hyper.is_hyperedge());
    }

    #[test]
    fn test_matching_weight_int() {
        // p=0.5 should give weight ~0 (odds ratio is 1)
        let m1 = ErrorMechanism::new(0.5, &[0, 1], 0);
        assert_eq!(m1.matching_weight_int(), 0);

        // Low probability should give positive weight
        let m2 = ErrorMechanism::new(0.001, &[0, 1], 0);
        assert!(m2.matching_weight_int() > 0);

        // High probability should give negative weight
        let m3 = ErrorMechanism::new(0.999, &[0, 1], 0);
        assert!(m3.matching_weight_int() < 0);
    }

    #[test]
    fn test_compiled_dem() {
        let detectors = [
            Detector::new(0, 0.0, 0.0, 0.0),
            Detector::new(1, 1.0, 0.0, 0.0),
        ];
        let det_ids: &[u32] = &[0, 1];
        let mechanisms = [ErrorMechanism::new(0.001, det_ids, 0)];

        let dem = CompiledDem::new(2, 1, &detectors, &mechanisms);

        assert_eq!(dem.num_detectors, 2);
        assert_eq!(dem.num_observables, 1);
        assert_eq!(dem.syndrome_words(), 1);
        assert!(!dem.has_hyperedges());
        assert_eq!(dem.count_boundary_errors(), 0);
        assert_eq!(dem.count_edge_errors(), 1);
    }

    #[test]
    fn test_syndrome_words() {
        let dem = CompiledDem::new(64, 1, &[], &[]);
        assert_eq!(dem.syndrome_words(), 1);

        let dem = CompiledDem::new(65, 1, &[], &[]);
        assert_eq!(dem.syndrome_words(), 2);

        let dem = CompiledDem::new(128, 1, &[], &[]);
        assert_eq!(dem.syndrome_words(), 2);

        let dem = CompiledDem::new(129, 1, &[], &[]);
        assert_eq!(dem.syndrome_words(), 3);
    }
}

// ============================================================================
// Kani Formal Verification Proofs
// ============================================================================

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify logical observable mask is a power of two.
    #[kani::proof]
    fn verify_observable_mask_power_of_two() {
        let id: u8 = kani::any();
        kani::assume(id < 8); // Only 8 bits in u8 mask

        let obs = LogicalObservable::new(id);
        let mask = obs.to_mask();

        kani::assert(mask.is_power_of_two(), "mask must be power of two");
        kani::assert(mask == (1u8 << id), "mask must equal 1 << id");
    }

    /// Verify syndrome_words calculation is correct.
    #[kani::proof]
    fn verify_syndrome_words() {
        let num_detectors: u32 = kani::any();
        kani::assume(num_detectors <= 10000);

        let dem = CompiledDem::new(num_detectors, 1, &[], &[]);
        let words = dem.syndrome_words();

        // Should hold all detector bits
        kani::assert(
            words * 64 >= num_detectors as usize,
            "words must hold all bits",
        );

        // Should not overallocate by more than 63 bits
        if num_detectors > 0 {
            kani::assert(
                (words - 1) * 64 < num_detectors as usize,
                "words should not overallocate",
            );
        }
    }

    /// Verify ErrorTarget accessors are consistent.
    #[kani::proof]
    fn verify_error_target_consistency() {
        let det_id: u32 = kani::any();
        let obs_id: u8 = kani::any();

        let det = ErrorTarget::Detector(det_id);
        kani::assert(det.is_detector(), "Detector target must be detector");
        kani::assert(
            !det.is_observable(),
            "Detector target must not be observable",
        );
        kani::assert(det.detector_id() == Some(det_id), "detector_id must match");
        kani::assert(det.observable_id().is_none(), "observable_id must be None");

        let obs = ErrorTarget::Observable(obs_id);
        kani::assert(!obs.is_detector(), "Observable target must not be detector");
        kani::assert(obs.is_observable(), "Observable target must be observable");
        kani::assert(obs.detector_id().is_none(), "detector_id must be None");
        kani::assert(
            obs.observable_id() == Some(obs_id),
            "observable_id must match",
        );
    }
}
