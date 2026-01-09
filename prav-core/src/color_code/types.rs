//! Core types for color code decoding.
//!
//! Color codes are topological QEC codes defined on 3-colorable lattices.
//! Unlike surface codes where any two defects can be paired, color codes
//! require defects to be matched within their color class.
//!
//! # Color Code Structure
//!
//! In a triangular color code:
//! - **Faces (plaquettes)** are assigned one of three colors: Red, Green, Blue
//! - **Adjacent faces** always have different colors (3-coloring property)
//! - **Single qubit errors** create defects on faces of the **same** color
//! - **Decoding** must respect color class boundaries
//!
//! # Restriction Decoder Approach
//!
//! This implementation uses the restriction decoder method:
//! 1. Project the color code onto three restricted subgraphs (one per color)
//! 2. Run Union-Find on each restricted subgraph independently
//! 3. Lift corrections back to the full color code lattice

/// The three face colors in a triangular color code.
///
/// Each face (stabilizer) in a color code lattice is assigned exactly one
/// of these three colors such that no two adjacent faces share the same color.
///
/// # Color Assignment
///
/// For a triangular lattice using Morton-encoded coordinates:
/// ```text
/// color = (x + y) % 3
/// ```
///
/// This ensures the 3-coloring property is satisfied.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum FaceColor {
    /// Red faces (color index 0).
    Red = 0,
    /// Green faces (color index 1).
    Green = 1,
    /// Blue faces (color index 2).
    Blue = 2,
}

impl FaceColor {
    /// Compute face color from 2D coordinates.
    ///
    /// For triangular lattices, the color is determined by `(x + y) % 3`.
    /// This ensures that adjacent faces (differing by 1 in x or y) have
    /// different colors.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use prav_core::color_code::FaceColor;
    ///
    /// assert_eq!(FaceColor::from_coords(0, 0), FaceColor::Red);
    /// assert_eq!(FaceColor::from_coords(1, 0), FaceColor::Green);
    /// assert_eq!(FaceColor::from_coords(0, 1), FaceColor::Green);
    /// assert_eq!(FaceColor::from_coords(1, 1), FaceColor::Blue);
    /// ```
    #[inline(always)]
    #[must_use]
    pub const fn from_coords(x: u32, y: u32) -> Self {
        match (x + y) % 3 {
            0 => FaceColor::Red,
            1 => FaceColor::Green,
            _ => FaceColor::Blue,
        }
    }

    /// Compute face color from 3D coordinates (space-time).
    ///
    /// The temporal dimension (t) does not affect the color - colors are
    /// consistent across measurement rounds.
    #[inline(always)]
    #[must_use]
    pub const fn from_coords_3d(x: u32, y: u32, _t: u32) -> Self {
        Self::from_coords(x, y)
    }

    /// Get the color index (0, 1, or 2).
    #[inline(always)]
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Get the next color in cyclic order (R -> G -> B -> R).
    #[inline(always)]
    #[must_use]
    pub const fn next(self) -> Self {
        match self {
            FaceColor::Red => FaceColor::Green,
            FaceColor::Green => FaceColor::Blue,
            FaceColor::Blue => FaceColor::Red,
        }
    }

    /// Get the previous color in cyclic order (R -> B -> G -> R).
    #[inline(always)]
    #[must_use]
    pub const fn prev(self) -> Self {
        match self {
            FaceColor::Red => FaceColor::Blue,
            FaceColor::Green => FaceColor::Red,
            FaceColor::Blue => FaceColor::Green,
        }
    }

    /// Returns all three colors as an array.
    #[inline(always)]
    #[must_use]
    pub const fn all() -> [FaceColor; 3] {
        [FaceColor::Red, FaceColor::Green, FaceColor::Blue]
    }

    /// Create from a raw index (0, 1, or 2).
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `idx > 2`.
    #[inline(always)]
    #[must_use]
    pub const fn from_index(idx: usize) -> Self {
        debug_assert!(idx < 3, "FaceColor index must be 0, 1, or 2");
        match idx {
            0 => FaceColor::Red,
            1 => FaceColor::Green,
            _ => FaceColor::Blue,
        }
    }
}

/// Boundary configuration for color codes.
///
/// In triangular color codes, there are three boundary edges, each associated
/// with one color. Defects of a given color can only terminate at the boundary
/// edge of the same color.
///
/// ```text
///        Red boundary
///        ___________
///       /           \
///      /             \
///     /               \
///    /_________________\
///   Green             Blue
/// boundary          boundary
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ColorCodeBoundaryConfig {
    /// Which colors have active boundaries.
    /// If true, defects of that color can terminate at the corresponding boundary.
    pub active: [bool; 3],
}

impl Default for ColorCodeBoundaryConfig {
    fn default() -> Self {
        Self::all_boundaries()
    }
}

impl ColorCodeBoundaryConfig {
    /// All three colored boundaries are active.
    #[must_use]
    pub const fn all_boundaries() -> Self {
        Self {
            active: [true, true, true],
        }
    }

    /// No boundaries (for testing toric-like codes).
    #[must_use]
    pub const fn no_boundaries() -> Self {
        Self {
            active: [false, false, false],
        }
    }

    /// Only the boundary for the given color is active.
    #[must_use]
    pub const fn single_color(color: FaceColor) -> Self {
        let mut active = [false, false, false];
        active[color.index()] = true;
        Self { active }
    }

    /// Check if the boundary for a given color is active.
    #[inline(always)]
    #[must_use]
    pub const fn is_active(&self, color: FaceColor) -> bool {
        self.active[color.index()]
    }
}

/// Result of decoding a color code.
///
/// Contains corrections for each color class and the overall logical frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct ColorCodeResult {
    /// Number of corrections in each color class.
    pub correction_counts: [usize; 3],
    /// Logical observable frame (XOR of all color frames).
    pub logical_frame: u8,
    /// Per-color logical frames.
    pub color_frames: [u8; 3],
}

impl ColorCodeResult {
    /// Total number of corrections across all colors.
    #[inline(always)]
    #[must_use]
    pub const fn total_corrections(&self) -> usize {
        self.correction_counts[0] + self.correction_counts[1] + self.correction_counts[2]
    }

    /// Check if a logical error occurred (non-zero frame).
    #[inline(always)]
    #[must_use]
    pub const fn has_logical_error(&self) -> bool {
        self.logical_frame != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_from_coords_origin() {
        assert_eq!(FaceColor::from_coords(0, 0), FaceColor::Red);
    }

    #[test]
    fn test_color_from_coords_adjacent() {
        // Adjacent faces must have different colors
        let c00 = FaceColor::from_coords(0, 0);
        let c10 = FaceColor::from_coords(1, 0);
        let c01 = FaceColor::from_coords(0, 1);
        let c11 = FaceColor::from_coords(1, 1);

        assert_ne!(c00, c10);
        assert_ne!(c00, c01);
        assert_ne!(c10, c11);
        assert_ne!(c01, c11);
    }

    #[test]
    fn test_color_cyclic() {
        assert_eq!(FaceColor::Red.next(), FaceColor::Green);
        assert_eq!(FaceColor::Green.next(), FaceColor::Blue);
        assert_eq!(FaceColor::Blue.next(), FaceColor::Red);

        assert_eq!(FaceColor::Red.prev(), FaceColor::Blue);
        assert_eq!(FaceColor::Green.prev(), FaceColor::Red);
        assert_eq!(FaceColor::Blue.prev(), FaceColor::Green);
    }

    #[test]
    fn test_3_coloring_property() {
        // Verify that no two adjacent nodes have the same color
        // in a 10x10 grid
        for x in 0u32..10 {
            for y in 0u32..10 {
                let color = FaceColor::from_coords(x, y);

                // Check right neighbor
                if x + 1 < 10 {
                    assert_ne!(color, FaceColor::from_coords(x + 1, y));
                }
                // Check bottom neighbor
                if y + 1 < 10 {
                    assert_ne!(color, FaceColor::from_coords(x, y + 1));
                }
            }
        }
    }

    #[test]
    fn test_color_index_roundtrip() {
        for color in FaceColor::all() {
            let idx = color.index();
            let recovered = FaceColor::from_index(idx);
            assert_eq!(color, recovered);
        }
    }

    #[test]
    fn test_boundary_config_defaults() {
        let all = ColorCodeBoundaryConfig::all_boundaries();
        assert!(all.is_active(FaceColor::Red));
        assert!(all.is_active(FaceColor::Green));
        assert!(all.is_active(FaceColor::Blue));

        let none = ColorCodeBoundaryConfig::no_boundaries();
        assert!(!none.is_active(FaceColor::Red));
        assert!(!none.is_active(FaceColor::Green));
        assert!(!none.is_active(FaceColor::Blue));
    }

    #[test]
    fn test_color_3d_time_invariant() {
        // Color should not change with time
        for t in 0..10 {
            assert_eq!(
                FaceColor::from_coords_3d(5, 3, t),
                FaceColor::from_coords(5, 3)
            );
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify that color assignment is deterministic.
    #[kani::proof]
    #[kani::unwind(1)]
    fn verify_color_assignment_deterministic() {
        let x: u32 = kani::any();
        let y: u32 = kani::any();
        kani::assume(x < 1000 && y < 1000);

        let c1 = FaceColor::from_coords(x, y);
        let c2 = FaceColor::from_coords(x, y);
        assert!(c1 == c2);
    }

    /// Verify the 3-coloring property: adjacent faces have different colors.
    #[kani::proof]
    #[kani::unwind(1)]
    fn verify_3_coloring_cardinal_neighbors() {
        let x: u32 = kani::any();
        let y: u32 = kani::any();
        kani::assume(x > 0 && x < 1000 && y > 0 && y < 1000);

        let color = FaceColor::from_coords(x, y);

        // Cardinal neighbors must have different colors
        assert!(color != FaceColor::from_coords(x - 1, y));
        assert!(color != FaceColor::from_coords(x + 1, y));
        assert!(color != FaceColor::from_coords(x, y - 1));
        assert!(color != FaceColor::from_coords(x, y + 1));
    }

    /// Verify that color index is always 0, 1, or 2.
    #[kani::proof]
    #[kani::unwind(1)]
    fn verify_color_index_range() {
        let x: u32 = kani::any();
        let y: u32 = kani::any();
        kani::assume(x < 10000 && y < 10000);

        let color = FaceColor::from_coords(x, y);
        let idx = color.index();
        assert!(idx < 3);
    }

    /// Verify that cyclic operations are correct.
    #[kani::proof]
    fn verify_color_cyclic_properties() {
        for color in FaceColor::all() {
            // next(next(next(c))) == c
            assert!(color.next().next().next() == color);
            // prev(prev(prev(c))) == c
            assert!(color.prev().prev().prev() == color);
            // next(prev(c)) == c
            assert!(color.prev().next() == color);
        }
    }

    /// Verify that from_index is inverse of index.
    #[kani::proof]
    fn verify_index_roundtrip() {
        let idx: usize = kani::any();
        kani::assume(idx < 3);

        let color = FaceColor::from_index(idx);
        assert!(color.index() == idx);
    }
}
