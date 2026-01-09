//! Restriction maps for color code decoding.
//!
//! The restriction decoder projects the full color code lattice onto three
//! restricted subgraphs (one per color class). This module provides mappings
//! between full grid indices and restricted grid indices.
//!
//! # Restriction Mapping
//!
//! Given a full grid with (x, y, t) coordinates:
//! - Detectors are colored by `(x + y) % 3`
//! - Each color class is compacted into a restricted grid
//! - The restricted grid has approximately 1/3 the detectors
//!
//! # Index Mapping
//!
//! ```text
//! Full Grid (6x6):           Restricted Grid (Red, 2x6):
//!
//!   R G B R G B              R R
//!   G B R G B R              R R
//!   B R G B R G   ──────►    R R
//!   R G B R G B              R R
//!   G B R G B R              R R
//!   B R G B R G              R R
//!
//! Mapping: full_idx → (color, restricted_idx)
//! Lifting: (color, restricted_idx) → full_idx
//! ```

use crate::color_code::types::FaceColor;
use crate::color_code::grid_3d::ColorCodeGrid3DConfig;

/// Pre-computed restriction maps for efficient index conversion.
///
/// In a `no_std` environment, these maps are stored in arena-allocated slices.
/// The maps provide O(1) lookup for both restriction (full → restricted) and
/// lifting (restricted → full).
#[derive(Debug)]
pub struct RestrictionMaps<'a> {
    /// For each full grid index: which restricted index it maps to.
    /// Value is the restricted index within that color's subgraph.
    /// Use `detector_color_at()` to determine which color subgraph.
    pub full_to_restricted: &'a [u32],

    /// For each color class: mapping from restricted index to full index.
    /// `restricted_to_full[color][restricted_idx] = full_idx`
    pub restricted_to_full: [&'a [u32]; 3],

    /// Number of detectors in each restricted subgraph.
    pub restricted_counts: [usize; 3],

    /// Grid configuration.
    config: ColorCodeGrid3DConfig,
}

impl<'a> RestrictionMaps<'a> {
    /// Build restriction maps for the given configuration.
    ///
    /// # Arguments
    /// * `config` - Grid configuration
    /// * `full_to_restricted` - Pre-allocated slice of size `config.alloc_nodes()`
    /// * `red_to_full` - Pre-allocated slice for red restricted indices
    /// * `green_to_full` - Pre-allocated slice for green restricted indices
    /// * `blue_to_full` - Pre-allocated slice for blue restricted indices
    ///
    /// # Safety
    ///
    /// Caller must ensure slices are large enough:
    /// - `full_to_restricted.len() >= config.alloc_nodes()`
    /// - `*_to_full.len() >= count_by_color()[color_index]`
    pub fn build(
        config: ColorCodeGrid3DConfig,
        full_to_restricted: &'a mut [u32],
        red_to_full: &'a mut [u32],
        green_to_full: &'a mut [u32],
        blue_to_full: &'a mut [u32],
    ) -> Self {
        let mut restricted_counts = [0usize; 3];

        // First pass: count detectors per color and build full_to_restricted
        for t in 0..config.depth {
            for y in 0..config.height {
                for x in 0..config.width {
                    let full_idx = config.coord_to_linear(x, y, t);
                    let color = config.detector_color(x, y);
                    let color_idx = color.index();

                    // This detector's index within its color class
                    let restricted_idx = restricted_counts[color_idx];
                    full_to_restricted[full_idx] = restricted_idx as u32;
                    restricted_counts[color_idx] += 1;
                }
            }
        }

        // Reset counts for second pass
        let final_counts = restricted_counts;
        restricted_counts = [0; 3];

        // Second pass: build restricted_to_full mappings
        for t in 0..config.depth {
            for y in 0..config.height {
                for x in 0..config.width {
                    let full_idx = config.coord_to_linear(x, y, t);
                    let color = config.detector_color(x, y);
                    let color_idx = color.index();
                    let restricted_idx = restricted_counts[color_idx];

                    match color {
                        FaceColor::Red => red_to_full[restricted_idx] = full_idx as u32,
                        FaceColor::Green => green_to_full[restricted_idx] = full_idx as u32,
                        FaceColor::Blue => blue_to_full[restricted_idx] = full_idx as u32,
                    }
                    restricted_counts[color_idx] += 1;
                }
            }
        }

        Self {
            full_to_restricted,
            restricted_to_full: [red_to_full, green_to_full, blue_to_full],
            restricted_counts: final_counts,
            config,
        }
    }

    /// Get the restricted index for a full grid index.
    ///
    /// Returns the index within the appropriate color's restricted subgraph.
    /// Use `detector_color_at()` to determine which color subgraph.
    #[inline(always)]
    pub fn restrict(&self, full_idx: usize) -> u32 {
        self.full_to_restricted[full_idx]
    }

    /// Get the full grid index from a restricted index and color.
    #[inline(always)]
    pub fn lift(&self, color: FaceColor, restricted_idx: usize) -> u32 {
        self.restricted_to_full[color.index()][restricted_idx]
    }

    /// Get the color of a detector at a full grid index.
    #[inline(always)]
    pub fn color_at(&self, full_idx: usize) -> FaceColor {
        self.config.detector_color_at(full_idx)
    }

    /// Number of detectors in the restricted subgraph for a color.
    #[inline(always)]
    pub fn count(&self, color: FaceColor) -> usize {
        self.restricted_counts[color.index()]
    }

    /// Get the grid configuration.
    #[inline(always)]
    pub fn config(&self) -> &ColorCodeGrid3DConfig {
        &self.config
    }
}

/// Compute the required buffer sizes for restriction maps.
///
/// Returns `(full_to_restricted_size, red_size, green_size, blue_size)`.
#[must_use]
pub const fn required_buffer_sizes(config: &ColorCodeGrid3DConfig) -> (usize, usize, usize, usize) {
    let counts = config.count_by_color();
    (config.alloc_nodes(), counts[0], counts[1], counts[2])
}

/// Restrict a syndrome (defect list) by color.
///
/// Given a list of defect indices in the full grid, separates them into
/// three lists (one per color) with indices converted to restricted form.
///
/// # Arguments
/// * `defects` - Full grid defect indices
/// * `maps` - Restriction maps
/// * `red_out`, `green_out`, `blue_out` - Output buffers for restricted defects
///
/// # Returns
/// `(red_count, green_count, blue_count)` - Number of defects in each color class
pub fn restrict_syndrome<'a>(
    defects: &[u32],
    maps: &RestrictionMaps<'a>,
    red_out: &mut [u32],
    green_out: &mut [u32],
    blue_out: &mut [u32],
) -> (usize, usize, usize) {
    let mut counts = [0usize; 3];

    for &full_idx in defects {
        let color = maps.color_at(full_idx as usize);
        let restricted_idx = maps.restrict(full_idx as usize);

        match color {
            FaceColor::Red => {
                red_out[counts[0]] = restricted_idx;
                counts[0] += 1;
            }
            FaceColor::Green => {
                green_out[counts[1]] = restricted_idx;
                counts[1] += 1;
            }
            FaceColor::Blue => {
                blue_out[counts[2]] = restricted_idx;
                counts[2] += 1;
            }
        }
    }

    (counts[0], counts[1], counts[2])
}

/// Lift corrections from a restricted subgraph back to the full grid.
///
/// # Arguments
/// * `restricted_corrections` - Corrections in restricted indices
/// * `color` - Which color's subgraph these corrections are from
/// * `maps` - Restriction maps
/// * `full_out` - Output buffer for full grid corrections
///
/// # Returns
/// Number of corrections written to `full_out`
pub fn lift_corrections<'a>(
    restricted_corrections: &[(u32, u32)],
    color: FaceColor,
    maps: &RestrictionMaps<'a>,
    full_out: &mut [(u32, u32)],
) -> usize {
    let mut count = 0;

    for &(u_restricted, v_restricted) in restricted_corrections {
        let u_full = maps.lift(color, u_restricted as usize);
        let v_full = if v_restricted == u32::MAX {
            // Boundary correction
            u32::MAX
        } else {
            maps.lift(color, v_restricted as usize)
        };

        full_out[count] = (u_full, v_full);
        count += 1;
    }

    count
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_maps() -> (
        ColorCodeGrid3DConfig,
        [u32; 512], // full_to_restricted
        [u32; 128], // red
        [u32; 128], // green
        [u32; 128], // blue
    ) {
        let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(5);
        let full_to_restricted = [0u32; 512];
        let red = [0u32; 128];
        let green = [0u32; 128];
        let blue = [0u32; 128];
        (config, full_to_restricted, red, green, blue)
    }

    #[test]
    fn test_restriction_roundtrip() {
        let (config, mut full_to_restricted, mut red, mut green, mut blue) = make_test_maps();

        let maps = RestrictionMaps::build(
            config,
            &mut full_to_restricted,
            &mut red,
            &mut green,
            &mut blue,
        );

        // Verify roundtrip for all valid indices
        for t in 0..config.depth {
            for y in 0..config.height {
                for x in 0..config.width {
                    let full_idx = config.coord_to_linear(x, y, t);
                    let color = maps.color_at(full_idx);
                    let restricted_idx = maps.restrict(full_idx);
                    let lifted = maps.lift(color, restricted_idx as usize);

                    assert_eq!(
                        full_idx as u32, lifted,
                        "Roundtrip failed for ({}, {}, {})",
                        x, y, t
                    );
                }
            }
        }
    }

    #[test]
    fn test_color_count_consistency() {
        let (config, mut full_to_restricted, mut red, mut green, mut blue) = make_test_maps();

        let maps = RestrictionMaps::build(
            config,
            &mut full_to_restricted,
            &mut red,
            &mut green,
            &mut blue,
        );

        // Count should match config
        let expected = config.count_by_color();
        assert_eq!(maps.count(FaceColor::Red), expected[0]);
        assert_eq!(maps.count(FaceColor::Green), expected[1]);
        assert_eq!(maps.count(FaceColor::Blue), expected[2]);

        // Total should equal num_detectors
        let total = maps.count(FaceColor::Red)
            + maps.count(FaceColor::Green)
            + maps.count(FaceColor::Blue);
        assert_eq!(total, config.num_detectors());
    }

    #[test]
    fn test_restrict_syndrome() {
        let (config, mut full_to_restricted, mut red, mut green, mut blue) = make_test_maps();

        let maps = RestrictionMaps::build(
            config,
            &mut full_to_restricted,
            &mut red,
            &mut green,
            &mut blue,
        );

        // Create some defects at known positions
        let defects = [
            config.coord_to_linear(0, 0, 0) as u32, // Red (0+0=0)
            config.coord_to_linear(1, 0, 0) as u32, // Green (1+0=1)
            config.coord_to_linear(2, 0, 0) as u32, // Blue (2+0=2)
            config.coord_to_linear(3, 0, 0) as u32, // Red (3+0=0)
        ];

        let mut red_out = [0u32; 8];
        let mut green_out = [0u32; 8];
        let mut blue_out = [0u32; 8];

        let (r, g, b) = restrict_syndrome(&defects, &maps, &mut red_out, &mut green_out, &mut blue_out);

        assert_eq!(r, 2); // Two red defects
        assert_eq!(g, 1); // One green defect
        assert_eq!(b, 1); // One blue defect
    }

    #[test]
    fn test_restricted_indices_unique() {
        let (config, mut full_to_restricted, mut red, mut green, mut blue) = make_test_maps();

        let maps = RestrictionMaps::build(
            config,
            &mut full_to_restricted,
            &mut red,
            &mut green,
            &mut blue,
        );

        // Within each color, restricted indices should be unique and contiguous 0..count
        for color in FaceColor::all() {
            let count = maps.count(color);
            let mut seen = [false; 256];

            for t in 0..config.depth {
                for y in 0..config.height {
                    for x in 0..config.width {
                        let full_idx = config.coord_to_linear(x, y, t);
                        if maps.color_at(full_idx) == color {
                            let restricted_idx = maps.restrict(full_idx) as usize;
                            assert!(restricted_idx < count);
                            assert!(!seen[restricted_idx], "Duplicate restricted index");
                            seen[restricted_idx] = true;
                        }
                    }
                }
            }

            // All indices 0..count should be used
            for i in 0..count {
                assert!(seen[i], "Missing restricted index {}", i);
            }
        }
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    // Note: Kani proofs for this module would require heap allocation
    // or static buffers, which is complex in no_std. The test coverage
    // above provides sufficient verification for the mapping logic.
    // Consider adding Kani proofs when the arena allocator is available
    // in the proof context.
}
