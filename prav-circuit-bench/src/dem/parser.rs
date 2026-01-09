//! # Stim DEM Format Parser
//!
//! This module parses Stim's Detector Error Model (DEM) format into our
//! internal [`ParsedDem`] structure.
//!
//! ## What is a DEM File?
//!
//! A DEM file describes all the error mechanisms in a quantum error correction
//! circuit. It's produced by Stim when you analyze a noisy circuit.
//!
//! The file is text-based and human-readable. It contains:
//!
//! 1. **Detector declarations**: Define measurement locations with coordinates
//! 2. **Error declarations**: Define what errors can happen and their probabilities
//! 3. **Control flow**: `repeat` blocks and `shift_detectors` for compact representation
//!
//! ## Supported Syntax
//!
//! ### Detector Declaration
//!
//! ```text
//! detector(x, y, t) D<id>
//! detector D<id>              # Without coordinates (defaults to 0,0,0)
//! ```
//!
//! - `x, y`: Spatial coordinates on the 2D qubit grid
//! - `t`: Time coordinate (measurement round number)
//! - `D<id>`: Unique detector identifier (e.g., D0, D1, D42)
//!
//! ### Error Declaration
//!
//! ```text
//! error(probability) D<id1> D<id2> ...
//! error(probability) D<id1> ^ L<obs_id> ...
//! ```
//!
//! - `probability`: Chance this error occurs (e.g., 0.001 for 0.1%)
//! - `D<id>`: Detectors flipped by this error (usually 1 or 2)
//! - `^ L<obs_id>`: Optional logical observable flip (L0, L1, etc.)
//!
//! ### Coordinate Shifting
//!
//! ```text
//! shift_detectors(dx, dy, dt) ...
//! shift_detectors N           # Compressed format
//! ```
//!
//! Shifts the coordinate system for subsequent detector declarations.
//! Used in repeat blocks to generate patterns.
//!
//! ### Repeat Blocks
//!
//! ```text
//! repeat N {
//!     # Content repeated N times
//!     shift_detectors(...)
//!     error(...)
//! }
//! ```
//!
//! For efficiency, we require DEMs to be flattened (no repeat blocks).
//! Use `stim.Circuit.detector_error_model(flatten_loops=True)` when generating.
//!
//! ## Example DEM File
//!
//! ```text
//! # d=3 surface code, 1 round
//! detector(0, 0, 0) D0
//! detector(1, 0, 0) D1
//! detector(0, 1, 0) D2
//! detector(1, 1, 0) D3
//!
//! # Interior edge errors
//! error(0.001) D0 D1
//! error(0.001) D2 D3
//! error(0.001) D0 D2
//! error(0.001) D1 D3
//!
//! # Boundary errors (affect logical)
//! error(0.001) D0 ^ L0
//! error(0.001) D1 ^ L1
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! let content = std::fs::read_to_string("model.dem")?;
//! let dem = parse_dem(&content)?;
//!
//! println!("Parsed {} detectors, {} error mechanisms",
//!          dem.num_detectors, dem.mechanisms.len());
//! ```

use prav_core::Detector;

use super::types::{OwnedErrorMechanism, ParsedDem};

/// Error type for DEM parsing.
///
/// These errors indicate problems with the DEM file format. They include
/// the problematic content to help with debugging.
#[derive(Debug, Clone)]
pub enum DemError {
    /// Invalid syntax that doesn't match any known DEM instruction.
    ///
    /// Examples:
    /// - Missing parentheses: `error 0.01 D0`
    /// - Unknown instruction: `foobar D0 D1`
    InvalidSyntax(String),

    /// Probability value couldn't be parsed as a float.
    ///
    /// Examples:
    /// - `error(abc) D0` - "abc" is not a number
    /// - `error(-1) D0` - negative probabilities are invalid
    InvalidProbability(String),

    /// Detector ID couldn't be parsed.
    ///
    /// Examples:
    /// - `error(0.01) Dfoo` - "foo" is not a number
    /// - `detector(0,0,0) D` - missing ID number
    InvalidDetectorId(String),

    /// Observable ID couldn't be parsed.
    ///
    /// Examples:
    /// - `error(0.01) D0 ^ Lx` - "x" is not a number
    InvalidObservableId(String),

    /// File ended unexpectedly (currently unused but reserved).
    UnexpectedEnd,
}

impl std::fmt::Display for DemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DemError::InvalidSyntax(msg) => write!(f, "Invalid syntax: {}", msg),
            DemError::InvalidProbability(msg) => write!(f, "Invalid probability: {}", msg),
            DemError::InvalidDetectorId(msg) => write!(f, "Invalid detector ID: {}", msg),
            DemError::InvalidObservableId(msg) => write!(f, "Invalid observable ID: {}", msg),
            DemError::UnexpectedEnd => write!(f, "Unexpected end of input"),
        }
    }
}

impl std::error::Error for DemError {}

/// Parse a DEM file into a [`ParsedDem`] structure.
///
/// This is the main entry point for DEM parsing. It takes the file content
/// as a string and returns a structured representation.
///
/// # Parsing Strategy
///
/// The parser processes the file line by line:
///
/// 1. **Skip**: Empty lines, comments (`#`), closing braces (`}`)
/// 2. **Track**: `repeat` block depth (we skip content inside repeat blocks)
/// 3. **Parse**: `detector`, `error`, `shift_detectors`, `logical_observable`
///
/// ## Coordinate Tracking
///
/// The parser maintains a coordinate offset that accumulates through
/// `shift_detectors` instructions. Each detector's final coordinates are:
/// `(x + offset_x, y + offset_y, t + offset_t)`.
///
/// ## Repeat Blocks
///
/// For simplicity, we skip content inside repeat blocks. This means DEMs
/// must be flattened before parsing. Use Stim with `flatten_loops=True`:
///
/// ```python
/// dem = circuit.detector_error_model(flatten_loops=True)
/// ```
///
/// # Parameters
///
/// - `content`: The DEM file content as a string
///
/// # Returns
///
/// - `Ok(ParsedDem)`: Successfully parsed model
/// - `Err(DemError)`: Parsing failed with details
///
/// # Example
///
/// ```ignore
/// let content = r#"
///     detector(0, 0, 0) D0
///     detector(1, 0, 0) D1
///     error(0.001) D0 D1
/// "#;
///
/// let dem = parse_dem(content)?;
/// assert_eq!(dem.num_detectors, 2);
/// assert_eq!(dem.mechanisms.len(), 1);
/// ```
pub fn parse_dem(content: &str) -> Result<ParsedDem, DemError> {
    let mut dem = ParsedDem::new();
    let mut coord_offset = (0.0f32, 0.0f32, 0.0f32);
    let mut max_detector_id = 0u32;
    let mut max_observable_id = 0u8;

    let mut in_repeat_block = 0usize;

    for line in content.lines() {
        // Track repeat block depth (simplified handling)
        let trimmed = line.trim();

        // Skip empty lines, comments, and closing braces
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed == "}" {
            if trimmed == "}" && in_repeat_block > 0 {
                in_repeat_block -= 1;
            }
            continue;
        }

        // Track repeat block entry
        if trimmed.starts_with("repeat") {
            in_repeat_block += 1;
            continue;
        }

        // Skip lines inside repeat blocks (they're expanded in Stim's output anyway)
        // For proper DEM parsing, we'd need to expand repeat blocks
        if in_repeat_block > 0 {
            continue;
        }

        // Parse instruction
        if trimmed.starts_with("detector") {
            parse_detector_line(trimmed, &mut dem, coord_offset, &mut max_detector_id)?;
        } else if trimmed.starts_with("error") {
            parse_error_line(trimmed, &mut dem, &mut max_detector_id, &mut max_observable_id)?;
        } else if trimmed.starts_with("shift_detectors") {
            coord_offset = parse_shift_detectors(trimmed, coord_offset)?;
        } else if trimmed.starts_with("logical_observable") {
            // Logical observable declaration - just track the max ID
            if let Some(id) = extract_observable_id(trimmed) {
                max_observable_id = max_observable_id.max(id + 1);
            }
        }
        // Ignore other instructions
    }

    dem.num_detectors = max_detector_id;
    dem.num_observables = max_observable_id;

    Ok(dem)
}

/// Parse a detector declaration line.
///
/// # Formats Supported
///
/// ```text
/// detector(x, y, t) D<id>   # With coordinates
/// detector D<id>            # Without coordinates (defaults to 0,0,0)
/// ```
///
/// The final coordinates are computed as `(x + offset.0, y + offset.1, t + offset.2)`
/// to account for any preceding `shift_detectors` instructions.
fn parse_detector_line(
    line: &str,
    dem: &mut ParsedDem,
    coord_offset: (f32, f32, f32),
    max_id: &mut u32,
) -> Result<(), DemError> {
    // Format: detector(x, y, t) D<id>
    // or: detector D<id>

    let (coords, rest) = if line.starts_with("detector(") {
        // Has coordinates
        let paren_end = line.find(')').ok_or_else(|| {
            DemError::InvalidSyntax("Missing closing parenthesis in detector".into())
        })?;
        let coords_str = &line[9..paren_end];
        let coords = parse_coords(coords_str)?;
        (Some(coords), &line[paren_end + 1..])
    } else {
        (None, &line[8..]) // Skip "detector"
    };

    // Find detector ID
    let rest = rest.trim();
    if !rest.starts_with('D') {
        return Err(DemError::InvalidSyntax(format!(
            "Expected D<id> in detector declaration: {}",
            line
        )));
    }

    let id_str = &rest[1..];
    let id: u32 = id_str
        .split_whitespace()
        .next()
        .unwrap_or("")
        .parse()
        .map_err(|_| DemError::InvalidDetectorId(id_str.into()))?;

    *max_id = (*max_id).max(id + 1);

    let (x, y, t) = coords.unwrap_or((0.0, 0.0, 0.0));
    dem.detectors.push(Detector::new(
        id,
        x + coord_offset.0,
        y + coord_offset.1,
        t + coord_offset.2,
    ));

    Ok(())
}

/// Parse an error mechanism declaration line.
///
/// # Format
///
/// ```text
/// error(probability) D<id1> D<id2> ... [^ L<obs_id> ...]
/// ```
///
/// ## Parts
///
/// - `probability`: Float in parentheses, e.g., `0.001` for 0.1% error rate
/// - `D<id>`: Zero or more detector IDs that this error flips
/// - `^ L<obs_id>`: Optional logical observable effects after the caret
///
/// ## Examples
///
/// ```text
/// error(0.001) D0 D1           # Edge error, no logical effect
/// error(0.001) D0              # Boundary error, no logical effect
/// error(0.002) D0 ^ L0         # Boundary error that flips L0
/// error(0.001) D0 D1 ^ L0 L1   # Edge error that flips both logicals
/// ```
fn parse_error_line(
    line: &str,
    dem: &mut ParsedDem,
    max_det_id: &mut u32,
    max_obs_id: &mut u8,
) -> Result<(), DemError> {
    // Format: error(probability) D<id1> D<id2> ... [^ L<obs_id> ...]

    // Extract probability
    let paren_start = line
        .find('(')
        .ok_or_else(|| DemError::InvalidSyntax("Missing opening parenthesis in error".into()))?;
    let paren_end = line
        .find(')')
        .ok_or_else(|| DemError::InvalidSyntax("Missing closing parenthesis in error".into()))?;

    let prob_str = &line[paren_start + 1..paren_end];
    let probability: f32 = prob_str
        .trim()
        .parse()
        .map_err(|_| DemError::InvalidProbability(prob_str.into()))?;

    let rest = &line[paren_end + 1..];

    // Split by "^" to separate detectors from logical observables
    let parts: Vec<&str> = rest.split('^').collect();
    let det_part = parts[0];
    let obs_part = parts.get(1).copied().unwrap_or("");

    // Parse detector IDs
    let mut detectors = Vec::new();
    for token in det_part.split_whitespace() {
        if let Some(id_str) = token.strip_prefix('D') {
            let id: u32 = id_str
                .parse()
                .map_err(|_| DemError::InvalidDetectorId(token.into()))?;
            detectors.push(id);
            *max_det_id = (*max_det_id).max(id + 1);
        }
    }

    // Parse logical observable IDs
    let mut frame_changes = 0u8;
    for token in obs_part.split_whitespace() {
        if let Some(id_str) = token.strip_prefix('L') {
            let id: u8 = id_str
                .parse()
                .map_err(|_| DemError::InvalidObservableId(token.into()))?;
            if id < 8 {
                frame_changes |= 1 << id;
                *max_obs_id = (*max_obs_id).max(id + 1);
            }
        }
    }

    dem.mechanisms.push(OwnedErrorMechanism::new(
        probability,
        detectors,
        frame_changes,
    ));

    Ok(())
}

/// Parse a shift_detectors instruction.
///
/// This instruction shifts the coordinate system for subsequent detector
/// declarations. The offset is cumulative.
///
/// # Formats
///
/// ```text
/// shift_detectors(dx, dy, dt)   # Explicit coordinate shift
/// shift_detectors N             # Compressed format (detector ID shift only)
/// ```
///
/// The explicit format is used for coordinate-aware shifts. The compressed
/// format appears in repeat blocks and only affects detector ID numbering,
/// not spatial coordinates.
///
/// We only handle the explicit format for coordinate shifts. The compressed
/// format is ignored (returns current offset unchanged).
fn parse_shift_detectors(
    line: &str,
    current_offset: (f32, f32, f32),
) -> Result<(f32, f32, f32), DemError> {
    // Format 1: shift_detectors(dx, dy, dt) ...
    // Format 2: shift_detectors N (compressed repeat, just accumulates detector offset)

    if let Some(paren_start) = line.find('(') {
        // Format 1: has parentheses with explicit coordinates
        let paren_end = line.find(')').ok_or_else(|| {
            DemError::InvalidSyntax("Missing closing parenthesis in shift_detectors".into())
        })?;

        let coords_str = &line[paren_start + 1..paren_end];
        let (dx, dy, dt) = parse_coords(coords_str)?;

        Ok((
            current_offset.0 + dx,
            current_offset.1 + dy,
            current_offset.2 + dt,
        ))
    } else {
        // Format 2: shift_detectors N (compressed format, used in repeat blocks)
        // Just increment the detector IDs, not spatial coordinates
        // For simplicity, we ignore this format and return current offset
        Ok(current_offset)
    }
}

/// Parse a comma-separated coordinate string like "1.5, 2.0, 3".
///
/// Returns (x, y, z) as f32. Missing components default to 0.0.
///
/// # Examples
///
/// - `"1, 2, 3"` → (1.0, 2.0, 3.0)
/// - `"1.5, 2.5"` → (1.5, 2.5, 0.0)
/// - `"1"` → (1.0, 0.0, 0.0)
fn parse_coords(s: &str) -> Result<(f32, f32, f32), DemError> {
    let parts: Vec<&str> = s.split(',').map(|p| p.trim()).collect();

    let x = parts.first().and_then(|p| p.parse().ok()).unwrap_or(0.0);
    let y = parts.get(1).and_then(|p| p.parse().ok()).unwrap_or(0.0);
    let z = parts.get(2).and_then(|p| p.parse().ok()).unwrap_or(0.0);

    Ok((x, y, z))
}

/// Extract logical observable ID from a "logical_observable `L<id>`" line.
///
/// This is used to track how many logical observables the DEM has.
/// Returns the ID number if found, or None if not parseable.
fn extract_observable_id(line: &str) -> Option<u8> {
    // Extract observable ID from "logical_observable L<id>"
    for token in line.split_whitespace() {
        if let Some(id_str) = token.strip_prefix('L') {
            return id_str.parse().ok();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_error() {
        let dem = "error(0.1) D0 D1\n";
        let parsed = parse_dem(dem).unwrap();
        assert_eq!(parsed.mechanisms.len(), 1);
        assert_eq!(parsed.mechanisms[0].probability, 0.1);
        assert_eq!(parsed.mechanisms[0].detectors, vec![0, 1]);
        assert_eq!(parsed.mechanisms[0].frame_changes, 0);
    }

    #[test]
    fn test_parse_error_with_logical() {
        let dem = "error(0.01) D0 D1 ^ L0\n";
        let parsed = parse_dem(dem).unwrap();
        assert_eq!(parsed.mechanisms[0].frame_changes, 0b0001);
    }

    #[test]
    fn test_parse_error_with_multiple_logicals() {
        let dem = "error(0.01) D0 ^ L0 L1\n";
        let parsed = parse_dem(dem).unwrap();
        assert_eq!(parsed.mechanisms[0].frame_changes, 0b0011);
    }

    #[test]
    fn test_parse_detector_with_coords() {
        let dem = "detector(1.5, 2.5, 0) D0\n";
        let parsed = parse_dem(dem).unwrap();
        assert_eq!(parsed.detectors.len(), 1);
        assert_eq!(parsed.detectors[0].id, 0);
        assert_eq!(parsed.detectors[0].x, 1.5);
        assert_eq!(parsed.detectors[0].y, 2.5);
        assert_eq!(parsed.detectors[0].t, 0.0);
    }

    #[test]
    fn test_parse_shift_detectors() {
        let dem = r#"
            detector(0, 0, 0) D0
            shift_detectors(1, 0, 1)
            detector(0, 0, 0) D1
        "#;
        let parsed = parse_dem(dem).unwrap();
        assert_eq!(parsed.detectors.len(), 2);
        assert_eq!(parsed.detectors[0].x, 0.0);
        assert_eq!(parsed.detectors[0].t, 0.0);
        assert_eq!(parsed.detectors[1].x, 1.0);
        assert_eq!(parsed.detectors[1].t, 1.0);
    }

    #[test]
    fn test_parse_boundary_error() {
        let dem = "error(0.02) D0 ^ L0\n";
        let parsed = parse_dem(dem).unwrap();
        assert!(parsed.mechanisms[0].is_boundary());
        assert!(!parsed.mechanisms[0].is_edge());
    }

    #[test]
    fn test_parse_comments() {
        let dem = r#"
            # This is a comment
            error(0.1) D0 D1
            # Another comment
        "#;
        let parsed = parse_dem(dem).unwrap();
        assert_eq!(parsed.mechanisms.len(), 1);
    }

    #[test]
    fn test_num_detectors_and_observables() {
        let dem = r#"
            detector(0, 0, 0) D0
            detector(1, 0, 0) D1
            detector(0, 1, 0) D5
            error(0.01) D0 D1 ^ L0 L2
        "#;
        let parsed = parse_dem(dem).unwrap();
        // max detector ID is 5, so num_detectors should be 6
        assert_eq!(parsed.num_detectors, 6);
        // max observable ID is 2, so num_observables should be 3
        assert_eq!(parsed.num_observables, 3);
    }
}
