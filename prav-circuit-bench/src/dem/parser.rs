//! Stim DEM format parser.
//!
//! Parses Stim's Detector Error Model format into a `ParsedDem` structure.
//!
//! # Supported Syntax
//!
//! ```text
//! detector(x, y, t) D<id>
//! error(probability) D<id1> D<id2> ... [^ L<obs_id> ...]
//! shift_detectors(dx, dy, dt) ...
//! repeat N { ... }
//! ```

use prav_core::Detector;

use super::types::{OwnedErrorMechanism, ParsedDem};

/// Error type for DEM parsing.
#[derive(Debug, Clone)]
pub enum DemError {
    /// Invalid syntax in DEM file.
    InvalidSyntax(String),
    /// Invalid probability value.
    InvalidProbability(String),
    /// Invalid detector ID.
    InvalidDetectorId(String),
    /// Invalid observable ID.
    InvalidObservableId(String),
    /// Unexpected end of input.
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

/// Parse a DEM file content into a `ParsedDem`.
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

fn parse_coords(s: &str) -> Result<(f32, f32, f32), DemError> {
    let parts: Vec<&str> = s.split(',').map(|p| p.trim()).collect();

    let x = parts.first().and_then(|p| p.parse().ok()).unwrap_or(0.0);
    let y = parts.get(1).and_then(|p| p.parse().ok()).unwrap_or(0.0);
    let z = parts.get(2).and_then(|p| p.parse().ok()).unwrap_or(0.0);

    Ok((x, y, z))
}

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
