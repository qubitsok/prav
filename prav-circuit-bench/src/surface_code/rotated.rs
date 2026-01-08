//! Rotated surface code detector layout.
//!
//! In a rotated surface code of distance d:
//! - Physical qubits form a d x d grid
//! - X-stabilizers and Z-stabilizers each cover (d-1)^2 locations
//! - Detectors are placed at stabilizer locations

use prav_core::Detector;

/// Rotated surface code layout generator.
pub struct RotatedSurfaceCode {
    /// Code distance.
    pub distance: usize,
}

impl RotatedSurfaceCode {
    /// Create a new rotated surface code layout.
    pub fn new(distance: usize) -> Self {
        Self { distance }
    }

    /// Generate X-stabilizer detector coordinates for multiple rounds.
    ///
    /// For a distance-d rotated surface code, there are (d-1)^2 / 2 X-stabilizers
    /// (approximately, depends on boundary conditions).
    pub fn x_detector_layout(&self, num_rounds: usize) -> Vec<Detector> {
        let mut detectors = Vec::new();
        let mut det_id = 0u32;

        let d = self.distance;
        let grid_size = d - 1;

        for round in 0..num_rounds {
            let t = round as f32;

            // X-stabilizers at checkerboard pattern
            for y in 0..grid_size {
                for x in 0..grid_size {
                    // X-stabilizers at positions where (x + y) is even
                    if (x + y) % 2 == 0 {
                        detectors.push(Detector::new(det_id, x as f32, y as f32, t));
                        det_id += 1;
                    }
                }
            }
        }

        detectors
    }

    /// Generate Z-stabilizer detector coordinates for multiple rounds.
    pub fn z_detector_layout(&self, num_rounds: usize) -> Vec<Detector> {
        let mut detectors = Vec::new();
        let mut det_id = 0u32;

        let d = self.distance;
        let grid_size = d - 1;

        for round in 0..num_rounds {
            let t = round as f32;

            // Z-stabilizers at checkerboard pattern (opposite parity from X)
            for y in 0..grid_size {
                for x in 0..grid_size {
                    // Z-stabilizers at positions where (x + y) is odd
                    if (x + y) % 2 == 1 {
                        detectors.push(Detector::new(det_id, x as f32, y as f32, t));
                        det_id += 1;
                    }
                }
            }
        }

        detectors
    }

    /// Generate combined detector layout (X and Z together).
    ///
    /// This is the layout used for decoding a single stabilizer type.
    pub fn detector_layout(&self, num_rounds: usize) -> Vec<Detector> {
        let mut detectors = Vec::new();
        let mut det_id = 0u32;

        let grid_size = self.distance - 1;

        for round in 0..num_rounds {
            let t = round as f32;

            // All stabilizers in the (d-1) x (d-1) grid
            for y in 0..grid_size {
                for x in 0..grid_size {
                    detectors.push(Detector::new(det_id, x as f32, y as f32, t));
                    det_id += 1;
                }
            }
        }

        detectors
    }

    /// Get the number of detectors per round.
    pub fn detectors_per_round(&self) -> usize {
        let grid_size = self.distance - 1;
        grid_size * grid_size
    }

    /// Get total number of detectors for given rounds.
    pub fn total_detectors(&self, num_rounds: usize) -> usize {
        self.detectors_per_round() * num_rounds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotated_d3() {
        let code = RotatedSurfaceCode::new(3);

        // d=3 has a 2x2 detector grid per round
        assert_eq!(code.detectors_per_round(), 4);

        let detectors = code.detector_layout(3);
        assert_eq!(detectors.len(), 12); // 4 per round * 3 rounds
    }

    #[test]
    fn test_rotated_d5() {
        let code = RotatedSurfaceCode::new(5);

        // d=5 has a 4x4 detector grid per round
        assert_eq!(code.detectors_per_round(), 16);

        let detectors = code.detector_layout(5);
        assert_eq!(detectors.len(), 80); // 16 per round * 5 rounds
    }

    #[test]
    fn test_detector_coordinates() {
        let code = RotatedSurfaceCode::new(3);
        let detectors = code.detector_layout(2);

        // First round detectors at t=0
        assert_eq!(detectors[0].t, 0.0);
        assert_eq!(detectors[3].t, 0.0);

        // Second round detectors at t=1
        assert_eq!(detectors[4].t, 1.0);
        assert_eq!(detectors[7].t, 1.0);
    }

    #[test]
    fn test_x_z_checkerboard() {
        let code = RotatedSurfaceCode::new(5);

        let x_dets = code.x_detector_layout(1);
        let z_dets = code.z_detector_layout(1);

        // X and Z should cover different positions
        for x_det in &x_dets {
            let (x, y) = (x_det.x as usize, x_det.y as usize);
            assert_eq!((x + y) % 2, 0, "X-stabilizer should be at even parity");
        }

        for z_det in &z_dets {
            let (x, y) = (z_det.x as usize, z_det.y as usize);
            assert_eq!((x + y) % 2, 1, "Z-stabilizer should be at odd parity");
        }

        // Together they should cover the full grid
        assert_eq!(x_dets.len() + z_dets.len(), code.detectors_per_round());
    }
}
