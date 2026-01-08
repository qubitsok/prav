//! Unrotated (CSS) planar surface code detector layout.
//!
//! In an unrotated planar surface code of distance d:
//! - X-stabilizers form a d x (d-1) grid
//! - Z-stabilizers form a (d-1) x d grid
//! - The code has rough and smooth boundaries on opposite sides

use prav_core::Detector;

/// Unrotated (CSS) planar surface code layout generator.
pub struct UnrotatedSurfaceCode {
    /// Code distance.
    pub distance: usize,
}

impl UnrotatedSurfaceCode {
    /// Create a new unrotated surface code layout.
    pub fn new(distance: usize) -> Self {
        Self { distance }
    }

    /// Generate X-stabilizer detector coordinates for multiple rounds.
    ///
    /// X-stabilizers form a d x (d-1) grid.
    pub fn x_detector_layout(&self, num_rounds: usize) -> Vec<Detector> {
        let mut detectors = Vec::new();
        let mut det_id = 0u32;

        let d = self.distance;
        let width = d;
        let height = d - 1;

        for round in 0..num_rounds {
            let t = round as f32;

            for y in 0..height {
                for x in 0..width {
                    detectors.push(Detector::new(det_id, x as f32, y as f32, t));
                    det_id += 1;
                }
            }
        }

        detectors
    }

    /// Generate Z-stabilizer detector coordinates for multiple rounds.
    ///
    /// Z-stabilizers form a (d-1) x d grid.
    pub fn z_detector_layout(&self, num_rounds: usize) -> Vec<Detector> {
        let mut detectors = Vec::new();
        let mut det_id = 0u32;

        let d = self.distance;
        let width = d - 1;
        let height = d;

        for round in 0..num_rounds {
            let t = round as f32;

            for y in 0..height {
                for x in 0..width {
                    detectors.push(Detector::new(det_id, x as f32, y as f32, t));
                    det_id += 1;
                }
            }
        }

        detectors
    }

    /// Generate detector layout for a single stabilizer type.
    ///
    /// Uses the X-stabilizer grid by default (d x (d-1)).
    pub fn detector_layout(&self, num_rounds: usize) -> Vec<Detector> {
        self.x_detector_layout(num_rounds)
    }

    /// Get the number of X-stabilizer detectors per round.
    pub fn x_detectors_per_round(&self) -> usize {
        self.distance * (self.distance - 1)
    }

    /// Get the number of Z-stabilizer detectors per round.
    pub fn z_detectors_per_round(&self) -> usize {
        (self.distance - 1) * self.distance
    }

    /// Get grid dimensions for X-stabilizer decoding.
    pub fn x_grid_dims(&self) -> (usize, usize) {
        (self.distance, self.distance - 1)
    }

    /// Get grid dimensions for Z-stabilizer decoding.
    pub fn z_grid_dims(&self) -> (usize, usize) {
        (self.distance - 1, self.distance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unrotated_d3() {
        let code = UnrotatedSurfaceCode::new(3);

        // d=3 X-stabilizers: 3 x 2 = 6 per round
        assert_eq!(code.x_detectors_per_round(), 6);

        // d=3 Z-stabilizers: 2 x 3 = 6 per round
        assert_eq!(code.z_detectors_per_round(), 6);

        let x_dets = code.x_detector_layout(3);
        assert_eq!(x_dets.len(), 18); // 6 per round * 3 rounds
    }

    #[test]
    fn test_unrotated_d5() {
        let code = UnrotatedSurfaceCode::new(5);

        // d=5 X-stabilizers: 5 x 4 = 20 per round
        assert_eq!(code.x_detectors_per_round(), 20);

        // d=5 Z-stabilizers: 4 x 5 = 20 per round
        assert_eq!(code.z_detectors_per_round(), 20);
    }

    #[test]
    fn test_grid_dims() {
        let code = UnrotatedSurfaceCode::new(5);

        assert_eq!(code.x_grid_dims(), (5, 4));
        assert_eq!(code.z_grid_dims(), (4, 5));
    }

    #[test]
    fn test_detector_coordinates() {
        let code = UnrotatedSurfaceCode::new(3);
        let detectors = code.x_detector_layout(2);

        // First round should have t=0
        for det in &detectors[0..6] {
            assert_eq!(det.t, 0.0);
        }

        // Second round should have t=1
        for det in &detectors[6..12] {
            assert_eq!(det.t, 1.0);
        }
    }
}
