//! Detector coordinate to Morton index mapping.

use prav_core::{Detector, Grid3DConfig};

/// Maps detector coordinates to prav's linear indices.
///
/// This handles the translation between DEM detector coordinates
/// and the stride-based layout used by `DecodingState`.
pub struct DetectorMapper {
    width: usize,
    height: usize,
    depth: usize,
    stride_y: usize,
    stride_z: usize,
}

impl DetectorMapper {
    /// Create a new detector mapper for the given grid configuration.
    pub fn new(config: &Grid3DConfig) -> Self {
        Self {
            width: config.width,
            height: config.height,
            depth: config.depth,
            stride_y: config.stride_y,
            stride_z: config.stride_z,
        }
    }

    /// Convert detector (x, y, t) coordinates to linear index.
    ///
    /// Uses stride-based layout: `t * stride_z + y * stride_y + x`
    pub fn detector_to_linear(&self, det: &Detector) -> u32 {
        let (x, y, t) = det.int_coords();
        self.coord_to_linear(x, y, t)
    }

    /// Convert (x, y, t) coordinates to linear index.
    pub fn coord_to_linear(&self, x: usize, y: usize, t: usize) -> u32 {
        (t * self.stride_z + y * self.stride_y + x) as u32
    }

    /// Convert linear index back to (x, y, t) coordinates.
    pub fn linear_to_coord(&self, idx: u32) -> (usize, usize, usize) {
        let idx = idx as usize;
        let t = idx / self.stride_z;
        let rem = idx % self.stride_z;
        let y = rem / self.stride_y;
        let x = rem % self.stride_y;
        (x, y, t)
    }

    /// Remap a DEM syndrome to prav format.
    ///
    /// Takes a syndrome in DEM format (detectors ordered by ID) and remaps
    /// it to prav's stride-based layout.
    pub fn remap_syndrome(&self, dem_syndrome: &[u64], detectors: &[Detector]) -> Vec<u64> {
        let num_nodes = self.stride_z * self.depth;
        let num_words = (num_nodes + 63) / 64;
        let mut prav_syndrome = vec![0u64; num_words];

        // Iterate over set bits in the DEM syndrome
        for (word_idx, &word) in dem_syndrome.iter().enumerate() {
            let mut w = word;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                w &= w - 1;

                let det_id = word_idx * 64 + bit;
                if let Some(det) = detectors.iter().find(|d| d.id == det_id as u32) {
                    let linear = self.detector_to_linear(det) as usize;
                    let blk = linear / 64;
                    let b = linear % 64;
                    if blk < prav_syndrome.len() {
                        prav_syndrome[blk] |= 1 << b;
                    }
                }
            }
        }

        prav_syndrome
    }

    /// Check if coordinates are within bounds.
    pub fn in_bounds(&self, x: usize, y: usize, t: usize) -> bool {
        x < self.width && y < self.height && t < self.depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prav_core::TestGrids3D;

    #[test]
    fn test_coord_roundtrip() {
        let config = TestGrids3D::D5;
        let mapper = DetectorMapper::new(&config);

        for x in 0..config.width {
            for y in 0..config.height {
                for t in 0..config.depth {
                    let linear = mapper.coord_to_linear(x, y, t);
                    let (x2, y2, t2) = mapper.linear_to_coord(linear);
                    assert_eq!((x, y, t), (x2, y2, t2));
                }
            }
        }
    }

    #[test]
    fn test_detector_to_linear() {
        let config = TestGrids3D::D3;
        let mapper = DetectorMapper::new(&config);

        let det = Detector::new(0, 1.0, 1.0, 2.0);
        let linear = mapper.detector_to_linear(&det);
        let (x, y, t) = mapper.linear_to_coord(linear);
        assert_eq!((x, y, t), (1, 1, 2));
    }

    #[test]
    fn test_remap_syndrome() {
        let config = TestGrids3D::D3;
        let mapper = DetectorMapper::new(&config);

        // Create detectors at known positions
        let detectors = vec![
            Detector::new(0, 0.0, 0.0, 0.0),
            Detector::new(1, 1.0, 0.0, 0.0),
        ];

        // DEM syndrome with detector 0 and 1 active
        let dem_syndrome = vec![0b11u64];

        let prav_syndrome = mapper.remap_syndrome(&dem_syndrome, &detectors);

        // Check that both detectors are set
        let linear_0 = mapper.coord_to_linear(0, 0, 0) as usize;
        let linear_1 = mapper.coord_to_linear(1, 0, 0) as usize;

        let word_0 = linear_0 / 64;
        let bit_0 = linear_0 % 64;
        let word_1 = linear_1 / 64;
        let bit_1 = linear_1 % 64;

        assert!(prav_syndrome[word_0] & (1 << bit_0) != 0);
        assert!(prav_syndrome[word_1] & (1 << bit_1) != 0);
    }

    #[test]
    fn test_in_bounds() {
        let config = TestGrids3D::D5;
        let mapper = DetectorMapper::new(&config);

        assert!(mapper.in_bounds(0, 0, 0));
        assert!(mapper.in_bounds(config.width - 1, config.height - 1, config.depth - 1));
        assert!(!mapper.in_bounds(config.width, 0, 0));
        assert!(!mapper.in_bounds(0, config.height, 0));
        assert!(!mapper.in_bounds(0, 0, config.depth));
    }
}
