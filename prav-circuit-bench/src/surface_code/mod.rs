//! Surface code layouts and detector mapping.
//!
//! Provides detector coordinate mapping and surface code geometry.

mod detector_map;
mod rotated;
mod unrotated;

pub use detector_map::DetectorMapper;
pub use rotated::RotatedSurfaceCode;
pub use unrotated::UnrotatedSurfaceCode;
