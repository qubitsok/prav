//! Circuit-level QEC types and utilities.
//!
//! This module provides `no_std` compatible types for circuit-level noise
//! simulation and Stim DEM (Detector Error Model) representation.
//!
//! # Overview
//!
//! Circuit-level QEC differs from code-capacity noise in that:
//! - Errors occur during gate operations (CNOTs, measurements)
//! - Syndrome extraction is noisy (measurement errors)
//! - Multiple measurement rounds are needed (3D decoding)
//!
//! # Module Organization
//!
//! - [`dem_types`] - Core DEM types (`Detector`, `ErrorMechanism`, etc.)

pub mod dem_types;

pub use dem_types::{
    CompiledDem, Detector, ErrorMechanism, ErrorTarget, LogicalObservable,
};
