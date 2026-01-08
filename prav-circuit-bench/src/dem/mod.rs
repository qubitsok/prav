//! Stim Detector Error Model (DEM) parsing.
//!
//! This module provides a parser for Stim's DEM format, which describes
//! the probabilistic error model for circuit-level QEC simulation.

pub mod parser;
pub mod types;

pub use parser::parse_dem;
pub use types::ParsedDem;
