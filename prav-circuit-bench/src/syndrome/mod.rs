//! Syndrome generation for circuit-level benchmarks.
//!
//! Provides both DEM-based sampling and phenomenological noise generation.

mod circuit_sampler;
mod phenomenological;

pub use circuit_sampler::CircuitSampler;
pub use phenomenological::{
    count_defects, generate_correlated_syndromes, generate_phenomenological_syndromes,
    SyndromeWithLogical,
};
