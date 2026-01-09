//! Syndrome generation for circuit-level benchmarks.
//!
//! Provides both DEM-based sampling and phenomenological noise generation,
//! as well as utilities for splitting syndromes for X/Z basis decoding.

mod circuit_sampler;
mod phenomenological;
mod splitter;

pub use circuit_sampler::CircuitSampler;
#[allow(unused_imports)]
pub use phenomenological::{
    SyndromeWithLogical, count_defects, generate_correlated_syndromes,
    generate_phenomenological_syndromes,
};
pub use splitter::{SplitSyndromes, SyndromeSplitter};
