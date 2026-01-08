// =============================================================================
// Decoder Submodules
// =============================================================================

/// Builder pattern for ergonomic decoder construction.
pub mod builder;

/// Core types: EdgeCorrection, BlockStateHot, BoundaryConfig.
pub mod types;

/// Static graph structure holding topology and neighbor information.
pub mod graph;

/// Cluster growth algorithms for syndrome spreading.
pub mod growth;

/// Peeling decoder for forest reconstruction and path tracing.
pub mod peeling;

/// Core decoding state structures.
pub mod state;

/// Tiled decoder for large grids with 32x32 tile optimization.
pub mod tiled;

/// Union-Find data structure for efficient cluster merging.
pub mod union_find;

/// Kani formal verification proofs for decoder core modules.
#[cfg(kani)]
pub mod kani_proofs;

// =============================================================================
// Public Re-exports
// =============================================================================

// Builder pattern (ergonomic API)
pub use builder::{DecoderBuilder, DynDecoder};

// Core types (from types module)
pub use types::{BlockStateHot, BoundaryConfig, EdgeCorrection, FLAG_VALID_FULL};

// DecodingState (from state module)
pub use state::DecodingState;

// Tiled decoder
pub use tiled::TiledDecodingState;

// Graph structure
pub use graph::StaticGraph;

// Traits (for advanced usage and benchmarks)
pub use growth::ClusterGrowth;
pub use peeling::Peeling;
pub use union_find::UnionFind;
