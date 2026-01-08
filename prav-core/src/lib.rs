//! # prav-core: High-Performance Union Find Decoder for Quantum Error Correction
//!
//! `prav-core` is a `no_std`, zero-allocation library implementing a Union Find-based
//! decoder for quantum error correction (QEC) codes, particularly optimized for surface
//! codes and related topological codes.
//!
//! ## Overview
//!
//! Quantum computers are inherently noisy, and quantum error correction is essential
//! for fault-tolerant quantum computation. This library implements a decoder that:
//!
//! 1. **Receives syndrome measurements** - Parity check outcomes indicating where errors occurred
//! 2. **Groups syndromes into clusters** - Using Union Find to track connected components
//! 3. **Extracts correction operators** - Edges that, when applied, restore the code state
//!
//! ## Architecture
//!
//! The decoder uses a block-based approach where the lattice is divided into 64-node
//! blocks organized in Morton (Z-order) layout for cache efficiency. Key optimizations:
//!
//! - **Path halving** in Union Find for O(α(n)) amortized complexity
//! - **Monochromatic fast-path** - 95% of blocks at typical error rates
//! - **SWAR (SIMD Within A Register)** bit operations for syndrome spreading
//! - **Sparse reset** - Only reset modified state, not entire data structures
//!
//! ## Quick Start
//!
//! ```ignore
//! use prav_core::{Arena, QecEngine, SquareGrid, EdgeCorrection};
//!
//! // Allocate memory buffer (no heap allocation)
//! let mut buffer = [0u8; 1024 * 1024];
//! let mut arena = Arena::new(&mut buffer);
//!
//! // Create decoder for 32x32 grid
//! let mut engine: QecEngine<SquareGrid, 32> = QecEngine::new(&mut arena, 32, 32, 1);
//!
//! // Load syndrome measurements (one u64 per 64 nodes)
//! let syndromes: &[u64] = &[/* syndrome data */];
//! let mut corrections = [EdgeCorrection { u: 0, v: 0 }; 1024];
//!
//! // Decode and get corrections
//! let num_corrections = engine.process_cycle_dense(syndromes, &mut corrections);
//! ```
//!
//! ## Module Organization
//!
//! - [`arena`] - Bump allocator for `no_std` memory management
//! - [`decoder`] - Core decoding logic (Union Find, cluster growth, peeling)
//! - [`topology`] - Lattice connectivity definitions (square, 3D, triangular, honeycomb)
//! - [`qec_engine`] - High-level API wrapping the decoder
//! - [`intrinsics`] - Low-level bit manipulation and Morton encoding
//! - [`testing_grids`] - Standard grid configurations for testing and benchmarks

#![no_std]
#![deny(missing_docs)]
#![allow(internal_features)]

// =============================================================================
// Module Declarations
// =============================================================================

/// Arena-based memory allocator for no_std environments.
pub mod arena;

/// Core decoder types, traits, and implementations.
pub mod decoder;

/// Low-level bit manipulation, Morton encoding, and syndrome spreading.
pub mod intrinsics;

/// High-level QEC engine wrapper.
pub mod qec_engine;

/// Pre-configured test grid sizes and error probabilities.
pub mod testing_grids;

/// Grid topology definitions (square, 3D, triangular, honeycomb).
pub mod topology;

/// Kani formal verification proofs for arena allocator.
#[cfg(kani)]
mod arena_kani;

// =============================================================================
// Convenience Re-exports (Clean Public API)
// =============================================================================

// Memory allocation and sizing
pub use arena::{Arena, required_buffer_size};

// Builder pattern (ergonomic API for hiding STRIDE_Y)
pub use decoder::{DecoderBuilder, DynDecoder};

// Core decoder types
pub use decoder::{
    BlockStateHot, BoundaryConfig, DecodingState, EdgeCorrection, TiledDecodingState,
    FLAG_VALID_FULL,
};

// Decoder traits (for advanced users and benchmarks)
pub use decoder::{ClusterGrowth, Peeling, StaticGraph, UnionFind};

// High-level engine
pub use qec_engine::QecEngine;

// Testing utilities
pub use testing_grids::{isqrt, GridConfig, TestGrids, ERROR_PROBS};

// Topology types
pub use topology::{Grid3D, HoneycombGrid, SquareGrid, Topology, TriangularGrid};
pub use topology::INTRA_BLOCK_NEIGHBORS;

// Morton encoding (commonly needed for defect generation)
pub use intrinsics::morton_encode_2d;
