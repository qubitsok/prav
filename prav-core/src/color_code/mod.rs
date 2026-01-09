//! Color code decoder for triangular lattices.
//!
//! This module implements a Union-Find based decoder for triangular color codes
//! using the **restriction decoder** approach. Color codes have fundamentally
//! different conservation laws than surface codes:
//!
//! - **Surface codes**: Single error creates 2 defects (anywhere), pair via Union-Find
//! - **Color codes**: Single error creates defects on **same-colored** faces only
//!
//! # Restriction Decoder
//!
//! The restriction decoder projects the color code onto three surface-code-like
//! subgraphs (one per color class) and decodes each independently:
//!
//! ```text
//! Full Color Code Syndrome
//!          │
//!     ┌────┼────┐
//!     ▼    ▼    ▼
//!   ┌───┐┌───┐┌───┐
//!   │Red││Grn││Blu│  ← Three restricted subgraphs
//!   │UF ││UF ││UF │  ← Run Union-Find on each
//!   └─┬─┘└─┬─┘└─┬─┘
//!     │    │    │
//!     └────┼────┘
//!          ▼
//!      Lift & Combine
//!          │
//!          ▼
//!   Color Code Corrections
//! ```
//!
//! # 3-Coloring Property
//!
//! The triangular lattice is 3-colorable: each face (stabilizer) is assigned
//! Red, Green, or Blue such that no two adjacent faces share the same color.
//!
//! ```text
//!     R ─── G ─── B ─── R
//!    / \ / \ / \ / \
//!   B ─── R ─── G ─── B
//!    \ / \ / \ / \ /
//!     G ─── B ─── R ─── G
//! ```
//!
//! # References
//!
//! - [Efficient color code decoders from toric code decoders](https://quantum-journal.org/papers/q-2023-02-21-929/)
//! - [Decoder for Triangular Color Code by Matching on Möbius Strip](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.3.010310)

// Submodules
mod types;
pub mod grid_3d;
pub mod restriction;
pub mod splitter;
pub mod decoder;
pub mod observables;

// Re-exports
pub use types::{ColorCodeBoundaryConfig, ColorCodeResult, FaceColor};
pub use grid_3d::ColorCodeGrid3DConfig;
