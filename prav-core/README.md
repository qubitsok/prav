# prav-core

[![Crates.io](https://img.shields.io/crates/v/prav-core.svg)](https://crates.io/crates/prav-core)
[![Documentation](https://docs.rs/prav-core/badge.svg)](https://docs.rs/prav-core)

High-performance, `no_std`, zero-heap Union Find decoder for quantum error correction.

## Features

- **Zero heap allocations**: All memory from user-provided buffer via arena allocator
- **`no_std` compatible**: Works on embedded systems, FPGAs, and WebAssembly
- **Multiple topologies**: Square (surface codes), 3D, triangular (color codes), honeycomb
- **Blazingly fast**: SWAR bit operations, cache-optimized layout, path halving

## Quick Start

```rust
use prav_core::{Arena, DecoderBuilder, SquareGrid, EdgeCorrection, required_buffer_size};

// Calculate and allocate buffer
let size = required_buffer_size(32, 32, 1);
let mut buffer = vec![0u8; size];
let mut arena = Arena::new(&mut buffer);

// Create decoder using builder (automatically calculates STRIDE_Y)
let mut decoder = DecoderBuilder::<SquareGrid>::new()
    .dimensions(32, 32)
    .build(&mut arena)
    .expect("Failed to build decoder");

// Decode syndromes
let syndromes = vec![0u64; 16]; // Empty syndromes for demo
let mut corrections = [EdgeCorrection::default(); 512];

decoder.load_dense_syndromes(&syndromes);
decoder.grow_clusters();
let count = decoder.peel_forest(&mut corrections);

// Reset for next cycle
decoder.reset_for_next_cycle();
```

## Understanding STRIDE_Y

The `DecodingState` struct requires a `STRIDE_Y` const generic equal to `max(width, height, depth).next_power_of_two()`. If using `DecodingState` directly:

```rust
use prav_core::{Arena, DecodingState, SquareGrid, required_buffer_size};

let size = required_buffer_size(32, 32, 1);
let mut buffer = vec![0u8; size];
let mut arena = Arena::new(&mut buffer);

// STRIDE_Y = 32 for a 32x32 grid (32.next_power_of_two() = 32)
let mut decoder: DecodingState<SquareGrid, 32> = DecodingState::new(&mut arena, 32, 32, 1);
```

Use `DecoderBuilder` to avoid this manual calculation.

## Supported Topologies

| Topology | Neighbors | Use Case |
|----------|-----------|----------|
| `SquareGrid` | 4 | Surface codes, toric codes |
| `Grid3D` | 6 | 3D topological codes |
| `TriangularGrid` | 6 | Color codes |
| `HoneycombGrid` | 3 | Kitaev honeycomb model |

## Memory Requirements

Use `required_buffer_size(width, height, depth)` to calculate exact buffer size:

| Grid Size | Approximate Buffer |
|-----------|-------------------|
| 32x32 | ~100 KB |
| 64x64 | ~400 KB |
| 128x128 | ~1.6 MB |
| 256x256 | ~6.4 MB |

## Minimum Supported Rust Version

Rust 1.85 or later (Edition 2024).

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
