# prav-core

[![Crates.io](https://img.shields.io/crates/v/prav-core.svg)](https://crates.io/crates/prav-core)
[![Documentation](https://docs.rs/prav-core/badge.svg)](https://docs.rs/prav-core)

High-performance, `no_std`, zero-heap Union Find decoder for quantum error correction.

## Features

- **Zero heap allocations**: All memory from user-provided buffer via arena allocator
- **`no_std` compatible**: Works on embedded systems, FPGAs, and WebAssembly
- **Multiple topologies**: Square (surface codes), 3D, triangular (color codes), honeycomb
- **Streaming decoder**: Real-time QEC with sliding window for round-by-round processing
- **Color code decoder**: Triangular lattice decoding via restriction approach (3 parallel RGB decoders)
- **Observable tracking**: Phenomenological and circuit-level modes for logical error detection
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

## Module Organization

| Module | Description |
|--------|-------------|
| `arena` | Bump allocator for `no_std` memory management |
| `decoder` | Core decoding logic (Union Find, cluster growth, peeling) |
| `decoder::streaming` | Sliding window streaming decoder for real-time QEC |
| `topology` | Lattice connectivity (square, 3D, triangular, honeycomb) |
| `color_code` | Color code decoder using restriction approach |
| `circuit` | Circuit-level QEC types (DEM, detectors, error mechanisms) |
| `testing_grids` | Standard 2D grid configurations for testing |
| `testing_grids_3d` | 3D grid configurations for circuit-level benchmarks |
| `intrinsics` | Low-level bit manipulation and Morton encoding |
| `qec_engine` | High-level API wrapping the decoder |

## Streaming Decoder

For real-time QEC, use the sliding window streaming decoder that processes syndromes round-by-round:

```rust
use prav_core::{Arena, StreamingConfig, StreamingDecoder, streaming_buffer_size, Grid3D};

// Configure: distance 5, window size 3 rounds
let config = StreamingConfig::for_rotated_surface(5, 3);
let buf_size = streaming_buffer_size(config.width, config.height, config.window_size);
let mut buffer = vec![0u8; buf_size];
let mut arena = Arena::new(&mut buffer);

// Create streaming decoder (STRIDE_Y = 4 for d=5)
let mut decoder: StreamingDecoder<Grid3D, 4> = StreamingDecoder::new(&mut arena, config);

// Process rounds as they arrive from the quantum computer
for round_syndromes in measurement_stream {
    if let Some(committed) = decoder.ingest_round(&round_syndromes) {
        // Round exited window - apply corrections immediately
        apply_corrections(committed.round, committed.corrections);
    }
}

// Flush remaining rounds at end of computation
for committed in decoder.flush() {
    apply_corrections(committed.round, committed.corrections);
}
```

**Key properties:**
- **Circular Z-indexing**: No data copying when window slides
- **Guaranteed correctness**: Corrections only committed when rounds exit window
- **Low latency**: Process each round as it arrives
- **Arena-only allocation**: No heap, deterministic timing

## Color Code Decoder

Decode triangular color codes using the restriction decoder approach with three parallel Union-Find decoders (one per color class):

```rust
use prav_core::color_code::{ColorCodeDecoder, ColorCodeGrid3DConfig};
use prav_core::Arena;

// Configure for distance-5 triangular (6,6,6) color code
let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(5);
let mut buffer = [0u8; 1024 * 1024];
let mut arena = Arena::new(&mut buffer);

// Create color code decoder (STRIDE_Y = 8)
let mut decoder: ColorCodeDecoder<'_, 8> = ColorCodeDecoder::new(&mut arena, config)?;

// Decode sparse defects
let defects = [/* sparse defect indices */];
let result = decoder.decode(&defects);

if result.has_logical_error() {
    // Handle logical error
}
```

**How it works:**
1. Splits syndrome by color class (Red, Green, Blue)
2. Runs three parallel Union-Find decoders
3. Combines results into unified correction

Reference: [Efficient color code decoders from toric code decoders](https://quantum-journal.org/papers/q-2023-02-21-929/)

## Observable Tracking

Track logical observables during decoding for logical error detection:

```rust
use prav_core::{DecoderBuilder, SquareGrid, ObservableMode, Arena, required_buffer_size};

let size = required_buffer_size(32, 32, 1);
let mut buffer = vec![0u8; size];
let mut arena = Arena::new(&mut buffer);

let mut decoder = DecoderBuilder::<SquareGrid>::new()
    .dimensions(32, 32)
    .build(&mut arena)?;

// Enable phenomenological observable tracking
decoder.set_observable_mode(ObservableMode::Phenomenological);

decoder.load_dense_syndromes(&syndromes);
let _ = decoder.decode(&mut corrections);

// Get predicted logical frame
let predicted = decoder.predicted_observables();
if predicted != ground_truth {
    // Logical error occurred
}
```

**Modes:**
- `ObservableMode::Disabled`: No tracking (fastest)
- `ObservableMode::Phenomenological`: Boundary-based inference for simplified noise
- `ObservableMode::CircuitLevel`: Use edge observable LUT from DEM for realistic noise

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
| `Grid3D` | 6 | 3D topological codes, space-time decoding |
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

For streaming decoder, use `streaming_buffer_size(width, height, window_size)`.

## Minimum Supported Rust Version

Rust 1.85 or later (Edition 2024).

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
