Contact: qubits@qubitsok.com | [LinkedIn](https://www.linkedin.com/in/piotrlewandowski1/)

---

# prav

Union Find decoder for Quantum Error Correction.

[![Crates.io](https://img.shields.io/crates/v/prav-core.svg)](https://crates.io/crates/prav-core)
[![docs.rs](https://docs.rs/prav-core/badge.svg)](https://docs.rs/prav-core)
[![PyPI](https://badge.fury.io/py/prav.svg)](https://pypi.org/project/prav/)

## What is this?

Quantum computers accumulate errors during computation. Quantum Error Correction (QEC) detects these errors by measuring parity checks called syndromes. This library uses the Union Find algorithm to group syndromes into clusters and determine which corrections to apply.

## Packages

| Package | Description | Links |
|---------|-------------|-------|
| [prav-core](prav-core/) | Core Rust library (`no_std`, zero-heap) | [crates.io](https://crates.io/crates/prav-core) · [docs.rs](https://docs.rs/prav-core) |
| [prav-py](prav-py/) | Python bindings via PyO3/maturin | [PyPI](https://pypi.org/project/prav/) |
| [prav-circuit-bench](prav-circuit-bench/) | Circuit-level QEC benchmarks and threshold studies | |
| [prav-fb-bench](prav-fb-bench/) | Rust benchmark: prav-core vs fusion-blossom | |
| [prav-py-bench](prav-py-bench/) | Python benchmark: prav vs PyMatching | |

## Quick Start

### Rust

```rust
use prav_core::{Arena, DecoderBuilder, SquareGrid, EdgeCorrection, required_buffer_size};

let size = required_buffer_size(32, 32, 1);
let mut buffer = vec![0u8; size];
let mut arena = Arena::new(&mut buffer);

let mut decoder = DecoderBuilder::<SquareGrid>::new()
    .dimensions(32, 32)
    .build(&mut arena)
    .expect("Failed to build decoder");

let syndromes = vec![0u64; 16];
let mut corrections = [EdgeCorrection::default(); 512];

decoder.load_dense_syndromes(&syndromes);
decoder.grow_clusters();
let count = decoder.peel_forest(&mut corrections);
decoder.reset_for_next_cycle();
```

### Python

```bash
pip install prav
```

```python
import prav
import numpy as np

decoder = prav.Decoder(17, 17, topology="square")
syndromes = np.zeros(8, dtype=np.uint64)
syndromes[0] = 0b11  # Two adjacent defects
corrections = decoder.decode(syndromes)
```

## Core Properties

- **`no_std`**: No standard library dependency. Runs on bare metal.
- **Portable**: Compiles to x86-64, ARM64, Cortex-R5, and WebAssembly.
- **Zero heap**: Uses an arena bump allocator. No dynamic memory allocation at runtime.
- **Performance**: SWAR bit operations, 64-byte cache-aligned blocks, Morton (Z-order) encoding for spatial locality.

These properties enable deployment on embedded systems and FPGAs. Memory usage is fixed at initialization. There is no garbage collection and no allocation latency. Timing is deterministic.

## Advanced Decoders

### Streaming Decoder (Real-time QEC)

Process syndrome measurements round-by-round with a sliding window for low-latency decoding:

```rust
use prav_core::{Arena, StreamingConfig, StreamingDecoder, streaming_buffer_size, Grid3D};

let config = StreamingConfig::for_rotated_surface(5, 3); // d=5, window=3
let buf_size = streaming_buffer_size(config.width, config.height, config.window_size);
let mut buffer = vec![0u8; buf_size];
let mut arena = Arena::new(&mut buffer);

let mut decoder: StreamingDecoder<Grid3D, 4> = StreamingDecoder::new(&mut arena, config);

// Process rounds as they arrive
for round_syndromes in syndrome_stream {
    if let Some(committed) = decoder.ingest_round(&round_syndromes) {
        apply_corrections(committed.round, committed.corrections);
    }
}
// Flush remaining at end of stream
for committed in decoder.flush() {
    apply_corrections(committed.round, committed.corrections);
}
```

### Color Code Decoder (Triangular Lattices)

Decode triangular color codes using the restriction decoder approach with three parallel RGB decoders:

```rust
use prav_core::color_code::{ColorCodeDecoder, ColorCodeGrid3DConfig};
use prav_core::Arena;

let config = ColorCodeGrid3DConfig::for_triangular_6_6_6(5);
let mut buffer = [0u8; 1024 * 1024];
let mut arena = Arena::new(&mut buffer);

let mut decoder: ColorCodeDecoder<'_, 8> = ColorCodeDecoder::new(&mut arena, config)?;
let defects = [/* sparse defect indices */];
let result = decoder.decode(&defects);
```

### Dual X/Z Decoding

For fault-tolerant QEC, decode X and Z error bases separately. See [prav-circuit-bench](prav-circuit-bench/) for the `--dual-decode` mode.

## Supported Topologies

| Topology | Neighbors | Use Case |
|----------|-----------|----------|
| Square | 4 | Surface codes, toric codes |
| Triangular | 6 | Color codes |
| Honeycomb | 3 | Kitaev honeycomb model |
| 3D | 6 | 3D topological codes |

## Benchmarks

### prav-core vs fusion-blossom (Rust)

```bash
cargo run --release -p prav-fb-bench
```

Compares Union Find (prav-core) against MWPM (fusion-blossom) on square surface codes. Reports latency percentiles and speedup across grid sizes and error rates.

### prav vs PyMatching (Python)

```bash
cd prav-py-bench
pip install -r requirements.txt
python benchmark.py
```

Compares prav Python bindings against PyMatching. Tests both individual `decode()` and batch `decode_batch()` methods.

### Circuit-Level Threshold Studies

```bash
cargo run --release -p prav-circuit-bench -- --distances 3,5,7 --shots 10000
```

Measures logical error rate on 3D space-time decoding problems. Supports phenomenological noise, Stim DEM files, color codes (`--color-code`), and dual X/Z decoding (`--dual-decode`).

## Examples

The `prav-core/examples/` directory contains tutorials for each grid topology:

```bash
cargo run --example tutorial_square
cargo run --example tutorial_triangular
cargo run --example tutorial_honeycomb
```

## Testing

```bash
# Unit tests
cargo test

# Property tests (random inputs, various grid sizes)
cargo test --test '*prop*'

# Formal verification (39 Kani proofs)
cargo kani
```

## Cross-Compilation

```bash
cargo xtask check-all  # Verify all targets compile
cargo xtask bench --target wasm32  # WebAssembly
cargo xtask bench --target aarch64  # ARM64
cargo xtask bench --target armv7r  # Bare-metal Cortex-R5
```

## License

Apache-2.0 OR MIT

---

Contact: qubits@qubitsok.com | [LinkedIn](https://www.linkedin.com/in/piotrlewandowski1/)
