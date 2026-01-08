Contact: qubits@qubitsok.com | [LinkedIn](https://www.linkedin.com/in/piotrlewandowski1/)

---

# prav

Union Find decoder for Quantum Error Correction.

## What is this?

Quantum computers accumulate errors during computation. Quantum Error Correction (QEC) detects these errors by measuring parity checks called syndromes. This library uses the Union Find algorithm to group syndromes into clusters and determine which corrections to apply.

## Core Properties

prav is `no_std`, portable, heap-free, and built for performance.

- **no_std**: No standard library dependency. Runs on bare metal.
- **Portable**: Compiles to x86-64, ARM64, Cortex-R5, and WebAssembly.
- **No heap**: Uses an arena bump allocator. No dynamic memory allocation at runtime.
- **Performance**: SWAR bit operations, 64-byte cache-aligned blocks, Morton (Z-order) encoding for spatial locality.

## Why this matters

These properties enable deployment on embedded systems and FPGAs. Memory usage is fixed at initialization. There is no garbage collection and no allocation latency. Timing is deterministic. The decoder integrates with real-time quantum control systems that require sub-microsecond response times.

## Supported Topologies

| Topology | Neighbors | Use Case |
|----------|-----------|----------|
| Square | 4 | Surface codes, toric codes |
| Rectangular | 4 | Asymmetric surface codes |
| Triangular | 6 | Color codes |
| Honeycomb | 3 | Kitaev honeycomb model |

## Examples

The `prav-core/examples/` directory contains educational examples demonstrating each grid topology:

| Example | Grid | Description |
|---------|------|-------------|
| `tutorial_square.rs` | 4x4 square | Surface code basics with syndrome visualization |
| `tutorial_rectangular.rs` | 5x3 rectangle | Asymmetric boundaries and distance calculations |
| `tutorial_triangular.rs` | 4x4 triangular | Parity-dependent diagonal neighbors |
| `tutorial_honeycomb.rs` | 4x4 honeycomb | Minimal 3-neighbor connectivity |

Run an example:

```bash
cargo run --example tutorial_square
```

## Performance Measurement

### Instruction-level profiling

```bash
cargo bench
```

Uses iai-callgrind to count instructions. Measures individual decoder operations: reset, load, grow, peel.

### Cluster growth timing

```bash
cargo run --example growth_bench --release
```

Runs 10,000 decoding cycles across all topologies. Reports average time per cycle at multiple error rates.

### Stage breakdown

```bash
cargo run --example stage_bench --release
```

Shows time distribution across decoder stages: reset, load syndromes, grow clusters, peel corrections, compact.

## Compilation Targets

The `prav-core/xtask` tool builds and runs benchmarks on four targets:

| Target | Description | Command |
|--------|-------------|---------|
| x86-64 | Native host | `cargo xtask bench --target x86-64` |
| aarch64 | ARM64 Linux | `cargo xtask bench --target aarch64` |
| armv7r | Bare-metal Cortex-R5 | `cargo xtask bench --target armv7r` |
| wasm32 | WebAssembly | `cargo xtask bench --target wasm32` |

Verify all targets compile:

```bash
cargo xtask check-all
```

## Testing Strategy

### Unit tests

```bash
cargo test
```

In-source tests verify module-level correctness.

### Property tests

```bash
cargo test --test '*prop*'
```

Uses proptest to generate random inputs and verify invariants hold across grid sizes (3x3 to 60x60) and defect counts (0 to 50).

### Formal verification

```bash
cargo kani
```

39 Kani proofs verify memory safety and arithmetic correctness. Proofs cover arena allocation bounds, Union Find determinism, Morton encoding round-trips, and peeling coordinate calculations.

## License

Apache 2.0

---

Contact: qubits@qubitsok.com | [LinkedIn](https://www.linkedin.com/in/piotrlewandowski1/)
