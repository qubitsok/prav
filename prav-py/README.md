# prav

[![PyPI version](https://badge.fury.io/py/prav.svg)](https://badge.fury.io/py/prav)
[![Python versions](https://img.shields.io/pypi/pyversions/prav.svg)](https://pypi.org/project/prav/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

High-performance Union Find decoder for quantum error correction.

## Installation

```bash
pip install prav
```

## Quick Start

```python
import prav
import numpy as np

# Create a decoder for a 17x17 square lattice
decoder = prav.Decoder(17, 17, topology="square")

# Create syndrome data (bitpacked u64 array)
syndromes = np.zeros(8, dtype=np.uint64)
syndromes[0] = 0b11  # Two adjacent defects at nodes 0 and 1

# Decode and get corrections
corrections = decoder.decode(syndromes)
# corrections is a flat array [u0, v0, u1, v1, ...]
```

## Supported Topologies

| Topology | Description | Use Case |
|----------|-------------|----------|
| `square` | 4-neighbor square lattice | Surface codes |
| `triangular` | 6-neighbor triangular lattice | Color codes |
| `honeycomb` | 3-neighbor honeycomb lattice | Kitaev model |
| `3d` | 6-neighbor 3D cubic lattice | 3D codes |

## API Reference

### `prav.Decoder(width, height, topology="square", depth=1)`

Create a new decoder instance.

**Parameters:**
- `width` (int): Grid width in nodes (must be > 0)
- `height` (int): Grid height in nodes (must be > 0)
- `topology` (str): Grid topology (default: "square")
- `depth` (int): Grid depth for 3D codes (default: 1, must be > 0)

**Raises:**
- `ValueError`: If width, height, or depth is 0, or if topology is invalid

### `decoder.decode(syndromes) -> np.ndarray[np.uint32]`

Decode syndromes and return edge corrections.

**Parameters:**
- `syndromes` (np.ndarray[np.uint64]): Dense bitpacked syndrome array

**Returns:**
- Flat correction array `[u0, v0, u1, v1, ...]` where `(u, v)` are edge endpoints.
  `v=0xFFFFFFFF` indicates a boundary correction.

### `decoder.reset()`

Reset decoder state for next decoding cycle.

### `prav.required_buffer_size(width, height, depth=1) -> int`

Calculate required buffer size for a decoder in bytes.

**Parameters:**
- `width` (int): Grid width in nodes (must be > 0)
- `height` (int): Grid height in nodes (must be > 0)
- `depth` (int): Grid depth for 3D codes (default: 1, must be > 0)

**Raises:**
- `ValueError`: If width, height, or depth is 0

### Properties

- `decoder.width`: Grid width in nodes
- `decoder.height`: Grid height in nodes
- `decoder.depth`: Grid depth (1 for 2D codes)
- `decoder.topology`: Grid topology type

## Performance

prav is built on [prav-core](https://crates.io/crates/prav-core), a zero-allocation
Rust implementation optimized for real-time QEC decoding. Key features:

- **No heap allocations**: Uses arena allocator from user-provided buffer
- **Deterministic timing**: No garbage collection, fixed memory layout
- **Cache-optimized**: 64-byte aligned blocks, Morton encoding for spatial locality
- **SWAR bit operations**: Parallel bitwise operations for syndrome processing

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Build the Rust extension
maturin develop

# Run tests
pytest tests/ -v

# Run linting
ruff check .
mypy python/prav
```

## License

MIT OR Apache-2.0
