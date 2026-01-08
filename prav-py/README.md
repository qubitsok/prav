# prav

High-performance Union Find decoder for quantum error correction.

## Installation

```bash
pip install prav
```

## Usage

```python
import prav
import numpy as np

# Create a decoder for a 17x17 square lattice
decoder = prav.Decoder(17, 17, topology="square")

# Create syndrome data (bitpacked u64 array)
# Each u64 represents 64 consecutive nodes
syndromes = np.zeros(8, dtype=np.uint64)
syndromes[0] = 0b11  # Two adjacent defects at nodes 0 and 1

# Decode and get corrections
corrections = decoder.decode(syndromes)
# corrections is a flat array [u0, v0, u1, v1, ...]
# where (u, v) are edge endpoints, v=0xFFFFFFFF means boundary
```

## Supported Topologies

- `square`: 4-neighbor square lattice (surface codes)
- `triangular`: 6-neighbor triangular lattice (color codes)
- `honeycomb`: 3-neighbor honeycomb lattice
- `3d`: 6-neighbor 3D cubic lattice

## API

### `prav.Decoder(width, height, topology="square", depth=1)`

Create a new decoder instance.

- `width`: Grid width in nodes
- `height`: Grid height in nodes
- `topology`: Grid topology (default: "square")
- `depth`: Grid depth for 3D codes (default: 1)

### `decoder.decode(syndromes)`

Decode syndromes and return edge corrections.

- `syndromes`: Dense bitpacked syndrome array (`np.ndarray[np.uint64]`)
- Returns: Flat correction array (`np.ndarray[np.uint32]`)

### `prav.required_buffer_size(width, height, depth=1)`

Calculate required buffer size for a decoder.

## License

MIT OR Apache-2.0
