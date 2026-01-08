# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.1] - 2025-01-08

### Added

- Initial public release
- `Decoder` class for Union Find QEC decoding
- `TopologyType` enum with support for:
  - `square`: 4-neighbor square lattice (surface codes)
  - `triangular`: 6-neighbor triangular lattice (color codes)
  - `honeycomb`: 3-neighbor honeycomb lattice
  - `3d`: 6-neighbor 3D cubic lattice
- `required_buffer_size()` helper function
- Input validation for width, height, and depth parameters
- Type stubs (.pyi) for IDE support
- Comprehensive test suite with pytest and hypothesis

### Python Bindings

- `Decoder(width, height, topology="square", depth=1)` - Create decoder instance
- `decoder.decode(syndromes)` - Decode bitpacked syndromes
- `decoder.reset()` - Reset decoder state
- Properties: `width`, `height`, `depth`, `topology`

[Unreleased]: https://github.com/qubitsok/prav/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/qubitsok/prav/releases/tag/v0.0.1
