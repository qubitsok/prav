# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-XX-XX

### Added

- Initial public release
- `DecodingState` for Union Find QEC decoding
- `QecEngine` high-level wrapper
- `TiledDecodingState` for large grids (>64x64)
- `Arena` bump allocator for zero-heap operation
- `DecoderBuilder` for ergonomic construction without manual STRIDE_Y calculation
- `DynDecoder` dynamic dispatch wrapper
- `required_buffer_size()` helper function for memory planning
- Support for multiple topologies:
  - `SquareGrid` (4-neighbor, surface codes)
  - `Grid3D` (6-neighbor, 3D codes)
  - `TriangularGrid` (6-neighbor, color codes)
  - `HoneycombGrid` (3-neighbor, Kitaev model)
- Kani formal verification proofs for memory safety
- Multi-platform support: x86-64, aarch64, armv7r, wasm32

### Changed

- License changed from AGPL-3.0 to MIT/Apache-2.0 dual license

### API

- `reset_for_next_cycle()` - efficient sparse reset between decoding cycles
- `full_reset()` - complete state reset
