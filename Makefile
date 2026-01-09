# =============================================================================
# prav - High-Performance QEC Decoder
# =============================================================================
# Usage: make [target]
# Run 'make help' to see all available targets.

.PHONY: all build release clean test help
.PHONY: prav-core prav-py dev watch
.PHONY: examples tutorial tutorial-rect tutorial-tri tutorial-hex
.PHONY: bench bench-aarch64 bench-wasm bench-arm bench-all-topologies
.PHONY: bench-py bench-py-3d bench-fb bench-circuit
.PHONY: check-all fmt clippy check lint doc

# Default target
all: build

# =============================================================================
# Build Targets
# =============================================================================

## Build all workspace members (debug)
build:
	cargo build --workspace

## Build all workspace members (release, optimized)
release:
	cargo build --workspace --release

## Build prav-core only
prav-core:
	cargo build -p prav-core --release

## Build Python wheel with maturin
prav-py:
	cd prav-py && maturin build --release

# =============================================================================
# Test Targets
# =============================================================================

## Run all tests
test:
	cargo test --workspace

## Run prav-core tests only
test-core:
	cargo test -p prav-core

## Run tests in release mode
test-release:
	cargo test --workspace --release

# =============================================================================
# Examples
# =============================================================================

## Build all examples
examples:
	cargo build -p prav-core --examples --release

## Run square grid tutorial
tutorial:
	cargo run -p prav-core --example tutorial_square --release

## Run rectangular grid tutorial
tutorial-rect:
	cargo run -p prav-core --example tutorial_rectangular --release

## Run triangular grid tutorial
tutorial-tri:
	cargo run -p prav-core --example tutorial_triangular --release

## Run honeycomb grid tutorial
tutorial-hex:
	cargo run -p prav-core --example tutorial_honeycomb --release

## Run growth benchmark example
bench-growth:
	cargo run -p prav-core --example growth_bench --release

## Run stage benchmark example
bench-stage:
	cargo run -p prav-core --example stage_bench --release

# =============================================================================
# Benchmarks (via xtask)
# =============================================================================

## Run bench-suite on native x86_64
bench:
	cargo run -p xtask -- bench --target x86-64

## Run bench-suite on native x86_64 (pinned to core 0)
bench-pinned:
	cargo run -p xtask -- bench --target x86-64 --pin-core 0

## Run bench-suite on ARM64 (via cross)
bench-aarch64:
	cargo run -p xtask -- bench --target aarch64

## Run bench-suite on WebAssembly (Node.js)
bench-wasm:
	cargo run -p xtask -- bench --target wasm32

## Run bench-suite on ARM Cortex-R (QEMU)
bench-arm:
	cargo run -p xtask -- bench --target armv7r

## Run all topology benchmarks (square, rectangle, triangular, honeycomb)
bench-all-topologies:
	cargo run -p xtask -- bench --target x86-64 --topology all

## Run bench-suite for specific topology (usage: make bench-topo TOPO=triangular)
bench-topo:
	cargo run -p xtask -- bench --target x86-64 --topology $(TOPO)

# =============================================================================
# Benchmarks (standalone)
# =============================================================================

## Run Python benchmarks (prav vs PyMatching)
bench-py: dev
	cd prav-py-bench && python benchmark.py

## Run Python benchmarks with custom args (usage: make bench-py-custom ARGS="--grids 32 64 --shots 5000")
bench-py-custom: dev
	cd prav-py-bench && python benchmark.py $(ARGS)

## Run 3D Python benchmarks
bench-py-3d: dev
	cd prav-py-bench && python benchmark_3d.py

## Run prav-fb-bench
bench-fb:
	cargo run -p prav-fb-bench --release

## Run prav-circuit-bench
bench-circuit:
	cargo run -p prav-circuit-bench --release

# =============================================================================
# Cross-compilation
# =============================================================================

## Check compilation for all targets (x86_64, aarch64, armv7r, wasm32)
check-all:
	cargo run -p xtask -- check-all

# =============================================================================
# Development
# =============================================================================

## Rebuild and reinstall prav Python package locally
dev:
	cd prav-py && maturin build --release
	pip install target/wheels/prav-*.whl --force-reinstall

## Build prav-py in development mode (faster, unoptimized)
dev-debug:
	cd prav-py && maturin develop

## Watch for changes and rebuild (requires cargo-watch)
watch:
	cargo watch -x "build -p prav-core"

## Watch and run tests on changes
watch-test:
	cargo watch -x "test -p prav-core"

# =============================================================================
# Code Quality
# =============================================================================

## Format all Rust code
fmt:
	cargo fmt --all

## Check formatting without modifying
fmt-check:
	cargo fmt --all -- --check

## Run clippy lints
clippy:
	cargo clippy --workspace --all-targets -- -D warnings

## Run cargo check
check:
	cargo check --workspace --all-targets

## Run all lints (format check + clippy)
lint: fmt-check clippy

# =============================================================================
# Documentation
# =============================================================================

## Generate documentation
doc:
	cargo doc --workspace --no-deps

## Generate and open documentation in browser
doc-open:
	cargo doc --workspace --no-deps --open

# =============================================================================
# Cleanup
# =============================================================================

## Clean all build artifacts
clean:
	cargo clean
	rm -rf target/wheels
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".hypothesis" -exec rm -rf {} + 2>/dev/null || true

## Clean only prav-core build artifacts
clean-core:
	cargo clean -p prav-core

## Clean Python build artifacts
clean-py:
	rm -rf target/wheels
	rm -rf prav-py/target
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# =============================================================================
# CI/CD Helpers
# =============================================================================

## Run full CI pipeline locally
ci: lint test-release doc

## Run quick validation (format + check + test)
validate: fmt-check check test

# =============================================================================
# Help
# =============================================================================

## Show this help message
help:
	@echo "prav - High-Performance QEC Decoder"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Build:"
	@echo "  build              Build all workspace members (debug)"
	@echo "  release            Build all workspace members (release)"
	@echo "  prav-core          Build prav-core only"
	@echo "  prav-py            Build Python wheel with maturin"
	@echo ""
	@echo "Test:"
	@echo "  test               Run all tests"
	@echo "  test-core          Run prav-core tests only"
	@echo "  test-release       Run tests in release mode"
	@echo ""
	@echo "Examples:"
	@echo "  examples           Build all examples"
	@echo "  tutorial           Run square grid tutorial"
	@echo "  tutorial-rect      Run rectangular grid tutorial"
	@echo "  tutorial-tri       Run triangular grid tutorial"
	@echo "  tutorial-hex       Run honeycomb grid tutorial"
	@echo ""
	@echo "Benchmarks (xtask):"
	@echo "  bench              Run bench-suite on native x86_64"
	@echo "  bench-pinned       Run bench-suite pinned to core 0"
	@echo "  bench-aarch64      Run bench-suite on ARM64 (via cross)"
	@echo "  bench-wasm         Run bench-suite on WebAssembly"
	@echo "  bench-arm          Run bench-suite on ARM Cortex-R (QEMU)"
	@echo "  bench-all-topologies  Run all topology benchmarks"
	@echo "  bench-topo TOPO=x  Run benchmark for specific topology"
	@echo ""
	@echo "Benchmarks (standalone):"
	@echo "  bench-py           Run Python benchmarks (prav vs PyMatching)"
	@echo "  bench-py-3d        Run 3D Python benchmarks"
	@echo "  bench-fb           Run prav-fb-bench"
	@echo "  bench-circuit      Run prav-circuit-bench"
	@echo ""
	@echo "Cross-compilation:"
	@echo "  check-all          Check compilation for all targets"
	@echo ""
	@echo "Development:"
	@echo "  dev                Rebuild and reinstall prav Python package"
	@echo "  dev-debug          Build prav-py in development mode"
	@echo "  watch              Watch for changes and rebuild"
	@echo "  watch-test         Watch and run tests on changes"
	@echo ""
	@echo "Quality:"
	@echo "  fmt                Format all Rust code"
	@echo "  clippy             Run clippy lints"
	@echo "  lint               Run all lints (fmt + clippy)"
	@echo "  check              Run cargo check"
	@echo ""
	@echo "Documentation:"
	@echo "  doc                Generate documentation"
	@echo "  doc-open           Generate and open documentation"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean              Clean all build artifacts"
	@echo "  clean-core         Clean only prav-core artifacts"
	@echo "  clean-py           Clean Python build artifacts"
	@echo ""
	@echo "CI/CD:"
	@echo "  ci                 Run full CI pipeline locally"
	@echo "  validate           Run quick validation"
