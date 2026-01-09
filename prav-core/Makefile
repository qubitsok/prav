# prav-core Makefile
# Comprehensive build, test, and verification targets

.PHONY: all test coverage coverage-html kani lint check check-lib check-tests fmt clean help

# Default target: run all checks and tests
all: check test

# Run all lint checks (formatting + clippy)
check: check-lib check-tests
	@echo "All checks passed!"

# Check library code only
check-lib:
	cargo fmt --check
	cargo clippy --lib -- -D warnings

# Check test code as well
check-tests:
	cargo clippy --tests -- -D warnings

# Auto-fix formatting and lint issues
lint:
	cargo fmt
	cargo clippy --fix --allow-dirty --allow-staged

# Format code only
fmt:
	cargo fmt

# Run all tests
test:
	cargo test --all-features

# Run tests with verbose output
test-verbose:
	cargo test --all-features -- --nocapture

# Generate text coverage report (quick)
coverage:
	cargo llvm-cov --text

# Generate text coverage report with summary only
coverage-summary:
	cargo llvm-cov --summary-only

# Generate HTML coverage report
coverage-html:
	cargo llvm-cov --html --output-dir coverage
	@echo "Coverage report generated at coverage/html/index.html"

# Run Kani formal verification proofs
kani:
	cargo kani --package prav-core

# Run Kani with verbose output
kani-verbose:
	cargo kani --package prav-core -v

# Clean build artifacts
clean:
	cargo clean
	rm -rf coverage/

# Show help
help:
	@echo "prav-core Makefile targets:"
	@echo ""
	@echo "  all            - Run check + test (default)"
	@echo "  check          - Run fmt --check and clippy on lib + tests"
	@echo "  check-lib      - Run fmt --check and clippy on lib only"
	@echo "  check-tests    - Run clippy on tests"
	@echo "  lint           - Auto-fix formatting and clippy issues"
	@echo "  fmt            - Format code with rustfmt"
	@echo "  test           - Run all tests"
	@echo "  test-verbose   - Run tests with output capture disabled"
	@echo "  coverage       - Generate text coverage report"
	@echo "  coverage-summary - Show coverage summary only"
	@echo "  coverage-html  - Generate HTML coverage report"
	@echo "  kani           - Run Kani formal verification"
	@echo "  kani-verbose   - Run Kani with verbose output"
	@echo "  clean          - Remove build artifacts and coverage"
	@echo "  help           - Show this help message"
