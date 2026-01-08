# prav-py-bench

Benchmark comparison of prav (Python bindings) vs PyMatching QEC decoders.

## Overview

This tool measures decode latency on square surface code grids and compares the performance of two quantum error correction decoders:

- **prav**: Union Find decoder via Python bindings (this project)
- **PyMatching**: Minimum Weight Perfect Matching (MWPM) decoder

Both decoders solve the same problem: given syndrome measurements, determine which corrections to apply. Union Find is an approximate algorithm that runs in near-linear time. MWPM finds optimal solutions but has higher computational cost.

## Features

- **Multiple grid sizes**: Default 17x17, 32x32, 64x64 (configurable)
- **Multiple error rates**: Test across different physical error probabilities
- **Latency percentiles**: Reports avg, p50, p95, p99 for both decoders
- **Speedup calculation**: Shows how many times faster prav is than PyMatching
- **Batch benchmarking**: Tests both PyMatching `decode()` and `decode_batch()` methods
- **Correctness verification**: Verifies both decoders resolve all defects
- **Warmup phase**: 200 warmup shots before measurement to avoid cold-start bias

## Prerequisites

### 1. Install prav Python bindings

From the repository root:

```bash
cd prav-py
pip install maturin
maturin develop --release
```

### 2. Install benchmark dependencies

```bash
cd prav-py-bench
pip install -r requirements.txt
```

## Quick Start

```bash
# Run with default settings (17x17, 32x32, 64x64 grids)
python benchmark.py

# Faster run with smaller grid
python benchmark.py --grids 17

# Custom configuration
python benchmark.py --grids 32 64 --shots 5000 --error-probs 0.01 0.05
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--grids` | 17 32 64 | Grid sizes to benchmark (square grids) |
| `--shots` | 10000 | Number of decoding cycles per error rate |
| `--error-probs` | 0.001 0.003 0.006 0.01 0.03 0.06 | Physical error probabilities |
| `--seed` | 42 | Random seed for reproducibility |

## Output

The benchmark produces six tables per grid size:

### 1. prav Latencies

Decode latency in microseconds for the Union Find decoder:

```
+--------------+----------+--------+--------+--------+--------+
| Error Rate   | Defects  |    avg |    p50 |    p95 |    p99 |
+==============+==========+========+========+========+========+
| 0.001        |      340 |   1.42 |   1.38 |   1.58 |   1.72 |
| 0.010        |    3,264 |   2.89 |   2.82 |   3.35 |   3.68 |
| 0.060        |   18,432 |   6.45 |   6.31 |   7.52 |   8.21 |
+--------------+----------+--------+--------+--------+--------+
```

### 2. PyMatching Latencies - decode()

Decode latency in microseconds for individual decode calls:

```
+--------------+----------+--------+--------+--------+--------+
| Error Rate   | Defects  |    avg |    p50 |    p95 |    p99 |
+==============+==========+========+========+========+========+
| 0.001        |      340 |   5.21 |   4.95 |   6.82 |   8.14 |
| 0.010        |    3,264 |  15.45 |  14.82 |  21.35 |  25.68 |
| 0.060        |   18,432 |  52.67 |  50.21 |  72.42 |  85.91 |
+--------------+----------+--------+--------+--------+--------+
```

### 3. PyMatching Latencies - decode_batch()

Average latency per shot when using batch decoding:

```
+--------------+----------+--------+
| Error Rate   | Defects  |    avg |
+==============+==========+========+
| 0.001        |      340 |   3.21 |
| 0.010        |    3,264 |   9.45 |
| 0.060        |   18,432 |  32.67 |
+--------------+----------+--------+
```

### 4. Speedup vs decode()

How many times faster prav is compared to individual PyMatching calls:

```
+--------------+--------+--------+--------+--------+
| Error Rate   |    avg |    p50 |    p95 |    p99 |
+==============+========+========+========+========+
| 0.001        |  3.67x |  3.59x |  4.32x |  4.73x |
| 0.010        |  5.35x |  5.26x |  6.37x |  6.98x |
| 0.060        |  8.17x |  7.96x |  9.63x | 10.47x |
+--------------+--------+--------+--------+--------+
```

### 5. Speedup vs decode_batch()

How many times faster prav is compared to PyMatching batch decoding:

```
+--------------+--------+
| Error Rate   |    avg |
+==============+========+
| 0.001        |  2.26x |
| 0.010        |  3.27x |
| 0.060        |  5.06x |
+--------------+--------+
```

### 6. Correctness Verification

Confirms both decoders properly resolve all syndrome defects:

```
+--------------+-------------------+---------------------+----------------+
| Error Rate   | prav Success      | PyMatching Success  | Feature Parity |
+==============+===================+=====================+================+
| 0.001        | 100.00% (10,000)  | 100.00% (10,000)    | 100.00%        |
| 0.010        | 100.00% (10,000)  | 100.00% (10,000)    | 100.00%        |
| 0.060        | 100.00% (10,000)  | 100.00% (10,000)    | 100.00%        |
+--------------+-------------------+---------------------+----------------+
```

## How Verification Works

Both decoders return edge corrections. Verification applies these corrections to the original syndrome:

**prav verification:**
1. For each correction edge (u, v), toggle the syndrome bits at both endpoints
2. Boundary corrections have `v = 0xFFFFFFFF`
3. After applying all corrections, count remaining defects

**PyMatching verification:**
1. Compute syndrome change: `H @ corrections (mod 2)` where H is the parity check matrix
2. XOR the change with the original syndrome
3. Count remaining defects

Success = all defects resolved (count = 0).

## Graph Construction

The benchmark constructs equivalent graphs for both decoders:

- **prav**: Uses `square` topology with 4-neighbor connectivity
- **PyMatching**: Surface code parity check matrix with:
  - Horizontal edges between adjacent columns
  - Vertical edges between adjacent rows
  - Boundary edges on all four sides
  - Weights computed as `log((1-p)/p)` where p is error probability

## File Structure

| File | Description |
|------|-------------|
| `benchmark.py` | Main benchmark script with CLI interface |
| `syndrome_generator.py` | Generates random syndromes in both formats |
| `verification.py` | Validates decoder correctness |
| `requirements.txt` | Python dependencies |

## Dependencies

- **prav**: Python bindings for the Union Find decoder (must be installed from prav-py)
- **pymatching**: Reference MWPM implementation
- **numpy**: Array operations and random number generation
- **scipy**: Sparse matrix for parity check matrix
- **tabulate**: Formatted table output

## Notes

- prav Python bindings must be built with `--release` for accurate benchmarks
- The warmup phase (200 shots) prevents cold-cache effects and JIT compilation overhead
- Error probability affects defect density and decode time
- Higher error rates show larger speedup (more defects to process)
- `decode_batch()` amortizes PyMatching's Python overhead across multiple shots
- Both decoders achieve 100% defect resolution on surface codes

## License

Apache-2.0 OR MIT
