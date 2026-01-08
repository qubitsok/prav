# prav-fb-bench

Benchmark comparison of prav-core vs fusion-blossom QEC decoders.

## Overview

This tool measures decode latency on square surface code grids and compares the performance of two quantum error correction decoders:

- **prav-core**: Union Find decoder (this project)
- **fusion-blossom**: Minimum Weight Perfect Matching (MWPM) decoder

Both decoders solve the same problem: given syndrome measurements, determine which corrections to apply. Union Find is an approximate algorithm that runs in near-linear time. MWPM finds optimal solutions but has higher computational cost.

## Features

- **Multiple grid sizes**: Default 17x17, 32x32, 64x64 (configurable)
- **Multiple error rates**: Test across different physical error probabilities
- **Latency percentiles**: Reports avg, p50, p95, p99 for both decoders
- **Speedup calculation**: Shows how many times faster prav is than fusion-blossom
- **Correctness verification**: Verifies both decoders resolve all defects
- **Warmup phase**: 200 warmup shots before measurement to avoid cold-start bias

## Quick Start

```bash
# Run with default settings (17x17, 32x32, 64x64 grids)
cargo run --release -p prav-fb-bench

# Faster run with smaller grid
cargo run --release -p prav-fb-bench -- --grids 17

# Custom configuration
cargo run --release -p prav-fb-bench -- --grids 32 64 --shots 5000 --error-probs 0.01 0.05
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--grids` | 17 32 64 | Grid sizes to benchmark (square grids) |
| `--shots` | 10000 | Number of decoding cycles per error rate |
| `--error-probs` | 0.001 0.01 0.06 | Physical error probabilities to test |
| `--seed` | 42 | Random seed for reproducibility |

## Output

The benchmark produces four tables per grid size:

### 1. prav Latencies

Decode latency in microseconds for the Union Find decoder:

```
    Error Rate|     Defects|       avg|       p50|       p95|       p99
--------------+------------+----------+----------+----------+----------
         0.001|         340|      0.42|      0.38|      0.58|      0.72
          0.01|       3,264|      0.89|      0.82|      1.35|      1.68
          0.06|      18,432|      2.45|      2.31|      3.52|      4.21
```

### 2. fusion-blossom Latencies

Decode latency in microseconds for the MWPM decoder:

```
    Error Rate|     Defects|       avg|       p50|       p95|       p99
--------------+------------+----------+----------+----------+----------
         0.001|         340|      3.21|      2.95|      4.82|      6.14
          0.01|       3,264|     12.45|     11.82|     18.35|     22.68
          0.06|      18,432|     45.67|     43.21|     65.42|     78.91
```

### 3. Speedup

How many times faster prav is compared to fusion-blossom:

```
    Error Rate|       avg|       p50|       p95|       p99
--------------+----------+----------+----------+----------
         0.001|     7.64x|     7.76x|     8.31x|     8.53x
          0.01|    13.99x|    14.41x|    13.59x|    13.50x
          0.06|    18.64x|    18.70x|    18.58x|    18.74x
```

### 4. Correctness Verification

Confirms both decoders properly resolve all syndrome defects:

```
    Error Rate|        prav Success|  fusion-blossom Success|    Feature Parity
--------------+--------------------+------------------------+------------------
         0.001|100.00% (  10,000)|          100.00% (  10,000)|          100.00%
          0.01|100.00% (  10,000)|          100.00% (  10,000)|          100.00%
          0.06|100.00% (  10,000)|          100.00% (  10,000)|          100.00%
```

## How Verification Works

Both decoders return edge corrections. Verification applies these corrections to the original syndrome:

1. For each correction edge (u, v), toggle the syndrome bits at both endpoints
2. After applying all corrections, count remaining defects
3. Success = all defects resolved (count = 0)

This verifies functional correctness without comparing specific matchings, since different valid matchings may exist.

## Graph Construction

The benchmark constructs equivalent graphs for both decoders:

- **prav-core**: Uses `SquareGrid` topology with 4-neighbor connectivity
- **fusion-blossom**: Surface code graph with:
  - Horizontal edges between adjacent columns
  - Vertical edges between adjacent rows
  - Single virtual boundary vertex connected to all boundary nodes
  - Weights computed as `1000 * ln((1-p)/p)` where p is error probability

## Dependencies

- **prav-core**: The Union Find decoder being benchmarked
- **fusion-blossom**: Reference MWPM implementation
- **rand** / **rand_xoshiro**: Syndrome generation
- **clap**: Command line parsing

## Notes

- Always run with `--release` for accurate timing
- The warmup phase (200 shots) prevents cold-cache effects
- Error probability affects defect density and decode time
- Higher error rates show larger speedup (more defects to process)
- Both decoders achieve 100% defect resolution on surface codes

## License

Apache-2.0 OR MIT
