# prav-circuit-bench

A benchmarking tool for the **prav** quantum error correction decoder. This tool measures how well prav corrects errors in simulated quantum computers.

**Who is this for?** Researchers and students studying quantum error correction who want to test decoder performance on realistic error models.

**What does it do?** Generates noisy quantum measurements, runs the decoder, and reports how often logical errors occur.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Background: Quantum Error Correction](#background-quantum-error-correction)
   - [Why Errors Matter](#why-errors-matter)
   - [The Surface Code](#the-surface-code)
   - [Syndromes and Detectors](#syndromes-and-detectors)
   - [Logical Errors](#logical-errors)
3. [The Union-Find Decoder](#the-union-find-decoder)
   - [The Decoding Problem](#the-decoding-problem)
   - [How Union-Find Works](#how-union-find-works)
   - [Comparison with MWPM](#comparison-with-mwpm)
4. [How This Benchmark Works](#how-this-benchmark-works)
   - [Pipeline Overview](#pipeline-overview)
   - [Two Syndrome Sources](#two-syndrome-sources)
   - [What We Measure](#what-we-measure)
5. [Installation](#installation)
6. [Quick Start](#quick-start)
7. [Command-Line Reference](#command-line-reference)
8. [Understanding the Output](#understanding-the-output)
9. [Threshold Studies](#threshold-studies)
10. [Using Stim DEM Files](#using-stim-dem-files)
11. [Architecture Overview](#architecture-overview)
12. [Glossary](#glossary)

---

## Introduction

`prav-circuit-bench` tests the **prav** Union-Find decoder on 3D quantum error correction problems. It answers a simple question: *How often does the decoder fail?*

A decoder "fails" when it makes a **logical error** - a correction that changes the stored quantum information. The fewer logical errors, the better the decoder.

This tool supports two modes:
1. **Phenomenological noise**: A simplified error model built into the tool
2. **Stim DEM files**: Realistic circuit-level noise from the Stim simulator

For serious threshold studies, use Stim DEM files. The phenomenological model is useful for quick sanity checks.

---

## Background: Quantum Error Correction

### Why Errors Matter

A quantum computer stores information in **qubits**. Unlike classical bits (0 or 1), qubits can be in **superposition** - a combination of 0 and 1 at the same time. This is what makes quantum computers powerful.

The problem: qubits are fragile. They interact with their environment and accumulate errors. An error might:
- Flip a qubit from 0 to 1 (a **bit-flip** or **X error**)
- Change the phase of a superposition (a **phase-flip** or **Z error**)
- Do both at once (a **Y error**)

Without error correction, errors accumulate and destroy the quantum computation within microseconds.

**Why not just copy the qubit?** In classical computing, we protect data by making copies. If one copy gets corrupted, we use the others. But quantum mechanics forbids copying unknown quantum states (the **no-cloning theorem**). We need a different approach.

**The solution: redundancy without copying.** We spread the information of one "logical" qubit across many "physical" qubits. Errors on individual physical qubits can be detected and corrected without measuring (and destroying) the logical qubit.

### The Surface Code

The **surface code** is the most promising error correction scheme for near-term quantum computers. It uses a 2D grid of physical qubits:

```
    Data qubit:     ○
    Measure qubit:  ●

    d=3 rotated surface code (9 data qubits, 8 measure qubits):

           ○───●───○
           │ Z │ X │
           ●───○───●
           │ X │ Z │
           ○───●───○
```

**Data qubits** (○) store the actual quantum information. **Measure qubits** (●) detect errors by measuring correlations between neighboring data qubits.

There are two types of measurements:
- **Z stabilizers**: Detect X (bit-flip) errors
- **X stabilizers**: Detect Z (phase-flip) errors

The **code distance** `d` is the size of the grid. A distance-d surface code:
- Uses roughly `d²` physical qubits
- Can correct up to `(d-1)/2` errors
- Has a grid of `(d-1) × (d-1)` measurement qubits

Larger distance = better error protection, but more qubits needed.

### Syndromes and Detectors

When we measure the stabilizers, each measurement gives a result: +1 or -1. If no errors occurred, all measurements return +1. When an error happens, some measurements flip to -1.

A **syndrome** is the pattern of -1 measurements. Each measurement location is called a **detector**.

```
    An X error on the center data qubit:

           ○───●───○          Syndrome:
           │ Z │ X │
           ●───○*──●           Z  X
           │ X │ Z │           -1 +1
           ○───●───○           +1 +1

    The error flips the Z stabilizer it touches (marked -1).
```

Key insight: An error flips the detectors on either side of it. Interior errors always flip exactly 2 detectors. Boundary errors flip only 1 detector.

**The 3D picture**: In a real quantum computer, we measure the stabilizers repeatedly over time. This creates a 3D structure:
- X and Y axes: spatial positions of detectors
- T axis: time (measurement round)

```
    3D syndrome (space × space × time):

    t=2:   ○──○──○
           │  │  │
    t=1:   ○──●──○   ← defect at (1,0,1)
           │  │  │
    t=0:   ○──●──○   ← defect at (1,0,0)

    A time-like error (measurement error) creates two defects
    at the same spatial position but adjacent time steps.
```

### Logical Errors

The decoder's job is to figure out which errors caused the observed syndrome and apply corrections. But here's the catch: **multiple error patterns can produce the same syndrome**.

```
    Two error chains with the same syndrome:

    Chain A:   ○───●───○       Chain B:   ○───●───○
               │   │   │                  │   │   │
               ●───○───●                  ●───○───●
               │ * │   │                  │ * * * │
               ○───●───○                  ○───●───○
                   ↑                              ↑
               1 error                        3 errors
```

Both chains flip the same detectors! The decoder must guess which one occurred.

If the decoder guesses wrong and the two chains differ by a path that crosses the entire code, the correction **changes the logical qubit**. This is a **logical error** - the decoder "fixed" the syndrome but corrupted the stored information.

The **threshold** is the physical error rate below which increasing the code distance reduces logical errors. Above threshold, bigger codes do worse. Below threshold, bigger codes do better.

A good decoder has a high threshold (around 1% for the surface code).

---

## The Union-Find Decoder

### The Decoding Problem

Given a syndrome (the pattern of triggered detectors), find a set of corrections that:
1. Resolves all detector triggers (pairs them up or connects them to boundaries)
2. Minimizes the chance of a logical error

The optimal solution is called **Minimum-Weight Perfect Matching (MWPM)**. It finds the correction with the smallest total weight (fewest errors assumed). But MWPM is slow: O(n³) for n detectors.

### How Union-Find Works

The **Union-Find** decoder is an approximate algorithm that trades optimality for speed. It runs in near-linear time: O(n α(n)) where α is the inverse Ackermann function (effectively constant).

Here's how it works:

**Step 1: Initialize clusters**

Each triggered detector starts as its own cluster.

```
    Syndrome:        Initial clusters:
    ●   ○   ●        {A}     {B}
    ○   ○   ○
    ○   ●   ○              {C}
```

**Step 2: Grow clusters**

All clusters grow outward simultaneously. When two clusters touch, they merge.

```
    After growing:
      A A A B B
      A A ○ B B
      ○ C C C ○

    Clusters A and B merged, C reached a boundary.
```

**Step 3: Extract corrections**

For each merged cluster, find a spanning tree connecting all defects. The edges of this tree are the corrections.

**Key properties:**
- Very fast: O(n α(n)) time complexity
- Parallelizable: clusters can grow independently
- Approximate: may not find optimal solution
- Good enough: threshold only slightly lower than MWPM

### Comparison with MWPM

| Property | Union-Find | MWPM |
|----------|------------|------|
| Time complexity | O(n α(n)) ≈ O(n) | O(n³) |
| Optimality | Approximate | Optimal |
| Threshold | ~10.5% | ~10.9% |
| Parallelization | Excellent | Limited |

For large codes and real-time decoding, Union-Find is the practical choice. The small threshold penalty is worth the massive speedup.

---

## How This Benchmark Works

### Pipeline Overview

The benchmark follows these steps:

```
    ┌─────────────────┐
    │ Generate/Load   │  Create noisy syndromes
    │   Syndromes     │  (from model or DEM file)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   Run Decoder   │  prav Union-Find decoder
    │                 │  produces corrections
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │    Verify       │  Apply corrections,
    │  Corrections    │  check if syndrome resolved
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Track Logical   │  Compare predicted vs actual
    │    Errors       │  logical frame changes
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   Statistics    │  LER, Lambda, latency
    └─────────────────┘
```

### Two Syndrome Sources

**1. Phenomenological Model (built-in)**

A simplified noise model with two error types:
- **Space-like errors**: Data qubit errors that flip adjacent detectors
- **Time-like errors**: Measurement errors that flip the same detector at consecutive times

```
    Space-like error:        Time-like error:

    t:  ○──●──●──○           t+1: ○──●──○
           └──┘                     │
       adjacent detectors       same detector
           flipped              at two times
```

Good for quick tests. Not accurate for threshold estimation.

**2. Stim DEM Files (recommended)**

Stim is a fast stabilizer circuit simulator. It outputs **Detector Error Models (DEM)** that capture the true error structure of quantum circuits including:
- Gate errors (depolarizing noise after each operation)
- Measurement errors (bit flips when reading out)
- Reset errors (starting in wrong state)
- Correlated errors (errors that affect multiple qubits)

For accurate threshold studies, always use Stim DEM files.

### What We Measure

**Logical Error Rate (LER)**: The fraction of shots where the decoder made a logical error.

```
LER = (logical errors) / (total shots × rounds)
```

We report LER per round because more rounds means more chances for errors.

**Error Suppression Factor (Lambda, Λ)**: How much better a larger code performs.

```
Λ = LER(distance d) / LER(distance d+2)
```

- Λ > 1: Larger code is better (below threshold) ✓
- Λ < 1: Larger code is worse (above threshold) ✗
- Λ = 1: At threshold

**Decode Latency**: How long each decode takes (in microseconds). Important for real-time decoding.

---

## Installation

### Prerequisites

- Rust toolchain (1.70+)
- Python 3.8+ (for Stim DEM generation)

### Build

```bash
# Build the benchmark
cargo build --release -p prav-circuit-bench

# Optional: install Stim for DEM generation
pip install stim
```

### Verify Installation

```bash
# Run a quick test
cargo run --release -p prav-circuit-bench -- --shots 1000 --distances 3,5
```

---

## Quick Start

### Basic benchmark with phenomenological noise

```bash
cargo run --release -p prav-circuit-bench
```

### Threshold study with Stim DEM files (recommended)

```bash
# Generate DEM files
make generate-dems

# Run threshold study
make threshold-study
```

### Custom distances and error rates

```bash
cargo run --release -p prav-circuit-bench -- \
    --distances 3,5,7,9 \
    --error-probs 0.001,0.003,0.005,0.007,0.01 \
    --shots 50000
```

---

## Command-Line Reference

| Option | Default | Description |
|--------|---------|-------------|
| `--dem <PATH>` | - | Single Stim DEM file to use |
| `--dem-dir <DIR>` | - | Directory of DEM files (batch mode) |
| `--shots <N>` | 10000 | Syndrome samples per configuration |
| `--distances <D1,D2,...>` | 3,5,7 | Code distances to test |
| `--error-probs <P1,P2,...>` | 0.1% to 1% | Physical error rates |
| `--seed <N>` | 42 | Random seed for reproducibility |
| `--no-verify` | false | Skip verification (faster but no correctness check) |
| `--csv` | false | Output results in CSV format |
| `--threshold-study` | false | Use denser error rate sweep |
| `--min-errors <N>` | 100 | Minimum logical errors before stopping |
| `--max-shots <N>` | 1000000 | Maximum shots per data point |
| `--color-code` | false | Run triangular color code benchmark |
| `--dual-decode` | false | Separate X/Z basis decoding |
| `--streaming` | false | Run streaming decoder benchmark (sliding window) |
| `--mode-2d` | false | Single measurement round (depth=1) |
| `--helios` | false | Helios-compatible benchmark (d=13, p=0.1%) |
| `--quick-bench <D>` | - | Quick single-point benchmark at distance D |

### Examples

**Save results to CSV:**
```bash
cargo run --release -p prav-circuit-bench -- --csv > results.csv
```

**High-statistics run:**
```bash
cargo run --release -p prav-circuit-bench -- \
    --shots 100000 \
    --min-errors 500 \
    --distances 3,5,7,9,11
```

**Use specific DEM file:**
```bash
cargo run --release -p prav-circuit-bench -- \
    --dem path/to/model.dem \
    --shots 50000
```

---

## Understanding the Output

### Sample Output

```
3D Circuit-Level Threshold Study: prav Union-Find Decoder
======================================================================

Distance d=3, 2x2x3 grid, 3 rounds
  p=0.0010: LER=1.20e-04/rnd [8.50e-05,1.70e-04], n=10000, t=0.15µs
  p=0.0030: LER=4.80e-04/rnd [3.90e-04,5.90e-04], n=10000, t=0.18µs
  p=0.0050: LER=9.50e-04/rnd [8.20e-04,1.10e-03], n=10000, t=0.21µs

Distance d=5, 4x4x5 grid, 5 rounds
  p=0.0010: LER=2.40e-05/rnd [1.50e-05,3.80e-05], n=10000, t=0.45µs
  p=0.0030: LER=1.80e-04/rnd [1.40e-04,2.30e-04], n=10000, t=0.52µs
  p=0.0050: LER=5.20e-04/rnd [4.40e-04,6.10e-04], n=10000, t=0.58µs

Error Suppression Factor Λ (= ε_d / ε_{d+2}):

       p      Λ(3→5)
--------------------
  0.0010  5.00±1.20
  0.0030  2.67±0.45
  0.0050  1.83±0.28

Benchmark complete.
```

### Field Explanations

**Header line:**
- `Distance d=5`: Code distance being tested
- `4x4x5 grid`: Detector grid dimensions (width × height × depth)
- `5 rounds`: Number of measurement rounds

**Per-error-rate line:**
- `p=0.0030`: Physical error probability (0.3%)
- `LER=1.80e-04/rnd`: Logical error rate per round (1.8 × 10⁻⁴)
- `[1.40e-04,2.30e-04]`: 95% confidence interval
- `n=10000`: Number of shots sampled
- `t=0.52µs`: Average decode time in microseconds

**Lambda table:**
- `Λ(3→5)`: Error suppression going from d=3 to d=5
- `2.67±0.45`: Lambda value with uncertainty
- **Λ > 1 means we're below threshold** (good!)

---

## Threshold Studies

### What is a Threshold?

The **threshold** is the critical physical error rate. Below it, increasing the code distance reduces logical errors exponentially. Above it, larger codes actually perform worse.

```
    Logical Error Rate vs Physical Error Rate:

    LER
     │
     │    d=3
     │      \
     │       \   d=5
     │        \ /
     │         X ← threshold
     │        / \
     │       /   \
     │      d=5   d=3
     └──────────────────► p
              p_th
```

For the surface code with circuit-level noise, the threshold is around **0.5-1%** depending on the decoder.

### How to Run a Threshold Study

**Option 1: Using Stim DEMs (recommended)**

```bash
# Generate DEM files for various distances and error rates
make generate-dems

# Run the threshold study
make threshold-study

# Get CSV output for analysis
make threshold-study-csv
```

**Option 2: Using phenomenological model**

```bash
cargo run --release -p prav-circuit-bench -- --threshold-study
```

### Interpreting Results

Look at the Lambda table:

| Physical Error Rate | Λ(3→5) | Λ(5→7) | Interpretation |
|---------------------|--------|--------|----------------|
| 0.1% | 5.2 | 4.8 | Far below threshold |
| 0.3% | 2.1 | 1.9 | Below threshold |
| 0.5% | 1.3 | 1.1 | Near threshold |
| 0.7% | 0.9 | 0.8 | Above threshold |
| 1.0% | 0.6 | 0.5 | Far above threshold |

The threshold is where Λ crosses 1.0.

---

## Color Code Benchmarking

Run triangular color code benchmarks using the restriction decoder approach:

```bash
# Default color code benchmark (d=3,5,7, p=1%)
cargo run --release -p prav-circuit-bench -- --color-code

# Specific distances and error rates
cargo run --release -p prav-circuit-bench -- --color-code --distances 5,7,9 --error-probs 0.005,0.01

# With CSV output
cargo run --release -p prav-circuit-bench -- --color-code --csv > color_code_results.csv
```

The color code decoder uses three parallel Union-Find decoders (one per color class: Red, Green, Blue) and combines their results using the restriction decoder approach.

**Output includes:**
- Decode latency (total and per-color breakdown)
- Logical error rates
- Defect distribution by color class
- Error suppression factor (Lambda)

---

## Dual X/Z Decoding

For realistic fault-tolerant QEC, decode X and Z error bases separately:

```bash
# Enable dual decoding mode
cargo run --release -p prav-circuit-bench -- --dual-decode

# With specific configuration
cargo run --release -p prav-circuit-bench -- --dual-decode --distances 5,7,9 --shots 20000

# CSV output with dual metrics
cargo run --release -p prav-circuit-bench -- --dual-decode --csv > dual_results.csv
```

**How it works:**
1. Splits the unified syndrome into X-only and Z-only components using `SyndromeSplitter`
2. Decodes each basis with an independent `DynDecoder`
3. Reports separate X and Z timing and logical error rates

**CSV output uses `DUAL_CSV_HEADER` with columns:**
- `distance`, `physical_p`, `rounds`, `shots`
- `x_logical_errors`, `z_logical_errors`, `combined_logical_errors`
- `x_ler_per_round`, `z_ler_per_round`, `combined_ler_per_round`
- `x_decode_us`, `z_decode_us`

---

## Streaming Decoder Benchmarking

The streaming decoder processes syndromes round-by-round with a sliding window, enabling real-time QEC:

```bash
# Default streaming benchmark
cargo run --release -p prav-circuit-bench -- --streaming

# Specific distances and error rates
cargo run --release -p prav-circuit-bench -- --streaming --distances 5,7,9,13 --error-probs 0.001,0.003,0.005,0.01

# With CSV output
cargo run --release -p prav-circuit-bench -- --streaming --csv > streaming_results.csv
```

### Architecture

```
Round N arrives:
┌─────────────────────────────────────────────────────┐
│ Sliding Window (size W)                             │
│  ┌───────┬───────┬───────┬───────┐                 │
│  │ R(N-W)│  ...  │ R(N-1)│  R(N) │  ← New round    │
│  │ EXIT  │       │       │ LOAD  │                  │
│  └───┬───┴───────┴───────┴───────┘                 │
│      │                                              │
│      ▼                                              │
│  Commit corrections for R(N-W)                      │
└─────────────────────────────────────────────────────┘
```

**How it works:**
1. Syndromes arrive round-by-round as measurements complete
2. Each round is ingested into the sliding window
3. When the window is full, the oldest round is committed (corrections extracted)
4. At stream end, remaining rounds are flushed

**Output includes:**
- **Ingest latency**: Time to load one round's syndromes and grow clusters
- **Commit latency**: Time to extract corrections when round exits window
- **Flush latency**: Time to commit remaining rounds at stream end
- **Memory usage**: Bytes allocated for the streaming decoder
- **Per-round latency**: Total processing time per measurement round

**Key properties:**
- Circular Z-indexing eliminates data copying when window slides
- Corrections committed only when rounds exit window (guaranteed correctness)
- Arena-only allocation maintains deterministic timing
- Suitable for FPGA/embedded systems with limited memory

---

## Using Stim DEM Files

### What is Stim?

[Stim](https://github.com/quantumlib/Stim) is a fast Clifford circuit simulator. It can model realistic quantum circuits with various noise models and export **Detector Error Models (DEMs)**.

A DEM describes:
- Which detectors exist and their coordinates
- Which error mechanisms exist
- The probability of each error
- Which detectors each error affects
- Which logical observables each error affects

### Generating DEM Files

We provide a script to generate DEMs for rotated surface codes:

```bash
# Using the Makefile
make generate-dems

# Or directly
python prav-circuit-bench/scripts/generate_dems.py \
    --distances 3 5 7 9 11 \
    --noise-levels 0.001 0.002 0.003 0.004 0.005 \
    --output-dir prav-circuit-bench/dems
```

### DEM File Format

DEM files are text files with this structure:

```
detector(0, 0, 0) D0
detector(1, 0, 0) D1
detector(0, 1, 0) D2
detector(1, 1, 0) D3

error(0.001) D0 D1
error(0.001) D0 D2
error(0.002) D0 ^ L0
```

- `detector(x, y, t) D<id>`: Declares a detector with coordinates
- `error(p) D<id1> D<id2> ...`: An error with probability p affecting listed detectors
- `^ L<id>`: This error also affects logical observable L<id>

### Using a Single DEM File

```bash
cargo run --release -p prav-circuit-bench -- \
    --dem path/to/surface_d5_r5_p0.0010.dem \
    --shots 50000
```

### Using a Directory of DEM Files

```bash
cargo run --release -p prav-circuit-bench -- \
    --dem-dir prav-circuit-bench/dems \
    --shots 10000
```

The tool automatically parses filenames to extract distance, rounds, and noise level. Expected format: `surface_d{distance}_r{rounds}_p{noise}.dem`

---

## Architecture Overview

### Module Structure

```
prav-circuit-bench/
├── src/
│   ├── main.rs              # CLI, benchmarking loop
│   ├── dual_decoder.rs      # Dual X/Z basis decoding
│   ├── color_code_bench.rs  # Color code benchmarking
│   ├── streaming_bench.rs   # Streaming decoder benchmarking
│   ├── stats.rs             # Statistics (ThresholdPoint, DualThresholdPoint, LatencyStats)
│   ├── verification.rs      # Correction verification
│   ├── dem/
│   │   ├── mod.rs           # DEM module exports
│   │   ├── types.rs         # ParsedDem, OwnedErrorMechanism
│   │   └── parser.rs        # DEM file parser
│   ├── syndrome/
│   │   ├── mod.rs           # Syndrome module exports
│   │   ├── phenomenological.rs  # Built-in noise model
│   │   ├── circuit_sampler.rs   # Sample from DEM
│   │   └── splitter.rs          # X/Z syndrome splitting
│   └── surface_code/
│       ├── mod.rs           # Surface code module exports
│       └── detector_map.rs  # Coordinate mapping
└── scripts/
    └── generate_dems.py     # Stim DEM generation
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Sources                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐          ┌─────────────────┐          │
│  │ Phenomenological │          │   Stim DEM      │          │
│  │     Model       │          │     Files       │          │
│  └────────┬────────┘          └────────┬────────┘          │
│           │                            │                    │
│           │ generate_correlated_       │ CircuitSampler    │
│           │ syndromes()                │ .sample()         │
│           │                            │                    │
│           └────────────┬───────────────┘                    │
│                        │                                    │
│                        ▼                                    │
│              ┌─────────────────┐                            │
│              │ SyndromeWith    │                            │
│              │   Logical       │                            │
│              │ - syndrome bits │                            │
│              │ - logical_flips │                            │
│              └────────┬────────┘                            │
│                       │                                     │
└───────────────────────│─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                       Decoder                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    ┌────────────────────────────────────────────────┐      │
│    │              prav Union-Find                    │      │
│    │  - load_dense_syndromes()                      │      │
│    │  - decode() → Vec<EdgeCorrection>              │      │
│    │  - reset_for_next_cycle()                      │      │
│    └────────────────────────────────────────────────┘      │
│                                                             │
└───────────────────────│─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Verification                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    verify_with_logical()                                    │
│    - Apply corrections to syndrome                          │
│    - Check all defects resolved                             │
│    - Track boundary corrections → predicted logical         │
│    - Compare predicted vs actual logical_flips              │
│                                                             │
└───────────────────────│─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                     Statistics                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    ThresholdPoint                                           │
│    - LER per round with Wilson CI                           │
│    - Decode latency percentiles                             │
│                                                             │
│    SuppressionFactor (Λ)                                    │
│    - Ratio of LER between adjacent distances                │
│    - Error propagation for uncertainty                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Types

| Type | Location | Purpose |
|------|----------|---------|
| `ParsedDem` | `dem/types.rs` | Holds parsed DEM data: detectors, error mechanisms |
| `OwnedErrorMechanism` | `dem/types.rs` | Single error: probability, affected detectors, logical |
| `SyndromeWithLogical` | `syndrome/` | Syndrome bits + ground-truth logical flips |
| `SplitSyndromes` | `syndrome/splitter.rs` | Split X/Z syndromes with compact indexing |
| `SyndromeSplitter` | `syndrome/splitter.rs` | Splits unified syndrome by X/Z basis |
| `CircuitSampler` | `syndrome/circuit_sampler.rs` | Samples syndromes from a DEM |
| `DetectorMapper` | `surface_code/detector_map.rs` | Maps DEM coordinates to prav indices |
| `VerificationResult` | `verification.rs` | Defects resolved + predicted logical |
| `ThresholdPoint` | `stats.rs` | Results for one (distance, error_rate) configuration |
| `DualThresholdPoint` | `stats.rs` | Results for dual X/Z decoding with separate metrics |
| `LatencyStats` | `stats.rs` | Timing percentiles (avg, p50, p95, p99) |
| `SuppressionFactor` | `stats.rs` | Lambda between two distances |
| `DualDecoderConfig` | `dual_decoder.rs` | Configuration for dual X/Z decoder |
| `ColorCodeBenchConfig` | `color_code_bench.rs` | Configuration for color code benchmarks |
| `StreamingBenchConfig` | `streaming_bench.rs` | Configuration for streaming decoder benchmarks |
| `StreamingThresholdPoint` | `streaming_bench.rs` | Results with ingest/commit latency metrics |

---

## Glossary

**Bit-flip error (X error)**: An error that flips a qubit from |0⟩ to |1⟩ or vice versa.

**Code distance (d)**: The minimum number of physical errors needed to cause a logical error. Larger distance = better protection.

**Decoder**: An algorithm that takes a syndrome and outputs corrections.

**Defect**: A detector that triggered (-1 measurement).

**Detector**: A measurement location in space-time. Maps to a stabilizer measurement at a specific round.

**Detector Error Model (DEM)**: A description of all error mechanisms, their probabilities, and which detectors they affect.

**Error suppression factor (Lambda, Λ)**: The ratio of logical error rates between code distances. Λ > 1 means we're below threshold.

**Logical error**: When the decoder's correction changes the encoded quantum information. The syndrome is resolved, but the data is corrupted.

**Logical Error Rate (LER)**: The probability of a logical error per shot or per round.

**Logical qubit**: The protected quantum information encoded across many physical qubits.

**MWPM (Minimum-Weight Perfect Matching)**: The optimal decoding algorithm. Finds the most likely error pattern. Slow: O(n³).

**Phenomenological noise**: A simplified error model with independent space-like and time-like errors.

**Phase-flip error (Z error)**: An error that changes the phase of a superposition: α|0⟩ + β|1⟩ → α|0⟩ - β|1⟩.

**Physical error rate (p)**: The probability of an error on a single physical qubit per gate or per time step.

**Physical qubit**: An actual qubit in the hardware.

**Rotated surface code**: A variant of the surface code where the grid is rotated 45°, using fewer qubits for the same distance.

**Stabilizer**: A multi-qubit measurement that detects errors without measuring the logical qubit. Returns +1 (no error) or -1 (error detected).

**Stim**: A fast quantum circuit simulator that can model stabilizer circuits with noise.

**Surface code**: A 2D topological error-correcting code. Uses a grid of qubits with nearest-neighbor interactions.

**Syndrome**: The pattern of triggered detectors. Tells us where errors occurred (but not exactly which errors).

**Threshold**: The critical physical error rate. Below threshold, larger codes are better. Above threshold, larger codes are worse.

**Union-Find**: A fast approximate decoding algorithm. Near-linear time complexity. Used in prav.

---

## See Also

- [prav main repository](../) - The Union-Find decoder implementation
- [Stim](https://github.com/quantumlib/Stim) - Quantum circuit simulator
- [PyMatching](https://github.com/oscarhiggott/PyMatching) - MWPM decoder for comparison

---

## References

1. Fowler, A. G., et al. "Surface codes: Towards practical large-scale quantum computation." Physical Review A 86.3 (2012): 032324.

2. Delfosse, N., and N. H. Nickerson. "Almost-linear time decoding algorithm for topological codes." Quantum 5 (2021): 595.

3. Dennis, E., et al. "Topological quantum memory." Journal of Mathematical Physics 43.9 (2002): 4452-4505.
