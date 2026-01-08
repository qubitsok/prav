#!/usr/bin/env python3
"""
3D Phenomenological Benchmark: prav vs PyMatching

Compares Union Find (prav) vs MWPM (PyMatching) on 3D cubic lattice
with phenomenological noise model (random defects in space-time volume).

This tests the temporal/multi-round aspect of QEC decoding.

Metrics:
- Decode latency: avg, p50, p95, p99 (microseconds)
- Speedup factor (prav / PyMatching)
- Correctness verification (both decoders resolve all defects)
"""

import argparse
import time
from typing import List, Dict, Any, Tuple

import numpy as np
from tabulate import tabulate

import prav
from pymatching import Matching

from syndrome_generator_3d import (
    Grid3DConfig,
    create_3d_matching_graph,
    generate_paired_3d_syndromes,
    count_defects_3d,
    ERROR_PROBS_3D,
)


def create_pymatching_decoder_3d(
    config: Grid3DConfig,
    error_prob: float,
) -> Tuple[Matching, Any]:
    """
    Create a properly configured PyMatching decoder for 3D grid.

    Parameters
    ----------
    config : Grid3DConfig
        Grid configuration.
    error_prob : float
        Error probability per edge.

    Returns
    -------
    Tuple[Matching, csr_matrix]
        (Configured PyMatching decoder, parity check matrix H)
    """
    H, num_edges = create_3d_matching_graph(config)

    # Compute weights as log-likelihood ratios
    p = np.clip(error_prob, 1e-10, 1 - 1e-10)
    weight = np.log((1 - p) / p)
    weights = np.ones(num_edges) * weight

    matcher = Matching.from_check_matrix(H, weights=weights)

    return matcher, H


def warmup_prav_3d(
    decoder: prav.Decoder,
    config: Grid3DConfig,
    num_warmup: int = 200,
) -> None:
    """Warmup prav decoder with random syndromes."""
    warmup_syndromes, _ = generate_paired_3d_syndromes(
        config,
        error_prob=0.01,
        num_shots=num_warmup,
        seed=0xDEADBEEF,
    )
    for s in warmup_syndromes:
        decoder.decode(s)


def warmup_pymatching_3d(
    matcher: Matching,
    config: Grid3DConfig,
    num_warmup: int = 200,
) -> None:
    """Warmup PyMatching decoder with random syndromes."""
    _, warmup_syndromes = generate_paired_3d_syndromes(
        config,
        error_prob=0.01,
        num_shots=num_warmup,
        seed=0xCAFEBABE,
    )
    for s in warmup_syndromes:
        matcher.decode(s)


def verify_prav_3d(
    syndrome: np.ndarray,
    corrections: np.ndarray,
    config: Grid3DConfig,
) -> bool:
    """
    Verify prav corrections resolve all defects.

    XOR endpoint bits for each correction edge.
    """
    state = syndrome.copy()

    for i in range(len(corrections) // 2):
        u = int(corrections[2 * i])
        v = int(corrections[2 * i + 1])

        # Flip u
        if u != 0xFFFFFFFF:
            blk_u = u // 64
            bit_u = u % 64
            if blk_u < len(state):
                state[blk_u] ^= np.uint64(1) << np.uint64(bit_u)

        # Flip v
        if v != 0xFFFFFFFF:
            blk_v = v // 64
            bit_v = v % 64
            if blk_v < len(state):
                state[blk_v] ^= np.uint64(1) << np.uint64(bit_v)

    # Check all bits are zero
    return all(s == 0 for s in state)


def verify_pymatching_3d(
    syndrome: np.ndarray,
    corrections: np.ndarray,
    H: Any,
) -> bool:
    """
    Verify PyMatching corrections resolve all defects.

    Compute H @ corrections (mod 2) and XOR with syndrome.
    """
    syndrome_change = (H @ corrections) % 2
    # Handle both sparse and dense results
    if hasattr(syndrome_change, 'A1'):
        syndrome_change = syndrome_change.A1
    else:
        syndrome_change = np.asarray(syndrome_change).flatten()
    remaining = (syndrome.astype(int) ^ syndrome_change.astype(int)) % 2
    return np.sum(remaining) == 0


def benchmark_single_config_3d(
    width: int,
    height: int,
    depth: int,
    error_prob: float,
    num_shots: int,
    seed: int,
) -> Dict[str, Any]:
    """
    Run benchmark for a single 3D configuration.

    Returns
    -------
    Dict with timing and verification results.
    """
    config = Grid3DConfig.from_dimensions(width, height, depth)

    # Create decoders
    prav_decoder = prav.Decoder(
        width=width,
        height=height,
        topology="3d",
        depth=depth,
    )
    pm_decoder, H = create_pymatching_decoder_3d(config, error_prob)

    # Warmup
    warmup_prav_3d(prav_decoder, config, num_warmup=200)
    warmup_pymatching_3d(pm_decoder, config, num_warmup=200)

    # Generate syndromes
    prav_syndromes, pm_syndromes = generate_paired_3d_syndromes(
        config, error_prob, num_shots, seed
    )

    # Benchmark
    prav_times = []
    pm_times = []
    prav_verified = 0
    pm_verified = 0
    total_defects = 0

    for prav_syn, pm_syn in zip(prav_syndromes, pm_syndromes):
        defects = count_defects_3d(prav_syn)
        total_defects += defects

        # Time prav decode
        t0 = time.perf_counter()
        prav_corr = prav_decoder.decode(prav_syn)
        t1 = time.perf_counter()
        prav_times.append((t1 - t0) * 1_000_000)

        # Time PyMatching decode
        t0 = time.perf_counter()
        pm_corr = pm_decoder.decode(pm_syn)
        t1 = time.perf_counter()
        pm_times.append((t1 - t0) * 1_000_000)

        # Verify
        if verify_prav_3d(prav_syn, prav_corr, config):
            prav_verified += 1
        if verify_pymatching_3d(pm_syn, pm_corr, H):
            pm_verified += 1

    return {
        "prav_times": prav_times,
        "pm_times": pm_times,
        "prav_verified": prav_verified,
        "pm_verified": pm_verified,
        "total_defects": total_defects,
        "num_shots": num_shots,
    }


def calculate_percentiles(times_us: List[float]) -> Dict[str, float]:
    """Calculate latency percentiles."""
    arr = np.array(times_us)
    return {
        "avg": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def print_latency_table(
    decoder_name: str,
    results_by_prob: Dict[float, Dict[str, Any]],
) -> None:
    """Print latency table for a decoder."""
    headers = ["p", "Shots", "avg", "p50", "p95", "p99"]
    rows = []
    for prob, res in sorted(results_by_prob.items()):
        times = res["prav_times"] if "prav" in decoder_name.lower() else res["pm_times"]
        perc = calculate_percentiles(times)
        rows.append([
            f"{prob:.4f}",
            f"{res['num_shots']:,}",
            f"{perc['avg']:.2f}",
            f"{perc['p50']:.2f}",
            f"{perc['p95']:.2f}",
            f"{perc['p99']:.2f}",
        ])
    print(f"\n{decoder_name} Latencies (microseconds):")
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def print_speedup_table(
    results_by_prob: Dict[float, Dict[str, Any]],
) -> None:
    """Print speedup table (prav vs PyMatching)."""
    headers = ["p", "avg", "p50", "p95", "p99"]
    rows = []
    for prob, res in sorted(results_by_prob.items()):
        prav_perc = calculate_percentiles(res["prav_times"])
        pm_perc = calculate_percentiles(res["pm_times"])
        rows.append([
            f"{prob:.4f}",
            f"{pm_perc['avg'] / prav_perc['avg']:.2f}x",
            f"{pm_perc['p50'] / prav_perc['p50']:.2f}x",
            f"{pm_perc['p95'] / prav_perc['p95']:.2f}x",
            f"{pm_perc['p99'] / prav_perc['p99']:.2f}x",
        ])
    print("\nSpeedup (prav vs PyMatching):")
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def print_verification_table(
    results_by_prob: Dict[float, Dict[str, Any]],
) -> None:
    """Print verification results table."""
    headers = ["p", "Shots", "prav OK", "PM OK", "Defects"]
    rows = []
    for prob, res in sorted(results_by_prob.items()):
        rows.append([
            f"{prob:.4f}",
            f"{res['num_shots']:,}",
            f"{res['prav_verified']} ({100*res['prav_verified']/res['num_shots']:.1f}%)",
            f"{res['pm_verified']} ({100*res['pm_verified']/res['num_shots']:.1f}%)",
            f"{res['total_defects']:,}",
        ])
    print("\nVerification Results:")
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def run_benchmark_for_grid(
    width: int,
    height: int,
    depth: int,
    error_probs: List[float],
    num_shots: int,
    seed: int,
) -> Dict[float, Dict[str, Any]]:
    """Run benchmark for all error probabilities at a given grid size."""
    results = {}

    for prob in error_probs:
        print(f"  p={prob:.4f}...", end=" ", flush=True)
        res = benchmark_single_config_3d(
            width, height, depth, prob, num_shots, seed
        )
        results[prob] = res

        prav_perc = calculate_percentiles(res["prav_times"])
        pm_perc = calculate_percentiles(res["pm_times"])
        speedup = pm_perc["avg"] / prav_perc["avg"]
        print(
            f"prav: {prav_perc['avg']:.1f}us, "
            f"pm: {pm_perc['avg']:.1f}us, "
            f"speedup: {speedup:.2f}x"
        )

    return results


def print_summary(all_results: List[Dict[float, Dict[str, Any]]]) -> None:
    """Print overall summary."""
    total_shots = 0
    total_prav_ok = 0
    total_pm_ok = 0
    speedups = []

    for results in all_results:
        for prob, res in results.items():
            total_shots += res["num_shots"]
            total_prav_ok += res["prav_verified"]
            total_pm_ok += res["pm_verified"]
            prav_perc = calculate_percentiles(res["prav_times"])
            pm_perc = calculate_percentiles(res["pm_times"])
            speedups.append(pm_perc["avg"] / prav_perc["avg"])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total shots:         {total_shots:,}")
    print(f"prav verified:       {total_prav_ok:,} ({100*total_prav_ok/total_shots:.1f}%)")
    print(f"PyMatching verified: {total_pm_ok:,} ({100*total_pm_ok/total_shots:.1f}%)")
    print(f"Average speedup:     {np.mean(speedups):.2f}x")
    print("=" * 70)


def run_benchmark(args: argparse.Namespace) -> None:
    """Run the full benchmark suite."""
    print("3D Phenomenological Benchmark: prav vs PyMatching")
    print("=" * 70)
    print(f"Grid sizes: {args.grids}")
    print(f"Error probs: {args.error_probs}")
    print(f"Shots per config: {args.shots:,}")
    print(f"Seed: {args.seed}")
    print("=" * 70)

    all_results = []

    for size in args.grids:
        # Use cubic grid: size × size × size
        w = h = d = size
        num_detectors = w * h * d

        print(f"\nGrid: {w}×{h}×{d}, {num_detectors} nodes")
        print("-" * 70)

        results = run_benchmark_for_grid(
            w, h, d, args.error_probs, args.shots, args.seed
        )

        # Print tables for this grid
        print_latency_table("prav", results)
        print_latency_table("PyMatching", results)
        print_speedup_table(results)
        print_verification_table(results)

        all_results.append(results)

    # Print overall summary
    print_summary(all_results)


def main():
    parser = argparse.ArgumentParser(
        description="3D phenomenological benchmark: prav vs PyMatching",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--grids",
        type=int,
        nargs="+",
        default=[7, 11, 17],
        help="Grid sizes (cubic: NxNxN)",
    )

    parser.add_argument(
        "--error-probs",
        type=float,
        nargs="+",
        default=ERROR_PROBS_3D,
        help="Error probabilities",
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=10000,
        help="Number of shots per configuration",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
