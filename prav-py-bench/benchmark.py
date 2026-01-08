#!/usr/bin/env python3
"""
Benchmark comparison: prav vs PyMatching

Compares decode latency on a 17x17 square grid surface code.
Outputs timing percentiles (avg, p50, p95, p99) in console tables.

PyMatching is configured with proper weights based on error probability
as recommended in the official documentation.
"""

import argparse
import time
from typing import List, Dict, Any, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from tabulate import tabulate

import prav
from pymatching import Matching

from syndrome_generator import (
    generate_syndromes_prav,
    prav_to_pymatching,
    count_defects_prav,
)


def create_surface_code_graph(width: int, height: int) -> Tuple[csr_matrix, int]:
    """
    Create a simple surface code matching graph for PyMatching.

    This creates a parity check matrix where each check (node) is
    connected to its neighboring data qubits (edges).

    Parameters
    ----------
    width : int
        Grid width.
    height : int
        Grid height.

    Returns
    -------
    Tuple[csr_matrix, int]
        (Parity check matrix H, number of edges)
    """
    num_nodes = width * height

    rows = []
    cols = []
    edge_idx = 0

    # Horizontal edges (between horizontally adjacent nodes)
    for y in range(height):
        for x in range(width - 1):
            node1 = y * width + x
            node2 = y * width + x + 1
            rows.extend([node1, node2])
            cols.extend([edge_idx, edge_idx])
            edge_idx += 1

    # Vertical edges (between vertically adjacent nodes)
    for y in range(height - 1):
        for x in range(width):
            node1 = y * width + x
            node2 = (y + 1) * width + x
            rows.extend([node1, node2])
            cols.extend([edge_idx, edge_idx])
            edge_idx += 1

    # Boundary edges - left and right
    for y in range(height):
        # Left boundary
        node = y * width
        rows.append(node)
        cols.append(edge_idx)
        edge_idx += 1

        # Right boundary
        node = y * width + (width - 1)
        rows.append(node)
        cols.append(edge_idx)
        edge_idx += 1

    # Boundary edges - top and bottom
    for x in range(width):
        # Top boundary
        node = x
        rows.append(node)
        cols.append(edge_idx)
        edge_idx += 1

        # Bottom boundary
        node = (height - 1) * width + x
        rows.append(node)
        cols.append(edge_idx)
        edge_idx += 1

    data = np.ones(len(rows), dtype=np.uint8)
    H = csr_matrix((data, (rows, cols)), shape=(num_nodes, edge_idx))

    return H, edge_idx


def create_pymatching_decoder(
    width: int, height: int, error_prob: float
) -> Matching:
    """
    Create a properly configured PyMatching decoder.

    Uses the recommended weight formula: weight = log((1-p)/p)
    where p is the error probability.

    Parameters
    ----------
    width : int
        Grid width.
    height : int
        Grid height.
    error_prob : float
        Error probability per edge.

    Returns
    -------
    Matching
        Configured PyMatching decoder.
    """
    H, num_edges = create_surface_code_graph(width, height)

    # Compute weights as log-likelihood ratios (PyMatching recommendation)
    # weight = log((1-p)/p) for error probability p
    # Clamp p to avoid log(0) or log(inf)
    p = np.clip(error_prob, 1e-10, 1 - 1e-10)
    weight = np.log((1 - p) / p)
    weights = np.ones(num_edges) * weight

    # Create decoder using from_check_matrix with weights
    matcher = Matching.from_check_matrix(H, weights=weights)

    return matcher


def benchmark_prav(
    decoder: prav.Decoder,
    syndromes_list: List[np.ndarray],
    warmup: int = 100,
) -> Tuple[List[float], int]:
    """
    Benchmark prav decoder.

    Parameters
    ----------
    decoder : prav.Decoder
        The decoder instance.
    syndromes_list : List[np.ndarray]
        List of syndrome arrays.
    warmup : int
        Number of warmup iterations.

    Returns
    -------
    Tuple[List[float], int]
        (per_decode_times_us, total_corrections)
    """
    # Warmup - ensure JIT/cache is warm
    for i in range(min(warmup, len(syndromes_list))):
        decoder.decode(syndromes_list[i])

    times_us = []
    total_corrections = 0

    for syndromes in syndromes_list:
        start = time.perf_counter()
        corrections = decoder.decode(syndromes)
        end = time.perf_counter()

        times_us.append((end - start) * 1_000_000)  # Convert to microseconds
        total_corrections += len(corrections) // 2

    return times_us, total_corrections


def benchmark_pymatching(
    matcher: Matching,
    syndromes_list: List[np.ndarray],
    warmup: int = 100,
) -> Tuple[List[float], int]:
    """
    Benchmark PyMatching decoder with individual decode calls.

    Parameters
    ----------
    matcher : Matching
        The PyMatching instance.
    syndromes_list : List[np.ndarray]
        List of syndrome arrays in PyMatching format.
    warmup : int
        Number of warmup iterations.

    Returns
    -------
    Tuple[List[float], int]
        (per_decode_times_us, total_corrections)
    """
    # Warmup - ensure C++ representation is cached (per PyMatching docs)
    for i in range(min(warmup, len(syndromes_list))):
        matcher.decode(syndromes_list[i])

    times_us = []
    total_corrections = 0

    for syndromes in syndromes_list:
        start = time.perf_counter()
        corrections = matcher.decode(syndromes)
        end = time.perf_counter()

        times_us.append((end - start) * 1_000_000)  # Convert to microseconds
        total_corrections += int(np.sum(corrections))

    return times_us, total_corrections


def benchmark_pymatching_batch(
    matcher: Matching,
    syndromes_batch: np.ndarray,
    warmup: int = 100,
) -> Tuple[float, int]:
    """
    Benchmark PyMatching decoder with batch decoding (decode_batch).

    This is the recommended way to use PyMatching for throughput.

    Parameters
    ----------
    matcher : Matching
        The PyMatching instance.
    syndromes_batch : np.ndarray
        2D array of syndromes (num_shots x num_nodes).
    warmup : int
        Number of warmup shots.

    Returns
    -------
    Tuple[float, int]
        (total_time_us, total_corrections)
    """
    # Warmup with a small batch
    if warmup > 0 and len(syndromes_batch) > warmup:
        matcher.decode_batch(syndromes_batch[:warmup])

    start = time.perf_counter()
    corrections = matcher.decode_batch(syndromes_batch)
    end = time.perf_counter()

    total_time_us = (end - start) * 1_000_000
    total_corrections = int(np.sum(corrections))

    return total_time_us, total_corrections


def calculate_percentiles(times_us: List[float]) -> Dict[str, float]:
    """
    Calculate latency percentiles.

    Parameters
    ----------
    times_us : List[float]
        List of decode times in microseconds.

    Returns
    -------
    Dict[str, float]
        Dictionary with avg, p50, p95, p99.
    """
    arr = np.array(times_us)
    return {
        "avg": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def run_benchmark(
    width: int = 17,
    height: int = 17,
    error_probs: List[float] = None,
    num_shots: int = 10000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Run benchmark comparison.

    Parameters
    ----------
    width : int
        Grid width.
    height : int
        Grid height.
    error_probs : List[float]
        List of error probabilities to test.
    num_shots : int
        Number of shots per error rate.
    seed : int
        Random seed.

    Returns
    -------
    List[Dict[str, Any]]
        List of result dictionaries.
    """
    if error_probs is None:
        error_probs = [0.001, 0.003, 0.006, 0.01, 0.03, 0.06]

    results = []

    # Initialize prav decoder (reused across error rates)
    print(f"Initializing decoders for {width}x{height} grid...")
    prav_decoder = prav.Decoder(width, height, topology="square")

    print(f"Running {num_shots:,} shots per error rate...\n")

    for p in error_probs:
        print(f"  Error rate {p:.3f}... ", end="", flush=True)

        # Create PyMatching decoder with proper weights for this error rate
        # Per PyMatching docs: weight = log((1-p)/p)
        pymatching_decoder = create_pymatching_decoder(width, height, p)

        # Generate syndromes
        prav_syndromes = generate_syndromes_prav(
            width, height, p, num_shots, seed + int(p * 1e6)
        )

        # Convert to PyMatching format
        pm_syndromes = [
            prav_to_pymatching(s, width, height) for s in prav_syndromes
        ]

        # Also create batch array for decode_batch
        pm_syndromes_batch = np.array(pm_syndromes, dtype=np.uint8)

        # Count total defects
        total_defects = sum(count_defects_prav(s) for s in prav_syndromes)

        # Benchmark prav (individual decodes for latency)
        prav_times, prav_corrections = benchmark_prav(
            prav_decoder, prav_syndromes
        )
        prav_stats = calculate_percentiles(prav_times)

        # Benchmark PyMatching (individual decodes for latency comparison)
        pm_times, pm_corrections = benchmark_pymatching(
            pymatching_decoder, pm_syndromes
        )
        pm_stats = calculate_percentiles(pm_times)

        # Also benchmark PyMatching with decode_batch for throughput comparison
        pm_batch_time, pm_batch_corrections = benchmark_pymatching_batch(
            pymatching_decoder, pm_syndromes_batch
        )
        pm_batch_avg = pm_batch_time / num_shots

        result = {
            "p": p,
            "defects": total_defects,
            "prav": prav_stats,
            "prav_corrections": prav_corrections,
            "pymatching": pm_stats,
            "pm_corrections": pm_corrections,
            "pm_batch_avg": pm_batch_avg,
            "pm_batch_corrections": pm_batch_corrections,
        }
        results.append(result)

        print(f"done (prav avg: {prav_stats['avg']:.2f}us, PM avg: {pm_stats['avg']:.2f}us, PM batch: {pm_batch_avg:.2f}us)")

    return results


def print_results(results: List[Dict[str, Any]], width: int, height: int, num_shots: int):
    """Print results as formatted tables."""
    print(f"\n{'='*80}")
    print(f"Benchmark: {width}x{height} Square Grid | {num_shots:,} shots per error rate")
    print("PyMatching configured with weights = log((1-p)/p) per documentation")
    print("="*80)

    # prav latencies table
    print("\nprav Latencies (microseconds):")
    prav_headers = ["Error Rate", "Defects", "avg", "p50", "p95", "p99"]
    prav_rows = []
    for r in results:
        prav_rows.append([
            f"{r['p']:.3f}",
            f"{r['defects']:,}",
            f"{r['prav']['avg']:.2f}",
            f"{r['prav']['p50']:.2f}",
            f"{r['prav']['p95']:.2f}",
            f"{r['prav']['p99']:.2f}",
        ])
    print(tabulate(prav_rows, headers=prav_headers, tablefmt="grid"))

    # PyMatching latencies table (individual decode calls)
    print("\nPyMatching Latencies - decode() (microseconds):")
    pm_headers = ["Error Rate", "Defects", "avg", "p50", "p95", "p99"]
    pm_rows = []
    for r in results:
        pm_rows.append([
            f"{r['p']:.3f}",
            f"{r['defects']:,}",
            f"{r['pymatching']['avg']:.2f}",
            f"{r['pymatching']['p50']:.2f}",
            f"{r['pymatching']['p95']:.2f}",
            f"{r['pymatching']['p99']:.2f}",
        ])
    print(tabulate(pm_rows, headers=pm_headers, tablefmt="grid"))

    # PyMatching batch latencies
    print("\nPyMatching Latencies - decode_batch() (microseconds per shot):")
    batch_headers = ["Error Rate", "Defects", "avg"]
    batch_rows = []
    for r in results:
        batch_rows.append([
            f"{r['p']:.3f}",
            f"{r['defects']:,}",
            f"{r['pm_batch_avg']:.2f}",
        ])
    print(tabulate(batch_rows, headers=batch_headers, tablefmt="grid"))

    # Speedup table (vs individual decode)
    print("\nSpeedup vs PyMatching decode():")
    speedup_headers = ["Error Rate", "avg", "p50", "p95", "p99"]
    speedup_rows = []
    for r in results:
        avg_speedup = r['pymatching']['avg'] / r['prav']['avg'] if r['prav']['avg'] > 0 else 0
        p50_speedup = r['pymatching']['p50'] / r['prav']['p50'] if r['prav']['p50'] > 0 else 0
        p95_speedup = r['pymatching']['p95'] / r['prav']['p95'] if r['prav']['p95'] > 0 else 0
        p99_speedup = r['pymatching']['p99'] / r['prav']['p99'] if r['prav']['p99'] > 0 else 0
        speedup_rows.append([
            f"{r['p']:.3f}",
            f"{avg_speedup:.2f}x",
            f"{p50_speedup:.2f}x",
            f"{p95_speedup:.2f}x",
            f"{p99_speedup:.2f}x",
        ])
    print(tabulate(speedup_rows, headers=speedup_headers, tablefmt="grid"))

    # Speedup vs decode_batch
    print("\nSpeedup vs PyMatching decode_batch():")
    batch_speedup_headers = ["Error Rate", "avg"]
    batch_speedup_rows = []
    for r in results:
        batch_speedup = r['pm_batch_avg'] / r['prav']['avg'] if r['prav']['avg'] > 0 else 0
        batch_speedup_rows.append([
            f"{r['p']:.3f}",
            f"{batch_speedup:.2f}x",
        ])
    print(tabulate(batch_speedup_rows, headers=batch_speedup_headers, tablefmt="grid"))

    # Correction counts
    print("\nCorrection Counts:")
    corr_headers = ["Error Rate", "prav", "PyMatching"]
    corr_rows = []
    for r in results:
        corr_rows.append([
            f"{r['p']:.3f}",
            f"{r['prav_corrections']:,}",
            f"{r['pm_corrections']:,}",
        ])
    print(tabulate(corr_rows, headers=corr_headers, tablefmt="grid"))

    # Summary
    avg_speedups = []
    batch_speedups = []
    for r in results:
        if r['prav']['avg'] > 0:
            avg_speedups.append(r['pymatching']['avg'] / r['prav']['avg'])
            batch_speedups.append(r['pm_batch_avg'] / r['prav']['avg'])

    if avg_speedups:
        overall_avg_speedup = sum(avg_speedups) / len(avg_speedups)
        overall_batch_speedup = sum(batch_speedups) / len(batch_speedups)
        print(f"\nOverall average speedup vs decode(): {overall_avg_speedup:.2f}x")
        print(f"Overall average speedup vs decode_batch(): {overall_batch_speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark prav vs PyMatching QEC decoders"
    )
    parser.add_argument(
        "--width", type=int, default=17,
        help="Grid width (default: 17)"
    )
    parser.add_argument(
        "--height", type=int, default=17,
        help="Grid height (default: 17)"
    )
    parser.add_argument(
        "--shots", type=int, default=10000,
        help="Number of shots per error rate (default: 10000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--error-probs", type=float, nargs="+",
        default=[0.001, 0.003, 0.006, 0.01, 0.03, 0.06],
        help="Error probabilities to test"
    )

    args = parser.parse_args()

    results = run_benchmark(
        width=args.width,
        height=args.height,
        error_probs=args.error_probs,
        num_shots=args.shots,
        seed=args.seed,
    )

    print_results(results, args.width, args.height, args.shots)


if __name__ == "__main__":
    main()
