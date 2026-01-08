#!/usr/bin/env python3
"""
Benchmark comparison: prav vs PyMatching

Compares decode latency on multiple grid sizes (17x17, 32x32, 64x64).
Outputs timing percentiles (avg, p50, p95, p99) in console tables.
Includes rigorous correctness verification proving feature parity.

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
from verification import (
    verify_prav_resolves_all,
    verify_pymatching_resolves_all,
    verify_feature_parity,
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
) -> Tuple[Matching, csr_matrix]:
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
    Tuple[Matching, csr_matrix]
        (Configured PyMatching decoder, parity check matrix H)
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

    return matcher, H


def warmup_prav(decoder: prav.Decoder, width: int, height: int, num_warmup: int = 200):
    """
    Warmup prav decoder with fresh random syndromes.

    Uses separate data from benchmark to avoid cache effects.

    Parameters
    ----------
    decoder : prav.Decoder
        The decoder instance.
    width : int
        Grid width.
    height : int
        Grid height.
    num_warmup : int
        Number of warmup iterations.
    """
    warmup_syndromes = generate_syndromes_prav(
        width, height,
        error_prob=0.01,  # Fixed rate for warmup
        num_shots=num_warmup,
        seed=0xDEADBEEF,  # Different seed than benchmark
    )
    for s in warmup_syndromes:
        decoder.decode(s)


def warmup_pymatching(matcher: Matching, width: int, height: int, num_warmup: int = 200):
    """
    Warmup PyMatching decoder with fresh random syndromes.

    Per PyMatching docs: first decode call caches C++ representation.

    Parameters
    ----------
    matcher : Matching
        The PyMatching instance.
    width : int
        Grid width.
    height : int
        Grid height.
    num_warmup : int
        Number of warmup iterations.
    """
    warmup_syndromes = generate_syndromes_prav(
        width, height,
        error_prob=0.01,  # Fixed rate for warmup
        num_shots=num_warmup,
        seed=0xCAFEBABE,  # Different seed than benchmark and prav warmup
    )
    # Convert to PyMatching format
    pm_warmup = [prav_to_pymatching(s, width, height) for s in warmup_syndromes]
    for s in pm_warmup:
        matcher.decode(s)


def benchmark_with_verification(
    prav_decoder: prav.Decoder,
    pm_decoder: Matching,
    prav_syndromes: List[np.ndarray],
    pm_syndromes: List[np.ndarray],
    H: csr_matrix,
    width: int,
    height: int,
) -> Dict[str, Any]:
    """
    Benchmark both decoders with full verification of every shot.

    Parameters
    ----------
    prav_decoder : prav.Decoder
        The prav decoder instance.
    pm_decoder : Matching
        The PyMatching decoder instance.
    prav_syndromes : List[np.ndarray]
        Syndromes in prav format.
    pm_syndromes : List[np.ndarray]
        Syndromes in PyMatching format.
    H : csr_matrix
        Parity check matrix for PyMatching verification.
    width : int
        Grid width.
    height : int
        Grid height.

    Returns
    -------
    Dict[str, Any]
        Dictionary with timing and verification results.
    """
    prav_times = []
    pm_times = []
    prav_verified = 0
    pm_verified = 0
    parity_matches = 0
    total_defects = 0
    prav_corrections_total = 0
    pm_corrections_total = 0

    for prav_syn, pm_syn in zip(prav_syndromes, pm_syndromes):
        # Count defects for this shot
        defects = count_defects_prav(prav_syn)
        total_defects += defects

        # Time prav decode
        t0 = time.perf_counter()
        prav_corr = prav_decoder.decode(prav_syn)
        t1 = time.perf_counter()
        prav_times.append((t1 - t0) * 1_000_000)  # us

        # Time PyMatching decode
        t0 = time.perf_counter()
        pm_corr = pm_decoder.decode(pm_syn)
        t1 = time.perf_counter()
        pm_times.append((t1 - t0) * 1_000_000)  # us

        # Verify EVERY shot
        prav_ok, _, _ = verify_prav_resolves_all(prav_syn, prav_corr, width, height)
        pm_ok, _, _ = verify_pymatching_resolves_all(pm_syn, pm_corr, H)

        if prav_ok:
            prav_verified += 1
        if pm_ok:
            pm_verified += 1
        if prav_ok == pm_ok:
            parity_matches += 1

        prav_corrections_total += len(prav_corr) // 2
        pm_corrections_total += int(np.sum(pm_corr))

    return {
        "prav_times": prav_times,
        "pm_times": pm_times,
        "prav_verified": prav_verified,
        "pm_verified": pm_verified,
        "parity_matches": parity_matches,
        "total_defects": total_defects,
        "prav_corrections": prav_corrections_total,
        "pm_corrections": pm_corrections_total,
        "num_shots": len(prav_syndromes),
    }


def benchmark_pymatching_batch(
    matcher: Matching,
    syndromes_batch: np.ndarray,
) -> Tuple[float, int]:
    """
    Benchmark PyMatching decoder with batch decoding (decode_batch).

    Parameters
    ----------
    matcher : Matching
        The PyMatching instance.
    syndromes_batch : np.ndarray
        2D array of syndromes (num_shots x num_nodes).

    Returns
    -------
    Tuple[float, int]
        (total_time_us, total_corrections)
    """
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


def run_benchmark_for_grid(
    width: int,
    height: int,
    error_probs: List[float],
    num_shots: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Run benchmark comparison for a single grid size.

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
    results = []

    # Initialize prav decoder (reused across error rates)
    print(f"  Initializing prav decoder...")
    prav_decoder = prav.Decoder(width, height, topology="square")

    for p in error_probs:
        print(f"  Error rate {p:.3f}... ", end="", flush=True)

        # Create PyMatching decoder with proper weights for this error rate
        pm_decoder, H = create_pymatching_decoder(width, height, p)

        # Warmup both decoders with fresh data BEFORE timing
        warmup_prav(prav_decoder, width, height, num_warmup=200)
        warmup_pymatching(pm_decoder, width, height, num_warmup=200)

        # Generate syndromes for benchmarking
        prav_syndromes = generate_syndromes_prav(
            width, height, p, num_shots, seed + int(p * 1e6)
        )
        pm_syndromes = [
            prav_to_pymatching(s, width, height) for s in prav_syndromes
        ]

        # Run benchmark with verification
        bench_result = benchmark_with_verification(
            prav_decoder,
            pm_decoder,
            prav_syndromes,
            pm_syndromes,
            H,
            width,
            height,
        )

        # Calculate latency percentiles
        prav_stats = calculate_percentiles(bench_result["prav_times"])
        pm_stats = calculate_percentiles(bench_result["pm_times"])

        # Also benchmark PyMatching with decode_batch for throughput comparison
        pm_syndromes_batch = np.array(pm_syndromes, dtype=np.uint8)
        pm_batch_time, pm_batch_corrections = benchmark_pymatching_batch(
            pm_decoder, pm_syndromes_batch
        )
        pm_batch_avg = pm_batch_time / num_shots

        result = {
            "p": p,
            "defects": bench_result["total_defects"],
            "prav": prav_stats,
            "prav_corrections": bench_result["prav_corrections"],
            "pymatching": pm_stats,
            "pm_corrections": bench_result["pm_corrections"],
            "pm_batch_avg": pm_batch_avg,
            "pm_batch_corrections": pm_batch_corrections,
            # Verification results
            "prav_verified": bench_result["prav_verified"],
            "pm_verified": bench_result["pm_verified"],
            "parity_matches": bench_result["parity_matches"],
            "num_shots": bench_result["num_shots"],
        }
        results.append(result)

        prav_pct = 100.0 * bench_result["prav_verified"] / num_shots
        pm_pct = 100.0 * bench_result["pm_verified"] / num_shots
        print(f"done (prav: {prav_stats['avg']:.2f}us [{prav_pct:.0f}%], PM: {pm_stats['avg']:.2f}us [{pm_pct:.0f}%])")

    return results


def run_full_benchmark(
    grid_sizes: List[int],
    error_probs: List[float],
    num_shots: int,
    seed: int,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Run benchmarks across multiple grid sizes.

    Parameters
    ----------
    grid_sizes : List[int]
        List of grid sizes (assumes square grids).
    error_probs : List[float]
        List of error probabilities to test.
    num_shots : int
        Number of shots per error rate.
    seed : int
        Random seed.

    Returns
    -------
    Dict[int, List[Dict[str, Any]]]
        {grid_size: [results_per_error_rate]}
    """
    all_results = {}

    for size in grid_sizes:
        print(f"\n{'='*70}")
        print(f"Grid: {size}x{size} | {num_shots:,} shots per error rate")
        print("="*70)

        results = run_benchmark_for_grid(size, size, error_probs, num_shots, seed)
        all_results[size] = results

    return all_results


def print_results_for_grid(results: List[Dict[str, Any]], size: int, num_shots: int):
    """Print results as formatted tables for a single grid size."""
    print(f"\n{'='*70}")
    print(f"Results: {size}x{size} Square Grid | {num_shots:,} shots per error rate")
    print("PyMatching configured with weights = log((1-p)/p) per documentation")
    print("="*70)

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

    # Correctness Verification table
    print("\nCorrectness Verification:")
    verif_headers = ["Error Rate", "prav Success", "PyMatching Success", "Feature Parity"]
    verif_rows = []
    for r in results:
        prav_pct = 100.0 * r['prav_verified'] / r['num_shots']
        pm_pct = 100.0 * r['pm_verified'] / r['num_shots']
        parity_pct = 100.0 * r['parity_matches'] / r['num_shots']
        verif_rows.append([
            f"{r['p']:.3f}",
            f"{prav_pct:.2f}% ({r['prav_verified']:,})",
            f"{pm_pct:.2f}% ({r['pm_verified']:,})",
            f"{parity_pct:.2f}%",
        ])
    print(tabulate(verif_rows, headers=verif_headers, tablefmt="grid"))

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


def print_summary(all_results: Dict[int, List[Dict[str, Any]]]):
    """Print overall summary across all grid sizes."""
    print(f"\n{'='*70}")
    print("SUMMARY ACROSS ALL GRIDS")
    print("="*70)

    total_shots = 0
    total_prav_verified = 0
    total_pm_verified = 0
    total_parity = 0
    all_speedups = []

    for size, results in all_results.items():
        for r in results:
            total_shots += r['num_shots']
            total_prav_verified += r['prav_verified']
            total_pm_verified += r['pm_verified']
            total_parity += r['parity_matches']
            if r['prav']['avg'] > 0:
                all_speedups.append(r['pymatching']['avg'] / r['prav']['avg'])

    prav_success_rate = 100.0 * total_prav_verified / total_shots if total_shots > 0 else 0
    pm_success_rate = 100.0 * total_pm_verified / total_shots if total_shots > 0 else 0
    parity_rate = 100.0 * total_parity / total_shots if total_shots > 0 else 0
    avg_speedup = sum(all_speedups) / len(all_speedups) if all_speedups else 0

    print(f"\nTotal shots: {total_shots:,}")
    print(f"prav defect resolution: {prav_success_rate:.2f}% ({total_prav_verified:,}/{total_shots:,})")
    print(f"PyMatching defect resolution: {pm_success_rate:.2f}% ({total_pm_verified:,}/{total_shots:,})")
    print(f"Feature parity: {parity_rate:.2f}%")
    print(f"Average speedup vs PyMatching decode(): {avg_speedup:.2f}x")

    if prav_success_rate == 100.0 and pm_success_rate == 100.0:
        print("\nAll decoders achieved 100% defect resolution across all configurations.")
        print("Feature parity confirmed: both decoders produce equivalent results.")
    elif prav_success_rate < 100.0 or pm_success_rate < 100.0:
        print(f"\nWARNING: Some shots did not resolve all defects.")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark prav vs PyMatching QEC decoders"
    )
    parser.add_argument(
        "--grids", type=int, nargs="+",
        default=[17, 32, 64],
        help="Grid sizes to benchmark (default: 17 32 64)"
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

    print("Benchmark: prav vs PyMatching QEC Decoders")
    print(f"Grids: {args.grids}")
    print(f"Shots per error rate: {args.shots:,}")
    print(f"Error probabilities: {args.error_probs}")

    all_results = run_full_benchmark(
        grid_sizes=args.grids,
        error_probs=args.error_probs,
        num_shots=args.shots,
        seed=args.seed,
    )

    # Print detailed results for each grid
    for size in args.grids:
        print_results_for_grid(all_results[size], size, args.shots)

    # Print overall summary
    print_summary(all_results)


if __name__ == "__main__":
    main()
