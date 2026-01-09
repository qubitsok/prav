#!/usr/bin/env python3
"""Generate Stim DEM files for prav-circuit-bench threshold studies.

This script generates Detector Error Model (DEM) files using Stim's
circuit-level noise simulation for rotated surface codes.

Usage:
    python generate_dems.py --output-dir dems
    python generate_dems.py --distances 3 5 7 9 11 --noise-levels 0.001 0.002 0.003
"""

import argparse
from pathlib import Path

import stim


def generate_dem(
    distance: int,
    rounds: int,
    noise: float,
    output_dir: Path,
) -> Path:
    """Generate a rotated surface code DEM file.

    Parameters
    ----------
    distance : int
        Code distance.
    rounds : int
        Number of measurement rounds.
    noise : float
        Physical error rate (depolarizing noise strength).
    output_dir : Path
        Directory to write DEM files.

    Returns
    -------
    Path
        Path to the generated DEM file.
    """
    # Generate a rotated surface code memory-Z circuit with noise
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=noise,
        after_reset_flip_probability=noise,
        before_measure_flip_probability=noise,
        before_round_data_depolarization=noise,
    )

    # Extract detector error model (flattened to avoid repeat blocks)
    dem = circuit.detector_error_model(flatten_loops=True)

    # Write to file
    output_file = output_dir / f"surface_d{distance}_r{rounds}_p{noise:.4f}.dem"
    output_file.write_text(str(dem))

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate Stim DEM files for threshold studies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--distances",
        nargs="+",
        type=int,
        default=[3, 5, 7, 9, 11],
        help="Code distances to generate",
    )
    parser.add_argument(
        "--noise-levels",
        nargs="+",
        type=float,
        default=[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01],
        help="Physical error rates",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dems"),
        help="Output directory for DEM files",
    )
    parser.add_argument(
        "--rounds-factor",
        type=float,
        default=1.0,
        help="Multiply distance by this to get rounds (default: rounds = distance)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating DEMs for distances {args.distances}")
    print(f"Noise levels: {args.noise_levels}")
    print(f"Output directory: {args.output_dir}")
    print()

    total = len(args.distances) * len(args.noise_levels)
    count = 0

    for d in args.distances:
        for p in args.noise_levels:
            rounds = max(1, int(d * args.rounds_factor))
            dem_file = generate_dem(d, rounds, p, args.output_dir)
            count += 1
            print(f"[{count}/{total}] Generated: {dem_file.name}")

    print()
    print(f"Done. Generated {count} DEM files in {args.output_dir}")


if __name__ == "__main__":
    main()
