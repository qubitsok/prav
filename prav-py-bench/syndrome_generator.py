"""
Syndrome generation utilities for benchmarking.

Generates random syndromes in both prav format (bitpacked u64) and
PyMatching format (boolean array).
"""

import numpy as np
from typing import List, Tuple


def generate_syndromes_prav(
    width: int,
    height: int,
    error_prob: float,
    num_shots: int,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Generate random syndromes for prav decoder.

    Uses the same format as prav-core: dense bitpacked u64 arrays
    with power-of-2 stride.

    Parameters
    ----------
    width : int
        Grid width in nodes.
    height : int
        Grid height in nodes.
    error_prob : float
        Probability of defect at each node.
    num_shots : int
        Number of syndrome samples to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    List[np.ndarray[np.uint64]]
        List of syndrome arrays in prav format.
    """
    rng = np.random.default_rng(seed)

    # Calculate stride (power of 2)
    max_dim = max(width, height)
    stride = 1 << (max_dim - 1).bit_length()

    # Calculate number of u64 blocks needed
    alloc_size = height * stride
    num_blocks = (alloc_size + 63) // 64

    syndromes_list = []

    for _ in range(num_shots):
        # Generate random defects
        defects = rng.random((height, width)) < error_prob

        # Pack into u64 array
        packed = np.zeros(num_blocks, dtype=np.uint64)

        for y in range(height):
            for x in range(width):
                if defects[y, x]:
                    idx = y * stride + x
                    block = idx // 64
                    bit = idx % 64
                    packed[block] |= np.uint64(1) << np.uint64(bit)

        syndromes_list.append(packed)

    return syndromes_list


def generate_syndromes_pymatching(
    width: int,
    height: int,
    error_prob: float,
    num_shots: int,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Generate random syndromes for PyMatching decoder.

    Uses boolean array format with row-major layout.

    Parameters
    ----------
    width : int
        Grid width in nodes.
    height : int
        Grid height in nodes.
    error_prob : float
        Probability of defect at each node.
    num_shots : int
        Number of syndrome samples to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    List[np.ndarray[bool]]
        List of syndrome arrays in PyMatching format.
    """
    rng = np.random.default_rng(seed)

    syndromes_list = []

    for _ in range(num_shots):
        # Generate random defects as flat boolean array
        defects = rng.random(width * height) < error_prob
        syndromes_list.append(defects)

    return syndromes_list


def prav_to_pymatching(
    packed: np.ndarray, width: int, height: int
) -> np.ndarray:
    """
    Convert prav-format syndromes to PyMatching format.

    Parameters
    ----------
    packed : np.ndarray[np.uint64]
        Packed syndrome array in prav format.
    width : int
        Grid width.
    height : int
        Grid height.

    Returns
    -------
    np.ndarray[bool]
        Boolean syndrome array for PyMatching.
    """
    max_dim = max(width, height)
    stride = 1 << (max_dim - 1).bit_length()

    result = np.zeros(width * height, dtype=bool)

    for y in range(height):
        for x in range(width):
            idx = y * stride + x
            block = idx // 64
            bit = idx % 64
            if block < len(packed) and (packed[block] & (np.uint64(1) << np.uint64(bit))):
                result[y * width + x] = True

    return result


def count_defects_prav(packed: np.ndarray) -> int:
    """Count number of defects in prav-format syndrome array."""
    return sum(bin(int(x)).count("1") for x in packed)


def count_defects_pymatching(syndromes: np.ndarray) -> int:
    """Count number of defects in PyMatching-format syndrome array."""
    return int(np.sum(syndromes))


def generate_paired_syndromes(
    width: int,
    height: int,
    error_prob: float,
    num_shots: int,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate paired syndromes for both prav and PyMatching.

    Both lists contain equivalent data but in different formats.

    Parameters
    ----------
    width : int
        Grid width in nodes.
    height : int
        Grid height in nodes.
    error_prob : float
        Probability of defect at each node.
    num_shots : int
        Number of syndrome samples to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[List[np.ndarray], List[np.ndarray]]
        (prav_syndromes, pymatching_syndromes)
    """
    prav_syndromes = generate_syndromes_prav(width, height, error_prob, num_shots, seed)
    pymatching_syndromes = [
        prav_to_pymatching(s, width, height) for s in prav_syndromes
    ]
    return prav_syndromes, pymatching_syndromes
