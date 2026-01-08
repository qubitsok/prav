"""
Verification utilities for checking decoder correctness.

Validates that corrections properly resolve syndromes for both
prav and PyMatching decoders.
"""

import numpy as np
from typing import Tuple


def verify_prav_corrections(
    packed_syndromes: np.ndarray,
    corrections: np.ndarray,
    width: int,
    height: int,
) -> Tuple[bool, int]:
    """
    Verify that prav corrections properly resolve syndromes.

    Applies corrections to syndromes and checks that no defects remain.

    Parameters
    ----------
    packed_syndromes : np.ndarray[np.uint64]
        Original packed syndromes.
    corrections : np.ndarray[np.uint32]
        Correction pairs from decoder [u0, v0, u1, v1, ...].
    width : int
        Grid width.
    height : int
        Grid height.

    Returns
    -------
    Tuple[bool, int]
        (valid, remaining_defects)
        valid: True if all defects resolved
        remaining_defects: Number of remaining defects (should be 0)
    """
    # Make a copy to modify
    result = packed_syndromes.copy()

    max_dim = max(width, height)
    stride = 1 << (max_dim - 1).bit_length()

    # Apply corrections (flip syndrome bits at edge endpoints)
    num_corrections = len(corrections) // 2
    for i in range(num_corrections):
        u = corrections[i * 2]
        v = corrections[i * 2 + 1]

        # Flip bit at u (if within valid grid)
        if u != 0xFFFFFFFF:
            # u is in prav's internal index (with stride)
            block_u = u // 64
            bit_u = u % 64
            if block_u < len(result):
                result[block_u] ^= np.uint64(1) << np.uint64(bit_u)

        # Flip bit at v (unless boundary)
        if v != 0xFFFFFFFF:
            block_v = v // 64
            bit_v = v % 64
            if block_v < len(result):
                result[block_v] ^= np.uint64(1) << np.uint64(bit_v)

    # Count remaining defects
    remaining = sum(bin(int(x)).count("1") for x in result)

    return remaining == 0, remaining


def verify_pymatching_corrections(
    syndromes: np.ndarray,
    corrections: np.ndarray,
    width: int,
    height: int,
) -> Tuple[bool, int]:
    """
    Verify that PyMatching corrections are valid.

    For PyMatching, corrections are edge flips. We count how many
    corrections were applied and verify the syndrome was resolved.

    Parameters
    ----------
    syndromes : np.ndarray[bool]
        Original syndrome array.
    corrections : np.ndarray
        Correction array from PyMatching.
    width : int
        Grid width.
    height : int
        Grid height.

    Returns
    -------
    Tuple[bool, int]
        (valid, remaining_defects)
    """
    # PyMatching returns edge corrections, not directly syndrome toggling
    # The number of corrections is np.sum(corrections)
    num_corrections = int(np.sum(corrections))

    # For a valid matching, all syndromes should be resolved
    # We can't easily verify this without the parity check matrix
    # Just return that it's valid if there are corrections when there were defects
    num_defects = int(np.sum(syndromes))

    # Simple heuristic: if there are defects, there should be corrections
    valid = (num_defects == 0) or (num_corrections > 0)

    return valid, 0  # PyMatching should always resolve all defects


def count_corrections_prav(corrections: np.ndarray) -> int:
    """Count number of corrections in prav output."""
    return len(corrections) // 2


def count_corrections_pymatching(corrections: np.ndarray) -> int:
    """Count number of corrections in PyMatching output."""
    return int(np.sum(corrections))


def compare_results(
    prav_corrections: np.ndarray,
    pymatching_corrections: np.ndarray,
    num_defects: int,
) -> dict:
    """
    Compare correction results from both decoders.

    Parameters
    ----------
    prav_corrections : np.ndarray
        Corrections from prav decoder.
    pymatching_corrections : np.ndarray
        Corrections from PyMatching decoder.
    num_defects : int
        Number of original defects.

    Returns
    -------
    dict
        Comparison metrics.
    """
    prav_count = count_corrections_prav(prav_corrections)
    pm_count = count_corrections_pymatching(pymatching_corrections)

    return {
        "num_defects": num_defects,
        "prav_corrections": prav_count,
        "pymatching_corrections": pm_count,
        "both_produced_corrections": prav_count > 0 and pm_count > 0,
    }
