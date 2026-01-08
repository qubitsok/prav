"""
Verification utilities for checking decoder correctness.

Validates that corrections properly resolve syndromes for both
prav and PyMatching decoders, proving feature parity.
"""

import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple, Dict, Any


def count_defects_prav(packed_syndromes: np.ndarray) -> int:
    """Count number of defects in prav-format syndrome array."""
    return sum(bin(int(x)).count("1") for x in packed_syndromes)


def count_defects_pymatching(syndromes: np.ndarray) -> int:
    """Count number of defects in PyMatching-format syndrome array."""
    return int(np.sum(syndromes))


def verify_prav_resolves_all(
    packed_syndromes: np.ndarray,
    corrections: np.ndarray,
    width: int,
    height: int,
) -> Tuple[bool, int, int]:
    """
    Verify that prav corrections resolve all defects.

    Applies corrections by XOR-toggling syndrome bits at edge endpoints.
    A valid matching should resolve all defects to zero.

    Parameters
    ----------
    packed_syndromes : np.ndarray[np.uint64]
        Original packed syndromes in prav format.
    corrections : np.ndarray[np.uint32]
        Correction pairs from decoder [u0, v0, u1, v1, ...].
        v=0xFFFFFFFF indicates a boundary correction.
    width : int
        Grid width.
    height : int
        Grid height.

    Returns
    -------
    Tuple[bool, int, int]
        (all_resolved, original_defects, remaining_defects)
    """
    # Count original defects
    original = count_defects_prav(packed_syndromes)

    # No defects means nothing to verify
    if original == 0:
        return True, 0, 0

    # Make a copy to modify
    result = packed_syndromes.copy()

    # Apply corrections by XOR-toggling endpoints
    num_corrections = len(corrections) // 2
    for i in range(num_corrections):
        u = int(corrections[2 * i])
        v = int(corrections[2 * i + 1])

        # Toggle bit at u (boundary marker = 0xFFFFFFFF)
        if u != 0xFFFFFFFF:
            block_u = u // 64
            bit_u = u % 64
            if block_u < len(result):
                result[block_u] ^= np.uint64(1) << np.uint64(bit_u)

        # Toggle bit at v (boundary marker = 0xFFFFFFFF)
        if v != 0xFFFFFFFF:
            block_v = v // 64
            bit_v = v % 64
            if block_v < len(result):
                result[block_v] ^= np.uint64(1) << np.uint64(bit_v)

    # Count remaining defects
    remaining = count_defects_prav(result)

    return remaining == 0, original, remaining


def verify_pymatching_resolves_all(
    syndromes: np.ndarray,
    corrections: np.ndarray,
    H: csr_matrix,
) -> Tuple[bool, int, int]:
    """
    Verify that PyMatching corrections resolve all defects.

    PyMatching returns edge corrections as a binary array. We verify by:
    1. Computing syndrome change = H @ corrections (mod 2)
    2. Applying change to original syndrome
    3. Checking all defects are resolved

    Parameters
    ----------
    syndromes : np.ndarray[bool/uint8]
        Original syndrome array in PyMatching format.
    corrections : np.ndarray
        Correction array from PyMatching (binary over edges).
    H : scipy.sparse.csr_matrix
        Parity check matrix used by PyMatching.

    Returns
    -------
    Tuple[bool, int, int]
        (all_resolved, original_defects, remaining_defects)
    """
    # Count original defects
    original = int(np.sum(syndromes))

    # No defects means nothing to verify
    if original == 0:
        return True, 0, 0

    # corrections is a binary array over edges
    # H @ corrections (mod 2) gives the syndrome change
    corrections_uint8 = corrections.astype(np.uint8)
    syndrome_change = np.array((H @ corrections_uint8) % 2).flatten()

    # Apply change to original syndrome (XOR)
    syndromes_uint8 = syndromes.astype(np.uint8)
    remaining_syndrome = (syndromes_uint8 ^ syndrome_change) % 2

    # Count remaining defects
    remaining = int(np.sum(remaining_syndrome))

    return remaining == 0, original, remaining


def verify_feature_parity(
    prav_resolved: bool,
    pm_resolved: bool,
    original_defects: int,
) -> Dict[str, Any]:
    """
    Compare that both decoders achieve the same result.

    Feature parity means both decoders either:
    - Both resolve all defects successfully, OR
    - Both fail to resolve (shouldn't happen with valid input)

    Note: The decoders may produce different corrections (different
    valid matchings exist), but the end result should be the same.

    Parameters
    ----------
    prav_resolved : bool
        Whether prav resolved all defects.
    pm_resolved : bool
        Whether PyMatching resolved all defects.
    original_defects : int
        Number of original defects.

    Returns
    -------
    Dict[str, Any]
        Comparison metrics.
    """
    return {
        "original_defects": original_defects,
        "prav_resolved": prav_resolved,
        "pymatching_resolved": pm_resolved,
        "feature_parity": prav_resolved == pm_resolved,
        "both_succeeded": prav_resolved and pm_resolved,
    }


def count_corrections_prav(corrections: np.ndarray) -> int:
    """Count number of corrections in prav output."""
    return len(corrections) // 2


def count_corrections_pymatching(corrections: np.ndarray) -> int:
    """Count number of corrections in PyMatching output."""
    return int(np.sum(corrections))
