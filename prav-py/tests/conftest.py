"""Pytest fixtures and test utilities for prav tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import prav
import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence


@pytest.fixture
def small_decoder() -> prav.Decoder:
    """Create a small 8x8 decoder for basic tests."""
    return prav.Decoder(8, 8)


@pytest.fixture
def medium_decoder() -> prav.Decoder:
    """Create a medium 17x17 decoder (standard surface code)."""
    return prav.Decoder(17, 17)


@pytest.fixture
def large_decoder() -> prav.Decoder:
    """Create a large 33x33 decoder."""
    return prav.Decoder(33, 33)


def make_syndromes(
    width: int, height: int, defect_positions: Sequence[tuple[int, int]]
) -> np.ndarray:
    """
    Create syndrome array with defects at specified (x, y) positions.

    Uses power-of-2 stride matching prav-core's internal layout.

    Parameters
    ----------
    width : int
        Grid width.
    height : int
        Grid height.
    defect_positions : Sequence[tuple[int, int]]
        List of (x, y) positions where defects should be placed.

    Returns
    -------
    np.ndarray
        Dense bitpacked syndrome array (uint64).
    """
    max_dim = max(width, height)
    stride = 1 << (max_dim - 1).bit_length()
    alloc_size = height * stride
    num_blocks = (alloc_size + 63) // 64

    syndromes = np.zeros(num_blocks, dtype=np.uint64)

    for x, y in defect_positions:
        idx = y * stride + x
        block = idx // 64
        bit = idx % 64
        syndromes[block] |= np.uint64(1) << np.uint64(bit)

    return syndromes


def verify_corrections_resolve_defects(
    syndromes: np.ndarray,
    corrections: np.ndarray,
) -> bool:
    """
    Verify that corrections properly resolve all defects.

    Applies corrections to syndromes and checks that all defects are resolved
    (XOR to zero).

    Parameters
    ----------
    syndromes : np.ndarray
        Original syndrome array.
    corrections : np.ndarray
        Correction edges as flat array [u0, v0, u1, v1, ...].

    Returns
    -------
    bool
        True if all defects are resolved, False otherwise.
    """
    # Count original defects
    original = sum(bin(int(x)).count("1") for x in syndromes)

    if original == 0:
        return True

    result = syndromes.copy()
    num_corrections = len(corrections) // 2

    for i in range(num_corrections):
        u = int(corrections[2 * i])
        v = int(corrections[2 * i + 1])

        # Toggle bit at u
        if u != 0xFFFFFFFF:
            block_u = u // 64
            bit_u = u % 64
            if block_u < len(result):
                result[block_u] ^= np.uint64(1) << np.uint64(bit_u)

        # Toggle bit at v
        if v != 0xFFFFFFFF:
            block_v = v // 64
            bit_v = v % 64
            if block_v < len(result):
                result[block_v] ^= np.uint64(1) << np.uint64(bit_v)

    remaining = sum(bin(int(x)).count("1") for x in result)
    return remaining == 0
