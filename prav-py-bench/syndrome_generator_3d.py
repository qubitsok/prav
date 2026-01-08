"""
3D syndrome generation utilities for phenomenological noise model.

Generates random syndromes for 3D cubic lattice decoding.
Supports both prav (bitpacked) and PyMatching (boolean array) formats.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class Grid3DConfig:
    """Configuration for 3D decoder grid."""

    width: int
    height: int
    depth: int
    stride_y: int
    stride_z: int

    @classmethod
    def from_dimensions(cls, width: int, height: int, depth: int) -> "Grid3DConfig":
        """Create config from grid dimensions."""
        max_dim = max(width, height, depth)
        stride_y = 1 << (max_dim - 1).bit_length() if max_dim > 1 else 1
        stride_z = stride_y * stride_y
        return cls(
            width=width,
            height=height,
            depth=depth,
            stride_y=stride_y,
            stride_z=stride_z,
        )

    def coord_to_linear(self, x: int, y: int, t: int) -> int:
        """Convert (x, y, t) coordinates to linear index."""
        return t * self.stride_z + y * self.stride_y + x

    def linear_to_coord(self, idx: int) -> Tuple[int, int, int]:
        """Convert linear index to (x, y, t) coordinates."""
        t = idx // self.stride_z
        rem = idx % self.stride_z
        y = rem // self.stride_y
        x = rem % self.stride_y
        return (x, y, t)

    def num_detectors(self) -> int:
        """Total number of detector nodes."""
        return self.width * self.height * self.depth

    def alloc_blocks(self) -> int:
        """Number of u64 blocks needed for syndrome storage."""
        alloc_size = self.stride_z * self.depth
        return (alloc_size + 63) // 64


def create_3d_matching_graph(config: Grid3DConfig) -> Tuple[csr_matrix, int]:
    """
    Create a 3D matching graph for PyMatching.

    Creates a parity check matrix where each node connects to its
    6 neighbors (up, down, left, right, front, back) and to boundary.

    Parameters
    ----------
    config : Grid3DConfig
        Grid configuration.

    Returns
    -------
    Tuple[csr_matrix, int]
        (Parity check matrix H, number of edges)
    """
    w, h, d = config.width, config.height, config.depth

    def node_idx(x: int, y: int, t: int) -> int:
        return t * w * h + y * w + x

    rows = []
    cols = []
    edge_idx = 0

    # Interior edges: connect adjacent nodes
    for t in range(d):
        for y in range(h):
            for x in range(w):
                n1 = node_idx(x, y, t)

                # X-direction edges (within same time slice)
                if x < w - 1:
                    n2 = node_idx(x + 1, y, t)
                    rows.extend([n1, n2])
                    cols.extend([edge_idx, edge_idx])
                    edge_idx += 1

                # Y-direction edges (within same time slice)
                if y < h - 1:
                    n2 = node_idx(x, y + 1, t)
                    rows.extend([n1, n2])
                    cols.extend([edge_idx, edge_idx])
                    edge_idx += 1

                # T-direction edges (between time slices)
                if t < d - 1:
                    n2 = node_idx(x, y, t + 1)
                    rows.extend([n1, n2])
                    cols.extend([edge_idx, edge_idx])
                    edge_idx += 1

    # Boundary edges (all 6 faces)
    # X boundaries
    for t in range(d):
        for y in range(h):
            # Left boundary (x=0)
            n = node_idx(0, y, t)
            rows.append(n)
            cols.append(edge_idx)
            edge_idx += 1
            # Right boundary (x=w-1)
            n = node_idx(w - 1, y, t)
            rows.append(n)
            cols.append(edge_idx)
            edge_idx += 1

    # Y boundaries
    for t in range(d):
        for x in range(w):
            # Top boundary (y=0)
            n = node_idx(x, 0, t)
            rows.append(n)
            cols.append(edge_idx)
            edge_idx += 1
            # Bottom boundary (y=h-1)
            n = node_idx(x, h - 1, t)
            rows.append(n)
            cols.append(edge_idx)
            edge_idx += 1

    # T boundaries
    for y in range(h):
        for x in range(w):
            # First time slice (t=0)
            n = node_idx(x, y, 0)
            rows.append(n)
            cols.append(edge_idx)
            edge_idx += 1
            # Last time slice (t=d-1)
            n = node_idx(x, y, d - 1)
            rows.append(n)
            cols.append(edge_idx)
            edge_idx += 1

    num_nodes = w * h * d
    data = np.ones(len(rows), dtype=np.uint8)
    H = csr_matrix((data, (rows, cols)), shape=(num_nodes, edge_idx))

    return H, edge_idx


def generate_3d_syndromes_prav(
    config: Grid3DConfig,
    error_prob: float,
    num_shots: int,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Generate random 3D syndromes for prav decoder.

    Parameters
    ----------
    config : Grid3DConfig
        Grid configuration.
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
    w, h, d = config.width, config.height, config.depth
    num_blocks = config.alloc_blocks()

    syndromes_list = []

    for _ in range(num_shots):
        # Generate random defects in 3D
        defects = rng.random((d, h, w)) < error_prob

        # Pack into u64 array
        packed = np.zeros(num_blocks, dtype=np.uint64)

        for t in range(d):
            for y in range(h):
                for x in range(w):
                    if defects[t, y, x]:
                        idx = config.coord_to_linear(x, y, t)
                        block = idx // 64
                        bit = idx % 64
                        if block < num_blocks:
                            packed[block] |= np.uint64(1) << np.uint64(bit)

        syndromes_list.append(packed)

    return syndromes_list


def prav_to_pymatching_3d(
    packed: np.ndarray,
    config: Grid3DConfig,
) -> np.ndarray:
    """
    Convert prav-format 3D syndromes to PyMatching format.

    Parameters
    ----------
    packed : np.ndarray[np.uint64]
        Packed syndrome array in prav format.
    config : Grid3DConfig
        Grid configuration.

    Returns
    -------
    np.ndarray[bool]
        Boolean syndrome array for PyMatching.
    """
    w, h, d = config.width, config.height, config.depth
    result = np.zeros(w * h * d, dtype=bool)

    for t in range(d):
        for y in range(h):
            for x in range(w):
                prav_idx = config.coord_to_linear(x, y, t)
                block = prav_idx // 64
                bit = prav_idx % 64
                if block < len(packed) and (
                    packed[block] & (np.uint64(1) << np.uint64(bit))
                ):
                    pm_idx = t * w * h + y * w + x
                    result[pm_idx] = True

    return result


def generate_paired_3d_syndromes(
    config: Grid3DConfig,
    error_prob: float,
    num_shots: int,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate paired syndromes for both prav and PyMatching.

    Parameters
    ----------
    config : Grid3DConfig
        Grid configuration.
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
    prav_syndromes = generate_3d_syndromes_prav(config, error_prob, num_shots, seed)
    pymatching_syndromes = [prav_to_pymatching_3d(s, config) for s in prav_syndromes]
    return prav_syndromes, pymatching_syndromes


def count_defects_3d(syndrome: np.ndarray) -> int:
    """Count number of defects in prav-format 3D syndrome array."""
    return sum(bin(int(x)).count("1") for x in syndrome)


# Default error probabilities for 3D benchmarking
ERROR_PROBS_3D = [0.001, 0.003, 0.006, 0.01, 0.03, 0.06]
