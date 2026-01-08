"""Type stubs for prav._prav native module."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

class TopologyType:
    """
    Supported grid topologies for QEC decoding.

    Available topologies:
    - Square: 4-neighbor square lattice (surface codes)
    - Triangular: 6-neighbor triangular lattice (color codes)
    - Honeycomb: 3-neighbor honeycomb lattice
    - Grid3D: 6-neighbor 3D cubic lattice
    """

    Square: TopologyType
    Triangular: TopologyType
    Honeycomb: TopologyType
    Grid3D: TopologyType

    def __new__(
        cls,
        name: Literal["square", "triangular", "honeycomb", "3d", "grid3d"] = "square",
    ) -> TopologyType: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class Decoder:
    """
    Union Find decoder for quantum error correction.

    Supports 2D surface codes, triangular codes, honeycomb codes, and 3D codes.

    Parameters
    ----------
    width : int
        Grid width in nodes. Must be > 0.
    height : int
        Grid height in nodes. Must be > 0.
    topology : str, optional
        Grid topology: 'square' (default), 'triangular', 'honeycomb', or '3d'.
    depth : int, optional
        Grid depth for 3D codes (default: 1). Must be > 0.

    Raises
    ------
    ValueError
        If width, height, or depth is 0, or if topology is invalid.

    Examples
    --------
    >>> import prav
    >>> import numpy as np
    >>> decoder = prav.Decoder(17, 17)
    >>> syndromes = np.zeros(8, dtype=np.uint64)
    >>> syndromes[0] = 0b11
    >>> corrections = decoder.decode(syndromes)
    """

    @property
    def width(self) -> int:
        """Grid width in nodes."""
        ...

    @property
    def height(self) -> int:
        """Grid height in nodes."""
        ...

    @property
    def depth(self) -> int:
        """Grid depth (1 for 2D codes)."""
        ...

    @property
    def topology(self) -> TopologyType:
        """Grid topology type."""
        ...

    def __new__(
        cls,
        width: int,
        height: int,
        topology: Literal[
            "square", "triangular", "honeycomb", "3d", "grid3d"
        ] = "square",
        depth: int = 1,
    ) -> Decoder: ...
    def decode(self, syndromes: NDArray[np.uint64]) -> NDArray[np.uint32]:
        """
        Decode syndromes and return edge corrections.

        Parameters
        ----------
        syndromes : np.ndarray[np.uint64]
            Dense bitpacked syndrome array. Each u64 represents 64 nodes.
            Bit i set means node (block_index * 64 + i) has a syndrome.

        Returns
        -------
        np.ndarray[np.uint32]
            Correction edges as flat array [u0, v0, u1, v1, ...].
            v=0xFFFFFFFF indicates a boundary correction.
        """
        ...

    def reset(self) -> None:
        """
        Reset decoder state for next decoding cycle.

        This is called automatically after decode(), but can be called
        manually if needed.
        """
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

def required_buffer_size(width: int, height: int, depth: int = 1) -> int:
    """
    Calculate required buffer size for a decoder.

    Parameters
    ----------
    width : int
        Grid width in nodes. Must be > 0.
    height : int
        Grid height in nodes. Must be > 0.
    depth : int, optional
        Grid depth for 3D codes (default: 1). Must be > 0.

    Returns
    -------
    int
        Required buffer size in bytes.

    Raises
    ------
    ValueError
        If width, height, or depth is 0.
    """
    ...
