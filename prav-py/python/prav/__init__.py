"""
prav - High-performance Union Find decoder for quantum error correction.

This module provides Python bindings for the prav-core Rust QEC decoder,
supporting surface codes, color codes, and other topological codes.

Example
-------
>>> import prav
>>> import numpy as np
>>> decoder = prav.Decoder(17, 17)
>>> syndromes = np.zeros(8, dtype=np.uint64)
>>> corrections = decoder.decode(syndromes)
"""

from prav._prav import Decoder, TopologyType, required_buffer_size

__version__ = "0.0.1"
__all__ = ["Decoder", "TopologyType", "required_buffer_size"]
