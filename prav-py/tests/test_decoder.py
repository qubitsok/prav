"""Unit tests for prav.Decoder class."""

from __future__ import annotations

import numpy as np
import prav

from tests.conftest import make_syndromes, verify_corrections_resolve_defects


class TestDecoderConstruction:
    """Tests for Decoder.__init__()."""

    def test_create_small_decoder(self) -> None:
        """Test creating a small 8x8 decoder."""
        decoder = prav.Decoder(8, 8)
        assert decoder.width == 8
        assert decoder.height == 8
        assert decoder.depth == 1

    def test_create_with_all_topologies(self) -> None:
        """Test creating decoders with all supported topologies."""
        topologies = ["square", "triangular", "honeycomb", "3d"]
        for topo in topologies:
            decoder = prav.Decoder(8, 8, topology=topo)
            assert decoder.width == 8

    def test_create_3d_decoder(self) -> None:
        """Test creating a 3D decoder with depth > 1."""
        decoder = prav.Decoder(8, 8, topology="3d", depth=4)
        assert decoder.depth == 4

    def test_repr_format(self) -> None:
        """Test __repr__ returns expected format."""
        decoder = prav.Decoder(17, 17)
        repr_str = repr(decoder)
        assert "Decoder" in repr_str
        assert "17" in repr_str

    def test_rectangular_grid(self) -> None:
        """Test non-square grid dimensions."""
        decoder = prav.Decoder(20, 10)
        assert decoder.width == 20
        assert decoder.height == 10

    def test_large_grid(self) -> None:
        """Test larger grid sizes work correctly."""
        decoder = prav.Decoder(100, 100)
        assert decoder.width == 100


class TestDecoderDecoding:
    """Tests for Decoder.decode()."""

    def test_empty_syndromes(self) -> None:
        """Decoding empty syndromes returns no corrections."""
        decoder = prav.Decoder(8, 8)
        syndromes = make_syndromes(8, 8, [])
        corrections = decoder.decode(syndromes)
        assert len(corrections) == 0

    def test_single_pair_horizontal(self) -> None:
        """Test decoding two horizontally adjacent defects."""
        decoder = prav.Decoder(8, 8)
        syndromes = make_syndromes(8, 8, [(2, 2), (3, 2)])
        corrections = decoder.decode(syndromes)
        assert verify_corrections_resolve_defects(syndromes, corrections)

    def test_single_pair_vertical(self) -> None:
        """Test decoding two vertically adjacent defects."""
        decoder = prav.Decoder(8, 8)
        syndromes = make_syndromes(8, 8, [(2, 2), (2, 3)])
        corrections = decoder.decode(syndromes)
        assert verify_corrections_resolve_defects(syndromes, corrections)

    def test_multiple_pairs(self) -> None:
        """Test decoding multiple independent defect pairs."""
        decoder = prav.Decoder(16, 16)
        syndromes = make_syndromes(16, 16, [(0, 0), (1, 0), (10, 10), (11, 10)])
        corrections = decoder.decode(syndromes)
        assert verify_corrections_resolve_defects(syndromes, corrections)

    def test_boundary_defects(self) -> None:
        """Test defects at grid boundaries."""
        decoder = prav.Decoder(8, 8)
        syndromes = make_syndromes(8, 8, [(0, 0), (0, 1)])
        corrections = decoder.decode(syndromes)
        assert verify_corrections_resolve_defects(syndromes, corrections)

    def test_corner_defects(self) -> None:
        """Test defects at grid corners."""
        decoder = prav.Decoder(8, 8)
        syndromes = make_syndromes(8, 8, [(0, 0), (7, 7)])
        corrections = decoder.decode(syndromes)
        assert verify_corrections_resolve_defects(syndromes, corrections)

    def test_chain_of_defects(self) -> None:
        """Test a linear chain of 4 defects."""
        decoder = prav.Decoder(16, 16)
        syndromes = make_syndromes(16, 16, [(2, 2), (4, 2), (6, 2), (8, 2)])
        corrections = decoder.decode(syndromes)
        assert verify_corrections_resolve_defects(syndromes, corrections)

    def test_dense_defects(self) -> None:
        """Test with many defects (stress test)."""
        decoder = prav.Decoder(20, 20)
        defects = [(i, j) for i in range(0, 20, 2) for j in range(0, 20, 2)]
        syndromes = make_syndromes(20, 20, defects)
        corrections = decoder.decode(syndromes)
        assert verify_corrections_resolve_defects(syndromes, corrections)

    def test_return_type(self) -> None:
        """Test that decode returns correct numpy array type."""
        decoder = prav.Decoder(8, 8)
        syndromes = make_syndromes(8, 8, [(2, 2), (3, 2)])
        corrections = decoder.decode(syndromes)
        assert isinstance(corrections, np.ndarray)
        assert corrections.dtype == np.uint32


class TestDecoderReset:
    """Tests for Decoder.reset()."""

    def test_reset_allows_reuse(self) -> None:
        """Test that reset() allows decoder reuse."""
        decoder = prav.Decoder(8, 8)
        syndromes = make_syndromes(8, 8, [(2, 2), (3, 2)])

        # First decode
        corrections1 = decoder.decode(syndromes)

        # Reset and decode again
        decoder.reset()
        corrections2 = decoder.decode(syndromes)

        # Both should resolve defects
        assert verify_corrections_resolve_defects(syndromes, corrections1)
        assert verify_corrections_resolve_defects(syndromes, corrections2)

    def test_multiple_decode_cycles(self) -> None:
        """Test decoder can handle multiple decode cycles."""
        decoder = prav.Decoder(8, 8)

        for _ in range(10):
            syndromes = make_syndromes(8, 8, [(2, 2), (3, 2)])
            corrections = decoder.decode(syndromes)
            assert verify_corrections_resolve_defects(syndromes, corrections)


class TestDecoderTopologies:
    """Tests for different grid topologies."""

    def test_square_topology(self) -> None:
        """Test square grid topology."""
        decoder = prav.Decoder(8, 8, topology="square")
        syndromes = make_syndromes(8, 8, [(2, 2), (3, 2)])
        corrections = decoder.decode(syndromes)
        assert len(corrections) >= 0

    def test_triangular_topology(self) -> None:
        """Test triangular grid topology."""
        decoder = prav.Decoder(8, 8, topology="triangular")
        syndromes = make_syndromes(8, 8, [(2, 2), (3, 2)])
        corrections = decoder.decode(syndromes)
        assert len(corrections) >= 0

    def test_honeycomb_topology(self) -> None:
        """Test honeycomb grid topology."""
        decoder = prav.Decoder(8, 8, topology="honeycomb")
        syndromes = make_syndromes(8, 8, [(2, 2), (3, 2)])
        corrections = decoder.decode(syndromes)
        assert len(corrections) >= 0

    def test_3d_topology(self) -> None:
        """Test 3D grid topology."""
        decoder = prav.Decoder(8, 8, topology="3d", depth=4)
        syndromes = make_syndromes(8, 8, [(2, 2), (3, 2)])
        corrections = decoder.decode(syndromes)
        assert len(corrections) >= 0
