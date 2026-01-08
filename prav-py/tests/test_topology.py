"""Tests for TopologyType enum."""

from __future__ import annotations

import prav
import pytest


class TestTopologyType:
    """Tests for TopologyType."""

    def test_square_topology(self) -> None:
        """Test square topology creation."""
        topo = prav.TopologyType("square")
        assert str(topo) == "square"

    def test_triangular_topology(self) -> None:
        """Test triangular topology creation."""
        topo = prav.TopologyType("triangular")
        assert str(topo) == "triangular"

    def test_honeycomb_topology(self) -> None:
        """Test honeycomb topology creation."""
        topo = prav.TopologyType("honeycomb")
        assert str(topo) == "honeycomb"

    def test_3d_topology(self) -> None:
        """Test 3D topology creation."""
        topo = prav.TopologyType("3d")
        assert str(topo) == "3d"

    def test_grid3d_alias(self) -> None:
        """Test 'grid3d' is alias for '3d'."""
        topo = prav.TopologyType("grid3d")
        assert str(topo) == "3d"

    def test_case_insensitive(self) -> None:
        """Test topology names are case-insensitive."""
        topo = prav.TopologyType("SQUARE")
        assert str(topo) == "square"

    def test_repr_format(self) -> None:
        """Test __repr__ returns expected format."""
        topo = prav.TopologyType("square")
        assert "TopologyType" in repr(topo)

    def test_invalid_raises_value_error(self) -> None:
        """Test invalid topology name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown topology"):
            prav.TopologyType("invalid")

    def test_decoder_topology_property(self) -> None:
        """Test decoder.topology returns TopologyType."""
        decoder = prav.Decoder(8, 8, topology="triangular")
        assert isinstance(decoder.topology, prav.TopologyType)
        assert str(decoder.topology) == "triangular"
