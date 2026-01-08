"""Tests for input validation."""

from __future__ import annotations

import prav
import pytest


class TestDecoderValidation:
    """Tests for Decoder input validation."""

    def test_zero_width_raises(self) -> None:
        """Creating decoder with width=0 raises ValueError."""
        with pytest.raises(ValueError, match="width must be greater than 0"):
            prav.Decoder(0, 8)

    def test_zero_height_raises(self) -> None:
        """Creating decoder with height=0 raises ValueError."""
        with pytest.raises(ValueError, match="height must be greater than 0"):
            prav.Decoder(8, 0)

    def test_zero_depth_raises(self) -> None:
        """Creating decoder with depth=0 raises ValueError."""
        with pytest.raises(ValueError, match="depth must be greater than 0"):
            prav.Decoder(8, 8, depth=0)

    def test_invalid_topology_raises(self) -> None:
        """Creating decoder with invalid topology raises ValueError."""
        with pytest.raises(ValueError, match="Unknown topology"):
            prav.Decoder(8, 8, topology="invalid")

    def test_valid_minimum_dimensions(self) -> None:
        """Minimum valid dimensions (1x1) should work."""
        decoder = prav.Decoder(1, 1)
        assert decoder.width == 1
        assert decoder.height == 1


class TestBufferSizeValidation:
    """Tests for required_buffer_size validation."""

    def test_zero_width_raises(self) -> None:
        """Buffer size with width=0 raises ValueError."""
        with pytest.raises(ValueError, match="width must be greater than 0"):
            prav.required_buffer_size(0, 8, 1)

    def test_zero_height_raises(self) -> None:
        """Buffer size with height=0 raises ValueError."""
        with pytest.raises(ValueError, match="height must be greater than 0"):
            prav.required_buffer_size(8, 0, 1)

    def test_zero_depth_raises(self) -> None:
        """Buffer size with depth=0 raises ValueError."""
        with pytest.raises(ValueError, match="depth must be greater than 0"):
            prav.required_buffer_size(8, 8, 0)

    def test_valid_returns_positive(self) -> None:
        """Valid inputs return positive buffer size."""
        size = prav.required_buffer_size(8, 8, 1)
        assert size > 0
