"""Tests for required_buffer_size function."""

from __future__ import annotations

import prav


class TestRequiredBufferSize:
    """Tests for required_buffer_size()."""

    def test_small_grid(self) -> None:
        """Test buffer size for small grid."""
        size = prav.required_buffer_size(8, 8)
        assert size > 0

    def test_medium_grid(self) -> None:
        """Test buffer size for medium grid."""
        size = prav.required_buffer_size(17, 17)
        assert size > 0

    def test_large_grid(self) -> None:
        """Test buffer size for large grid."""
        size = prav.required_buffer_size(100, 100)
        assert size > 0

    def test_3d_grid(self) -> None:
        """Test buffer size for 3D grid."""
        size_2d = prav.required_buffer_size(8, 8, 1)
        size_3d = prav.required_buffer_size(8, 8, 4)
        assert size_3d > size_2d

    def test_larger_grid_needs_more_memory(self) -> None:
        """Test larger grids need more memory."""
        small = prav.required_buffer_size(8, 8)
        large = prav.required_buffer_size(32, 32)
        assert large > small

    def test_default_depth_is_one(self) -> None:
        """Test default depth=1."""
        explicit = prav.required_buffer_size(8, 8, 1)
        implicit = prav.required_buffer_size(8, 8)
        assert explicit == implicit

    def test_rectangular_grid(self) -> None:
        """Test buffer size for rectangular grid."""
        size = prav.required_buffer_size(20, 10)
        assert size > 0

    def test_minimum_grid(self) -> None:
        """Test buffer size for minimum 1x1 grid."""
        size = prav.required_buffer_size(1, 1)
        assert size > 0
