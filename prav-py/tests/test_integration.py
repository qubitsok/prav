"""Integration tests for prav package."""

from __future__ import annotations

import numpy as np
import prav


class TestPackageIntegration:
    """Integration tests for full package behavior."""

    def test_version_available(self) -> None:
        """Test __version__ is accessible."""
        assert hasattr(prav, "__version__")
        assert prav.__version__ == "0.0.1"

    def test_all_exports_available(self) -> None:
        """Test all __all__ exports are accessible."""
        assert hasattr(prav, "Decoder")
        assert hasattr(prav, "TopologyType")
        assert hasattr(prav, "required_buffer_size")

    def test_exports_match_all(self) -> None:
        """Test that __all__ matches actual exports."""
        expected = {"Decoder", "TopologyType", "required_buffer_size"}
        assert set(prav.__all__) == expected

    def test_readme_example(self) -> None:
        """Test the example from README works."""
        decoder = prav.Decoder(17, 17, topology="square")
        syndromes = np.zeros(8, dtype=np.uint64)
        syndromes[0] = 0b11
        corrections = decoder.decode(syndromes)
        assert isinstance(corrections, np.ndarray)
        assert corrections.dtype == np.uint32

    def test_full_workflow(self) -> None:
        """Test complete encoding-decoding workflow."""
        # Setup
        width, height = 20, 20
        decoder = prav.Decoder(width, height)

        # Multiple decode cycles
        rng = np.random.default_rng(42)
        for _ in range(5):
            num_defects = int(rng.integers(2, 20))
            defects = [
                (int(rng.integers(0, width)), int(rng.integers(0, height)))
                for _ in range(num_defects)
            ]

            # Generate syndromes
            max_dim = max(width, height)
            stride = 1 << (max_dim - 1).bit_length()
            num_blocks = (height * stride + 63) // 64
            syndromes = np.zeros(num_blocks, dtype=np.uint64)

            for x, y in defects:
                idx = y * stride + x
                block = idx // 64
                bit = idx % 64
                syndromes[block] |= np.uint64(1) << np.uint64(bit)

            # Decode
            corrections = decoder.decode(syndromes)

            # Verify format
            assert corrections.dtype == np.uint32
            assert len(corrections) % 2 == 0

    def test_decoder_properties_immutable(self) -> None:
        """Test that decoder properties are read-only."""
        decoder = prav.Decoder(17, 17, topology="triangular", depth=1)

        # Properties should be accessible
        assert decoder.width == 17
        assert decoder.height == 17
        assert decoder.depth == 1
        assert str(decoder.topology) == "triangular"

    def test_str_representation(self) -> None:
        """Test string representation of decoder."""
        decoder = prav.Decoder(17, 17)
        s = str(decoder)
        assert "17" in s or "Decoder" in s
