"""Property-based tests using Hypothesis."""

from __future__ import annotations

import numpy as np
import prav
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.conftest import make_syndromes, verify_corrections_resolve_defects


class TestDecoderProperties:
    """Property-based tests for decoder correctness."""

    @given(
        width=st.integers(min_value=3, max_value=30),
        height=st.integers(min_value=3, max_value=30),
        num_defects=st.integers(min_value=0, max_value=20),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=50, deadline=10000)
    def test_decoder_always_resolves_defects(
        self, width: int, height: int, num_defects: int, seed: int
    ) -> None:
        """Decoder should always resolve all defects."""
        decoder = prav.Decoder(width, height)

        # Generate random defect positions
        rng = np.random.default_rng(seed)
        defects = [
            (int(rng.integers(0, width)), int(rng.integers(0, height)))
            for _ in range(num_defects)
        ]

        syndromes = make_syndromes(width, height, defects)
        corrections = decoder.decode(syndromes)

        assert verify_corrections_resolve_defects(syndromes, corrections)

    @given(
        width=st.integers(min_value=3, max_value=20),
        height=st.integers(min_value=3, max_value=20),
    )
    @settings(max_examples=30, deadline=5000)
    def test_empty_syndromes_produce_no_corrections(
        self, width: int, height: int
    ) -> None:
        """Empty syndromes should produce no corrections."""
        decoder = prav.Decoder(width, height)
        syndromes = make_syndromes(width, height, [])
        corrections = decoder.decode(syndromes)
        assert len(corrections) == 0

    @given(
        width=st.integers(min_value=3, max_value=20),
        height=st.integers(min_value=3, max_value=20),
    )
    @settings(max_examples=20, deadline=5000)
    def test_single_pair_produces_corrections(self, width: int, height: int) -> None:
        """Single defect pair should produce corrections."""
        decoder = prav.Decoder(width, height)
        x, y = width // 2, height // 2
        syndromes = make_syndromes(width, height, [(x, y), (x + 1, y)])
        corrections = decoder.decode(syndromes)
        assert len(corrections) > 0

    @given(topology=st.sampled_from(["square", "triangular", "honeycomb", "3d"]))
    @settings(max_examples=10, deadline=5000)
    def test_all_topologies_work(self, topology: str) -> None:
        """All topologies should successfully decode."""
        decoder = prav.Decoder(10, 10, topology=topology)
        syndromes = make_syndromes(10, 10, [(3, 3), (4, 3)])
        corrections = decoder.decode(syndromes)
        # Basic check: no exceptions thrown
        assert isinstance(corrections, np.ndarray)

    @given(
        width=st.integers(min_value=1, max_value=100),
        height=st.integers(min_value=1, max_value=100),
        depth=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=20, deadline=5000)
    def test_buffer_size_always_positive(
        self, width: int, height: int, depth: int
    ) -> None:
        """Buffer size should always be positive for valid inputs."""
        size = prav.required_buffer_size(width, height, depth)
        assert size > 0
