"""
Tests for rendering helpers that do not need an OpenGL context.

Rendering itself requires a GPU/EGL context and mesh fixtures; these tests
cover the pure-numpy helpers around it.
"""

import numpy as np
import pytest

from body2colmap.renderer import _flat_color_rgba8


class TestFlatColorRGBA8:
    """Vertex colors for the ambient-only face pass."""

    def test_opaque(self):
        """Flat colors are always fully opaque."""
        assert _flat_color_rgba8((0.5, 0.5, 0.5))[3] == 255

    def test_black_and_white_are_exact(self):
        """The gamma curve fixes both endpoints."""
        assert list(_flat_color_rgba8((0.0, 0.0, 0.0))[:3]) == [0, 0, 0]
        assert list(_flat_color_rgba8((1.0, 1.0, 1.0))[:3]) == [255, 255, 255]

    def test_midtones_are_pre_darkened(self):
        """pyrender gamma-encodes its output, so mid-tones go in darker."""
        # 0.5 ** 2.2 = 0.2176 -> 55; without compensation this would be 128
        assert _flat_color_rgba8((0.5, 0.5, 0.5))[0] == 55

    def test_round_trips_through_the_shader_gamma(self):
        """Encoding then applying pow(c, 1/2.2) returns the requested color."""
        for channel in (0.15, 0.35, 0.5, 0.85, 0.95):
            encoded = _flat_color_rgba8((channel, channel, channel))[0] / 255.0
            rendered = encoded ** (1.0 / 2.2)
            assert rendered == pytest.approx(channel, abs=1.5 / 255)

    def test_channels_are_independent(self):
        """Each channel is converted on its own."""
        rgba = _flat_color_rgba8((1.0, 0.0, 0.5))
        assert list(rgba) == [255, 0, 55, 255]

    def test_clamps_out_of_range_input(self):
        """Values outside 0-1 do not wrap around or produce NaNs."""
        assert list(_flat_color_rgba8((-1.0, 2.0, 0.0))[:3]) == [0, 255, 0]

    def test_accepts_arrays(self):
        """Works on numpy colors as well as tuples."""
        assert np.array_equal(
            _flat_color_rgba8(np.array([1.0, 0.0, 0.5])),
            _flat_color_rgba8((1.0, 0.0, 0.5)),
        )
