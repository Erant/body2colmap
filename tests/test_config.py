"""
Tests for configuration validation.
"""

import pytest

from body2colmap.config import (
    SkeletonConfig,
    _validate_eye_style,
    _validate_pupil_scale,
)


class TestEyeStyleValidation:
    """The eye_style escape hatch only accepts known styles."""

    def test_accepts_known_styles(self):
        """Both shipped styles pass."""
        for style in ("shape", "dots"):
            assert _validate_eye_style(style) == style

    def test_rejects_unknown_style(self):
        """A typo is an error rather than a silent fallback to the default."""
        with pytest.raises(ValueError, match="eye_style"):
            _validate_eye_style("blobs")

    def test_default_is_filled_shapes(self):
        """Filled eyes are the default; dots are opt-in."""
        assert SkeletonConfig().eye_style == "shape"


class TestPupilScaleValidation:
    """A pupil larger than the eye it sits in is rejected, not clamped."""

    def test_accepts_values_in_range(self):
        """Anything in (0, 1] is a pupil that fits inside the eye."""
        for value in (0.01, 0.5, 0.75, 1.0):
            assert _validate_pupil_scale(value) == value

    def test_rejects_above_one(self):
        """1.0 already spans the full eye height; more would spill past the lids."""
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            _validate_pupil_scale(1.01)

    def test_rejects_zero_and_negative(self):
        """A zero or negative pupil is not a shape."""
        for value in (0.0, -0.5):
            with pytest.raises(ValueError):
                _validate_pupil_scale(value)

    def test_default_is_in_range(self):
        """The shipped default passes its own validation."""
        default = SkeletonConfig().pupil_scale
        assert _validate_pupil_scale(default) == default
