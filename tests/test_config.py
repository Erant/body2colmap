"""
Tests for configuration validation.
"""

import pytest

from body2colmap.background import DEFAULT_RADIUS_SCALE
from body2colmap.config import (
    BackgroundConfig,
    Config,
    SkeletonConfig,
    _validate_background_geometry,
    _validate_eye_style,
    _validate_pupil_scale,
    create_argument_parser,
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


class TestBackgroundGeometryValidation:
    """Only the two surfaces the renderer knows how to intersect."""

    def test_accepts_the_known_surfaces(self):
        for value in ("sphere", "cube"):
            assert _validate_background_geometry(value) == value

    def test_rejects_anything_else(self):
        with pytest.raises(ValueError, match="Use 'sphere' or 'cube'"):
            _validate_background_geometry("dome")


class TestBackgroundConfigValidation:
    """The radius fields, which are easy to set into a contradiction."""

    def test_default_is_valid_and_disabled(self):
        config = BackgroundConfig()
        config.validate()
        assert config.enabled is False

    def test_default_is_a_grid_cube_at_a_finite_radius(self):
        """
        The three settings are a set. A cube is only a room -- corners, a
        floor and a ceiling that read apart -- at a finite radius; at infinity
        it flattens to a plain ruled field with no parallax, which is the weak
        version of the backdrop rather than the one measured to carry the cue.
        """
        config = BackgroundConfig(enabled=True)
        config.validate()
        assert config.texture == "grid"
        assert config.geometry == "cube"
        assert config.radius is None
        assert config.radius_scale == DEFAULT_RADIUS_SCALE

    def test_infinity_is_still_reachable(self):
        """Nulling both forms is the escape hatch back to a distant sky."""
        config = BackgroundConfig(
            enabled=True, texture="blender_sky", geometry="sphere",
            radius=None, radius_scale=None,
        )
        config.validate()
        assert config.radius is None and config.radius_scale is None

    def test_both_radius_forms_rejected(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            BackgroundConfig(radius=5.0, radius_scale=3.0).validate()

    def test_the_conflict_names_the_default_that_caused_it(self):
        """
        Since radius_scale is defaulted, the likeliest way to hit this is
        setting only a radius in Python. The message has to say so, or it
        reads as a complaint about a field the caller never touched.
        """
        with pytest.raises(ValueError, match="radius_scale defaults to"):
            BackgroundConfig(radius=5.0).validate()

    def test_radius_scale_must_exceed_one(self):
        """At or below 1.0 the orbit would pass through the backdrop."""
        for value in (1.0, 0.5):
            with pytest.raises(ValueError, match="stays inside"):
                BackgroundConfig(radius_scale=value).validate()

    def test_non_positive_radius_rejected(self):
        with pytest.raises(ValueError, match="must be > 0"):
            BackgroundConfig(radius=0.0).validate()

    def test_tiny_resolution_rejected(self):
        with pytest.raises(ValueError, match="must be >= 8"):
            BackgroundConfig(resolution=4).validate()


class TestBackgroundYamlRadius:
    """
    How a config file interacts with the defaulted radius_scale. The file is
    the one place a user writes `radius` without writing anything else, so the
    default has to get out of the way there.
    """

    @staticmethod
    def _background(body, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(body)
        return Config.from_yaml(str(path), input_file_override="in.npz").background

    def test_a_file_without_a_background_section_gets_the_defaults(self, tmp_path):
        background = self._background("render:\n  modes: [\"mesh\"]\n", tmp_path)
        assert background.texture == "grid"
        assert background.geometry == "cube"
        assert background.radius_scale == DEFAULT_RADIUS_SCALE

    def test_a_radius_in_the_file_supersedes_the_default_scale(self, tmp_path):
        """
        Writing one of a mutually exclusive pair is not a contradiction, and
        must not be reported as one.
        """
        background = self._background(
            "background:\n  enabled: true\n  radius: 5.0\n", tmp_path
        )
        assert background.radius == 5.0
        assert background.radius_scale is None

    def test_a_null_radius_scale_asks_for_infinity(self, tmp_path):
        background = self._background(
            "background:\n  enabled: true\n  radius_scale: null\n", tmp_path
        )
        assert background.radius is None and background.radius_scale is None

    def test_writing_both_is_still_a_contradiction(self, tmp_path):
        with pytest.raises(ValueError, match="mutually exclusive"):
            self._background(
                "background:\n  radius: 5.0\n  radius_scale: 3.0\n", tmp_path
            )


class TestBackgroundCliOverrides:
    """--background and friends, layered over a config file's values."""

    @staticmethod
    def _config(*argv):
        args = create_argument_parser().parse_args(
            ["in.npz", "-o", "out", *argv]
        )
        return Config.from_args(args).background

    def test_absent_by_default(self):
        assert self._config().enabled is False

    def test_naming_a_texture_enables_it(self):
        """One flag is enough; --background grid should not also need a toggle."""
        background = self._config("--background", "grid")
        assert background.enabled is True
        assert background.texture == "grid"

    def test_no_background_wins_over_a_config_file(self):
        assert self._config("--background", "grid", "--no-background").enabled is False

    def test_defaults_to_a_grid_cube_at_the_default_scale(self):
        background = self._config("--background", "grid")
        assert background.geometry == "cube"
        assert background.radius_scale == DEFAULT_RADIUS_SCALE
        assert background.radius is None

    def test_geometry_and_radius(self):
        background = self._config(
            "--background", "checker",
            "--background-geometry", "cube",
            "--background-radius", "8.5",
            "--background-rotation", "30",
        )
        assert background.geometry == "cube"
        assert background.radius == 8.5
        assert background.rotation_deg == 30.0
        # An explicit radius has to displace the defaulted scale, or every
        # --background-radius would land on the mutual-exclusion error.
        assert background.radius_scale is None

    def test_infinite_flag_clears_both_radius_forms(self):
        background = self._config("--background", "blender_sky",
                                  "--background-infinite")
        assert background.radius is None and background.radius_scale is None
        background.validate()

    def test_radius_scale_clears_a_radius_from_the_config_file(self):
        """
        Otherwise the two would coexist and trip the mutual-exclusion check,
        turning a plain CLI override into an error.
        """
        args = create_argument_parser().parse_args(
            ["in.npz", "-o", "out", "--background", "grid",
             "--background-radius-scale", "3.0"]
        )
        config = Config.from_args(args)
        config.background.radius = None  # as if a config file had set it
        assert config.background.radius_scale == 3.0
        config.background.validate()

    def test_keep_alpha(self):
        assert self._config("--background", "grid",
                            "--background-keep-alpha").opaque is False

    def test_bad_geometry_is_caught_by_argparse(self):
        with pytest.raises(SystemExit):
            self._config("--background-geometry", "dome")

    def test_both_radius_forms_are_caught_by_argparse(self):
        with pytest.raises(SystemExit):
            self._config("--background-radius", "5",
                         "--background-radius-scale", "3")


class TestBackgroundFadeConfigValidation:
    """The fade settings, checked as a set."""

    def test_off_by_default(self):
        config = BackgroundConfig()
        assert config.fade.enabled is False
        config.validate()

    def test_rejects_an_unknown_profile(self):
        config = BackgroundConfig()
        config.fade.profile = "logarithmic"
        with pytest.raises(ValueError, match="smoothstep"):
            config.validate()

    def test_rejects_an_unknown_target(self):
        config = BackgroundConfig()
        config.fade.target = "local"
        with pytest.raises(ValueError, match="plain, color, blur"):
            config.validate()

    @pytest.mark.parametrize(
        "field,value", [("falloff", 0.0), ("rate", -1.0),
                        ("margin", 0.0), ("detail", 0)]
    )
    def test_rejects_non_positive_scalars(self, field, value):
        config = BackgroundConfig()
        setattr(config.fade, field, value)
        with pytest.raises(ValueError, match=field):
            config.validate()

    def test_falloff_error_points_at_the_step_profile(self):
        """falloff=0 is a plausible way to ask for a hard edge; say where it is."""
        config = BackgroundConfig()
        config.fade.falloff = 0.0
        with pytest.raises(ValueError, match="step"):
            config.validate()

    def test_rejects_a_colour_out_of_range(self):
        config = BackgroundConfig()
        config.fade.color = (1.0, 1.5, 0.0)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            config.validate()


class TestBackgroundFadeCliOverrides:
    """--background-fade and friends."""

    @staticmethod
    def _fade(*argv):
        args = create_argument_parser().parse_args(
            ["in.npz", "-o", "out", *argv]
        )
        return Config.from_args(args).background.fade

    def test_off_by_default(self):
        assert self._fade("--background", "grid").enabled is False

    def test_defaults_to_the_pattern_free_target(self):
        """Not 'blur': averaging smears the lines rather than removing them."""
        fade = self._fade("--background", "grid", "--background-fade", "linear")
        assert fade.target == "plain"

    def test_naming_a_profile_enables_it(self):
        """One flag is enough, matching --background's own shape."""
        fade = self._fade("--background", "grid", "--background-fade", "gaussian")
        assert fade.enabled is True
        assert fade.profile == "gaussian"

    def test_no_fade_wins_over_a_config_file(self):
        assert self._fade(
            "--background", "grid", "--background-fade", "linear",
            "--no-background-fade",
        ).enabled is False

    def test_scalars(self):
        fade = self._fade(
            "--background", "grid", "--background-fade", "exponential",
            "--background-fade-falloff", "0.4",
            "--background-fade-rate", "6",
            "--background-fade-margin", "1.25",
            "--background-fade-detail", "48",
        )
        assert (fade.falloff, fade.rate, fade.margin, fade.detail) == (
            0.4, 6.0, 1.25, 48
        )

    def test_a_colour_implies_the_flat_target(self):
        """
        Otherwise the colour would be accepted and then silently ignored,
        which looks like the colour did not work.
        """
        fade = self._fade("--background", "grid", "--background-fade", "linear",
                          "--background-fade-color", "0.5,0.5,0.5")
        assert fade.color == (0.5, 0.5, 0.5)
        assert fade.target == "color"

    def test_an_explicit_target_still_wins(self):
        fade = self._fade("--background", "grid", "--background-fade", "linear",
                          "--background-fade-color", "0.5,0.5,0.5",
                          "--background-fade-target", "blur")
        assert fade.target == "blur"

    def test_a_malformed_colour_is_reported(self):
        with pytest.raises(ValueError, match="background-fade-color"):
            self._fade("--background", "grid", "--background-fade", "linear",
                       "--background-fade-color", "grey")
