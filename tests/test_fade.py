"""
Tests for the subject fade over the backdrop.

The load-bearing test is :class:`TestClearZoneCoversTheSilhouette`: the whole
point of fitting an ellipsoid rather than using one frame's outline is that the
clear zone must cover the silhouette from *every* viewpoint on an orbit, and a
fade that is merely centred on the subject looks fine in a single frame while
letting the grid run into the outline halfway round.
"""

import numpy as np
import pytest

from body2colmap.background import Background
from body2colmap.camera import Camera
from body2colmap.coordinates import spherical_to_cartesian
from body2colmap.fade import (
    DECAY_PROFILES,
    Ellipsoid,
    SubjectFade,
    decay_weight,
)


def make_camera(width=96, height=72, focal=90.0, position=(0.0, 0.0, 0.0)):
    """A camera at ``position`` with identity rotation (looking down -Z)."""
    return Camera(
        focal_length=(focal, focal),
        image_size=(width, height),
        position=np.asarray(position, dtype=np.float32),
        rotation=np.eye(3, dtype=np.float32),
    )


def body_points(n=2000, seed=0):
    """A standing-figure-ish point cloud: tall, narrow, off the origin."""
    rng = np.random.default_rng(seed)
    shell = rng.normal(size=(n, 3))
    shell /= np.linalg.norm(shell, axis=1, keepdims=True)
    return (shell * np.array([0.28, 0.9, 0.16])
            + np.array([0.1, 0.95, -0.2])).astype(np.float32)


def world_rays(camera):
    """Per-pixel world ray directions, straight from the background module."""
    from body2colmap.background import camera_ray_directions
    return camera_ray_directions(camera)


class TestDecayProfiles:
    """Every profile is a monotone fall from 1 at the surface."""

    @pytest.mark.parametrize("profile", sorted(DECAY_PROFILES))
    def test_starts_at_full_fade(self, profile):
        assert decay_weight(np.zeros(1), profile) == pytest.approx(1.0)

    @pytest.mark.parametrize("profile", sorted(DECAY_PROFILES))
    def test_monotonically_decreasing(self, profile):
        u = np.linspace(0.0, 4.0, 400)
        weight = decay_weight(u, profile)
        assert np.all(np.diff(weight) <= 1e-6)

    @pytest.mark.parametrize("profile", sorted(DECAY_PROFILES))
    def test_bounded(self, profile):
        weight = decay_weight(np.linspace(0.0, 20.0, 500), profile)
        assert weight.min() >= 0.0 and weight.max() <= 1.0

    @pytest.mark.parametrize(
        "profile", ["step", "linear", "smoothstep", "cosine"]
    )
    def test_compact_profiles_reach_zero_at_the_band_edge(self, profile):
        assert decay_weight(np.array([1.0001]), profile) == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "profile", ["exponential", "gaussian", "inverse_square"]
    )
    def test_tailed_profiles_do_not(self, profile):
        """
        The three shaped profiles never quite reach zero. That is their
        character -- and the reason a compact one is the default.
        """
        assert decay_weight(np.array([1.0]), profile) > 0.0

    def test_inverse_square_has_the_heavy_tail(self):
        """Documented in the module: still a visible wash well out at u = 3."""
        far = {
            name: float(decay_weight(np.array([3.0]), name)[0])
            for name in ("exponential", "gaussian", "inverse_square")
        }
        assert far["inverse_square"] > 0.02
        assert far["exponential"] < 1e-4
        assert far["gaussian"] < 1e-4

    def test_rate_tightens_the_shaped_profiles(self):
        u = np.array([0.5])
        for profile in ("exponential", "gaussian", "inverse_square"):
            loose = decay_weight(u, profile, rate=1.0)
            tight = decay_weight(u, profile, rate=8.0)
            assert tight < loose, profile

    def test_unknown_profile_names_the_alternatives(self):
        with pytest.raises(ValueError, match="smoothstep"):
            decay_weight(np.zeros(1), "linear-ish")

    def test_rate_must_be_positive(self):
        with pytest.raises(ValueError, match="rate"):
            decay_weight(np.zeros(1), "exponential", rate=0.0)


class TestEllipsoidFit:
    """The fit must enclose everything, and must be tight when it can be."""

    def test_recovers_a_known_ellipsoid(self):
        rng = np.random.default_rng(3)
        axes = np.array([3.0, 1.0, 0.5])
        angle = 0.7
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0.0],
             [np.sin(angle), np.cos(angle), 0.0],
             [0.0, 0.0, 1.0]]
        )
        shell = rng.normal(size=(4000, 3))
        shell /= np.linalg.norm(shell, axis=1, keepdims=True)
        center = np.array([1.0, 2.0, -3.0])
        points = (shell * axes) @ rotation.T + center

        fitted = Ellipsoid.fit(points)

        assert np.sort(fitted.axes)[::-1] == pytest.approx(axes, rel=0.02)
        assert fitted.center == pytest.approx(center, abs=0.02)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_encloses_every_point(self, seed):
        points = body_points(seed=seed)
        assert Ellipsoid.fit(points).contains(points).all()

    def test_encloses_every_point_even_when_subsampled(self):
        """
        The fit runs on a stride of the vertices for speed, so enclosure is
        imposed afterwards against the full set. Squeezing max_points down to
        a handful makes the solver's answer useless and the guarantee is all
        that is left -- which is exactly what must still hold.
        """
        points = body_points(n=3000, seed=7)
        fitted = Ellipsoid.fit(points, max_points=20)
        assert fitted.contains(points).all()

    def test_margin_inflates_without_moving_the_centre(self):
        points = body_points()
        tight = Ellipsoid.fit(points)
        loose = Ellipsoid.fit(points, margin=2.0)

        assert loose.center == pytest.approx(tight.center)
        assert np.sort(loose.axes) == pytest.approx(2.0 * np.sort(tight.axes))

    def test_scaled_matches_a_margin_at_fit_time(self):
        points = body_points()
        assert np.sort(Ellipsoid.fit(points).scaled(1.7).axes) == pytest.approx(
            np.sort(Ellipsoid.fit(points, margin=1.7).axes)
        )

    def test_beats_the_bounding_box_ellipsoid(self):
        """
        Why pipeline._resolve_fade() fits the vertices rather than the bounds:
        a box ellipsoid has to clear the box's corners, so it clears far more
        of the backdrop than the subject warrants.
        """
        points = body_points()
        fitted = Ellipsoid.fit(points)
        boxed = Ellipsoid.from_bounds(points.min(axis=0), points.max(axis=0))

        assert boxed.contains(points).all()
        assert np.prod(fitted.axes) < 0.5 * np.prod(boxed.axes)

    def test_rejects_a_degenerate_point_set(self):
        flat = np.zeros((200, 3), dtype=np.float32)
        flat[:, :2] = np.random.default_rng(0).normal(size=(200, 2))
        with pytest.raises(ValueError, match="coplanar|degenerate|singular"):
            Ellipsoid.fit(flat)

    def test_rejects_too_few_points(self):
        with pytest.raises(ValueError, match="at least 4"):
            Ellipsoid.fit(np.zeros((3, 3), dtype=np.float32))


class TestRayDistance:
    """The per-pixel scalar the whole fade is built on."""

    def test_through_the_centre_is_zero(self):
        sphere = Ellipsoid(np.array([0.0, 0.0, -5.0]), np.eye(3))
        assert sphere.ray_distance(
            np.zeros(3), np.array([[0.0, 0.0, -1.0]])
        ) == pytest.approx(0.0, abs=1e-6)

    def test_surface_grazing_ray_is_one(self):
        sphere = Ellipsoid(np.array([0.0, 0.0, -5.0]), np.eye(3))
        # Tangent from the origin to a unit sphere 5 away: sin(theta) = 1/5.
        theta = np.arcsin(0.2)
        direction = np.array([[np.sin(theta), 0.0, -np.cos(theta)]])
        assert sphere.ray_distance(np.zeros(3), direction) == pytest.approx(
            1.0, abs=1e-5
        )

    def test_is_measured_in_radius_units_not_world_units(self):
        """
        Scale-invariance is what makes `falloff` a subject-relative width, so
        one setting works across an auto-framed orbit at any subject size.
        """
        rays = np.array([[0.3, 0.1, -1.0], [0.0, 0.5, -1.0]])
        small = Ellipsoid(np.zeros(3), np.eye(3) / 1.0)
        big = Ellipsoid(np.zeros(3), np.eye(3) / 10.0)
        origin_small = np.array([0.0, 0.0, 4.0])
        origin_big = origin_small * 10.0

        assert small.ray_distance(origin_small, rays) == pytest.approx(
            big.ray_distance(origin_big, rays), rel=1e-5
        )

    def test_rays_pointing_away_are_not_extended_backwards(self):
        sphere = Ellipsoid(np.array([0.0, 0.0, -5.0]), np.eye(3))
        away = sphere.ray_distance(np.zeros(3), np.array([[0.0, 0.0, 1.0]]))
        assert away == pytest.approx(5.0)


class TestClearZoneCoversTheSilhouette:
    """
    The reason the shell is a fitted ellipsoid and not a per-frame outline.

    Every pixel that the subject actually projects onto must be fully faded,
    from every viewpoint on an orbit -- otherwise the grid runs into the
    silhouette on some frame and the model reads it as an occlusion boundary
    again, which is the failure the fade exists to remove.
    """

    @pytest.mark.parametrize("azimuth_deg", [0.0, 55.0, 130.0, 215.0, 300.0])
    @pytest.mark.parametrize("elevation_deg", [-25.0, 0.0, 30.0])
    def test_every_projected_vertex_is_fully_faded(
        self, azimuth_deg, elevation_deg
    ):
        points = body_points(n=1500)
        target = (points.min(axis=0) + points.max(axis=0)) / 2.0

        camera = make_camera(width=192, height=256, focal=200.0)
        camera.position = (
            target + spherical_to_cartesian(4.0, azimuth_deg, elevation_deg)
        ).astype(np.float32)
        camera.look_at(target.astype(np.float32))

        fade = SubjectFade(Ellipsoid.fit(points), profile="step")
        weight = fade.weights(camera, world_rays(camera))

        projected = camera.project(points)
        cols = np.clip(np.round(projected[:, 0]).astype(int), 0, camera.width - 1)
        rows = np.clip(np.round(projected[:, 1]).astype(int), 0, camera.height - 1)

        assert weight[rows, cols].min() == pytest.approx(1.0)

    def test_the_far_field_is_untouched(self):
        """The rotation cue has to survive; a fade over the whole frame is
        the same failure with extra steps."""
        points = body_points()
        camera = make_camera(width=192, height=256, focal=200.0)
        camera.position = np.array([0.0, 0.95, 3.5], dtype=np.float32)
        camera.look_at(np.array([0.1, 0.95, -0.2], dtype=np.float32))

        fade = SubjectFade(Ellipsoid.fit(points), profile="smoothstep")
        weight = fade.weights(camera, world_rays(camera))

        assert weight.max() == pytest.approx(1.0)
        assert weight[0, 0] == pytest.approx(0.0)
        assert (weight < 0.01).mean() > 0.4

    def test_a_wider_falloff_clears_more(self):
        points = body_points()
        camera = make_camera(width=192, height=256, focal=200.0)
        camera.position = np.array([0.0, 0.95, 3.5], dtype=np.float32)
        camera.look_at(np.array([0.1, 0.95, -0.2], dtype=np.float32))
        dirs = world_rays(camera)

        ellipsoid = Ellipsoid.fit(points)
        narrow = SubjectFade(ellipsoid, falloff=0.3).weights(camera, dirs)
        wide = SubjectFade(ellipsoid, falloff=2.0).weights(camera, dirs)

        assert np.all(wide >= narrow - 1e-6)
        assert wide.sum() > 2.0 * narrow.sum()

    def test_a_wider_margin_clears_more(self):
        points = body_points()
        camera = make_camera(width=192, height=256, focal=200.0)
        camera.position = np.array([0.0, 0.95, 3.5], dtype=np.float32)
        camera.look_at(np.array([0.1, 0.95, -0.2], dtype=np.float32))
        dirs = world_rays(camera)

        tight = SubjectFade(Ellipsoid.fit(points), profile="step")
        loose = SubjectFade(Ellipsoid.fit(points, margin=1.5), profile="step")

        assert (loose.weights(camera, dirs) >= tight.weights(camera, dirs)).all()


class TestApply:
    """Blending the fade into a rendered backdrop."""

    @staticmethod
    def _setup(profile="step", **kwargs):
        camera = make_camera(width=96, height=96, focal=80.0,
                             position=(0.0, 0.0, 3.0))
        ellipsoid = Ellipsoid(np.zeros(3), np.eye(3) * 2.0)
        fade = SubjectFade(ellipsoid, profile=profile, **kwargs)
        return camera, fade, world_rays(camera)

    def test_flat_colour_target(self):
        camera, fade, dirs = self._setup(target="color", color=(1.0, 0.0, 0.0))
        image = np.full((96, 96, 3), 40, dtype=np.uint8)

        faded = fade.apply(image, camera, dirs)

        assert tuple(faded[48, 48]) == (255, 0, 0)
        assert tuple(faded[0, 0]) == (40, 40, 40)

    def test_default_colour_is_used_when_none_is_set(self):
        camera, fade, dirs = self._setup(target="color")
        image = np.full((96, 96, 3), 40, dtype=np.uint8)

        faded = fade.apply(image, camera, dirs, default_color=(0.0, 1.0, 0.0))

        assert tuple(faded[48, 48]) == (0, 255, 0)

    def test_explicit_colour_beats_the_default(self):
        camera, fade, dirs = self._setup(target="color", color=(1.0, 0.0, 0.0))
        image = np.full((96, 96, 3), 40, dtype=np.uint8)

        faded = fade.apply(image, camera, dirs, default_color=(0.0, 1.0, 0.0))

        assert tuple(faded[48, 48]) == (255, 0, 0)

    def test_local_target_erases_detail_but_keeps_tone(self):
        """
        What `target="local"` is for: the lines go, the wall tone stays, so
        the clear zone has no edge against the surrounding backdrop.
        """
        # detail=8 puts the averaging box at 12 px, two whole line periods, so
        # the box average is exact rather than beating against the pattern.
        camera, fade, dirs = self._setup(target="local", detail=8)
        image = np.full((96, 96, 3), 100, dtype=np.uint8)
        image[:, ::6] = 200                       # "grid lines"

        faded = fade.apply(image, camera, dirs)
        cleared = fade.weights(camera, dirs) > 0.99

        assert faded[cleared].std() < 3.0                 # detail gone
        assert abs(float(faded[cleared].mean()) - float(image.mean())) < 6.0

    def test_a_fade_that_reaches_nothing_returns_the_image_untouched(self):
        camera = make_camera(position=(0.0, 0.0, 3.0))
        far = Ellipsoid(np.array([0.0, 0.0, 500.0]), np.eye(3))
        fade = SubjectFade(far, profile="step")
        image = np.full((72, 96, 3), 40, dtype=np.uint8)

        assert fade.apply(image, camera, world_rays(camera)) is image

    def test_rejects_bad_settings(self):
        ellipsoid = Ellipsoid(np.zeros(3), np.eye(3))
        with pytest.raises(ValueError, match="profile"):
            SubjectFade(ellipsoid, profile="nope")
        with pytest.raises(ValueError, match="target"):
            SubjectFade(ellipsoid, target="nope")
        with pytest.raises(ValueError, match="falloff"):
            SubjectFade(ellipsoid, falloff=0.0)
        with pytest.raises(ValueError, match="rate"):
            SubjectFade(ellipsoid, rate=-1.0)
        with pytest.raises(ValueError, match="detail"):
            SubjectFade(ellipsoid, detail=0)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            SubjectFade(ellipsoid, color=(2.0, 0.0, 0.0))
        with pytest.raises(ValueError, match="3 RGB"):
            SubjectFade(ellipsoid, color=(0.5, 0.5))


class TestBackgroundIntegration:
    """The fade is applied by Background.render(), to the backdrop only."""

    RADIUS = 1.0

    @classmethod
    def _ellipsoid(cls):
        return Ellipsoid(np.zeros(3), np.eye(3) / cls.RADIUS)

    @classmethod
    def _background(cls, **fade_kwargs):
        return cls._plain(fade=SubjectFade(cls._ellipsoid(), **fade_kwargs))

    @staticmethod
    def _plain(**kwargs):
        return Background.create(
            "grid", "cube", resolution=256, center=np.zeros(3), radius=8.0,
            **kwargs,
        )

    @staticmethod
    def _camera():
        return make_camera(width=160, height=160, focal=140.0,
                           position=(0.0, 0.0, 4.0))

    def test_render_clears_the_middle_and_keeps_the_edges(self):
        camera = self._camera()
        plain = self._plain().render(camera)
        # detail=8 rather than the default: this camera sees the cube's grid
        # cells at ~37 px, so the default's 7 px averaging box is only just
        # wide enough to swallow a line.
        faded = self._background(
            profile="step", target="local", detail=8
        ).render(camera)

        cleared = SubjectFade(
            self._ellipsoid(), profile="step"
        ).weights(camera, world_rays(camera)) > 0.99

        # Measured as neighbour-to-neighbour variation, not as a plain
        # standard deviation: "local" deliberately keeps the low-frequency
        # wall/floor tone, and only the grid lines are supposed to go.
        inside = cleared[:, :-1] & cleared[:, 1:]

        def detail(image):
            steps = np.abs(np.diff(image.astype(float), axis=1)).mean(axis=2)
            return float(steps[inside].mean())

        assert detail(faded) < detail(plain) / 10.0
        np.testing.assert_array_equal(faded[~cleared], plain[~cleared])

    def test_texture_mean_is_the_fallback_colour(self):
        background = self._background(profile="step", target="color")
        camera = self._camera()

        faded = background.render(camera)
        expected = np.round(background._texture_mean() * 255.0)

        assert faded[80, 80] == pytest.approx(expected, abs=1)

    def test_the_fade_never_touches_the_subject(self):
        """
        The fade runs inside render(), so composite() lays the base layer over
        an already-faded backdrop. An opaque subject pixel must come through
        exactly as rendered.
        """
        camera = self._camera()
        background = self._background(profile="step", target="color",
                                      color=(1.0, 0.0, 0.0))

        layer = np.zeros((160, 160, 4), dtype=np.uint8)
        layer[70:90, 70:90] = (10, 200, 30, 255)

        composited = background.composite(layer, camera)

        # (80, 80) is the subject; (80, 55) is backdrop inside the clear zone,
        # which at radius 1.0 and focal 140 over 4 units reaches +-35 px.
        assert tuple(composited[80, 80]) == (10, 200, 30, 255)
        assert tuple(composited[80, 55][:3]) == (255, 0, 0)

    def test_describe_mentions_the_fade(self):
        described = self._background(profile="gaussian").describe()
        assert "gaussian" in described and "ellipsoid" in described

    def test_a_fade_out_of_frame_leaves_the_render_untouched(self):
        camera = self._camera()
        elsewhere = SubjectFade(
            Ellipsoid(np.array([0.0, 0.0, 50.0]), np.eye(3)), profile="step"
        )
        np.testing.assert_array_equal(
            self._plain(fade=elsewhere).render(camera),
            self._plain().render(camera),
        )
