"""
Tests for the environment backdrop.

The backdrop is pure numpy plus cv2.remap, so all of it is testable without an
OpenGL context. The load-bearing test is
:class:`TestMarkerLandsWhereProjectionSaysItShould`: a mirrored, transposed or
half-turned lookup all still produce a plausible-looking sky, and only checking
a known direction against the camera's own projection catches them.
"""

import numpy as np
import pytest

from body2colmap.background import (
    CUBE_FACES,
    Background,
    camera_ray_directions,
    cube_face_directions,
    cube_face_uv,
    equirect_directions,
    equirect_to_cube,
    equirect_uv,
    generate_texture,
    load_texture,
)
from body2colmap.camera import Camera
from body2colmap.coordinates import spherical_to_cartesian


def make_camera(width=64, height=48, focal=90.0, position=(0.0, 0.0, 0.0)):
    """A camera at ``position`` with identity rotation (looking down -Z)."""
    camera = Camera(
        focal_length=(focal, focal),
        image_size=(width, height),
        position=np.asarray(position, dtype=np.float32),
        rotation=np.eye(3, dtype=np.float32),
    )
    return camera


def flat_texture(color, geometry="sphere", size=32):
    """A uniformly coloured texture of the right shape for ``geometry``."""
    if geometry == "sphere":
        return np.tile(np.uint8(color), (size, 2 * size, 1))
    return [np.tile(np.uint8(color), (size, size, 1)) for _ in CUBE_FACES]


class TestRayDirections:
    """Per-pixel rays must invert Camera.project exactly."""

    def test_centre_pixel_is_the_forward_vector(self):
        camera = make_camera(width=65, height=49)
        camera.position = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        camera.look_at(np.zeros(3, dtype=np.float32))

        dirs = camera_ray_directions(camera)
        assert dirs[24, 32] == pytest.approx(camera.get_forward_vector(), abs=1e-5)

    def test_axes_point_the_expected_way(self):
        """Right of centre leans right; above centre leans up."""
        camera = make_camera(width=65, height=49)
        dirs = camera_ray_directions(camera)

        assert np.dot(dirs[24, 64], camera.get_right_vector()) > 0
        assert np.dot(dirs[24, 0], camera.get_right_vector()) < 0
        assert np.dot(dirs[0, 32], camera.get_up_vector()) > 0
        assert np.dot(dirs[48, 32], camera.get_up_vector()) < 0

    def test_unit_length(self):
        dirs = camera_ray_directions(make_camera())
        assert np.linalg.norm(dirs, axis=-1) == pytest.approx(1.0, abs=1e-5)

    def test_matches_the_cached_world_rays(self):
        """Background caches the camera-space grid; the result must agree."""
        camera = make_camera()
        camera.position = np.array([0.5, -1.0, 2.0], dtype=np.float32)
        camera.look_at(np.array([0.0, 1.0, 0.0], dtype=np.float32))

        background = Background(flat_texture((0, 0, 0)))
        assert background._world_rays(camera) == pytest.approx(
            camera_ray_directions(camera), abs=1e-5
        )


class TestEquirectParameterization:
    """equirect_uv and equirect_directions are inverses."""

    def test_round_trip(self):
        height = 32
        u, v = equirect_uv(equirect_directions(height))

        expected_u = (np.arange(2 * height) + 0.5) / (2 * height)
        expected_v = (np.arange(height) + 0.5) / height

        assert u == pytest.approx(np.broadcast_to(expected_u, u.shape), abs=1e-6)
        assert v == pytest.approx(
            np.broadcast_to(expected_v[:, None], v.shape), abs=1e-6
        )

    @pytest.mark.parametrize("azimuth_deg,expected_u", [
        (0.0, 0.5),      # +Z, toward the viewer
        (90.0, 0.75),    # +X, right
        (-90.0, 0.25),   # -X, left
    ])
    def test_matches_the_project_spherical_convention(self, azimuth_deg, expected_u):
        direction = spherical_to_cartesian(1.0, azimuth_deg, 0.0)[None]
        u, v = equirect_uv(direction.astype(np.float32))
        assert u[0] == pytest.approx(expected_u, abs=1e-5)
        assert v[0] == pytest.approx(0.5, abs=1e-5)

    def test_v_runs_zenith_to_nadir(self):
        up = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        down = np.array([[0.0, -1.0, 0.0]], dtype=np.float32)
        assert equirect_uv(up)[1][0] == pytest.approx(0.0, abs=1e-5)
        assert equirect_uv(down)[1][0] == pytest.approx(1.0, abs=1e-5)


class TestCubeParameterization:
    """cube_face_uv and cube_face_directions are inverses."""

    @pytest.mark.parametrize("face", CUBE_FACES)
    def test_round_trip(self, face):
        dirs, s, t = cube_face_directions(face, 16)
        index, s_back, t_back = cube_face_uv(dirs)

        assert (index == CUBE_FACES.index(face)).all()
        assert s_back == pytest.approx(s, abs=1e-5)
        assert t_back == pytest.approx(t, abs=1e-5)

    @pytest.mark.parametrize("axis,face", [
        ((1, 0, 0), "px"), ((-1, 0, 0), "nx"),
        ((0, 1, 0), "py"), ((0, -1, 0), "ny"),
        ((0, 0, 1), "pz"), ((0, 0, -1), "nz"),
    ])
    def test_axis_directions_hit_their_own_face_centre(self, axis, face):
        index, s, t = cube_face_uv(np.array([axis], dtype=np.float32))
        assert CUBE_FACES[index[0]] == face
        assert s[0] == pytest.approx(0.5, abs=1e-6)
        assert t[0] == pytest.approx(0.5, abs=1e-6)

    def test_unknown_face_rejected(self):
        with pytest.raises(ValueError, match="Unknown cube face"):
            cube_face_directions("up", 8)


class TestMarkerLandsWhereProjectionSaysItShould:
    """
    Render a texture with one bright texel and check it lands on the pixel the
    camera's own projection predicts.

    For an infinite backdrop the point at direction ``d`` is seen along ``d``,
    so ``camera.project(camera.position + d)`` is the answer, independently of
    every convention inside the background code.
    """

    MARKERS = [(0.0, 0.0), (35.0, 20.0), (-70.0, -15.0), (150.0, 40.0)]

    @staticmethod
    def _centroid(image):
        """Intensity-weighted centre of the bright blob, as (x, y)."""
        weight = image.sum(axis=2).astype(np.float64)
        assert weight.max() > 0, "marker is not in frame"
        rows, cols = np.indices(weight.shape)
        total = weight.sum()
        return np.array([(cols * weight).sum() / total,
                         (rows * weight).sum() / total])

    @staticmethod
    def _camera_looking_near(azimuth_deg, elevation_deg):
        """Wide-angle camera aimed off-target, so the marker is off-centre."""
        camera = make_camera(width=256, height=192, focal=140.0)
        camera.look_at(
            spherical_to_cartesian(
                1.0, azimuth_deg - 22.0, elevation_deg - 9.0
            ).astype(np.float32)
        )
        return camera

    @pytest.mark.parametrize("azimuth_deg,elevation_deg", MARKERS)
    def test_sphere(self, azimuth_deg, elevation_deg):
        height = 512
        texture = np.zeros((height, 2 * height, 3), dtype=np.uint8)

        # Snap the marker to a texel and take its exact centre direction back
        # from equirect_directions, so the test measures the render rather than
        # a half-texel quantization of where the marker was put.
        target = spherical_to_cartesian(1.0, azimuth_deg, elevation_deg)
        u, v = equirect_uv(target.astype(np.float32)[None])
        col = int(u[0] * 2 * height)
        row = int(v[0] * height)
        texture[row, col] = 255
        marker = equirect_directions(height)[row, col]

        camera = self._camera_looking_near(azimuth_deg, elevation_deg)
        found = self._centroid(Background(texture, "sphere").render(camera))
        expected = camera.project((camera.position + marker)[None])[0]

        assert found == pytest.approx(expected, abs=0.75)

    @pytest.mark.parametrize("azimuth_deg,elevation_deg", MARKERS)
    def test_cube(self, azimuth_deg, elevation_deg):
        # Small enough that _fit_resolution leaves it alone: area-averaging a
        # one-texel marker snaps it to the coarser grid, which would blunt the
        # sub-pixel comparison this test exists to make.
        size = 256
        target = spherical_to_cartesian(1.0, azimuth_deg, elevation_deg)
        index, s, t = cube_face_uv(target.astype(np.float32)[None])

        faces = [np.zeros((size, size, 3), dtype=np.uint8) for _ in CUBE_FACES]
        col = int(s[0] * size)
        row = int(t[0] * size)
        faces[index[0]][row, col] = 255
        marker = cube_face_directions(CUBE_FACES[index[0]], size)[0][row, col]

        camera = self._camera_looking_near(azimuth_deg, elevation_deg)
        found = self._centroid(Background(faces, "cube").render(camera))
        expected = camera.project((camera.position + marker)[None])[0]

        assert found == pytest.approx(expected, abs=0.75)


class TestFiniteGeometry:
    """A finite surface must be hit exactly, and must contain the camera."""

    def test_sphere_hits_lie_on_the_sphere(self):
        background = Background(flat_texture((0, 0, 0)), "sphere", radius=5.0)
        camera = make_camera(position=(0.4, -0.7, 1.1))
        camera.look_at(np.array([1.0, 0.5, -2.0], dtype=np.float32))

        hits = background._surface_vectors(camera, background._world_rays(camera))
        assert np.linalg.norm(hits, axis=-1) == pytest.approx(5.0, abs=1e-3)

    def test_cube_hits_lie_on_the_cube(self):
        background = Background(flat_texture((0, 0, 0), "cube"), "cube", radius=5.0)
        camera = make_camera(position=(0.4, -0.7, 1.1))
        camera.look_at(np.array([1.0, 0.5, -2.0], dtype=np.float32))

        hits = background._surface_vectors(camera, background._world_rays(camera))
        assert np.abs(hits).max(axis=-1) == pytest.approx(5.0, abs=1e-3)

    def test_hits_are_in_front_of_the_camera(self):
        background = Background(flat_texture((0, 0, 0)), "sphere", radius=4.0)
        camera = make_camera(position=(1.0, 0.0, 2.0))
        dirs = background._world_rays(camera)
        hits = background._surface_vectors(camera, dirs)

        origin = np.asarray(camera.position) - background.center
        along = np.einsum("...i,...i->...", hits - origin, dirs)
        assert (along > 0).all()

    def test_centre_ray_of_an_identity_camera(self):
        """Looking down -Z from inside, the centre ray exits at -radius on Z."""
        for geometry in ("sphere", "cube"):
            background = Background(
                flat_texture((0, 0, 0), geometry), geometry, radius=5.0
            )
            camera = make_camera(width=9, height=9, position=(0.0, 0.0, 2.0))
            hits = background._surface_vectors(camera, background._world_rays(camera))
            assert hits[4, 4] == pytest.approx([0.0, 0.0, -5.0], abs=1e-4)

    @pytest.mark.parametrize("geometry", ["sphere", "cube"])
    def test_camera_outside_is_an_error(self, geometry):
        background = Background(
            flat_texture((0, 0, 0), geometry), geometry, radius=1.0
        )
        camera = make_camera(position=(0.0, 0.0, 9.0))
        with pytest.raises(ValueError, match="must be inside"):
            background.render(camera)

    def test_centre_offsets_the_surface(self):
        """A camera outside a centred sphere is inside one centred on it."""
        camera = make_camera(position=(0.0, 0.0, 9.0))
        centred = Background(flat_texture((0, 0, 0)), "sphere", radius=2.0)
        with pytest.raises(ValueError):
            centred.render(camera)

        moved = Background(
            flat_texture((0, 0, 0)), "sphere",
            center=np.array([0.0, 0.0, 9.0], dtype=np.float32), radius=2.0,
        )
        assert moved.render(camera).shape == (48, 64, 3)

    def test_non_positive_radius_rejected(self):
        with pytest.raises(ValueError, match="radius must be > 0"):
            Background(flat_texture((0, 0, 0)), "sphere", radius=0.0)


class TestParallax:
    """
    What a finite radius actually buys.

    An infinite backdrop is a function of ray direction alone, so translating
    the camera without rotating it changes nothing. That is correct for a
    distant sky and useless as a depth cue; a finite surface is what makes the
    backdrop move relative to the subject.
    """

    @staticmethod
    def _render_pair(background):
        near = make_camera(width=96, height=96, focal=60.0, position=(0.0, 0.0, 0.0))
        far = make_camera(width=96, height=96, focal=60.0, position=(1.5, 0.0, 0.0))
        return background.render(near).astype(float), background.render(far).astype(float)

    def test_infinite_backdrop_ignores_translation(self):
        background = Background.create("checker", "sphere", resolution=256)
        a, b = self._render_pair(background)
        assert np.abs(a - b).mean() == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("geometry", ["sphere", "cube"])
    def test_finite_backdrop_shifts_with_translation(self, geometry):
        background = Background.create(
            "checker", geometry, resolution=256, radius=6.0
        )
        a, b = self._render_pair(background)
        assert np.abs(a - b).mean() > 5.0


class TestRotation:
    """rotation_deg turns the environment, not the camera."""

    def test_rotating_the_environment_matches_rotating_the_camera(self):
        """
        Turning the world by +40 degrees looks like turning the camera by -40:
        both put the same part of the texture in front of the lens.
        """
        texture = generate_texture("checker", "sphere", 512)

        still = Background(texture, "sphere", rotation_deg=40.0)
        camera_a = make_camera(width=96, height=96, focal=60.0)
        image_a = still.render(camera_a)

        turned = Background(texture, "sphere")
        camera_b = make_camera(width=96, height=96, focal=60.0)
        camera_b.look_at(
            spherical_to_cartesian(1.0, -40.0, 0.0).astype(np.float32)
        )
        image_b = turned.render(camera_b)

        # Resampling differs slightly between the two paths; the point is that
        # they show the same thing, not that they are bit-identical.
        assert np.abs(image_a.astype(float) - image_b.astype(float)).mean() < 4.0

    def test_a_cube_rotates_with_its_texture(self):
        """
        The rotation is applied to the ray, not to the texture lookup, so a
        cube's walls turn too. If it were applied after intersection the
        texture would slide across fixed walls, and rotating by 90 degrees --
        a symmetry of the cube itself -- would not be a symmetry of the render.
        """
        texture = generate_texture("checker", "cube", 256, {"n_per_face": 4})
        camera = make_camera(width=96, height=96, focal=60.0, position=(0.0, 0.0, 1.0))

        base = Background(texture, "cube", radius=5.0).render(camera)
        quarter = Background(
            texture, "cube", radius=5.0, rotation_deg=90.0
        ).render(camera)

        # All six faces carry the same checker, so a quarter turn about Y maps
        # the cube onto itself.
        assert np.abs(base.astype(float) - quarter.astype(float)).mean() < 1.0


class TestCompositing:
    """The backdrop goes behind the base layer without eating its alpha."""

    @staticmethod
    def _layer(width=64, height=48):
        image = np.zeros((height, width, 4), dtype=np.uint8)
        image[10:30, 10:30] = (255, 0, 0, 255)
        return image

    def test_opaque_fills_alpha_and_keeps_the_subject(self):
        background = Background(flat_texture((0, 200, 0)), opaque=True)
        camera = make_camera()
        out = background.composite(self._layer(), camera)

        assert (out[:, :, 3] == 255).all()
        assert list(out[20, 20, :3]) == [255, 0, 0]
        assert list(out[0, 0, :3]) == [0, 200, 0]

    def test_transparent_mode_preserves_the_mask(self):
        background = Background(flat_texture((0, 200, 0)), opaque=False)
        camera = make_camera()
        out = background.composite(self._layer(), camera)

        assert out[0, 0, 3] == 0
        assert out[20, 20, 3] == 255
        assert list(out[0, 0, :3]) == [0, 200, 0]

    def test_partial_alpha_blends(self):
        """A soft outline edge blends rather than stepping."""
        background = Background(flat_texture((0, 0, 0)), opaque=False)
        camera = make_camera()
        image = np.zeros((48, 64, 4), dtype=np.uint8)
        image[:, :] = (255, 255, 255, 128)

        out = background.composite(image, camera)
        assert out[0, 0, 0] == pytest.approx(128, abs=2)

    def test_size_mismatch_rejected(self):
        background = Background(flat_texture((0, 0, 0)))
        with pytest.raises(ValueError, match="but the camera is"):
            background.composite(self._layer(width=32), make_camera())

    def test_rgb_input_rejected(self):
        background = Background(flat_texture((0, 0, 0)))
        with pytest.raises(ValueError, match="needs an RGBA image"):
            background.composite(np.zeros((48, 64, 3), np.uint8), make_camera())


class TestGenerators:
    """The built-in procedural textures."""

    @pytest.mark.parametrize("name", ["blender_sky", "gradient", "checker", "grid"])
    def test_sphere_shape(self, name):
        texture = generate_texture(name, "sphere", 32)
        assert texture.shape == (32, 64, 3)
        assert texture.dtype == np.uint8

    @pytest.mark.parametrize("name", ["blender_sky", "gradient", "checker", "grid"])
    def test_cube_shape(self, name):
        faces = generate_texture(name, "cube", 32)
        assert len(faces) == 6
        assert all(f.shape == (32, 32, 3) and f.dtype == np.uint8 for f in faces)

    def test_blender_sky_is_blue_above_and_dark_below(self):
        texture = generate_texture("blender_sky", "sphere", 64, {"sun_intensity": 0.0})
        zenith = texture[0].mean(axis=0)
        nadir = texture[-1].mean(axis=0)

        assert zenith[2] > zenith[0]          # more blue than red
        assert nadir.mean() < zenith.mean()   # ground is darker than sky

    @staticmethod
    def _azimuthal_signal(texture):
        """
        Variation along each line of latitude, in 8-bit levels.

        This is the quantity that matters for the whole feature: it is how much
        the backdrop changes as the camera swings around, and so how much cue
        it can possibly give that the camera -- not the subject -- is moving.
        """
        return texture.astype(float).std(axis=1).mean(axis=-1)

    def test_blender_sky_carries_no_azimuthal_signal_without_its_sun(self):
        """
        The documented caveat, pinned as a test: a Nishita sky is rotationally
        symmetric, so on its own it cannot tell a diffusion model anything
        about an orbit. If this ever failed, the warnings carried in the module
        docs, the config template and --help would have gone stale.
        """
        texture = generate_texture(
            "blender_sky", "sphere", 128, {"sun_intensity": 0.0}
        )
        assert self._azimuthal_signal(texture).max() < 0.5

    def test_the_sun_is_the_only_azimuthal_feature(self):
        """It is a real feature, but a local one: most of the sky is flat."""
        rows = self._azimuthal_signal(generate_texture("blender_sky", "sphere", 128))
        assert rows.max() > 5.0
        assert np.median(rows) < 0.1

    def test_checker_carries_far_more_azimuthal_signal_than_the_sky(self):
        """The reason the docs point at checker/grid for validating an orbit."""
        sky = self._azimuthal_signal(generate_texture("blender_sky", "sphere", 128))
        checker = self._azimuthal_signal(generate_texture("checker", "sphere", 128))
        grid = self._azimuthal_signal(generate_texture("grid", "sphere", 128))

        assert checker.mean() > 10 * sky.mean()
        assert grid.mean() > 10 * sky.mean()

    def test_checker_is_two_toned(self):
        texture = generate_texture("checker", "sphere", 64)
        assert len(np.unique(texture.reshape(-1, 3), axis=0)) == 2

    def test_grid_tints_floor_and_ceiling_apart(self):
        faces = generate_texture("grid", "cube", 32)
        floor = faces[CUBE_FACES.index("ny")].mean()
        ceiling = faces[CUBE_FACES.index("py")].mean()
        assert ceiling > floor

    def test_unknown_generator_names_the_alternatives(self):
        with pytest.raises(ValueError, match="Unknown background texture generator"):
            generate_texture("nishita", "sphere", 32)

    def test_unknown_parameter_rejected(self):
        with pytest.raises(ValueError, match="does not accept"):
            generate_texture("checker", "sphere", 32, {"sun_elevation_deg": 10.0})

    def test_parameters_take_effect(self):
        a = generate_texture("checker", "sphere", 64, {"n_azimuth": 4})
        b = generate_texture("checker", "sphere", 64, {"n_azimuth": 32})
        assert not np.array_equal(a, b)

    def test_unknown_geometry_rejected(self):
        with pytest.raises(ValueError, match="Unknown background geometry"):
            generate_texture("checker", "dome", 32)


class TestCreate:
    """Background.create dispatches between generators and files."""

    def test_generator_name(self):
        background = Background.create("grid", "sphere", resolution=64)
        assert background.geometry == "sphere"
        assert background.radius is None

    def test_file_path(self, tmp_path):
        import cv2
        path = tmp_path / "sky.png"
        cv2.imwrite(str(path), np.zeros((64, 128, 3), np.uint8))

        background = Background.create(str(path), "sphere")
        assert background.render(make_camera()).shape == (48, 64, 3)

    def test_params_with_a_file_are_rejected(self, tmp_path):
        """
        Silently ignoring them would leave a config that reads as if it tuned
        the texture while doing nothing at all.
        """
        import cv2
        path = tmp_path / "sky.png"
        cv2.imwrite(str(path), np.zeros((64, 128, 3), np.uint8))

        with pytest.raises(ValueError, match="is a file, not a generator"):
            Background.create(str(path), "sphere", params={"n_azimuth": 4})

    def test_a_name_that_is_neither_names_the_generators(self):
        """
        A near-miss on a generator name is likelier than a genuine missing
        file, so the error lists the built-ins rather than reporting a bad
        path -- "nishita" is a much easier mistake than a typo'd filename.
        """
        with pytest.raises(ValueError, match="neither a built-in generator"):
            Background.create("nishita", "sphere")

    def test_missing_file_lists_the_generators_too(self):
        with pytest.raises(ValueError, match="nor an existing path"):
            Background.create("/nonexistent/sky.png", "sphere")


class TestTextureLoading:
    """Reading textures off disk, in each accepted layout."""

    @staticmethod
    def _write_faces(directory, names):
        import cv2
        for i, name in enumerate(names):
            face = np.full((16, 16, 3), i * 40, dtype=np.uint8)
            cv2.imwrite(str(directory / f"{name}.png"), face)

    def test_directory_of_faces(self, tmp_path):
        self._write_faces(tmp_path, CUBE_FACES)
        faces = load_texture(tmp_path, "cube")
        assert len(faces) == 6
        assert [int(f[0, 0, 0]) for f in faces] == [0, 40, 80, 120, 160, 200]

    def test_directory_accepts_alias_names(self, tmp_path):
        self._write_faces(
            tmp_path, ["right", "left", "top", "bottom", "front", "back"]
        )
        assert len(load_texture(tmp_path, "cube")) == 6

    def test_missing_face_names_what_it_wanted(self, tmp_path):
        self._write_faces(tmp_path, CUBE_FACES[:5])
        with pytest.raises(ValueError, match="No image for cube face 'nz'"):
            load_texture(tmp_path, "cube")

    def test_horizontal_cross(self, tmp_path):
        import cv2
        # Place a unique value at each face's slot in the 4x3 cross.
        size = 16
        cross = np.zeros((3 * size, 4 * size, 3), dtype=np.uint8)
        slots = {"px": (1, 2), "nx": (1, 0), "py": (0, 1),
                 "ny": (2, 1), "pz": (1, 1), "nz": (1, 3)}
        for i, face in enumerate(CUBE_FACES):
            row, col = slots[face]
            cross[row * size:(row + 1) * size, col * size:(col + 1) * size] = i * 40

        path = tmp_path / "cross.png"
        cv2.imwrite(str(path), cross)

        faces = load_texture(path, "cube")
        assert [int(f[0, 0, 0]) for f in faces] == [0, 40, 80, 120, 160, 200]

    def test_strip(self, tmp_path):
        import cv2
        size = 16
        strip = np.zeros((size, 6 * size, 3), dtype=np.uint8)
        for i in range(6):
            strip[:, i * size:(i + 1) * size] = i * 40

        path = tmp_path / "strip.png"
        cv2.imwrite(str(path), strip)
        assert [int(f[0, 0, 0]) for f in load_texture(path, "cube")] == [
            0, 40, 80, 120, 160, 200
        ]

    def test_equirect_is_resampled_onto_a_cube(self, tmp_path):
        import cv2
        path = tmp_path / "sky.png"
        equirect = generate_texture("checker", "sphere", 64)
        cv2.imwrite(str(path), equirect[:, :, ::-1])

        faces = load_texture(path, "cube")
        assert len(faces) == 6
        assert all(f.shape[0] == f.shape[1] for f in faces)

    def test_unusable_aspect_lists_the_layouts(self, tmp_path):
        import cv2
        path = tmp_path / "odd.png"
        cv2.imwrite(str(path), np.zeros((40, 97, 3), np.uint8))
        with pytest.raises(ValueError, match="Cannot infer a cubemap layout"):
            load_texture(path, "cube")

    def test_sphere_requires_a_2_to_1_image(self, tmp_path):
        import cv2
        path = tmp_path / "square.png"
        cv2.imwrite(str(path), np.zeros((64, 64, 3), np.uint8))
        with pytest.raises(ValueError, match="expected 2:1"):
            load_texture(path, "sphere")

    def test_sphere_rejects_a_directory(self, tmp_path):
        with pytest.raises(ValueError, match="is a directory"):
            load_texture(tmp_path, "sphere")

    def test_greyscale_and_alpha_are_normalized(self, tmp_path):
        import cv2
        grey = tmp_path / "grey.png"
        cv2.imwrite(str(grey), np.full((32, 64), 128, np.uint8))
        assert load_texture(grey, "sphere").shape == (32, 64, 3)

        rgba = tmp_path / "rgba.png"
        cv2.imwrite(str(rgba), np.zeros((32, 64, 4), np.uint8))
        assert load_texture(rgba, "sphere").shape == (32, 64, 3)

    def test_equirect_to_cube_preserves_direction(self):
        """A marker in the equirect ends up on the face that direction hits."""
        height = 256
        equirect = np.zeros((height, 2 * height, 3), dtype=np.uint8)
        marker = spherical_to_cartesian(1.0, 90.0, 0.0).astype(np.float32)
        u, v = equirect_uv(marker[None])
        equirect[int(v[0] * height), int(u[0] * 2 * height)] = 255

        faces = equirect_to_cube(equirect, 128)
        brightest = int(np.argmax([f.max() for f in faces]))
        assert CUBE_FACES[brightest] == "px"


class TestSeams:
    """The wrap-around meridian and the poles must not show a seam."""

    def test_no_discontinuity_across_the_wrap_meridian(self):
        """
        The equirect wraps at -Z. Sampled naively that edge shows a hard line,
        which is why the texture is padded with its opposite column.
        """
        texture = generate_texture("gradient", "sphere", 128, {
            "top_color": (1.0, 1.0, 1.0), "bottom_color": (0.0, 0.0, 0.0)
        })
        background = Background(texture, "sphere")

        camera = make_camera(width=128, height=32, focal=40.0)
        camera.look_at(spherical_to_cartesian(1.0, 180.0, 0.0).astype(np.float32))
        image = background.render(camera).astype(float)

        # A vertical gradient has no horizontal structure, so any column-to-
        # column jump is a seam artifact.
        row = image[16, :, 0]
        assert np.abs(np.diff(row)).max() < 3.0

    def test_poles_are_finite(self):
        """Looking straight up must not sample past the top row."""
        background = Background.create("gradient", "sphere", resolution=64)
        camera = make_camera(width=32, height=32, focal=10.0)
        camera.rotation = np.array(
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32
        )
        assert np.isfinite(background.render(camera)).all()


class TestResolutionFitting:
    """
    Over-detailed textures are area-averaged down to the render's own angular
    resolution. Point-sampling a 4K panorama into a small frame resamples
    differently every frame, and the shimmer reads as motion to a video model.
    """

    def test_oversized_equirect_is_downsampled(self):
        background = Background.create("checker", "sphere", resolution=2048)
        assert background._equirect.shape[0] == 2048

        background.render(make_camera(focal=90.0))
        assert background._equirect.shape[0] < 2048

    def test_oversized_faces_are_downsampled(self):
        background = Background.create("checker", "cube", resolution=2048)
        background.render(make_camera(focal=90.0))
        assert background._faces[0].shape[0] < 2048

    def test_already_matched_texture_is_left_alone(self):
        background = Background.create("checker", "sphere", resolution=64)
        background.render(make_camera(focal=600.0))
        assert background._equirect.shape[0] == 64

    def test_fitting_happens_once_for_the_same_camera(self):
        background = Background.create("checker", "sphere", resolution=2048)
        camera = make_camera(focal=90.0)
        background.render(camera)
        fitted = background._equirect.shape[0]
        background.render(camera)
        assert background._equirect.shape[0] == fitted

    def test_a_longer_lens_refits_from_the_source(self):
        """
        render_original_view() typically zooms in relative to the orbit. Fitting
        the already-downsampled copy again would leave the sharper camera stuck
        with detail that had been thrown away for the wider one.
        """
        background = Background.create("checker", "sphere", resolution=2048)
        background.render(make_camera(focal=60.0))
        wide = background._equirect.shape[0]

        background.render(make_camera(focal=240.0))
        assert background._equirect.shape[0] > wide

    def test_a_shorter_lens_does_not_refit(self):
        """Going wider needs no more detail, so the fitted texture stands."""
        background = Background.create("checker", "sphere", resolution=2048)
        background.render(make_camera(focal=240.0))
        narrow = background._equirect.shape[0]

        background.render(make_camera(focal=60.0))
        assert background._equirect.shape[0] == narrow


class TestConstruction:
    """Shape validation, so a malformed texture fails at construction."""

    def test_unknown_geometry(self):
        with pytest.raises(ValueError, match="Unknown background geometry"):
            Background(flat_texture((0, 0, 0)), "dome")

    def test_sphere_needs_three_channels(self):
        with pytest.raises(ValueError, match=r"needs an \(H, W, 3\) image"):
            Background(np.zeros((32, 64), np.uint8), "sphere")

    def test_cube_needs_six_faces(self):
        with pytest.raises(ValueError, match="needs 6 faces"):
            Background([np.zeros((8, 8, 3), np.uint8)] * 4, "cube")

    def test_cube_faces_must_agree(self):
        faces = [np.zeros((8, 8, 3), np.uint8)] * 5 + [np.zeros((4, 4, 3), np.uint8)]
        with pytest.raises(ValueError, match="same size"):
            Background(faces, "cube")

    def test_cube_faces_must_be_square(self):
        with pytest.raises(ValueError, match="must be square"):
            Background([np.zeros((8, 16, 3), np.uint8)] * 6, "cube")

    def test_describe_mentions_the_essentials(self):
        text = Background.create("grid", "cube", resolution=32, radius=4.0).describe()
        assert "cube" in text and "4.0" in text
        assert "infinite" in Background.create("grid", resolution=32).describe()
