"""
Tests for anchoring an externally-produced Gaussian splat to the world frame.

The load-bearing property is the reprojection identity: a Gaussian built from
pixel (u, v) of the photo must, after anchoring, project back onto that same
pixel under SAM-3D-Body's intrinsics. Everything else the feature does rests on
it, so it is tested directly rather than through a rendered image.
"""

import json

import numpy as np
import pytest

from body2colmap.camera import Camera
from body2colmap.renderer import parse_composite_modes
from body2colmap.splat_anchor import (
    compute_anchor_transform,
    estimate_depth_scale,
    load_splat_meta,
    transform_splat_scene,
)
from body2colmap.splat_scene import SplatScene


# --- fixtures ---------------------------------------------------------------

FULL_W, FULL_H = 757, 1536
SAM3D_FOCAL = 1509.33
CROP_BOX = (141, 0, 594, 477)

SPLAT_META = {
    "width": 453,
    "height": 477,
    "intrinsics": {"f": 1122.03, "cx": 224.89, "cy": 284.39},
    "centroid": [0.0087, 0.0753, -1.2168],
}


def _random_splat(n=64, seed=0, sh_degree=0):
    """A splat scene in the source frame: in front of the camera, facing it."""
    rng = np.random.default_rng(seed)
    centroid = np.asarray(SPLAT_META["centroid"])
    # Spread around the origin of the splat frame, i.e. around the centroid in
    # camera coordinates. Negative Z in camera coords = in front of the camera.
    means = rng.normal(0.0, 0.06, size=(n, 3)).astype(np.float32)

    quats = rng.normal(size=(n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)

    n_coeffs = (sh_degree + 1) ** 2
    return SplatScene(
        means=means,
        scales=rng.uniform(-9.0, -6.0, size=(n, 3)).astype(np.float32),
        quats=quats,
        opacities=rng.normal(size=n).astype(np.float32),
        sh_coeffs=rng.normal(size=(n, n_coeffs, 3)).astype(np.float32),
        sh_degree=sh_degree,
    ), centroid


def _project_with_splat_intrinsics(means, centroid):
    """Where each Gaussian sits in the full photo, per the splat's own model."""
    meta = SPLAT_META
    q = np.asarray(means, np.float64) + centroid
    cv = q * np.array([1.0, -1.0, -1.0])          # OpenGL -> OpenCV
    uv = cv[:, :2] / cv[:, 2:3] * meta["intrinsics"]["f"] + np.array(
        [meta["intrinsics"]["cx"], meta["intrinsics"]["cy"]]
    )
    r = (CROP_BOX[2] - CROP_BOX[0]) / meta["width"]
    return uv * r + np.array([CROP_BOX[0], CROP_BOX[1]])


def _covariances(scene):
    from body2colmap.splat_anchor import _quats_to_matrices

    R = _quats_to_matrices(scene.quats)
    sigma = np.exp(np.asarray(scene.scales, np.float64))
    RS = R * sigma[:, None, :]
    return RS @ np.transpose(RS, (0, 2, 1))


# --- the reprojection identity ---------------------------------------------


@pytest.mark.parametrize("scale", [1.0, 0.6, 1.3726, 3.0])
def test_anchored_splat_reprojects_onto_its_source_pixels(scale):
    """The whole feature rests on this: 2-D alignment is exact by construction."""
    scene, centroid = _random_splat()
    expected = _project_with_splat_intrinsics(scene.means, centroid)

    M, cen = compute_anchor_transform(
        SPLAT_META, SAM3D_FOCAL, (FULL_W, FULL_H), crop_box=CROP_BOX, scale=scale
    )
    world = transform_splat_scene(scene, M, cen)

    camera = Camera(
        focal_length=(SAM3D_FOCAL, SAM3D_FOCAL), image_size=(FULL_W, FULL_H)
    )
    got = camera.project(world.means.astype(np.float32))

    assert np.abs(got - expected).max() < 0.01


def test_scale_does_not_move_the_projection():
    """The depth gauge is invisible at the anchor frame -- that is what makes it a gauge."""
    scene, _ = _random_splat(seed=3)
    camera = Camera(
        focal_length=(SAM3D_FOCAL, SAM3D_FOCAL), image_size=(FULL_W, FULL_H)
    )

    projections = []
    for scale in (0.5, 1.0, 2.0):
        M, cen = compute_anchor_transform(
            SPLAT_META, SAM3D_FOCAL, (FULL_W, FULL_H), crop_box=CROP_BOX, scale=scale
        )
        world = transform_splat_scene(scene, M, cen)
        projections.append(camera.project(world.means.astype(np.float32)))

    assert np.abs(projections[1] - projections[0]).max() < 0.01
    assert np.abs(projections[2] - projections[0]).max() < 0.01


def test_scale_scales_depth_linearly():
    scene, _ = _random_splat(seed=4)
    depths = []
    for scale in (1.0, 2.5):
        M, cen = compute_anchor_transform(
            SPLAT_META, SAM3D_FOCAL, (FULL_W, FULL_H), crop_box=CROP_BOX, scale=scale
        )
        depths.append(-transform_splat_scene(scene, M, cen).means[:, 2])

    np.testing.assert_allclose(depths[1], depths[0] * 2.5, rtol=1e-5)


def test_transform_is_re_unprojection_at_gauge_one():
    """
    At s=1 the map must reproduce "unproject the same depths with f_m".

    This is the justification for the anisotropy, so it is worth pinning.
    """
    scene, centroid = _random_splat(seed=7)
    M, cen = compute_anchor_transform(
        SPLAT_META, SAM3D_FOCAL, (FULL_W, FULL_H), crop_box=CROP_BOX, scale=1.0
    )
    world = transform_splat_scene(scene, M, cen)

    uv = _project_with_splat_intrinsics(scene.means, centroid)
    depth = -(np.asarray(scene.means, np.float64) + centroid)[:, 2]
    expected_cv = np.stack([
        (uv[:, 0] - FULL_W / 2.0) * depth / SAM3D_FOCAL,
        (uv[:, 1] - FULL_H / 2.0) * depth / SAM3D_FOCAL,
        depth,
    ], axis=1)
    expected = expected_cv * np.array([1.0, -1.0, -1.0])  # back to OpenGL

    np.testing.assert_allclose(world.means, expected, atol=1e-5)


# --- covariance re-decomposition -------------------------------------------


def test_covariance_transforms_as_M_Sigma_MT():
    scene, _ = _random_splat(n=200, seed=11)
    M, cen = compute_anchor_transform(
        SPLAT_META, SAM3D_FOCAL, (FULL_W, FULL_H), crop_box=CROP_BOX, scale=1.4
    )
    world = transform_splat_scene(scene, M, cen)

    expected = M @ _covariances(scene) @ M.T
    # Tolerance is scaled to the size of the covariance rather than applied
    # per-element. A SplatScene stores log-scales and quaternions as float32,
    # which is ~1e-7 relative; on the near-cancelling off-diagonal terms (three
    # orders of magnitude below the diagonal, since these Gaussians are flat
    # sub-millimetre discs) that amplifies to ~1e-5 relative. Judging each
    # element against its own magnitude would be measuring the cancellation,
    # not the transform.
    np.testing.assert_allclose(
        _covariances(world), expected,
        rtol=1e-4, atol=1e-6 * np.abs(expected).max(),
    )


def test_output_rotations_are_proper_and_scales_descending():
    from body2colmap.splat_anchor import _quats_to_matrices

    scene, _ = _random_splat(n=200, seed=12)
    M, cen = compute_anchor_transform(
        SPLAT_META, SAM3D_FOCAL, (FULL_W, FULL_H), crop_box=CROP_BOX, scale=1.4
    )
    world = transform_splat_scene(scene, M, cen)

    np.testing.assert_allclose(
        np.linalg.norm(world.quats, axis=1), 1.0, atol=1e-6
    )
    dets = np.linalg.det(_quats_to_matrices(world.quats))
    np.testing.assert_allclose(dets, 1.0, atol=1e-6)

    # Smallest axis stays in column 2, keeping the upstream
    # "surface normal in the shortest-scale axis" convention.
    assert np.all(np.diff(world.scales, axis=1) <= 1e-6)


def test_uniform_scale_fast_path_matches_the_general_path():
    """
    With matching intrinsics the map is diag(s,s,s) and quats are untouched.

    The fast path exists to skip re-decomposing every rotation, so it has to
    agree with the general result it replaces.
    """
    scene, _ = _random_splat(n=100, seed=13)
    centred_meta = {
        "width": FULL_W,
        "height": FULL_H,
        "intrinsics": {"f": SAM3D_FOCAL, "cx": FULL_W / 2.0, "cy": FULL_H / 2.0},
        "centroid": SPLAT_META["centroid"],
    }
    M, cen = compute_anchor_transform(
        centred_meta, SAM3D_FOCAL, (FULL_W, FULL_H), scale=1.7
    )
    np.testing.assert_allclose(M, np.eye(3) * 1.7, atol=1e-12)

    fast = transform_splat_scene(scene, M, cen)
    # Nudge M off-diagonal by less than nothing is impossible, so compare
    # covariances instead: those are basis-independent.
    np.testing.assert_allclose(
        _covariances(fast), M @ _covariances(scene) @ M.T, rtol=1e-6
    )
    np.testing.assert_allclose(fast.quats, scene.quats, atol=0)


def test_no_reconcile_gives_a_pure_uniform_scale():
    M, _ = compute_anchor_transform(
        SPLAT_META, SAM3D_FOCAL, (FULL_W, FULL_H), crop_box=CROP_BOX,
        scale=1.4, reconcile_intrinsics=False,
    )
    np.testing.assert_allclose(M, np.eye(3) * 1.4, atol=1e-12)


# --- guards -----------------------------------------------------------------


def test_higher_order_sh_is_refused():
    scene, _ = _random_splat(sh_degree=3)
    M, cen = compute_anchor_transform(
        SPLAT_META, SAM3D_FOCAL, (FULL_W, FULL_H), crop_box=CROP_BOX, scale=1.0
    )
    with pytest.raises(ValueError, match="SH degree 0"):
        transform_splat_scene(scene, M, cen)


def test_size_mismatch_without_a_crop_box_is_refused():
    with pytest.raises(ValueError, match="crop box"):
        compute_anchor_transform(SPLAT_META, SAM3D_FOCAL, (FULL_W, FULL_H))


def test_non_uniform_crop_resize_is_refused():
    with pytest.raises(ValueError, match="non-uniform resize"):
        compute_anchor_transform(
            SPLAT_META, SAM3D_FOCAL, (FULL_W, FULL_H), crop_box=(0, 0, 906, 477)
        )


def test_degenerate_crop_box_is_refused():
    with pytest.raises(ValueError, match="Degenerate"):
        compute_anchor_transform(
            SPLAT_META, SAM3D_FOCAL, (FULL_W, FULL_H), crop_box=(100, 0, 100, 477)
        )


def test_load_splat_meta_reports_missing_keys(tmp_path):
    path = tmp_path / "splat_meta.json"
    path.write_text(json.dumps({"width": 10, "height": 10}))
    with pytest.raises(ValueError, match="centroid"):
        load_splat_meta(str(path))

    path.write_text(json.dumps({**SPLAT_META, "intrinsics": {"f": 1.0}}))
    with pytest.raises(ValueError, match="cx"):
        load_splat_meta(str(path))


def test_load_splat_meta_roundtrips(tmp_path):
    path = tmp_path / "splat_meta.json"
    path.write_text(json.dumps(SPLAT_META))
    assert load_splat_meta(str(path))["intrinsics"]["f"] == SPLAT_META["intrinsics"]["f"]


# --- depth gauge ------------------------------------------------------------


def _flat_sheet_depth_map(camera, depth):
    """A synthetic 'mesh' depth buffer: a fronto-parallel plane."""
    return np.full((camera.height, camera.width), depth, dtype=np.float32)


def test_estimate_depth_scale_recovers_a_known_ratio():
    camera = Camera(focal_length=(500.0, 500.0), image_size=(64, 64))
    rng = np.random.default_rng(5)

    # Gaussians on a plane 2 m in front of the camera (world Z = -2), spread
    # over a patch that lands well inside the 64x64 frame.
    means = np.column_stack([
        rng.uniform(-0.1, 0.1, 2000),
        rng.uniform(-0.1, 0.1, 2000),
        np.full(2000, -2.0),
    ])
    mesh_depth = _flat_sheet_depth_map(camera, 3.0)

    assert estimate_depth_scale(means, camera, mesh_depth) == pytest.approx(1.5, rel=1e-6)


def test_estimate_depth_scale_is_robust_to_silhouette_outliers():
    """A few pixels pairing the face against the background must not move the fit."""
    camera = Camera(focal_length=(500.0, 500.0), image_size=(64, 64))
    rng = np.random.default_rng(6)
    means = np.column_stack([
        rng.uniform(-0.1, 0.1, 2000),
        rng.uniform(-0.1, 0.1, 2000),
        np.full(2000, -2.0),
    ])
    mesh_depth = _flat_sheet_depth_map(camera, 3.0)
    # A handful of wildly wrong "background" depths.
    mesh_depth[:3, :] = 40.0

    assert estimate_depth_scale(means, camera, mesh_depth) == pytest.approx(1.5, rel=1e-3)


def test_estimate_depth_scale_refuses_a_size_mismatch():
    camera = Camera(focal_length=(500.0, 500.0), image_size=(64, 64))
    means = np.array([[0.0, 0.0, -2.0]])
    with pytest.raises(ValueError, match="but the camera is"):
        estimate_depth_scale(means, camera, np.ones((32, 32), np.float32))


def test_estimate_depth_scale_refuses_too_little_overlap():
    camera = Camera(focal_length=(500.0, 500.0), image_size=(64, 64))
    means = np.array([[0.0, 0.0, -2.0]])
    with pytest.raises(ValueError, match="overlap"):
        estimate_depth_scale(means, camera, _flat_sheet_depth_map(camera, 3.0))


# --- mode parsing -----------------------------------------------------------


@pytest.mark.parametrize("mode_str,expected", [
    ("mesh", ("mesh", [])),
    ("skeleton+splat", ("skeleton", ["splat"])),
    ("depth+skeleton+face", ("depth", ["skeleton", "face"])),
    ("outline + skeleton", ("outline", ["skeleton"])),
])
def test_parse_composite_modes(mode_str, expected):
    assert parse_composite_modes(mode_str) == expected


@pytest.mark.parametrize("mode_str,match", [
    ("face+skeleton", "cannot be the base layer"),
    ("mesh+bogus", "Unknown render layer"),
    ("skeleton++face", "Empty layer"),
    ("skeleton+skeleton", "repeated"),
    ("skeleton+mesh", "cannot be an overlay"),
])
def test_parse_composite_modes_rejects(mode_str, match):
    with pytest.raises(ValueError, match=match):
        parse_composite_modes(mode_str)
