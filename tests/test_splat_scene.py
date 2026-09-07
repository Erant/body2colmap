"""
Tests for SplatScene's PLY round trip.

The property that matters is losslessness: a .ply loaded and written back
carries every vertex column it came with. A trainer attaches per-Gaussian
data the renderer needs (brush's ``ev_*`` evidence block, which
``brush-splat-render --confidence`` gates on), and silently dropping it
turned that gate into plain alpha once.
"""

import numpy as np
import pytest
from plyfile import PlyData

from body2colmap.splat_anchor import transform_splat_scene
from body2colmap.splat_scene import SplatScene

EVIDENCE = ["ev_w_in", "ev_w_all", "ev_err", "ev_views",
            "ev_dir_0", "ev_dir_1", "ev_dir_2"]


def _scene(n=32, sh_degree=3, seed=0, extras=None):
    rng = np.random.default_rng(seed)
    quats = rng.normal(size=(n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    k = (sh_degree + 1) ** 2
    return SplatScene(
        means=rng.normal(size=(n, 3)).astype(np.float32),
        scales=rng.uniform(-6.0, -3.0, size=(n, 3)).astype(np.float32),
        quats=quats,
        opacities=rng.normal(size=n).astype(np.float32),
        sh_coeffs=rng.normal(size=(n, k, 3)).astype(np.float32),
        sh_degree=sh_degree,
        extras=extras,
    )


def _evidence(n, seed=1):
    rng = np.random.default_rng(seed)
    return {name: rng.uniform(size=n).astype(np.float32) for name in EVIDENCE}


def _columns(path):
    return list(PlyData.read(str(path))["vertex"].data.dtype.names)


# --- round trip -------------------------------------------------------------

def test_round_trip_keeps_evidence_block(tmp_path):
    n = 32
    scene = _scene(n, extras=_evidence(n))
    path = tmp_path / "scene.ply"
    scene.to_ply(str(path))

    columns = _columns(path)
    assert columns[-7:] == EVIDENCE, "extras follow the standard columns, in order"

    loaded = SplatScene.from_ply(str(path))
    assert list(loaded.extras) == EVIDENCE
    for name in EVIDENCE:
        np.testing.assert_array_equal(loaded.extras[name], scene.extras[name])
        assert loaded.extras[name].dtype == np.float32

    # And the standard fields still come back intact.
    np.testing.assert_array_equal(loaded.means, scene.means)
    np.testing.assert_array_equal(loaded.sh_coeffs, scene.sh_coeffs)
    assert loaded.sh_degree == 3


def test_second_round_trip_is_stable(tmp_path):
    """Load-save-load-save: same columns, same extras (from_ply renormalises
    quaternions, so this is a column and value check, not a byte one)."""
    n = 16
    scene = _scene(n, extras=_evidence(n))
    a, b = tmp_path / "a.ply", tmp_path / "b.ply"
    scene.to_ply(str(a))
    SplatScene.from_ply(str(a)).to_ply(str(b))
    assert _columns(a) == _columns(b)
    again = SplatScene.from_ply(str(b))
    for name in EVIDENCE:
        np.testing.assert_array_equal(again.extras[name], scene.extras[name])


def test_plain_scene_writes_only_standard_columns(tmp_path):
    scene = _scene(8, sh_degree=1)
    path = tmp_path / "plain.ply"
    scene.to_ply(str(path))
    columns = _columns(path)
    assert not any(c.startswith("ev_") for c in columns)
    assert columns[:14] == ["x", "y", "z", "scale_0", "scale_1", "scale_2",
                            "rot_0", "rot_1", "rot_2", "rot_3", "opacity",
                            "f_dc_0", "f_dc_1", "f_dc_2"]
    assert SplatScene.from_ply(str(path)).extras == {}


def test_extras_keep_their_dtype(tmp_path):
    n = 8
    scene = _scene(n, sh_degree=0, extras={
        "label": np.arange(n, dtype=np.int32),
        "weight": np.linspace(0, 1, n, dtype=np.float64),
    })
    path = tmp_path / "typed.ply"
    scene.to_ply(str(path))
    loaded = SplatScene.from_ply(str(path))
    assert loaded.extras["label"].dtype == np.int32
    assert loaded.extras["weight"].dtype == np.float64
    np.testing.assert_array_equal(loaded.extras["label"], np.arange(n))


def test_loaded_extras_are_owned_copies(tmp_path):
    """Mutating the scene must not depend on plyfile's buffer staying alive."""
    n = 8
    _scene(n, sh_degree=0, extras=_evidence(n)).to_ply(str(tmp_path / "s.ply"))
    loaded = SplatScene.from_ply(str(tmp_path / "s.ply"))
    for arr in loaded.extras.values():
        assert arr.flags.owndata or arr.base is None or isinstance(arr.base, np.ndarray)
        arr[0] = 42.0  # writable


# --- validation ---------------------------------------------------------------

def test_extra_must_be_one_value_per_gaussian():
    with pytest.raises(ValueError, match="one value per Gaussian"):
        _scene(8, sh_degree=0, extras={"ev_w_in": np.zeros(7, np.float32)})
    with pytest.raises(ValueError, match="one value per Gaussian"):
        _scene(8, sh_degree=0, extras={"ev_dir": np.zeros((8, 3), np.float32)})


def test_extra_may_not_shadow_a_standard_column():
    with pytest.raises(ValueError, match="collides"):
        _scene(8, sh_degree=0, extras={"opacity": np.zeros(8, np.float32)})
    with pytest.raises(ValueError, match="collides"):
        _scene(8, sh_degree=0, extras={"f_rest_0": np.zeros(8, np.float32)})


def test_repr_names_extras():
    assert "extras=['ev_w_in'" in repr(_scene(4, sh_degree=0, extras=_evidence(4)))
    assert "extras" not in repr(_scene(4, sh_degree=0))


# --- anchoring ------------------------------------------------------------------

@pytest.mark.parametrize("M", [
    np.eye(3) * 2.0,                                   # uniform-scale fast path
    np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]) * 1.5,  # general
])
def test_transform_carries_extras_through(M):
    n = 16
    scene = _scene(n, sh_degree=0, extras=_evidence(n))
    world = transform_splat_scene(scene, M, np.zeros(3))
    assert list(world.extras) == EVIDENCE
    for name in EVIDENCE:
        np.testing.assert_array_equal(world.extras[name], scene.extras[name])
        assert world.extras[name] is not scene.extras[name], "a copy, not a view"
