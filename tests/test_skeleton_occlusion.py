"""
Tests for the skeleton's occlusion test — `Renderer._joints_visible`, the
rule `render_skeleton(occlusion_tolerance=...)` draws by.

A joint is hidden when the mesh's nearest surface at its pixel is more than
the tolerance in front of it, and the tolerance is scaled per joint class
(`OCCLUSION_DEPTH_SCALE_BODY25`) because a hip sits far deeper in its own
flesh than a wrist. No GL: the depth buffer is hand-built.
"""

import numpy as np
import pytest

from body2colmap.camera import Camera
from body2colmap.renderer import Renderer
from body2colmap.skeleton import OCCLUSION_DEPTH_SCALE_BODY25


SIZE = (40, 30)  # (width, height)


class _FakeDepthRenderer(Renderer):
    def __init__(self, depth):
        self.width, self.height = depth.shape[1], depth.shape[0]
        self.background = None
        self._depth = depth

    def _render_depth_buffer(self, camera):
        return self._depth


def _camera():
    # At the origin looking down -Z, principal point at the frame's centre.
    return Camera(focal_length=(20.0, 20.0), image_size=SIZE,
                  position=np.zeros(3, np.float32),
                  rotation=np.eye(3, dtype=np.float32))


def _joint(depth, dx=0.0, dy=0.0):
    """A world point `depth` metres in front of the camera, offset in x/y."""
    return np.array([dx, dy, -depth], dtype=np.float32)


class TestVisibility:
    def test_a_joint_inside_its_own_limb_is_seen(self):
        depth = np.full((SIZE[1], SIZE[0]), 1.0, np.float32)
        renderer = _FakeDepthRenderer(depth)
        joints = np.stack([_joint(1.05), _joint(1.20)])
        visible = renderer._joints_visible(joints, _camera(), tolerance=0.1)
        assert visible.tolist() == [True, False]

    def test_the_tolerance_is_scaled_per_joint(self):
        depth = np.full((SIZE[1], SIZE[0]), 1.0, np.float32)
        renderer = _FakeDepthRenderer(depth)
        joints = np.stack([_joint(1.15)] * 3)
        visible = renderer._joints_visible(
            joints, _camera(), tolerance=0.1, depth_scale={0: 0.5, 1: 2.0}
        )
        # 15 cm behind: hidden at x0.5 (5 cm) and x1 (10 cm), seen at x2.
        assert visible.tolist() == [False, True, False]

    def test_the_body25_table_keeps_the_hips_and_neck_and_drops_the_nose(self):
        """The measured depths of a SAM-3D-Body fit: hips ~20 cm inside the
        pelvis, the neck ~27 cm behind the near shoulder in profile, both
        still found by a 2-D detector; the nose 30 cm behind the back of
        the head is not."""
        depth = np.full((SIZE[1], SIZE[0]), 1.0, np.float32)
        renderer = _FakeDepthRenderer(depth)
        joints = np.zeros((25, 3), np.float32)
        joints[:] = _joint(1.05)
        joints[1] = _joint(1.27)   # Neck
        joints[9] = _joint(1.20)   # RHip
        joints[0] = _joint(1.30)   # Nose
        joints[4] = _joint(1.40)   # RWrist, behind the torso
        visible = renderer._joints_visible(
            joints, _camera(), tolerance=0.12,
            depth_scale=OCCLUSION_DEPTH_SCALE_BODY25,
        )
        assert visible[1] and visible[9]
        assert not visible[0] and not visible[4]
        assert visible[2]

    def test_nothing_hides_a_joint_off_the_mesh_or_off_the_frame(self):
        depth = np.zeros((SIZE[1], SIZE[0]), np.float32)
        depth[:, :20] = 0.5  # the left half is covered, close
        renderer = _FakeDepthRenderer(depth)
        joints = np.stack([
            _joint(2.0, dx=-0.5),   # left of centre: on the mesh -> hidden
            _joint(2.0, dx=0.5),    # right: no mesh there -> seen
            _joint(2.0, dx=50.0),   # off the frame -> seen
            _joint(-1.0),           # behind the camera -> seen (not tested)
        ])
        visible = renderer._joints_visible(joints, _camera(), tolerance=0.1)
        assert visible.tolist() == [False, True, True, True]

    def test_a_negative_tolerance_is_refused(self):
        renderer = _FakeDepthRenderer(np.zeros((SIZE[1], SIZE[0]), np.float32))
        with pytest.raises(ValueError, match="occlusion_tolerance"):
            renderer._joints_visible(np.zeros((1, 3), np.float32), _camera(), -0.1)

    def test_no_joints(self):
        renderer = _FakeDepthRenderer(np.zeros((SIZE[1], SIZE[0]), np.float32))
        assert renderer._joints_visible(np.zeros((0, 3), np.float32), _camera(), 0.1).shape == (0,)


class TestTheTable:
    def test_torso_deep_face_shallow_limbs_unit(self):
        table = OCCLUSION_DEPTH_SCALE_BODY25
        assert table[1] > table[2] > 1.0            # neck deeper than shoulder
        assert table[8] == table[9] == table[12]    # the three hips alike
        assert all(table[i] < 1.0 for i in (0, 15, 16, 17, 18))
        assert all(i not in table for i in (3, 4, 6, 7, 10, 11, 13, 14))
