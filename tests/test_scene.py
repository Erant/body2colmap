"""
Tests for scene module: torso facing direction and Y-axis rotation.
"""

import numpy as np
import pytest

from body2colmap.scene import Scene


def _make_scene_with_skeleton(
    skeleton_joints, skeleton_format="mhr70", n_verts=10
):
    """Create a minimal Scene with the given skeleton."""
    rng = np.random.RandomState(0)
    vertices = rng.rand(n_verts, 3).astype(np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    return Scene(
        vertices=vertices,
        faces=faces,
        skeleton_joints=skeleton_joints,
        skeleton_format=skeleton_format,
    )


class TestComputeTorsoFacingDirection:
    """Test torso facing direction computation."""

    def test_facing_negative_z(self):
        """Body facing -Z should return approximately [0, 0, -1]."""
        # MHR70: 5=left_shoulder, 6=right_shoulder, 9=left_hip, 10=right_hip
        joints = np.zeros((70, 3), dtype=np.float32)
        # Shoulders and hips spread along X, body faces -Z
        joints[5] = [0.2, 1.4, 0.0]   # left_shoulder at +X
        joints[6] = [-0.2, 1.4, 0.0]  # right_shoulder at -X
        joints[9] = [0.1, 0.9, 0.0]   # left_hip at +X
        joints[10] = [-0.1, 0.9, 0.0] # right_hip at -X

        scene = _make_scene_with_skeleton(joints, "mhr70")
        facing = scene.compute_torso_facing_direction()

        assert facing is not None
        assert abs(facing[1]) < 1e-6  # No Y component (XZ plane only)
        assert facing[2] < -0.9       # Facing -Z

    def test_facing_positive_x(self):
        """Body facing +X should return approximately [1, 0, 0]."""
        joints = np.zeros((70, 3), dtype=np.float32)
        # Shoulders/hips spread along Z, body faces +X
        joints[5] = [0.0, 1.4, 0.2]   # left_shoulder at +Z
        joints[6] = [0.0, 1.4, -0.2]  # right_shoulder at -Z
        joints[9] = [0.0, 0.9, 0.1]   # left_hip at +Z
        joints[10] = [0.0, 0.9, -0.1] # right_hip at -Z

        scene = _make_scene_with_skeleton(joints, "mhr70")
        facing = scene.compute_torso_facing_direction()

        assert facing is not None
        assert facing[0] > 0.9  # Facing +X

    def test_openpose_format(self):
        """Should work with OpenPose Body25 format too."""
        # OpenPose: 5=LShoulder, 2=RShoulder, 12=LHip, 9=RHip
        joints = np.zeros((25, 3), dtype=np.float32)
        joints[5] = [0.2, 1.4, 0.0]   # LShoulder
        joints[2] = [-0.2, 1.4, 0.0]  # RShoulder
        joints[12] = [0.1, 0.9, 0.0]  # LHip
        joints[9] = [-0.1, 0.9, 0.0]  # RHip

        scene = _make_scene_with_skeleton(joints, "openpose_body25_hands")
        facing = scene.compute_torso_facing_direction()

        assert facing is not None
        assert facing[2] < -0.9

    def test_no_skeleton_returns_none(self):
        """No skeleton → None."""
        vertices = np.random.rand(10, 3).astype(np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        scene = Scene(vertices=vertices, faces=faces)

        assert scene.compute_torso_facing_direction() is None

    def test_unknown_format_returns_none(self):
        """Unknown skeleton format → None."""
        joints = np.zeros((50, 3), dtype=np.float32)
        scene = _make_scene_with_skeleton(joints, "unknown_format")

        assert scene.compute_torso_facing_direction() is None

    def test_result_is_unit_vector(self):
        """Facing direction should be a unit vector."""
        joints = np.zeros((70, 3), dtype=np.float32)
        joints[5] = [0.3, 1.4, 0.1]
        joints[6] = [-0.2, 1.4, -0.05]
        joints[9] = [0.15, 0.9, 0.08]
        joints[10] = [-0.1, 0.9, -0.03]

        scene = _make_scene_with_skeleton(joints, "mhr70")
        facing = scene.compute_torso_facing_direction()

        assert facing is not None
        assert abs(np.linalg.norm(facing) - 1.0) < 1e-5

    def test_y_component_is_zero(self):
        """Facing direction should be in XZ plane (no Y)."""
        joints = np.zeros((70, 3), dtype=np.float32)
        # Shoulders at different heights to make sure Y is projected out
        joints[5] = [0.2, 1.5, 0.0]
        joints[6] = [-0.2, 1.3, 0.0]
        joints[9] = [0.1, 1.0, 0.0]
        joints[10] = [-0.1, 0.8, 0.0]

        scene = _make_scene_with_skeleton(joints, "mhr70")
        facing = scene.compute_torso_facing_direction()

        assert facing is not None
        assert abs(facing[1]) < 1e-6


class TestRotateAroundY:
    """Test Y-axis rotation of scene."""

    def test_180_rotation_flips_z(self):
        """180° rotation should negate X and Z around center."""
        vertices = np.array([
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, -1.0],
        ], dtype=np.float32)
        faces = np.array([[0, 0, 1]], dtype=np.int32)
        scene = Scene(vertices=vertices, faces=faces)

        # bbox center is at [0, 0, 0]
        scene.rotate_around_y(180.0)

        np.testing.assert_allclose(scene.vertices[0], [-1, 0, -1], atol=1e-5)
        np.testing.assert_allclose(scene.vertices[1], [1, 0, 1], atol=1e-5)

    def test_90_rotation(self):
        """90° rotation should send +Z to +X."""
        # Two vertices so bbox center is at origin
        vertices = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ], dtype=np.float32)
        faces = np.array([[0, 0, 1]], dtype=np.int32)
        scene = Scene(vertices=vertices, faces=faces)

        # bbox center = [0, 0, 0]
        # 90° Y rotation: [0,0,1] → [1,0,0]
        scene.rotate_around_y(90.0)

        np.testing.assert_allclose(scene.vertices[0], [1, 0, 0], atol=1e-5)
        np.testing.assert_allclose(scene.vertices[1], [-1, 0, 0], atol=1e-5)

    def test_preserves_y(self):
        """Y coordinates should be unchanged by Y-axis rotation."""
        rng = np.random.RandomState(42)
        vertices = rng.rand(100, 3).astype(np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        original_y = vertices[:, 1].copy()

        scene = Scene(vertices=vertices.copy(), faces=faces)
        scene.rotate_around_y(73.0)

        np.testing.assert_allclose(scene.vertices[:, 1], original_y, atol=1e-5)

    def test_rotates_skeleton_too(self):
        """Skeleton joints should be rotated along with mesh."""
        vertices = np.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([[0, 0, 1]], dtype=np.int32)
        joints = np.array([
            [0.5, 1.0, 0.0],
            [-0.5, 1.0, 0.0],
        ], dtype=np.float32)

        scene = Scene(vertices=vertices, faces=faces,
                      skeleton_joints=joints, skeleton_format="mhr70")
        scene.rotate_around_y(180.0)

        # Center is [0, 0, 0]; 180° flips X and Z
        np.testing.assert_allclose(scene.skeleton_joints[0], [-0.5, 1.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(scene.skeleton_joints[1], [0.5, 1.0, 0.0], atol=1e-5)

    def test_zero_rotation_is_noop(self):
        """0° rotation should not modify anything."""
        vertices = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        faces = np.array([[0, 0, 0]], dtype=np.int32)
        scene = Scene(vertices=vertices.copy(), faces=faces)

        scene.rotate_around_y(0.0)
        np.testing.assert_array_equal(scene.vertices, vertices)

    def test_invalidates_cache(self):
        """Rotation should invalidate cached bounds."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ], dtype=np.float32)
        faces = np.array([[0, 0, 1]], dtype=np.int32)
        scene = Scene(vertices=vertices, faces=faces)

        # Populate cache
        _ = scene.get_bounds()
        assert scene._bounds is not None

        scene.rotate_around_y(45.0)
        assert scene._bounds is None


class TestAutoOrientIntegration:
    """Test the full auto-orient flow on Scene."""

    def test_auto_orient_faces_negative_z(self):
        """After auto-orient, torso should face -Z."""
        # Body initially facing +X
        joints = np.zeros((70, 3), dtype=np.float32)
        joints[5] = [0.0, 1.4, 0.2]   # left_shoulder
        joints[6] = [0.0, 1.4, -0.2]  # right_shoulder
        joints[9] = [0.0, 0.9, 0.1]   # left_hip
        joints[10] = [0.0, 0.9, -0.1] # right_hip

        scene = _make_scene_with_skeleton(joints, "mhr70")

        # Verify initially facing +X
        facing_before = scene.compute_torso_facing_direction()
        assert facing_before[0] > 0.9

        # Compute correction and rotate
        current_angle = float(np.arctan2(facing_before[0], facing_before[2]))
        target_angle = float(np.arctan2(0.0, -1.0))
        correction = np.degrees(target_angle - current_angle)
        scene.rotate_around_y(correction)

        # Should now face -Z
        facing_after = scene.compute_torso_facing_direction()
        assert facing_after[2] < -0.9
        assert abs(facing_after[0]) < 0.1


def _make_full_skeleton():
    """Create a realistic MHR70 skeleton for camera height tests."""
    joints = np.zeros((70, 3), dtype=np.float32)
    # Feet / ankles
    joints[13] = [-0.1, 0.05, 0.0]   # left_ankle
    joints[14] = [0.1, 0.05, 0.0]    # right_ankle
    # Knees
    joints[11] = [-0.1, 0.45, 0.0]   # left_knee
    joints[12] = [0.1, 0.45, 0.0]    # right_knee
    # Hips
    joints[9] = [-0.1, 0.90, 0.0]    # left_hip
    joints[10] = [0.1, 0.90, 0.0]    # right_hip
    # Shoulders
    joints[5] = [-0.2, 1.40, 0.0]    # left_shoulder
    joints[6] = [0.2, 1.40, 0.0]     # right_shoulder
    # Neck
    joints[69] = [0.0, 1.55, 0.0]    # neck
    # Head
    joints[0] = [0.0, 1.70, 0.0]     # nose
    return joints


class TestGetCameraHeightY:
    """Test skeleton-based camera orbit height."""

    def test_bbox_center(self):
        """bbox_center should return Y center of bounding box."""
        joints = _make_full_skeleton()
        scene = _make_scene_with_skeleton(joints, "mhr70")
        # Override vertices to have predictable bounds
        scene.vertices = np.array([
            [-0.3, 0.0, -0.1],
            [0.3, 1.8, 0.1],
        ], dtype=np.float32)
        scene._bounds = None

        y = scene.get_camera_height_y("bbox_center")
        assert abs(y - 0.9) < 1e-5  # (0 + 1.8) / 2

    def test_feet(self):
        """feet should return average ankle Y."""
        joints = _make_full_skeleton()
        scene = _make_scene_with_skeleton(joints, "mhr70")
        y = scene.get_camera_height_y("feet")
        assert abs(y - 0.05) < 1e-5

    def test_knees(self):
        """knees should return average knee Y."""
        joints = _make_full_skeleton()
        scene = _make_scene_with_skeleton(joints, "mhr70")
        y = scene.get_camera_height_y("knees")
        assert abs(y - 0.45) < 1e-5

    def test_waist(self):
        """waist should return average hip Y."""
        joints = _make_full_skeleton()
        scene = _make_scene_with_skeleton(joints, "mhr70")
        y = scene.get_camera_height_y("waist")
        assert abs(y - 0.90) < 1e-5

    def test_chest(self):
        """chest should return midpoint between hips and shoulders."""
        joints = _make_full_skeleton()
        scene = _make_scene_with_skeleton(joints, "mhr70")
        y = scene.get_camera_height_y("chest")
        expected = (0.90 + 1.40) / 2.0  # midpoint hip-shoulder
        assert abs(y - expected) < 1e-5

    def test_shoulders(self):
        """shoulders should return average shoulder Y."""
        joints = _make_full_skeleton()
        scene = _make_scene_with_skeleton(joints, "mhr70")
        y = scene.get_camera_height_y("shoulders")
        assert abs(y - 1.40) < 1e-5

    def test_head(self):
        """head should return nose Y."""
        joints = _make_full_skeleton()
        scene = _make_scene_with_skeleton(joints, "mhr70")
        y = scene.get_camera_height_y("head")
        assert abs(y - 1.70) < 1e-5

    def test_no_skeleton_raises(self):
        """Skeleton-based preset without skeleton should raise ValueError."""
        vertices = np.random.rand(10, 3).astype(np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        scene = Scene(vertices=vertices, faces=faces)

        with pytest.raises(ValueError, match="requires skeleton data"):
            scene.get_camera_height_y("waist")

    def test_unknown_preset_raises(self):
        """Unknown preset should raise ValueError."""
        joints = _make_full_skeleton()
        scene = _make_scene_with_skeleton(joints, "mhr70")

        with pytest.raises(ValueError, match="Unknown camera height preset"):
            scene.get_camera_height_y("belly_button")

    def test_non_mhr70_raises(self):
        """Non-MHR70 skeleton format should raise ValueError."""
        joints = np.zeros((25, 3), dtype=np.float32)
        scene = _make_scene_with_skeleton(joints, "openpose_body25")

        with pytest.raises(ValueError, match="MHR70"):
            scene.get_camera_height_y("waist")
