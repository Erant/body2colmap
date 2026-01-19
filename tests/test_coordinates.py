"""
Tests for coordinate system conversions.

This is a HIGH PRIORITY test module - coordinate transforms must be correct
for the entire system to work.
"""

import numpy as np
import pytest

from body2colmap import coordinates


class TestWorldCoordinates:
    """Test coordinate system definitions."""

    def test_axes_are_unit_vectors(self):
        """Coordinate axes should be unit length."""
        assert np.allclose(np.linalg.norm(coordinates.WorldCoordinates.UP_AXIS), 1.0)
        assert np.allclose(np.linalg.norm(coordinates.WorldCoordinates.FORWARD_AXIS), 1.0)
        assert np.allclose(np.linalg.norm(coordinates.WorldCoordinates.RIGHT_AXIS), 1.0)

    def test_axes_are_orthogonal(self):
        """Coordinate axes should be orthogonal (right-handed system)."""
        up = coordinates.WorldCoordinates.UP_AXIS
        forward = coordinates.WorldCoordinates.FORWARD_AXIS
        right = coordinates.WorldCoordinates.RIGHT_AXIS

        # Right cross forward should give up (or -up depending on convention)
        # Actually: right, up, -forward should form right-handed system
        # So: right x up = -forward
        assert np.allclose(np.cross(right, up), -forward)


class TestSAM3DToWorld:
    """Test SAM-3D-Body to world conversion."""

    def test_simple_translation(self):
        """Vertices should be translated by cam_t."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=np.float32)
        cam_t = np.array([0, 0, 5], dtype=np.float32)

        result = coordinates.sam3d_to_world(vertices, cam_t)

        expected = vertices + cam_t
        assert np.allclose(result, expected)

    def test_preserves_shape(self):
        """Output should have same shape as input."""
        vertices = np.random.randn(100, 3).astype(np.float32)
        cam_t = np.array([0, 0, 5], dtype=np.float32)

        result = coordinates.sam3d_to_world(vertices, cam_t)

        assert result.shape == vertices.shape


class TestRotationToQuaternion:
    """Test rotation matrix to quaternion conversion."""

    def test_identity_rotation(self):
        """Identity rotation should give quaternion [1, 0, 0, 0]."""
        R = np.eye(3, dtype=np.float32)
        quat = coordinates.rotation_to_quaternion_wxyz(R)

        expected = np.array([1, 0, 0, 0], dtype=np.float32)
        assert np.allclose(quat, expected)

    def test_90deg_rotation_around_z(self):
        """90° rotation around Z axis."""
        # Rotation matrix for 90° around Z
        R = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ], dtype=np.float32)

        quat = coordinates.rotation_to_quaternion_wxyz(R)

        # Expected quaternion: [cos(45°), 0, 0, sin(45°)]
        # = [√2/2, 0, 0, √2/2]
        sqrt2_over_2 = np.sqrt(2) / 2
        expected = np.array([sqrt2_over_2, 0, 0, sqrt2_over_2], dtype=np.float32)

        assert np.allclose(quat, expected, atol=1e-6)

    def test_quaternion_is_unit(self):
        """Quaternion should have unit length."""
        R = np.random.randn(3, 3).astype(np.float32)
        # Orthonormalize (QR decomposition)
        R, _ = np.linalg.qr(R)

        quat = coordinates.rotation_to_quaternion_wxyz(R)

        assert np.allclose(np.linalg.norm(quat), 1.0)


class TestLookAtMatrix:
    """Test look-at matrix construction."""

    def test_camera_at_origin_looking_at_z(self):
        """Camera at origin looking down -Z should give identity rotation."""
        eye = np.array([0, 0, 0], dtype=np.float32)
        target = np.array([0, 0, -1], dtype=np.float32)

        c2w = coordinates.look_at_matrix(eye, target)

        # Rotation should be identity (camera aligned with world)
        assert np.allclose(c2w[:3, :3], np.eye(3), atol=1e-6)

        # Position should be eye
        assert np.allclose(c2w[:3, 3], eye)

    def test_forward_vector_points_to_target(self):
        """Camera forward direction should point toward target."""
        eye = np.array([0, 0, 5], dtype=np.float32)
        target = np.array([0, 0, 0], dtype=np.float32)

        c2w = coordinates.look_at_matrix(eye, target)

        # Forward is -Z column of rotation matrix
        forward = -c2w[:3, 2]

        expected_forward = (target - eye) / np.linalg.norm(target - eye)
        assert np.allclose(forward, expected_forward, atol=1e-6)

    def test_returns_4x4_matrix(self):
        """Should return 4x4 homogeneous matrix."""
        eye = np.array([1, 2, 3], dtype=np.float32)
        target = np.array([0, 0, 0], dtype=np.float32)

        c2w = coordinates.look_at_matrix(eye, target)

        assert c2w.shape == (4, 4)
        assert np.allclose(c2w[3, :], [0, 0, 0, 1])


class TestSphericalToCartesian:
    """Test spherical to Cartesian conversion."""

    def test_radius_5_azimuth_0_elevation_0(self):
        """Origin: azimuth 0, elevation 0 should be at +Z."""
        result = coordinates.spherical_to_cartesian(5.0, 0.0, 0.0)
        expected = np.array([0, 0, 5], dtype=np.float32)
        assert np.allclose(result, expected, atol=1e-6)

    def test_radius_5_azimuth_90_elevation_0(self):
        """Azimuth 90° should be at +X."""
        result = coordinates.spherical_to_cartesian(5.0, 90.0, 0.0)
        expected = np.array([5, 0, 0], dtype=np.float32)
        assert np.allclose(result, expected, atol=1e-6)

    def test_radius_5_azimuth_0_elevation_90(self):
        """Elevation 90° should be at +Y (straight up)."""
        result = coordinates.spherical_to_cartesian(5.0, 0.0, 90.0)
        expected = np.array([0, 5, 0], dtype=np.float32)
        assert np.allclose(result, expected, atol=1e-6)

    def test_distance_from_origin(self):
        """Point should be at specified radius from origin."""
        radius = 5.0
        result = coordinates.spherical_to_cartesian(radius, 45.0, 30.0)
        distance = np.linalg.norm(result)
        assert np.allclose(distance, radius, atol=1e-6)


# TODO: Add tests for world_to_colmap_camera conversion
# This is CRITICAL - needs tests with known ground truth
