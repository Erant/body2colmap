"""
Tests for Camera class.

This is a HIGH PRIORITY test module - camera math must be correct.
"""

import numpy as np
import pytest

from body2colmap.camera import Camera


class TestCameraInitialization:
    """Test Camera initialization."""

    def test_default_initialization(self):
        """Basic camera creation with minimal params."""
        camera = Camera(
            focal_length=(1000.0, 1000.0),
            image_size=(512, 512)
        )

        assert camera.fx == 1000.0
        assert camera.fy == 1000.0
        assert camera.width == 512
        assert camera.height == 512

        # Should default to image center
        assert camera.cx == 256.0
        assert camera.cy == 256.0

        # Should default to origin and identity rotation
        assert np.allclose(camera.position, [0, 0, 0])
        assert np.allclose(camera.rotation, np.eye(3))

    def test_custom_principal_point(self):
        """Camera with custom principal point."""
        camera = Camera(
            focal_length=(1000.0, 1000.0),
            image_size=(512, 512),
            principal_point=(250.0, 250.0)
        )

        assert camera.cx == 250.0
        assert camera.cy == 250.0

    def test_from_fov(self):
        """Camera created from field of view."""
        camera = Camera.from_fov(
            fov_deg=90.0,
            image_size=(512, 512)
        )

        # 90° FOV: tan(45°) = 1
        # fx = (width/2) / tan(fov/2) = 256 / 1 = 256
        expected_fx = 256.0
        assert np.allclose(camera.fx, expected_fx)


class TestCameraTransforms:
    """Test camera transformation matrices."""

    def test_c2w_matrix(self):
        """Camera-to-world matrix construction."""
        camera = Camera(
            focal_length=(1000.0, 1000.0),
            image_size=(512, 512),
            position=np.array([1, 2, 3]),
            rotation=np.eye(3)
        )

        c2w = camera.get_c2w()

        # Should be 4x4
        assert c2w.shape == (4, 4)

        # Upper-left 3x3 should be rotation
        assert np.allclose(c2w[:3, :3], np.eye(3))

        # Upper-right 3x1 should be position
        assert np.allclose(c2w[:3, 3], [1, 2, 3])

        # Bottom row should be [0, 0, 0, 1]
        assert np.allclose(c2w[3, :], [0, 0, 0, 1])

    def test_w2c_is_inverse_of_c2w(self):
        """w2c should be inverse of c2w."""
        camera = Camera(
            focal_length=(1000.0, 1000.0),
            image_size=(512, 512),
            position=np.array([1, 2, 3]),
            rotation=np.eye(3)
        )

        c2w = camera.get_c2w()
        w2c = camera.get_w2c()

        # w2c @ c2w should be identity
        identity = w2c @ c2w
        assert np.allclose(identity, np.eye(4), atol=1e-6)

    def test_look_at(self):
        """look_at should orient camera toward target."""
        camera = Camera(
            focal_length=(1000.0, 1000.0),
            image_size=(512, 512),
            position=np.array([0, 0, 5])
        )

        target = np.array([0, 0, 0])
        camera.look_at(target)

        # Forward vector should point toward target
        forward = camera.get_forward_vector()
        expected_forward = (target - camera.position) / np.linalg.norm(target - camera.position)

        assert np.allclose(forward, expected_forward, atol=1e-6)


class TestCameraVectors:
    """Test camera direction vectors."""

    def test_default_camera_vectors(self):
        """Default camera (identity rotation) should have standard vectors."""
        camera = Camera(
            focal_length=(1000.0, 1000.0),
            image_size=(512, 512)
        )

        # Identity rotation: camera aligned with world axes
        # Forward: -Z = [0, 0, -1]
        # Up: +Y = [0, 1, 0]
        # Right: +X = [1, 0, 0]

        assert np.allclose(camera.get_forward_vector(), [0, 0, -1])
        assert np.allclose(camera.get_up_vector(), [0, 1, 0])
        assert np.allclose(camera.get_right_vector(), [1, 0, 0])

    def test_vectors_are_unit_length(self):
        """Direction vectors should be unit length."""
        camera = Camera(
            focal_length=(1000.0, 1000.0),
            image_size=(512, 512),
            rotation=np.random.randn(3, 3).astype(np.float32)
        )

        # Orthonormalize rotation
        camera.rotation, _ = np.linalg.qr(camera.rotation)

        assert np.allclose(np.linalg.norm(camera.get_forward_vector()), 1.0)
        assert np.allclose(np.linalg.norm(camera.get_up_vector()), 1.0)
        assert np.allclose(np.linalg.norm(camera.get_right_vector()), 1.0)


# TODO: Add tests for:
# - get_colmap_extrinsics() with known ground truth
# - project() with known 3D points
# - Verify quaternion is (w, x, y, z) order
