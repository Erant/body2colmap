"""
Tests for orbit path generation, original-camera orbit, and warp homography.
"""

import numpy as np
import pytest

from body2colmap import coordinates
from body2colmap.camera import Camera
from body2colmap.path import OrbitPath, compute_original_camera_orbit_params
from body2colmap.utils import compute_warp_to_camera


class TestCartesianToSpherical:
    """Test cartesian_to_spherical inverse of spherical_to_cartesian."""

    def test_roundtrip_basic(self):
        """spherical -> cartesian -> spherical should be identity."""
        radius, azimuth, elevation = 5.0, 45.0, 30.0
        cart = coordinates.spherical_to_cartesian(radius, azimuth, elevation)
        r2, a2, e2 = coordinates.cartesian_to_spherical(cart)
        assert np.isclose(r2, radius, atol=1e-5)
        assert np.isclose(a2, azimuth, atol=1e-5)
        assert np.isclose(e2, elevation, atol=1e-5)

    def test_roundtrip_negative_azimuth(self):
        """Handles negative Z (azimuth near 180)."""
        radius, azimuth, elevation = 3.0, 170.0, -10.0
        cart = coordinates.spherical_to_cartesian(radius, azimuth, elevation)
        r2, a2, e2 = coordinates.cartesian_to_spherical(cart)
        assert np.isclose(r2, radius, atol=1e-5)
        assert np.isclose(a2, azimuth, atol=1e-5)
        assert np.isclose(e2, elevation, atol=1e-5)

    def test_origin_returns_zeros(self):
        """Zero vector should return all zeros."""
        r, a, e = coordinates.cartesian_to_spherical(np.zeros(3))
        assert r == 0.0
        assert a == 0.0
        assert e == 0.0

    def test_pure_up(self):
        """Point straight up should have elevation 90."""
        r, a, e = coordinates.cartesian_to_spherical(np.array([0, 5, 0]))
        assert np.isclose(r, 5.0, atol=1e-5)
        assert np.isclose(e, 90.0, atol=1e-5)

    def test_plus_z(self):
        """Point along +Z should have azimuth 0."""
        r, a, e = coordinates.cartesian_to_spherical(np.array([0, 0, 3]))
        assert np.isclose(r, 3.0, atol=1e-5)
        assert np.isclose(a, 0.0, atol=1e-5)
        assert np.isclose(e, 0.0, atol=1e-5)


class TestComputeOriginalCameraOrbitParams:
    """Test the standalone orbit parameter computation."""

    def test_camera_at_origin_target_along_negative_z(self):
        """Standard SAM-3D-Body case: camera at origin, mesh along -Z."""
        target = np.array([0.0, 0.0, -2.0], dtype=np.float32)
        params = compute_original_camera_orbit_params(target)

        # Distance from origin to target
        assert np.isclose(params['radius'], 2.0, atol=1e-5)
        # Camera is at origin, target at [0,0,-2], so offset is [0,0,+2]
        # That's along +Z -> azimuth 0
        assert np.isclose(params['start_azimuth_deg'], 0.0, atol=1e-3)
        assert np.isclose(params['elevation_deg'], 0.0, atol=1e-3)

    def test_camera_at_origin_target_off_axis(self):
        """Mesh center slightly off the -Z axis."""
        target = np.array([0.1, -0.2, -1.5], dtype=np.float32)
        params = compute_original_camera_orbit_params(target)

        # Verify roundtrip: spherical -> cartesian gives back the offset
        offset_expected = -target  # camera(origin) - target
        offset_recovered = coordinates.spherical_to_cartesian(
            params['radius'], params['start_azimuth_deg'], params['elevation_deg']
        )
        assert np.allclose(offset_recovered, offset_expected, atol=1e-4)

    def test_custom_camera_position(self):
        """Non-origin camera position."""
        target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        camera_pos = np.array([4.0, 0.0, 0.0], dtype=np.float32)
        params = compute_original_camera_orbit_params(target, camera_pos)

        # Offset is [3, 0, 0] -> azimuth 90, elevation 0
        assert np.isclose(params['radius'], 3.0, atol=1e-5)
        assert np.isclose(params['start_azimuth_deg'], 90.0, atol=1e-3)
        assert np.isclose(params['elevation_deg'], 0.0, atol=1e-3)


class TestOrbitPathCircular:
    """Test basic circular orbit generation."""

    def test_frame_count(self):
        """Should generate requested number of frames."""
        orbit = OrbitPath(
            target=np.zeros(3, dtype=np.float32),
            radius=5.0
        )
        cameras = orbit.circular(n_frames=30, overlap=0)
        assert len(cameras) == 30

    def test_frame_count_with_overlap(self):
        """Overlap should produce correct total frame count."""
        orbit = OrbitPath(
            target=np.zeros(3, dtype=np.float32),
            radius=5.0
        )
        cameras = orbit.circular(n_frames=30, overlap=1)
        assert len(cameras) == 30

    def test_all_cameras_at_correct_distance(self):
        """All cameras should be at the orbit radius from the target."""
        target = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        radius = 5.0
        orbit = OrbitPath(target=target, radius=radius)
        cameras = orbit.circular(n_frames=20, overlap=0)

        for cam in cameras:
            dist = np.linalg.norm(cam.position - target)
            assert np.isclose(dist, radius, atol=1e-4)

    def test_cameras_look_at_target(self):
        """All cameras should have forward vectors pointing toward target."""
        target = np.array([0, 0, 0], dtype=np.float32)
        orbit = OrbitPath(target=target, radius=3.0)
        cameras = orbit.circular(n_frames=12, overlap=0)

        for cam in cameras:
            forward = cam.get_forward_vector()
            expected = (target - cam.position) / np.linalg.norm(target - cam.position)
            assert np.allclose(forward, expected, atol=1e-5)

    def test_start_azimuth(self):
        """Custom start azimuth should shift all positions."""
        target = np.zeros(3, dtype=np.float32)
        orbit = OrbitPath(target=target, radius=5.0)

        cams_default = orbit.circular(n_frames=4, overlap=0, start_azimuth_deg=0.0)
        cams_offset = orbit.circular(n_frames=4, overlap=0, start_azimuth_deg=90.0)

        # First camera of default should be at +Z
        assert np.isclose(cams_default[0].position[2], 5.0, atol=1e-4)
        # First camera of offset should be at +X
        assert np.isclose(cams_offset[0].position[0], 5.0, atol=1e-4)


class TestEndToEndOriginalCameraOrbit:
    """Integration test: compute params + create orbit without pin."""

    def test_frame0_at_origin_via_geometric_radius(self):
        """Geometric radius places frame 0 at the origin via spherical roundtrip."""
        # Simulate SAM-3D-Body: mesh centered at [0.02, -0.15, -1.8]
        target = np.array([0.02, -0.15, -1.8], dtype=np.float32)
        original_fl = 600.0

        # Step 1: compute orbit params (geometric radius + start angles)
        params = compute_original_camera_orbit_params(target)

        # Step 2: create orbit with geometric radius (no pin)
        template = Camera(
            focal_length=(original_fl, original_fl),
            image_size=(512, 512)
        )
        orbit = OrbitPath(
            target=params['target'],
            radius=params['radius'],
        )
        cameras = orbit.circular(
            n_frames=30,
            elevation_deg=params['elevation_deg'],
            start_azimuth_deg=params['start_azimuth_deg'],
            overlap=0,
            camera_template=template
        )

        # Frame 0 is at the origin (spherical roundtrip)
        assert np.allclose(cameras[0].position, [0, 0, 0], atol=1e-4)

        # Frame 0 has near-identity rotation (look_at from origin to
        # nearby-on-axis target), NOT exact identity
        assert np.allclose(cameras[0].rotation, np.eye(3), atol=0.1)

        # Frame 1 is NOT at origin (it's the next orbit position)
        assert not np.allclose(cameras[1].position, [0, 0, 0])

        # All frames have correct focal length
        for cam in cameras:
            assert cam.fx == original_fl

        # ALL frames are at the correct radius (no special-casing)
        for cam in cameras:
            dist = np.linalg.norm(cam.position - target)
            assert np.isclose(dist, params['radius'], atol=1e-4)

    def test_rotation_smooth_between_frame0_and_frame1(self):
        """Rotation should change smoothly (no discontinuity) across all frames."""
        target = np.array([0.02, -0.15, -1.8], dtype=np.float32)
        params = compute_original_camera_orbit_params(target)

        template = Camera(
            focal_length=(600.0, 600.0),
            image_size=(512, 512)
        )
        orbit = OrbitPath(target=params['target'], radius=params['radius'])
        cameras = orbit.circular(
            n_frames=60,
            elevation_deg=params['elevation_deg'],
            start_azimuth_deg=params['start_azimuth_deg'],
            overlap=0,
            camera_template=template
        )

        # Compute angular difference between consecutive frames
        # using Frobenius norm of rotation difference
        max_delta = 0.0
        for i in range(len(cameras) - 1):
            R_diff = cameras[i + 1].rotation - cameras[i].rotation
            delta = np.linalg.norm(R_diff, 'fro')
            max_delta = max(max_delta, delta)

        # All consecutive rotation differences should be similar
        # (no single jump much larger than the average)
        avg_delta = 0.0
        for i in range(len(cameras) - 1):
            R_diff = cameras[i + 1].rotation - cameras[i].rotation
            avg_delta += np.linalg.norm(R_diff, 'fro')
        avg_delta /= (len(cameras) - 1)

        # Max should be within 2x of average (no outlier jumps)
        assert max_delta < 2.0 * avg_delta


class TestComputeWarpToCamera:
    """Test homography computation for original image warping."""

    def test_identity_camera_gives_scale_only(self):
        """Identity rotation camera → homography is pure scale + translate."""
        cam = Camera(
            focal_length=(1200.0, 1200.0),
            image_size=(512, 512),
            position=np.zeros(3, dtype=np.float32),
            rotation=np.eye(3, dtype=np.float32)
        )
        H = compute_warp_to_camera(
            original_focal_length=600.0,
            original_image_size=(512, 512),
            target_camera=cam,
        )
        # For identity rotation, H should be an affine-like matrix:
        # [[s, 0, tx], [0, s, ty], [0, 0, 1]]
        assert np.isclose(H[2, 0], 0.0, atol=1e-10)
        assert np.isclose(H[2, 1], 0.0, atol=1e-10)
        assert np.isclose(H[2, 2], 1.0, atol=1e-10)
        # Scale should be 1200/600 = 2.0
        assert np.isclose(H[0, 0], 2.0, atol=1e-5)
        assert np.isclose(H[1, 1], 2.0, atol=1e-5)

    def test_look_at_camera_maps_mesh_center_correctly(self):
        """Homography should map the mesh center projection to image center."""
        target = np.array([0.02, -0.15, -1.8], dtype=np.float32)
        original_fl = 600.0
        framed_fl = 1086.0

        # Create a look_at camera at origin
        cam = Camera(
            focal_length=(framed_fl, framed_fl),
            image_size=(720, 1280),
            position=np.zeros(3, dtype=np.float32),
        )
        cam.look_at(target)

        H = compute_warp_to_camera(
            original_focal_length=original_fl,
            original_image_size=(720, 1280),
            target_camera=cam,
        )

        # Project mesh center through original camera (identity, original FL)
        # OpenCV projection: u = fx * x/(-z) + cx, v = fy * (-y)/(-z) + cy
        cx_orig = 720 / 2.0
        cy_orig = 1280 / 2.0
        u_orig = original_fl * target[0] / (-target[2]) + cx_orig
        v_orig = original_fl * (-target[1]) / (-target[2]) + cy_orig

        # Apply homography to this point
        p_orig = np.array([u_orig, v_orig, 1.0])
        p_warped = H @ p_orig
        p_warped = p_warped[:2] / p_warped[2]

        # Project mesh center through look_at camera
        points_2d = cam.project(target.reshape(1, 3))
        u_cam, v_cam = points_2d[0]

        # The warped point should match the camera projection
        assert np.isclose(p_warped[0], u_cam, atol=1.0)
        assert np.isclose(p_warped[1], v_cam, atol=1.0)

    def test_homography_is_invertible(self):
        """Homography should be invertible (non-singular)."""
        cam = Camera(
            focal_length=(1000.0, 1000.0),
            image_size=(512, 512),
            position=np.zeros(3, dtype=np.float32),
        )
        cam.look_at(np.array([0.1, -0.2, -2.0]))

        H = compute_warp_to_camera(
            original_focal_length=600.0,
            original_image_size=(512, 512),
            target_camera=cam,
        )
        assert np.linalg.det(H) != 0.0
