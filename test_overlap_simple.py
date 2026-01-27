#!/usr/bin/env python3
"""Simple test for circular path overlap functionality."""

import sys
import numpy as np

# Add body2colmap to path
sys.path.insert(0, '/home/user/body2colmap')

# Mock the Camera class for testing
class MockCamera:
    def __init__(self, position=None, focal_length=None, image_size=None, principal_point=None):
        self.position = position if position is not None else np.array([0, 0, 0], dtype=np.float32)
        self.fx, self.fy = focal_length if focal_length else (500, 500)
        self.width, self.height = image_size if image_size else (512, 512)
        self.cx, self.cy = principal_point if principal_point else (256, 256)
        self.rotation = np.eye(3)

    def look_at(self, target, up):
        # Simplified look_at for testing
        direction = target - self.position
        direction = direction / np.linalg.norm(direction)
        # Store for verification
        self._target = target

    @staticmethod
    def from_fov(fov_deg, image_size, position=None, is_horizontal_fov=True):
        return MockCamera(position=position, image_size=image_size)

# Monkey patch
import body2colmap.camera
body2colmap.camera.Camera = MockCamera

# Now import OrbitPath
from body2colmap.path import OrbitPath


def test_circular_overlap():
    """Test circular path with various overlap values."""

    # Setup
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    radius = 2.0
    path = OrbitPath(target=target, radius=radius)
    camera_template = MockCamera.from_fov(fov_deg=50.0, image_size=(512, 512))

    print("Testing circular path overlap functionality\n")
    print("=" * 60)

    # Test overlap=0 (no overlap)
    print("\nTest 1: overlap=0 (no overlap)")
    cameras = path.circular(n_frames=8, overlap=0, camera_template=camera_template)
    print(f"  Generated {len(cameras)} cameras")
    assert len(cameras) == 8, f"Expected 8 cameras, got {len(cameras)}"

    # Check first and last are different
    pos_first = cameras[0].position
    pos_last = cameras[-1].position
    diff = np.linalg.norm(pos_first - pos_last)
    print(f"  Distance between first and last: {diff:.6f}")
    assert diff > 0.1, "First and last should be different with overlap=0"
    print("  ✓ First and last positions are different")

    # Test overlap=1 (first and last are same)
    print("\nTest 2: overlap=1 (first and last identical)")
    cameras = path.circular(n_frames=8, overlap=1, camera_template=camera_template)
    print(f"  Generated {len(cameras)} cameras")
    assert len(cameras) == 8, f"Expected 8 cameras, got {len(cameras)}"

    # Check first and last are identical
    pos_first = cameras[0].position
    pos_last = cameras[-1].position
    diff = np.linalg.norm(pos_first - pos_last)
    print(f"  Distance between first and last: {diff:.6f}")
    assert diff < 1e-6, f"First and last should be identical with overlap=1, diff={diff}"

    # Verify they are the same object
    assert cameras[0] is cameras[7], "cameras[0] and cameras[7] should be the same object"
    print("  ✓ First and last positions are identical (same object)")

    # Check angular spacing (7 unique positions evenly distributed over 360°)
    expected_angle_step = 360.0 / 7  # 7 unique frames
    print(f"  Expected angle step: {expected_angle_step:.2f}°")

    # Test overlap=2
    print("\nTest 3: overlap=2 (first 2 and last 2 match)")
    cameras = path.circular(n_frames=10, overlap=2, camera_template=camera_template)
    print(f"  Generated {len(cameras)} cameras")
    assert len(cameras) == 10, f"Expected 10 cameras, got {len(cameras)}"

    # Check cameras[0] == cameras[8]
    assert cameras[0] is cameras[8], "cameras[0] and cameras[8] should be the same object"
    diff_0 = np.linalg.norm(cameras[0].position - cameras[8].position)
    print(f"  Distance between cameras[0] and cameras[8]: {diff_0:.6f}")
    assert diff_0 < 1e-6, f"cameras[0] and cameras[8] should match, diff={diff_0}"

    # Check cameras[1] == cameras[9]
    assert cameras[1] is cameras[9], "cameras[1] and cameras[9] should be the same object"
    diff_1 = np.linalg.norm(cameras[1].position - cameras[9].position)
    print(f"  Distance between cameras[1] and cameras[9]: {diff_1:.6f}")
    assert diff_1 < 1e-6, f"cameras[1] and cameras[9] should match, diff={diff_1}"
    print("  ✓ First 2 and last 2 positions match correctly")

    # Test error handling
    print("\nTest 4: Error handling")
    try:
        path.circular(n_frames=8, overlap=8, camera_template=camera_template)
        assert False, "Should have raised ValueError for overlap >= n_frames"
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")

    try:
        path.circular(n_frames=8, overlap=-1, camera_template=camera_template)
        assert False, "Should have raised ValueError for negative overlap"
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")

    # Test with actual use case: 81 frames with overlap=1
    print("\nTest 5: Realistic case - 81 frames with overlap=1")
    cameras = path.circular(n_frames=81, overlap=1, elevation_deg=0.0,
                           camera_template=camera_template)
    print(f"  Generated {len(cameras)} cameras")
    assert len(cameras) == 81, f"Expected 81 cameras, got {len(cameras)}"

    # Verify first and last are identical
    assert cameras[0] is cameras[80], "cameras[0] and cameras[80] should be the same object"
    diff = np.linalg.norm(cameras[0].position - cameras[80].position)
    print(f"  Distance between cameras[0] and cameras[80]: {diff:.6f}")
    assert diff < 1e-6, "First and last should be identical"

    # Check angular spacing (80 unique positions over 360°)
    expected_angle = 360.0 / 80  # 4.5 degrees
    print(f"  Expected angle step: {expected_angle:.2f}°")
    print("  ✓ 81 frames with overlap=1 works correctly")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")


if __name__ == "__main__":
    test_circular_overlap()
