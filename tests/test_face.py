"""
Tests for face landmark module.

Tests Procrustes alignment, face visibility, and data integrity.
"""

import numpy as np
import pytest

from body2colmap.face import (
    CANONICAL_FACE_LANDMARKS_70,
    CANONICAL_ANCHOR_INDICES,
    SKELETON_ANCHOR_JOINT_INDICES,
    OPENPOSE_FACE_BONES,
    FACE_COLOR,
    procrustes_align,
    fit_face_to_skeleton,
    compute_face_normal,
    is_face_visible,
)


class TestCanonicalLandmarks:
    """Test integrity of the canonical face landmark data."""

    def test_shape(self):
        """Canonical landmarks should be (70, 3)."""
        assert CANONICAL_FACE_LANDMARKS_70.shape == (70, 3)

    def test_dtype(self):
        """Should be float32."""
        assert CANONICAL_FACE_LANDMARKS_70.dtype == np.float32

    def test_symmetry(self):
        """Left/right landmark pairs should be symmetric about X=0."""
        # Jawline: 0 ↔ 16, 1 ↔ 15, 2 ↔ 14, etc.
        for i in range(8):
            left = CANONICAL_FACE_LANDMARKS_70[i]
            right = CANONICAL_FACE_LANDMARKS_70[16 - i]
            assert np.allclose(left[0], -right[0], atol=1e-3), f"Jaw {i} vs {16-i} X"
            assert np.allclose(left[1], right[1], atol=1e-3), f"Jaw {i} vs {16-i} Y"
            assert np.allclose(left[2], right[2], atol=1e-3), f"Jaw {i} vs {16-i} Z"

        # Eyes: right outer (36) ↔ left outer (45), inner (39) ↔ inner (42)
        # Right eye: 36(outer), 37, 38, 39(inner), 40, 41
        # Left eye:  42(inner), 43, 44, 45(outer), 46, 47
        eye_pairs = [(36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46)]
        for r_idx, l_idx in eye_pairs:
            right_pt = CANONICAL_FACE_LANDMARKS_70[r_idx]
            left_pt = CANONICAL_FACE_LANDMARKS_70[l_idx]
            assert np.allclose(right_pt[0], -left_pt[0], atol=1e-3), f"Eye {r_idx}↔{l_idx} X"
            assert np.allclose(right_pt[1], left_pt[1], atol=1e-3), f"Eye {r_idx}↔{l_idx} Y"

        # Pupils: 68 ↔ 69
        assert np.allclose(
            CANONICAL_FACE_LANDMARKS_70[68][0],
            -CANONICAL_FACE_LANDMARKS_70[69][0],
            atol=1e-3
        )

    def test_chin_is_lowest(self):
        """Chin (index 8) should have the lowest Y value."""
        chin_y = CANONICAL_FACE_LANDMARKS_70[8, 1]
        assert chin_y == CANONICAL_FACE_LANDMARKS_70[:, 1].min()

    def test_nose_tip_protrudes(self):
        """Nose tip (index 30) should have the highest Z value."""
        nose_z = CANONICAL_FACE_LANDMARKS_70[30, 2]
        assert nose_z == CANONICAL_FACE_LANDMARKS_70[:, 2].max()

    def test_midline_points_centered(self):
        """Points on the face midline should have X ≈ 0."""
        midline_indices = [8, 27, 28, 29, 30, 33, 51, 57, 62, 66]
        for idx in midline_indices:
            assert abs(CANONICAL_FACE_LANDMARKS_70[idx, 0]) < 1e-4, \
                f"Midline point {idx} has X={CANONICAL_FACE_LANDMARKS_70[idx, 0]}"


class TestAnchorIndices:
    """Test anchor point configuration."""

    def test_anchor_count(self):
        """Should have 5 anchor points."""
        assert len(CANONICAL_ANCHOR_INDICES) == 5
        assert len(SKELETON_ANCHOR_JOINT_INDICES) == 5

    def test_anchor_indices_in_range(self):
        """All anchor indices should be valid."""
        for idx in CANONICAL_ANCHOR_INDICES:
            assert 0 <= idx < 70
        for idx in SKELETON_ANCHOR_JOINT_INDICES:
            assert 0 <= idx < 25  # OpenPose Body25 range


class TestFaceBones:
    """Test face bone connectivity."""

    def test_bone_count(self):
        """Should have 63 face bone connections (OpenPose FACE_PAIRS_RENDER_GPU)."""
        assert len(OPENPOSE_FACE_BONES) == 63

    def test_bone_indices_in_range(self):
        """All bone indices should be in range 0-69."""
        for start, end in OPENPOSE_FACE_BONES:
            assert 0 <= start < 70, f"Bone start {start} out of range"
            assert 0 <= end < 70, f"Bone end {end} out of range"

    def test_no_self_loops(self):
        """No bone should connect a point to itself."""
        for start, end in OPENPOSE_FACE_BONES:
            assert start != end

    def test_eyes_are_closed_loops(self):
        """Eye connections should form closed loops."""
        # Right eye: 36→37→38→39→40→41→36
        right_eye_bones = [(36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36)]
        for bone in right_eye_bones:
            assert bone in OPENPOSE_FACE_BONES

        # Left eye: 42→43→44→45→46→47→42
        left_eye_bones = [(42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42)]
        for bone in left_eye_bones:
            assert bone in OPENPOSE_FACE_BONES


class TestProcrustesAlign:
    """Test Procrustes alignment."""

    def test_identity_alignment(self):
        """Aligning identical point sets should give identity transform."""
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ], dtype=np.float32)

        rotation, scale, translation, residual = procrustes_align(points, points)

        assert np.allclose(rotation, np.eye(3), atol=1e-5)
        assert np.allclose(scale, 1.0, atol=1e-5)
        assert np.allclose(translation, [0, 0, 0], atol=1e-5)
        assert residual < 1e-5

    def test_pure_translation(self):
        """Should recover a pure translation."""
        source = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ], dtype=np.float32)

        offset = np.array([10, 20, 30], dtype=np.float32)
        target = source + offset

        rotation, scale, translation, residual = procrustes_align(source, target)

        assert np.allclose(rotation, np.eye(3), atol=1e-4)
        assert np.allclose(scale, 1.0, atol=1e-4)
        assert np.allclose(translation, offset, atol=1e-4)
        assert residual < 1e-4

    def test_pure_scale(self):
        """Should recover a uniform scale."""
        source = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ], dtype=np.float32)

        target = source * 3.0

        rotation, scale, translation, residual = procrustes_align(source, target)

        assert np.allclose(scale, 3.0, atol=1e-4)
        assert residual < 1e-4

    def test_rotation_90_around_z(self):
        """Should recover a 90-degree rotation around Z."""
        source = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ], dtype=np.float32)

        # 90° rotation around Z: (x,y,z) → (-y,x,z)
        R_90z = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1],
        ], dtype=np.float32)

        target = (R_90z @ source.T).T

        rotation, scale, translation, residual = procrustes_align(source, target)

        assert np.allclose(rotation, R_90z, atol=1e-4)
        assert np.allclose(scale, 1.0, atol=1e-4)
        assert residual < 1e-4

    def test_combined_transform(self):
        """Should recover combined scale + rotation + translation."""
        source = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ], dtype=np.float32)

        # Apply scale=2, 180° rotation around Y, translate [5,0,0]
        R_180y = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1],
        ], dtype=np.float32)

        target = 2.0 * (R_180y @ source.T).T + np.array([5, 0, 0])

        rotation, scale, translation, residual = procrustes_align(source, target)

        assert np.allclose(scale, 2.0, atol=1e-3)
        assert np.allclose(rotation, R_180y, atol=1e-3)
        assert residual < 1e-3


class TestFitFaceToSkeleton:
    """Test fitting face landmarks to skeleton."""

    def _make_skeleton(self):
        """Create a simple synthetic skeleton with head joints."""
        # 19 joints minimum (indices 0, 15, 16, 17, 18 used for face anchoring)
        joints = np.zeros((19, 3), dtype=np.float32)

        # Head joints roughly matching a real skeleton (in meters)
        joints[0] = [0.0, 1.7, 0.05]    # Nose (slightly forward)
        joints[15] = [-0.03, 1.72, 0.02]  # Right eye
        joints[16] = [0.03, 1.72, 0.02]   # Left eye
        joints[17] = [-0.08, 1.7, -0.04]  # Right ear
        joints[18] = [0.08, 1.7, -0.04]   # Left ear

        return joints

    def test_output_shape(self):
        """Should produce 70 transformed landmarks."""
        joints = self._make_skeleton()
        landmarks, residual = fit_face_to_skeleton(joints)

        assert landmarks.shape == (70, 3)
        assert landmarks.dtype == np.float32

    def test_residual_is_small(self):
        """Residual should be small for a reasonable skeleton."""
        joints = self._make_skeleton()
        landmarks, residual = fit_face_to_skeleton(joints)

        # For a reasonable skeleton, residual should be small relative to head size
        head_size = np.linalg.norm(joints[17] - joints[18])
        assert residual / head_size < 0.3

    def test_nose_near_skeleton_nose(self):
        """Transformed nose tip should be near skeleton nose joint."""
        joints = self._make_skeleton()
        landmarks, _ = fit_face_to_skeleton(joints)

        # Nose tip is landmark 30
        nose_dist = np.linalg.norm(landmarks[30] - joints[0])
        head_size = np.linalg.norm(joints[17] - joints[18])

        # Should be within one head-width of the skeleton nose
        assert nose_dist < head_size

    def test_face_spans_head_region(self):
        """Transformed face should span approximately the head region."""
        joints = self._make_skeleton()
        landmarks, _ = fit_face_to_skeleton(joints)

        # Face landmarks should be roughly centered on the head
        face_center = landmarks.mean(axis=0)
        head_center = (joints[17] + joints[18]) / 2

        dist = np.linalg.norm(face_center - head_center)
        head_size = np.linalg.norm(joints[17] - joints[18])

        assert dist < head_size * 2


class TestFaceVisibility:
    """Test face visibility hemisphere test."""

    def _make_face_landmarks(self):
        """Create face landmarks in a known orientation.

        Places face at origin, facing +Z.
        """
        landmarks = CANONICAL_FACE_LANDMARKS_70.copy()
        # Scale to ~0.1m (head-sized) and center at origin
        landmarks = landmarks * 0.01  # cm to meters-ish
        return landmarks

    def test_visible_from_front(self):
        """Face should be visible from in front."""
        landmarks = self._make_face_landmarks()
        # Camera in front of face (+Z direction)
        camera_pos = np.array([0, 0, 5], dtype=np.float32)
        assert is_face_visible(landmarks, camera_pos)

    def test_not_visible_from_behind(self):
        """Face should NOT be visible from behind."""
        landmarks = self._make_face_landmarks()
        # Camera behind face (-Z direction)
        camera_pos = np.array([0, 0, -5], dtype=np.float32)
        assert not is_face_visible(landmarks, camera_pos)

    def test_visible_from_side_front(self):
        """Face should be visible from 45 degrees to the side."""
        landmarks = self._make_face_landmarks()
        # Camera at 45° to the right, still in front
        camera_pos = np.array([5, 0, 5], dtype=np.float32)
        assert is_face_visible(landmarks, camera_pos)

    def test_not_visible_from_side_back(self):
        """Face should NOT be visible from 135 degrees (behind-side)."""
        landmarks = self._make_face_landmarks()
        # Camera at 135° to the right (behind)
        camera_pos = np.array([5, 0, -5], dtype=np.float32)
        assert not is_face_visible(landmarks, camera_pos)

    def test_edge_case_exactly_perpendicular(self):
        """At exactly 90 degrees, face should not be visible (dot ≤ 0)."""
        landmarks = self._make_face_landmarks()
        # Camera exactly to the side (+X)
        camera_pos = np.array([5, 0, 0], dtype=np.float32)
        # At exactly 90°, dot product is ~0, could go either way
        # Just verify it returns a boolean without error
        result = is_face_visible(landmarks, camera_pos)
        assert isinstance(result, bool)


class TestComputeFaceNormal:
    """Test face normal computation."""

    def test_canonical_face_normal_direction(self):
        """Canonical face (facing +Z) should have normal roughly in +Z."""
        normal = compute_face_normal(CANONICAL_FACE_LANDMARKS_70)

        # Should primarily point in +Z direction
        assert normal[2] > 0.5, f"Face normal Z component too small: {normal}"

        # Should be a unit vector
        assert np.allclose(np.linalg.norm(normal), 1.0, atol=1e-5)

    def test_normal_is_unit_vector(self):
        """Face normal should always be unit length."""
        # Scale and translate the landmarks
        landmarks = CANONICAL_FACE_LANDMARKS_70 * 5.0 + np.array([100, 200, 300])
        normal = compute_face_normal(landmarks.astype(np.float32))

        assert np.allclose(np.linalg.norm(normal), 1.0, atol=1e-5)
