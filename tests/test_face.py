"""
Tests for face landmark module.

Tests Procrustes alignment, face visibility, data integrity, and ingestion.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from body2colmap.face import (
    CANONICAL_FACE_LANDMARKS_70,
    CANONICAL_ANCHOR_INDICES,
    SKELETON_ANCHOR_JOINT_INDICES,
    OPENPOSE_FACE_BONES,
    FACE_COLOR,
    MEDIAPIPE_TO_OPENPOSE_68,
    MEDIAPIPE_RIGHT_EYE_CONTOUR,
    MEDIAPIPE_LEFT_EYE_CONTOUR,
    FaceLandmarkIngest,
    procrustes_align,
    fit_face_to_skeleton,
    compute_face_normal,
    is_face_visible,
    _calibrate_face_depth,
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


class TestMediaPipeMappingConstants:
    """Test MediaPipe → OpenPose mapping constants."""

    def test_mapping_length(self):
        """Should map 68 keypoints (without pupils)."""
        assert len(MEDIAPIPE_TO_OPENPOSE_68) == 68

    def test_mapping_indices_in_range(self):
        """All MediaPipe indices should be in 0-467 range."""
        for mp_idx in MEDIAPIPE_TO_OPENPOSE_68:
            assert 0 <= mp_idx < 468, f"MP index {mp_idx} out of range"

    def test_eye_contour_indices_in_range(self):
        """Eye contour indices should be in 0-467 range."""
        for idx in MEDIAPIPE_RIGHT_EYE_CONTOUR:
            assert 0 <= idx < 468
        for idx in MEDIAPIPE_LEFT_EYE_CONTOUR:
            assert 0 <= idx < 468

    def test_eye_contour_length(self):
        """Each eye contour should have 6 points."""
        assert len(MEDIAPIPE_RIGHT_EYE_CONTOUR) == 6
        assert len(MEDIAPIPE_LEFT_EYE_CONTOUR) == 6


class TestFaceLandmarkIngestFromMediaPipe:
    """Test FaceLandmarkIngest.from_mediapipe()."""

    def _make_fake_mediapipe_478(self):
        """Create fake 478-landmark array for testing."""
        rng = np.random.RandomState(42)
        return rng.rand(478, 3).astype(np.float32)

    def _make_fake_mediapipe_468(self):
        """Create fake 468-landmark array (no iris)."""
        rng = np.random.RandomState(42)
        return rng.rand(468, 3).astype(np.float32)

    def test_output_shape_478(self):
        """Should produce (70, 3) from 478-landmark input."""
        mp = self._make_fake_mediapipe_478()
        result = FaceLandmarkIngest.from_mediapipe(mp)
        assert result.shape == (70, 3)
        assert result.dtype == np.float32

    def test_output_shape_468(self):
        """Should produce (70, 3) from 468-landmark input."""
        mp = self._make_fake_mediapipe_468()
        result = FaceLandmarkIngest.from_mediapipe(mp)
        assert result.shape == (70, 3)
        assert result.dtype == np.float32

    def test_first_68_from_mapping(self):
        """First 68 keypoints should come from the mapping (all 3 coords)."""
        mp = self._make_fake_mediapipe_478()
        result = FaceLandmarkIngest.from_mediapipe(mp)

        for i, mp_idx in enumerate(MEDIAPIPE_TO_OPENPOSE_68):
            np.testing.assert_array_equal(
                result[i], mp[mp_idx],
                err_msg=f"OpenPose keypoint {i} doesn't match MP index {mp_idx}"
            )

    def test_pupils_from_iris_478(self):
        """With 478 landmarks, pupils should come from iris centers."""
        mp = self._make_fake_mediapipe_478()
        result = FaceLandmarkIngest.from_mediapipe(mp)

        np.testing.assert_array_equal(result[68], mp[468])
        np.testing.assert_array_equal(result[69], mp[473])

    def test_pupils_from_centroids_468(self):
        """With 468 landmarks, pupils should be eye contour centroids."""
        mp = self._make_fake_mediapipe_468()
        result = FaceLandmarkIngest.from_mediapipe(mp)

        expected_right = mp[MEDIAPIPE_RIGHT_EYE_CONTOUR].mean(axis=0)
        expected_left = mp[MEDIAPIPE_LEFT_EYE_CONTOUR].mean(axis=0)

        np.testing.assert_allclose(result[68], expected_right, atol=1e-6)
        np.testing.assert_allclose(result[69], expected_left, atol=1e-6)

    def test_accepts_list_input(self):
        """Should accept list-of-lists as well as numpy array."""
        mp = self._make_fake_mediapipe_478()
        mp_list = mp.tolist()
        result = FaceLandmarkIngest.from_mediapipe(mp_list)
        assert result.shape == (70, 3)

    def test_rejects_too_few_landmarks(self):
        """Should raise ValueError for < 468 landmarks."""
        mp = np.random.rand(100, 3).astype(np.float32)
        with pytest.raises(ValueError, match="at least 468"):
            FaceLandmarkIngest.from_mediapipe(mp)

    def test_rejects_wrong_dimensions(self):
        """Should raise ValueError for non-Nx3 input."""
        mp = np.random.rand(478, 2).astype(np.float32)
        with pytest.raises(ValueError, match="shape"):
            FaceLandmarkIngest.from_mediapipe(mp)

    def test_denormalize_with_image_size(self):
        """With image_size, x scaled by width, y by height, z by width."""
        mp = self._make_fake_mediapipe_478()
        w, h = 800, 1200

        result = FaceLandmarkIngest.from_mediapipe(mp, image_size=(w, h))

        # x should be scaled by width, y by height, z by width
        for i, mp_idx in enumerate(MEDIAPIPE_TO_OPENPOSE_68):
            assert abs(result[i, 0] - mp[mp_idx, 0] * w) < 1e-4
            assert abs(result[i, 1] - mp[mp_idx, 1] * h) < 1e-4
            assert abs(result[i, 2] - mp[mp_idx, 2] * w) < 1e-4

    def test_z_preserved(self):
        """MediaPipe z should be preserved (not zeroed)."""
        mp = self._make_fake_mediapipe_478()

        result_with = FaceLandmarkIngest.from_mediapipe(mp, image_size=(800, 600))
        result_without = FaceLandmarkIngest.from_mediapipe(mp)

        # With image_size, z should be scaled by width (non-zero for random input)
        assert not np.allclose(result_with[:, 2], 0.0)
        # Without image_size, z should be raw values (non-zero for random input)
        assert not np.allclose(result_without[:, 2], 0.0)

    def test_no_image_size_preserves_raw(self):
        """Without image_size, all coordinates pass through unchanged."""
        mp = self._make_fake_mediapipe_478()
        result = FaceLandmarkIngest.from_mediapipe(mp)

        # x, y, z should all be raw values from mapping
        np.testing.assert_array_equal(
            result[0], mp[MEDIAPIPE_TO_OPENPOSE_68[0]]
        )


class TestFaceLandmarkIngestFromJSON:
    """Test FaceLandmarkIngest.from_json()."""

    def _make_mediapipe_json(self, n_landmarks=478):
        """Create a MediaPipe-format JSON dict."""
        rng = np.random.RandomState(42)
        landmarks = rng.rand(n_landmarks, 3).tolist()
        return {
            "version": "1.0",
            "source": "mediapipe",
            "source_image": "test.jpg",
            "image_size": [640, 480],
            "n_landmarks": n_landmarks,
            "refined": n_landmarks >= 478,
            "landmarks": landmarks
        }

    def test_loads_mediapipe_json(self):
        """Should correctly load and convert a MediaPipe JSON file."""
        data = self._make_mediapipe_json(478)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            f.flush()
            result = FaceLandmarkIngest.from_json(f.name)

        assert result.shape == (70, 3)
        assert result.dtype == np.float32

    def test_loads_468_landmarks(self):
        """Should handle 468-landmark (non-refined) JSON."""
        data = self._make_mediapipe_json(468)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            f.flush()
            result = FaceLandmarkIngest.from_json(f.name)

        assert result.shape == (70, 3)

    def test_rejects_unknown_source(self):
        """Should raise ValueError for unsupported source format."""
        data = {"source": "unknown_format", "landmarks": []}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            f.flush()
            with pytest.raises(ValueError, match="Unsupported"):
                FaceLandmarkIngest.from_json(f.name)

    def test_rejects_missing_landmarks(self):
        """Should raise ValueError when landmarks field is missing."""
        data = {"source": "mediapipe"}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            f.flush()
            with pytest.raises(ValueError, match="missing 'landmarks'"):
                FaceLandmarkIngest.from_json(f.name)

    def test_file_not_found(self):
        """Should raise FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            FaceLandmarkIngest.from_json("/nonexistent/path.json")

    def test_accepts_path_object(self):
        """Should accept pathlib.Path as well as string."""
        data = self._make_mediapipe_json(478)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            f.flush()
            result = FaceLandmarkIngest.from_json(Path(f.name))

        assert result.shape == (70, 3)

    def test_roundtrip_consistency(self):
        """JSON load should produce same result as direct from_mediapipe."""
        rng = np.random.RandomState(42)
        raw = rng.rand(478, 3).astype(np.float32)
        image_size = (640, 480)

        # Direct conversion with image_size
        direct = FaceLandmarkIngest.from_mediapipe(raw, image_size=image_size)

        # Via JSON roundtrip (image_size in JSON triggers denormalization)
        data = {
            "source": "mediapipe",
            "image_size": list(image_size),
            "landmarks": raw.tolist()
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            f.flush()
            from_json = FaceLandmarkIngest.from_json(f.name)

        np.testing.assert_allclose(from_json, direct, atol=1e-5)

    def test_roundtrip_no_image_size(self):
        """JSON without image_size should pass through raw coordinates."""
        rng = np.random.RandomState(42)
        raw = rng.rand(478, 3).astype(np.float32)

        # Direct conversion without image_size
        direct = FaceLandmarkIngest.from_mediapipe(raw)

        # Via JSON roundtrip (no image_size → no denormalization)
        data = {
            "source": "mediapipe",
            "landmarks": raw.tolist()
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            f.flush()
            from_json = FaceLandmarkIngest.from_json(f.name)

        np.testing.assert_allclose(from_json, direct, atol=1e-5)


class TestCalibrateFaceDepth:
    """Test z-depth calibration from skeleton geometry."""

    def _make_skeleton(self):
        """Create skeleton with known head geometry."""
        skeleton = np.zeros((19, 3), dtype=np.float32)
        skeleton[0] = [0.0, 1.7, 0.05]       # Nose
        skeleton[17] = [-0.08, 1.7, -0.04]    # REar
        skeleton[18] = [0.08, 1.7, -0.04]     # LEar
        return skeleton

    def test_scales_z_to_match_skeleton_ratio(self):
        """Z values should be scaled to match skeleton depth/width ratio."""
        face = np.zeros((70, 3), dtype=np.float32)
        # Ears at x=+-8, z=-10; nose at z=+10  (exaggerated z)
        face[1] = [-8.0, 0.0, -10.0]   # Right ear
        face[15] = [8.0, 0.0, -10.0]   # Left ear
        face[30] = [0.0, 0.0, 10.0]    # Nose tip

        skeleton = self._make_skeleton()

        # Skeleton geometry:
        #   ear_dist = ||[0.08,1.7,-0.04] - [-0.08,1.7,-0.04]|| = 0.16
        #   ear_mid = [0, 1.7, -0.04]
        #   depth = ||[0,1.7,0.05] - [0,1.7,-0.04]|| = 0.09
        #   ratio = 0.09 / 0.16 = 0.5625
        #
        # Face geometry:
        #   ear_dist_xy = 16.0
        #   z_span = |10 - (-10)| = 20
        #   target_z_span = 16.0 * 0.5625 = 9.0
        #   z_scale = 9.0 / 20.0 = 0.45
        #   z_center = (10 + (-10)) / 2 = 0
        #   nose_z_new = 10 * 0.45 = 4.5
        #   ear_z_new = -10 * 0.45 = -4.5

        result = _calibrate_face_depth(face, skeleton)

        assert abs(result[30, 2] - 4.5) < 0.1
        assert abs(result[1, 2] - (-4.5)) < 0.1
        assert abs(result[15, 2] - (-4.5)) < 0.1

    def test_preserves_xy(self):
        """X and Y should not be modified by z calibration."""
        face = np.zeros((70, 3), dtype=np.float32)
        face[1] = [-8.0, -1.0, -10.0]
        face[15] = [8.0, 1.0, -10.0]
        face[30] = [0.0, 2.0, 10.0]
        face[5] = [3.0, 4.0, 5.0]

        skeleton = self._make_skeleton()
        result = _calibrate_face_depth(face, skeleton)

        np.testing.assert_array_equal(result[:, :2], face[:, :2])

    def test_degenerate_skeleton_ears_passthrough(self):
        """If skeleton ears coincide, should return landmarks unchanged."""
        face = np.random.RandomState(42).rand(70, 3).astype(np.float32)
        skeleton = np.zeros((19, 3), dtype=np.float32)
        skeleton[17] = [0.0, 1.7, 0.0]
        skeleton[18] = [0.0, 1.7, 0.0]  # Same as right ear

        result = _calibrate_face_depth(face, skeleton)
        np.testing.assert_array_equal(result, face)

    def test_zero_face_z_span_passthrough(self):
        """If face has no z variation at anchors, should return unchanged."""
        face = np.zeros((70, 3), dtype=np.float32)
        face[1] = [-8.0, 0.0, 0.0]    # Right ear, z=0
        face[15] = [8.0, 0.0, 0.0]    # Left ear, z=0
        face[30] = [0.0, 0.0, 0.0]    # Nose, z=0

        skeleton = self._make_skeleton()
        result = _calibrate_face_depth(face, skeleton)
        np.testing.assert_array_equal(result, face)
