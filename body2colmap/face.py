"""
Face landmark generation and anchoring for OpenPose Face 70-keypoint rendering.

This module provides:
- Canonical face landmarks (70 points) extracted from MediaPipe's canonical face model
- Procrustes alignment to anchor face landmarks to skeleton head joints
- Face visibility test (frontal hemisphere only)
- OpenPose Face 70 connectivity for rendering

The 70 keypoints follow the OpenPose face convention:
  0-16:  Jawline contour (17 points, open chain)
  17-21: Right eyebrow (5 points, open chain)
  22-26: Left eyebrow (5 points, open chain)
  27-30: Nose bridge (4 points, open chain)
  31-35: Nose bottom/nostrils (5 points, open chain)
  36-41: Right eye (6 points, closed loop)
  42-47: Left eye (6 points, closed loop)
  48-59: Outer lip (12 points, closed loop)
  60-67: Inner lip (8 points, closed loop)
  68:    Right pupil (isolated point)
  69:    Left pupil (isolated point)

No MediaPipe dependency is required. The canonical face model geometry is embedded
as a constant extracted from MediaPipe's canonical_face_model.obj.
"""

import logging
import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# =============================================================================
# Canonical Face Landmarks (70 points)
# =============================================================================
# Extracted from MediaPipe's canonical_face_model.obj using the MaixPy
# single-index mapping from MediaPipe 468 → OpenPose Face 68, plus
# synthesized pupils from eye contour centroids.
#
# Coordinate system: Right-handed, Y-up, face looks toward +Z.
# Units: approximately centimeters.
# Source: https://github.com/google-ai-edge/mediapipe/blob/master/
#         mediapipe/modules/face_geometry/data/canonical_face_model.obj

CANONICAL_FACE_LANDMARKS_70 = np.array([
    # Jawline 0-16
    (-7.555811,   4.106811,  -0.991917),  #  0 jaw
    (-7.664182,   0.673132,  -2.435867),  #  1 jaw
    (-7.542244,  -1.049282,  -2.431321),  #  2 jaw
    (-6.719682,  -4.788645,  -1.745401),  #  3 jaw
    (-5.940524,  -6.223629,  -0.631468),  #  4 jaw
    (-5.085276,  -7.178590,   0.714711),  #  5 jaw
    (-3.210651,  -8.533278,   2.802001),  #  6 jaw
    (-1.292924,  -9.295920,   4.094063),  #  7 jaw
    ( 0.000000,  -9.403378,   4.264492),  #  8 jaw (chin)
    ( 1.292924,  -9.295920,   4.094063),  #  9 jaw
    ( 3.210651,  -8.533278,   2.802001),  # 10 jaw
    ( 5.085276,  -7.178590,   0.714711),  # 11 jaw
    ( 5.940524,  -6.223629,  -0.631468),  # 12 jaw
    ( 6.719682,  -4.788645,  -1.745401),  # 13 jaw
    ( 7.542244,  -1.049282,  -2.431321),  # 14 jaw
    ( 7.664182,   0.673132,  -2.435867),  # 15 jaw
    ( 7.555811,   4.106811,  -0.991917),  # 16 jaw
    # Right eyebrow 17-21
    (-6.374393,   4.785590,   1.591691),  # 17 r_eyebrow
    (-4.985894,   4.802461,   3.751977),  # 18 r_eyebrow
    (-3.986562,   5.109487,   4.466315),  # 19 r_eyebrow
    (-2.760292,   5.100971,   5.015990),  # 20 r_eyebrow
    (-1.395634,   5.011963,   5.316032),  # 21 r_eyebrow
    # Left eyebrow 22-26
    ( 1.395634,   5.011963,   5.316032),  # 22 l_eyebrow
    ( 2.760292,   5.100971,   5.015990),  # 23 l_eyebrow
    ( 3.986562,   5.109487,   4.466315),  # 24 l_eyebrow
    ( 4.985894,   4.802461,   3.751977),  # 25 l_eyebrow
    ( 6.374393,   4.785590,   1.591691),  # 26 l_eyebrow
    # Nose bridge 27-30
    ( 0.000000,   3.271027,   5.236015),  # 27 nose_bridge
    ( 0.000000,   1.728369,   6.316750),  # 28 nose_bridge
    ( 0.000000,   0.365669,   7.242870),  # 29 nose_bridge
    ( 0.000000,  -0.463170,   7.586580),  # 30 nose_bridge (tip)
    # Nose bottom 31-35
    (-1.043625,  -1.464973,   5.662455),  # 31 nose_bottom
    (-0.597442,  -2.013686,   5.866456),  # 32 nose_bottom
    ( 0.000000,  -2.089024,   6.058267),  # 33 nose_bottom
    ( 0.597442,  -2.013686,   5.866456),  # 34 nose_bottom
    ( 1.043625,  -1.464973,   5.662455),  # 35 nose_bottom
    # Right eye 36-41
    (-4.445859,   2.663991,   3.173422),  # 36 r_eye (outer corner)
    (-3.670075,   2.927714,   3.724325),  # 37 r_eye
    (-2.724032,   2.961810,   3.871767),  # 38 r_eye
    (-1.856432,   2.585245,   3.757904),  # 39 r_eye (inner corner)
    (-2.724032,   2.315802,   3.777151),  # 40 r_eye
    (-3.670075,   2.360153,   3.635230),  # 41 r_eye
    # Left eye 42-47
    ( 1.856432,   2.585245,   3.757904),  # 42 l_eye (inner corner)
    ( 2.724032,   2.961810,   3.871767),  # 43 l_eye
    ( 3.670075,   2.927714,   3.724325),  # 44 l_eye
    ( 4.445859,   2.663991,   3.173422),  # 45 l_eye (outer corner)
    ( 3.670075,   2.360153,   3.635230),  # 46 l_eye
    ( 2.724032,   2.315802,   3.777151),  # 47 l_eye
    # Outer lip 48-59
    (-2.456206,  -4.342621,   4.283884),  # 48 outer_lip (R corner)
    (-1.431615,  -3.500953,   5.496189),  # 49 outer_lip
    (-0.711452,  -3.329355,   5.877044),  # 50 outer_lip
    ( 0.000000,  -3.406404,   5.979507),  # 51 outer_lip (top center)
    ( 0.711452,  -3.329355,   5.877044),  # 52 outer_lip
    ( 1.431615,  -3.500953,   5.496189),  # 53 outer_lip
    ( 2.456206,  -4.342621,   4.283884),  # 54 outer_lip (L corner)
    ( 1.325085,  -5.106507,   5.205010),  # 55 outer_lip
    ( 0.699606,  -5.291850,   5.448304),  # 56 outer_lip
    ( 0.000000,  -5.365123,   5.535441),  # 57 outer_lip (bottom center)
    (-0.699606,  -5.291850,   5.448304),  # 58 outer_lip
    (-1.325085,  -5.106507,   5.205010),  # 59 outer_lip
    # Inner lip 60-67
    (-2.153084,  -4.276322,   4.038093),  # 60 inner_lip
    (-0.533422,  -3.993222,   5.138202),  # 61 inner_lip
    ( 0.000000,  -3.994436,   5.219482),  # 62 inner_lip (top center)
    ( 0.533422,  -3.993222,   5.138202),  # 63 inner_lip
    ( 2.153084,  -4.276322,   4.038093),  # 64 inner_lip
    ( 0.583218,  -4.517982,   5.339869),  # 65 inner_lip
    ( 0.000000,  -4.542400,   5.404754),  # 66 inner_lip (bottom center)
    (-0.583218,  -4.517982,   5.339869),  # 67 inner_lip
    # Pupils 68-69 (synthesized from eye contour centroids)
    (-3.181751,   2.635786,   3.656633),  # 68 r_pupil
    ( 3.181751,   2.635786,   3.656633),  # 69 l_pupil
], dtype=np.float32)


# =============================================================================
# Canonical Face Anchor Points
# =============================================================================
# These are the 5 anchor points in the canonical face model that correspond
# to the skeleton's head joints. Used for Procrustes alignment.
#
# Indices into CANONICAL_FACE_LANDMARKS_70:
#   Nose tip:       index 30 (nose bridge tip, MP vertex 4)
#   Right eye:      index 68 (right pupil, centroid of eye contour)
#   Left eye:       index 69 (left pupil, centroid of eye contour)
#   Right ear:      index 1  (jawline point near right ear, MP vertex 234)
#   Left ear:       index 15 (jawline point near left ear, MP vertex 454)

CANONICAL_ANCHOR_INDICES = [30, 68, 69, 1, 15]

# Corresponding skeleton joint indices (OpenPose Body25 format):
#   0: Nose, 15: REye, 16: LEye, 17: REar, 18: LEar
SKELETON_ANCHOR_JOINT_INDICES = [0, 15, 16, 17, 18]


# =============================================================================
# OpenPose Face 70 Connectivity
# =============================================================================
# From OpenPose FACE_PAIRS_RENDER_GPU: 71 bone connections.

OPENPOSE_FACE_BONES = [
    # Jawline (open chain, 16 segments)
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
    (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
    # Right eyebrow (open chain, 4 segments)
    (17, 18), (18, 19), (19, 20), (20, 21),
    # Left eyebrow (open chain, 4 segments)
    (22, 23), (23, 24), (24, 25), (25, 26),
    # Nose bridge (open chain, 3 segments)
    (27, 28), (28, 29), (29, 30),
    # Nose bottom (open chain, 4 segments)
    (31, 32), (32, 33), (33, 34), (34, 35),
    # Right eye (closed loop, 6 segments)
    (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),
    # Left eye (closed loop, 6 segments)
    (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),
    # Outer lip (closed loop, 12 segments)
    (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54),
    (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48),
    # Inner lip (closed loop, 8 segments)
    (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 60),
]

# Face color: solid white (OpenPose convention)
FACE_COLOR = (1.0, 1.0, 1.0)


# =============================================================================
# Procrustes Alignment
# =============================================================================

def procrustes_align(
    source: NDArray[np.float32],
    target: NDArray[np.float32]
) -> Tuple[NDArray[np.float32], float, NDArray[np.float32], float]:
    """
    Compute similarity transform (scale + rotation + translation) from source to target.

    Uses Procrustes analysis via SVD of the cross-covariance matrix.

    Args:
        source: Source points, shape (N, 3)
        target: Target points, shape (N, 3)

    Returns:
        rotation: 3x3 rotation matrix
        scale: Uniform scale factor
        translation: 3D translation vector
        residual: Mean distance between transformed source and target
    """
    assert source.shape == target.shape
    n = source.shape[0]

    # Center both point sets
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)

    source_centered = source - source_mean
    target_centered = target - target_mean

    # Compute scale
    source_norm = np.sqrt(np.sum(source_centered ** 2) / n)
    target_norm = np.sqrt(np.sum(target_centered ** 2) / n)

    if source_norm < 1e-10:
        logger.warning("Face anchor points are degenerate (near-zero spread)")
        return np.eye(3, dtype=np.float32), 1.0, target_mean - source_mean, float('inf')

    scale = target_norm / source_norm

    # Compute optimal rotation via SVD
    # Cross-covariance matrix
    H = (source_centered.T @ target_centered) / n
    U, S, Vt = np.linalg.svd(H)

    # Ensure proper rotation (not reflection)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])
    rotation = (Vt.T @ sign_matrix @ U.T).astype(np.float32)

    # Compute translation: target_mean = scale * R @ source_mean + t
    translation = target_mean - scale * (rotation @ source_mean)

    # Compute residual
    transformed = scale * (source @ rotation.T) + translation
    residual = float(np.mean(np.linalg.norm(transformed - target, axis=1)))

    return rotation, float(scale), translation.astype(np.float32), residual


def fit_face_to_skeleton(
    skeleton_joints: NDArray[np.float32],
    face_landmarks: Optional[NDArray[np.float32]] = None,
    warn_threshold: float = 0.2
) -> Tuple[NDArray[np.float32], float]:
    """
    Fit face landmarks to skeleton head joints via Procrustes alignment.

    Uses 5 anchor points (nose, eyes, ears) shared between the face model
    and the skeleton to compute a similarity transform, then applies it
    to all 70 face keypoints.

    Args:
        skeleton_joints: OpenPose Body25(+Hands) joints, shape (N, 3) where N >= 19.
            Uses joints 0 (nose), 15 (REye), 16 (LEye), 17 (REar), 18 (LEar).
        face_landmarks: Optional custom face landmarks, shape (70, 3).
            If None, uses CANONICAL_FACE_LANDMARKS_70.
        warn_threshold: If the Procrustes residual exceeds this fraction of
            head size, log a warning.

    Returns:
        transformed_landmarks: Face landmarks in world coordinates, shape (70, 3)
        residual: Mean anchor point alignment error (meters)
    """
    if face_landmarks is None:
        face_landmarks = CANONICAL_FACE_LANDMARKS_70.copy()

    # Extract anchor points from face landmarks
    source_anchors = face_landmarks[CANONICAL_ANCHOR_INDICES]

    # Extract anchor points from skeleton
    target_anchors = skeleton_joints[SKELETON_ANCHOR_JOINT_INDICES]

    # Run Procrustes alignment
    rotation, scale, translation, residual = procrustes_align(source_anchors, target_anchors)

    # Apply transform to all 70 landmarks
    transformed = scale * (face_landmarks @ rotation.T) + translation

    # Warn if fit quality is poor
    # Compute head size as distance between ears for reference
    right_ear = skeleton_joints[17]
    left_ear = skeleton_joints[18]
    head_size = float(np.linalg.norm(right_ear - left_ear))

    if head_size > 1e-6 and residual / head_size > warn_threshold:
        logger.warning(
            f"Face-to-skeleton fit quality is poor: residual={residual:.4f}m, "
            f"head_size={head_size:.4f}m, ratio={residual/head_size:.2f}. "
            f"Face landmarks may not align well with skeleton."
        )

    return transformed.astype(np.float32), residual


# =============================================================================
# Face Visibility Test
# =============================================================================

def compute_face_normal(
    face_landmarks: NDArray[np.float32]
) -> NDArray[np.float32]:
    """
    Compute face normal direction from transformed face landmarks.

    Uses cross product of (left_eye - right_eye) x (nose_bridge_top - chin)
    to determine the direction the face is pointing. These two vectors span
    the face plane (horizontal and vertical), so their cross product gives
    the outward-facing normal.

    Args:
        face_landmarks: Transformed face landmarks, shape (70, 3)

    Returns:
        Unit normal vector pointing outward from face, shape (3,)
    """
    right_eye = face_landmarks[68]   # Right pupil
    left_eye = face_landmarks[69]    # Left pupil
    nose_bridge_top = face_landmarks[27]  # Top of nose bridge
    chin = face_landmarks[8]         # Chin

    # Two vectors spanning the face plane
    eye_vector = left_eye - right_eye              # horizontal (left-right)
    vertical_vector = nose_bridge_top - chin       # vertical (chin-to-forehead)

    # Cross product: horizontal × vertical = outward face normal
    normal = np.cross(eye_vector, vertical_vector)
    norm = np.linalg.norm(normal)

    if norm < 1e-10:
        # Degenerate case, return forward (+Z as fallback)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    return (normal / norm).astype(np.float32)


def is_face_visible(
    face_landmarks: NDArray[np.float32],
    camera_position: NDArray[np.float32]
) -> bool:
    """
    Determine if the face is visible from a camera position.

    The face is considered visible when the face normal points toward
    the camera (dot product > 0), giving a clean 180-degree frontal
    hemisphere cutoff.

    Args:
        face_landmarks: Transformed face landmarks, shape (70, 3)
        camera_position: Camera position in world coordinates, shape (3,)

    Returns:
        True if face is visible (facing camera), False if occluded by head
    """
    face_normal = compute_face_normal(face_landmarks)

    # Head center: midpoint of ears
    head_center = (face_landmarks[1] + face_landmarks[15]) / 2.0

    # View direction: from head center toward camera
    view_direction = camera_position - head_center
    view_norm = np.linalg.norm(view_direction)

    if view_norm < 1e-10:
        return True  # Camera at head center, show face

    view_direction = view_direction / view_norm

    # Dot product > 0 means face points toward camera
    return float(np.dot(face_normal, view_direction)) > 0.0
