"""
Face landmark generation and anchoring for OpenPose Face 70-keypoint rendering.

This module provides:
- Canonical face landmarks (70 points) extracted from MediaPipe's canonical face model
- Procrustes alignment to anchor face landmarks to skeleton head joints
- Face visibility test (frontal hemisphere only)
- OpenPose Face 70 connectivity for rendering
- FaceLandmarkIngest: Convert external face landmark formats to OpenPose Face 70

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

import json
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, NamedTuple, Union
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
# From OpenPose FACE_PAIRS_RENDER_GPU: 63 bone connections.

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
# Eye Geometry
# =============================================================================
# The eyes are not drawn as landmark dots.  A ring of dots plus a pupil dot
# carries almost no gaze information at video-diffusion resolutions, so the
# eye contour is instead filled as a flat two-tone shape: a solid sclera in
# `eye_color` with a `pupil_color` disc sitting on top of it.  The pupil
# position comes from landmarks 68/69, which is where gaze direction lives.

RIGHT_EYE_CONTOUR_INDICES = [36, 37, 38, 39, 40, 41]
LEFT_EYE_CONTOUR_INDICES = [42, 43, 44, 45, 46, 47]
RIGHT_PUPIL_INDEX = 68
LEFT_PUPIL_INDEX = 69

# Landmarks consumed by the eye shapes.  Drawing these as dots as well would
# just speckle the sclera.
EYE_LANDMARK_INDICES = frozenset(
    RIGHT_EYE_CONTOUR_INDICES + LEFT_EYE_CONTOUR_INDICES
    + [RIGHT_PUPIL_INDEX, LEFT_PUPIL_INDEX]
)

# All face keypoints, drawn as dots under eye_style="dots".
ALL_FACE_POINT_INDICES = list(range(len(CANONICAL_FACE_LANDMARKS_70)))

# Face keypoints still rendered as dots under eye_style="shape"
# (everything except the eyes).
FACE_POINT_INDICES_EXCLUDING_EYES = [
    i for i in ALL_FACE_POINT_INDICES if i not in EYE_LANDMARK_INDICES
]

# Face bones still rendered as cylinders under eye_style="shape".  The two
# eye loops are dropped: the filled eye surface already draws that outline.
FACE_BONES_EXCLUDING_EYES = [
    (start, end) for (start, end) in OPENPOSE_FACE_BONES
    if start not in EYE_LANDMARK_INDICES and end not in EYE_LANDMARK_INDICES
]

# Eye rendering styles:
#   "shape" - filled sclera with a pupil disc (default)
#   "dots"  - the original OpenPose landmark dots and eye outline
EYE_STYLES = ("shape", "dots")
DEFAULT_EYE_STYLE = "shape"

# Eye rendering defaults
DEFAULT_EYE_COLOR = (1.0, 1.0, 1.0)     # sclera, matches FACE_COLOR
DEFAULT_PUPIL_COLOR = (0.0, 0.0, 0.0)
DEFAULT_PUPIL_SCALE = 0.75              # fraction of eye height (1.0 = full)
PUPIL_SEGMENTS = 32                     # disc tessellation

# The 6-point eye contour is resampled to this many points before filling.
# A raw hexagon reads as angular and its straight edges cut the corners off
# the opening, which would shrink the largest pupil that fits inside it.
EYE_CONTOUR_SEGMENTS = 48

# The pupil disc is pushed this fraction of the eye height along the eye
# normal so it never z-fights with the sclera it sits on.
PUPIL_LIFT_RATIO = 0.05


def get_face_draw_lists(
    eye_style: str = DEFAULT_EYE_STYLE,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Get the face keypoints and bones to draw for an eye rendering style.

    Under "shape" the eye landmarks and the eye contour bones are withheld,
    because build_eye_geometry() draws filled eye shapes in their place.
    Under "dots" everything is drawn, which is the original OpenPose Face 70
    rendering.

    Args:
        eye_style: "shape" or "dots"

    Returns:
        point_indices: Landmark indices to draw as spheres
        bones: (start, end) landmark index pairs to draw as cylinders

    Raises:
        ValueError: If eye_style is not a known style
    """
    if eye_style == "shape":
        return FACE_POINT_INDICES_EXCLUDING_EYES, FACE_BONES_EXCLUDING_EYES
    if eye_style == "dots":
        return ALL_FACE_POINT_INDICES, OPENPOSE_FACE_BONES

    raise ValueError(
        f"Unknown eye_style: {eye_style!r}. Expected one of {EYE_STYLES}"
    )


class EyeGeometry(NamedTuple):
    """
    Renderable geometry for a single eye, in world coordinates.

    Both meshes are flat (they lie in the eye's own least-squares plane) and
    are wound so their front faces point along `normal`, i.e. out of the face.

    Attributes:
        eye_vertices: Sclera vertices, shape (N + 1, 3) — centroid then
            resampled contour
        eye_faces: Sclera triangle fan, shape (N, 3)
        pupil_vertices: Pupil disc vertices, shape (segments + 1, 3)
        pupil_faces: Pupil triangle fan, shape (segments, 3)
        normal: Unit outward normal of the eye plane, shape (3,)
        pupil_center: Disc center in world coords, shape (3,)
        pupil_radius: Disc radius in world units
        eye_height: Height of the eye opening at the pupil, in world units
    """
    eye_vertices: NDArray[np.float32]
    eye_faces: NDArray[np.int32]
    pupil_vertices: NDArray[np.float32]
    pupil_faces: NDArray[np.int32]
    normal: NDArray[np.float32]
    pupil_center: NDArray[np.float32]
    pupil_radius: float
    eye_height: float


def _fit_plane_normal(
    points: NDArray[np.float32],
    reference: NDArray[np.float32],
) -> NDArray[np.float32]:
    """
    Least-squares plane normal of a point set, oriented to agree with a reference.

    Args:
        points: Points to fit, shape (N, 3)
        reference: Direction the returned normal must point along (dot > 0)

    Returns:
        Unit normal, shape (3,)
    """
    centered = points - points.mean(axis=0)
    # Smallest singular vector = plane normal
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]

    norm = np.linalg.norm(normal)
    if norm < 1e-10:
        return np.asarray(reference, dtype=np.float32)

    normal = normal / norm
    if np.dot(normal, reference) < 0:
        normal = -normal

    return normal.astype(np.float32)


def _distance_to_polygon_edges(
    point: NDArray[np.float32],
    polygon: NDArray[np.float32],
) -> float:
    """
    Minimum distance from a 2D point to the edges of a closed 2D polygon.

    Args:
        point: Query point, shape (2,)
        polygon: Closed polygon vertices in order, shape (N, 2)

    Returns:
        Distance to the nearest edge
    """
    starts = polygon
    ends = np.roll(polygon, -1, axis=0)

    edges = ends - starts
    lengths_sq = np.sum(edges ** 2, axis=1)
    lengths_sq = np.maximum(lengths_sq, 1e-20)

    # Project the point onto each edge, clamped to the segment
    t = np.sum((point - starts) * edges, axis=1) / lengths_sq
    t = np.clip(t, 0.0, 1.0)
    closest = starts + t[:, np.newaxis] * edges

    return float(np.min(np.linalg.norm(point - closest, axis=1)))


def _resample_closed_curve(
    points: NDArray[np.float32],
    n_samples: int,
) -> NDArray[np.float32]:
    """
    Smooth a closed polygon with a Catmull-Rom spline through its vertices.

    The spline passes through every input point, so the eye corners and lid
    midpoints stay exactly where the landmarks put them; only the straight
    edges between them become curved.

    Args:
        points: Closed loop vertices in order, shape (N, 3)
        n_samples: Total number of output samples (>= N)

    Returns:
        Resampled loop, shape (n_samples, 3)
    """
    n = len(points)
    per_segment = max(1, int(round(n_samples / n)))

    # Catmull-Rom needs the neighbours on both sides of each segment
    p0 = np.roll(points, 1, axis=0)
    p1 = points
    p2 = np.roll(points, -1, axis=0)
    p3 = np.roll(points, -2, axis=0)

    t = np.linspace(0.0, 1.0, per_segment, endpoint=False)[:, np.newaxis, np.newaxis]
    t2 = t * t
    t3 = t2 * t

    # Uniform Catmull-Rom basis
    samples = 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )

    # samples is (per_segment, N, 3); interleave so the loop stays in order
    return samples.transpose(1, 0, 2).reshape(-1, 3).astype(np.float32)


def _build_single_eye(
    contour: NDArray[np.float32],
    pupil_landmark: NDArray[np.float32],
    face_normal: NDArray[np.float32],
    pupil_scale: float,
    segments: int,
) -> EyeGeometry:
    """
    Build sclera and pupil meshes for one eye.

    Args:
        contour: Eye contour landmarks in loop order, shape (6, 3)
            (outer corner, upper lid, inner corner, lower lid)
        pupil_landmark: Pupil landmark, shape (3,)
        face_normal: Outward face normal, used to orient the eye plane
        pupil_scale: Pupil diameter as a fraction of the eye height (0, 1]
        segments: Number of segments in the pupil disc

    Returns:
        EyeGeometry for this eye
    """
    normal = _fit_plane_normal(contour, face_normal)

    # Corner indices are needed for the in-plane basis before resampling
    outer_corner = contour[0]
    inner_corner = contour[3]

    contour = _resample_closed_curve(contour, EYE_CONTOUR_SEGMENTS)
    center = contour.mean(axis=0)

    # Flatten the contour into its own plane.  The real eye opening curves
    # around the eyeball, and that curvature would bulge in front of the
    # (flat) pupil disc and clip it.  Both shapes are meant to read as flat
    # two-tone areas anyway, so the projection costs nothing visually.
    contour = contour - ((contour - center) @ normal)[:, np.newaxis] * normal

    # In-plane basis: u runs corner to corner, v is the perpendicular
    # (roughly vertical) axis the eye height is measured along.
    u = inner_corner - outer_corner
    u = u - np.dot(u, normal) * normal
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-10:
        # Degenerate contour: fall back to any in-plane axis
        u = np.cross(normal, [0.0, 1.0, 0.0])
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-10:
            u = np.cross(normal, [1.0, 0.0, 0.0])
            u_norm = np.linalg.norm(u)
    u = (u / u_norm).astype(np.float32)
    v = np.cross(normal, u).astype(np.float32)

    # Project the contour and the pupil into the plane
    offsets = contour - center
    contour_2d = np.stack([offsets @ u, offsets @ v], axis=1)
    pupil_offset = pupil_landmark - center
    pupil_2d = np.array([pupil_offset @ u, pupil_offset @ v], dtype=np.float32)

    # Eye height is measured at the pupil, not across the whole contour: the
    # pupil sits off-center, so the tallest part of the opening is elsewhere
    # and a disc sized from it would poke through a lid.  The distance from
    # the pupil to the nearest lid is half the opening it sits in, so at
    # pupil_scale = 1.0 the disc touches both lids and never spills out.
    eye_height = 2.0 * _distance_to_polygon_edges(pupil_2d, contour_2d)
    radius = pupil_scale * eye_height / 2.0

    # Sclera: triangle fan from the contour centroid
    n_contour = len(contour)
    eye_vertices = np.vstack([center[np.newaxis], contour]).astype(np.float32)
    eye_faces = np.array(
        [(0, i + 1, (i + 1) % n_contour + 1) for i in range(n_contour)],
        dtype=np.int32
    )

    # Pupil: triangle fan disc, lifted off the sclera to avoid z-fighting
    lift = normal * (eye_height * PUPIL_LIFT_RATIO)
    pupil_center = (center + pupil_2d[0] * u + pupil_2d[1] * v + lift).astype(np.float32)

    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = (
        pupil_center[np.newaxis]
        + radius * (np.cos(angles)[:, np.newaxis] * u[np.newaxis]
                    + np.sin(angles)[:, np.newaxis] * v[np.newaxis])
    )
    pupil_vertices = np.vstack([pupil_center[np.newaxis], ring]).astype(np.float32)
    pupil_faces = np.array(
        [(0, i + 1, (i + 1) % segments + 1) for i in range(segments)],
        dtype=np.int32
    )

    # Wind both fans so their front faces look out of the face.  The contour
    # loop direction differs between the left and right eye, so this is
    # decided from the geometry rather than assumed.
    first = eye_vertices[eye_faces[0]]
    winding_normal = np.cross(first[1] - first[0], first[2] - first[0])
    if np.dot(winding_normal, normal) < 0:
        eye_faces = eye_faces[:, ::-1].copy()

    pupil_first = pupil_vertices[pupil_faces[0]]
    pupil_winding = np.cross(
        pupil_first[1] - pupil_first[0], pupil_first[2] - pupil_first[0]
    )
    if np.dot(pupil_winding, normal) < 0:
        pupil_faces = pupil_faces[:, ::-1].copy()

    return EyeGeometry(
        eye_vertices=eye_vertices,
        eye_faces=eye_faces,
        pupil_vertices=pupil_vertices,
        pupil_faces=pupil_faces,
        normal=normal,
        pupil_center=pupil_center,
        pupil_radius=float(radius),
        eye_height=eye_height,
    )


def build_eye_geometry(
    face_landmarks: NDArray[np.float32],
    pupil_scale: float = DEFAULT_PUPIL_SCALE,
    segments: int = PUPIL_SEGMENTS,
) -> List[EyeGeometry]:
    """
    Build flat sclera + pupil meshes for both eyes.

    Each eye contour (landmarks 36-41 and 42-47) is filled as a triangle fan
    lying in the contour's own least-squares plane, and a pupil disc centered
    on the pupil landmark (68/69) is lifted slightly in front of it.

    Args:
        face_landmarks: Fitted OpenPose Face 70 landmarks in world
            coordinates, shape (70, 3)
        pupil_scale: Pupil diameter as a fraction of the eye height at the
            pupil.  1.0 gives a disc that touches the upper and lower lid;
            smaller values shrink it proportionally.  Must be in (0, 1].
        segments: Number of segments in the pupil disc

    Returns:
        [right_eye, left_eye] geometry

    Raises:
        AssertionError: If pupil_scale is outside (0, 1], or the landmark
            array is the wrong shape.
    """
    assert 0.0 < pupil_scale <= 1.0, (
        f"pupil_scale must be in (0, 1], got {pupil_scale}"
    )
    assert face_landmarks.shape == (70, 3), (
        f"Expected face landmarks shape (70, 3), got {face_landmarks.shape}"
    )
    assert segments >= 3, f"Pupil needs at least 3 segments, got {segments}"

    face_normal = compute_face_normal(face_landmarks)

    return [
        _build_single_eye(
            face_landmarks[contour_indices],
            face_landmarks[pupil_index],
            face_normal,
            pupil_scale,
            segments,
        )
        for contour_indices, pupil_index in (
            (RIGHT_EYE_CONTOUR_INDICES, RIGHT_PUPIL_INDEX),
            (LEFT_EYE_CONTOUR_INDICES, LEFT_PUPIL_INDEX),
        )
    ]



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


def _calibrate_face_depth(
    face_landmarks: NDArray[np.float32],
    skeleton_joints: NDArray[np.float32],
) -> NDArray[np.float32]:
    """
    Scale face landmark z values to match the skeleton's head depth.

    MediaPipe z is a relative depth estimate with ~2-3x the range of the
    actual head depth. This function computes the skeleton's head
    depth-to-width ratio (nose-to-ear-midpoint / ear-to-ear) and scales
    the face z values to match.

    Args:
        face_landmarks: OpenPose Face 70 landmarks with uncalibrated z
        skeleton_joints: Skeleton joints in world coordinates

    Returns:
        Copy of face_landmarks with z values scaled to match skeleton geometry
    """
    # Skeleton head geometry
    nose = skeleton_joints[0]       # Nose
    right_ear = skeleton_joints[17] # REar
    left_ear = skeleton_joints[18]  # LEar

    ear_mid = (right_ear + left_ear) / 2.0
    skel_ear_dist = float(np.linalg.norm(left_ear - right_ear))
    skel_depth = float(np.linalg.norm(nose - ear_mid))

    if skel_ear_dist < 1e-6:
        logger.debug("Skeleton ear distance near zero, skipping z calibration")
        return face_landmarks

    skel_ratio = skel_depth / skel_ear_dist

    # Face landmark geometry (same anchor indices: nose=30, ears=1,15)
    face_nose = face_landmarks[30]
    face_right_ear = face_landmarks[1]
    face_left_ear = face_landmarks[15]

    face_ear_mid_z = (face_right_ear[2] + face_left_ear[2]) / 2.0
    face_nose_z = face_nose[2]
    face_z_span = abs(face_nose_z - face_ear_mid_z)

    face_ear_dist_xy = float(np.linalg.norm(
        face_left_ear[:2] - face_right_ear[:2]
    ))

    if face_z_span < 1e-6 or face_ear_dist_xy < 1e-6:
        logger.debug("Face z span or ear distance near zero, skipping z calibration")
        return face_landmarks

    # Target z span: match the skeleton's depth-to-width ratio
    target_z_span = face_ear_dist_xy * skel_ratio
    z_scale = target_z_span / face_z_span

    logger.debug(
        "Z calibration: skeleton depth/width=%.3f, face z_span=%.1f, "
        "target=%.1f, z_scale=%.3f",
        skel_ratio, face_z_span, target_z_span, z_scale
    )

    result = face_landmarks.copy()
    z_center = (face_nose_z + face_ear_mid_z) / 2.0
    result[:, 2] = (result[:, 2] - z_center) * z_scale + z_center

    return result


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
        logger.debug("Using canonical face landmarks")
    else:
        logger.debug(
            "Using custom face landmarks: shape=%s, "
            "x range=[%.3f, %.3f], y range=[%.3f, %.3f], z range=[%.3f, %.3f]",
            face_landmarks.shape,
            face_landmarks[:, 0].min(), face_landmarks[:, 0].max(),
            face_landmarks[:, 1].min(), face_landmarks[:, 1].max(),
            face_landmarks[:, 2].min(), face_landmarks[:, 2].max(),
        )
        # Calibrate z depth using skeleton's known head geometry
        face_landmarks = _calibrate_face_depth(face_landmarks, skeleton_joints)

    # Extract anchor points from face landmarks
    source_anchors = face_landmarks[CANONICAL_ANCHOR_INDICES]

    # Extract anchor points from skeleton
    target_anchors = skeleton_joints[SKELETON_ANCHOR_JOINT_INDICES]

    logger.debug(
        "Procrustes anchors — source (face): %s, target (skeleton): %s",
        source_anchors.tolist(), target_anchors.tolist()
    )

    # Run Procrustes alignment
    rotation, scale, translation, residual = procrustes_align(source_anchors, target_anchors)
    logger.debug(
        "Procrustes result: scale=%.6f, residual=%.6f", scale, residual
    )

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
    camera_position: NDArray[np.float32],
    max_angle_deg: float = 90.0,
) -> bool:
    """
    Determine if the face is visible from a camera position.

    The face is visible when the angle between the face normal and the
    view direction is less than max_angle_deg.  At the default of 90
    degrees this gives the full frontal hemisphere (dot > 0).  Smaller
    values restrict visibility to more frontal views.

    Args:
        face_landmarks: Transformed face landmarks, shape (70, 3)
        camera_position: Camera position in world coordinates, shape (3,)
        max_angle_deg: Maximum angle off the face normal (in degrees) at
            which the face is still rendered.  90 = full hemisphere,
            45 = +/-45 degrees from straight-on.

    Returns:
        True if face is visible (facing camera within threshold)
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

    # cos(angle) between face normal and view direction
    cos_angle = float(np.dot(face_normal, view_direction))
    cos_threshold = float(np.cos(np.radians(max_angle_deg)))

    return cos_angle >= cos_threshold


# =============================================================================
# Face Landmark Ingestion
# =============================================================================
# Converts external face landmark formats to our canonical OpenPose Face 70.
# No dependency on MediaPipe or other detection libraries.

# MediaPipe Face Mesh 478 → OpenPose Face 68 index mapping (MaixPy convention).
# Each entry is a single MediaPipe vertex index that maps to the corresponding
# OpenPose face keypoint. Pupils (68-69) are handled separately.
MEDIAPIPE_TO_OPENPOSE_68 = [
    # Jawline 0-16
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,
    # Right eyebrow 17-21
    71, 63, 105, 66, 107,
    # Left eyebrow 22-26
    336, 296, 334, 293, 301,
    # Nose bridge 27-30
    168, 197, 5, 4,
    # Nose bottom 31-35
    75, 97, 2, 326, 305,
    # Right eye 36-41
    33, 160, 158, 133, 153, 144,
    # Left eye 42-47
    362, 385, 387, 263, 373, 380,
    # Outer lip 48-59
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    # Inner lip 60-67
    78, 82, 13, 312, 308, 317, 14, 87,
]

# Iris center indices in MediaPipe's refined 478-landmark model
MEDIAPIPE_RIGHT_IRIS_CENTER = 468
MEDIAPIPE_LEFT_IRIS_CENTER = 473

# Eye contour indices (in MediaPipe space) for pupil centroid fallback
MEDIAPIPE_RIGHT_EYE_CONTOUR = [33, 160, 158, 133, 153, 144]
MEDIAPIPE_LEFT_EYE_CONTOUR = [362, 385, 387, 263, 373, 380]


class FaceLandmarkIngest:
    """
    Convert external face landmark formats to OpenPose Face 70.

    Supported input formats:
    - "mediapipe": MediaPipe Face Mesh (468 or 478 landmarks)

    The canonical output is always OpenPose Face 70 keypoints as (70, 3) float32.

    Usage:
        # From a JSON file saved by extract_face_landmarks.py:
        landmarks = FaceLandmarkIngest.from_json("landmarks.json")

        # From raw MediaPipe landmarks (list of [x, y, z]):
        landmarks = FaceLandmarkIngest.from_mediapipe(mp_landmarks)
    """

    @staticmethod
    def from_mediapipe(
        landmarks: Union[List[List[float]], NDArray[np.float32]],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> NDArray[np.float32]:
        """
        Convert MediaPipe Face Mesh landmarks to OpenPose Face 70.

        Applies the MaixPy single-index mapping from MediaPipe 468/478
        landmarks to OpenPose Face 68, then adds pupils (indices 68-69)
        from iris centers (if 478 landmarks) or eye contour centroids.

        MediaPipe landmarks are normalized: x to image width, y to image
        height, z relative depth (roughly same scale as x). If image_size
        is provided, coordinates are denormalized to pixel space so that
        the face geometry has correct proportions for Procrustes alignment.

        Args:
            landmarks: MediaPipe face landmarks, shape (N, 3) where N is
                468 (standard) or 478 (refined with iris tracking).
                Coordinates are normalized [0,1] x [0,1] x relative_depth.
            image_size: (width, height) of the source image. Required for
                correct proportions — without it, portrait/landscape images
                produce distorted face geometry.

        Returns:
            OpenPose Face 70 landmarks, shape (70, 3), float32.

        Raises:
            ValueError: If landmarks have wrong shape or too few points.
        """
        lm = np.asarray(landmarks, dtype=np.float32)

        if lm.ndim != 2 or lm.shape[1] != 3:
            raise ValueError(
                f"Expected landmarks shape (N, 3), got {lm.shape}"
            )

        n = lm.shape[0]
        if n < 468:
            raise ValueError(
                f"MediaPipe landmarks require at least 468 points, got {n}"
            )

        # Denormalize to pixel coordinates for correct aspect ratio.
        # MediaPipe x is normalized to width, y to height. Without this,
        # portrait images produce faces wider than tall in normalized space.
        #
        # z is also denormalized by width (MediaPipe z is "roughly same
        # scale as x"). The z values will still have wrong absolute scale
        # (2-3x too large), but fit_face_to_skeleton() calibrates z using
        # the skeleton's known head depth-to-width ratio.
        lm = lm.copy()
        if image_size is not None:
            w, h = image_size
            logger.debug(
                "Denormalizing MediaPipe landmarks with image_size=(%d, %d)", w, h
            )
            lm[:, 0] *= w
            lm[:, 1] *= h
            lm[:, 2] *= w
        else:
            logger.debug("No image_size provided, using raw normalized coordinates")

        # Map the first 68 keypoints
        openpose_68 = lm[MEDIAPIPE_TO_OPENPOSE_68]  # (68, 3)

        # Pupils (indices 68-69)
        if n >= 478:
            # Use iris center landmarks directly
            right_pupil = lm[MEDIAPIPE_RIGHT_IRIS_CENTER]
            left_pupil = lm[MEDIAPIPE_LEFT_IRIS_CENTER]
        else:
            # Synthesize from eye contour centroids
            right_pupil = lm[MEDIAPIPE_RIGHT_EYE_CONTOUR].mean(axis=0)
            left_pupil = lm[MEDIAPIPE_LEFT_EYE_CONTOUR].mean(axis=0)

        # Combine into 70-point array
        openpose_70 = np.vstack([
            openpose_68,
            right_pupil[np.newaxis],
            left_pupil[np.newaxis],
        ]).astype(np.float32)

        return openpose_70

    @staticmethod
    def from_json(filepath: Union[str, Path]) -> NDArray[np.float32]:
        """
        Load face landmarks from a JSON file and convert to OpenPose Face 70.

        Auto-detects the source format from the JSON "source" field and
        dispatches to the appropriate converter.

        Supported JSON formats:
        - MediaPipe: {"source": "mediapipe", "landmarks": [[x,y,z], ...]}

        Args:
            filepath: Path to JSON file.

        Returns:
            OpenPose Face 70 landmarks, shape (70, 3), float32.

        Raises:
            ValueError: If format is unrecognized or data is invalid.
            FileNotFoundError: If file does not exist.
        """
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)

        source = data.get("source", "").lower()

        if source == "mediapipe":
            raw_landmarks = data.get("landmarks")
            if raw_landmarks is None:
                raise ValueError(
                    f"MediaPipe JSON missing 'landmarks' field in {filepath}"
                )
            image_size = data.get("image_size")
            logger.debug(
                "Loading MediaPipe JSON from %s: %d landmarks, image_size=%s",
                filepath, len(raw_landmarks), image_size
            )
            if image_size is not None:
                image_size = tuple(image_size)
            return FaceLandmarkIngest.from_mediapipe(
                raw_landmarks, image_size=image_size
            )
        else:
            raise ValueError(
                f"Unsupported face landmark source: '{source}' in {filepath}. "
                f"Supported: 'mediapipe'"
            )
