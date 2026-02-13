"""
Utility functions for body2colmap.

This module provides helper functions used across the library:
- Auto-framing calculations for orbit cameras
- Original-view framing computation
- Default focal length computation
"""

import numpy as np
from typing import Any, Dict, Tuple
from numpy.typing import NDArray


def compute_default_focal_length(
    width: int,
    fov_deg: float = 47.0
) -> float:
    """
    Compute focal length for a given horizontal field of view.

    Args:
        width: Image width in pixels
        fov_deg: Desired horizontal field of view in degrees (default: 47°)

    Returns:
        Focal length in pixels

    Note:
        The default 47° FOV approximates a standard 50mm lens on full-frame camera.
    """
    return width / (2.0 * np.tan(np.radians(fov_deg / 2.0)))


def compute_auto_orbit_radius(
    bounds: Tuple[NDArray[np.float32], NDArray[np.float32]],
    render_size: Tuple[int, int],
    focal_length: float,
    fill_ratio: float = 0.8
) -> float:
    """
    Compute orbit radius that frames the scene properly in the viewport.

    This algorithm handles all aspect ratios correctly, including portrait mode.
    It computes the required distance separately for horizontal and vertical
    dimensions, then uses the maximum to ensure the scene fits in both.

    Args:
        bounds: Tuple of (min_corner, max_corner) arrays, each shape (3,)
                representing the axis-aligned bounding box
        render_size: (width, height) in pixels
        focal_length: Camera focal length in pixels
        fill_ratio: How much of the viewport should be filled (0.0 to 1.0)
                   Default 0.8 leaves 20% padding around the scene

    Returns:
        Orbit radius in world units (distance from orbit center to camera)

    Note:
        For the horizontal extent, we use max(X_extent, Z_extent) because
        the camera orbits around the Y axis and will see different projections
        of the scene at different azimuth angles.
    """
    width, height = render_size
    min_corner, max_corner = bounds

    # Scene extents - for width, use max of X and Z since camera orbits around
    # and will see different projections at different angles
    scene_width = max(
        max_corner[0] - min_corner[0],  # X extent
        max_corner[2] - min_corner[2]   # Z extent (depth)
    )
    scene_height = max_corner[1] - min_corner[1]  # Y extent (up)

    # Compute FOVs in radians
    horizontal_fov_rad = 2 * np.arctan(width / (2 * focal_length))
    vertical_fov_rad = 2 * np.arctan(height / (2 * focal_length))

    # Radius needed to fit horizontal extent in horizontal FOV
    desired_h_angle = horizontal_fov_rad * fill_ratio
    radius_h = (scene_width / 2.0) / np.tan(desired_h_angle / 2.0)

    # Radius needed to fit vertical extent in vertical FOV
    desired_v_angle = vertical_fov_rad * fill_ratio
    radius_v = (scene_height / 2.0) / np.tan(desired_v_angle / 2.0)

    # Use the larger radius to ensure scene fits in both dimensions
    return max(radius_h, radius_v)


def compute_original_view_framing(
    vertices: NDArray[np.float32],
    render_size: Tuple[int, int],
    original_focal_length: float,
    fill_ratio: float = 0.8
) -> Dict[str, Any]:
    """
    Compute auto-framing parameters for the original SAM-3D-Body viewpoint.

    Projects the mesh through the original camera (at origin, identity rotation)
    and computes the scale and translation needed to center the subject and
    fill the frame.

    Since the camera stays at the origin (same viewpoint), this is a pure
    2D operation: scale + translate. The result is a new focal length and
    principal point, plus a 2x3 affine matrix that can be applied to the
    original input image with cv2.warpAffine().

    This is a standalone utility so it can be used outside of OrbitPipeline.

    Args:
        vertices: Mesh vertices in world coordinates, shape (N, 3)
        render_size: (width, height) in pixels
        original_focal_length: Focal length from .npz, in pixels
        fill_ratio: How much of the frame the subject should fill (0-1)

    Returns:
        Dictionary with framing parameters:
            scale_factor, framed_focal_length, framed_principal_point,
            affine_matrix (2x3, for cv2.warpAffine on the original image),
            inverse_affine_matrix (2x3, maps framed coords back to original),
            original_2d_bbox [u_min, v_min, u_max, v_max],
            crop_box_in_original (what region of the original maps to output)
    """
    # Import here to avoid circular dependency
    from .camera import Camera

    w, h = render_size
    cx_orig = w / 2.0
    cy_orig = h / 2.0

    # Create the original camera
    orig_cam = Camera(
        focal_length=(original_focal_length, original_focal_length),
        image_size=render_size,
        position=np.zeros(3, dtype=np.float32),
        rotation=np.eye(3, dtype=np.float32)
    )

    # Project all mesh vertices to 2D image coordinates
    points_2d = orig_cam.project(vertices)

    # 2D bounding box of the projected mesh
    u_min, v_min = points_2d.min(axis=0)
    u_max, v_max = points_2d.max(axis=0)
    bbox_w = u_max - u_min
    bbox_h = v_max - v_min
    bbox_cx = (u_min + u_max) / 2.0
    bbox_cy = (v_min + v_max) / 2.0

    # Scale factor: make the subject fill fill_ratio of the output
    if bbox_w < 1e-6 or bbox_h < 1e-6:
        s = 1.0
    else:
        s = fill_ratio * min(w / bbox_w, h / bbox_h)

    f_new = original_focal_length * s

    # New principal point: centers the subject in the output
    cx_new = w / 2.0 - s * (bbox_cx - cx_orig)
    cy_new = h / 2.0 - s * (bbox_cy - cy_orig)

    # Affine matrix: maps original image coords -> auto-framed coords
    # u' = s * u + tx, v' = s * v + ty
    tx = cx_new - s * cx_orig
    ty = cy_new - s * cy_orig
    affine = [[s, 0.0, tx], [0.0, s, ty]]

    # Inverse affine: maps framed coords -> original image coords
    inv_s = 1.0 / s
    inv_tx = -tx / s
    inv_ty = -ty / s
    inv_affine = [[inv_s, 0.0, inv_tx], [0.0, inv_s, inv_ty]]

    # Crop box: what region of the original image maps to the output
    orig_u_left = -tx / s
    orig_v_top = -ty / s
    orig_u_right = (w - tx) / s
    orig_v_bottom = (h - ty) / s

    return {
        'scale_factor': float(s),
        'framed_focal_length': float(f_new),
        'framed_principal_point': [float(cx_new), float(cy_new)],
        'original_focal_length': float(original_focal_length),
        'original_principal_point': [float(cx_orig), float(cy_orig)],
        'affine_matrix': affine,
        'inverse_affine_matrix': inv_affine,
        'original_2d_bbox': [float(u_min), float(v_min),
                             float(u_max), float(v_max)],
        'crop_box_in_original': [float(orig_u_left), float(orig_v_top),
                                 float(orig_u_right), float(orig_v_bottom)],
    }
