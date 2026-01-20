"""
Utility functions for body2colmap.

This module provides helper functions used across the library:
- Auto-framing calculations for orbit cameras
- Default focal length computation
"""

import numpy as np
from typing import Tuple
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
