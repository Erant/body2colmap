"""Path generation utilities wrapping body2colmap pipeline functionality."""

import numpy as np
from typing import Tuple, Optional
from body2colmap.scene import Scene


def compute_smart_orbit_radius(
    scene: Scene,
    render_size: Tuple[int, int],
    focal_length: float,
    fill_ratio: float = 0.8
) -> float:
    """
    Compute orbit radius using smart algorithm from body2colmap pipeline.

    This is the CORRECT auto-framing that works for all aspect ratios including portrait.
    Uses the algorithm from body2colmap/pipeline.py:141-172.

    Args:
        scene: Scene to frame
        render_size: (width, height) in pixels
        focal_length: Focal length in pixels
        fill_ratio: How much of viewport to fill (0.0 to 1.0)

    Returns:
        Orbit radius in world units
    """
    bounds = scene.get_bounds()
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
