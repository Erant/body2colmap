"""Custom data types for Body2COLMAP ComfyUI nodes."""

from typing import TypeAlias, TypedDict, List, Tuple
import numpy as np
from numpy.typing import NDArray
from body2colmap.camera import Camera
from body2colmap.scene import Scene


class B2C_PATH(TypedDict):
    """Camera path data passed between path generator and render nodes.

    Attributes:
        cameras: List of Camera objects with intrinsics/extrinsics
        orbit_center: (3,) center point cameras look at
        orbit_radius: Distance from center in meters
        pattern: Path pattern type ("circular", "sinusoidal", or "helical")
        n_frames: Number of camera positions
    """
    cameras: List[Camera]
    orbit_center: NDArray[np.float32]
    orbit_radius: float
    pattern: str
    n_frames: int


class B2C_RENDER_DATA(TypedDict):
    """Render data passed to COLMAP export node.

    Attributes:
        cameras: Same cameras used for rendering
        scene: body2colmap Scene object
        resolution: (width, height) in pixels
        focal_length: Focal length in pixels
    """
    cameras: List[Camera]
    scene: Scene
    resolution: Tuple[int, int]
    focal_length: float
