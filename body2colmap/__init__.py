"""
Body2COLMAP - Generate synthetic multi-view training data for Gaussian Splatting.

This package converts SAM-3D-Body output (3D mesh from single image) into:
- Multi-view rendered images from orbit camera paths
- COLMAP format camera parameters
- Initial point cloud for 3D Gaussian Splatting training

Example usage:
    from body2colmap import OrbitPipeline

    pipeline = OrbitPipeline.from_sam3d_file("estimation.npz")
    pipeline.set_orbit_params(pattern="helical", n_frames=120)
    pipeline.render_all(modes=["mesh", "depth"])
    pipeline.export_colmap("./output")
    pipeline.export_images("./output")
"""

__version__ = "0.2.0"
__author__ = "Your Name"

from .camera import Camera
from .coordinates import cartesian_to_spherical, spherical_to_cartesian
from .face import FaceLandmarkIngest
from .renderer import parse_composite_modes
from .scene import Scene
from .pipeline import OrbitPipeline
from .splat_anchor import (
    anchor_splat_to_world,
    compute_anchor_transform,
    estimate_depth_scale,
    load_splat_meta,
    transform_splat_scene,
)
from .path import (
    OrbitPath,
    compute_helical_anchor_params,
    compute_original_camera_orbit_params,
    helical_elevation_deg,
)
from .utils import (
    compute_auto_orbit_radius,
    compute_default_focal_length,
    compute_original_view_framing,
    compute_warp_to_camera,
)

__all__ = [
    "Camera",
    "FaceLandmarkIngest",
    "OrbitPath",
    "Scene",
    "OrbitPipeline",
    "anchor_splat_to_world",
    "cartesian_to_spherical",
    "compute_auto_orbit_radius",
    "compute_default_focal_length",
    "compute_helical_anchor_params",
    "compute_original_camera_orbit_params",
    "compute_original_view_framing",
    "compute_anchor_transform",
    "compute_warp_to_camera",
    "estimate_depth_scale",
    "helical_elevation_deg",
    "load_splat_meta",
    "parse_composite_modes",
    "spherical_to_cartesian",
    "transform_splat_scene",
]
