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

__version__ = "0.1.0"
__author__ = "Your Name"

from .camera import Camera
from .scene import Scene
from .pipeline import OrbitPipeline

__all__ = [
    "Camera",
    "Scene",
    "OrbitPipeline",
]
