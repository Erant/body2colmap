"""Camera path generator nodes for Body2COLMAP."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from body2colmap.path import OrbitPath
from body2colmap.camera import Camera
from core.sam3d_adapter import sam3d_output_to_scene


class Body2COLMAP_CircularPath:
    """Generate circular camera orbit path around mesh."""

    CATEGORY = "Body2COLMAP/Path"
    FUNCTION = "generate_path"
    RETURN_TYPES = ("B2C_PATH",)
    RETURN_NAMES = ("camera_path",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_data": ("SAM3D_OUTPUT",),
                "n_frames": ("INT", {
                    "default": 36,
                    "min": 4,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Number of camera positions around the orbit"
                }),
                "elevation_deg": ("FLOAT", {
                    "default": 0.0,
                    "min": -89.0,
                    "max": 89.0,
                    "step": 1.0,
                    "tooltip": "Camera elevation angle (0=eye level, positive=above)"
                }),
            },
            "optional": {
                "radius": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Orbit radius in meters (0=auto-compute from mesh bounds)"
                }),
                "fill_ratio": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "How much of viewport should contain mesh (for auto-radius)"
                }),
                "start_azimuth_deg": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 5.0,
                    "tooltip": "Starting azimuth angle (0=front)"
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Image width for auto-radius calculation"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Image height for auto-radius calculation"
                }),
                "focal_length": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10000.0,
                    "step": 1.0,
                    "tooltip": "Focal length in pixels (0=auto for ~47° FOV)"
                }),
            }
        }

    def generate_path(self, mesh_data, n_frames, elevation_deg,
                      radius=0.0, fill_ratio=0.8, start_azimuth_deg=0.0,
                      width=512, height=512, focal_length=0.0):
        """Generate circular orbit camera path."""

        # Convert SAM3D output to Scene
        scene = sam3d_output_to_scene(mesh_data, include_skeleton=False)

        # Get orbit center from mesh bounding box
        orbit_center = scene.get_bbox_center()

        # Auto-compute radius if needed
        if radius <= 0.0:
            radius = OrbitPath.auto_compute_radius(
                scene_bounds=scene.get_bounds(),
                fill_ratio=fill_ratio,
                fov_deg=47.0
            )

        # Create camera template with specified intrinsics
        if focal_length <= 0:
            # Auto: ~47° horizontal FOV
            camera_template = Camera.from_fov(
                fov_deg=47.0,
                image_size=(width, height),
                is_horizontal_fov=True
            )
        else:
            camera_template = Camera(
                focal_length=(focal_length, focal_length),
                image_size=(width, height)
            )

        # Create OrbitPath and generate cameras
        path_gen = OrbitPath(
            target=orbit_center,
            radius=radius
        )

        cameras = path_gen.circular(
            n_frames=n_frames,
            elevation_deg=elevation_deg,
            start_azimuth_deg=start_azimuth_deg,
            camera_template=camera_template
        )

        # Return B2C_PATH dict
        return ({
            "cameras": cameras,
            "orbit_center": orbit_center,
            "orbit_radius": float(radius),
            "pattern": "circular",
            "n_frames": int(n_frames),
        },)


class Body2COLMAP_SinusoidalPath:
    """Generate sinusoidal camera orbit with oscillating elevation."""

    CATEGORY = "Body2COLMAP/Path"
    FUNCTION = "generate_path"
    RETURN_TYPES = ("B2C_PATH",)
    RETURN_NAMES = ("camera_path",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_data": ("SAM3D_OUTPUT",),
                "n_frames": ("INT", {
                    "default": 60,
                    "min": 4,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Number of camera positions"
                }),
                "amplitude_deg": ("FLOAT", {
                    "default": 30.0,
                    "min": 0.0,
                    "max": 89.0,
                    "step": 1.0,
                    "tooltip": "Maximum elevation deviation from center"
                }),
                "n_cycles": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of up/down oscillations per full rotation"
                }),
            },
            "optional": {
                "radius": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Orbit radius in meters (0=auto-compute)"
                }),
                "fill_ratio": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Viewport fill ratio for auto-radius"
                }),
                "start_azimuth_deg": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 5.0,
                    "tooltip": "Starting azimuth angle"
                }),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "focal_length": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0}),
            }
        }

    def generate_path(self, mesh_data, n_frames, amplitude_deg, n_cycles,
                      radius=0.0, fill_ratio=0.8, start_azimuth_deg=0.0,
                      width=512, height=512, focal_length=0.0):
        """Generate sinusoidal orbit camera path."""

        # Convert SAM3D output to Scene
        scene = sam3d_output_to_scene(mesh_data, include_skeleton=False)

        # Get orbit center
        orbit_center = scene.get_bbox_center()

        # Auto-compute radius if needed
        if radius <= 0.0:
            radius = OrbitPath.auto_compute_radius(
                scene_bounds=scene.get_bounds(),
                fill_ratio=fill_ratio,
                fov_deg=47.0
            )

        # Create camera template
        if focal_length <= 0:
            camera_template = Camera.from_fov(
                fov_deg=47.0,
                image_size=(width, height),
                is_horizontal_fov=True
            )
        else:
            camera_template = Camera(
                focal_length=(focal_length, focal_length),
                image_size=(width, height)
            )

        # Create OrbitPath and generate cameras
        path_gen = OrbitPath(
            target=orbit_center,
            radius=radius
        )

        cameras = path_gen.sinusoidal(
            n_frames=n_frames,
            amplitude_deg=amplitude_deg,
            n_cycles=n_cycles,
            start_azimuth_deg=start_azimuth_deg,
            camera_template=camera_template
        )

        return ({
            "cameras": cameras,
            "orbit_center": orbit_center,
            "orbit_radius": float(radius),
            "pattern": "sinusoidal",
            "n_frames": int(n_frames),
        },)


class Body2COLMAP_HelicalPath:
    """Generate helical camera path - best for 3D Gaussian Splatting training."""

    CATEGORY = "Body2COLMAP/Path"
    FUNCTION = "generate_path"
    RETURN_TYPES = ("B2C_PATH",)
    RETURN_NAMES = ("camera_path",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_data": ("SAM3D_OUTPUT",),
                "n_frames": ("INT", {
                    "default": 120,
                    "min": 4,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Total number of frames across all loops"
                }),
                "n_loops": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of full 360° rotations"
                }),
                "amplitude_deg": ("FLOAT", {
                    "default": 30.0,
                    "min": 0.0,
                    "max": 89.0,
                    "step": 1.0,
                    "tooltip": "Elevation range: camera goes from -amplitude to +amplitude"
                }),
            },
            "optional": {
                "lead_in_deg": ("FLOAT", {
                    "default": 45.0,
                    "min": 0.0,
                    "max": 180.0,
                    "step": 5.0,
                    "tooltip": "Degrees of rotation at bottom before ascending"
                }),
                "lead_out_deg": ("FLOAT", {
                    "default": 45.0,
                    "min": 0.0,
                    "max": 180.0,
                    "step": 5.0,
                    "tooltip": "Degrees of rotation at top after ascending"
                }),
                "radius": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Orbit radius (0=auto-compute)"
                }),
                "fill_ratio": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "start_azimuth_deg": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 5.0,
                }),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64}),
                "focal_length": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0}),
            }
        }

    def generate_path(self, mesh_data, n_frames, n_loops, amplitude_deg,
                      lead_in_deg=45.0, lead_out_deg=45.0,
                      radius=0.0, fill_ratio=0.8, start_azimuth_deg=0.0,
                      width=512, height=512, focal_length=0.0):
        """Generate helical orbit camera path."""

        # Convert SAM3D output to Scene
        scene = sam3d_output_to_scene(mesh_data, include_skeleton=False)

        # Get orbit center
        orbit_center = scene.get_bbox_center()

        # Auto-compute radius if needed
        if radius <= 0.0:
            radius = OrbitPath.auto_compute_radius(
                scene_bounds=scene.get_bounds(),
                fill_ratio=fill_ratio,
                fov_deg=47.0
            )

        # Create camera template
        if focal_length <= 0:
            camera_template = Camera.from_fov(
                fov_deg=47.0,
                image_size=(width, height),
                is_horizontal_fov=True
            )
        else:
            camera_template = Camera(
                focal_length=(focal_length, focal_length),
                image_size=(width, height)
            )

        # Create OrbitPath and generate cameras
        path_gen = OrbitPath(
            target=orbit_center,
            radius=radius
        )

        cameras = path_gen.helical(
            n_frames=n_frames,
            n_loops=n_loops,
            amplitude_deg=amplitude_deg,
            lead_in_deg=lead_in_deg,
            lead_out_deg=lead_out_deg,
            start_azimuth_deg=start_azimuth_deg,
            camera_template=camera_template
        )

        return ({
            "cameras": cameras,
            "orbit_center": orbit_center,
            "orbit_radius": float(radius),
            "pattern": "helical",
            "n_frames": int(n_frames),
        },)
