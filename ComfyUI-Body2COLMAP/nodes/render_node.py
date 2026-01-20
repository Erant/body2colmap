"""Render node for Body2COLMAP - generates multi-view images."""

import numpy as np
from typing import Tuple
from body2colmap.renderer import Renderer
from body2colmap.scene import Scene
from ..core.sam3d_adapter import sam3d_output_to_scene
from ..core.comfy_utils import rendered_to_comfy


class Body2COLMAP_Render:
    """Render multi-view images of mesh from camera path."""

    CATEGORY = "Body2COLMAP"
    FUNCTION = "render"
    RETURN_TYPES = ("IMAGE", "B2C_RENDER_DATA")
    RETURN_NAMES = ("images", "render_data")
    OUTPUT_TOOLTIPS = (
        "Batch of rendered images (connect to SaveImage or PreviewImage)",
        "Render metadata for COLMAP export (connect to ExportCOLMAP)"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_data": ("SAM3D_OUTPUT",),
                "camera_path": ("B2C_PATH",),
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Image width in pixels"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Image height in pixels"
                }),
                "render_mode": ([
                    "mesh",
                    "depth",
                    "skeleton",
                    "mesh+skeleton",
                    "depth+skeleton"
                ], {
                    "default": "mesh",
                    "tooltip": "What to render: mesh surface, depth map, skeleton, or composites"
                }),
            },
            "optional": {
                # Mesh rendering options
                "mesh_color_r": ("FLOAT", {
                    "default": 0.65,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Mesh color red channel"
                }),
                "mesh_color_g": ("FLOAT", {
                    "default": 0.74,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Mesh color green channel"
                }),
                "mesh_color_b": ("FLOAT", {
                    "default": 0.86,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Mesh color blue channel"
                }),
                "bg_color_r": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Background color red channel"
                }),
                "bg_color_g": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Background color green channel"
                }),
                "bg_color_b": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Background color blue channel"
                }),

                # Skeleton rendering options (for skeleton modes)
                "skeleton_format": ([
                    "openpose_body25_hands",
                    "mhr70"
                ], {
                    "default": "openpose_body25_hands",
                    "tooltip": "Skeleton format for rendering"
                }),
                "joint_radius": ("FLOAT", {
                    "default": 0.015,
                    "min": 0.001,
                    "max": 0.1,
                    "step": 0.001,
                    "tooltip": "Sphere radius for skeleton joints (meters)"
                }),
                "bone_radius": ("FLOAT", {
                    "default": 0.008,
                    "min": 0.001,
                    "max": 0.05,
                    "step": 0.001,
                    "tooltip": "Cylinder radius for skeleton bones (meters)"
                }),

                # Depth rendering options
                "depth_colormap": ([
                    "grayscale",
                    "viridis",
                    "plasma",
                    "inferno",
                    "magma"
                ], {
                    "default": "grayscale",
                    "tooltip": "Colormap for depth visualization"
                }),

                # Camera options
                "focal_length": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10000.0,
                    "step": 1.0,
                    "tooltip": "Focal length in pixels (0=auto for ~47° FOV)"
                }),
            }
        }

    def render(self, mesh_data, camera_path, width, height, render_mode,
               mesh_color_r=0.65, mesh_color_g=0.74, mesh_color_b=0.86,
               bg_color_r=1.0, bg_color_g=1.0, bg_color_b=1.0,
               skeleton_format="openpose_body25_hands",
               joint_radius=0.015, bone_radius=0.008,
               depth_colormap="grayscale",
               focal_length=0.0):
        """
        Render all camera positions and return batch of images.

        Returns:
            images: Tensor of shape [N, H, W, 3] in [0,1] range (ComfyUI IMAGE format)
            render_data: Dict containing cameras and scene for COLMAP export
        """
        # Convert SAM3D output to Scene (with skeleton for skeleton modes)
        include_skeleton = "skeleton" in render_mode
        scene = sam3d_output_to_scene(mesh_data, include_skeleton=include_skeleton)

        # Extract cameras from path
        cameras = camera_path["cameras"]

        # Update camera intrinsics if needed
        if focal_length > 0:
            # Update all cameras with new focal length
            for cam in cameras:
                cam.width = width
                cam.height = height
                cam.fx = focal_length
                cam.fy = focal_length
                # Update principal point to image center
                cam.cx = width / 2.0
                cam.cy = height / 2.0
        else:
            # Just update resolution
            for cam in cameras:
                cam.width = width
                cam.height = height
                # Update principal point to image center
                cam.cx = width / 2.0
                cam.cy = height / 2.0

        # Get actual focal length
        actual_focal_length = cameras[0].fx

        # Prepare render colors
        mesh_color = (mesh_color_r, mesh_color_g, mesh_color_b)
        bg_color = (bg_color_r, bg_color_g, bg_color_b)

        # Create renderer - requires scene and render_size tuple
        renderer = Renderer(scene=scene, render_size=(width, height))

        # Render all frames
        rendered_images = []

        for i, camera in enumerate(cameras):
            # Determine render mode
            if render_mode == "mesh":
                img = renderer.render_mesh(
                    camera=camera,
                    mesh_color=mesh_color,
                    bg_color=bg_color,
                )
            elif render_mode == "depth":
                img = renderer.render_depth(
                    camera=camera,
                    colormap=depth_colormap,
                )
            elif render_mode == "skeleton":
                img = renderer.render_skeleton(
                    camera=camera,
                    target_format=skeleton_format,
                    joint_radius=joint_radius,
                    bone_radius=bone_radius,
                )
            elif render_mode == "mesh+skeleton":
                img = renderer.render_composite(
                    camera=camera,
                    modes={
                        "mesh": {"color": mesh_color, "bg_color": bg_color},
                        "skeleton": {
                            "target_format": skeleton_format,
                            "joint_radius": joint_radius,
                            "bone_radius": bone_radius
                        }
                    }
                )
            elif render_mode == "depth+skeleton":
                img = renderer.render_composite(
                    camera=camera,
                    modes={
                        "depth": {"colormap": depth_colormap},
                        "skeleton": {
                            "target_format": skeleton_format,
                            "joint_radius": joint_radius,
                            "bone_radius": bone_radius
                        }
                    }
                )
            else:
                raise ValueError(f"Unknown render mode: {render_mode}")

            rendered_images.append(img)

        # Convert to ComfyUI IMAGE format
        images_tensor = rendered_to_comfy(rendered_images)

        # Package render data for COLMAP export
        render_data = {
            "cameras": cameras,
            "scene": scene,
            "resolution": (width, height),
            "focal_length": actual_focal_length,
        }

        return (images_tensor, render_data)
