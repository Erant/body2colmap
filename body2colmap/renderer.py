"""
Rendering engine using pyrender.

This module provides the Renderer class which handles actual image generation
using pyrender. Supports multiple rendering modes: mesh, depth, skeleton, and
composite modes.

All rendering happens in world coordinates (no coordinate conversion needed
since pyrender uses OpenGL convention matching our world coords).
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from numpy.typing import NDArray

from .scene import Scene
from .camera import Camera


class Renderer:
    """
    Render images from Scene using pyrender.

    Supports multiple rendering modes:
    - mesh: Colored mesh with lighting
    - depth: Depth buffer
    - skeleton: 3D skeleton as spheres (joints) and cylinders (bones)
    - composite: Combinations of the above

    All modes support alpha channel output.
    """

    def __init__(
        self,
        scene: Scene,
        render_size: Tuple[int, int] = (512, 512)
    ):
        """
        Initialize Renderer.

        Args:
            scene: Scene to render
            render_size: (width, height) in pixels
        """
        self.scene = scene
        self.width, self.height = render_size

        # Will be lazily initialized
        self._pyrender_scene = None
        self._renderer = None

    def _get_pyrender_renderer(self):
        """
        Get or create pyrender OffscreenRenderer.

        Lazily creates the renderer to avoid OpenGL initialization
        until actually needed.

        Returns:
            pyrender.OffscreenRenderer instance
        """
        if self._renderer is None:
            try:
                import pyrender
            except ImportError:
                raise ImportError(
                    "pyrender is required for rendering. "
                    "Install with: pip install pyrender"
                )

            self._renderer = pyrender.OffscreenRenderer(
                viewport_width=self.width,
                viewport_height=self.height
            )

        return self._renderer

    def _create_pyrender_scene(
        self,
        mesh_color: Optional[Tuple[float, float, float]] = None,
        include_skeleton: bool = False
    ):
        """
        Create pyrender.Scene with mesh and optional skeleton.

        Args:
            mesh_color: RGB color (0-1 range) for mesh
                       If None, uses default light gray
            include_skeleton: Whether to add skeleton to scene

        Returns:
            pyrender.Scene instance

        Note:
            The scene geometry is in world coordinates.
            No coordinate conversion needed here.
        """
        try:
            import pyrender
            import trimesh
        except ImportError:
            raise ImportError(
                "pyrender and trimesh are required. "
                "Install with: pip install pyrender trimesh"
            )

        # Create pyrender scene
        pr_scene = pyrender.Scene(
            bg_color=[0, 0, 0, 0],  # Transparent background
            ambient_light=[0.3, 0.3, 0.3]
        )

        # Add mesh
        if mesh_color is None:
            mesh_color = (0.65, 0.74, 0.86)  # Light gray-blue

        # Create trimesh with vertex colors
        mesh_tm = trimesh.Trimesh(
            vertices=self.scene.vertices,
            faces=self.scene.faces,
            process=False
        )

        # Set vertex colors (RGBA, 0-255)
        color_rgba = np.array([
            int(mesh_color[0] * 255),
            int(mesh_color[1] * 255),
            int(mesh_color[2] * 255),
            255
        ], dtype=np.uint8)
        mesh_tm.visual.vertex_colors = np.tile(color_rgba, (len(mesh_tm.vertices), 1))

        # Create pyrender mesh
        pr_mesh = pyrender.Mesh.from_trimesh(mesh_tm, smooth=True)
        pr_scene.add(pr_mesh)

        # Add lighting
        # Directional light from above-front
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        pr_scene.add(light, pose=np.eye(4))

        # TODO: Add skeleton rendering if requested
        # if include_skeleton and self.scene.skeleton_joints is not None:
        #     self._add_skeleton_to_scene(pr_scene)

        return pr_scene

    def render_mesh(
        self,
        camera: Camera,
        mesh_color: Optional[Tuple[float, float, float]] = None,
        bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> NDArray[np.uint8]:
        """
        Render mesh with lighting.

        Args:
            camera: Camera to render from
            mesh_color: RGB color (0-1 range) for mesh
            bg_color: RGB color (0-1 range) for background

        Returns:
            RGBA image, shape (height, width, 4), dtype uint8
            Alpha = 255 where mesh, 0 where background
        """
        try:
            import pyrender
        except ImportError:
            raise ImportError("pyrender is required")

        # Create scene
        pr_scene = self._create_pyrender_scene(mesh_color=mesh_color)

        # Add camera to scene
        pr_camera = pyrender.PerspectiveCamera(
            yfov=2 * np.arctan(self.height / (2 * camera.fy)),
            aspectRatio=self.width / self.height
        )
        camera_pose = camera.get_c2w()
        pr_scene.add(pr_camera, pose=camera_pose)

        # Render
        renderer = self._get_pyrender_renderer()
        color, depth = renderer.render(pr_scene, flags=pyrender.RenderFlags.RGBA)

        # Replace background color (where alpha=0)
        bg_rgba = np.array([
            int(bg_color[0] * 255),
            int(bg_color[1] * 255),
            int(bg_color[2] * 255),
            0
        ], dtype=np.uint8)

        mask = color[:, :, 3] == 0
        color[mask] = bg_rgba

        return color

    def render_depth(
        self,
        camera: Camera,
        normalize: bool = True,
        colormap: Optional[str] = None
    ) -> NDArray[np.uint8]:
        """
        Render depth buffer.

        Args:
            camera: Camera to render from
            normalize: If True, normalize depth to 0-1 range for visualization
            colormap: Optional colormap name ("viridis", "plasma", etc.)
                     If None, returns grayscale depth

        Returns:
            RGBA image, shape (height, width, 4), dtype uint8
            Alpha = 255 where depth exists, 0 where no geometry
        """
        try:
            import pyrender
        except ImportError:
            raise ImportError("pyrender is required")

        # Create scene
        pr_scene = self._create_pyrender_scene()

        # Add camera
        pr_camera = pyrender.PerspectiveCamera(
            yfov=2 * np.arctan(self.height / (2 * camera.fy)),
            aspectRatio=self.width / self.height
        )
        pr_scene.add(pr_camera, pose=camera.get_c2w())

        # Render
        renderer = self._get_pyrender_renderer()
        color, depth = renderer.render(pr_scene)

        # Create alpha mask (1 where depth exists)
        alpha = (depth > 0).astype(np.uint8) * 255

        # Normalize depth if requested
        if normalize:
            valid_depth = depth[depth > 0]
            if len(valid_depth) > 0:
                min_depth = valid_depth.min()
                max_depth = valid_depth.max()
                depth_normalized = np.zeros_like(depth)
                mask = depth > 0
                # Invert so closer = white (1.0), farther = black (0.0)
                depth_normalized[mask] = 1.0 - (depth[mask] - min_depth) / (max_depth - min_depth)
            else:
                depth_normalized = depth
        else:
            depth_normalized = depth

        # Apply colormap if requested
        if colormap is not None:
            try:
                import matplotlib.cm as cm
            except ImportError:
                raise ImportError("matplotlib is required for colormaps")

            cmap = cm.get_cmap(colormap)
            depth_colored = cmap(depth_normalized)[:, :, :3]  # RGB
            depth_colored = (depth_colored * 255).astype(np.uint8)
        else:
            # Grayscale
            depth_gray = (depth_normalized * 255).astype(np.uint8)
            depth_colored = np.stack([depth_gray] * 3, axis=-1)

        # Combine with alpha
        rgba = np.dstack([depth_colored, alpha])

        return rgba

    def render_skeleton(
        self,
        camera: Camera,
        joint_radius: float = 0.02,
        bone_radius: float = 0.012,
        joint_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        bone_color: Tuple[float, float, float] = None,
        use_openpose_colors: bool = True
    ) -> NDArray[np.uint8]:
        """
        Render 3D skeleton as spheres (joints) and cylinders (bones).

        Args:
            camera: Camera to render from
            joint_radius: Radius of joint spheres in meters
            bone_radius: Radius of bone cylinders in meters
            joint_color: RGB color (0-1) for joints
            bone_color: RGB color (0-1) for bones (if not using OpenPose colors)
            use_openpose_colors: If True, use OpenPose color scheme for bones

        Returns:
            RGBA image, shape (height, width, 4), dtype uint8

        Note:
            Skeleton does NOT contribute to alpha for masking purposes.
            This is intentional - skeleton is overlay only.

        Raises:
            ValueError: If scene has no skeleton data
        """
        if self.scene.skeleton_joints is None:
            raise ValueError("Scene has no skeleton data")

        try:
            import pyrender
            import trimesh
            from . import skeleton as skel_module
        except ImportError as e:
            raise ImportError(f"Required dependencies not available: {e}")

        # Create pyrender scene
        pr_scene = pyrender.Scene(
            bg_color=[0, 0, 0, 0],  # Transparent background
            ambient_light=[1.0, 1.0, 1.0]  # Full bright ambient for clear skeleton
        )

        # Get bone connectivity
        bones = skel_module.get_skeleton_bones(self.scene.skeleton_format)

        # Get bone colors (OpenPose style or single color)
        if use_openpose_colors:
            bone_colors = skel_module.get_bone_colors_openpose_style(self.scene.skeleton_format)
        else:
            default_bone_color = bone_color if bone_color is not None else (0.0, 1.0, 0.0)
            bone_colors = {bone: default_bone_color for bone in bones}

        # Add bones as cylinders FIRST (so joints render on top)
        for start_idx, end_idx in bones:
            if start_idx >= len(self.scene.skeleton_joints) or end_idx >= len(self.scene.skeleton_joints):
                continue  # Skip invalid bone indices

            start_pos = self.scene.skeleton_joints[start_idx]
            end_pos = self.scene.skeleton_joints[end_idx]

            # Create cylinder connecting start to end
            direction = end_pos - start_pos
            length = np.linalg.norm(direction)

            if length < 1e-6:
                continue  # Skip zero-length bones

            # Create cylinder along Z axis
            cylinder = trimesh.creation.cylinder(
                radius=bone_radius,
                height=length,
                sections=8
            )

            # Rotate and translate to connect joints
            # Cylinder default: along Z axis, centered at origin
            # We need: from start_pos to end_pos

            # Compute rotation to align Z axis with bone direction
            z_axis = np.array([0, 0, 1], dtype=np.float32)
            bone_dir = direction / length

            # Rotation axis: cross product
            rot_axis = np.cross(z_axis, bone_dir)
            rot_axis_len = np.linalg.norm(rot_axis)

            if rot_axis_len > 1e-6:
                rot_axis = rot_axis / rot_axis_len
                # Rotation angle
                cos_angle = np.dot(z_axis, bone_dir)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

                # Build rotation matrix
                rot_matrix = trimesh.transformations.rotation_matrix(angle, rot_axis)
            else:
                # Parallel or anti-parallel
                if np.dot(z_axis, bone_dir) > 0:
                    rot_matrix = np.eye(4)
                else:
                    # 180 degree rotation
                    rot_matrix = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])

            # Translate to center between joints
            center = (start_pos + end_pos) / 2
            rot_matrix[:3, 3] = center

            cylinder.apply_transform(rot_matrix)

            # Get color for this bone
            this_bone_color = bone_colors.get((start_idx, end_idx), (0.0, 1.0, 0.0))
            cylinder.visual.vertex_colors = np.array([
                int(this_bone_color[0] * 255),
                int(this_bone_color[1] * 255),
                int(this_bone_color[2] * 255),
                255
            ], dtype=np.uint8)

            mesh = pyrender.Mesh.from_trimesh(cylinder, smooth=False)
            pr_scene.add(mesh)

        # Add joints as spheres LAST (render on top of bones)
        for joint_pos in self.scene.skeleton_joints:
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=joint_radius)
            sphere.vertices += joint_pos
            sphere.visual.vertex_colors = np.array([
                int(joint_color[0] * 255),
                int(joint_color[1] * 255),
                int(joint_color[2] * 255),
                255
            ], dtype=np.uint8)

            mesh = pyrender.Mesh.from_trimesh(sphere, smooth=False)
            pr_scene.add(mesh)

        # Add camera
        pr_camera = pyrender.PerspectiveCamera(
            yfov=2 * np.arctan(self.height / (2 * camera.fy)),
            aspectRatio=self.width / self.height
        )
        pr_scene.add(pr_camera, pose=camera.get_c2w())

        # Render
        renderer = self._get_pyrender_renderer()
        color, depth = renderer.render(pr_scene, flags=pyrender.RenderFlags.RGBA)

        return color

    def render_composite(
        self,
        camera: Camera,
        modes: Dict[str, Any]
    ) -> NDArray[np.uint8]:
        """
        Render composite of multiple modes (e.g., mesh + skeleton overlay).

        Args:
            camera: Camera to render from
            modes: Dictionary specifying which modes to render and their options
                  Example: {
                      "mesh": {"color": (0.65, 0.74, 0.86)},
                      "skeleton": {"joint_radius": 0.015}
                  }

        Returns:
            RGBA image with composited modes

        Note:
            Modes are composited in order:
            1. mesh or depth (base layer)
            2. skeleton (overlay)

            Alpha channel comes from base layer only (mesh or depth).
        """
        # Render base layer
        base_image = None

        if "mesh" in modes:
            mesh_opts = modes["mesh"] if isinstance(modes["mesh"], dict) else {}
            base_image = self.render_mesh(
                camera,
                mesh_color=mesh_opts.get("color"),
                bg_color=mesh_opts.get("bg_color", (1.0, 1.0, 1.0))
            )
        elif "depth" in modes:
            depth_opts = modes["depth"] if isinstance(modes["depth"], dict) else {}
            base_image = self.render_depth(
                camera,
                normalize=depth_opts.get("normalize", True),
                colormap=depth_opts.get("colormap")
            )

        if base_image is None:
            raise ValueError("Must specify 'mesh' or 'depth' as base layer")

        # Overlay skeleton if requested
        if "skeleton" in modes and self.scene.skeleton_joints is not None:
            skel_opts = modes["skeleton"] if isinstance(modes["skeleton"], dict) else {}

            # Render skeleton
            skel_image = self.render_skeleton(
                camera,
                joint_radius=skel_opts.get("joint_radius", 0.015),
                bone_radius=skel_opts.get("bone_radius", 0.008),
                joint_color=skel_opts.get("joint_color", (1.0, 0.0, 0.0)),
                bone_color=skel_opts.get("bone_color", (0.0, 1.0, 0.0))
            )

            # Composite skeleton over base using alpha blending
            # Skeleton alpha determines blending
            skel_alpha = skel_image[:, :, 3:4] / 255.0
            base_image[:, :, :3] = (
                skel_image[:, :, :3] * skel_alpha +
                base_image[:, :, :3] * (1 - skel_alpha)
            ).astype(np.uint8)

            # Keep base layer's alpha (skeleton doesn't affect masking)

        return base_image

    def __del__(self):
        """Clean up renderer resources."""
        if self._renderer is not None:
            self._renderer.delete()
