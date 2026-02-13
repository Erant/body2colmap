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

        # Create pyrender scene with strong ambient light
        pr_scene = pyrender.Scene(
            bg_color=[0, 0, 0, 0],  # Transparent background
            ambient_light=[0.7, 0.7, 0.7]  # Strong ambient for even lighting
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

        # Add multiple directional lights for even illumination from all sides
        # Front light (positive Z)
        light_front = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
        pr_scene.add(light_front, pose=np.eye(4))

        # Back light (negative Z)
        light_back = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
        back_pose = np.eye(4)
        back_pose[:3, :3] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])  # Rotate 180° around Y
        pr_scene.add(light_back, pose=back_pose)

        # Left light (positive X)
        light_left = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.5)
        left_pose = np.eye(4)
        left_pose[:3, :3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])  # Rotate 90° around Y
        pr_scene.add(light_left, pose=left_pose)

        # Right light (negative X)
        light_right = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.5)
        right_pose = np.eye(4)
        right_pose[:3, :3] = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])  # Rotate -90° around Y
        pr_scene.add(light_right, pose=right_pose)

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
        pr_camera = pyrender.IntrinsicsCamera(
            fx=camera.fx, fy=camera.fy,
            cx=camera.cx, cy=camera.cy
        )
        camera_pose = camera.get_c2w()
        pr_scene.add(pr_camera, pose=camera_pose)

        # Render
        renderer = self._get_pyrender_renderer()
        color, depth = renderer.render(pr_scene, flags=pyrender.RenderFlags.RGBA)

        # Make color array writable (pyrender may return read-only array on some platforms)
        if not color.flags.writeable:
            color = np.array(color, copy=True)

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
        pr_camera = pyrender.IntrinsicsCamera(
            fx=camera.fx, fy=camera.fy,
            cx=camera.cx, cy=camera.cy
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
        use_openpose_colors: bool = True,
        render_bones: bool = True,
        target_format: str = "openpose_body25_hands",
        face_mode: str = None,
        face_landmarks: Optional[NDArray[np.float32]] = None,
        face_max_angle: float = 90.0,
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
            render_bones: If False, only render joints (no bone cylinders)
            target_format: Skeleton format to render ("openpose_body25_hands", "mhr70", etc.)
            face_mode: Face landmark rendering mode:
                - None: No face landmarks (default)
                - "full": Points + connectivity lines
                - "points": Points only, no connecting lines
            face_landmarks: Optional custom face landmarks in OpenPose Face 70
                format, shape (70, 3). If provided, used instead of the
                canonical face model for Procrustes fitting.
            face_max_angle: Maximum angle (degrees) off the face normal at which
                face landmarks are rendered. 90 = full frontal hemisphere.

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

        # Convert skeleton format if needed
        skeleton_joints = self.scene.skeleton_joints
        skeleton_format = target_format

        if self.scene.skeleton_format == "mhr70" and target_format == "openpose_body25_hands":
            # Convert MHR70 → OpenPose Body25+Hands
            skeleton_joints = skel_module.convert_mhr70_to_openpose_body25_hands(self.scene.skeleton_joints)
        elif self.scene.skeleton_format != target_format:
            raise ValueError(
                f"Conversion from {self.scene.skeleton_format} to {target_format} not implemented"
            )

        # Create pyrender scene
        pr_scene = pyrender.Scene(
            bg_color=[0, 0, 0, 0],  # Transparent background
            ambient_light=[1.0, 1.0, 1.0]  # Full bright ambient for clear skeleton
        )

        # Get bone connectivity and colors (needed for both bones and joints)
        bones = skel_module.get_skeleton_bones(skeleton_format)

        # Get bone colors (OpenPose style or single color)
        if use_openpose_colors:
            bone_colors = skel_module.get_bone_colors_openpose_style(skeleton_format)
        else:
            default_bone_color = bone_color if bone_color is not None else (0.0, 1.0, 0.0)
            bone_colors = {bone: default_bone_color for bone in bones}

        # Add bones as cylinders FIRST (so joints render on top)
        if render_bones:

            for start_idx, end_idx in bones:
                if start_idx >= len(skeleton_joints) or end_idx >= len(skeleton_joints):
                    continue  # Skip invalid bone indices

                start_pos = skeleton_joints[start_idx]
                end_pos = skeleton_joints[end_idx]

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

        # Compute per-joint colors from bone colors
        if use_openpose_colors:
            joint_colors_list = skel_module.get_joint_colors_from_bones(bone_colors, len(skeleton_joints))
        else:
            # Use single color for all joints
            joint_colors_list = [joint_color] * len(skeleton_joints)

        # Add joints as spheres LAST (render on top of bones)
        for joint_idx, joint_pos in enumerate(skeleton_joints):
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=joint_radius)
            sphere.vertices += joint_pos

            # Use per-joint color
            this_joint_color = joint_colors_list[joint_idx]
            sphere.visual.vertex_colors = np.array([
                int(this_joint_color[0] * 255),
                int(this_joint_color[1] * 255),
                int(this_joint_color[2] * 255),
                255
            ], dtype=np.uint8)

            mesh = pyrender.Mesh.from_trimesh(sphere, smooth=False)
            pr_scene.add(mesh)

        # Add camera
        pr_camera = pyrender.IntrinsicsCamera(
            fx=camera.fx, fy=camera.fy,
            cx=camera.cx, cy=camera.cy
        )
        pr_scene.add(pr_camera, pose=camera.get_c2w())

        # Render skeleton
        renderer = self._get_pyrender_renderer()
        skel_color, _ = renderer.render(pr_scene, flags=pyrender.RenderFlags.RGBA)

        # Render face landmarks as separate pass and composite on top
        if face_mode is not None:
            face_image = self._render_face(
                skeleton_joints, camera, face_mode,
                face_joint_radius=joint_radius * 0.35,
                face_bone_radius=bone_radius * 0.35,
                face_landmarks=face_landmarks,
                face_max_angle=face_max_angle,
            )
            if face_image is not None:
                skel_color = np.array(skel_color, copy=True)
                face_alpha = face_image[:, :, 3:4] / 255.0
                skel_color[:, :, :3] = (
                    face_image[:, :, :3] * face_alpha +
                    skel_color[:, :, :3] * (1 - face_alpha)
                ).astype(np.uint8)
                skel_color[:, :, 3] = np.maximum(
                    skel_color[:, :, 3], face_image[:, :, 3]
                )

        return skel_color

    def _render_face(
        self,
        skeleton_joints: NDArray[np.float32],
        camera: Camera,
        face_mode: str,
        face_joint_radius: float = 0.005,
        face_bone_radius: float = 0.003,
        face_landmarks: Optional[NDArray[np.float32]] = None,
        face_max_angle: float = 90.0,
    ) -> Optional[NDArray[np.uint8]]:
        """
        Render face landmarks as a separate RGBA image.

        Rendered in its own pyrender scene so it can be composited on top
        of the body skeleton without depth-test occlusion from the larger
        skeleton head joints.

        Args:
            skeleton_joints: Skeleton joints in world coords (OpenPose Body25+)
            camera: Camera for visibility test and rendering
            face_mode: "full" (points + connectivity) or "points" (points only)
            face_joint_radius: Radius for face keypoint spheres
            face_bone_radius: Radius for face connection cylinders
            face_landmarks: Optional custom face landmarks in OpenPose Face 70
                format, shape (70, 3). Passed to fit_face_to_skeleton() to use
                instead of the canonical face model.

        Returns:
            RGBA image with face landmarks, or None if face is not visible
        """
        import trimesh
        import pyrender
        from . import face as face_module

        # Check we have enough joints for the 5 anchor points
        max_anchor = max(face_module.SKELETON_ANCHOR_JOINT_INDICES)
        if len(skeleton_joints) <= max_anchor:
            return None

        # Fit face to skeleton (custom landmarks override canonical model)
        fitted_landmarks, residual = face_module.fit_face_to_skeleton(
            skeleton_joints, face_landmarks=face_landmarks
        )

        # Check visibility (angle threshold from face normal)
        if not face_module.is_face_visible(
            fitted_landmarks, camera.position, max_angle_deg=face_max_angle
        ):
            return None

        # Create separate scene for face
        pr_scene = pyrender.Scene(
            bg_color=[0, 0, 0, 0],
            ambient_light=[1.0, 1.0, 1.0]
        )

        face_color_uint8 = np.array([255, 255, 255, 255], dtype=np.uint8)

        # Add face bones as cylinders (if full mode)
        if face_mode == "full":
            for start_idx, end_idx in face_module.OPENPOSE_FACE_BONES:
                start_pos = fitted_landmarks[start_idx]
                end_pos = fitted_landmarks[end_idx]

                direction = end_pos - start_pos
                length = np.linalg.norm(direction)

                if length < 1e-6:
                    continue

                cylinder = trimesh.creation.cylinder(
                    radius=face_bone_radius,
                    height=length,
                    sections=6
                )

                z_axis = np.array([0, 0, 1], dtype=np.float32)
                bone_dir = direction / length

                rot_axis = np.cross(z_axis, bone_dir)
                rot_axis_len = np.linalg.norm(rot_axis)

                if rot_axis_len > 1e-6:
                    rot_axis = rot_axis / rot_axis_len
                    cos_angle = np.dot(z_axis, bone_dir)
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                    rot_matrix = trimesh.transformations.rotation_matrix(angle, rot_axis)
                else:
                    if np.dot(z_axis, bone_dir) > 0:
                        rot_matrix = np.eye(4)
                    else:
                        rot_matrix = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])

                center = (start_pos + end_pos) / 2
                rot_matrix[:3, 3] = center
                cylinder.apply_transform(rot_matrix)

                cylinder.visual.vertex_colors = face_color_uint8
                mesh = pyrender.Mesh.from_trimesh(cylinder, smooth=False)
                pr_scene.add(mesh)

        # Add face joints as spheres
        for face_pos in fitted_landmarks:
            sphere = trimesh.creation.icosphere(subdivisions=1, radius=face_joint_radius)
            sphere.vertices += face_pos
            sphere.visual.vertex_colors = face_color_uint8
            mesh = pyrender.Mesh.from_trimesh(sphere, smooth=False)
            pr_scene.add(mesh)

        # Add camera and render
        pr_camera = pyrender.IntrinsicsCamera(
            fx=camera.fx, fy=camera.fy,
            cx=camera.cx, cy=camera.cy
        )
        pr_scene.add(pr_camera, pose=camera.get_c2w())

        renderer = self._get_pyrender_renderer()
        color, _ = renderer.render(pr_scene, flags=pyrender.RenderFlags.RGBA)
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

        # Determine face mode and custom landmarks from composite modes
        face_mode = None
        custom_face_landmarks = None
        face_max_angle = 90.0
        if "face" in modes and self.scene.skeleton_joints is not None:
            face_opts = modes["face"] if isinstance(modes["face"], dict) else {}
            face_mode = face_opts.get("face_mode", "full")
            custom_face_landmarks = face_opts.get("face_landmarks")
            face_max_angle = face_opts.get("face_max_angle", 90.0)

        # If skeleton is present but no mesh/depth base, render skeleton directly
        if base_image is None:
            if "skeleton" not in modes:
                raise ValueError("Must specify 'mesh', 'depth', or 'skeleton' as base layer")

            skel_opts = modes["skeleton"] if isinstance(modes["skeleton"], dict) else {}
            return self.render_skeleton(
                camera,
                joint_radius=skel_opts.get("joint_radius", 0.015),
                bone_radius=skel_opts.get("bone_radius", 0.008),
                joint_color=skel_opts.get("joint_color", (1.0, 0.0, 0.0)),
                bone_color=skel_opts.get("bone_color", (0.0, 1.0, 0.0)),
                face_mode=face_mode,
                face_landmarks=custom_face_landmarks,
                face_max_angle=face_max_angle,
            )

        # Overlay skeleton if requested
        if "skeleton" in modes and self.scene.skeleton_joints is not None:
            skel_opts = modes["skeleton"] if isinstance(modes["skeleton"], dict) else {}

            # Render skeleton (with optional face landmarks)
            skel_image = self.render_skeleton(
                camera,
                joint_radius=skel_opts.get("joint_radius", 0.015),
                bone_radius=skel_opts.get("bone_radius", 0.008),
                joint_color=skel_opts.get("joint_color", (1.0, 0.0, 0.0)),
                bone_color=skel_opts.get("bone_color", (0.0, 1.0, 0.0)),
                face_mode=face_mode,
                face_landmarks=custom_face_landmarks,
                face_max_angle=face_max_angle,
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
