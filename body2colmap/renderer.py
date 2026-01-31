"""
Rendering engine using pyribbit (pyrender fork).

This module provides the Renderer class which handles actual image generation
using pyribbit. Supports multiple rendering modes: mesh, depth, skeleton, and
composite modes.

All rendering happens in world coordinates (no coordinate conversion needed
since pyribbit uses OpenGL convention matching our world coords).
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from numpy.typing import NDArray

from .scene import Scene
from .camera import Camera
from . import edges as edge_module


class Renderer:
    """
    Render images from Scene using pyribbit.

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
                import pyribbit as pyrender
            except ImportError:
                raise ImportError(
                    "pyribbit is required for rendering. "
                    "Install with: pip install pyribbit"
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
            import pyribbit as pyrender
            import trimesh
        except ImportError:
            raise ImportError(
                "pyribbit and trimesh are required. "
                "Install with: pip install pyribbit trimesh"
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
            import pyribbit as pyrender
        except ImportError:
            raise ImportError("pyribbit is required")

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

    def render_depth_raw(
        self,
        camera: Camera
    ) -> NDArray[np.float32]:
        """
        Render raw depth buffer (linear depth values).

        Args:
            camera: Camera to render from

        Returns:
            Depth buffer, shape (height, width), dtype float32
            Values are linear depth in world units, 0 for background
        """
        try:
            import pyribbit as pyrender
        except ImportError:
            raise ImportError("pyribbit is required")

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
        _, depth = renderer.render(pr_scene)

        return depth.astype(np.float32)

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
            import pyribbit as pyrender
        except ImportError:
            raise ImportError("pyribbit is required")

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

    def render_normals(
        self,
        camera: Camera,
        space: str = "camera"
    ) -> NDArray[np.uint8]:
        """
        Render surface normal map.

        Normal maps encode surface orientation as RGB colors, providing
        view-independent 3D structure information useful for ControlNet
        conditioning.

        Args:
            camera: Camera to render from
            space: Coordinate space for normals:
                  - "camera": Normals in camera space (blue = facing camera)
                  - "world": Normals in world space

        Returns:
            RGBA image, shape (height, width, 4), dtype uint8
            RGB encodes normal direction: R=(Nx+1)/2, G=(Ny+1)/2, B=(Nz+1)/2
            Alpha = 255 where geometry exists, 0 for background

        Note:
            Camera-space normals are more useful for ControlNet as they're
            view-consistent (surfaces facing the camera are always blue).
        """
        try:
            import pyribbit as pyrender
            import trimesh
        except ImportError:
            raise ImportError("pyribbit is required")

        # Create trimesh to compute normals
        mesh_tm = trimesh.Trimesh(
            vertices=self.scene.vertices,
            faces=self.scene.faces,
            process=False
        )

        # Get vertex normals (trimesh computes these automatically)
        vertex_normals = mesh_tm.vertex_normals.copy()

        # Transform normals to camera space if requested
        if space == "camera":
            # Get world-to-camera rotation (no translation for normals)
            c2w = camera.get_c2w()
            w2c_rot = c2w[:3, :3].T  # Inverse rotation
            vertex_normals = (w2c_rot @ vertex_normals.T).T

        # Normalize (should already be normalized, but ensure)
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        vertex_normals = vertex_normals / norms

        # Encode normals as colors: (n + 1) / 2 maps [-1, 1] to [0, 1]
        normal_colors = ((vertex_normals + 1) / 2 * 255).astype(np.uint8)
        # Add alpha channel
        normal_colors_rgba = np.hstack([
            normal_colors,
            np.full((len(normal_colors), 1), 255, dtype=np.uint8)
        ])

        # Create mesh with normal colors as vertex colors
        mesh_tm.visual.vertex_colors = normal_colors_rgba

        # Create pyrender scene
        pr_scene = pyrender.Scene(
            bg_color=[0, 0, 0, 0],
            ambient_light=[1.0, 1.0, 1.0]  # Full ambient - no shading
        )

        # Create pyrender mesh with flat shading to preserve normal colors
        pr_mesh = pyrender.Mesh.from_trimesh(mesh_tm, smooth=False)
        pr_scene.add(pr_mesh)

        # Add camera
        pr_camera = pyrender.PerspectiveCamera(
            yfov=2 * np.arctan(self.height / (2 * camera.fy)),
            aspectRatio=self.width / self.height
        )
        pr_scene.add(pr_camera, pose=camera.get_c2w())

        # Render
        renderer = self._get_pyrender_renderer()
        color, depth = renderer.render(pr_scene, flags=pyrender.RenderFlags.RGBA)

        # Ensure writable
        if not color.flags.writeable:
            color = np.array(color, copy=True)

        return color

    def render_skeleton(
        self,
        camera: Camera,
        joint_radius: float = 0.02,
        bone_radius: float = 0.012,
        joint_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        bone_color: Tuple[float, float, float] = None,
        use_openpose_colors: bool = True,
        render_bones: bool = True,
        target_format: str = "openpose_body25_hands"
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
            import pyribbit as pyrender
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
        pr_camera = pyrender.PerspectiveCamera(
            yfov=2 * np.arctan(self.height / (2 * camera.fy)),
            aspectRatio=self.width / self.height
        )
        pr_scene.add(pr_camera, pose=camera.get_c2w())

        # Render
        renderer = self._get_pyrender_renderer()
        color, depth = renderer.render(pr_scene, flags=pyrender.RenderFlags.RGBA)

        return color

    def render_edges(
        self,
        camera: Camera,
        source: str = "mesh",
        method: str = "canny",
        edge_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        mesh_color: Optional[Tuple[float, float, float]] = None,
        bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        **edge_kwargs
    ) -> NDArray[np.uint8]:
        """
        Render edges detected from mesh or depth.

        This follows the same pattern as skeleton rendering - edges are rendered
        with a transparent background so they can be composited onto other modes.

        Args:
            camera: Camera to render from
            source: What to detect edges on:
                   - "mesh": Detect edges on rendered mesh image
                   - "depth": Detect edges on depth buffer
            method: Edge detection method ("canny", "sobel", "laplacian")
            edge_color: RGB color (0-1 range) for edge pixels
            mesh_color: RGB color for mesh (only used when source="mesh")
            bg_color: RGB color for background (only used when source="mesh")
            **edge_kwargs: Method-specific parameters:
                - canny: low_threshold (int), high_threshold (int), blur_kernel (int)
                - sobel: kernel_size (int), threshold (int)
                - laplacian: kernel_size (int), threshold (int)

        Returns:
            RGBA image, shape (height, width, 4), dtype uint8
            - Edge pixels: edge_color with alpha=255
            - Non-edge pixels: transparent (alpha=0)

        Note:
            Like skeleton, edges do NOT contribute to alpha for masking purposes.
            The alpha channel only indicates where edges exist for compositing.
        """
        # Render source image
        if source == "mesh":
            source_image = self.render_mesh(
                camera,
                mesh_color=mesh_color,
                bg_color=bg_color
            )
        elif source == "depth":
            source_image = self.render_depth(
                camera,
                normalize=True,
                colormap=None  # Grayscale works best for edge detection
            )
        else:
            raise ValueError(f"Unknown edge source: {source}. Use 'mesh' or 'depth'.")

        # Detect edges
        edges = edge_module.detect_edges(
            source_image[:, :, :3],  # Use RGB, ignore alpha
            method=method,
            **edge_kwargs
        )

        # Convert to RGBA with specified color
        rgba = edge_module.edges_to_rgba(
            edges,
            color=edge_color,
            background_alpha=0  # Transparent background for compositing
        )

        return rgba

    def render_face_ids(
        self,
        camera: Camera
    ) -> NDArray[np.int32]:
        """
        Render face ID buffer for visibility determination.

        Each pixel contains the index of the mesh face visible at that pixel,
        or -1 for background pixels. This is used for projecting textures
        onto the mesh from rendered views.

        Args:
            camera: Camera to render from

        Returns:
            Face ID buffer, shape (height, width), dtype int32
            Values are face indices (0 to num_faces-1) or -1 for background

        Note:
            Uses flat shading with unique colors per face to avoid interpolation.
            Face index is encoded in RGB channels: R + G*256 + B*65536
            Supports up to ~16.7 million faces.
        """
        try:
            import pyribbit as pyrender
            import trimesh
        except ImportError:
            raise ImportError(
                "pyribbit and trimesh are required. "
                "Install with: pip install pyribbit trimesh"
            )

        # Create a mesh with unique color per face
        # We need to split vertices so each face has its own vertices
        num_faces = len(self.scene.faces)

        # Create new vertices array: 3 vertices per face
        split_vertices = self.scene.vertices[self.scene.faces.flatten()]

        # Create new faces array: sequential indices
        split_faces = np.arange(num_faces * 3).reshape(-1, 3)

        # Create colors: encode face index in RGB
        # Face index = R + G*256 + B*65536
        face_colors = np.zeros((num_faces * 3, 4), dtype=np.uint8)
        for face_idx in range(num_faces):
            r = face_idx % 256
            g = (face_idx // 256) % 256
            b = (face_idx // 65536) % 256
            # All 3 vertices of this face get the same color
            face_colors[face_idx * 3:(face_idx + 1) * 3] = [r, g, b, 255]

        # Create trimesh with flat vertex colors
        mesh_tm = trimesh.Trimesh(
            vertices=split_vertices,
            faces=split_faces,
            process=False
        )
        mesh_tm.visual.vertex_colors = face_colors

        # Create pyrender scene with no lighting (flat shading)
        pr_scene = pyrender.Scene(
            bg_color=[0, 0, 0, 0],  # Transparent background (will decode as -1)
            ambient_light=[1.0, 1.0, 1.0]  # Full ambient, no shading
        )

        # Create mesh with flat shading (no smooth normals)
        pr_mesh = pyrender.Mesh.from_trimesh(mesh_tm, smooth=False)
        pr_scene.add(pr_mesh)

        # Add camera
        pr_camera = pyrender.PerspectiveCamera(
            yfov=2 * np.arctan(self.height / (2 * camera.fy)),
            aspectRatio=self.width / self.height
        )
        pr_scene.add(pr_camera, pose=camera.get_c2w())

        # Render with flat shading flag
        renderer = self._get_pyrender_renderer()
        color, depth = renderer.render(
            pr_scene,
            flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.FLAT
        )

        # Decode face IDs from RGB
        face_ids = np.full((self.height, self.width), -1, dtype=np.int32)

        # Where alpha > 0, decode face index
        mask = color[:, :, 3] > 0
        face_ids[mask] = (
            color[mask, 0].astype(np.int32) +
            color[mask, 1].astype(np.int32) * 256 +
            color[mask, 2].astype(np.int32) * 65536
        )

        return face_ids

    def render_textured(
        self,
        camera: Camera,
        texture: NDArray[np.uint8],
        uv_coords: NDArray[np.float32],
        bg_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    ) -> NDArray[np.uint8]:
        """
        Render mesh with a texture applied.

        Args:
            camera: Camera to render from
            texture: Texture image, shape (tex_height, tex_width, 4), RGBA uint8
            uv_coords: UV coordinates per vertex, shape (num_vertices, 2)
                      Values in [0, 1], with (0,0) at bottom-left
            bg_color: Background color RGBA (0-1 range)

        Returns:
            RGBA image, shape (height, width, 4), dtype uint8
        """
        try:
            import pyribbit as pyrender
            import trimesh
            from PIL import Image
        except ImportError:
            raise ImportError(
                "pyribbit, trimesh, and Pillow are required. "
                "Install with: pip install pyribbit trimesh Pillow"
            )

        # Create trimesh with texture
        mesh_tm = trimesh.Trimesh(
            vertices=self.scene.vertices,
            faces=self.scene.faces,
            process=False
        )

        # Create texture as PIL Image
        # Ensure it's a proper contiguous array for PIL
        texture_arr = np.ascontiguousarray(texture)
        texture_image = Image.fromarray(texture_arr, mode='RGBA')

        # Create TextureVisuals with UV coordinates
        # Only pass uv and image - trimesh will create the material internally
        mesh_tm.visual = trimesh.visual.TextureVisuals(
            uv=uv_coords.astype(np.float64),  # trimesh expects float64 for UVs
            image=texture_image
        )

        # Create pyrender scene
        bg_rgba = [
            bg_color[0], bg_color[1], bg_color[2],
            bg_color[3] if len(bg_color) > 3 else 0.0
        ]
        pr_scene = pyrender.Scene(
            bg_color=bg_rgba,
            ambient_light=[1.0, 1.0, 1.0]  # Full ambient for texture visibility
        )

        # Create pyrender mesh from trimesh
        pr_mesh = pyrender.Mesh.from_trimesh(mesh_tm, smooth=False)
        pr_scene.add(pr_mesh)

        # Add camera
        pr_camera = pyrender.PerspectiveCamera(
            yfov=2 * np.arctan(self.height / (2 * camera.fy)),
            aspectRatio=self.width / self.height
        )
        pr_scene.add(pr_camera, pose=camera.get_c2w())

        # Render
        renderer = self._get_pyrender_renderer()
        color, depth = renderer.render(pr_scene, flags=pyrender.RenderFlags.RGBA)

        # Ensure writable
        if not color.flags.writeable:
            color = np.array(color, copy=True)

        return color

    def render_vertex_colors(
        self,
        camera: Camera,
        vertex_colors: NDArray[np.uint8],
        bg_color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    ) -> NDArray[np.uint8]:
        """
        Render mesh with vertex colors applied.

        This is an alternative to texture-based rendering that avoids
        UV overlap issues by storing colors directly on vertices.

        Args:
            camera: Camera to render from
            vertex_colors: Colors per vertex, shape (num_vertices, 4), RGBA uint8
            bg_color: Background color RGBA (0-1 range)

        Returns:
            RGBA image, shape (height, width, 4), dtype uint8
        """
        try:
            import pyribbit as pyrender
            import trimesh
        except ImportError:
            raise ImportError(
                "pyribbit and trimesh are required. "
                "Install with: pip install pyribbit trimesh"
            )

        # Create trimesh with vertex colors
        mesh_tm = trimesh.Trimesh(
            vertices=self.scene.vertices,
            faces=self.scene.faces,
            process=False
        )
        mesh_tm.visual.vertex_colors = vertex_colors

        # Create pyrender scene
        bg_rgba = [
            bg_color[0], bg_color[1], bg_color[2],
            bg_color[3] if len(bg_color) > 3 else 0.0
        ]
        pr_scene = pyrender.Scene(
            bg_color=bg_rgba,
            ambient_light=[1.0, 1.0, 1.0]  # Full ambient for color visibility
        )

        # Create pyrender mesh - use smooth=True for interpolated vertex colors
        pr_mesh = pyrender.Mesh.from_trimesh(mesh_tm, smooth=True)
        pr_scene.add(pr_mesh)

        # Add camera
        pr_camera = pyrender.PerspectiveCamera(
            yfov=2 * np.arctan(self.height / (2 * camera.fy)),
            aspectRatio=self.width / self.height
        )
        pr_scene.add(pr_camera, pose=camera.get_c2w())

        # Render
        renderer = self._get_pyrender_renderer()
        color, depth = renderer.render(pr_scene, flags=pyrender.RenderFlags.RGBA)

        # Ensure writable
        if not color.flags.writeable:
            color = np.array(color, copy=True)

        return color

    def render_composite(
        self,
        camera: Camera,
        modes: Dict[str, Any]
    ) -> NDArray[np.uint8]:
        """
        Render composite of multiple modes (e.g., depth + textured + skeleton).

        Args:
            camera: Camera to render from
            modes: Dictionary specifying which modes to render and their options
                  Example: {
                      "depth": {"normalize": True},
                      "normals": {"space": "camera"},
                      "textured": {"texture": atlas, "uv_coords": uvs},
                      "skeleton": {"joint_radius": 0.015},
                      "edges": {"method": "canny", "color": (1.0, 1.0, 1.0)}
                  }

        Returns:
            RGBA image with composited modes

        Note:
            Modes are composited in order:
            1. mesh, depth, or normals (base layer)
            2. textured (overlay, if present) - mesh with projected texture
            3. skeleton (overlay, if present)
            4. edges (overlay, if present) - per-view edge detection

            Alpha channel comes from base layer only (mesh, depth, or normals).
            Overlays (textured, skeleton, edges) don't affect masking.

            For the texture projection workflow (circular→helical):
            - Use "depth" as base layer
            - Use "textured" with Canny atlas from circular orbit
            - Optionally add "skeleton" for pose guidance

            For ControlNet conditioning:
            - "depth" provides depth information
            - "normals" provides surface orientation (good for normal ControlNet)
            - "textured" with Canny provides edge structure
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
        elif "normals" in modes:
            normal_opts = modes["normals"] if isinstance(modes["normals"], dict) else {}
            base_image = self.render_normals(
                camera,
                space=normal_opts.get("space", "camera")
            )

        if base_image is None:
            raise ValueError("Must specify 'mesh', 'depth', or 'normals' as base layer")

        # Ensure base_image is writable before compositing
        if not base_image.flags.writeable:
            base_image = np.array(base_image, copy=True)

        # Overlay textured mesh if requested (e.g., projected Canny edges)
        # This comes BEFORE skeleton so skeleton renders on top
        if "textured" in modes:
            tex_opts = modes["textured"] if isinstance(modes["textured"], dict) else {}

            texture = tex_opts.get("texture")
            uv_coords = tex_opts.get("uv_coords")

            if texture is None or uv_coords is None:
                raise ValueError(
                    "textured mode requires 'texture' (RGBA array) and 'uv_coords' (UV array)"
                )

            # Render textured mesh with transparent background
            textured_image = self.render_textured(
                camera,
                texture=texture,
                uv_coords=uv_coords,
                bg_color=(0.0, 0.0, 0.0, 0.0)  # Transparent background
            )

            # Composite textured over base using alpha blending
            textured_alpha = textured_image[:, :, 3:4] / 255.0
            base_image[:, :, :3] = (
                textured_image[:, :, :3] * textured_alpha +
                base_image[:, :, :3] * (1 - textured_alpha)
            ).astype(np.uint8)

            # Keep base layer's alpha (textured doesn't affect masking)

        # Overlay vertex-colored mesh if requested (alternative to textured)
        if "vertex_colors" in modes:
            vc_opts = modes["vertex_colors"] if isinstance(modes["vertex_colors"], dict) else {}

            colors = vc_opts.get("colors")

            if colors is None:
                raise ValueError(
                    "vertex_colors mode requires 'colors' (RGBA array per vertex)"
                )

            # Render vertex-colored mesh with transparent background
            vc_image = self.render_vertex_colors(
                camera,
                vertex_colors=colors,
                bg_color=(0.0, 0.0, 0.0, 0.0)
            )

            # Composite over base using alpha blending
            vc_alpha = vc_image[:, :, 3:4] / 255.0
            base_image[:, :, :3] = (
                vc_image[:, :, :3] * vc_alpha +
                base_image[:, :, :3] * (1 - vc_alpha)
            ).astype(np.uint8)

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

        # Overlay edges if requested
        if "edges" in modes:
            edges_opts = modes["edges"] if isinstance(modes["edges"], dict) else {}

            # Determine edge source - defaults to base layer type
            edge_source = edges_opts.get("source")
            if edge_source is None:
                edge_source = "mesh" if "mesh" in modes else "depth"

            # Get mesh color for edge source rendering (if source is mesh)
            edge_mesh_color = None
            edge_bg_color = (1.0, 1.0, 1.0)
            if "mesh" in modes:
                mesh_opts = modes["mesh"] if isinstance(modes["mesh"], dict) else {}
                edge_mesh_color = mesh_opts.get("color")
                edge_bg_color = mesh_opts.get("bg_color", (1.0, 1.0, 1.0))

            # Render edges
            edges_image = self.render_edges(
                camera,
                source=edge_source,
                method=edges_opts.get("method", "canny"),
                edge_color=edges_opts.get("color", (1.0, 1.0, 1.0)),
                mesh_color=edge_mesh_color,
                bg_color=edge_bg_color,
                low_threshold=edges_opts.get("low_threshold", 50),
                high_threshold=edges_opts.get("high_threshold", 150),
                blur_kernel=edges_opts.get("blur_kernel", 5)
            )

            # Composite edges over base using alpha blending
            edges_alpha = edges_image[:, :, 3:4] / 255.0
            base_image[:, :, :3] = (
                edges_image[:, :, :3] * edges_alpha +
                base_image[:, :, :3] * (1 - edges_alpha)
            ).astype(np.uint8)

            # Keep base layer's alpha (edges don't affect masking)

        return base_image

    def __del__(self):
        """Clean up renderer resources."""
        if self._renderer is not None:
            self._renderer.delete()
