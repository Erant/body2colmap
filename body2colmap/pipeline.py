"""
High-level pipeline orchestrating all components.

This module provides the OrbitPipeline class which ties together:
- Scene loading (mesh or Gaussian splat)
- Orbit path generation
- Rendering
- Export to COLMAP and images

This is the main API for users of the library.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray

from .scene import Scene
from .camera import Camera
from .path import OrbitPath
from .renderer import Renderer
from .exporter import ColmapExporter, ImageExporter
from .utils import compute_default_focal_length, compute_auto_orbit_radius


class OrbitPipeline:
    """
    High-level pipeline for orbit rendering and COLMAP export.

    This class orchestrates the entire process:
    1. Load scene from SAM-3D-Body output
    2. Generate camera orbit path
    3. Render frames from each camera
    4. Export images and COLMAP files

    Example:
        pipeline = OrbitPipeline.from_npz_file("estimation.npz")
        pipeline.set_orbit_params(pattern="helical", n_frames=120)
        images = pipeline.render_all(modes=["mesh"])
        pipeline.export_colmap("./output")
        pipeline.export_images("./output", images["mesh"])
    """

    def __init__(
        self,
        scene: Scene,
        render_size: Tuple[int, int] = (512, 512),
        focal_length: Optional[float] = None
    ):
        """
        Initialize pipeline.

        Args:
            scene: Scene to render
            render_size: (width, height) for rendering
            focal_length: Camera focal length in pixels
                         If None, computed for ~47° FOV
        """
        self.scene = scene
        self.render_size = render_size

        # Compute focal length if not provided
        if focal_length is None:
            self.focal_length = compute_default_focal_length(render_size[0])
        else:
            self.focal_length = focal_length

        # Will be set by set_orbit_params()
        self.cameras: Optional[List[Camera]] = None
        self.orbit_params: Optional[Dict[str, Any]] = None

        # Renderer (lazily created)
        self._renderer: Optional[Renderer] = None

    @classmethod
    def from_npz_file(
        cls,
        filepath: str,
        render_size: Tuple[int, int] = (512, 512),
        include_skeleton: bool = False
    ) -> "OrbitPipeline":
        """
        Create pipeline from SAM-3D-Body .npz file.

        Args:
            filepath: Path to .npz file
            render_size: (width, height) for rendering
            include_skeleton: Whether to load skeleton data

        Returns:
            OrbitPipeline instance
        """
        scene = Scene.from_npz_file(filepath, include_skeleton=include_skeleton)
        return cls(scene, render_size)

    @classmethod
    def from_sam3d_output(
        cls,
        output_dict: Dict[str, Any],
        render_size: Tuple[int, int] = (512, 512),
        include_skeleton: bool = False
    ) -> "OrbitPipeline":
        """
        Create pipeline from SAM-3D-Body output dictionary.

        Args:
            output_dict: Dictionary with SAM-3D-Body output
            render_size: (width, height) for rendering
            include_skeleton: Whether to load skeleton data

        Returns:
            OrbitPipeline instance
        """
        scene = Scene.from_sam3d_output(output_dict, include_skeleton)
        return cls(scene, render_size)

    @classmethod
    def from_ply_file(
        cls,
        filepath: str,
        render_size: Tuple[int, int] = (512, 512)
    ) -> "OrbitPipeline":
        """
        Create pipeline from Gaussian Splat PLY file.

        Args:
            filepath: Path to .ply file
            render_size: (width, height) for rendering

        Returns:
            OrbitPipeline instance with SplatScene

        Note:
            Requires gsplat dependencies. Install with:
            pip install body2colmap[splat]
        """
        from .splat_scene import SplatScene
        scene = SplatScene.from_ply(filepath)
        return cls(scene, render_size)

    def _is_splat_scene(self) -> bool:
        """Check if scene is a SplatScene (vs mesh Scene)."""
        # Import here to avoid circular imports and allow optional dependency
        try:
            from .splat_scene import SplatScene
            return isinstance(self.scene, SplatScene)
        except ImportError:
            return False

    @property
    def renderer(self):
        """Get or create the renderer for this pipeline's scene type."""
        if self._renderer is None:
            if self._is_splat_scene():
                from .splat_renderer import SplatRenderer
                self._renderer = SplatRenderer(self.scene, self.render_size)
            else:
                self._renderer = Renderer(self.scene, self.render_size)
        return self._renderer

    def auto_orient(self, rotation_offset_deg: float = 0.0) -> None:
        """
        Rotate the scene so the body faces the camera at orbit frame 0.

        Computes the torso facing direction from the skeleton's shoulder
        and hip joints, then rotates the entire scene (mesh + skeleton)
        around the Y axis so that the body faces -Z (toward the camera).

        An optional offset rotates further from that facing position.

        Args:
            rotation_offset_deg: Additional rotation in degrees after
                auto-facing. 0 = face camera directly, 90 = turned 90
                degrees to the right, etc.
        """
        facing = self.scene.compute_torso_facing_direction()

        if facing is not None:
            # Angle of current facing direction from +Z axis
            current_angle = float(np.arctan2(facing[0], facing[2]))
            # Angle of target (-Z) from +Z axis
            target_angle = float(np.arctan2(0.0, -1.0))  # pi
            correction_deg = float(np.degrees(target_angle - current_angle))
        else:
            correction_deg = 0.0

        total_rotation = correction_deg + rotation_offset_deg
        self.scene.rotate_around_y(total_rotation)

    def set_orbit_params(
        self,
        pattern: str = "helical",
        n_frames: int = 120,
        radius: Optional[float] = None,
        framing: str = "full",
        **kwargs
    ) -> "OrbitPipeline":
        """
        Set orbit path parameters and generate cameras.

        Args:
            pattern: Orbit pattern type ("circular", "sinusoidal", "helical")
            n_frames: Number of frames to generate
            radius: Orbit radius (distance from target)
                   If None, auto-computed to frame scene
            framing: Body framing preset ("full", "torso", "bust", "head")
                    Non-full presets use skeleton joints to determine Y threshold
                    and filter mesh vertices for accurate framing bounds.
            **kwargs: Pattern-specific parameters:
                - circular: elevation_deg
                - sinusoidal: amplitude_deg, n_cycles
                - helical: n_loops, amplitude_deg, lead_in_deg, lead_out_deg

        Returns:
            self (for method chaining)
        """
        # Get framing bounds based on preset
        # For partial body presets, this filters mesh vertices by Y coordinate
        framing_bounds = self.scene.get_framing_bounds(preset=framing)

        # Compute orbit center (look-at target) from framing region
        target = (framing_bounds[0] + framing_bounds[1]) / 2.0

        # Auto-compute radius if not provided
        if radius is None:
            fill_ratio = kwargs.pop('fill_ratio', 0.8)
            radius = compute_auto_orbit_radius(
                bounds=framing_bounds,
                render_size=self.render_size,
                focal_length=self.focal_length,
                fill_ratio=fill_ratio
            )

        # Create orbit path generator
        orbit = OrbitPath(target=target, radius=radius)

        # Create camera template with correct intrinsics
        camera_template = Camera(
            focal_length=(self.focal_length, self.focal_length),
            image_size=self.render_size
        )

        # Generate cameras based on pattern
        if pattern == "circular":
            elevation_deg = kwargs.pop('elevation_deg', 0.0)
            self.cameras = orbit.circular(
                n_frames=n_frames,
                elevation_deg=elevation_deg,
                camera_template=camera_template,
                **kwargs
            )
        elif pattern == "sinusoidal":
            amplitude_deg = kwargs.pop('amplitude_deg', 30.0)
            n_cycles = kwargs.pop('n_cycles', 2)
            self.cameras = orbit.sinusoidal(
                n_frames=n_frames,
                amplitude_deg=amplitude_deg,
                n_cycles=n_cycles,
                camera_template=camera_template,
                **kwargs
            )
        elif pattern == "helical":
            n_loops = kwargs.pop('n_loops', 3)
            amplitude_deg = kwargs.pop('amplitude_deg', 30.0)
            lead_in_deg = kwargs.pop('lead_in_deg', 45.0)
            lead_out_deg = kwargs.pop('lead_out_deg', 45.0)
            self.cameras = orbit.helical(
                n_frames=n_frames,
                n_loops=n_loops,
                amplitude_deg=amplitude_deg,
                lead_in_deg=lead_in_deg,
                lead_out_deg=lead_out_deg,
                camera_template=camera_template,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown orbit pattern: {pattern}")

        # Store params for reference
        self.orbit_params = {
            'pattern': pattern,
            'n_frames': n_frames,
            'radius': radius,
            **kwargs
        }

        return self

    def render_all(
        self,
        modes: List[str] = ["mesh"],
        **render_kwargs
    ) -> Dict[str, List[NDArray[np.uint8]]]:
        """
        Render all frames for specified modes.

        Args:
            modes: List of render modes
                - For mesh scenes: "mesh", "depth", "skeleton"
                - For splat scenes: "splat"
            **render_kwargs: Mode-specific rendering options:
                - mesh_color: RGB tuple (0-1) for mesh
                - bg_color: RGB tuple (0-1) for background
                - normalize_depth: bool for depth rendering
                - etc.

        Returns:
            Dictionary mapping mode name to list of rendered images
            Example: {"mesh": [img1, img2, ...], "depth": [img1, img2, ...]}

        Raises:
            RuntimeError: If cameras haven't been set (call set_orbit_params first)
        """
        if self.cameras is None:
            raise RuntimeError("Cameras not set. Call set_orbit_params() first.")

        # Get appropriate renderer for scene type
        renderer = self.renderer
        is_splat = self._is_splat_scene()

        results = {}

        for mode in modes:
            images = []

            # Render each frame
            for i, camera in enumerate(self.cameras):
                if mode == "splat":
                    # Splat rendering (only valid for SplatScene)
                    if not is_splat:
                        raise ValueError("'splat' mode only valid for SplatScene")
                    image = renderer.render(
                        camera,
                        bg_color=render_kwargs.get('bg_color', (1.0, 1.0, 1.0))
                    )
                elif mode == "mesh":
                    if is_splat:
                        raise ValueError("'mesh' mode not valid for SplatScene, use 'splat'")
                    image = renderer.render_mesh(
                        camera,
                        mesh_color=render_kwargs.get('mesh_color'),
                        bg_color=render_kwargs.get('bg_color', (1.0, 1.0, 1.0))
                    )
                elif mode == "depth":
                    if is_splat:
                        raise ValueError("'depth' mode not yet supported for SplatScene")
                    image = renderer.render_depth(
                        camera,
                        normalize=render_kwargs.get('normalize_depth', True),
                        colormap=render_kwargs.get('depth_colormap')
                    )
                elif mode == "skeleton":
                    if is_splat:
                        raise ValueError("'skeleton' mode not valid for SplatScene")
                    image = renderer.render_skeleton(
                        camera,
                        joint_radius=render_kwargs.get('joint_radius', 0.015),
                        bone_radius=render_kwargs.get('bone_radius', 0.008),
                        face_mode=render_kwargs.get('face_mode'),
                        face_landmarks=render_kwargs.get('face_landmarks')
                    )
                else:
                    raise ValueError(f"Unknown render mode: {mode}")

                images.append(image)

            results[mode] = images

        return results

    def render_composite_all(
        self,
        composite_modes: Dict[str, Dict[str, Any]]
    ) -> List[NDArray[np.uint8]]:
        """
        Render all frames with composite modes (e.g., mesh+skeleton).

        Args:
            composite_modes: Dictionary specifying modes and their options
                Example: {
                    "mesh": {"color": (0.65, 0.74, 0.86), "bg_color": (1, 1, 1)},
                    "skeleton": {"joint_radius": 0.02, "use_openpose_colors": True}
                }

        Returns:
            List of composite rendered images (one per camera)

        Raises:
            RuntimeError: If cameras haven't been set (call set_orbit_params first)
            ValueError: If used with SplatScene (composites not supported)
        """
        if self.cameras is None:
            raise RuntimeError("Cameras not set. Call set_orbit_params() first.")

        if self._is_splat_scene():
            raise ValueError("Composite rendering not supported for SplatScene")

        # Create renderer if needed (mesh renderer for composites)
        renderer = self.renderer

        images = []
        for camera in self.cameras:
            image = renderer.render_composite(camera, composite_modes)
            images.append(image)

        return images

    def compute_original_view_framing(
        self,
        original_focal_length: float,
        fill_ratio: float = 0.8
    ) -> Dict[str, Any]:
        """
        Compute auto-framing parameters for the original SAM-3D-Body viewpoint.

        Projects the mesh through the original camera and computes the scale
        and translation needed to center the subject and fill the frame.

        Since the camera stays at the origin (same viewpoint), this is a pure
        2D operation: scale + translate. The result is a new focal length and
        principal point, plus a 2x3 affine matrix that can be applied to the
        original input image with cv2.warpAffine().

        Args:
            original_focal_length: Focal length from .npz, in pixels
            fill_ratio: How much of the frame the subject should fill (0-1)

        Returns:
            Dictionary with framing parameters:
                scale_factor, framed_focal_length, framed_principal_point,
                affine_matrix (2x3, for cv2.warpAffine on the original image),
                inverse_affine_matrix (2x3, maps framed coords back to original),
                original_2d_bbox [u_min, v_min, u_max, v_max],
                crop_box_in_original (what region of the original maps to output)
        """
        w, h = self.render_size
        cx_orig = w / 2.0
        cy_orig = h / 2.0

        # Create the original camera
        orig_cam = Camera(
            focal_length=(original_focal_length, original_focal_length),
            image_size=self.render_size,
            position=np.zeros(3, dtype=np.float32),
            rotation=np.eye(3, dtype=np.float32)
        )

        # Project all mesh vertices to 2D image coordinates
        points_2d = orig_cam.project(self.scene.vertices)

        # 2D bounding box of the projected mesh
        u_min, v_min = points_2d.min(axis=0)
        u_max, v_max = points_2d.max(axis=0)
        bbox_w = u_max - u_min
        bbox_h = v_max - v_min
        bbox_cx = (u_min + u_max) / 2.0
        bbox_cy = (v_min + v_max) / 2.0

        # Scale factor: make the subject fill fill_ratio of the output
        if bbox_w < 1e-6 or bbox_h < 1e-6:
            s = 1.0
        else:
            s = fill_ratio * min(w / bbox_w, h / bbox_h)

        f_new = original_focal_length * s

        # New principal point: centers the subject in the output
        cx_new = w / 2.0 - s * (bbox_cx - cx_orig)
        cy_new = h / 2.0 - s * (bbox_cy - cy_orig)

        # Affine matrix: maps original image coords → auto-framed coords
        # u' = s * u + tx, v' = s * v + ty
        tx = cx_new - s * cx_orig
        ty = cy_new - s * cy_orig
        affine = [[s, 0.0, tx], [0.0, s, ty]]

        # Inverse affine: maps framed coords → original image coords
        # u = (u' - tx) / s, v = (v' - ty) / s
        inv_s = 1.0 / s
        inv_tx = -tx / s
        inv_ty = -ty / s
        inv_affine = [[inv_s, 0.0, inv_tx], [0.0, inv_s, inv_ty]]

        # Crop box: what region of the original image maps to the output
        # Output corners (0,0) and (w,h) in original-image coords:
        orig_u_left = -tx / s
        orig_v_top = -ty / s
        orig_u_right = (w - tx) / s
        orig_v_bottom = (h - ty) / s

        return {
            'scale_factor': float(s),
            'framed_focal_length': float(f_new),
            'framed_principal_point': [float(cx_new), float(cy_new)],
            'original_focal_length': float(original_focal_length),
            'original_principal_point': [float(cx_orig), float(cy_orig)],
            'affine_matrix': affine,
            'inverse_affine_matrix': inv_affine,
            'original_2d_bbox': [float(u_min), float(v_min),
                                 float(u_max), float(v_max)],
            'crop_box_in_original': [float(orig_u_left), float(orig_v_top),
                                     float(orig_u_right), float(orig_v_bottom)],
        }

    def render_original_view(
        self,
        original_focal_length: float,
        modes: List[str] = ["mesh"],
        auto_frame: bool = False,
        fill_ratio: float = 0.8,
        **render_kwargs
    ) -> Tuple[Dict[str, NDArray[np.uint8]], Optional[Dict[str, Any]]]:
        """
        Render a single frame from the original SAM-3D-Body viewpoint.

        After sam3d_to_world(), the original camera is at the origin with
        identity rotation (the 180-degree X-axis flip that converts SAM-3D
        coords to world coords also converts the camera from OpenCV to
        OpenGL convention). So we just need the original focal length.

        IMPORTANT: The scene must NOT have been auto-oriented or rotated
        for this to produce a correct overlay.

        Args:
            original_focal_length: Focal length from the .npz file, in pixels.
            modes: Render modes (same as render_all)
            auto_frame: If True, adjust focal length and principal point to
                center and fill the frame. Also returns framing metadata.
            fill_ratio: Target fill ratio when auto_frame=True (0-1)
            **render_kwargs: Passed through to render methods

        Returns:
            Tuple of (rendered_images, framing_info):
                rendered_images: Dict mapping mode name to single rendered image
                framing_info: Dict with framing metadata including the Camera
                    object used for rendering (key ``'camera'``).
        """
        if auto_frame:
            framing_info = self.compute_original_view_framing(
                original_focal_length, fill_ratio
            )
            fl = framing_info['framed_focal_length']
            cx, cy = framing_info['framed_principal_point']
        else:
            w, h = self.render_size
            fl = original_focal_length
            cx = w / 2.0
            cy = h / 2.0
            framing_info = {
                'scale_factor': 1.0,
                'framed_focal_length': float(fl),
                'framed_principal_point': [float(cx), float(cy)],
                'original_focal_length': float(original_focal_length),
                'original_principal_point': [float(cx), float(cy)],
                'affine_matrix': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                'inverse_affine_matrix': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            }

        # Camera at origin, identity rotation, chosen intrinsics
        camera = Camera(
            focal_length=(fl, fl),
            image_size=self.render_size,
            principal_point=(cx, cy),
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            rotation=np.eye(3, dtype=np.float32)
        )
        framing_info['camera'] = camera

        renderer = self.renderer
        results = {}

        for mode in modes:
            if '+' in mode:
                # Composite mode — delegate to render_composite
                parts = [p.strip() for p in mode.split('+')]
                base_mode = parts[0]
                overlays = parts[1:]

                # Build composite config from render_kwargs
                composite_modes = {}
                if base_mode == "mesh":
                    composite_modes["mesh"] = {
                        "color": render_kwargs.get('mesh_color'),
                        "bg_color": render_kwargs.get('bg_color', (1.0, 1.0, 1.0)),
                    }
                elif base_mode == "depth":
                    composite_modes["depth"] = {}

                for overlay in overlays:
                    if overlay == "skeleton":
                        composite_modes["skeleton"] = {
                            "joint_radius": render_kwargs.get('joint_radius', 0.015),
                            "bone_radius": render_kwargs.get('bone_radius', 0.008),
                        }
                    elif overlay == "face":
                        composite_modes["face"] = {
                            "face_mode": render_kwargs.get('face_mode', 'full'),
                            "face_landmarks": render_kwargs.get('face_landmarks'),
                        }

                image = renderer.render_composite(camera, composite_modes)
            elif mode == "mesh":
                image = renderer.render_mesh(
                    camera,
                    mesh_color=render_kwargs.get('mesh_color'),
                    bg_color=render_kwargs.get('bg_color', (1.0, 1.0, 1.0))
                )
            elif mode == "depth":
                image = renderer.render_depth(
                    camera,
                    normalize=render_kwargs.get('normalize_depth', True),
                    colormap=render_kwargs.get('depth_colormap')
                )
            elif mode == "skeleton":
                image = renderer.render_skeleton(
                    camera,
                    joint_radius=render_kwargs.get('joint_radius', 0.015),
                    bone_radius=render_kwargs.get('bone_radius', 0.008),
                    face_mode=render_kwargs.get('face_mode'),
                    face_landmarks=render_kwargs.get('face_landmarks')
                )
            else:
                raise ValueError(f"Unknown render mode: {mode}")

            results[mode] = image

        return results, framing_info

    def export_colmap(
        self,
        output_dir: str,
        n_pointcloud_samples: int = 50000,
        filename_pattern: str = "frame_{:04d}.png"
    ) -> Path:
        """
        Export COLMAP format files.

        Creates:
        - output_dir/cameras.txt
        - output_dir/images.txt
        - output_dir/points3D.txt

        Args:
            output_dir: Directory to write files to
            n_pointcloud_samples: Number of points to sample from mesh
            filename_pattern: Filename pattern for images (must match actual image files)

        Returns:
            Path to output directory

        Raises:
            RuntimeError: If cameras haven't been set
        """
        if self.cameras is None:
            raise RuntimeError("Cameras not set. Call set_orbit_params() first.")

        # Generate image filenames using the provided pattern
        image_names = ImageExporter.generate_filenames(
            n_frames=len(self.cameras),
            pattern=filename_pattern
        )

        # Create exporter
        exporter = ColmapExporter.from_scene_and_cameras(
            scene=self.scene,
            cameras=self.cameras,
            image_names=image_names,
            n_pointcloud_samples=n_pointcloud_samples
        )

        # Export
        output_path = Path(output_dir)
        exporter.export(output_path)

        return output_path

    def export_images(
        self,
        output_dir: str,
        images: List[NDArray[np.uint8]],
        filename_pattern: str = "frame_{:04d}.png"
    ) -> List[Path]:
        """
        Export rendered images to files.

        Args:
            output_dir: Directory to save images to
            images: List of images to save
            filename_pattern: Filename format string

        Returns:
            List of paths to saved images
        """
        # Generate filenames
        filenames = ImageExporter.generate_filenames(
            n_frames=len(images),
            pattern=filename_pattern
        )

        # Create exporter
        exporter = ImageExporter(images, filenames)

        # Export
        return exporter.export(Path(output_dir))

    def __repr__(self) -> str:
        """String representation for debugging."""
        n_cams = len(self.cameras) if self.cameras is not None else 0
        return (
            f"OrbitPipeline({self.scene}, "
            f"size={self.render_size}, "
            f"cameras={n_cams})"
        )
