"""
High-level pipeline for texture projection workflow.

This module provides the ProjectionPipeline class that orchestrates the
circular-orbit-to-helical-orbit Canny projection workflow:

1. Generate cameras for circular orbit (reference views)
2. Render mesh and detect Canny edges from circular views
3. Project edges onto UV atlas
4. Generate cameras for helical orbit (target views)
5. Render depth + projected-canny + skeleton composites for helical views

The output is suitable for ControlNet guidance in diffusion models.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from .scene import Scene
from .camera import Camera
from .path import OrbitPath
from .renderer import Renderer
from .exporter import ColmapExporter, ImageExporter
from .utils import compute_default_focal_length, compute_auto_orbit_radius
from . import edges as edge_module
from . import texture_projection as tex_proj


class ProjectionPipeline:
    """
    Pipeline for circular→helical Canny projection workflow.

    This pipeline:
    1. Renders mesh from circular orbit at fixed elevation
    2. Detects Canny edges from each circular view
    3. Projects edges onto UV atlas (texture)
    4. Renders depth + textured-canny + skeleton from helical orbit

    Example:
        pipeline = ProjectionPipeline.from_npz_file("mesh.npz")

        # Step 1: Generate Canny atlas from circular orbit
        pipeline.generate_canny_atlas(
            n_circular_frames=36,
            elevation_deg=0.0
        )

        # Step 2: Render helical orbit with projected Canny
        pipeline.set_helical_orbit(n_frames=120, n_loops=3)
        images = pipeline.render_helical_with_canny(
            modes=["depth", "textured", "skeleton"]
        )

        # Step 3: Export
        pipeline.export_colmap("./output")
        pipeline.export_images("./output", images)
    """

    def __init__(
        self,
        scene: Scene,
        render_size: Tuple[int, int] = (512, 512),
        focal_length: Optional[float] = None,
        atlas_size: Tuple[int, int] = (1024, 1024)
    ):
        """
        Initialize ProjectionPipeline.

        Args:
            scene: Scene containing mesh and optional skeleton
            render_size: (width, height) for rendering
            focal_length: Camera focal length in pixels (None = auto)
            atlas_size: (width, height) for UV atlas
        """
        self.scene = scene
        self.render_size = render_size
        self.atlas_size = atlas_size

        # Compute focal length if not provided
        if focal_length is None:
            self.focal_length = compute_default_focal_length(render_size[0])
        else:
            self.focal_length = focal_length

        # Will be set during processing
        self.circular_cameras: Optional[List[Camera]] = None
        self.helical_cameras: Optional[List[Camera]] = None
        self.uv_coords: Optional[NDArray[np.float32]] = None
        self.canny_atlas: Optional[NDArray[np.uint8]] = None

        # Renderer (lazily created)
        self._renderer: Optional[Renderer] = None

    @classmethod
    def from_npz_file(
        cls,
        filepath: str,
        render_size: Tuple[int, int] = (512, 512),
        include_skeleton: bool = True,
        atlas_size: Tuple[int, int] = (1024, 1024)
    ) -> "ProjectionPipeline":
        """
        Create pipeline from SAM-3D-Body .npz file.

        Args:
            filepath: Path to .npz file
            render_size: (width, height) for rendering
            include_skeleton: Whether to load skeleton data
            atlas_size: (width, height) for UV atlas

        Returns:
            ProjectionPipeline instance
        """
        scene = Scene.from_npz_file(filepath, include_skeleton=include_skeleton)
        return cls(scene, render_size, atlas_size=atlas_size)

    def _get_renderer(self) -> Renderer:
        """Get or create renderer."""
        if self._renderer is None:
            self._renderer = Renderer(self.scene, self.render_size)
        return self._renderer

    def _compute_orbit_radius(self, fill_ratio: float = 0.8) -> float:
        """Compute orbit radius to frame the scene properly."""
        return compute_auto_orbit_radius(
            bounds=self.scene.get_bounds(),
            render_size=self.render_size,
            focal_length=self.focal_length,
            fill_ratio=fill_ratio
        )

    def _create_camera_template(self) -> Camera:
        """Create camera with correct intrinsics."""
        return Camera(
            focal_length=(self.focal_length, self.focal_length),
            image_size=self.render_size
        )

    def generate_canny_atlas(
        self,
        n_circular_frames: int = 36,
        elevation_deg: float = 0.0,
        radius: Optional[float] = None,
        fill_ratio: float = 0.8,
        canny_low: int = 50,
        canny_high: int = 150,
        canny_blur: int = 5,
        edge_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        uv_method: str = "cylindrical",
        blend_mode: str = "max",
        mesh_color: Optional[Tuple[float, float, float]] = None,
        bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> NDArray[np.uint8]:
        """
        Generate Canny edge atlas from circular orbit views.

        This is the main method for step 1 of the workflow:
        - Generates circular orbit cameras
        - Renders mesh from each camera
        - Detects Canny edges
        - Projects edges onto UV atlas

        Args:
            n_circular_frames: Number of views in circular orbit
            elevation_deg: Elevation angle for circular orbit (0 = eye level)
            radius: Orbit radius (None = auto-compute)
            fill_ratio: How much of frame to fill (for auto radius)
            canny_low: Canny low threshold
            canny_high: Canny high threshold
            canny_blur: Gaussian blur kernel size before Canny
            edge_color: RGB color for edges in atlas
            uv_method: UV generation method ("cylindrical", "spherical")
            blend_mode: How to blend overlapping projections ("max", "average")
            mesh_color: Mesh color for rendering (None = default)
            bg_color: Background color for rendering

        Returns:
            Canny atlas, shape (atlas_height, atlas_width, 4), RGBA uint8
        """
        renderer = self._get_renderer()

        # Compute orbit parameters
        if radius is None:
            radius = self._compute_orbit_radius(fill_ratio)

        target = self.scene.get_bbox_center()
        orbit = OrbitPath(target=target, radius=radius)
        camera_template = self._create_camera_template()

        # Generate circular orbit cameras
        self.circular_cameras = orbit.circular(
            n_frames=n_circular_frames,
            elevation_deg=elevation_deg,
            camera_template=camera_template
        )

        # Generate UVs for the mesh
        self.uv_coords = tex_proj.generate_uvs(
            self.scene.vertices,
            self.scene.faces,
            method=uv_method
        )

        # Create projector
        projector = tex_proj.TextureProjector(
            self.scene.vertices,
            self.scene.faces,
            self.uv_coords,
            self.atlas_size
        )

        # Process each circular view
        for camera in self.circular_cameras:
            # Render mesh
            mesh_image = renderer.render_mesh(
                camera,
                mesh_color=mesh_color,
                bg_color=bg_color
            )

            # Detect Canny edges
            edges = edge_module.canny(
                mesh_image[:, :, :3],
                low_threshold=canny_low,
                high_threshold=canny_high,
                blur_kernel=canny_blur
            )

            # Convert to RGBA
            edge_rgba = edge_module.edges_to_rgba(edges, color=edge_color)

            # Render face IDs for this view
            face_ids = renderer.render_face_ids(camera)

            # Project onto atlas
            projector.project_view_fast(edge_rgba, face_ids, blend_mode)

        # Get final atlas
        self.canny_atlas = projector.get_atlas()

        return self.canny_atlas

    def set_helical_orbit(
        self,
        n_frames: int = 120,
        n_loops: int = 3,
        amplitude_deg: float = 30.0,
        lead_in_deg: float = 45.0,
        lead_out_deg: float = 45.0,
        radius: Optional[float] = None,
        fill_ratio: float = 0.8
    ) -> List[Camera]:
        """
        Set up helical orbit cameras for final rendering.

        Args:
            n_frames: Number of frames in helical orbit
            n_loops: Number of full rotations
            amplitude_deg: Maximum elevation deviation
            lead_in_deg: Azimuth range for lead-in (smooth start)
            lead_out_deg: Azimuth range for lead-out (smooth end)
            radius: Orbit radius (None = auto, should match circular orbit)
            fill_ratio: Fill ratio for auto radius

        Returns:
            List of Camera objects for helical orbit
        """
        if radius is None:
            radius = self._compute_orbit_radius(fill_ratio)

        target = self.scene.get_bbox_center()
        orbit = OrbitPath(target=target, radius=radius)
        camera_template = self._create_camera_template()

        self.helical_cameras = orbit.helical(
            n_frames=n_frames,
            n_loops=n_loops,
            amplitude_deg=amplitude_deg,
            lead_in_deg=lead_in_deg,
            lead_out_deg=lead_out_deg,
            camera_template=camera_template
        )

        return self.helical_cameras

    def render_helical_with_canny(
        self,
        include_depth: bool = True,
        include_skeleton: bool = True,
        depth_colormap: Optional[str] = None,
        skeleton_opts: Optional[Dict[str, Any]] = None
    ) -> List[NDArray[np.uint8]]:
        """
        Render helical orbit views with projected Canny overlay.

        Args:
            include_depth: Include depth as base layer
            include_skeleton: Include skeleton overlay
            depth_colormap: Colormap for depth (None = grayscale)
            skeleton_opts: Skeleton rendering options

        Returns:
            List of composite RGBA images, one per helical camera

        Raises:
            RuntimeError: If canny_atlas or helical_cameras not set
        """
        if self.canny_atlas is None or self.uv_coords is None:
            raise RuntimeError(
                "Canny atlas not generated. Call generate_canny_atlas() first."
            )

        if self.helical_cameras is None:
            raise RuntimeError(
                "Helical cameras not set. Call set_helical_orbit() first."
            )

        renderer = self._get_renderer()
        images = []

        for camera in self.helical_cameras:
            # Build composite modes
            modes: Dict[str, Any] = {}

            # Base layer: depth
            if include_depth:
                modes["depth"] = {
                    "normalize": True,
                    "colormap": depth_colormap
                }
            else:
                # Need some base layer - use mesh with neutral color
                modes["mesh"] = {
                    "color": (0.5, 0.5, 0.5),
                    "bg_color": (0.0, 0.0, 0.0)
                }

            # Textured overlay: projected Canny
            modes["textured"] = {
                "texture": self.canny_atlas,
                "uv_coords": self.uv_coords
            }

            # Skeleton overlay
            if include_skeleton and self.scene.skeleton_joints is not None:
                skel_defaults = {
                    "joint_radius": 0.015,
                    "bone_radius": 0.008,
                    "use_openpose_colors": True
                }
                if skeleton_opts:
                    skel_defaults.update(skeleton_opts)
                modes["skeleton"] = skel_defaults

            # Render composite
            image = renderer.render_composite(camera, modes)
            images.append(image)

        return images

    def export_colmap(
        self,
        output_dir: str,
        n_pointcloud_samples: int = 50000,
        filename_pattern: str = "frame_{:04d}.png"
    ) -> Path:
        """
        Export COLMAP format files for helical orbit.

        Args:
            output_dir: Directory for output files
            n_pointcloud_samples: Points to sample from mesh
            filename_pattern: Image filename pattern

        Returns:
            Path to output directory
        """
        if self.helical_cameras is None:
            raise RuntimeError("Helical cameras not set.")

        image_names = ImageExporter.generate_filenames(
            n_frames=len(self.helical_cameras),
            pattern=filename_pattern
        )

        exporter = ColmapExporter.from_scene_and_cameras(
            scene=self.scene,
            cameras=self.helical_cameras,
            image_names=image_names,
            n_pointcloud_samples=n_pointcloud_samples
        )

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
        Export rendered images.

        Args:
            output_dir: Directory for output images
            images: List of images to export
            filename_pattern: Filename pattern

        Returns:
            List of saved file paths
        """
        filenames = ImageExporter.generate_filenames(
            n_frames=len(images),
            pattern=filename_pattern
        )

        exporter = ImageExporter(images, filenames)
        return exporter.export(Path(output_dir))

    def export_canny_atlas(
        self,
        output_path: str
    ) -> Path:
        """
        Export the generated Canny atlas as an image.

        Args:
            output_path: Path for output image (e.g., "canny_atlas.png")

        Returns:
            Path to saved atlas
        """
        if self.canny_atlas is None:
            raise RuntimeError("Canny atlas not generated.")

        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow required for image export")

        img = Image.fromarray(self.canny_atlas)
        path = Path(output_path)
        img.save(path)

        return path

    def get_cameras(self) -> List[Camera]:
        """Get the helical orbit cameras (for COLMAP export)."""
        if self.helical_cameras is None:
            raise RuntimeError("Helical cameras not set.")
        return self.helical_cameras

    def __repr__(self) -> str:
        """String representation."""
        n_circ = len(self.circular_cameras) if self.circular_cameras else 0
        n_heli = len(self.helical_cameras) if self.helical_cameras else 0
        has_atlas = self.canny_atlas is not None

        return (
            f"ProjectionPipeline("
            f"scene={self.scene}, "
            f"circular_cams={n_circ}, "
            f"helical_cams={n_heli}, "
            f"has_atlas={has_atlas})"
        )
