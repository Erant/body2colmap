"""
High-level pipeline for texture projection workflow.

This module provides the ProjectionPipeline class that orchestrates the
circular-orbit-to-helical-orbit Canny projection workflow:

**Phase 1 - Generate reference views for diffusion:**
1. Generate cameras for circular orbit (eye-level reference views)
2. Render mesh from circular orbit
3. Export images + COLMAP for diffusion model input

**External step (not handled by this tool):**
- User processes circular orbit images through diffusion model

**Phase 2 - Project diffusion output and render guidance:**
4. Load diffusion-processed images (matching COLMAP filenames)
5. Detect Canny edges on diffusion output
6. Project edges onto UV atlas
7. Generate helical orbit cameras
8. Render depth + projected-canny + skeleton composites

The output is suitable for ControlNet guidance in subsequent diffusion passes.
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

    This is a two-phase pipeline with an external diffusion step in between:

    **Phase 1 - Generate reference views:**
        pipeline = ProjectionPipeline.from_npz_file("mesh.npz")
        pipeline.setup_circular_orbit(n_frames=36, elevation_deg=0.0)
        images = pipeline.render_circular_orbit()
        pipeline.export_circular_orbit("./circular_output")
        # → User runs diffusion on ./circular_output/*.png

    **Phase 2 - Project and render guidance:**
        pipeline.load_diffusion_images("./circular_output")  # or different dir
        pipeline.generate_canny_atlas_from_images()
        pipeline.setup_helical_orbit(n_frames=120, n_loops=3)
        guidance = pipeline.render_helical_with_canny()
        pipeline.export_helical_output("./helical_output", guidance)

    The circular and helical outputs use the same COLMAP camera format,
    so diffusion models can use consistent camera conditioning.
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
        self.texture_atlas: Optional[NDArray[np.uint8]] = None

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

    # =========================================================================
    # PHASE 1: Generate circular orbit for diffusion input
    # =========================================================================

    def setup_circular_orbit(
        self,
        n_frames: int = 36,
        elevation_deg: float = 0.0,
        radius: Optional[float] = None,
        fill_ratio: float = 0.8
    ) -> List[Camera]:
        """
        Set up circular orbit cameras for Phase 1 (diffusion input).

        Args:
            n_frames: Number of views in circular orbit
            elevation_deg: Elevation angle (0 = eye level)
            radius: Orbit radius (None = auto-compute)
            fill_ratio: How much of frame to fill (for auto radius)

        Returns:
            List of Camera objects for circular orbit
        """
        if radius is None:
            radius = self._compute_orbit_radius(fill_ratio)

        self._circular_radius = radius  # Store for helical orbit

        target = self.scene.get_bbox_center()
        orbit = OrbitPath(target=target, radius=radius)
        camera_template = self._create_camera_template()

        self.circular_cameras = orbit.circular(
            n_frames=n_frames,
            elevation_deg=elevation_deg,
            camera_template=camera_template
        )

        return self.circular_cameras

    def render_circular_orbit(
        self,
        mesh_color: Optional[Tuple[float, float, float]] = None,
        bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> List[NDArray[np.uint8]]:
        """
        Render mesh from all circular orbit cameras.

        These renders are intended as input for diffusion models.

        Args:
            mesh_color: Mesh color (None = default)
            bg_color: Background color

        Returns:
            List of RGBA images, one per camera

        Raises:
            RuntimeError: If circular orbit not set up
        """
        if self.circular_cameras is None:
            raise RuntimeError("Circular orbit not set up. Call setup_circular_orbit() first.")

        renderer = self._get_renderer()
        images = []

        for camera in self.circular_cameras:
            image = renderer.render_mesh(
                camera,
                mesh_color=mesh_color,
                bg_color=bg_color
            )
            images.append(image)

        return images

    def export_circular_orbit(
        self,
        output_dir: str,
        images: List[NDArray[np.uint8]],
        filename_pattern: str = "frame_{:04d}.png",
        n_pointcloud_samples: int = 50000
    ) -> Path:
        """
        Export circular orbit images and COLMAP files.

        After this, process images through diffusion, then call
        load_diffusion_images() with the processed images.

        Args:
            output_dir: Directory for output
            images: Rendered images from render_circular_orbit()
            filename_pattern: Image filename pattern
            n_pointcloud_samples: Points for COLMAP point cloud

        Returns:
            Path to output directory
        """
        if self.circular_cameras is None:
            raise RuntimeError("Circular orbit not set up.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export images
        filenames = ImageExporter.generate_filenames(
            n_frames=len(images),
            pattern=filename_pattern
        )
        exporter = ImageExporter(images, filenames)
        exporter.export(output_path)

        # Export COLMAP
        colmap_exporter = ColmapExporter.from_scene_and_cameras(
            scene=self.scene,
            cameras=self.circular_cameras,
            image_names=filenames,
            n_pointcloud_samples=n_pointcloud_samples
        )
        colmap_exporter.export(output_path)

        # Store filename pattern for later use
        self._circular_filename_pattern = filename_pattern

        return output_path

    # =========================================================================
    # PHASE 2: Load diffusion output and generate Canny atlas
    # =========================================================================

    def load_diffusion_images(
        self,
        image_dir: str,
        filename_pattern: Optional[str] = None
    ) -> List[NDArray[np.uint8]]:
        """
        Load diffusion-processed images for Canny projection.

        Images must correspond to the circular orbit cameras (same filenames
        as exported in Phase 1).

        Args:
            image_dir: Directory containing diffusion output images
            filename_pattern: Filename pattern (default: use pattern from export)

        Returns:
            List of loaded images (RGBA)

        Raises:
            RuntimeError: If circular cameras not set
            FileNotFoundError: If expected images are missing
        """
        if self.circular_cameras is None:
            raise RuntimeError(
                "Circular cameras not set. Either call setup_circular_orbit() "
                "or load cameras from COLMAP export."
            )

        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow required for image loading")

        if filename_pattern is None:
            filename_pattern = getattr(self, '_circular_filename_pattern', 'frame_{:04d}.png')

        image_dir = Path(image_dir)
        n_frames = len(self.circular_cameras)
        filenames = ImageExporter.generate_filenames(n_frames, filename_pattern)

        images = []
        for filename in filenames:
            filepath = image_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Expected diffusion image not found: {filepath}\n"
                    f"Make sure diffusion output uses same filenames as circular orbit export."
                )

            img = Image.open(filepath)
            # Convert to RGBA if needed
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            images.append(np.array(img))

        self._diffusion_images = images
        return images

    def generate_canny_atlas_from_images(
        self,
        images: Optional[List[NDArray[np.uint8]]] = None,
        canny_low: int = 50,
        canny_high: int = 150,
        canny_blur: int = 5,
        edge_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        uv_method: str = "cylindrical",
        blend_mode: str = "max"
    ) -> NDArray[np.uint8]:
        """
        Generate Canny atlas from diffusion-processed images.

        This detects Canny edges on the provided images and projects
        them onto the mesh UV atlas.

        Args:
            images: List of images to process. If None, uses images from
                   load_diffusion_images()
            canny_low: Canny low threshold
            canny_high: Canny high threshold
            canny_blur: Gaussian blur kernel size
            edge_color: RGB color for edges in atlas
            uv_method: UV generation method ("cylindrical", "spherical")
            blend_mode: Blending mode ("max", "average")

        Returns:
            Canny atlas, shape (atlas_height, atlas_width, 4), RGBA uint8

        Raises:
            RuntimeError: If no images available
        """
        if images is None:
            images = getattr(self, '_diffusion_images', None)

        if images is None:
            raise RuntimeError(
                "No images to process. Call load_diffusion_images() first, "
                "or pass images directly."
            )

        if self.circular_cameras is None:
            raise RuntimeError("Circular cameras not set.")

        if len(images) != len(self.circular_cameras):
            raise ValueError(
                f"Number of images ({len(images)}) doesn't match "
                f"number of cameras ({len(self.circular_cameras)})"
            )

        renderer = self._get_renderer()

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

        # Process each image with its corresponding camera
        for camera, image in zip(self.circular_cameras, images):
            # Detect Canny edges on diffusion output
            edges = edge_module.canny(
                image[:, :, :3],  # Use RGB
                low_threshold=canny_low,
                high_threshold=canny_high,
                blur_kernel=canny_blur
            )

            # Convert to RGBA
            edge_rgba = edge_module.edges_to_rgba(edges, color=edge_color)

            # Render face IDs for this camera view
            face_ids = renderer.render_face_ids(camera)

            # Project onto atlas with proper UV interpolation
            projector.project_view(edge_rgba, camera, face_ids, blend_mode)

        # Get final atlas
        self.canny_atlas = projector.get_atlas()

        return self.canny_atlas

    def generate_texture_atlas_from_images(
        self,
        images: Optional[List[NDArray[np.uint8]]] = None,
        mode: str = "canny",
        canny_low: int = 50,
        canny_high: int = 150,
        canny_blur: int = 5,
        edge_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        uv_method: str = "cylindrical",
        blend_mode: str = "max"
    ) -> NDArray[np.uint8]:
        """
        Generate texture atlas from diffusion-processed images.

        Supports multiple texture modes for different ControlNet conditioning:
        - "canny": Edge detection (good for structure)
        - "color": Direct color projection (preserves texture details)
        - "both": Canny edges overlaid on color

        Args:
            images: List of images to process. If None, uses cached images.
            mode: Texture mode - "canny", "color", or "both"
            canny_low: Canny low threshold (for canny/both modes)
            canny_high: Canny high threshold (for canny/both modes)
            canny_blur: Gaussian blur kernel size (for canny/both modes)
            edge_color: RGB color for edges (for canny mode only)
            uv_method: UV generation method ("cylindrical", "spherical")
            blend_mode: Blending mode ("max", "average")

        Returns:
            Texture atlas, shape (atlas_height, atlas_width, 4), RGBA uint8
        """
        if images is None:
            images = getattr(self, '_diffusion_images', None)

        if images is None:
            raise RuntimeError(
                "No images to process. Call load_diffusion_images() first, "
                "or pass images directly."
            )

        if self.circular_cameras is None:
            raise RuntimeError("Circular cameras not set.")

        if len(images) != len(self.circular_cameras):
            raise ValueError(
                f"Number of images ({len(images)}) doesn't match "
                f"number of cameras ({len(self.circular_cameras)})"
            )

        renderer = self._get_renderer()

        # Generate UVs for the mesh (reuse if already generated)
        if self.uv_coords is None:
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

        # Process each image with its corresponding camera
        for camera, image in zip(self.circular_cameras, images):
            # Render face IDs for visibility
            face_ids = renderer.render_face_ids(camera)

            if mode == "canny":
                # Detect Canny edges
                edges = edge_module.canny(
                    image[:, :, :3],
                    low_threshold=canny_low,
                    high_threshold=canny_high,
                    blur_kernel=canny_blur
                )
                proj_image = edge_module.edges_to_rgba(edges, color=edge_color)

            elif mode == "color":
                # Use image directly (ensure RGBA)
                if image.shape[2] == 3:
                    proj_image = np.dstack([image, np.full(image.shape[:2], 255, dtype=np.uint8)])
                else:
                    proj_image = image

            elif mode == "both":
                # Color with Canny edges overlaid
                edges = edge_module.canny(
                    image[:, :, :3],
                    low_threshold=canny_low,
                    high_threshold=canny_high,
                    blur_kernel=canny_blur
                )
                # Start with color
                if image.shape[2] == 3:
                    proj_image = np.dstack([image, np.full(image.shape[:2], 255, dtype=np.uint8)])
                else:
                    proj_image = image.copy()
                # Overlay white edges
                edge_mask = edges > 0
                proj_image[edge_mask] = [255, 255, 255, 255]

            else:
                raise ValueError(f"Unknown texture mode: {mode}. Use 'canny', 'color', or 'both'.")

            # Choose blend mode based on texture type
            # - "max" for edges: preserves all edges from any view
            # - "average" for colors: blends colors from multiple views
            effective_blend_mode = blend_mode
            if blend_mode == "max" and mode == "color":
                effective_blend_mode = "average"

            # Project onto atlas with proper UV interpolation
            projector.project_view(proj_image, camera, face_ids, effective_blend_mode)

        # Get and store atlas
        atlas = projector.get_atlas()

        # Store as texture_atlas (more general than canny_atlas)
        self.texture_atlas = atlas
        # Also store as canny_atlas for backwards compatibility
        if mode in ("canny", "both"):
            self.canny_atlas = atlas

        return atlas

    def generate_vertex_colors_from_images(
        self,
        images: Optional[List[NDArray[np.uint8]]] = None,
        blend_mode: str = "best_angle"
    ) -> NDArray[np.uint8]:
        """
        Generate vertex colors from diffusion-processed images.

        This is an alternative to texture atlas that avoids UV overlap issues
        inherent in cylindrical UV mapping. Colors are stored directly on
        vertices and interpolated during rendering.

        Args:
            images: List of images to process. If None, uses cached images.
            blend_mode: Blending mode ("best_angle", "average", "replace")

        Returns:
            Vertex colors, shape (num_vertices, 4), RGBA uint8
        """
        if images is None:
            images = getattr(self, '_diffusion_images', None)

        if images is None:
            raise RuntimeError(
                "No images to process. Call load_diffusion_images() first, "
                "or pass images directly."
            )

        if self.circular_cameras is None:
            raise RuntimeError("Circular cameras not set.")

        if len(images) != len(self.circular_cameras):
            raise ValueError(
                f"Number of images ({len(images)}) doesn't match "
                f"number of cameras ({len(self.circular_cameras)})"
            )

        renderer = self._get_renderer()

        # Create vertex color projector
        projector = tex_proj.VertexColorProjector(
            self.scene.vertices,
            self.scene.faces
        )

        # Process each view using direct depth-based projection
        # This avoids face_id buffer issues from anti-aliasing
        for camera, image in zip(self.circular_cameras, images):
            # Render depth buffer for visibility testing
            depth_buffer = renderer.render_depth_raw(camera)

            # Project directly to vertices using depth visibility
            projector.project_view_direct(image, depth_buffer, camera, blend_mode)

        # Get and store vertex colors
        self.vertex_colors = projector.get_vertex_colors()

        return self.vertex_colors

    # =========================================================================
    # PHASE 2 continued: Set up helical orbit and render with Canny
    # =========================================================================

    def setup_helical_orbit(
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
        Set up helical orbit cameras for Phase 2 rendering.

        Uses same orbit radius as circular orbit by default to ensure
        consistent framing.

        Args:
            n_frames: Number of frames in helical orbit
            n_loops: Number of full rotations
            amplitude_deg: Maximum elevation deviation from center
            lead_in_deg: Azimuth range for smooth lead-in
            lead_out_deg: Azimuth range for smooth lead-out
            radius: Orbit radius (None = use circular orbit radius or auto)
            fill_ratio: Fill ratio for auto radius (if no circular radius)

        Returns:
            List of Camera objects for helical orbit
        """
        if radius is None:
            # Use circular orbit radius if available, otherwise compute
            radius = getattr(self, '_circular_radius', None)
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

    def render_helical_with_texture(
        self,
        base_mode: str = "depth",
        include_texture: bool = True,
        include_skeleton: bool = True,
        depth_colormap: Optional[str] = None,
        normal_space: str = "camera",
        skeleton_opts: Optional[Dict[str, Any]] = None,
        use_vertex_colors: bool = False
    ) -> List[NDArray[np.uint8]]:
        """
        Render helical orbit views with projected texture overlay.

        This is a more general version of render_helical_with_canny that
        supports different base modes and texture sources.

        Args:
            base_mode: Base layer type - "depth", "normals", or "mesh"
            include_texture: Include projected texture overlay
            include_skeleton: Include skeleton overlay
            depth_colormap: Colormap for depth (None = grayscale)
            normal_space: Normal map space ("camera" or "world")
            skeleton_opts: Skeleton rendering options
            use_vertex_colors: Use vertex colors instead of texture atlas
                              (avoids UV overlap issues with cylindrical UVs)

        Returns:
            List of composite RGBA images, one per helical camera

        Raises:
            RuntimeError: If texture requested but atlas not generated
        """
        # Get texture data (atlas or vertex colors)
        texture_atlas = None
        vertex_colors = None

        if use_vertex_colors:
            vertex_colors = getattr(self, 'vertex_colors', None)
            if include_texture and vertex_colors is None:
                raise RuntimeError(
                    "Vertex colors not generated. Call generate_vertex_colors_from_images() first."
                )
        else:
            # Use texture atlas (prefer texture_atlas, fall back to canny_atlas)
            texture_atlas = getattr(self, 'texture_atlas', None)
            if texture_atlas is None:
                texture_atlas = self.canny_atlas

            if include_texture and (texture_atlas is None or self.uv_coords is None):
                raise RuntimeError(
                    "Texture atlas not generated. Call generate_texture_atlas_from_images() "
                    "or generate_canny_atlas_from_images() first."
                )

        if self.helical_cameras is None:
            raise RuntimeError(
                "Helical cameras not set. Call setup_helical_orbit() first."
            )

        renderer = self._get_renderer()
        images = []

        for camera in self.helical_cameras:
            # Build composite modes
            modes: Dict[str, Any] = {}

            # Base layer
            if base_mode == "depth":
                modes["depth"] = {
                    "normalize": True,
                    "colormap": depth_colormap
                }
            elif base_mode == "normals":
                modes["normals"] = {
                    "space": normal_space
                }
            elif base_mode == "mesh":
                modes["mesh"] = {
                    "color": (0.5, 0.5, 0.5),
                    "bg_color": (0.0, 0.0, 0.0)
                }
            else:
                raise ValueError(f"Unknown base_mode: {base_mode}")

            # Textured overlay: projected texture (atlas or vertex colors)
            if include_texture:
                if use_vertex_colors:
                    modes["vertex_colors"] = {
                        "colors": vertex_colors
                    }
                else:
                    modes["textured"] = {
                        "texture": texture_atlas,
                        "uv_coords": self.uv_coords
                    }

            # Skeleton overlay
            if include_skeleton and self.scene.skeleton_joints is not None:
                skel_defaults = {
                    "joint_radius": 0.006,
                    "bone_radius": 0.003,
                    "use_openpose_colors": True
                }
                if skeleton_opts:
                    skel_defaults.update(skeleton_opts)
                modes["skeleton"] = skel_defaults

            # Render composite
            image = renderer.render_composite(camera, modes)
            images.append(image)

        return images

    def render_helical_with_canny(
        self,
        include_depth: bool = True,
        include_skeleton: bool = True,
        depth_colormap: Optional[str] = None,
        skeleton_opts: Optional[Dict[str, Any]] = None
    ) -> List[NDArray[np.uint8]]:
        """
        Render helical orbit views with projected Canny overlay.

        This is a convenience wrapper around render_helical_with_texture
        for backwards compatibility.

        Args:
            include_depth: Include depth as base layer
            include_skeleton: Include skeleton overlay
            depth_colormap: Colormap for depth (None = grayscale)
            skeleton_opts: Skeleton rendering options

        Returns:
            List of composite RGBA images, one per helical camera
        """
        return self.render_helical_with_texture(
            base_mode="depth" if include_depth else "mesh",
            include_texture=True,
            include_skeleton=include_skeleton,
            depth_colormap=depth_colormap,
            skeleton_opts=skeleton_opts
        )

    def render_helical_normals(
        self,
        include_skeleton: bool = False,
        normal_space: str = "camera",
        skeleton_opts: Optional[Dict[str, Any]] = None
    ) -> List[NDArray[np.uint8]]:
        """
        Render helical orbit normal maps.

        Normal maps encode surface orientation and are useful for
        ControlNet-Normal conditioning.

        Args:
            include_skeleton: Include skeleton overlay
            normal_space: "camera" (recommended) or "world"
            skeleton_opts: Skeleton rendering options

        Returns:
            List of normal map RGBA images, one per helical camera
        """
        if self.helical_cameras is None:
            raise RuntimeError("Helical cameras not set.")

        renderer = self._get_renderer()
        images = []

        for camera in self.helical_cameras:
            modes: Dict[str, Any] = {
                "normals": {"space": normal_space}
            }

            if include_skeleton and self.scene.skeleton_joints is not None:
                skel_defaults = {
                    "joint_radius": 0.006,
                    "bone_radius": 0.003,
                    "use_openpose_colors": True
                }
                if skeleton_opts:
                    skel_defaults.update(skeleton_opts)
                modes["skeleton"] = skel_defaults

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

    def export_helical_output(
        self,
        output_dir: str,
        images: List[NDArray[np.uint8]],
        filename_pattern: str = "frame_{:04d}.png",
        n_pointcloud_samples: int = 50000
    ) -> Path:
        """
        Export helical orbit images and COLMAP files together.

        Convenience method that calls both export_images() and export_colmap().

        Args:
            output_dir: Directory for output
            images: Rendered images from render_helical_with_canny()
            filename_pattern: Image filename pattern
            n_pointcloud_samples: Points for COLMAP point cloud

        Returns:
            Path to output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.export_images(output_dir, images, filename_pattern)
        self.export_colmap(output_dir, n_pointcloud_samples, filename_pattern)

        return output_path

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
