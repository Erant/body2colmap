"""
High-level pipeline orchestrating all components.

This module provides the OrbitPipeline class which ties together:
- Scene loading
- Orbit path generation
- Rendering
- Export to COLMAP and images

This is the main API for users of the library.
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
            # ~47° horizontal FOV
            width = render_size[0]
            self.focal_length = width / (2.0 * np.tan(np.radians(47.0 / 2.0)))
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

    def set_orbit_params(
        self,
        pattern: str = "helical",
        n_frames: int = 120,
        radius: Optional[float] = None,
        **kwargs
    ) -> "OrbitPipeline":
        """
        Set orbit path parameters and generate cameras.

        Args:
            pattern: Orbit pattern type ("circular", "sinusoidal", "helical")
            n_frames: Number of frames to generate
            radius: Orbit radius (distance from target)
                   If None, auto-computed to frame scene
            **kwargs: Pattern-specific parameters:
                - circular: elevation_deg
                - sinusoidal: amplitude_deg, n_cycles
                - helical: n_loops, amplitude_deg, lead_in_deg, lead_out_deg

        Returns:
            self (for method chaining)
        """
        # Auto-compute radius if not provided
        if radius is None:
            bounds = self.scene.get_bounds()
            fill_ratio = kwargs.pop('fill_ratio', 0.8)

            # For proper framing, compute radius needed for each dimension separately
            # then use the max to ensure the scene fits in both dimensions
            width, height = self.render_size
            min_corner, max_corner = bounds

            # Scene extents in X (horizontal) and Y (vertical) dimensions
            scene_width = max_corner[0] - min_corner[0]  # X extent
            scene_height = max_corner[1] - min_corner[1]  # Y extent (up)

            # Compute FOVs in radians
            horizontal_fov_rad = 2 * np.arctan(width / (2 * self.focal_length))
            vertical_fov_rad = 2 * np.arctan(height / (2 * self.focal_length))

            # Radius needed to fit horizontal extent in horizontal FOV
            desired_h_angle = horizontal_fov_rad * fill_ratio
            radius_h = (scene_width / 2.0) / np.tan(desired_h_angle / 2.0)

            # Radius needed to fit vertical extent in vertical FOV
            desired_v_angle = vertical_fov_rad * fill_ratio
            radius_v = (scene_height / 2.0) / np.tan(desired_v_angle / 2.0)

            # Use the larger radius to ensure scene fits in both dimensions
            radius = max(radius_h, radius_v)

        # Create orbit path generator
        target = self.scene.get_centroid()
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
            modes: List of render modes ("mesh", "depth", "skeleton")
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

        # Create renderer if needed
        if self._renderer is None:
            self._renderer = Renderer(self.scene, self.render_size)

        results = {}

        for mode in modes:
            images = []

            # Render each frame
            for i, camera in enumerate(self.cameras):
                if mode == "mesh":
                    image = self._renderer.render_mesh(
                        camera,
                        mesh_color=render_kwargs.get('mesh_color'),
                        bg_color=render_kwargs.get('bg_color', (1.0, 1.0, 1.0))
                    )
                elif mode == "depth":
                    image = self._renderer.render_depth(
                        camera,
                        normalize=render_kwargs.get('normalize_depth', True),
                        colormap=render_kwargs.get('depth_colormap')
                    )
                elif mode == "skeleton":
                    image = self._renderer.render_skeleton(
                        camera,
                        joint_radius=render_kwargs.get('joint_radius', 0.015),
                        bone_radius=render_kwargs.get('bone_radius', 0.008)
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
        """
        if self.cameras is None:
            raise RuntimeError("Cameras not set. Call set_orbit_params() first.")

        # Create renderer if needed
        if self._renderer is None:
            self._renderer = Renderer(self.scene, self.render_size)

        images = []
        for camera in self.cameras:
            image = self._renderer.render_composite(camera, composite_modes)
            images.append(image)

        return images

    def export_colmap(
        self,
        output_dir: str,
        n_pointcloud_samples: int = 50000
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

        Returns:
            Path to output directory

        Raises:
            RuntimeError: If cameras haven't been set
        """
        if self.cameras is None:
            raise RuntimeError("Cameras not set. Call set_orbit_params() first.")

        # Generate image filenames (matches default rendering output)
        image_names = ImageExporter.generate_filenames(len(self.cameras))

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
