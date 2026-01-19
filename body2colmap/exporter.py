"""
Export to COLMAP format and image files.

This module provides exporters for:
- COLMAP sparse reconstruction format (cameras.txt, images.txt, points3D.txt)
- Image sequences (PNG, JPG)

This is the SECOND COORDINATE CONVERSION POINT.
World coordinates → COLMAP/OpenCV coordinates happens here.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from numpy.typing import NDArray

from .camera import Camera
from .scene import Scene


class ColmapExporter:
    """
    Export camera parameters and point cloud to COLMAP text format.

    COLMAP format consists of three files:
    - cameras.txt: Camera intrinsic parameters
    - images.txt: Per-image extrinsic parameters
    - points3D.txt: 3D point cloud (optional)

    COLMAP uses OpenCV camera convention (Y-down, Z-forward).
    This exporter handles conversion from world coordinates (Y-up, Z-out).
    """

    def __init__(
        self,
        cameras: List[Camera],
        image_names: List[str],
        points_3d: Optional[Tuple[NDArray[np.float32], NDArray[np.uint8]]] = None
    ):
        """
        Initialize COLMAP exporter.

        Args:
            cameras: List of Camera objects to export
            image_names: List of image filenames (must match camera list length)
            points_3d: Optional tuple of (points, colors)
                      points: (N, 3) array of 3D positions
                      colors: (N, 3) array of RGB colors (0-255)
        """
        if len(cameras) != len(image_names):
            raise ValueError(
                f"Number of cameras ({len(cameras)}) must match "
                f"number of image names ({len(image_names)})"
            )

        self.cameras = cameras
        self.image_names = image_names
        self.points_3d = points_3d

    def export(self, output_dir: Path) -> None:
        """
        Export COLMAP files to directory.

        Creates:
        - output_dir/cameras.txt
        - output_dir/images.txt
        - output_dir/points3D.txt (if points_3d provided)

        Args:
            output_dir: Directory to write files to (will be created if needed)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._export_cameras(output_dir / "cameras.txt")
        self._export_images(output_dir / "images.txt")

        if self.points_3d is not None:
            self._export_points3d(output_dir / "points3D.txt")

    def _export_cameras(self, filepath: Path) -> None:
        """
        Export cameras.txt file.

        Format (one line per camera):
        CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]

        For PINHOLE model: PARAMS = fx fy cx cy

        Note:
            We assume all cameras share the same intrinsics (single camera ID).
            This is typical for orbit rendering.
        """
        # Use intrinsics from first camera (assume all same)
        cam = self.cameras[0]

        with open(filepath, 'w') as f:
            # Write header
            f.write("# Camera list with one line of data per camera:\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write("# Number of cameras: 1\n")

            # Write camera
            model, params = cam.get_colmap_intrinsics()
            params_str = " ".join(f"{p:.6f}" for p in params)
            f.write(f"1 {model} {cam.width} {cam.height} {params_str}\n")

    def _export_images(self, filepath: Path) -> None:
        """
        Export images.txt file.

        Format (two lines per image):
        IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        POINTS2D[] as (X, Y, POINT3D_ID)

        The second line lists 2D points; we leave it empty.

        Note:
            QW QX QY QZ = quaternion (w, x, y, z) in COLMAP/OpenCV convention
            TX TY TZ = translation in COLMAP/OpenCV convention
            These are WORLD-TO-CAMERA transform.
        """
        with open(filepath, 'w') as f:
            # Write header
            f.write("# Image list with two lines of data per image:\n")
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f.write(f"# Number of images: {len(self.cameras)}\n")

            # Write each image
            for image_id, (camera, name) in enumerate(zip(self.cameras, self.image_names), start=1):
                # Get extrinsics in COLMAP format (OpenCV convention)
                quat_wxyz, t_w2c = camera.get_colmap_extrinsics()

                # Write image line
                f.write(
                    f"{image_id} "
                    f"{quat_wxyz[0]:.8f} {quat_wxyz[1]:.8f} {quat_wxyz[2]:.8f} {quat_wxyz[3]:.8f} "
                    f"{t_w2c[0]:.8f} {t_w2c[1]:.8f} {t_w2c[2]:.8f} "
                    f"1 {name}\n"
                )

                # Write empty points2D line
                f.write("\n")

    def _export_points3d(self, filepath: Path) -> None:
        """
        Export points3D.txt file.

        Format (one line per point):
        POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID, POINT2D_IDX)

        We export points with:
        - ERROR = 0.0 (unknown)
        - TRACK = empty (no observations)

        Note:
            Point positions are in world coordinates.
            COLMAP expects points in the same coordinate system as camera poses,
            so we don't need conversion here - the coordinate system is already
            consistent (cameras are in COLMAP coords via get_colmap_extrinsics).

            Actually WAIT - this is subtle. The camera extrinsics are in COLMAP
            convention, but that's just the transformation. The actual world
            coordinate system is still our world coords (Y-up, Z-out).

            COLMAP doesn't care what the world coordinate system is - as long as
            cameras and points are in the SAME world system. Since camera poses
            are expressed as transforms FROM world TO camera, and we converted
            those transforms to COLMAP convention, the world points should stay
            in world coords.

            Actually no - the camera pose IS in world coords. The quaternion and
            translation define how to transform world points to camera frame.
            The world points need to be in the same world system.

            Let me think... Our world coords are Y-up, Z-out. COLMAP cameras
            are in OpenCV convention (Y-down, Z-forward). But the camera POSE
            describes where the camera is in world space. The world space can
            be anything - COLMAP doesn't dictate it.

            So: we can keep points in our world coords, and the camera poses
            (which we converted to w2c in OpenCV convention) will correctly
            transform them.

            Actually, I need to reconsider. Let me check the coordinate conversion...

            In camera.get_colmap_extrinsics():
            - We apply opengl_to_opencv rotation to R_c2w.T
            - This gives R_w2c in OpenCV camera convention
            - But the WORLD is still in our original world coords

            The rotation R_w2c rotates world points to camera frame.
            If world is Y-up/Z-out and camera is Y-down/Z-forward (OpenCV),
            then R_w2c properly handles that conversion.

            So yes, points should stay in world coordinates (Y-up, Z-out).
            The camera w2c rotation handles converting them to camera frame.

            TL;DR: No conversion needed for points - they stay in world coords.
        """
        if self.points_3d is None:
            return

        points, colors = self.points_3d

        with open(filepath, 'w') as f:
            # Write header
            f.write("# 3D point list with one line of data per point:\n")
            f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write(f"# Number of points: {len(points)}\n")

            # Write each point
            for point_id, (pos, color) in enumerate(zip(points, colors), start=1):
                f.write(
                    f"{point_id} "
                    f"{pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f} "
                    f"{color[0]} {color[1]} {color[2]} "
                    f"0.0\n"  # ERROR=0.0, empty TRACK
                )

    @classmethod
    def from_scene_and_cameras(
        cls,
        scene: Scene,
        cameras: List[Camera],
        image_names: List[str],
        n_pointcloud_samples: int = 50000
    ) -> "ColmapExporter":
        """
        Create exporter from Scene and cameras.

        Automatically samples point cloud from scene mesh.

        Args:
            scene: Scene to export point cloud from
            cameras: List of cameras
            image_names: List of image filenames
            n_pointcloud_samples: Number of points to sample from mesh

        Returns:
            ColmapExporter instance
        """
        # Sample point cloud from mesh
        points, colors = scene.get_point_cloud(n_samples=n_pointcloud_samples)

        return cls(
            cameras=cameras,
            image_names=image_names,
            points_3d=(points, colors)
        )


class ImageExporter:
    """
    Export rendered images to files.

    Supports PNG and JPG formats with optional alpha channel.
    """

    def __init__(
        self,
        images: List[NDArray[np.uint8]],
        filenames: List[str]
    ):
        """
        Initialize image exporter.

        Args:
            images: List of images (H, W, C) where C is 3 (RGB) or 4 (RGBA)
            filenames: List of filenames to save as
        """
        if len(images) != len(filenames):
            raise ValueError(
                f"Number of images ({len(images)}) must match "
                f"number of filenames ({len(filenames)})"
            )

        self.images = images
        self.filenames = filenames

    def export(self, output_dir: Path) -> List[Path]:
        """
        Export images to directory.

        Args:
            output_dir: Directory to save images to (created if needed)

        Returns:
            List of paths to saved images
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for image, filename in zip(self.images, self.filenames):
            filepath = output_dir / filename
            self._save_image(image, filepath)
            saved_paths.append(filepath)

        return saved_paths

    def _save_image(self, image: NDArray[np.uint8], filepath: Path) -> None:
        """
        Save single image to file.

        Args:
            image: Image array (H, W, C)
            filepath: Path to save to

        Note:
            Uses OpenCV for saving, which expects BGR(A) order.
            We need to convert from RGB(A).
        """
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "opencv-python is required for image export. "
                "Install with: pip install opencv-python"
            )

        # Convert RGB(A) to BGR(A) for OpenCV
        if image.shape[2] == 4:  # RGBA
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        elif image.shape[2] == 3:  # RGB
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"Unexpected number of channels: {image.shape[2]}")

        # Save
        cv2.imwrite(str(filepath), image_bgr)

    @staticmethod
    def generate_filenames(
        n_frames: int,
        pattern: str = "frame_{:04d}.png",
        start_index: int = 1
    ) -> List[str]:
        """
        Generate sequential filenames.

        Args:
            n_frames: Number of filenames to generate
            pattern: Format string with one {} placeholder for frame number
            start_index: Starting frame number (default 1 for 1-based indexing)

        Returns:
            List of filenames

        Example:
            generate_filenames(3, "frame_{:04d}.png", 1)
            → ["frame_0001.png", "frame_0002.png", "frame_0003.png"]
        """
        return [pattern.format(i) for i in range(start_index, start_index + n_frames)]
