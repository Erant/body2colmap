"""
Scene management for 3D mesh, skeleton, and lighting.

This module provides the Scene class which encapsulates the 3D content to be
rendered: mesh geometry, optional skeleton, and lighting configuration.

The Scene class handles loading from SAM-3D-Body output and provides utilities
for querying scene properties (bounds, centroid, point cloud sampling).
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from numpy.typing import NDArray

from . import coordinates


class Scene:
    """
    3D scene containing mesh, optional skeleton, and lighting.

    All geometry is stored in world coordinates (see coordinates.py).

    The Scene is immutable after creation - geometry doesn't move.
    Cameras orbit around the scene.
    """

    def __init__(
        self,
        vertices: NDArray[np.float32],
        faces: NDArray[np.int32],
        skeleton_joints: Optional[NDArray[np.float32]] = None,
        skeleton_format: Optional[str] = None
    ):
        """
        Initialize a Scene.

        Args:
            vertices: Mesh vertex positions in world coords, shape (N, 3)
            faces: Mesh face indices (triangles), shape (M, 3)
            skeleton_joints: Optional 3D joint positions in world coords, shape (J, 3)
            skeleton_format: Skeleton format identifier ("mhr70", "coco", etc.)
        """
        self.vertices = vertices
        self.faces = faces
        self.skeleton_joints = skeleton_joints
        self.skeleton_format = skeleton_format

        # Will be lazily initialized
        self._mesh_trimesh = None
        self._bounds = None
        self._centroid = None

    @classmethod
    def from_sam3d_output(
        cls,
        output_dict: Dict[str, Any],
        include_skeleton: bool = False
    ) -> "Scene":
        """
        Create Scene from SAM-3D-Body output.

        This is the FIRST COORDINATE CONVERSION POINT.
        SAM-3D-Body coordinates are converted to world coordinates here.

        Args:
            output_dict: Dictionary loaded from .npz file with keys:
                - 'pred_vertices': (10475, 3) mesh vertices
                - 'pred_cam_t': (3,) camera translation
                - 'faces': (20908, 3) mesh face indices
                - 'pred_keypoints_3d': (optional) (N, 3) skeleton joints
            include_skeleton: Whether to load skeleton joints

        Returns:
            Scene instance with geometry in world coordinates

        Raises:
            ValueError: If required fields are missing
        """
        # Validate required fields
        required_fields = ['pred_vertices', 'pred_cam_t', 'faces']
        for field in required_fields:
            if field not in output_dict:
                raise ValueError(f"Missing required field: {field}")

        # Extract data
        vertices_sam3d = output_dict['pred_vertices']
        cam_t = output_dict['pred_cam_t']
        faces = output_dict['faces']

        # COORDINATE CONVERSION: SAM-3D-Body → World
        vertices_world = coordinates.sam3d_to_world(vertices_sam3d, cam_t)

        # Optional skeleton
        skeleton_joints = None
        skeleton_format = None
        if include_skeleton and 'pred_keypoints_3d' in output_dict:
            joints_sam3d = output_dict['pred_keypoints_3d']
            # Skeleton joints need same translation as mesh
            skeleton_joints = coordinates.sam3d_to_world(joints_sam3d, cam_t)
            # Infer format from number of joints
            skeleton_format = cls._infer_skeleton_format(len(skeleton_joints))

        return cls(
            vertices=vertices_world,
            faces=faces,
            skeleton_joints=skeleton_joints,
            skeleton_format=skeleton_format
        )

    @classmethod
    def from_npz_file(cls, filepath: str, include_skeleton: bool = False) -> "Scene":
        """
        Load Scene from .npz file.

        Args:
            filepath: Path to .npz file from SAM-3D-Body
            include_skeleton: Whether to load skeleton joints

        Returns:
            Scene instance
        """
        data = np.load(filepath, allow_pickle=True)
        output_dict = {key: data[key] for key in data.files}
        return cls.from_sam3d_output(output_dict, include_skeleton)

    @staticmethod
    def _infer_skeleton_format(n_joints: int) -> str:
        """
        Infer skeleton format from number of joints.

        Args:
            n_joints: Number of skeleton joints

        Returns:
            Format identifier string
        """
        format_map = {
            70: "mhr70",
            17: "coco",
            25: "openpose_body25",
            65: "openpose_body25_hands"
        }
        return format_map.get(n_joints, f"unknown_{n_joints}")

    def get_trimesh(self):
        """
        Get trimesh.Trimesh object for mesh operations.

        Lazily creates and caches the trimesh object.

        Returns:
            trimesh.Trimesh instance

        Note:
            Requires trimesh library to be installed.
        """
        if self._mesh_trimesh is None:
            try:
                import trimesh
            except ImportError:
                raise ImportError(
                    "trimesh is required for mesh operations. "
                    "Install with: pip install trimesh"
                )

            self._mesh_trimesh = trimesh.Trimesh(
                vertices=self.vertices,
                faces=self.faces,
                process=False  # Don't modify the mesh
            )

        return self._mesh_trimesh

    def get_bounds(self) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Get axis-aligned bounding box of the mesh.

        Returns:
            min_corner: Minimum (x, y, z) coordinates
            max_corner: Maximum (x, y, z) coordinates
        """
        if self._bounds is None:
            min_corner = np.min(self.vertices, axis=0)
            max_corner = np.max(self.vertices, axis=0)
            self._bounds = (min_corner, max_corner)

        return self._bounds

    def get_centroid(self) -> NDArray[np.float32]:
        """
        Get centroid of the mesh (mean of all vertices).

        Note: This can be biased by vertex density. For camera framing,
        consider using get_bbox_center() instead.

        Returns:
            Centroid position in world coordinates
        """
        if self._centroid is None:
            self._centroid = np.mean(self.vertices, axis=0)

        return self._centroid

    def get_bbox_center(self) -> NDArray[np.float32]:
        """
        Get center of the axis-aligned bounding box.

        This is the geometric center of the bounds, unaffected by vertex density.
        Preferred for camera look-at targets.

        Returns:
            Bounding box center position in world coordinates
        """
        min_corner, max_corner = self.get_bounds()
        return (min_corner + max_corner) / 2.0

    def get_bounding_sphere_radius(self) -> float:
        """
        Get radius of bounding sphere centered at centroid.

        Useful for computing camera orbit distances.

        Returns:
            Radius of smallest sphere containing all vertices,
            centered at mesh centroid
        """
        centroid = self.get_centroid()
        distances = np.linalg.norm(self.vertices - centroid, axis=1)
        return float(np.max(distances))

    def get_point_cloud(
        self,
        n_samples: int = 50000,
        color: Tuple[int, int, int] = (128, 128, 128)
    ) -> Tuple[NDArray[np.float32], NDArray[np.uint8]]:
        """
        Sample point cloud from mesh surface.

        Used for generating initial point cloud for COLMAP/3DGS.

        Args:
            n_samples: Number of points to sample
            color: RGB color (0-255) for all points

        Returns:
            points: Sampled point positions, shape (n_samples, 3)
            colors: RGB colors for each point, shape (n_samples, 3)

        Note:
            Uses trimesh.sample_surface_even for uniform sampling.
        """
        mesh = self.get_trimesh()

        # Sample points uniformly on surface
        points, face_indices = mesh.sample(n_samples, return_index=True)

        # Create color array
        colors = np.full((n_samples, 3), color, dtype=np.uint8)

        return points, colors

    def get_skeleton_bones(self) -> Optional[NDArray[np.int32]]:
        """
        Get skeleton bone connectivity.

        Returns:
            Array of bone indices (start_joint, end_joint), shape (B, 2)
            None if skeleton format is unknown

        Note:
            Bone connectivity depends on skeleton format.
            This should be implemented based on the specific skeleton topology.
        """
        if self.skeleton_joints is None:
            return None

        # TODO: Implement bone connectivity for each skeleton format
        # For now, return None
        # In full implementation, would have:
        # - MHR70_BONES = [(0, 1), (1, 2), ...]
        # - COCO_BONES = [(0, 1), (1, 2), ...]
        # etc.
        return None

    def __repr__(self) -> str:
        """String representation for debugging."""
        skeleton_info = ""
        if self.skeleton_joints is not None:
            skeleton_info = f", skeleton={self.skeleton_format}"

        return (
            f"Scene(vertices={len(self.vertices)}, "
            f"faces={len(self.faces)}{skeleton_info})"
        )
