"""
Scene management for 3D mesh, skeleton, and lighting.

This module provides the Scene class which encapsulates the 3D content to be
rendered: mesh geometry, optional skeleton, and lighting configuration.

The Scene class handles loading from SAM-3D-Body output and provides utilities
for querying scene properties (bounds, centroid, point cloud sampling).
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .camera import Camera

from . import coordinates


class Scene:
    """
    3D scene containing mesh, optional skeleton, and lighting.

    All geometry is stored in world coordinates (see coordinates.py).

    Cameras orbit around the scene. The scene may be rotated once
    during setup (e.g. to face the camera) but is otherwise static.
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
    def load_npz_metadata(filepath: str) -> Dict[str, Any]:
        """
        Load optional metadata fields from SAM-3D-Body .npz file.

        Returns a dictionary with whatever optional fields are present:
        focal_length, bbox, pred_keypoints_2d, global_rot, img_shape, etc.

        Args:
            filepath: Path to .npz file

        Returns:
            Dictionary of optional metadata (may be empty)
        """
        data = np.load(filepath, allow_pickle=True)
        metadata = {}
        optional_fields = [
            'focal_length', 'bbox', 'pred_keypoints_2d',
            'global_rot', 'body_pose_params', 'shape_params',
            'img_shape', 'original_img_shape',
        ]
        for field in optional_fields:
            if field in data.files:
                val = data[field]
                # Convert 0-d arrays to scalars
                if isinstance(val, np.ndarray) and val.ndim == 0:
                    val = val.item()
                metadata[field] = val

        # Also report all keys present for debugging
        metadata['_all_keys'] = list(data.files)
        return metadata

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

    def compute_torso_facing_direction(self) -> Optional[NDArray[np.float32]]:
        """
        Compute the direction the body's torso faces, projected to the XZ plane.

        Uses the average of the shoulder and hip left-to-right vectors, then
        takes the perpendicular in the XZ plane to get the forward direction.

        Returns:
            Unit vector in XZ plane pointing the direction the body faces,
            or None if skeleton data is unavailable.
        """
        if self.skeleton_joints is None or self.skeleton_format is None:
            return None

        # Look up shoulder/hip indices by format
        if self.skeleton_format == "mhr70":
            # MHR70: 5=left_shoulder, 6=right_shoulder, 9=left_hip, 10=right_hip
            l_shoulder = self.skeleton_joints[5]
            r_shoulder = self.skeleton_joints[6]
            l_hip = self.skeleton_joints[9]
            r_hip = self.skeleton_joints[10]
        elif self.skeleton_format in ("openpose_body25_hands", "openpose_body25"):
            # OpenPose: 5=LShoulder, 2=RShoulder, 12=LHip, 9=RHip
            l_shoulder = self.skeleton_joints[5]
            r_shoulder = self.skeleton_joints[2]
            l_hip = self.skeleton_joints[12]
            r_hip = self.skeleton_joints[9]
        else:
            return None

        # Average "across body" vector (right → left)
        shoulder_vec = l_shoulder - r_shoulder
        hip_vec = l_hip - r_hip
        across = (shoulder_vec + hip_vec) / 2.0

        # Project to XZ plane
        across_xz = np.array([across[0], 0.0, across[2]], dtype=np.float64)
        if np.linalg.norm(across_xz) < 1e-6:
            return None

        # Forward = Y_up × across (perpendicular in XZ plane, facing outward)
        y_up = np.array([0.0, 1.0, 0.0])
        forward = np.cross(y_up, across_xz)

        norm = np.linalg.norm(forward)
        if norm < 1e-6:
            return None

        return (forward / norm).astype(np.float32)

    def rotate_around_y(self, angle_deg: float) -> None:
        """
        Rotate mesh and skeleton around the Y axis through the bbox center.

        Args:
            angle_deg: Rotation angle in degrees (positive = counterclockwise
                from above, i.e. from +Z toward +X).
        """
        if abs(angle_deg) < 1e-6:
            return

        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        R = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a],
        ], dtype=np.float64)

        center = self.get_bbox_center().astype(np.float64)

        self.vertices = ((self.vertices - center) @ R.T + center).astype(np.float32)
        if self.skeleton_joints is not None:
            self.skeleton_joints = (
                (self.skeleton_joints - center) @ R.T + center
            ).astype(np.float32)

        # Invalidate cached bounds/centroid
        self._bounds = None
        self._centroid = None
        self._mesh_trimesh = None

    def get_framing_bounds(
        self,
        preset: str = "full"
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Get bounding box for a framing preset using mesh vertex filtering.

        For partial body presets (torso, bust, head), filters mesh vertices
        by Y coordinate (height) based on skeleton joint positions, then
        computes the bounding box of the filtered vertices.

        This provides accurate visual bounds for the selected body region
        without requiring heuristic padding.

        Args:
            preset: Framing preset name:
                - "full": Entire body (uses all mesh vertices)
                - "torso": Waist up (Y >= hip level)
                - "bust": Shoulders and head (Y >= upper chest)
                - "head": Head only (Y >= neck level)

        Returns:
            Tuple of (min_corner, max_corner) arrays, each shape (3,)

        Raises:
            ValueError: If preset requires skeleton but none is loaded,
                       or if skeleton format is not MHR70
        """
        from .skeleton import get_framing_y_threshold, FRAMING_PRESETS

        # Validate preset
        if preset not in FRAMING_PRESETS:
            raise ValueError(
                f"Unknown framing preset: '{preset}'. "
                f"Valid options: {', '.join(FRAMING_PRESETS)}"
            )

        # Full body - use all mesh vertices
        if preset == "full":
            return self.get_bounds()

        # Partial framing requires skeleton data
        if self.skeleton_joints is None:
            raise ValueError(
                f"Framing preset '{preset}' requires skeleton data. "
                "Load with include_skeleton=True or use --framing full"
            )

        if self.skeleton_format != "mhr70":
            raise ValueError(
                f"Framing presets only supported for MHR70 skeleton format, "
                f"got: {self.skeleton_format}"
            )

        # Get Y threshold from skeleton
        y_threshold = get_framing_y_threshold(self.skeleton_joints, preset)

        # Filter mesh vertices by height
        mask = self.vertices[:, 1] >= y_threshold
        filtered_vertices = self.vertices[mask]

        if len(filtered_vertices) == 0:
            raise ValueError(
                f"No vertices found above Y threshold for '{preset}' framing. "
                "This may indicate a problem with the skeleton data."
            )

        # Compute bbox from filtered vertices
        min_corner = np.min(filtered_vertices, axis=0)
        max_corner = np.max(filtered_vertices, axis=0)

        return min_corner.astype(np.float32), max_corner.astype(np.float32)

    def filter_mesh_to_viewport(self, camera: "Camera") -> "Scene":
        """
        Create a new Scene with only vertices visible in camera's viewport.

        Projects all mesh vertices through the camera and keeps only those
        that fall within the image bounds. Treats the mesh as transparent
        (no occlusion culling) - vertices behind other geometry are kept
        as long as they project into the viewport.

        Faces are kept if ANY of their vertices are in the viewport, which
        preserves boundary triangles that span the viewport edge.

        Args:
            camera: Camera defining the viewport to filter against

        Returns:
            New Scene with filtered mesh. Skeleton data is preserved unchanged.

        Raises:
            ValueError: If filtering results in no vertices
        """
        # Get camera transform and intrinsics
        w2c = camera.get_w2c()
        fx, fy = camera.fx, camera.fy
        cx, cy = camera.cx, camera.cy
        width, height = camera.width, camera.height

        # Transform vertices to camera space
        # vertices_h: (N, 4) homogeneous coordinates
        vertices_h = np.hstack([self.vertices, np.ones((len(self.vertices), 1))])
        vertices_cam = (w2c @ vertices_h.T).T[:, :3]

        # In OpenGL convention, camera looks down -Z axis
        # Points in front of camera have Z < 0
        in_front = vertices_cam[:, 2] < 0

        # Project to image coordinates
        # For points with Z < 0, we use -Z for the division
        z = -vertices_cam[:, 2]
        z[z <= 0] = 1e-6  # Avoid division by zero for points behind camera

        x_proj = fx * vertices_cam[:, 0] / z + cx
        y_proj = fy * vertices_cam[:, 1] / z + cy

        # Determine which vertices are in the viewport
        in_viewport = (
            in_front &
            (x_proj >= 0) & (x_proj < width) &
            (y_proj >= 0) & (y_proj < height)
        )

        # Get indices of kept vertices
        kept_indices = np.where(in_viewport)[0]
        kept_set = set(kept_indices)

        if len(kept_indices) == 0:
            raise ValueError(
                "No vertices visible in camera viewport. "
                "Check camera position and orientation."
            )

        # Create old-to-new index mapping
        old_to_new = {old: new for new, old in enumerate(kept_indices)}

        # Filter vertices
        new_vertices = self.vertices[in_viewport].astype(np.float32)

        # Keep faces where ANY vertex is visible (preserves boundary triangles)
        new_faces = []
        for face in self.faces:
            # Check if any vertex of this face is in the kept set
            visible_verts = [v for v in face if v in kept_set]
            if len(visible_verts) == 3:
                # All vertices visible - remap and keep
                new_faces.append([old_to_new[v] for v in face])
            elif len(visible_verts) > 0:
                # Partial visibility - still keep if all vertices happen to be kept
                # (This handles edge cases at viewport boundary)
                if all(v in kept_set for v in face):
                    new_faces.append([old_to_new[v] for v in face])

        if len(new_faces) == 0:
            raise ValueError(
                "No complete faces visible in camera viewport. "
                "This may indicate the mesh is partially outside the view."
            )

        new_faces = np.array(new_faces, dtype=np.int32)

        # Return new scene with filtered mesh, preserving skeleton
        return Scene(
            vertices=new_vertices,
            faces=new_faces,
            skeleton_joints=self.skeleton_joints,
            skeleton_format=self.skeleton_format
        )

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
