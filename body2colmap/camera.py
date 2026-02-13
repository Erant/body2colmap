"""
Camera class with intrinsics and extrinsics.

This module provides a Camera class that encapsulates both camera intrinsic
parameters (focal length, principal point, image size) and extrinsic parameters
(position and orientation in world space).

The Camera class handles conversions between different representations (c2w,
w2c, quaternions) and provides utilities for common operations.
"""

import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray

from . import coordinates


class Camera:
    """
    Camera with both intrinsic and extrinsic parameters.

    Intrinsics define the camera's internal properties:
    - focal_length: (fx, fy) in pixels
    - principal_point: (cx, cy) in pixels (image center)
    - image_size: (width, height) in pixels

    Extrinsics define the camera's pose in world coordinates:
    - position: 3D point in world space
    - rotation: 3x3 matrix (camera-to-world) or stored as quaternion

    All extrinsics are stored and manipulated in world coordinates
    (see coordinates.py for coordinate system definition).
    """

    def __init__(
        self,
        focal_length: Tuple[float, float],
        image_size: Tuple[int, int],
        principal_point: Optional[Tuple[float, float]] = None,
        position: Optional[NDArray[np.float32]] = None,
        rotation: Optional[NDArray[np.float32]] = None
    ):
        """
        Initialize a Camera.

        Args:
            focal_length: (fx, fy) in pixels
            image_size: (width, height) in pixels
            principal_point: (cx, cy) in pixels
                            If None, defaults to image center
            position: Camera position in world coords, shape (3,)
                     If None, defaults to origin [0, 0, 0]
            rotation: Camera-to-world rotation matrix, shape (3, 3)
                     Columns are camera's local axes in world coords
                     If None, defaults to identity (camera aligned with world axes)
        """
        self.fx, self.fy = focal_length
        self.width, self.height = image_size

        if principal_point is None:
            self.cx = self.width / 2.0
            self.cy = self.height / 2.0
        else:
            self.cx, self.cy = principal_point

        if position is None:
            self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        else:
            self.position = np.array(position, dtype=np.float32)

        if rotation is None:
            self.rotation = np.eye(3, dtype=np.float32)
        else:
            self.rotation = np.array(rotation, dtype=np.float32)

    @classmethod
    def from_fov(
        cls,
        fov_deg: float,
        image_size: Tuple[int, int],
        position: Optional[NDArray[np.float32]] = None,
        rotation: Optional[NDArray[np.float32]] = None,
        is_horizontal_fov: bool = True
    ) -> "Camera":
        """
        Create camera from field of view angle.

        Args:
            fov_deg: Field of view in degrees
            image_size: (width, height) in pixels
            position: Camera position in world coords
            rotation: Camera-to-world rotation matrix
            is_horizontal_fov: If True, fov_deg is horizontal FOV
                              If False, fov_deg is vertical FOV

        Returns:
            Camera instance
        """
        width, height = image_size

        if is_horizontal_fov:
            fx = (width / 2.0) / np.tan(np.radians(fov_deg / 2.0))
            fy = fx  # Assume square pixels
        else:
            fy = (height / 2.0) / np.tan(np.radians(fov_deg / 2.0))
            fx = fy

        return cls(
            focal_length=(fx, fy),
            image_size=image_size,
            position=position,
            rotation=rotation
        )

    def look_at(
        self,
        target: NDArray[np.float32],
        up: Optional[NDArray[np.float32]] = None
    ) -> None:
        """
        Orient camera to look at target point.

        Updates the camera's rotation matrix so it looks at the target.
        Position remains unchanged.

        Args:
            target: 3D point to look at in world coords
            up: Up direction hint in world coords
                Default: [0, 1, 0] (Y-up)
        """
        c2w_matrix = coordinates.look_at_matrix(self.position, target, up)
        self.rotation = c2w_matrix[:3, :3]

    def get_c2w(self) -> NDArray[np.float32]:
        """
        Get camera-to-world transformation matrix.

        Returns:
            4x4 matrix that transforms points from camera space to world space
            - Upper-left 3x3: rotation (c2w)
            - Upper-right 3x1: translation (camera position)
            - Bottom row: [0, 0, 0, 1]
        """
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = self.rotation
        c2w[:3, 3] = self.position
        return c2w

    def get_w2c(self) -> NDArray[np.float32]:
        """
        Get world-to-camera transformation matrix.

        Returns:
            4x4 matrix that transforms points from world space to camera space
            w2c = inverse(c2w)
        """
        R_w2c = self.rotation.T
        t_w2c = -R_w2c @ self.position

        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R_w2c
        w2c[:3, 3] = t_w2c
        return w2c

    def get_intrinsics_matrix(self) -> NDArray[np.float32]:
        """
        Get camera intrinsics as 3x3 matrix.

        Returns:
            Intrinsic matrix K:
            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
        """
        K = np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        return K

    def get_colmap_intrinsics(self) -> Tuple[str, Tuple[float, ...]]:
        """
        Get intrinsics in COLMAP format.

        Returns:
            model: Camera model name (e.g., "PINHOLE")
            params: Tuple of parameters (fx, fy, cx, cy)
        """
        return "PINHOLE", (self.fx, self.fy, self.cx, self.cy)

    def get_colmap_extrinsics(self) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Get extrinsics in COLMAP format (OpenCV convention).

        COLMAP stores world-to-camera transformation:
        - Quaternion (w, x, y, z): rotates world points to camera frame
        - Translation: -R_w2c @ position

        Returns:
            quat_wxyz: Quaternion in (w, x, y, z) order
            t_w2c: Translation vector
        """
        return coordinates.world_to_colmap_camera(self.rotation, self.position)

    def project(self, points_3d: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Project 3D points in world coordinates to 2D image coordinates.

        Uses OpenGL camera convention (Y-up, camera looks -Z) and converts
        to OpenCV for projection (Y-down, camera looks +Z), matching how
        pyrender renders images.

        Args:
            points_3d: 3D points in world coords, shape (N, 3)

        Returns:
            2D points in image coordinates, shape (N, 2)
            Points may be outside image bounds.
        """
        # Transform to OpenGL camera coordinates
        w2c = self.get_w2c()
        points_cam = (w2c[:3, :3] @ points_3d.T).T + w2c[:3, 3]

        # Convert OpenGL camera space (Y-up, -Z forward) to OpenCV (Y-down, +Z forward)
        # This ensures projection matches pyrender's rendered output.
        points_cv = points_cam * np.array([1.0, -1.0, -1.0], dtype=np.float32)

        # Standard pinhole projection (Z > 0 for visible points in OpenCV)
        points_normalized = points_cv[:, :2] / points_cv[:, 2:3]

        # Apply intrinsics
        K = self.get_intrinsics_matrix()
        points_2d_homogeneous = (K @ np.column_stack([
            points_normalized,
            np.ones(len(points_normalized))
        ]).T).T

        return points_2d_homogeneous[:, :2]

    def get_forward_vector(self) -> NDArray[np.float32]:
        """
        Get camera's forward direction in world coordinates.

        In camera local space, camera looks down -Z.
        In world space, this is the third column of c2w matrix, negated.

        Returns:
            Forward direction vector (unit length)
        """
        return -self.rotation[:, 2]

    def get_up_vector(self) -> NDArray[np.float32]:
        """
        Get camera's up direction in world coordinates.

        In camera local space, up is +Y.
        In world space, this is the second column of c2w matrix.

        Returns:
            Up direction vector (unit length)
        """
        return self.rotation[:, 1]

    def get_right_vector(self) -> NDArray[np.float32]:
        """
        Get camera's right direction in world coordinates.

        In camera local space, right is +X.
        In world space, this is the first column of c2w matrix.

        Returns:
            Right direction vector (unit length)
        """
        return self.rotation[:, 0]

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"Camera(pos={self.position}, "
            f"focal=({self.fx:.1f}, {self.fy:.1f}), "
            f"size=({self.width}, {self.height}))"
        )
