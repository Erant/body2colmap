"""
Coordinate system definitions and conversion functions.

This module defines the canonical coordinate system used throughout body2colmap
and provides conversion functions for interacting with external systems
(SAM-3D-Body input, COLMAP output).

Canonical Coordinate System (World/Renderer coords):
    Origin: At mesh position (after applying cam_t from SAM-3D-Body)
    +X: Right
    +Y: Up (toward head)
    +Z: Out of screen (toward viewer)
    Camera: Located in world space, looks down -Z axis

Coordinate conversions happen ONLY at system boundaries:
1. Input: SAM-3D-Body → World (in scene.py)
2. Output: World → COLMAP/OpenCV (in exporter.py)
"""

import numpy as np
from typing import Tuple
from numpy.typing import NDArray


class WorldCoordinates:
    """
    Documentation of the canonical world coordinate system.

    All internal computations use this coordinate system:
    - Origin: At mesh position (mesh vertices + cam_t from SAM-3D-Body)
    - +X: Right (from camera's perspective)
    - +Y: Up (toward sky/head)
    - +Z: Out of screen (toward viewer/camera)

    Camera convention:
    - Camera position: 3D point in world space
    - Camera orientation: c2w rotation matrix
    - Camera looks: Down -Z axis in its local frame
    - Camera up: +Y axis in its local frame

    This matches OpenGL/pyrender conventions, minimizing transforms needed
    for rendering.
    """

    UP_AXIS = np.array([0.0, 1.0, 0.0])
    FORWARD_AXIS = np.array([0.0, 0.0, -1.0])  # Camera looks down -Z
    RIGHT_AXIS = np.array([1.0, 0.0, 0.0])


def sam3d_to_world(
    vertices: NDArray[np.float32],
    cam_t: NDArray[np.float32]
) -> NDArray[np.float32]:
    """
    Convert SAM-3D-Body vertices to world coordinates.

    SAM-3D-Body outputs vertices centered at origin (roughly at pelvis).
    The cam_t vector positions the mesh in front of a camera at origin.

    Args:
        vertices: Mesh vertices in SAM-3D-Body coords, shape (N, 3)
        cam_t: Camera translation vector from SAM-3D-Body, shape (3,)

    Returns:
        Vertices in world coordinates, shape (N, 3)

    Note:
        SAM-3D-Body coordinate system has the body oriented such that when
        cam_t positions it in front of a camera at the origin, the coordinate
        frame needs to be flipped to match OpenGL rendering conventions.

        We apply a 180° rotation around X axis to convert:
        - SAM-3D-Body: body facing one direction, Y-up
        - World/Renderer: body facing camera, Y-up

        This rotation:
        - Keeps X axis (left-right) unchanged
        - Flips Y axis (makes body right-side up for rendering)
        - Flips Z axis (makes body face toward -Z, where camera looks from +Z)
    """
    # First, position the mesh
    positioned_vertices = vertices + cam_t

    # Apply 180° rotation around X axis
    # This is the critical transform that was hidden in the old implementation
    R_x_180 = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0]
    ], dtype=np.float32)

    # Apply rotation: vertices @ R.T (right-multiply by transpose)
    world_vertices = positioned_vertices @ R_x_180.T

    return world_vertices


def world_to_colmap_camera(
    R_c2w: NDArray[np.float32],
    position: NDArray[np.float32]
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Convert camera pose from world (OpenGL) to COLMAP (OpenCV) convention.

    OpenGL convention (our world coords):
        Camera looks: -Z
        Camera up: +Y

    OpenCV/COLMAP convention:
        Camera looks: +Z
        Camera up: -Y (Y points down!)

    The conversion is a 180° rotation around X axis.

    Args:
        R_c2w: Camera-to-world rotation matrix, shape (3, 3)
               Columns are camera's local axes (right, up, -forward) in world coords
        position: Camera position in world coords, shape (3,)

    Returns:
        quat_wxyz: Quaternion (w, x, y, z) for COLMAP images.txt
        t_w2c: Translation vector for COLMAP images.txt

    Note:
        COLMAP stores world-to-camera transform in images.txt.
        The quaternion rotates world points into camera coordinates.
        The translation is -R_w2c @ position.
    """
    # OpenGL to OpenCV conversion matrix (180° rotation around X)
    opengl_to_opencv = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0]
    ], dtype=np.float32)

    # Convert rotation: world-to-camera in OpenCV convention
    R_w2c = opengl_to_opencv @ R_c2w.T

    # Convert translation: -R_w2c @ position
    t_w2c = -R_w2c @ position

    # Convert rotation matrix to quaternion
    quat_wxyz = rotation_to_quaternion_wxyz(R_w2c)

    return quat_wxyz, t_w2c


def rotation_to_quaternion_wxyz(R: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Convert 3x3 rotation matrix to quaternion in (w, x, y, z) order.

    This is the quaternion convention used by COLMAP.

    Args:
        R: 3x3 rotation matrix

    Returns:
        Quaternion as [w, x, y, z]

    Note:
        Many libraries (scipy, pyquaternion) use (x, y, z, w) order.
        COLMAP uses (w, x, y, z) order - be careful!
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z], dtype=np.float32)


def look_at_matrix(
    eye: NDArray[np.float32],
    target: NDArray[np.float32],
    up: NDArray[np.float32] = None
) -> NDArray[np.float32]:
    """
    Construct camera-to-world matrix for camera at 'eye' looking at 'target'.

    This builds the extrinsic matrix that positions and orients a camera.

    Args:
        eye: Camera position in world coords, shape (3,)
        target: Point camera looks at in world coords, shape (3,)
        up: Up direction hint in world coords, shape (3,)
            Default: [0, 1, 0] (Y-up)

    Returns:
        4x4 camera-to-world transform matrix
        - Upper-left 3x3: rotation (c2w)
        - Upper-right 3x1: translation (camera position)
        - Bottom row: [0, 0, 0, 1]

    Note:
        The camera will be positioned at 'eye', looking toward 'target',
        with +Y roughly aligned with 'up' vector.

        Camera local axes:
        - forward = (target - eye) normalized
        - right = cross(forward, up) normalized
        - actual_up = cross(right, forward)

        In camera local frame, camera looks down -Z (so forward maps to -Z).
    """
    if up is None:
        up = WorldCoordinates.UP_AXIS

    # Compute camera's forward direction (looking from eye toward target)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    # Compute camera's right direction (perpendicular to forward and up)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    # Recompute actual up (perpendicular to right and forward)
    actual_up = np.cross(right, forward)

    # Build rotation matrix: columns are camera's local axes in world coords
    # Camera -Z is forward direction (OpenGL convention)
    R_c2w = np.column_stack([right, actual_up, -forward])

    # Build full 4x4 matrix
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = eye

    return c2w


def cartesian_to_spherical(
    position: NDArray[np.float32]
) -> Tuple[float, float, float]:
    """
    Convert Cartesian coordinates to spherical (Y-up convention).

    Inverse of spherical_to_cartesian().

    Args:
        position: Cartesian coordinates [x, y, z]

    Returns:
        Tuple of (radius, azimuth_deg, elevation_deg) where:
            radius: Distance from origin
            azimuth_deg: Angle in XZ plane from +Z axis (degrees)
            elevation_deg: Angle above XZ plane (degrees)
    """
    x, y, z = float(position[0]), float(position[1]), float(position[2])
    radius = float(np.sqrt(x * x + y * y + z * z))
    if radius < 1e-10:
        return 0.0, 0.0, 0.0
    elevation_deg = float(np.degrees(np.arcsin(np.clip(y / radius, -1.0, 1.0))))
    azimuth_deg = float(np.degrees(np.arctan2(x, z)))
    return radius, azimuth_deg, elevation_deg


def spherical_to_cartesian(
    radius: float,
    azimuth_deg: float,
    elevation_deg: float
) -> NDArray[np.float32]:
    """
    Convert spherical coordinates to Cartesian (Y-up convention).

    Used for generating camera positions on orbit paths.

    Args:
        radius: Distance from origin
        azimuth_deg: Angle in XZ plane from +Z axis (degrees)
                     Positive = rotating toward +X (counterclockwise from above)
        elevation_deg: Angle above XZ plane (degrees)
                       Positive = up, negative = down

    Returns:
        Cartesian coordinates [x, y, z]

    Note:
        Azimuth 0° = +Z direction (in front)
        Azimuth 90° = +X direction (right)
        Azimuth 180° = -Z direction (behind)
        Azimuth 270° = -X direction (left)

        Elevation 0° = on XZ plane (eye level)
        Elevation +45° = above, looking down
        Elevation -45° = below, looking up
    """
    azim_rad = np.radians(azimuth_deg)
    elev_rad = np.radians(elevation_deg)

    x = radius * np.cos(elev_rad) * np.sin(azim_rad)
    y = radius * np.sin(elev_rad)
    z = radius * np.cos(elev_rad) * np.cos(azim_rad)

    return np.array([x, y, z], dtype=np.float32)
