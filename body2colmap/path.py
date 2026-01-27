"""
Orbit path generation for camera trajectories.

This module provides the OrbitPath class which generates camera positions
following various orbit patterns (circular, helical, sinusoidal).

All camera positions and orientations are in world coordinates.
Cameras look at a target point (typically mesh centroid).
"""

import numpy as np
from typing import List, Optional, Tuple
from numpy.typing import NDArray

from .camera import Camera
from . import coordinates


class OrbitPath:
    """
    Generate camera orbit paths around a target point.

    Supports multiple orbit patterns:
    - Circular: Fixed elevation, rotating azimuth (turntable)
    - Sinusoidal: Oscillating elevation as camera rotates
    - Helical: Multiple full rotations with linear elevation change

    All patterns generate a list of Camera objects with positions and
    orientations set to orbit around a target point.
    """

    def __init__(
        self,
        target: NDArray[np.float32],
        radius: float,
        up_vector: Optional[NDArray[np.float32]] = None
    ):
        """
        Initialize OrbitPath.

        Args:
            target: 3D point to orbit around (typically mesh centroid)
            radius: Distance from target to camera
            up_vector: World up direction for elevation reference
                      Default: [0, 1, 0] (Y-up)
        """
        self.target = np.array(target, dtype=np.float32)
        self.radius = radius
        self.up_vector = up_vector if up_vector is not None else coordinates.WorldCoordinates.UP_AXIS

    def circular(
        self,
        n_frames: int,
        elevation_deg: float = 0.0,
        start_azimuth_deg: float = 0.0,
        overlap: int = 1,
        camera_template: Optional[Camera] = None
    ) -> List[Camera]:
        """
        Generate circular orbit path (turntable).

        Camera rotates around target at fixed elevation.

        Args:
            n_frames: Number of frames (camera positions)
            elevation_deg: Fixed elevation angle in degrees
                          0° = eye level, positive = above, negative = below
            start_azimuth_deg: Starting azimuth angle in degrees
            overlap: Number of camera positions that overlap between start and end
                    overlap=1 (default): first and last positions are identical
                    overlap=2: first 2 and last 2 positions overlap
                    overlap=0: no overlap (first and last positions differ)
            camera_template: Camera with intrinsics to copy
                            If None, creates default cameras

        Returns:
            List of Camera objects positioned on circular orbit

        Raises:
            ValueError: If overlap >= n_frames or overlap < 0
        """
        if overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap}")
        if overlap >= n_frames:
            raise ValueError(f"overlap ({overlap}) must be less than n_frames ({n_frames})")

        # Generate unique camera positions
        unique_frames = n_frames - overlap
        cameras = []

        for i in range(unique_frames):
            # Compute azimuth for this frame
            # Distribute unique frames evenly around full 360° rotation
            azimuth_deg = start_azimuth_deg + (i / unique_frames) * 360.0

            # Convert spherical to Cartesian (relative to target)
            position_rel = coordinates.spherical_to_cartesian(
                self.radius, azimuth_deg, elevation_deg
            )
            position = self.target + position_rel

            # Create camera
            camera = self._create_camera(position, camera_template)
            camera.look_at(self.target, self.up_vector)

            cameras.append(camera)

        # Append overlapping cameras from the beginning
        for i in range(overlap):
            cameras.append(cameras[i])

        return cameras

    def sinusoidal(
        self,
        n_frames: int,
        amplitude_deg: float = 30.0,
        n_cycles: int = 2,
        start_azimuth_deg: float = 0.0,
        camera_template: Optional[Camera] = None
    ) -> List[Camera]:
        """
        Generate sinusoidal orbit path.

        Camera rotates 360° while elevation oscillates in sine wave.

        Args:
            n_frames: Number of frames
            amplitude_deg: Amplitude of elevation oscillation in degrees
            n_cycles: Number of up/down cycles during rotation
            start_azimuth_deg: Starting azimuth angle in degrees
            camera_template: Camera with intrinsics to copy

        Returns:
            List of Camera objects positioned on sinusoidal orbit
        """
        cameras = []

        for i in range(n_frames):
            # Azimuth: full 360° rotation
            azimuth_deg = start_azimuth_deg + (i / n_frames) * 360.0

            # Elevation: sine wave oscillation
            phase = (i / n_frames) * n_cycles * 2 * np.pi
            elevation_deg = amplitude_deg * np.sin(phase)

            # Convert to Cartesian
            position_rel = coordinates.spherical_to_cartesian(
                self.radius, azimuth_deg, elevation_deg
            )
            position = self.target + position_rel

            # Create camera
            camera = self._create_camera(position, camera_template)
            camera.look_at(self.target, self.up_vector)

            cameras.append(camera)

        return cameras

    def helical(
        self,
        n_frames: int,
        n_loops: int = 3,
        amplitude_deg: float = 30.0,
        lead_in_deg: float = 45.0,
        lead_out_deg: float = 45.0,
        start_azimuth_deg: float = 0.0,
        camera_template: Optional[Camera] = None
    ) -> List[Camera]:
        """
        Generate helical orbit path (spiral).

        Camera makes multiple full rotations while elevation changes linearly
        from -amplitude to +amplitude. Includes lead-in and lead-out sections
        for smooth start/end.

        This pattern provides excellent coverage for 3D Gaussian Splatting.

        Args:
            n_frames: Number of frames
            n_loops: Number of full 360° rotations
            amplitude_deg: Elevation range (goes from -amplitude to +amplitude)
            lead_in_deg: Degrees of rotation before first loop starts
            lead_out_deg: Degrees of rotation after last loop ends
            start_azimuth_deg: Starting azimuth angle in degrees
            camera_template: Camera with intrinsics to copy

        Returns:
            List of Camera objects positioned on helical orbit

        Note:
            Total rotation = lead_in + (n_loops * 360) + lead_out degrees

            Elevation progression:
            - Lead-in: -amplitude (bottom)
            - Loops: linearly increase from -amplitude to +amplitude
            - Lead-out: +amplitude (top)
        """
        cameras = []

        # Total degrees of rotation
        total_deg = lead_in_deg + (n_loops * 360.0) + lead_out_deg

        for i in range(n_frames):
            # Current rotation angle
            angle_deg = start_azimuth_deg + (i / n_frames) * total_deg

            # Compute elevation based on position in sequence
            progress = i / n_frames  # 0 to 1

            if progress < (lead_in_deg / total_deg):
                # Lead-in: stay at bottom
                elevation_deg = -amplitude_deg
            elif progress > (1.0 - lead_out_deg / total_deg):
                # Lead-out: stay at top
                elevation_deg = amplitude_deg
            else:
                # Main loops: linear elevation change
                loop_progress = (progress - lead_in_deg / total_deg) / (
                    1.0 - (lead_in_deg + lead_out_deg) / total_deg
                )
                elevation_deg = -amplitude_deg + (2 * amplitude_deg * loop_progress)

            # Convert to Cartesian
            position_rel = coordinates.spherical_to_cartesian(
                self.radius, angle_deg, elevation_deg
            )
            position = self.target + position_rel

            # Create camera
            camera = self._create_camera(position, camera_template)
            camera.look_at(self.target, self.up_vector)

            cameras.append(camera)

        return cameras

    def _create_camera(
        self,
        position: NDArray[np.float32],
        template: Optional[Camera] = None
    ) -> Camera:
        """
        Create camera at position, optionally copying intrinsics from template.

        Args:
            position: Camera position in world coords
            template: Camera to copy intrinsics from
                     If None, creates default camera (512x512, ~47° FOV)

        Returns:
            Camera instance
        """
        if template is not None:
            # Copy intrinsics from template
            camera = Camera(
                focal_length=(template.fx, template.fy),
                image_size=(template.width, template.height),
                principal_point=(template.cx, template.cy),
                position=position
            )
        else:
            # Create default camera
            camera = Camera.from_fov(
                fov_deg=47.0,
                image_size=(512, 512),
                position=position,
                is_horizontal_fov=True
            )

        return camera

    @staticmethod
    def auto_compute_radius(
        scene_bounds: Tuple[NDArray[np.float32], NDArray[np.float32]],
        fill_ratio: float = 0.8,
        fov_deg: float = 47.0
    ) -> float:
        """
        Automatically compute orbit radius to frame the scene.

        Computes the distance needed so the scene fills a specified portion
        of the viewport.

        Args:
            scene_bounds: (min_corner, max_corner) from Scene.get_bounds()
            fill_ratio: How much of viewport to fill (0.0 to 1.0)
                       0.8 = scene occupies 80% of image
            fov_deg: Camera field of view in degrees

        Returns:
            Recommended orbit radius
        """
        min_corner, max_corner = scene_bounds
        scene_size = np.linalg.norm(max_corner - min_corner)

        # Distance needed for scene to subtend desired angle
        desired_angle_rad = np.radians(fov_deg * fill_ratio)
        radius = (scene_size / 2.0) / np.tan(desired_angle_rad / 2.0)

        return radius

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"OrbitPath(target={self.target}, radius={self.radius:.2f})"
