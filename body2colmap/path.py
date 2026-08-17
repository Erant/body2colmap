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


def compute_original_camera_orbit_params(
    target: NDArray[np.float32],
    camera_position: Optional[NDArray[np.float32]] = None
) -> dict:
    """
    Compute orbit parameters that place the original camera at frame 0.

    Given a look-at target (typically mesh bbox center) and the original
    camera position (typically the origin for SAM-3D-Body), computes the
    radius, start_azimuth_deg, and elevation_deg that would place
    the first orbit frame at the original camera's location.

    This is a standalone utility so it can be used outside of OrbitPipeline.

    Args:
        target: 3D point the orbit centers on (look-at target)
        camera_position: Original camera position in world coords.
                        Defaults to origin [0, 0, 0] (SAM-3D-Body convention).

    Returns:
        Dictionary with keys:
            radius: Distance from target to camera
            start_azimuth_deg: Azimuth angle of camera relative to target
            elevation_deg: Elevation angle of camera relative to target
            target: The target point (pass-through)
            camera_position: The camera position used
    """
    if camera_position is None:
        camera_position = np.zeros(3, dtype=np.float32)

    # Vector from target to camera
    offset = np.array(camera_position, dtype=np.float32) - np.array(target, dtype=np.float32)

    # Convert to spherical coordinates
    radius, azimuth_deg, elevation_deg = coordinates.cartesian_to_spherical(offset)

    return {
        'radius': radius,
        'start_azimuth_deg': azimuth_deg,
        'elevation_deg': elevation_deg,
        'target': np.array(target, dtype=np.float32),
        'camera_position': np.array(camera_position, dtype=np.float32),
    }


def helical_elevation_deg(
    progress: float,
    amplitude_deg: float,
    n_loops: int,
    lead_in_deg: float,
    lead_out_deg: float
) -> float:
    """
    Elevation of a helical orbit at a given fractional progress.

    Single source of truth for the helix elevation ramp, shared by
    :meth:`OrbitPath.helical` and :func:`compute_helical_anchor_params`.

    Args:
        progress: Position in the sequence, 0.0 (first frame) to 1.0 (end)
        amplitude_deg: Elevation range (ramp goes from -amplitude to +amplitude)
        n_loops: Number of full 360 degree rotations
        lead_in_deg: Degrees of rotation before the ramp starts
        lead_out_deg: Degrees of rotation after the ramp ends

    Returns:
        Elevation angle in degrees (before any elevation offset is applied)
    """
    total_deg = lead_in_deg + (n_loops * 360.0) + lead_out_deg

    if progress < (lead_in_deg / total_deg):
        # Lead-in: stay at bottom
        return -amplitude_deg
    if progress > (1.0 - lead_out_deg / total_deg):
        # Lead-out: stay at top
        return amplitude_deg

    # Main loops: linear elevation change
    loop_progress = (progress - lead_in_deg / total_deg) / (
        1.0 - (lead_in_deg + lead_out_deg) / total_deg
    )
    return -amplitude_deg + (2 * amplitude_deg * loop_progress)


def compute_helical_anchor_params(
    target: NDArray[np.float32],
    camera_position: Optional[NDArray[np.float32]] = None,
    *,
    n_frames: int,
    n_loops: int = 3,
    amplitude_deg: float = 30.0,
    lead_in_deg: float = 45.0,
    lead_out_deg: float = 45.0,
    max_elevation_error_deg: float = 2.0
) -> dict:
    """
    Compute helical orbit parameters that make one frame land on a given camera.

    Unlike a circular orbit — where the anchor can always be frame 0 because
    elevation is constant — a helix sweeps elevation monotonically, so the
    anchor's elevation dictates *where in the sequence* it can occur.

    Two knobs make the helix pass through the anchor pose:

    1. Inverting the elevation ramp at the anchor's elevation gives a
       fractional progress, which rounds to the nearest frame index ``k``.
    2. ``start_azimuth_deg`` is solved so frame ``k`` hits the anchor's
       azimuth exactly.
    3. The rounding in step 1 leaves a small elevation residual, returned as
       ``elevation_offset_deg``.  Applying it to *every* frame keeps the path
       a perfect helix (its band just shifts by that offset) while placing
       frame ``k`` exactly on the anchor.

    Use the returned ``radius``, ``start_azimuth_deg`` and
    ``elevation_offset_deg`` with :meth:`OrbitPath.helical` to generate the
    path; ``anchor_frame_index`` tells the caller which rendered frame
    corresponds to the anchor camera.

    Args:
        target: 3D point the orbit centers on (look-at target)
        camera_position: Anchor camera position in world coords.
                        Defaults to origin [0, 0, 0] (SAM-3D-Body convention).
        n_frames: Number of frames the helix will be sampled at
        n_loops: Number of full 360 degree rotations
        amplitude_deg: Elevation range (ramp goes from -amplitude to +amplitude)
        lead_in_deg: Degrees of rotation before the ramp starts
        lead_out_deg: Degrees of rotation after the ramp ends
        max_elevation_error_deg: Largest elevation correction tolerated before
                        raising, in degrees.  Guards against silently shifting
                        the elevation band by a large amount.

    Returns:
        Dictionary with keys:
            radius: Distance from target to anchor camera
            anchor_frame_index: Index of the frame that lands on the anchor
            start_azimuth_deg: Azimuth to start the helix at
            elevation_offset_deg: Uniform elevation correction to apply
            anchor_azimuth_deg: Anchor azimuth relative to target
            anchor_elevation_deg: Anchor elevation relative to target
            target: The target point (pass-through)
            camera_position: The anchor camera position used

    Raises:
        ValueError: If the anchor coincides with the target, the helix has no
            elevation ramp to solve on, the anchor lies outside the helix's
            elevation band, or the helix is sampled too coarsely to reach the
            anchor within ``max_elevation_error_deg``.
    """
    if camera_position is None:
        camera_position = np.zeros(3, dtype=np.float32)

    if n_frames < 1:
        raise ValueError(f"n_frames must be at least 1, got {n_frames}")

    # Vector from target to anchor camera
    offset = np.array(camera_position, dtype=np.float32) - np.array(target, dtype=np.float32)
    radius, anchor_azimuth_deg, anchor_elevation_deg = coordinates.cartesian_to_spherical(offset)

    if radius < 1e-6:
        raise ValueError(
            "Anchor camera coincides with the orbit target; cannot derive "
            "orbit parameters from a zero-length offset."
        )

    if n_loops < 1:
        raise ValueError(
            f"Helical anchoring requires at least 1 loop, got n_loops={n_loops}. "
            "With no loops the helix has no elevation ramp to solve on."
        )

    if amplitude_deg <= 0.0:
        raise ValueError(
            f"Helical anchoring requires a positive amplitude, got "
            f"amplitude_deg={amplitude_deg}."
        )

    if abs(anchor_elevation_deg) > amplitude_deg + max_elevation_error_deg:
        raise ValueError(
            f"Anchor camera sits at elevation {anchor_elevation_deg:.2f}deg, outside "
            f"the helix elevation band of +/-{amplitude_deg:.2f}deg. Increase "
            f"helical_amplitude_deg to at least {int(np.ceil(abs(anchor_elevation_deg)))}."
        )

    total_deg = lead_in_deg + (n_loops * 360.0) + lead_out_deg
    lead_in_frac = lead_in_deg / total_deg
    lead_out_frac = lead_out_deg / total_deg
    ramp_frac = 1.0 - lead_in_frac - lead_out_frac

    # Invert the elevation ramp: elevation -> fractional progress
    clamped_elevation = float(np.clip(anchor_elevation_deg, -amplitude_deg, amplitude_deg))
    ramp_t = (clamped_elevation + amplitude_deg) / (2.0 * amplitude_deg)
    progress_exact = lead_in_frac + ramp_t * ramp_frac

    # Snap to the nearest actual frame
    anchor_frame_index = int(np.clip(round(progress_exact * n_frames), 0, n_frames - 1))
    progress_frame = anchor_frame_index / n_frames

    # Residual elevation, applied uniformly to shift the whole helix
    base_elevation = helical_elevation_deg(
        progress_frame, amplitude_deg, n_loops, lead_in_deg, lead_out_deg
    )
    elevation_offset_deg = float(anchor_elevation_deg - base_elevation)

    if abs(elevation_offset_deg) > max_elevation_error_deg:
        raise ValueError(
            f"Helix is sampled too coarsely to reach the anchor: nearest frame "
            f"({anchor_frame_index}) is {elevation_offset_deg:.2f}deg off in elevation, "
            f"above the {max_elevation_error_deg:.2f}deg limit. Increase n_frames, "
            f"reduce helical_amplitude_deg, or raise max_elevation_error_deg."
        )

    # Solve start azimuth so frame `anchor_frame_index` hits the anchor azimuth
    start_azimuth_deg = anchor_azimuth_deg - progress_frame * total_deg
    # Wrap to (-180, 180] for readability; azimuth is modulo 360
    start_azimuth_deg = float((start_azimuth_deg + 180.0) % 360.0 - 180.0)

    return {
        'radius': radius,
        'anchor_frame_index': anchor_frame_index,
        'start_azimuth_deg': start_azimuth_deg,
        'elevation_offset_deg': elevation_offset_deg,
        'anchor_azimuth_deg': anchor_azimuth_deg,
        'anchor_elevation_deg': anchor_elevation_deg,
        'target': np.array(target, dtype=np.float32),
        'camera_position': np.array(camera_position, dtype=np.float32),
    }


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
        up_vector: Optional[NDArray[np.float32]] = None,
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
        elevation_offset_deg: float = 0.0,
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
            elevation_offset_deg: Uniform elevation shift applied to every
                                 frame. Used by anchored orbits (see
                                 :func:`compute_helical_anchor_params`) to
                                 make one frame land exactly on a given
                                 camera pose without breaking the helix.
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

            elevation_deg = helical_elevation_deg(
                progress, amplitude_deg, n_loops, lead_in_deg, lead_out_deg
            ) + elevation_offset_deg

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
