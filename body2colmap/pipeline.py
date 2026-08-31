"""
High-level pipeline orchestrating all components.

This module provides the OrbitPipeline class which ties together:
- Scene loading (mesh or Gaussian splat)
- Orbit path generation
- Rendering
- Export to COLMAP and images

This is the main API for users of the library.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray

from .scene import Scene
from .camera import Camera
from .path import (
    OrbitPath,
    compute_helical_anchor_params,
    compute_original_camera_orbit_params,
)
from .renderer import Renderer, parse_composite_modes
from .exporter import ColmapExporter, ImageExporter
from .utils import (
    compute_default_focal_length,
    compute_auto_orbit_radius,
    compute_original_view_framing as _compute_original_view_framing,
    compute_warp_to_camera,
)


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
            self.focal_length = compute_default_focal_length(render_size[0])
        else:
            self.focal_length = focal_length

        # Will be set by set_orbit_params()
        self.cameras: Optional[List[Camera]] = None
        self.orbit_params: Optional[Dict[str, Any]] = None

        # Renderer (lazily created)
        self._renderer: Optional[Renderer] = None

        # Optional Gaussian-splat overlay (see attach_splat_overlay)
        self._splat_overlay = None
        self._splat_overlay_renderer = None
        self.splat_overlay_params: Optional[Dict[str, Any]] = None

        # Splat backend settings (see configure_splat_renderer)
        self._splat_binary: Optional[str] = None
        self._splat_confidence = None
        self._splat_verbose = False
        self._splat_on_fault = None

        # Set by auto_orient(); the splat overlay is incompatible with it
        self._auto_oriented = False

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

    @classmethod
    def from_ply_file(
        cls,
        filepath: str,
        render_size: Tuple[int, int] = (512, 512)
    ) -> "OrbitPipeline":
        """
        Create pipeline from Gaussian Splat PLY file.

        Args:
            filepath: Path to .ply file
            render_size: (width, height) for rendering

        Returns:
            OrbitPipeline instance with SplatScene

        Note:
            Requires ``plyfile`` (``pip install body2colmap[splat]``) to read
            the ply, and the ``brush-splat-render`` binary to rasterize it --
            see :func:`~body2colmap.splat_renderer.resolve_binary`.
        """
        from .splat_scene import SplatScene
        scene = SplatScene.from_ply(filepath)
        return cls(scene, render_size)

    def _is_splat_scene(self) -> bool:
        """Check if scene is a SplatScene (vs mesh Scene)."""
        # Import here to avoid circular imports and allow optional dependency
        try:
            from .splat_scene import SplatScene
            return isinstance(self.scene, SplatScene)
        except ImportError:
            return False

    @property
    def renderer(self):
        """Get or create the renderer for this pipeline's scene type."""
        if self._renderer is None:
            if self._is_splat_scene():
                from .splat_renderer import SplatRenderer
                self._renderer = SplatRenderer(
                    self.scene,
                    self.render_size,
                    binary=self._splat_binary,
                    confidence=self._splat_confidence,
                    verbose=self._splat_verbose,
                    on_fault=self._splat_on_fault,
                )
            else:
                self._renderer = Renderer(self.scene, self.render_size)
        return self._renderer

    def configure_splat_renderer(
        self,
        binary: Optional[str] = None,
        confidence=None,
        verbose: bool = False,
        on_fault=None,
    ) -> "OrbitPipeline":
        """
        Set how Gaussian splats are rasterized.

        Applies to both the ``.ply``-input base render and the overlay. Call
        before the renderer is first used -- it is created lazily on first
        render and these settings are read then.

        Args:
            binary: Path to ``brush-splat-render``. None resolves it from
                ``$BRUSH_SPLAT_RENDER`` then ``PATH``.
            confidence: Optional
                :class:`~body2colmap.splat_renderer.ConfidenceOptions`. Valid
                only for a ``.ply``-input base render -- see
                :meth:`attach_splat_overlay`, which refuses it.
            verbose: Let the binary log per-frame progress to stderr.
            on_fault: Called with a
                :class:`~body2colmap.splat_renderer.RenderFault` when a render
                goes wrong, while its temp directory is still on disk. The
                seam for saving a crash report -- see that class.

        Returns:
            self (for method chaining)
        """
        self._splat_binary = binary
        self._splat_confidence = confidence
        self._splat_verbose = verbose
        self._splat_on_fault = on_fault
        return self

    @property
    def splat_confidence_maps(self) -> Optional[List[NDArray[np.uint8]]]:
        """
        Raw per-pixel confidence maps from the most recent splat render.

        Populated only when :meth:`configure_splat_renderer` was given
        ``ConfidenceOptions(sidecar=True)``. One 8-bit greyscale image per
        frame, before the ``gate_lo``/``gate_hi`` smoothstep that produces the
        rendered alpha — useful for choosing those thresholds.
        """
        if self._renderer is None:
            return None
        return getattr(self._renderer, "last_confidence_maps", None)

    def auto_orient(self, rotation_offset_deg: float = 0.0) -> None:
        """
        Rotate the scene so the body faces the camera at orbit frame 0.

        Computes the torso facing direction from the skeleton's shoulder
        and hip joints, then rotates the entire scene (mesh + skeleton)
        around the Y axis so that the body faces -Z (toward the camera).

        An optional offset rotates further from that facing position.

        Args:
            rotation_offset_deg: Additional rotation in degrees after
                auto-facing. 0 = face camera directly, 90 = turned 90
                degrees to the right, etc.
        """
        facing = self.scene.compute_torso_facing_direction()

        if facing is not None:
            # Angle of current facing direction from +Z axis
            current_angle = float(np.arctan2(facing[0], facing[2]))
            # Angle of target (-Z) from +Z axis
            target_angle = float(np.arctan2(0.0, -1.0))  # pi
            correction_deg = float(np.degrees(target_angle - current_angle))
        else:
            correction_deg = 0.0

        total_rotation = correction_deg + rotation_offset_deg
        self.scene.rotate_around_y(total_rotation)
        self._auto_oriented = True

    def attach_splat_overlay(
        self,
        ply_path: str,
        meta_path: str,
        original_focal_length: float,
        original_image_size: Tuple[int, int],
        crop_box: Optional[Tuple[int, int, int, int]] = None,
        scale: Optional[float] = None,
        reconcile_intrinsics: bool = True,
        max_angle_deg: float = 45.0,
    ) -> "OrbitPipeline":
        """
        Attach an externally-produced Gaussian splat as a composite overlay.

        The splat is anchored into this scene's world frame by
        :func:`body2colmap.splat_anchor.anchor_splat_to_world` — see that
        module for the geometry. Once attached, ``"splat"`` becomes a valid
        overlay in composite modes such as ``"skeleton+splat"``.

        Args:
            ply_path: The 3DGS ``.ply``.
            meta_path: Its ``splat_meta.json``.
            original_focal_length: SAM-3D-Body's ``focal_length`` from the .npz.
            original_image_size: ``(width, height)`` of the full original photo
                SAM-3D-Body was run on.
            crop_box: ``(x0, y0, x1, y1)`` in full-image pixels, the region the
                splat's input image was cut from. ``None`` if the splat was
                built from the whole photo.
            scale: The depth gauge. ``None`` fits it against the mesh.
            reconcile_intrinsics: See
                :func:`~body2colmap.splat_anchor.compute_anchor_transform`.
            max_angle_deg: Cull the splat on frames viewing it from more than
                this many degrees off its source view direction. The splat is a
                2.5-D shell reconstructed from one photo — there is nothing
                behind the subject — so as the camera turns away, the open rim
                of the shell swings into view as a flare of grazing-incidence
                splats.

                The default of 45 was measured, not assumed: on a Face_Neck
                head splat the face reads cleanly to about 30 degrees, the rim
                starts flaring by 45, and by 60 the shell is mostly edge. Raise
                it if you would rather have coverage than a clean silhouette.

        Returns:
            self (for method chaining)

        Raises:
            RuntimeError: If the scene is a SplatScene (nothing to composite
                against), if it has been auto-oriented — ``auto_orient()``
                rotates the scene about its bbox centre, which moves it out of
                the original camera's frame and invalidates the anchoring —
                or if confidence gating is configured, which an overlay splat
                cannot support.
        """
        from .splat_anchor import anchor_splat_to_world
        from .splat_renderer import SplatRenderer

        if self._is_splat_scene():
            raise RuntimeError(
                "Cannot attach a splat overlay to a SplatScene. The overlay "
                "composites against a mesh and skeleton; load the .npz instead."
            )
        if self._auto_oriented:
            raise RuntimeError(
                "Cannot attach a splat overlay after auto_orient(). The anchor "
                "places the splat relative to the original camera at the "
                "origin, and auto_orient() rotates the scene about its bbox "
                "centre, breaking that relationship. Drop --auto-orient."
            )
        if self._splat_confidence is not None:
            raise RuntimeError(
                "Confidence gating is not available for a splat overlay. It "
                "scores each Gaussian by how well the training views "
                "constrained it, and an overlay splat is reconstructed from a "
                "single photograph — there are no training views and no "
                "evidence block, so the gate would silently degenerate to "
                "plain alpha. It also composites over cull_color and writes "
                "the gate as alpha, which the overlay's straight-alpha "
                "compositing cannot use. Drop --splat-confidence."
            )

        w, h = original_image_size
        probe_renderer = Renderer(self.scene, render_size=(int(w), int(h)))
        try:
            splat, info = anchor_splat_to_world(
                ply_path=ply_path,
                meta_path=meta_path,
                original_focal_length=original_focal_length,
                original_image_size=(int(w), int(h)),
                crop_box=crop_box,
                scale=scale,
                reconcile_intrinsics=reconcile_intrinsics,
                depth_probe=probe_renderer._render_depth_buffer,
            )
        finally:
            # The probe renderer holds its own OpenGL context; it is only
            # needed for the one depth read used to fit the gauge. Release it
            # explicitly rather than leaving it to the garbage collector.
            probe_renderer.delete()

        self._splat_overlay = splat
        self._splat_overlay_renderer = SplatRenderer(
            splat,
            self.render_size,
            binary=self._splat_binary,
            verbose=self._splat_verbose,
            on_fault=self._splat_on_fault,
        )
        info["max_angle_deg"] = float(max_angle_deg)
        self.splat_overlay_params = info
        return self

    @property
    def has_splat_overlay(self) -> bool:
        """Whether a Gaussian-splat overlay has been attached."""
        return self._splat_overlay is not None

    def splat_view_angle_deg(self, camera: Camera) -> float:
        """
        Angle between a camera's view of the splat and the splat's source view.

        Zero at the original photograph's viewpoint, growing as the orbit turns
        away from it.

        Args:
            camera: Camera to measure.

        Returns:
            Angle in degrees, in [0, 180].

        Raises:
            RuntimeError: If no splat overlay is attached.
        """
        if not self.has_splat_overlay:
            raise RuntimeError("No splat overlay attached.")

        center = self._splat_overlay.get_bbox_center().astype(np.float64)
        to_splat = center - np.asarray(camera.position, dtype=np.float64)
        norm = np.linalg.norm(to_splat)
        if norm < 1e-9:
            return 0.0

        cos = float(np.dot(to_splat / norm, self.splat_overlay_params["source_view_dir"]))
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    def _splat_layer_kept(self, camera: Camera) -> bool:
        """Whether the splat overlay survives the view-angle cull for a camera."""
        if not self.has_splat_overlay:
            return False
        return (
            self.splat_view_angle_deg(camera)
            <= self.splat_overlay_params["max_angle_deg"]
        )

    def render_splat_layer(self, camera: Camera) -> Optional[NDArray[np.uint8]]:
        """
        Render the splat overlay for one camera, or None if it is culled.

        The layer comes back with **straight** alpha so it composites correctly
        over whatever is underneath.

        This costs a full ``brush-splat-render`` invocation, so use it for
        one-off frames only. For a sequence, call :meth:`render_splat_layers`,
        which renders the whole camera list in one invocation.

        Args:
            camera: Camera to render from.

        Returns:
            RGBA image, or None when no overlay is attached or the camera is
            beyond ``max_angle_deg`` of the splat's source view.
        """
        if not self._splat_layer_kept(camera):
            return None

        return self._splat_overlay_renderer.render(camera, bg_color=None)

    def render_splat_layers(
        self,
        cameras: List[Camera]
    ) -> List[Optional[NDArray[np.uint8]]]:
        """
        Render the splat overlay for many cameras in one binary invocation.

        The batched form of :meth:`render_splat_layer`, and the one to use for
        a sequence: ``brush-splat-render`` initializes wgpu and loads the ply
        once per invocation, so rendering 81 frames one at a time pays that
        setup 81 times.

        Culled cameras are not sent to the binary at all; their slots come back
        as ``None``, so the result lines up index-for-index with ``cameras``
        and ``render_composite(splat_layer=...)`` needs no special case.

        Args:
            cameras: Cameras to render from, in order.

        Returns:
            List of RGBA images with **straight** alpha, ``None`` at every
            index with no overlay or beyond ``max_angle_deg``.
        """
        layers: List[Optional[NDArray[np.uint8]]] = [None] * len(cameras)
        kept = [i for i, cam in enumerate(cameras) if self._splat_layer_kept(cam)]
        if not kept:
            return layers

        rendered = self._splat_overlay_renderer.render_many(
            [cameras[i] for i in kept], bg_color=None
        )
        for i, image in zip(kept, rendered):
            layers[i] = image
        return layers

    def set_orbit_params(
        self,
        pattern: str = "helical",
        n_frames: int = 120,
        radius: Optional[float] = None,
        framing: str = "full",
        original_focal_length: Optional[float] = None,
        **kwargs
    ) -> "OrbitPipeline":
        """
        Set orbit path parameters and generate cameras.

        Args:
            pattern: Orbit pattern type ("circular", "sinusoidal", "helical")
            n_frames: Number of frames to generate
            radius: Orbit radius (distance from target)
                   If None, auto-computed to frame scene
            framing: Body framing preset ("full", "torso", "bust", "head")
                    Non-full presets use skeleton joints to determine Y threshold
                    and filter mesh vertices for accurate framing bounds.
            original_focal_length: When set, anchors one frame of the orbit to
                the original SAM-3D-Body camera (at the origin). The orbit
                radius, target, and start azimuth are derived from the
                mesh's position relative to the origin, so the orbit
                smoothly passes through the original viewpoint.
                All cameras share the auto-framed focal length.
                The scene must NOT have been auto-oriented.

                For "circular" the anchor is frame 0. For "helical" the
                elevation ramp determines where the anchor can occur, so the
                index is solved for and reported as
                ``orbit_params['anchor_frame_index']``; always read it rather
                than assuming 0. "sinusoidal" is not anchored.
            **kwargs: Pattern-specific parameters:
                - circular: elevation_deg
                - sinusoidal: amplitude_deg, n_cycles
                - helical: n_loops, amplitude_deg, lead_in_deg, lead_out_deg

        Returns:
            self (for method chaining)
        """
        if original_focal_length is not None:
            # --- Original-camera mode ---
            # Auto-frame using the same logic as render_original_view so
            # that frame 0 matches the debug output (zoomed + centered).
            fill_ratio = kwargs.pop('fill_ratio', 0.8)
            framing_info = self.compute_original_view_framing(
                original_focal_length, fill_ratio
            )
            framed_fl = framing_info['framed_focal_length']

            # Derive orbit start position from mesh geometry
            framing_bounds = self.scene.get_framing_bounds(preset=framing)
            target = (framing_bounds[0] + framing_bounds[1]) / 2.0

            orbit_params = compute_original_camera_orbit_params(target)
            start_azimuth_deg = orbit_params['start_azimuth_deg']
            derived_elevation_deg = orbit_params['elevation_deg']

            # Use the geometric distance (||target||) as orbit radius.
            # This ensures frame 0 lands exactly at the origin (the
            # original camera position) via the spherical-coordinate
            # roundtrip, so there is NO position or rotation discontinuity
            # between frame 0 and frame 1.  The framed_fl was computed to
            # frame the subject at exactly this distance, so framing is
            # correct by construction.
            radius = float(orbit_params['radius'])

            # All frames (including frame 0) share the same template:
            # framed focal length, centered principal point.  look_at()
            # handles centering for every frame including frame 0.
            camera_template = Camera(
                focal_length=(framed_fl, framed_fl),
                image_size=self.render_size
            )

            orbit = OrbitPath(target=target, radius=radius)

            # Inject start_azimuth_deg into kwargs for all patterns
            kwargs['start_azimuth_deg'] = start_azimuth_deg

            # Which frame lands on the original camera.  Circular orbits have
            # constant elevation, so frame 0 always works; helical orbits sweep
            # elevation and must solve for the index (see below).
            anchor_frame_index = 0
            anchor_info: Optional[Dict[str, Any]] = None

            if pattern == "circular":
                # In original-camera mode the elevation is geometrically
                # determined — it's not a user-tunable parameter.  Always
                # use the derived value so frame 0 lands at the origin.
                kwargs.pop('elevation_deg', None)
                elevation_deg = derived_elevation_deg
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

                # A helix sweeps elevation, so the original camera can only be
                # reached at the frame whose elevation matches it.  Solve for
                # that frame, the start azimuth that lands on it, and the small
                # uniform elevation shift that makes it exact.
                anchor_info = compute_helical_anchor_params(
                    target=target,
                    n_frames=n_frames,
                    n_loops=n_loops,
                    amplitude_deg=amplitude_deg,
                    lead_in_deg=lead_in_deg,
                    lead_out_deg=lead_out_deg,
                )
                anchor_frame_index = anchor_info['anchor_frame_index']

                # Overrides the circular-mode value injected above, which
                # only places frame 0 correctly for a constant elevation.
                kwargs['start_azimuth_deg'] = anchor_info['start_azimuth_deg']

                self.cameras = orbit.helical(
                    n_frames=n_frames,
                    n_loops=n_loops,
                    amplitude_deg=amplitude_deg,
                    lead_in_deg=lead_in_deg,
                    lead_out_deg=lead_out_deg,
                    elevation_offset_deg=anchor_info['elevation_offset_deg'],
                    camera_template=camera_template,
                    **kwargs
                )
            else:
                raise ValueError(f"Unknown orbit pattern: {pattern}")

            # Compute the homography that warps the original image to align
            # with the anchor frame's look_at view.  This accounts for both
            # the focal-length zoom and the slight rotation correction.
            anchor_camera = self.cameras[anchor_frame_index]
            warp_homography = compute_warp_to_camera(
                original_focal_length=original_focal_length,
                original_image_size=self.render_size,
                target_camera=anchor_camera,
            )

            self.orbit_params = {
                'pattern': pattern,
                'n_frames': n_frames,
                'radius': radius,
                'original_focal_length': original_focal_length,
                'framed_focal_length': framed_fl,
                # start_azimuth_deg arrives via **kwargs below, so it always
                # reflects the value actually used (helical solves its own).
                'derived_elevation_deg': derived_elevation_deg,
                'framing_info': framing_info,
                'anchor_frame_index': anchor_frame_index,
                'anchor_camera': anchor_camera,
                'anchor_elevation_offset_deg': (
                    anchor_info['elevation_offset_deg'] if anchor_info else 0.0
                ),
                'warp_homography': warp_homography,
                **kwargs
            }

            # Store the pipeline focal length as the framed value
            self.focal_length = framed_fl

            return self

        # --- Standard orbit mode ---
        # Get framing bounds based on preset
        # For partial body presets, this filters mesh vertices by Y coordinate
        framing_bounds = self.scene.get_framing_bounds(preset=framing)

        # Compute orbit center (look-at target) from framing region
        target = (framing_bounds[0] + framing_bounds[1]) / 2.0

        # Auto-compute radius if not provided
        if radius is None:
            fill_ratio = kwargs.pop('fill_ratio', 0.8)
            radius = compute_auto_orbit_radius(
                bounds=framing_bounds,
                render_size=self.render_size,
                focal_length=self.focal_length,
                fill_ratio=fill_ratio
            )

        # Create orbit path generator
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
            modes: List of render modes
                - For mesh scenes: "mesh", "depth", "outline", "skeleton"
                - For splat scenes: "splat"
            **render_kwargs: Mode-specific rendering options:
                - mesh_color: RGB tuple (0-1) for mesh
                - bg_color: RGB tuple (0-1) for background
                - normalize_depth: bool for depth rendering
                - outline_color: RGB tuple (0-1) for the outline foreground
                - outline_bg_color: RGB tuple (0-1) for the outline background
                - outline_style: "filled" or "stroke"
                - outline_thickness: stroke width in px (style="stroke" only)
                - outline_blur: blur radius in px for the outline (0 = off)
                - eye_style: "shape" (filled eye + pupil disc) or "dots"
                - eye_color: RGB tuple (0-1) for the filled eye shape
                - pupil_color: RGB tuple (0-1) for the pupil disc
                - pupil_scale: pupil diameter as a fraction of eye height, (0, 1]
                - etc.

        Returns:
            Dictionary mapping mode name to list of rendered images
            Example: {"mesh": [img1, img2, ...], "depth": [img1, img2, ...]}

        Raises:
            RuntimeError: If cameras haven't been set (call set_orbit_params first)
        """
        if self.cameras is None:
            raise RuntimeError("Cameras not set. Call set_orbit_params() first.")

        # Get appropriate renderer for scene type
        renderer = self.renderer
        is_splat = self._is_splat_scene()

        results = {}

        for mode in modes:
            if mode == "splat":
                # Splat rendering (only valid for SplatScene). Batched: the
                # brush binary loads the ply and initializes wgpu once per
                # invocation, so a per-frame loop would pay that n_frames times.
                if not is_splat:
                    raise ValueError("'splat' mode only valid for SplatScene")
                results[mode] = renderer.render_many(
                    self.cameras,
                    bg_color=render_kwargs.get('bg_color', (1.0, 1.0, 1.0))
                )
                continue

            images = []

            # Render each frame
            for i, camera in enumerate(self.cameras):
                if mode == "mesh":
                    if is_splat:
                        raise ValueError("'mesh' mode not valid for SplatScene, use 'splat'")
                    image = renderer.render_mesh(
                        camera,
                        mesh_color=render_kwargs.get('mesh_color'),
                        bg_color=render_kwargs.get('bg_color', (1.0, 1.0, 1.0))
                    )
                elif mode == "depth":
                    if is_splat:
                        raise ValueError("'depth' mode not yet supported for SplatScene")
                    image = renderer.render_depth(
                        camera,
                        normalize=render_kwargs.get('normalize_depth', True),
                        colormap=render_kwargs.get('depth_colormap')
                    )
                elif mode == "outline":
                    if is_splat:
                        raise ValueError("'outline' mode not yet supported for SplatScene")
                    image = renderer.render_outline(
                        camera,
                        fg_color=render_kwargs.get('outline_color', (0.0, 0.0, 0.0)),
                        bg_color=render_kwargs.get('outline_bg_color', (1.0, 1.0, 1.0)),
                        style=render_kwargs.get('outline_style', 'filled'),
                        thickness=render_kwargs.get('outline_thickness', 3),
                        blur=render_kwargs.get('outline_blur', 4),
                    )
                elif mode == "skeleton":
                    if is_splat:
                        raise ValueError("'skeleton' mode not valid for SplatScene")
                    image = renderer.render_skeleton(
                        camera,
                        joint_radius=render_kwargs.get('joint_radius', 0.015),
                        bone_radius=render_kwargs.get('bone_radius', 0.008),
                        face_mode=render_kwargs.get('face_mode'),
                        face_landmarks=render_kwargs.get('face_landmarks'),
                        eye_style=render_kwargs.get('eye_style'),
                        eye_color=render_kwargs.get('eye_color'),
                        pupil_color=render_kwargs.get('pupil_color'),
                        pupil_scale=render_kwargs.get('pupil_scale'),
                        bg_color=render_kwargs.get('bg_color'),
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
            ValueError: If used with SplatScene (composites not supported)
        """
        if self.cameras is None:
            raise RuntimeError("Cameras not set. Call set_orbit_params() first.")

        if self._is_splat_scene():
            raise ValueError("Composite rendering not supported for SplatScene")

        # Create renderer if needed (mesh renderer for composites)
        renderer = self.renderer

        # Rendered up front, in one binary invocation, rather than per frame.
        splat_layers = self.render_splat_layers(self.cameras)

        images = []
        for camera, splat_layer in zip(self.cameras, splat_layers):
            image = renderer.render_composite(
                camera, composite_modes,
                splat_layer=splat_layer,
            )
            images.append(image)

        return images

    def compute_original_view_framing(
        self,
        original_focal_length: float,
        fill_ratio: float = 0.8
    ) -> Dict[str, Any]:
        """
        Compute auto-framing parameters for the original SAM-3D-Body viewpoint.

        Thin wrapper around the standalone
        :func:`body2colmap.utils.compute_original_view_framing` using
        this pipeline's scene vertices and render size.

        See that function for full documentation and return value details.
        """
        return _compute_original_view_framing(
            vertices=self.scene.vertices,
            render_size=self.render_size,
            original_focal_length=original_focal_length,
            fill_ratio=fill_ratio,
        )

    def render_original_view(
        self,
        original_focal_length: float,
        modes: List[str] = ["mesh"],
        auto_frame: bool = False,
        fill_ratio: float = 0.8,
        **render_kwargs
    ) -> Tuple[Dict[str, NDArray[np.uint8]], Optional[Dict[str, Any]]]:
        """
        Render a single frame from the original SAM-3D-Body viewpoint.

        After sam3d_to_world(), the original camera is at the origin with
        identity rotation (the 180-degree X-axis flip that converts SAM-3D
        coords to world coords also converts the camera from OpenCV to
        OpenGL convention). So we just need the original focal length.

        IMPORTANT: The scene must NOT have been auto-oriented or rotated
        for this to produce a correct overlay.

        Args:
            original_focal_length: Focal length from the .npz file, in pixels.
            modes: Render modes (same as render_all)
            auto_frame: If True, adjust focal length and principal point to
                center and fill the frame. Also returns framing metadata.
            fill_ratio: Target fill ratio when auto_frame=True (0-1)
            **render_kwargs: Passed through to render methods

        Returns:
            Tuple of (rendered_images, framing_info):
                rendered_images: Dict mapping mode name to single rendered image
                framing_info: Dict with framing metadata including the Camera
                    object used for rendering (key ``'camera'``).
        """
        if auto_frame:
            framing_info = self.compute_original_view_framing(
                original_focal_length, fill_ratio
            )
            fl = framing_info['framed_focal_length']
            cx, cy = framing_info['framed_principal_point']
        else:
            w, h = self.render_size
            fl = original_focal_length
            cx = w / 2.0
            cy = h / 2.0
            framing_info = {
                'scale_factor': 1.0,
                'framed_focal_length': float(fl),
                'framed_principal_point': [float(cx), float(cy)],
                'original_focal_length': float(original_focal_length),
                'original_principal_point': [float(cx), float(cy)],
                'affine_matrix': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                'inverse_affine_matrix': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            }

        # Camera at origin, identity rotation, chosen intrinsics
        camera = Camera(
            focal_length=(fl, fl),
            image_size=self.render_size,
            principal_point=(cx, cy),
            position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            rotation=np.eye(3, dtype=np.float32)
        )
        framing_info['camera'] = camera

        renderer = self.renderer
        results = {}

        for mode in modes:
            if '+' in mode:
                # Composite mode — delegate to render_composite
                base_mode, overlays = parse_composite_modes(mode)

                # Build composite config from render_kwargs
                composite_modes = {}
                if base_mode == "mesh":
                    composite_modes["mesh"] = {
                        "color": render_kwargs.get('mesh_color'),
                        "bg_color": render_kwargs.get('bg_color', (1.0, 1.0, 1.0)),
                    }
                elif base_mode == "depth":
                    composite_modes["depth"] = {}
                elif base_mode == "outline":
                    composite_modes["outline"] = {
                        "fg_color": render_kwargs.get('outline_color', (0.0, 0.0, 0.0)),
                        "bg_color": render_kwargs.get('outline_bg_color', (1.0, 1.0, 1.0)),
                        "style": render_kwargs.get('outline_style', 'filled'),
                        "thickness": render_kwargs.get('outline_thickness', 3),
                        "blur": render_kwargs.get('outline_blur', 4),
                    }

                for overlay in overlays:
                    if overlay == "skeleton":
                        composite_modes["skeleton"] = {
                            "joint_radius": render_kwargs.get('joint_radius', 0.015),
                            "bone_radius": render_kwargs.get('bone_radius', 0.008),
                        }
                    elif overlay == "face":
                        composite_modes["face"] = {
                            "face_mode": render_kwargs.get('face_mode', 'full'),
                            "face_landmarks": render_kwargs.get('face_landmarks'),
                            "eye_style": render_kwargs.get('eye_style'),
                            "eye_color": render_kwargs.get('eye_color'),
                            "pupil_color": render_kwargs.get('pupil_color'),
                            "pupil_scale": render_kwargs.get('pupil_scale'),
                        }

                image = renderer.render_composite(
                    camera, composite_modes,
                    splat_layer=self.render_splat_layer(camera),
                )
            elif mode == "mesh":
                image = renderer.render_mesh(
                    camera,
                    mesh_color=render_kwargs.get('mesh_color'),
                    bg_color=render_kwargs.get('bg_color', (1.0, 1.0, 1.0))
                )
            elif mode == "depth":
                image = renderer.render_depth(
                    camera,
                    normalize=render_kwargs.get('normalize_depth', True),
                    colormap=render_kwargs.get('depth_colormap')
                )
            elif mode == "outline":
                image = renderer.render_outline(
                    camera,
                    fg_color=render_kwargs.get('outline_color', (0.0, 0.0, 0.0)),
                    bg_color=render_kwargs.get('outline_bg_color', (1.0, 1.0, 1.0)),
                    style=render_kwargs.get('outline_style', 'filled'),
                    thickness=render_kwargs.get('outline_thickness', 3),
                    blur=render_kwargs.get('outline_blur', 4),
                )
            elif mode == "skeleton":
                image = renderer.render_skeleton(
                    camera,
                    joint_radius=render_kwargs.get('joint_radius', 0.015),
                    bone_radius=render_kwargs.get('bone_radius', 0.008),
                    face_mode=render_kwargs.get('face_mode'),
                    face_landmarks=render_kwargs.get('face_landmarks'),
                    eye_style=render_kwargs.get('eye_style'),
                    eye_color=render_kwargs.get('eye_color'),
                    pupil_color=render_kwargs.get('pupil_color'),
                    pupil_scale=render_kwargs.get('pupil_scale'),
                    bg_color=render_kwargs.get('bg_color'),
                )
            else:
                raise ValueError(f"Unknown render mode: {mode}")

            results[mode] = image

        return results, framing_info

    def export_colmap(
        self,
        output_dir: str,
        n_pointcloud_samples: int = 50000,
        filename_pattern: str = "frame_{:04d}.png"
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
            filename_pattern: Filename pattern for images (must match actual image files)

        Returns:
            Path to output directory

        Raises:
            RuntimeError: If cameras haven't been set
        """
        if self.cameras is None:
            raise RuntimeError("Cameras not set. Call set_orbit_params() first.")

        # Generate image filenames using the provided pattern
        image_names = ImageExporter.generate_filenames(
            n_frames=len(self.cameras),
            pattern=filename_pattern
        )

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
