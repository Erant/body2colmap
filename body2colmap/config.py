"""
Configuration management for body2colmap.

Handles:
- Command-line argument parsing
- YAML config file loading
- Configuration validation
- Merging configs with defaults
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import argparse

from .background import DEFAULT_RADIUS_SCALE
from .fade import (
    DECAY_PROFILES,
    DEFAULT_DETAIL,
    DEFAULT_FALLOFF,
    DEFAULT_MARGIN,
    DEFAULT_PROFILE,
    DEFAULT_RATE,
    FADE_TARGETS,
)
from .face import EYE_STYLES


@dataclass
class RenderConfig:
    """Rendering configuration."""
    resolution: Tuple[int, int] = (512, 512)
    mesh_color: Tuple[float, float, float] = (0.65, 0.74, 0.86)
    bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    modes: List[str] = field(default_factory=lambda: ["mesh"])

    # Outline mode: flat two-tone silhouette
    outline_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)      # foreground
    outline_bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)   # background
    outline_style: str = "filled"  # "filled" or "stroke"
    outline_thickness: int = 3     # stroke width in px (style="stroke" only)
    outline_blur: int = 4          # blur radius in px (0 = hard edges)


def _validate_background_geometry(value: str) -> str:
    """
    Validate a background surface type.

    Args:
        value: "sphere" or "cube"

    Returns:
        The validated value

    Raises:
        ValueError: If the value is not a known geometry
    """
    if value not in ("sphere", "cube"):
        raise ValueError(
            f"Invalid background geometry {value!r}. Use 'sphere' or 'cube'."
        )
    return value


@dataclass
class BackgroundFadeConfig:
    """
    Fade the backdrop out around the subject.

    Exists because the backdrop that fixes one failure causes another. In
    ``outline`` modes a grid that runs right up to the silhouette reads to a
    video model as a hard occlusion boundary, and it will not paint outside
    it -- bulky clothing and hair get squashed back onto the outline of the
    bare mesh. Clearing the backdrop in a shell around the subject keeps the
    rotation cue in the far field and gives the model room to expand.

    The shell is the projection of an ellipsoid fitted to the mesh, so it
    encloses the silhouette from every viewpoint on the orbit rather than
    tracking one frame's outline. See :mod:`body2colmap.fade`.
    """
    enabled: bool = False

    #: Decay profile: how the backdrop returns as you move away from the
    #: subject. See :data:`~body2colmap.fade.DECAY_PROFILES`.
    profile: str = DEFAULT_PROFILE

    #: Width of the fade band, as a multiple of the subject's own radius.
    #: Scale-free, so it holds up across an auto-framed orbit.
    falloff: float = DEFAULT_FALLOFF

    #: Shape constant for `exponential`, `gaussian` and `inverse_square`.
    #: Larger = tighter. Ignored by the other profiles.
    rate: float = DEFAULT_RATE

    #: Inflate the fitted ellipsoid before the fade is measured. Raise it
    #: when the mesh is a bare body and the target subject is not.
    margin: float = DEFAULT_MARGIN

    #: What the backdrop fades to. "local" averages the backdrop's own detail
    #: away, so the lines go but the wall/floor/ceiling tone carries through
    #: with no visible patch. "color" uses one flat colour.
    target: str = "local"

    #: Flat colour for target="color", RGB 0-1. null = the texture's mean.
    color: Optional[Tuple[float, float, float]] = None

    #: Long-side resolution the backdrop is averaged down to for
    #: target="local". Wants to be well below the texture's own frequency.
    detail: int = DEFAULT_DETAIL

    def validate(self) -> None:
        """
        Check the settings hang together.

        Raises:
            ValueError: On an unknown profile or target, a non-positive
                falloff, rate, margin or detail, or a malformed colour.
        """
        if self.profile not in DECAY_PROFILES:
            raise ValueError(
                f"Invalid background.fade.profile {self.profile!r}. "
                f"Choose from: {', '.join(sorted(DECAY_PROFILES))}"
            )
        if self.target not in FADE_TARGETS:
            raise ValueError(
                f"Invalid background.fade.target {self.target!r}. Use "
                f"{' or '.join(repr(t) for t in FADE_TARGETS)}."
            )
        if self.falloff <= 0.0:
            raise ValueError(
                f"background.fade.falloff must be > 0, got {self.falloff}. "
                f"Use profile: step for a hard-edged clear zone."
            )
        if self.rate <= 0.0:
            raise ValueError(
                f"background.fade.rate must be > 0, got {self.rate}"
            )
        if self.margin <= 0.0:
            raise ValueError(
                f"background.fade.margin must be > 0, got {self.margin}"
            )
        if self.detail < 1:
            raise ValueError(
                f"background.fade.detail must be >= 1 pixel, got {self.detail}"
            )
        if self.color is not None:
            if len(self.color) != 3:
                raise ValueError(
                    f"background.fade.color must be 3 RGB floats, got "
                    f"{self.color}"
                )
            if any(not 0.0 <= c <= 1.0 for c in self.color):
                raise ValueError(
                    f"background.fade.color components must be in [0, 1], "
                    f"got {self.color}"
                )


@dataclass
class BackgroundConfig:
    """
    Environment backdrop drawn behind the render.

    Exists to break a specific failure mode: with a blank background a video
    diffusion model reads an orbit as the *subject* rotating, and prompt
    conditioning is not strong enough to correct it. A world-fixed backdrop
    that sweeps past as the camera moves supplies the missing cue.

    Note that not every texture supplies it equally. A Nishita-style sky is
    azimuthally symmetric apart from the sun, so it barely changes as the
    camera orbits; ``checker`` and ``grid`` carry far more structure and are
    the honest test of whether the cue lands. See
    :mod:`body2colmap.background`.

    The defaults below are a ``grid`` cube at
    :data:`~body2colmap.background.DEFAULT_RADIUS_SCALE` times the orbit
    radius -- walls meeting at corners, over a floor and ceiling that read
    apart -- because that is the arrangement that carries the cue most
    strongly. The three settings are a set: a cube is only a room at a finite
    radius, and at infinity it flattens to a plain ruled field.

    The backdrop is for conditioning frames only: it is not exported to
    COLMAP, adds no points to the point cloud, and never enters the depth
    buffer or the silhouette mask.
    """
    enabled: bool = False

    #: "sphere" or "cube". At an infinite radius the two differ only in how
    #: the texture is parameterized; the geometric difference needs a radius.
    geometry: str = "cube"

    #: A built-in generator name (blender_sky, gradient, checker, grid) or a
    #: path to an image -- or, for a cube, a directory of six face images.
    texture: str = "grid"

    #: Equirect height / cube face size, for generated textures only.
    resolution: int = 1024

    #: Surface radius in world units. null leaves the surface at infinity, so
    #: it responds to camera rotation but not to translation.
    radius: Optional[float] = None

    #: Radius as a multiple of the orbit radius, for when the orbit is
    #: auto-framed and its scale is not known up front. Mutually exclusive
    #: with `radius`, and must exceed 1.0 so the camera stays inside.
    #: Defaulted rather than left at infinity because the default cube needs a
    #: finite radius to be a room at all. Set `radius` for world units, or
    #: both forms to None for an infinite backdrop.
    radius_scale: Optional[float] = DEFAULT_RADIUS_SCALE

    #: Rotate the environment about +Y, in degrees.
    rotation_deg: float = 0.0

    #: Force alpha to 255. True suits conditioning frames; False keeps the
    #: silhouette alpha intact and fills only RGB.
    opaque: bool = True

    #: Extra keyword arguments for a generator, e.g. {sun_azimuth_deg: 40}.
    #: Rejected when `texture` names a file.
    params: Dict[str, Any] = field(default_factory=dict)

    #: Fade the backdrop out around the subject, so an outline frame does not
    #: read as a hard occlusion boundary.
    fade: BackgroundFadeConfig = field(default_factory=BackgroundFadeConfig)

    def validate(self) -> None:
        """
        Check the settings hang together.

        Raises:
            ValueError: On an unknown geometry, both radius forms at once, a
                non-positive radius, a radius_scale that would put the camera
                outside the surface, or a bad resolution.
        """
        _validate_background_geometry(self.geometry)

        # Each value is checked before they are checked against each other:
        # radius_scale carries a default, so a caller who set only a bad radius
        # would otherwise be told about a conflict rather than about the value
        # they actually got wrong.
        if self.radius is not None and self.radius <= 0.0:
            raise ValueError(
                f"background.radius must be > 0, got {self.radius}"
            )
        if self.radius_scale is not None and self.radius_scale <= 1.0:
            raise ValueError(
                f"background.radius_scale must be > 1.0 so the camera stays "
                f"inside the backdrop, got {self.radius_scale}"
            )
        if self.radius is not None and self.radius_scale is not None:
            raise ValueError(
                "background.radius and background.radius_scale are mutually "
                "exclusive; set one or neither (neither = infinite backdrop). "
                f"Note that radius_scale defaults to {DEFAULT_RADIUS_SCALE}, "
                "so pass radius_scale=None alongside an explicit radius"
            )
        if self.resolution < 8:
            raise ValueError(
                f"background.resolution must be >= 8, got {self.resolution}"
            )
        self.fade.validate()


@dataclass
class CameraConfig:
    """Camera configuration."""
    focal_length: Optional[float] = None  # None = auto (47° FOV)
    auto_frame: bool = True
    fill_ratio: float = 0.8
    zoom: Optional[float] = None  # Overrides auto_frame if set


@dataclass
class PathConfig:
    """Orbit path configuration."""
    pattern: str = "helical"  # "circular", "sinusoidal", "helical"
    n_frames: int = 120
    radius: Optional[float] = None  # None = auto-compute
    framing: str = "full"  # "full", "torso", "bust", "head"
    crop_to_viewport: bool = False  # Filter mesh to first camera's viewport
    use_original_camera: bool = False  # Pin frame 0 to original SAM-3D-Body camera

    # Initial body orientation
    initial_rotation: float = 0.0  # Additional degrees offset after auto-facing

    # Circular-specific
    elevation_deg: float = 0.0

    # Sinusoidal-specific
    sinusoidal_amplitude_deg: float = 30.0
    sinusoidal_cycles: int = 2

    # Helical-specific
    helical_loops: int = 3
    helical_amplitude_deg: float = 30.0
    helical_lead_in_deg: float = 45.0
    helical_lead_out_deg: float = 45.0


def _validate_eye_style(value: str) -> str:
    """
    Validate an eye rendering style.

    Args:
        value: "shape" (filled eye with a pupil disc) or "dots" (the original
            OpenPose landmark dots)

    Returns:
        The validated value

    Raises:
        ValueError: If the value is not a known style
    """
    if value not in EYE_STYLES:
        raise ValueError(
            f"eye_style must be one of {EYE_STYLES}, got {value!r}"
        )
    return value


def _opt_tuple(value, length: int, cast):
    """Coerce an optional YAML list to a fixed-length tuple, or None."""
    if value is None:
        return None
    seq = tuple(cast(v) for v in value)
    if len(seq) != length:
        raise ValueError(f"Expected {length} values, got {len(seq)}: {value}")
    return seq


def _validate_crop_box(value):
    """
    Coerce and check a crop box: (x0, y0, x1, y1) in full-image pixels.

    Raises:
        ValueError: If it does not have four values or is degenerate.
    """
    box = _opt_tuple(value, 4, int)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"crop_box must be (x0, y0, x1, y1) with x1 > x0 and y1 > y0, got {box}"
        )
    return box


def _validate_pupil_scale(value: float) -> float:
    """
    Validate that a pupil scale is in (0, 1].

    Above 1.0 the pupil would be larger than the eye opening it sits in and
    would spill past the lids, so it is rejected rather than clamped.

    Args:
        value: Pupil diameter as a fraction of the eye height

    Returns:
        The validated value

    Raises:
        ValueError: If the value is outside (0, 1]
    """
    if not 0.0 < value <= 1.0:
        raise ValueError(
            f"pupil_scale must be in (0, 1], got {value}. "
            f"1.0 is a pupil as tall as the eye opening."
        )
    return value


@dataclass
class SkeletonConfig:
    """Skeleton rendering configuration."""
    enabled: bool = False
    format: str = "openpose_body25_hands"  # Default to OpenPose format
    joint_radius: float = 0.015
    bone_radius: float = 0.008
    face_mode: str = None  # None, "full", or "points"
    face_landmarks: str = None  # Path to face landmarks JSON file
    face_max_angle: float = 90.0  # Max degrees off face normal to render (90 = full hemisphere)
    eye_style: str = "shape"  # "shape" (filled eye + pupil) or "dots" (landmark dots)
    eye_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)   # filled eye shape (sclera)
    pupil_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    pupil_scale: float = 0.75  # Pupil diameter as a fraction of eye height, (0, 1]


@dataclass
class SplatConfig:
    """
    Gaussian-splat configuration.

    Two unrelated things live here, because both concern splats:

    **The overlay** (``overlay_ply`` and everything up to ``max_angle_deg``).
    A splat produced externally from the same photograph that feeds
    SAM-3D-Body (see ~/Projects/masktest), composited on top of the skeleton
    via the ``splat`` overlay layer, e.g. ``modes: ["skeleton+splat"]``.

    **The rasterizer** (``renderer_binary`` and the confidence fields).
    Applies to any splat rendering, including a ``.ply`` input rendered as the
    base layer. Confidence gating is valid only for that base case — an
    overlay splat has no training views to measure evidence against, and
    :meth:`~body2colmap.pipeline.OrbitPipeline.attach_splat_overlay` refuses it.
    """
    overlay_ply: Optional[str] = None   # None disables the overlay entirely
    meta_json: Optional[str] = None     # default: splat_meta.json beside the PLY
    original_image_size: Optional[Tuple[int, int]] = None  # default: from the .npz
    crop_box: Optional[Tuple[int, int, int, int]] = None   # x0,y0,x1,y1 in full-image px
    scale: Optional[float] = None       # None = fit the depth gauge against the mesh
    reconcile_intrinsics: bool = True
    max_angle_deg: float = 45.0         # cull past this far off the source view

    # Rasterizer: brush-splat-render. None resolves $BRUSH_SPLAT_RENDER, then PATH.
    renderer_binary: Optional[str] = None

    # Confidence gating. NOTE: this changes what the alpha channel means —
    # with it, alpha is the confidence gate, not accumulated opacity.
    confidence: bool = False
    cull_color: Optional[Tuple[float, float, float]] = None  # None = render.bg_color
    gate_lo: float = 0.45
    gate_hi: float = 0.65
    confidence_sidecar: bool = False
    confidence_dataset: Optional[str] = None   # measure evidence here if the ply has none
    confidence_extra_args: List[str] = field(default_factory=list)

    def confidence_options(self):
        """
        Build :class:`~body2colmap.splat_renderer.ConfidenceOptions`, or None.

        Returns None when confidence gating is off, which is exactly what
        :meth:`~body2colmap.pipeline.OrbitPipeline.configure_splat_renderer`
        wants for the ungated case.
        """
        if not self.confidence:
            return None
        from .splat_renderer import ConfidenceOptions
        return ConfidenceOptions(
            cull_color=self.cull_color,   # None = follow the render's bg_color
            gate_lo=self.gate_lo,
            gate_hi=self.gate_hi,
            sidecar=self.confidence_sidecar,
            dataset=self.confidence_dataset,
            extra_args=tuple(self.confidence_extra_args),
        )

    def resolved_meta_json(self) -> Optional[str]:
        """The metadata path, defaulting to splat_meta.json beside the PLY."""
        if self.meta_json is not None:
            return self.meta_json
        if self.overlay_ply is None:
            return None
        return str(Path(self.overlay_ply).parent / "splat_meta.json")


@dataclass
class ExportConfig:
    """Export configuration."""
    output_dir: str = "./output"
    image_format: str = "png"
    filename_pattern: str = "frame_{:04d}.png"
    colmap: bool = True
    pointcloud_samples: int = 50000


@dataclass
class Config:
    """Complete configuration."""
    input_file: str
    render: RenderConfig = field(default_factory=RenderConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    path: PathConfig = field(default_factory=PathConfig)
    skeleton: SkeletonConfig = field(default_factory=SkeletonConfig)
    splat: SplatConfig = field(default_factory=SplatConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """
        Create config from parsed command-line arguments.

        Loads config file if specified, then applies command-line overrides.

        Args:
            args: Parsed arguments from argparse

        Returns:
            Config instance

        Raises:
            ValueError: If neither config file nor required args are provided
        """
        # Load from config file if specified
        if args.config:
            config = cls.from_yaml(args.config, input_file_override=args.input)
        else:
            # Create from defaults
            if not args.input:
                raise ValueError("input file must be specified (positional argument or in config file)")
            if not args.output_dir:
                raise ValueError("--output-dir must be specified (or in config file)")

            config = cls(
                input_file=args.input,
                render=RenderConfig(),
                background=BackgroundConfig(),
                camera=CameraConfig(),
                path=PathConfig(),
                skeleton=SkeletonConfig(),
                export=ExportConfig(output_dir=args.output_dir)
            )

        # Apply command-line overrides
        if args.input:
            config.input_file = args.input
        if args.output_dir:
            config.export.output_dir = args.output_dir

        # Render overrides
        if args.resolution:
            try:
                w, h = args.resolution.lower().split('x')
                config.render.resolution = (int(w), int(h))
            except ValueError:
                raise ValueError(f"Invalid resolution format: {args.resolution}. Use WxH (e.g., 512x512)")

        # Individual width/height overrides (take precedence over --resolution)
        if args.width is not None or args.height is not None:
            width = args.width if args.width is not None else config.render.resolution[0]
            height = args.height if args.height is not None else config.render.resolution[1]
            config.render.resolution = (width, height)

        if args.render_modes:
            config.render.modes = [m.strip() for m in args.render_modes.split(',')]

        if args.mesh_color:
            try:
                r, g, b = [float(x) for x in args.mesh_color.split(',')]
                config.render.mesh_color = (r, g, b)
            except ValueError:
                raise ValueError(f"Invalid mesh-color format: {args.mesh_color}. Use R,G,B (e.g., 0.65,0.74,0.86)")

        if args.bg_color:
            try:
                r, g, b = [float(x) for x in args.bg_color.split(',')]
                config.render.bg_color = (r, g, b)
            except ValueError:
                raise ValueError(f"Invalid bg-color format: {args.bg_color}. Use R,G,B (e.g., 1.0,1.0,1.0)")

        if args.outline_color:
            try:
                r, g, b = [float(x) for x in args.outline_color.split(',')]
                config.render.outline_color = (r, g, b)
            except ValueError:
                raise ValueError(f"Invalid outline-color format: {args.outline_color}. Use R,G,B (e.g., 0.0,0.0,0.0)")

        if args.outline_bg_color:
            try:
                r, g, b = [float(x) for x in args.outline_bg_color.split(',')]
                config.render.outline_bg_color = (r, g, b)
            except ValueError:
                raise ValueError(f"Invalid outline-bg-color format: {args.outline_bg_color}. Use R,G,B (e.g., 1.0,1.0,1.0)")

        if args.outline_style:
            config.render.outline_style = args.outline_style

        if args.outline_thickness is not None:
            config.render.outline_thickness = args.outline_thickness

        if args.outline_blur is not None:
            config.render.outline_blur = args.outline_blur

        # Camera overrides
        if args.focal_length is not None:
            config.camera.focal_length = args.focal_length
        if args.fill_ratio is not None:
            config.camera.fill_ratio = args.fill_ratio

        # Path overrides
        if args.orbit_pattern:
            config.path.pattern = args.orbit_pattern
        if args.orbit_radius is not None:
            config.path.radius = args.orbit_radius
        if args.n_frames is not None:
            config.path.n_frames = args.n_frames
        if args.elevation is not None:
            config.path.elevation_deg = args.elevation
        if args.helical_loops is not None:
            config.path.helical_loops = args.helical_loops
        if args.amplitude is not None:
            config.path.helical_amplitude_deg = args.amplitude
            config.path.sinusoidal_amplitude_deg = args.amplitude
        if args.framing:
            config.path.framing = args.framing
        if args.crop_to_viewport:
            config.path.crop_to_viewport = True
        if args.initial_rotation is not None:
            config.path.initial_rotation = args.initial_rotation
        if args.use_original_camera:
            config.path.use_original_camera = True

        # Skeleton overrides
        if args.skeleton:
            config.skeleton.enabled = True
        if args.skeleton_format:
            config.skeleton.format = args.skeleton_format
        if args.joint_radius is not None:
            config.skeleton.joint_radius = args.joint_radius
        if args.bone_radius is not None:
            config.skeleton.bone_radius = args.bone_radius
        if args.face_mode:
            if args.face_mode == "none":
                config.skeleton.face_mode = None
            else:
                config.skeleton.face_mode = args.face_mode
        if args.face_landmarks:
            config.skeleton.face_landmarks = args.face_landmarks
            # Providing landmarks implies face rendering
            if config.skeleton.face_mode is None:
                config.skeleton.face_mode = "full"
        if args.face_max_angle is not None:
            config.skeleton.face_max_angle = args.face_max_angle
        if args.eye_style:
            config.skeleton.eye_style = _validate_eye_style(args.eye_style)
        if args.eye_color:
            try:
                r, g, b = [float(x) for x in args.eye_color.split(',')]
                config.skeleton.eye_color = (r, g, b)
            except ValueError:
                raise ValueError(f"Invalid eye-color format: {args.eye_color}. Use R,G,B (e.g., 1.0,1.0,1.0)")
        if args.pupil_color:
            try:
                r, g, b = [float(x) for x in args.pupil_color.split(',')]
                config.skeleton.pupil_color = (r, g, b)
            except ValueError:
                raise ValueError(f"Invalid pupil-color format: {args.pupil_color}. Use R,G,B (e.g., 0.0,0.0,0.0)")
        if args.pupil_scale is not None:
            config.skeleton.pupil_scale = _validate_pupil_scale(args.pupil_scale)

        # Splat overlay overrides
        if args.splat_overlay:
            config.splat.overlay_ply = args.splat_overlay
        if args.splat_meta:
            config.splat.meta_json = args.splat_meta
        if args.splat_crop:
            try:
                config.splat.crop_box = _validate_crop_box(
                    [int(x) for x in args.splat_crop.split(',')]
                )
            except ValueError as e:
                raise ValueError(
                    f"Invalid --splat-crop {args.splat_crop!r}: {e}. "
                    "Use X0,Y0,X1,Y1 in full-image pixels (e.g. 141,0,594,477)"
                )
        if args.splat_image_size:
            try:
                w, h = args.splat_image_size.lower().split('x')
                config.splat.original_image_size = (int(w), int(h))
            except ValueError:
                raise ValueError(
                    f"Invalid --splat-image-size format: {args.splat_image_size}. "
                    "Use WxH (e.g., 757x1536)"
                )
        if args.splat_scale is not None:
            config.splat.scale = args.splat_scale
        if args.splat_no_reconcile:
            config.splat.reconcile_intrinsics = False
        if args.splat_max_angle is not None:
            config.splat.max_angle_deg = args.splat_max_angle
        if args.splat_renderer is not None:
            config.splat.renderer_binary = args.splat_renderer
        if args.splat_confidence:
            config.splat.confidence = True
        if args.splat_cull_color is not None:
            try:
                config.splat.cull_color = _opt_tuple(
                    args.splat_cull_color.split(','), 3, float
                )
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Invalid --splat-cull-color {args.splat_cull_color!r}: {e}. "
                    "Expected R,G,B in 0-1, e.g. 0.5,0.5,0.5"
                )
        if args.splat_gate_lo is not None:
            config.splat.gate_lo = args.splat_gate_lo
        if args.splat_gate_hi is not None:
            config.splat.gate_hi = args.splat_gate_hi
        if args.splat_confidence_sidecar:
            config.splat.confidence_sidecar = True
        if args.splat_confidence_dataset is not None:
            config.splat.confidence_dataset = args.splat_confidence_dataset

        # Background overrides
        if args.background is not None:
            config.background.enabled = True
            config.background.texture = args.background
        if args.no_background:
            config.background.enabled = False
        if args.background_geometry is not None:
            config.background.geometry = _validate_background_geometry(
                args.background_geometry
            )
        if args.background_resolution is not None:
            config.background.resolution = args.background_resolution
        if args.background_radius is not None:
            config.background.radius = args.background_radius
            # An explicit radius supersedes a scale from the config file, which
            # would otherwise trip the mutual-exclusion check below.
            config.background.radius_scale = None
        if args.background_radius_scale is not None:
            config.background.radius_scale = args.background_radius_scale
            config.background.radius = None
        if args.background_infinite:
            config.background.radius = None
            config.background.radius_scale = None
        if args.background_rotation is not None:
            config.background.rotation_deg = args.background_rotation
        if args.background_keep_alpha:
            config.background.opaque = False
        if args.background_fade is not None:
            config.background.fade.enabled = True
            config.background.fade.profile = args.background_fade
        if args.no_background_fade:
            config.background.fade.enabled = False
        if args.background_fade_falloff is not None:
            config.background.fade.falloff = args.background_fade_falloff
        if args.background_fade_rate is not None:
            config.background.fade.rate = args.background_fade_rate
        if args.background_fade_margin is not None:
            config.background.fade.margin = args.background_fade_margin
        if args.background_fade_target is not None:
            config.background.fade.target = args.background_fade_target
        if args.background_fade_detail is not None:
            config.background.fade.detail = args.background_fade_detail
        if args.background_fade_color:
            try:
                r, g, b = [float(x) for x in args.background_fade_color.split(',')]
            except ValueError:
                raise ValueError(
                    f"Invalid --background-fade-color format: "
                    f"{args.background_fade_color}. Use R,G,B (e.g., 0.5,0.5,0.5)"
                )
            config.background.fade.color = (r, g, b)
            # Naming a colour is only meaningful against the flat target, and
            # silently ignoring it would look like the colour did not work.
            if args.background_fade_target is None:
                config.background.fade.target = "color"
        config.background.validate()

        # Export overrides
        if args.no_colmap:
            config.export.colmap = False
        if args.pointcloud_samples is not None:
            config.export.pointcloud_samples = args.pointcloud_samples
        if args.filename_pattern:
            config.export.filename_pattern = args.filename_pattern

        return config

    @classmethod
    def from_yaml(cls, filepath: str, input_file_override: Optional[str] = None) -> "Config":
        """
        Load config from YAML file.

        Args:
            filepath: Path to YAML config file
            input_file_override: Override input file from command line

        Returns:
            Config instance
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for config files. "
                "Install with: pip install pyyaml"
            )

        with open(filepath, 'r') as f:
            data = yaml.safe_load(f) or {}

        # Parse input file
        input_file = input_file_override or data.get('input_file', '')
        if not input_file:
            raise ValueError("input_file must be specified in config or command line")

        # Parse render config
        render_data = data.get('render', {})
        render = RenderConfig(
            resolution=tuple(render_data.get('resolution', [512, 512])),
            mesh_color=tuple(render_data.get('mesh_color', [0.65, 0.74, 0.86])),
            bg_color=tuple(render_data.get('bg_color', [1.0, 1.0, 1.0])),
            modes=render_data.get('modes', ['mesh']),
            outline_color=tuple(render_data.get('outline_color', [0.0, 0.0, 0.0])),
            outline_bg_color=tuple(render_data.get('outline_bg_color', [1.0, 1.0, 1.0])),
            outline_style=render_data.get('outline_style', 'filled'),
            outline_thickness=render_data.get('outline_thickness', 3),
            outline_blur=render_data.get('outline_blur', 4)
        )

        # Parse background config
        background_data = data.get('background', {})

        # A radius written in the file supersedes the default radius_scale,
        # which the two being mutually exclusive would otherwise turn into an
        # error for a perfectly reasonable file. Writing either key as null is
        # how a file asks for an infinite backdrop.
        if 'radius' in background_data and 'radius_scale' not in background_data:
            background_radius_scale = None
        else:
            background_radius_scale = background_data.get(
                'radius_scale', DEFAULT_RADIUS_SCALE
            )

        fade_data = background_data.get('fade') or {}
        fade_color = fade_data.get('color')
        background_fade = BackgroundFadeConfig(
            enabled=fade_data.get('enabled', False),
            profile=fade_data.get('profile', DEFAULT_PROFILE),
            falloff=fade_data.get('falloff', DEFAULT_FALLOFF),
            rate=fade_data.get('rate', DEFAULT_RATE),
            margin=fade_data.get('margin', DEFAULT_MARGIN),
            target=fade_data.get('target', 'local'),
            color=None if fade_color is None else tuple(fade_color),
            detail=fade_data.get('detail', DEFAULT_DETAIL),
        )

        background = BackgroundConfig(
            enabled=background_data.get('enabled', False),
            geometry=_validate_background_geometry(
                background_data.get('geometry', 'cube')
            ),
            texture=background_data.get('texture', 'grid'),
            resolution=background_data.get('resolution', 1024),
            radius=background_data.get('radius'),
            radius_scale=background_radius_scale,
            rotation_deg=background_data.get('rotation_deg', 0.0),
            opaque=background_data.get('opaque', True),
            params=dict(background_data.get('params') or {}),
            fade=background_fade,
        )
        background.validate()

        # Parse camera config
        camera_data = data.get('camera', {})
        camera = CameraConfig(
            focal_length=camera_data.get('focal_length'),
            auto_frame=camera_data.get('auto_frame', True),
            fill_ratio=camera_data.get('fill_ratio', 0.8),
            zoom=camera_data.get('zoom')
        )

        # Parse path config
        path_data = data.get('path', {})
        path = PathConfig(
            pattern=path_data.get('pattern', 'helical'),
            n_frames=path_data.get('n_frames', 120),
            radius=path_data.get('radius'),
            framing=path_data.get('framing', 'full'),
            crop_to_viewport=path_data.get('crop_to_viewport', False),
            use_original_camera=path_data.get('use_original_camera', False),
            initial_rotation=path_data.get('initial_rotation', 0.0),
            elevation_deg=path_data.get('elevation_deg', 0.0),
            sinusoidal_amplitude_deg=path_data.get('sinusoidal_amplitude_deg', 30.0),
            sinusoidal_cycles=path_data.get('sinusoidal_cycles', 2),
            helical_loops=path_data.get('helical_loops', 3),
            helical_amplitude_deg=path_data.get('helical_amplitude_deg', 30.0),
            helical_lead_in_deg=path_data.get('helical_lead_in_deg', 45.0),
            helical_lead_out_deg=path_data.get('helical_lead_out_deg', 45.0)
        )

        # Parse skeleton config
        skeleton_data = data.get('skeleton', {})
        skeleton = SkeletonConfig(
            enabled=skeleton_data.get('enabled', False),
            format=skeleton_data.get('format', 'openpose_body25_hands'),
            joint_radius=skeleton_data.get('joint_radius', 0.015),
            bone_radius=skeleton_data.get('bone_radius', 0.008),
            face_mode=skeleton_data.get('face_mode', None),
            face_landmarks=skeleton_data.get('face_landmarks', None),
            face_max_angle=skeleton_data.get('face_max_angle', 90.0),
            eye_style=_validate_eye_style(
                skeleton_data.get('eye_style', 'shape')
            ),
            eye_color=tuple(skeleton_data.get('eye_color', [1.0, 1.0, 1.0])),
            pupil_color=tuple(skeleton_data.get('pupil_color', [0.0, 0.0, 0.0])),
            pupil_scale=_validate_pupil_scale(
                skeleton_data.get('pupil_scale', 0.75)
            )
        )

        # Parse splat config
        splat_data = data.get('splat', {})
        if 'device' in splat_data:
            # Splats are rasterized by brush (wgpu/Vulkan), not gsplat on a
            # torch device, so this key no longer selects anything. Raise
            # rather than ignore it: a stale config would otherwise quietly
            # mean something other than what it says.
            raise ValueError(
                "splat.device is no longer supported. Gaussian splats are "
                "rasterized by the brush-splat-render binary, which picks its "
                "own wgpu adapter. Remove the key, and use "
                "splat.renderer_binary (or $BRUSH_SPLAT_RENDER) if you need to "
                "point at a specific build."
            )
        splat = SplatConfig(
            overlay_ply=splat_data.get('overlay_ply'),
            meta_json=splat_data.get('meta_json'),
            original_image_size=_opt_tuple(splat_data.get('original_image_size'), 2, int),
            crop_box=_validate_crop_box(splat_data.get('crop_box')),
            scale=splat_data.get('scale'),
            reconcile_intrinsics=splat_data.get('reconcile_intrinsics', True),
            max_angle_deg=splat_data.get('max_angle_deg', 45.0),
            renderer_binary=splat_data.get('renderer_binary'),
            confidence=splat_data.get('confidence', False),
            cull_color=_opt_tuple(splat_data.get('cull_color'), 3, float),
            gate_lo=splat_data.get('gate_lo', 0.45),
            gate_hi=splat_data.get('gate_hi', 0.65),
            confidence_sidecar=splat_data.get('confidence_sidecar', False),
            confidence_dataset=splat_data.get('confidence_dataset'),
            confidence_extra_args=list(splat_data.get('confidence_extra_args') or []),
        )

        # Parse export config
        export_data = data.get('export', {})
        export = ExportConfig(
            output_dir=export_data.get('output_dir', './output'),
            image_format=export_data.get('image_format', 'png'),
            filename_pattern=export_data.get('filename_pattern', 'frame_{:04d}.png'),
            colmap=export_data.get('colmap', True),
            pointcloud_samples=export_data.get('pointcloud_samples', 50000)
        )

        return cls(
            input_file=input_file,
            render=render,
            background=background,
            camera=camera,
            path=path,
            skeleton=skeleton,
            splat=splat,
            export=export
        )

    def to_yaml(self, filepath: str) -> None:
        """
        Save config to YAML file.

        Args:
            filepath: Path to save YAML config file
        """
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for config files. "
                "Install with: pip install pyyaml"
            )

        data = {
            'input_file': self.input_file,
            'render': {
                'resolution': list(self.render.resolution),
                'mesh_color': list(self.render.mesh_color),
                'bg_color': list(self.render.bg_color),
                'modes': self.render.modes,
                'outline_color': list(self.render.outline_color),
                'outline_bg_color': list(self.render.outline_bg_color),
                'outline_style': self.render.outline_style,
                'outline_thickness': self.render.outline_thickness,
                'outline_blur': self.render.outline_blur
            },
            'background': {
                'enabled': self.background.enabled,
                'geometry': self.background.geometry,
                'texture': self.background.texture,
                'resolution': self.background.resolution,
                'radius': self.background.radius,
                'radius_scale': self.background.radius_scale,
                'rotation_deg': self.background.rotation_deg,
                'opaque': self.background.opaque,
                'params': dict(self.background.params),
                'fade': {
                    'enabled': self.background.fade.enabled,
                    'profile': self.background.fade.profile,
                    'falloff': self.background.fade.falloff,
                    'rate': self.background.fade.rate,
                    'margin': self.background.fade.margin,
                    'target': self.background.fade.target,
                    'color': (
                        None if self.background.fade.color is None
                        else list(self.background.fade.color)
                    ),
                    'detail': self.background.fade.detail,
                },
            },
            'camera': {
                'focal_length': self.camera.focal_length,
                'auto_frame': self.camera.auto_frame,
                'fill_ratio': self.camera.fill_ratio,
                'zoom': self.camera.zoom
            },
            'path': {
                'pattern': self.path.pattern,
                'n_frames': self.path.n_frames,
                'radius': self.path.radius,
                'framing': self.path.framing,
                'crop_to_viewport': self.path.crop_to_viewport,
                'use_original_camera': self.path.use_original_camera,
                'initial_rotation': self.path.initial_rotation,
                'elevation_deg': self.path.elevation_deg,
                'sinusoidal_amplitude_deg': self.path.sinusoidal_amplitude_deg,
                'sinusoidal_cycles': self.path.sinusoidal_cycles,
                'helical_loops': self.path.helical_loops,
                'helical_amplitude_deg': self.path.helical_amplitude_deg,
                'helical_lead_in_deg': self.path.helical_lead_in_deg,
                'helical_lead_out_deg': self.path.helical_lead_out_deg
            },
            'skeleton': {
                'enabled': self.skeleton.enabled,
                'format': self.skeleton.format,
                'joint_radius': self.skeleton.joint_radius,
                'bone_radius': self.skeleton.bone_radius,
                'face_mode': self.skeleton.face_mode,
                'face_landmarks': self.skeleton.face_landmarks,
                'face_max_angle': self.skeleton.face_max_angle,
                'eye_style': self.skeleton.eye_style,
                'eye_color': list(self.skeleton.eye_color),
                'pupil_color': list(self.skeleton.pupil_color),
                'pupil_scale': self.skeleton.pupil_scale
            },
            'splat': {
                'overlay_ply': self.splat.overlay_ply,
                'meta_json': self.splat.meta_json,
                'original_image_size': (
                    list(self.splat.original_image_size)
                    if self.splat.original_image_size else None
                ),
                'crop_box': list(self.splat.crop_box) if self.splat.crop_box else None,
                'scale': self.splat.scale,
                'reconcile_intrinsics': self.splat.reconcile_intrinsics,
                'max_angle_deg': self.splat.max_angle_deg,
                'renderer_binary': self.splat.renderer_binary,
                'confidence': self.splat.confidence,
                'cull_color': (
                    list(self.splat.cull_color) if self.splat.cull_color else None
                ),
                'gate_lo': self.splat.gate_lo,
                'gate_hi': self.splat.gate_hi,
                'confidence_sidecar': self.splat.confidence_sidecar,
                'confidence_dataset': self.splat.confidence_dataset,
                'confidence_extra_args': list(self.splat.confidence_extra_args),
            },
            'export': {
                'output_dir': self.export.output_dir,
                'image_format': self.export.image_format,
                'filename_pattern': self.export.filename_pattern,
                'colmap': self.export.colmap,
                'pointcloud_samples': self.export.pointcloud_samples
            }
        }

        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def generate_default_config_template() -> str:
        """
        Generate a default configuration template with comments.

        Returns:
            YAML string with comments explaining each option
        """
        return """# Body2COLMAP Configuration File
#
# This file configures the multi-view rendering and COLMAP export pipeline.
# Command-line arguments override values specified here.

# Input file (required)
# Supported formats:
#   .npz - SAM-3D-Body mesh output (supports mesh, depth, skeleton modes)
#   .ply - Gaussian Splat file (renders splat directly)
input_file: "path/to/input.npz"

# Rendering configuration
render:
  # Output resolution [width, height] in pixels
  resolution: [512, 512]

  # Mesh color [R, G, B] in range 0-1
  mesh_color: [0.65, 0.74, 0.86]

  # Background color [R, G, B] in range 0-1
  bg_color: [1.0, 1.0, 1.0]

  # Render modes (for .npz): mesh, depth, outline, skeleton,
  #   depth+skeleton, outline+skeleton, skeleton+face, depth+skeleton+face
  # With a splat overlay configured below, "splat" is also available as an
  # overlay layer: skeleton+splat, depth+skeleton+splat, ...
  # For .ply files, "splat" mode is automatically used
  modes: ["mesh"]

  # Outline mode: flat two-tone silhouette of the mesh (no shading)
  # Foreground (mesh) color [R, G, B] in range 0-1
  outline_color: [0.0, 0.0, 0.0]

  # Background color [R, G, B] in range 0-1
  outline_bg_color: [1.0, 1.0, 1.0]

  # "filled" = solid silhouette, "stroke" = boundary band only
  outline_style: "filled"

  # Stroke width in pixels (only used when outline_style is "stroke")
  outline_thickness: 3

  # Blur radius in pixels applied to the outline (0 = hard two-tone edges).
  # Does not affect a skeleton overlay in "outline+skeleton" mode.
  outline_blur: 4

# Environment backdrop drawn behind the render.
#
# Purpose: a blank background lets a video diffusion model read an orbit as the
# subject spinning on a turntable. A world-fixed backdrop that sweeps past as
# the camera moves is the cue that says otherwise.
#
# The defaults are a "grid" cube at 3x the orbit radius: walls meeting at
# corners over a floor and ceiling that read apart. That is the arrangement
# that carries the cue most strongly, and the three settings are a set -- a
# cube is only a room at a finite radius.
#
# Caveat worth knowing before picking another texture: a Nishita-style sky is
# azimuthally symmetric apart from its sun, so it barely changes as the camera
# orbits and supplies almost none of the cue you are after. "checker" and
# "grid" carry real azimuthal structure and are the honest test.
#
# The backdrop is for conditioning frames only -- it is never exported to
# COLMAP, adds no points to the point cloud, and does not touch the depth
# buffer or the silhouette mask.
background:
  # Draw a backdrop at all
  enabled: false

  # Surface the texture is mapped onto: "sphere" or "cube".
  # At an infinite radius (see below) these differ only in how the texture is
  # parameterized -- the geometric difference needs a finite radius.
  geometry: "cube"

  # Either a built-in generator:
  #   blender_sky - approximation of Blender's default Sky Texture (Nishita)
  #   grid        - ruled walls with a darker floor and lighter ceiling
  #   checker     - two-tone checker; maximum azimuthal signal, for validation
  #   gradient    - plain vertical gradient; a control with no rotation cue
  # or a path to an image:
  #   sphere - a 2:1 equirectangular image
  #   cube   - a directory of six faces (px/nx/py/ny/pz/nz, or posx/... , or
  #            right/left/top/bottom/front/back), a 4:3 horizontal cross, a
  #            6:1 strip, a 1:6 column, or a 2:1 equirect resampled onto it
  texture: "grid"

  # Generated-texture resolution: equirect height, or cube face size.
  # Ignored for a loaded texture, which keeps its own.
  resolution: 1024

  # Surface radius in world units. Setting it here supersedes the
  # `radius_scale` default below. null on both puts the surface at infinity,
  # where it responds to camera rotation but not to camera translation --
  # correct for a distant sky, but it yields no parallax between subject and
  # backdrop, and a cube at infinity has no corners.
  radius: null

  # Radius as a multiple of the orbit radius, which is what keeps the backdrop
  # sized correctly when the orbit is auto-framed. Mutually exclusive with
  # `radius`; must be > 1.0 so the camera stays inside.
  radius_scale: 3.0

  # Rotate the environment about +Y, in degrees. Aims the sun, or turns a
  # cube's walls relative to the subject.
  rotation_deg: 0.0

  # Force alpha to 255, giving a flat conditioning frame. Set false to keep
  # the silhouette alpha usable as a mask, filling only RGB behind it.
  opaque: true

  # Extra arguments for a generator, e.g. {sun_azimuth_deg: 40, sun_size_deg: 6}
  # for blender_sky, or {n_per_face: 8} for a cube checker. Rejected when
  # `texture` names a file.
  params: {}

  # Fade the backdrop out around the subject.
  #
  # Purpose: the backdrop that fixes one failure causes another. In outline
  # modes a grid running right up to the silhouette reads to a video model as
  # a hard occlusion boundary, so it refuses to paint outside it and bulky
  # clothing or hair gets squashed onto the outline of the bare mesh. Clearing
  # the backdrop in a shell around the subject keeps the rotation cue in the
  # far field and leaves room to expand into.
  #
  # The shell is the projection of an ellipsoid fitted to the mesh, not of one
  # frame's outline, so it covers the silhouette from every viewpoint on the
  # orbit.
  fade:
    # Fade at all
    enabled: false

    # How the backdrop returns as you move away from the subject:
    #   step           - hard cut at the band edge; the control condition
    #   linear         - straight ramp, with a visible slope break
    #   smoothstep     - Hermite ramp, flat at both ends (default)
    #   cosine         - raised cosine; steeper through the middle
    #   exponential    - steepest at the silhouette, long thin tail
    #   gaussian       - flat at the silhouette, then falls away
    #   inverse_square - the heaviest tail; a faint wash over the whole frame
    # The first four reach zero exactly at the band edge; the last three have
    # tails that never quite do.
    profile: "smoothstep"

    # Width of the fade band, as a multiple of the subject's own radius.
    # 1.0 means the backdrop is fully back by twice the subject's extent.
    # Scale-free, so it holds up across an auto-framed orbit.
    falloff: 1.0

    # Shape constant for exponential / gaussian / inverse_square; larger is
    # tighter. Ignored by the other profiles.
    rate: 4.0

    # Inflate the fitted ellipsoid before the fade is measured. Raise it when
    # the mesh is a bare body and the subject you want generated is not.
    margin: 1.0

    # What the backdrop fades to:
    #   local - the backdrop's own colour with its detail averaged away, so
    #           the lines go but the wall/floor/ceiling tone carries through
    #           and the clear zone has no edge against its surroundings
    #   color - one flat colour over the whole clear zone
    target: "local"

    # Flat colour for target: color, as RGB 0-1. null = the texture's mean,
    # which is the one flat colour that leaves the frame's tone unchanged.
    color: null

    # Long-side resolution the backdrop is averaged down to for target: local.
    # Wants to be well below the texture's own frequency.
    detail: 24

# Camera configuration
camera:
  # Focal length in pixels (null = auto-compute for ~47° FOV)
  focal_length: null

  # Auto-frame the mesh to fit in view
  auto_frame: true

  # Fill ratio for auto-framing (0-1, how much of frame mesh should fill)
  fill_ratio: 0.8

  # Manual zoom override (null = use auto_frame)
  zoom: null

# Orbit path configuration
path:
  # Pattern type: circular, sinusoidal, helical
  pattern: "helical"

  # Number of frames to render
  n_frames: 120

  # Orbit radius in meters (null = auto-compute from mesh bounds)
  radius: null

  # Framing preset: full, torso, bust, head
  # - full: Entire body (default)
  # - torso: Waist up (requires skeleton data)
  # - bust: Shoulders and head (requires skeleton data)
  # - head: Head only (requires skeleton data)
  framing: "full"

  # Initial body rotation offset in degrees.
  # The body is auto-rotated to face the camera at frame 0.
  # This value adds additional rotation (0 = face camera, 90 = right side, etc.)
  initial_rotation: 0.0

  # Pin frame 0 to the original SAM-3D-Body camera.
  # When true, the orbit radius, target, and start azimuth are derived from
  # the mesh's position relative to the origin (the original camera). All
  # cameras share the original focal length. Skips auto-orient.
  # Requires 'focal_length' in the .npz file.
  use_original_camera: false

  # Crop mesh to initial viewport
  # When true, removes vertices not visible in the first camera's viewport.
  # Useful when you only want to render partial views and don't want
  # hidden geometry appearing as the camera orbits.
  crop_to_viewport: false

  # Circular mode: base elevation angle in degrees
  elevation_deg: 0.0

  # Sinusoidal mode: amplitude and cycles
  sinusoidal_amplitude_deg: 30.0
  sinusoidal_cycles: 2

  # Helical mode: loops and amplitude
  helical_loops: 3
  helical_amplitude_deg: 30.0
  helical_lead_in_deg: 45.0
  helical_lead_out_deg: 45.0

# Skeleton rendering configuration
skeleton:
  # Enable skeleton rendering
  enabled: false

  # Skeleton format: openpose_body25_hands, mhr70
  format: "openpose_body25_hands"

  # Joint sphere radius in meters
  joint_radius: 0.015

  # Bone cylinder radius in meters
  bone_radius: 0.008

  # Face landmark rendering mode: null (disabled), "full" (points + lines), "points" (points only)
  # Providing face_landmarks automatically sets this to "full" unless overridden
  face_mode: null

  # Path to face landmarks JSON file (from tools/extract_face_landmarks.py)
  # When provided, uses subject-specific face geometry instead of the canonical model
  face_landmarks: null

  # Maximum angle (degrees) off the face normal at which face landmarks are rendered.
  # 90 = full frontal hemisphere (default), 45 = only +/-45 degrees from straight-on.
  face_max_angle: 90.0

  # How to render the eyes: "shape" (a filled eye with a pupil disc) or
  # "dots" (the original OpenPose landmark dots and eye outline).
  eye_style: "shape"

  # Color of the eye shape (sclera) and of the pupil, as RGB floats 0-1.
  # Both apply to eye_style "shape" only.
  eye_color: [1.0, 1.0, 1.0]
  pupil_color: [0.0, 0.0, 0.0]

  # Pupil diameter as a fraction of the eye height, in (0, 1].
  # 1.0 = a disc touching the upper and lower lid.
  pupil_scale: 0.75

# Gaussian-splat overlay configuration
# Composites a splat built externally from the SAME photo that fed
# SAM-3D-Body on top of the skeleton, giving a conditioning frame with the
# real face instead of synthetic landmarks. Use with modes: ["skeleton+splat"].
splat:
  # 3DGS .ply. null disables the overlay.
  overlay_ply: null

  # Its metadata (fitted intrinsics + centroid).
  # null = splat_meta.json beside the .ply
  meta_json: null

  # Size [width, height] of the full photo SAM-3D-Body saw.
  # null = taken from img_shape in the .npz
  original_image_size: null

  # If the splat was built from a crop of that photo (recommended for a face:
  # the upstream models run at 1024x768, so cropping to the head spends that
  # budget on the face), the crop region in full-image pixels [x0, y0, x1, y1].
  # null = the splat used the whole photo.
  crop_box: null

  # Depth gauge: how far along the view rays the splat sits. It does not affect
  # the anchor frame at all — only how well the splat and skeleton stay
  # together as the orbit turns away. null fits it against the mesh.
  scale: null

  # Correct for the splat's own fitted focal length disagreeing with
  # SAM-3D-Body's. Leave true unless you trust the splat's focal more.
  reconcile_intrinsics: true

  # Drop the splat on frames more than this many degrees off its source view.
  # It is a 2.5-D shell with nothing behind the subject.
  max_angle_deg: 45.0

  # Path to the brush-splat-render binary that rasterizes splats.
  # null = $BRUSH_SPLAT_RENDER, then PATH.
  renderer_binary: null

  # Gate each pixel by per-splat multi-view confidence instead of leaving it to
  # a downstream alpha threshold. NOTE: this makes the alpha channel the gate,
  # not accumulated opacity. Only for a .ply input -- an overlay splat is built
  # from one photo and has no training views to score against.
  confidence: false

  # What culled pixels resolve to when confidence is on. The renderer uses one
  # colour for both this and the background composited under the splat, so it
  # is the whole background of a gated render. null = follow render.bg_color;
  # set it only to make culled regions stand out for inspection.
  cull_color: null

  # Confidence at or below which a pixel is fully culled / at or above which it
  # is fully kept. Equal values give a hard cut.
  gate_lo: 0.45
  gate_hi: 0.65

  # Also write <frame>.conf.png: the raw confidence, before the gate.
  confidence_sidecar: false

  # Training dataset to measure evidence against when the .ply carries no ev_*
  # block (i.e. was not trained with brush --export-evidence).
  confidence_dataset: null

  # Verbatim passthrough for brush-splat-render's tuning flags
  # (--conf-tau, --conf-min-views, --conf-facing, ...).
  confidence_extra_args: []

# Export configuration
export:
  # Output directory for rendered images and COLMAP files
  output_dir: "./output"

  # Image format: png, jpg
  image_format: "png"

  # Filename pattern (Python format string)
  filename_pattern: "frame_{:04d}.png"

  # Export COLMAP sparse reconstruction
  colmap: true

  # Number of points to sample from mesh surface for COLMAP points3D.txt
  pointcloud_samples: 50000
"""


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create command-line argument parser.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="body2colmap",
        description="Generate multi-view training data for Gaussian Splatting from SAM-3D-Body output (.npz) or re-render existing Gaussian Splats (.ply)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Configuration file (YAML) can be used to set all options. Command-line arguments override config file values."
    )

    # Config file
    parser.add_argument(
        "--config", "-c",
        help="Path to YAML configuration file"
    )

    # Input/output
    parser.add_argument(
        "input",
        nargs='?',
        help="Path to input file: .npz (SAM-3D-Body mesh) or .ply (Gaussian Splat)"
    )

    parser.add_argument(
        "--output-dir", "-o",
        help="Directory for rendered frames and COLMAP files"
    )

    # Generate default config
    parser.add_argument(
        "--save-config",
        metavar="PATH",
        help="Save default configuration to YAML file and exit"
    )

    # Rendering options
    render_group = parser.add_argument_group("Rendering Options")
    render_group.add_argument(
        "--resolution",
        type=str,
        metavar="WxH",
        help="Render resolution (e.g., 512x512, 1024x768)"
    )
    render_group.add_argument(
        "--width",
        type=int,
        metavar="PIXELS",
        help="Render width in pixels (alternative to --resolution)"
    )
    render_group.add_argument(
        "--height",
        type=int,
        metavar="PIXELS",
        help="Render height in pixels (alternative to --resolution)"
    )
    render_group.add_argument(
        "--render-modes",
        type=str,
        metavar="MODE[,MODE...]",
        help="Comma-separated render modes: mesh, depth, outline, skeleton, "
             "depth+skeleton, outline+skeleton, skeleton+face, "
             "depth+skeleton+face (for .npz). With --splat-overlay, 'splat' is "
             "also available as an overlay layer, e.g. skeleton+splat. For a "
             ".ply input, splat mode is auto-selected"
    )
    render_group.add_argument(
        "--mesh-color",
        type=str,
        metavar="R,G,B",
        help="Mesh color as RGB floats 0-1 (e.g., 0.65,0.74,0.86)"
    )
    render_group.add_argument(
        "--bg-color",
        type=str,
        metavar="R,G,B",
        help="Background color as RGB floats 0-1 (e.g., 1.0,1.0,1.0)"
    )
    render_group.add_argument(
        "--outline-color",
        type=str,
        metavar="R,G,B",
        help="Outline foreground (mesh) color as RGB floats 0-1 (default: 0,0,0)"
    )
    render_group.add_argument(
        "--outline-bg-color",
        type=str,
        metavar="R,G,B",
        help="Outline background color as RGB floats 0-1 (default: 1,1,1)"
    )
    render_group.add_argument(
        "--outline-style",
        type=str,
        choices=["filled", "stroke"],
        help="Outline style: filled silhouette or boundary stroke (default: filled)"
    )
    render_group.add_argument(
        "--outline-thickness",
        type=int,
        metavar="PIXELS",
        help="Outline stroke width in pixels (only for --outline-style stroke)"
    )
    render_group.add_argument(
        "--outline-blur",
        type=int,
        metavar="PIXELS",
        help="Blur radius in pixels for the outline, 0 to disable (default: 4). "
             "Does not blur a skeleton overlay."
    )

    # Background options
    bg_group = parser.add_argument_group(
        "Background Options",
        "A world-fixed backdrop, so an orbit reads as the camera moving rather "
        "than the subject spinning. The defaults -- a grid cube at 3x the "
        "orbit radius -- are the arrangement that carries that cue most "
        "strongly: walls meeting at corners over a floor and ceiling that read "
        "apart. Note that a Nishita-style sky is azimuthally symmetric apart "
        "from its sun and so carries almost none of it."
    )
    bg_group.add_argument(
        "--background",
        type=str,
        metavar="TEXTURE",
        help="Enable a backdrop with this texture: a built-in generator "
             "(grid, checker, blender_sky, gradient) or a path to an "
             "equirectangular image, packed cubemap, or directory of six "
             "cube faces. Pass 'grid' for the default backdrop"
    )
    bg_group.add_argument(
        "--no-background",
        action="store_true",
        help="Disable the backdrop, overriding a config file that enables it"
    )
    bg_group.add_argument(
        "--background-geometry",
        type=str,
        choices=["sphere", "cube"],
        help="Surface the texture is mapped onto (default: cube). Only "
             "differs geometrically with a finite radius, which is on by "
             "default -- see --background-radius-scale"
    )
    bg_group.add_argument(
        "--background-resolution",
        type=int,
        metavar="PIXELS",
        help="Generated texture resolution: equirect height or cube face size "
             "(default: 1024). Ignored for a loaded texture"
    )
    bg_radius = bg_group.add_mutually_exclusive_group()
    bg_radius.add_argument(
        "--background-radius",
        type=float,
        metavar="UNITS",
        help="Backdrop radius in world units, replacing the default "
             "--background-radius-scale"
    )
    bg_radius.add_argument(
        "--background-radius-scale",
        type=float,
        metavar="FACTOR",
        help="Backdrop radius as a multiple of the orbit radius (must be > 1, "
             "default: 3.0). Sized against the orbit, so it holds up when the "
             "orbit is auto-framed"
    )
    bg_radius.add_argument(
        "--background-infinite",
        action="store_true",
        help="Put the backdrop at infinity instead of the default finite "
             "radius. It then tracks camera rotation but not translation, so "
             "there is no parallax and a cube loses its corners -- right for a "
             "distant sky"
    )
    bg_group.add_argument(
        "--background-rotation",
        type=float,
        metavar="DEGREES",
        help="Rotate the environment about +Y, e.g. to aim the sun"
    )
    bg_group.add_argument(
        "--background-keep-alpha",
        action="store_true",
        help="Fill only RGB behind the subject, leaving the silhouette alpha "
             "intact as a mask. Default is to force alpha opaque"
    )

    # Subject fade
    fade_group = parser.add_argument_group(
        "Background Fade Options",
        "Fade the backdrop out around the subject. In outline modes a grid "
        "running right up to the silhouette reads to a video model as a hard "
        "occlusion boundary, so it will not paint bulky clothing or hair "
        "outside the bare mesh's outline. The clear zone is the projection of "
        "an ellipsoid fitted to the mesh, so it covers the silhouette from "
        "every viewpoint on the orbit."
    )
    fade_group.add_argument(
        "--background-fade",
        type=str,
        metavar="PROFILE",
        choices=sorted(DECAY_PROFILES),
        help="Enable the fade with this decay profile: "
             + ", ".join(sorted(DECAY_PROFILES))
             + ". 'step' is the hard-edged control; 'smoothstep' is the "
               "default shape"
    )
    fade_group.add_argument(
        "--no-background-fade",
        action="store_true",
        help="Disable the fade, overriding a config file that enables it"
    )
    fade_group.add_argument(
        "--background-fade-falloff",
        type=float,
        metavar="FACTOR",
        help=f"Width of the fade band, as a multiple of the subject's own "
             f"radius (default: {DEFAULT_FALLOFF:g}). Scale-free, so it holds "
             f"up across an auto-framed orbit"
    )
    fade_group.add_argument(
        "--background-fade-rate",
        type=float,
        metavar="K",
        help=f"Shape constant for the exponential, gaussian and "
             f"inverse_square profiles; larger is tighter (default: "
             f"{DEFAULT_RATE:g}). Ignored by the others"
    )
    fade_group.add_argument(
        "--background-fade-margin",
        type=float,
        metavar="FACTOR",
        help=f"Inflate the fitted ellipsoid before the fade is measured "
             f"(default: {DEFAULT_MARGIN:g}). Raise it when the mesh is a "
             f"bare body and the subject is not"
    )
    fade_group.add_argument(
        "--background-fade-target",
        type=str,
        choices=sorted(FADE_TARGETS),
        help="What the backdrop fades to: 'local' (default) averages its own "
             "detail away, so the lines go but the wall/floor/ceiling tone "
             "carries through; 'color' uses one flat colour"
    )
    fade_group.add_argument(
        "--background-fade-color",
        type=str,
        metavar="R,G,B",
        help="Flat fade colour as RGB floats 0-1 (default: the texture's "
             "mean). Implies --background-fade-target color"
    )
    fade_group.add_argument(
        "--background-fade-detail",
        type=int,
        metavar="PIXELS",
        help=f"Long-side resolution the backdrop is averaged down to for the "
             f"'local' target (default: {DEFAULT_DETAIL}). Wants to be well "
             f"below the texture's own frequency"
    )

    # Camera options
    camera_group = parser.add_argument_group("Camera Options")
    camera_group.add_argument(
        "--focal-length",
        type=float,
        help="Focal length in pixels (default: auto for ~47° FOV)"
    )
    camera_group.add_argument(
        "--fill-ratio",
        type=float,
        metavar="RATIO",
        help="How much of the viewport the scene should fill (0.0-1.0, default: 0.8)"
    )
    camera_group.add_argument(
        "--n-frames",
        type=int,
        help="Number of frames in orbit (default: 120)"
    )

    # Path options
    path_group = parser.add_argument_group("Orbit Path Options")
    path_group.add_argument(
        "--orbit-pattern",
        choices=["circular", "sinusoidal", "helical"],
        help="Orbit pattern"
    )
    path_group.add_argument(
        "--orbit-radius",
        type=float,
        metavar="METERS",
        help="Orbit radius in meters (default: auto-computed)"
    )
    path_group.add_argument(
        "--elevation",
        type=float,
        metavar="DEGREES",
        help="Base elevation angle in degrees (circular mode)"
    )
    path_group.add_argument(
        "--helical-loops",
        type=int,
        metavar="N",
        help="Number of full rotations (helical mode)"
    )
    path_group.add_argument(
        "--amplitude",
        type=float,
        metavar="DEGREES",
        help="Elevation amplitude in degrees (helical/sinusoidal)"
    )
    path_group.add_argument(
        "--framing",
        choices=["full", "torso", "bust", "head"],
        help="Body framing preset: full (entire body), torso (waist up), "
             "bust (shoulders and head), head (head only). "
             "Non-full presets require skeleton data."
    )
    path_group.add_argument(
        "--crop-to-viewport",
        action="store_true",
        help="Filter mesh to keep only vertices visible in the first camera's "
             "viewport. Vertices outside the initial view won't appear when "
             "the camera orbits around."
    )
    path_group.add_argument(
        "--initial-rotation",
        type=float,
        metavar="DEGREES",
        help="Additional rotation (degrees) after auto-facing the body toward "
             "the camera. 0 = face camera (default), 90 = right side, etc."
    )
    path_group.add_argument(
        "--use-original-camera",
        action="store_true",
        help="Pin frame 0 to the original SAM-3D-Body camera. The orbit "
             "radius, target, and start azimuth are derived from the mesh's "
             "position so the path smoothly continues from the original "
             "viewpoint. All cameras share the original focal length. "
             "Skips auto-orient. Requires 'focal_length' in the .npz file."
    )

    # Skeleton options
    skeleton_group = parser.add_argument_group("Skeleton Options")
    skeleton_group.add_argument(
        "--skeleton",
        action="store_true",
        help="Enable skeleton rendering"
    )
    skeleton_group.add_argument(
        "--skeleton-format",
        choices=["openpose_body25_hands", "mhr70"],
        help="Skeleton format for rendering"
    )
    skeleton_group.add_argument(
        "--joint-radius",
        type=float,
        metavar="METERS",
        help="Joint sphere radius in meters"
    )
    skeleton_group.add_argument(
        "--bone-radius",
        type=float,
        metavar="METERS",
        help="Bone cylinder radius in meters"
    )
    skeleton_group.add_argument(
        "--face-mode",
        choices=["full", "points", "none"],
        help="Face landmark rendering mode: full (points + connectivity), "
             "points (points only), none (disabled). "
             "Requires skeleton data. Rendered only for frontal views."
    )
    skeleton_group.add_argument(
        "--face-landmarks",
        metavar="PATH",
        help="Path to face landmarks JSON file (from extract_face_landmarks.py). "
             "When provided, uses subject-specific face geometry instead of "
             "the generic canonical face model. Implies --face-mode full."
    )
    skeleton_group.add_argument(
        "--eye-style",
        choices=["shape", "dots"],
        help="How to render the eyes: shape (filled eye with a pupil disc, "
             "default), dots (the original OpenPose landmark dots and eye "
             "outline). Colors and pupil size apply to 'shape' only."
    )
    skeleton_group.add_argument(
        "--eye-color",
        type=str,
        metavar="R,G,B",
        help="Color of the filled eye shape (sclera) as RGB floats 0-1 "
             "(default: 1,1,1)"
    )
    skeleton_group.add_argument(
        "--pupil-color",
        type=str,
        metavar="R,G,B",
        help="Color of the pupil disc as RGB floats 0-1 (default: 0,0,0)"
    )
    skeleton_group.add_argument(
        "--pupil-scale",
        type=float,
        metavar="SCALE",
        help="Pupil diameter as a fraction of the eye height, in (0, 1]. "
             "1.0 = a disc touching the upper and lower lid (default: 0.75)"
    )
    skeleton_group.add_argument(
        "--face-max-angle",
        type=float,
        metavar="DEGREES",
        help="Maximum angle (degrees) off the face normal at which face "
             "landmarks are rendered. 90 = full frontal hemisphere (default), "
             "45 = only within 45 degrees of straight-on."
    )

    # Export options
    # Gaussian-splat overlay options
    splat_group = parser.add_argument_group("Splat Overlay Options")
    splat_group.add_argument(
        "--splat-overlay",
        type=str,
        metavar="PLY",
        help="3DGS .ply built from the same photo as the .npz, composited on "
             "top of the skeleton. Enables the 'splat' overlay layer, e.g. "
             "--render-modes skeleton+splat"
    )
    splat_group.add_argument(
        "--splat-meta",
        type=str,
        metavar="JSON",
        help="Splat metadata (intrinsics, centroid). "
             "Default: splat_meta.json beside the .ply"
    )
    splat_group.add_argument(
        "--splat-crop",
        type=str,
        metavar="X0,Y0,X1,Y1",
        help="Region of the original photo the splat's input image was cut "
             "from, in full-image pixels. Omit if the splat used the whole photo"
    )
    splat_group.add_argument(
        "--splat-image-size",
        type=str,
        metavar="WxH",
        help="Size of the full original photo SAM-3D-Body saw. "
             "Default: img_shape from the .npz"
    )
    splat_group.add_argument(
        "--splat-scale",
        type=float,
        metavar="S",
        help="Depth gauge for the splat. Omit to fit it against the mesh"
    )
    splat_group.add_argument(
        "--splat-no-reconcile",
        action="store_true",
        help="Ignore the splat's own fitted intrinsics and place it with a "
             "uniform scale. Preserves its shape but can mis-size it"
    )
    splat_group.add_argument(
        "--splat-max-angle",
        type=float,
        metavar="DEGREES",
        help="Cull the splat past this many degrees off its source view "
             "(default 45). It is a 2.5-D shell with nothing behind it, so "
             "past roughly 45 deg its open edge flares into view"
    )
    splat_group.add_argument(
        "--splat-renderer",
        type=str,
        metavar="PATH",
        help="Path to the brush-splat-render binary that rasterizes splats. "
             "Default: $BRUSH_SPLAT_RENDER, then PATH"
    )
    splat_group.add_argument(
        "--splat-confidence",
        action="store_true",
        help="Gate each pixel by per-splat multi-view confidence instead of "
             "leaving it to a downstream alpha threshold. NOTE: makes the "
             "alpha channel the gate, not accumulated opacity. Needs a .ply "
             "input carrying ev_* properties (brush --export-evidence) or "
             "--splat-confidence-dataset; not available for --splat-overlay"
    )
    splat_group.add_argument(
        "--splat-cull-color",
        type=str,
        metavar="R,G,B",
        help="Colour culled pixels resolve to, 0-1. This is also the whole "
             "background of a confidence render, so it defaults to --bg-color; "
             "set it only to make culled regions stand out"
    )
    splat_group.add_argument(
        "--splat-gate-lo",
        type=float,
        metavar="C",
        help="Confidence at or below which a pixel is fully culled (default "
             "0.45). Set equal to --splat-gate-hi for a hard cut"
    )
    splat_group.add_argument(
        "--splat-gate-hi",
        type=float,
        metavar="C",
        help="Confidence at or above which a pixel is fully kept (default 0.65)"
    )
    splat_group.add_argument(
        "--splat-confidence-sidecar",
        action="store_true",
        help="Also write the raw per-pixel confidence beside each frame as "
             "<frame>.conf.png, before the gate thresholds are applied"
    )
    splat_group.add_argument(
        "--splat-confidence-dataset",
        type=str,
        metavar="DIR",
        help="Training dataset (COLMAP / nerfstudio) to measure evidence "
             "against when the .ply carries no ev_* block"
    )

    export_group = parser.add_argument_group("Export Options")
    export_group.add_argument(
        "--no-colmap",
        action="store_true",
        help="Skip COLMAP export"
    )
    export_group.add_argument(
        "--pointcloud-samples",
        type=int,
        metavar="N",
        help="Number of points to sample on mesh surface for COLMAP"
    )
    export_group.add_argument(
        "--filename-pattern",
        metavar="PATTERN",
        help="Output filename pattern (Python format string, e.g., 'frame_{:04d}.png')"
    )

    # Debug options
    debug_group = parser.add_argument_group("Debug Options")
    debug_group.add_argument(
        "--debug-original-view",
        action="store_true",
        help="Render a single frame from the original SAM-3D-Body camera viewpoint "
             "using the focal length stored in the .npz file. Use this to verify "
             "mesh/skeleton alignment with the original input image. "
             "Skips orbit generation and COLMAP export. "
             "Requires 'focal_length' to be present in the .npz file. "
             "Auto-frames by default; use --no-auto-frame to disable."
    )
    debug_group.add_argument(
        "--no-auto-frame",
        action="store_true",
        help="With --debug-original-view: render at exact original camera parameters "
             "without auto-framing. By default, the subject is zoomed and centered "
             "to fill the frame, and a framing.json with the affine transform is saved."
    )
    debug_group.add_argument(
        "--original-image",
        metavar="PATH",
        help="Path to the original input image (the SAM-3D-Body crop). "
             "Used with --debug-original-view to produce a warped version of "
             "the original that matches the auto-framed render, plus overlay "
             "composites for visual alignment verification."
    )

    # Other options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    return parser
