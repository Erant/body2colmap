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
