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


@dataclass
class RenderConfig:
    """Rendering configuration."""
    resolution: Tuple[int, int] = (512, 512)
    mesh_color: Tuple[float, float, float] = (0.65, 0.74, 0.86)
    bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    modes: List[str] = field(default_factory=lambda: ["mesh"])


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
    camera_height: str = "bbox_center"  # skeleton-based camera orbit height
    look_at_height: str = "bbox_center"  # skeleton-based look-at target height
    crop_to_viewport: bool = False  # Filter mesh to first camera's viewport

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

        # Camera overrides
        if args.focal_length is not None:
            config.camera.focal_length = args.focal_length

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
        if args.camera_height:
            config.path.camera_height = args.camera_height
        if args.look_at_height:
            config.path.look_at_height = args.look_at_height
        if args.crop_to_viewport:
            config.path.crop_to_viewport = True
        if args.initial_rotation is not None:
            config.path.initial_rotation = args.initial_rotation

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
            modes=render_data.get('modes', ['mesh'])
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
            camera_height=path_data.get('camera_height', 'bbox_center'),
            look_at_height=path_data.get('look_at_height', 'bbox_center'),
            crop_to_viewport=path_data.get('crop_to_viewport', False),
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
            face_max_angle=skeleton_data.get('face_max_angle', 90.0)
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
                'modes': self.render.modes
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
                'camera_height': self.path.camera_height,
                'look_at_height': self.path.look_at_height,
                'crop_to_viewport': self.path.crop_to_viewport,
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
                'face_max_angle': self.skeleton.face_max_angle
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

  # Render modes (for .npz): mesh, depth, skeleton, depth+skeleton, skeleton+face, depth+skeleton+face
  # For .ply files, "splat" mode is automatically used
  modes: ["mesh"]

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

  # Camera orbit height: determines the Y position of the orbit center.
  # Uses skeleton joint positions for body-part presets.
  # - bbox_center: Center of bounding box (default, no skeleton needed)
  # - feet: Ankle height
  # - knees: Knee height
  # - waist: Hip height
  # - chest: Midpoint between hips and shoulders
  # - shoulders: Shoulder height
  # - head: Nose height
  camera_height: "bbox_center"

  # Look-at target height: determines the Y position the cameras look at.
  # Same presets as camera_height. Independent of camera_height.
  # Example: camera_height=feet + look_at_height=head = low-angle upward shot
  look_at_height: "bbox_center"

  # Initial body rotation offset in degrees.
  # The body is auto-rotated to face the camera at frame 0.
  # This value adds additional rotation (0 = face camera, 90 = right side, etc.)
  initial_rotation: 0.0

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
        help="Comma-separated render modes: mesh, depth, skeleton, depth+skeleton, skeleton+face, depth+skeleton+face (for .npz); splat mode auto-selected for .ply"
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

    # Camera options
    camera_group = parser.add_argument_group("Camera Options")
    camera_group.add_argument(
        "--focal-length",
        type=float,
        help="Focal length in pixels (default: auto for ~47° FOV)"
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
        "--camera-height",
        choices=["bbox_center", "feet", "knees", "waist", "chest", "shoulders", "head"],
        help="Camera orbit height relative to body: bbox_center (default), "
             "feet, knees, waist, chest, shoulders, head. "
             "Non-bbox_center presets use skeleton joint positions."
    )
    path_group.add_argument(
        "--look-at-height",
        choices=["bbox_center", "feet", "knees", "waist", "chest", "shoulders", "head"],
        help="Look-at target height relative to body: bbox_center (default), "
             "feet, knees, waist, chest, shoulders, head. "
             "Non-bbox_center presets use skeleton joint positions."
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
        "--face-max-angle",
        type=float,
        metavar="DEGREES",
        help="Maximum angle (degrees) off the face normal at which face "
             "landmarks are rendered. 90 = full frontal hemisphere (default), "
             "45 = only within 45 degrees of straight-on."
    )

    # Export options
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

    # Other options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    return parser
