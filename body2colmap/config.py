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
    format: str = "mhr70"
    joint_radius: float = 0.015
    bone_radius: float = 0.008


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

        Args:
            args: Parsed arguments from argparse

        Returns:
            Config instance
        """
        # Build config from args
        # TODO: Implement full mapping from args to config
        config = cls(
            input_file=args.input,
            render=RenderConfig(
                resolution=(args.width, args.height),
                modes=[args.mode] if hasattr(args, 'mode') else ["mesh"]
            ),
            camera=CameraConfig(
                focal_length=getattr(args, 'focal_length', None)
            ),
            path=PathConfig(
                pattern=getattr(args, 'orbit_mode', 'helical'),
                n_frames=getattr(args, 'n_frames', 120)
            ),
            export=ExportConfig(
                output_dir=args.output_dir
            )
        )

        return config

    @classmethod
    def from_yaml(cls, filepath: str) -> "Config":
        """
        Load config from YAML file.

        Args:
            filepath: Path to YAML config file

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
            data = yaml.safe_load(f)

        # TODO: Implement full YAML parsing
        # For now, just raise NotImplementedError
        raise NotImplementedError("YAML config loading not yet implemented")


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create command-line argument parser.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Generate multi-view training data for Gaussian Splatting from SAM-3D-Body output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to .npz file from SAM-3D-Body"
    )

    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Directory for rendered frames and COLMAP files"
    )

    # Rendering options
    render_group = parser.add_argument_group("Rendering Options")
    render_group.add_argument(
        "--width",
        type=int,
        default=512,
        help="Render width in pixels"
    )
    render_group.add_argument(
        "--height",
        type=int,
        default=512,
        help="Render height in pixels"
    )
    render_group.add_argument(
        "--mode",
        choices=["mesh", "depth", "skeleton"],
        default="mesh",
        help="Render mode"
    )

    # Camera options
    camera_group = parser.add_argument_group("Camera Options")
    camera_group.add_argument(
        "--focal-length",
        type=float,
        default=None,
        help="Focal length in pixels (default: auto for ~47° FOV)"
    )
    camera_group.add_argument(
        "--n-frames",
        type=int,
        default=120,
        help="Number of frames in orbit"
    )

    # Path options
    path_group = parser.add_argument_group("Orbit Path Options")
    path_group.add_argument(
        "--orbit-mode",
        choices=["circular", "sinusoidal", "helical"],
        default="helical",
        help="Orbit pattern"
    )
    path_group.add_argument(
        "--elevation",
        type=float,
        default=0.0,
        dest="elevation_deg",
        help="Base elevation angle in degrees (circular mode)"
    )
    path_group.add_argument(
        "--helical-loops",
        type=int,
        default=3,
        help="Number of full rotations (helical mode)"
    )
    path_group.add_argument(
        "--amplitude",
        type=float,
        default=30.0,
        help="Elevation amplitude in degrees (helical/sinusoidal)"
    )

    # Export options
    export_group = parser.add_argument_group("Export Options")
    export_group.add_argument(
        "--pointcloud-samples",
        type=int,
        default=50000,
        help="Number of points to sample on mesh surface"
    )
    export_group.add_argument(
        "--filename-pattern",
        default="frame_{:04d}.png",
        help="Output filename pattern (Python format string)"
    )

    # Other options
    parser.add_argument(
        "--skeleton",
        action="store_true",
        help="Include skeleton rendering"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    return parser
