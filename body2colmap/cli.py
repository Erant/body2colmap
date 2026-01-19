"""
Command-line interface for body2colmap.

This module provides the main entry point for the CLI tool.
"""

import sys
from pathlib import Path
from typing import Optional

from .config import create_argument_parser, Config
from .pipeline import OrbitPipeline


def main(argv: Optional[list] = None) -> int:
    """
    Main entry point for CLI.

    Args:
        argv: Command-line arguments (None = sys.argv)

    Returns:
        Exit code (0 = success, non-zero = error)
    """
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args(argv)

    # Create config
    config = Config.from_args(args)

    # Print banner
    if args.verbose:
        print("=" * 60)
        print("Body2COLMAP - Multi-view Synthetic Data Generator")
        print("=" * 60)
        print(f"Input: {config.input_file}")
        print(f"Output: {config.export.output_dir}")
        print(f"Resolution: {config.render.resolution[0]}x{config.render.resolution[1]}")
        print(f"Frames: {config.path.n_frames}")
        print(f"Orbit: {config.path.pattern}")
        print("=" * 60)

    try:
        # Load scene
        if args.verbose:
            print("\n[1/4] Loading scene...")

        pipeline = OrbitPipeline.from_npz_file(
            config.input_file,
            render_size=config.render.resolution,
            include_skeleton=config.skeleton.enabled
        )

        if args.verbose:
            print(f"  Loaded: {pipeline.scene}")

        # Generate orbit
        if args.verbose:
            print("\n[2/4] Generating orbit path...")

        # Build orbit kwargs based on pattern
        orbit_kwargs = {
            'fill_ratio': config.camera.fill_ratio
        }

        if config.path.pattern == "circular":
            orbit_kwargs['elevation_deg'] = config.path.elevation_deg
        elif config.path.pattern == "sinusoidal":
            orbit_kwargs['amplitude_deg'] = config.path.sinusoidal_amplitude_deg
            orbit_kwargs['n_cycles'] = config.path.sinusoidal_cycles
        elif config.path.pattern == "helical":
            orbit_kwargs['n_loops'] = config.path.helical_loops
            orbit_kwargs['amplitude_deg'] = config.path.helical_amplitude_deg
            orbit_kwargs['lead_in_deg'] = config.path.helical_lead_in_deg
            orbit_kwargs['lead_out_deg'] = config.path.helical_lead_out_deg

        pipeline.set_orbit_params(
            pattern=config.path.pattern,
            n_frames=config.path.n_frames,
            radius=config.path.radius,
            **orbit_kwargs
        )

        if args.verbose:
            print(f"  Generated {len(pipeline.cameras)} camera positions")

        # Render
        if args.verbose:
            print("\n[3/4] Rendering frames...")
            print(f"  Mode: {config.render.modes[0]}")

        rendered = pipeline.render_all(
            modes=config.render.modes,
            mesh_color=config.render.mesh_color,
            bg_color=config.render.bg_color
        )

        if args.verbose:
            for mode, images in rendered.items():
                print(f"  Rendered {len(images)} {mode} frames")

        # Export
        if args.verbose:
            print("\n[4/4] Exporting...")

        # Export COLMAP
        if config.export.colmap:
            colmap_dir = pipeline.export_colmap(
                config.export.output_dir,
                n_pointcloud_samples=config.export.pointcloud_samples
            )
            if args.verbose:
                print(f"  COLMAP files → {colmap_dir}")

        # Export images
        for mode, images in rendered.items():
            saved_paths = pipeline.export_images(
                config.export.output_dir,
                images,
                filename_pattern=config.export.filename_pattern
            )
            if args.verbose:
                print(f"  {mode} images ({len(saved_paths)}) → {config.export.output_dir}")

        # Success
        if args.verbose:
            print("\n" + "=" * 60)
            print("✓ Complete!")
            print("=" * 60)

        return 0

    except FileNotFoundError as e:
        print(f"Error: File not found: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: Invalid configuration: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
