"""
Command-line interface for body2colmap.

This module provides the main entry point for the CLI tool.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from .config import create_argument_parser, Config
from .face import FaceLandmarkIngest
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

    # Handle --save-config
    if args.save_config:
        template = Config.generate_default_config_template()
        with open(args.save_config, 'w') as f:
            f.write(template)
        print(f"Default configuration saved to: {args.save_config}")
        print(f"Edit this file and use with: body2colmap --config {args.save_config}")
        return 0

    # Create config
    try:
        config = Config.from_args(args)
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        print(f"\nUse --help for usage information or --save-config to generate a template.", file=sys.stderr)
        return 1

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")

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
        # Load scene - detect file type from extension
        if args.verbose:
            print("\n[1/4] Loading scene...")

        input_path = Path(config.input_file)
        input_ext = input_path.suffix.lower()

        if input_ext == '.ply':
            # Gaussian Splat PLY file
            pipeline = OrbitPipeline.from_ply_file(
                config.input_file,
                render_size=config.render.resolution
            )
            is_splat = True
        elif input_ext == '.npz':
            # SAM-3D-Body NPZ file
            # Always load skeleton: needed for auto-orient (torso facing
            # direction), framing presets, and skeleton/face rendering.
            needs_skeleton = True
            pipeline = OrbitPipeline.from_npz_file(
                config.input_file,
                render_size=config.render.resolution,
                include_skeleton=needs_skeleton
            )
            is_splat = False
        else:
            raise ValueError(f"Unsupported input file type: {input_ext}. Use .npz or .ply")

        if args.verbose:
            print(f"  Loaded: {pipeline.scene}")

        # Load face landmarks if provided
        face_landmarks_70 = None
        if config.skeleton.face_landmarks:
            if args.verbose:
                print(f"  Loading face landmarks: {config.skeleton.face_landmarks}")
            face_landmarks_70 = FaceLandmarkIngest.from_json(config.skeleton.face_landmarks)
            if args.verbose:
                print(f"  Converted to OpenPose Face 70 ({face_landmarks_70.shape})")

        # Auto-orient: rotate body to face camera, plus any user offset
        if not is_splat:
            pipeline.auto_orient(rotation_offset_deg=config.path.initial_rotation)
            if args.verbose:
                print(f"  Auto-oriented body (offset: {config.path.initial_rotation}°)")

        # Generate orbit
        if args.verbose:
            print("\n[2/4] Generating orbit path...")

        # Build orbit kwargs based on pattern
        orbit_kwargs = {
            'fill_ratio': config.camera.fill_ratio,
            'framing': config.path.framing
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
            camera_height=config.path.camera_height,
            look_at_height=config.path.look_at_height,
            **orbit_kwargs
        )

        if args.verbose:
            print(f"  Generated {len(pipeline.cameras)} camera positions")

        # Apply viewport cropping if requested
        if config.path.crop_to_viewport and pipeline.cameras:
            if args.verbose:
                original_verts = len(pipeline.scene.vertices)
            pipeline.scene = pipeline.scene.filter_mesh_to_viewport(pipeline.cameras[0])
            if args.verbose:
                new_verts = len(pipeline.scene.vertices)
                print(f"  Cropped mesh to viewport: {original_verts} -> {new_verts} vertices")

        # Render
        if args.verbose:
            print("\n[3/4] Rendering frames...")

        rendered = {}

        if is_splat:
            # Splat rendering - only "splat" mode supported
            if args.verbose:
                print("  Mode: splat")

            mode_rendered = pipeline.render_all(
                modes=["splat"],
                bg_color=config.render.bg_color
            )
            rendered.update(mode_rendered)

            if args.verbose:
                print(f"  Rendered {len(mode_rendered.get('splat', []))} splat frames")
        else:
            # Mesh rendering - supports mesh, depth, skeleton, composites
            if args.verbose:
                print(f"  Modes: {', '.join(config.render.modes)}")

            # Parse render modes and handle composites
            for mode_str in config.render.modes:
                if '+' in mode_str:
                    # Composite mode (e.g., "mesh+skeleton" or "depth+skeleton")
                    parts = mode_str.split('+')
                    base_mode = parts[0].strip()
                    overlay_modes = [p.strip() for p in parts[1:]]

                    # Build composite rendering configuration
                    composite_modes = {base_mode: {}}

                    if base_mode == "mesh":
                        composite_modes[base_mode]["color"] = config.render.mesh_color
                        composite_modes[base_mode]["bg_color"] = config.render.bg_color

                    # Add overlays
                    for overlay in overlay_modes:
                        if overlay == "skeleton":
                            composite_modes["skeleton"] = {
                                "joint_radius": config.skeleton.joint_radius,
                                "bone_radius": config.skeleton.bone_radius,
                                "use_openpose_colors": True,
                                "target_format": config.skeleton.format
                            }
                        if overlay == "face" or (overlay == "skeleton" and config.skeleton.face_mode):
                            face_opts = {
                                "face_mode": config.skeleton.face_mode or "full",
                                "face_max_angle": config.skeleton.face_max_angle,
                            }
                            if face_landmarks_70 is not None:
                                face_opts["face_landmarks"] = face_landmarks_70
                            composite_modes["face"] = face_opts

                    # Render composite for all frames
                    mode_images = pipeline.render_composite_all(composite_modes)
                    rendered[mode_str] = mode_images

                    if args.verbose:
                        print(f"  Rendered {len(mode_images)} {mode_str} frames")
                else:
                    # Single mode rendering
                    mode_rendered = pipeline.render_all(
                        modes=[mode_str],
                        mesh_color=config.render.mesh_color,
                        bg_color=config.render.bg_color
                    )
                    rendered.update(mode_rendered)

                    if args.verbose:
                        for mode, images in mode_rendered.items():
                            print(f"  Rendered {len(images)} {mode} frames")

        # Export
        if args.verbose:
            print("\n[4/4] Exporting...")

        # Export COLMAP
        if config.export.colmap:
            colmap_dir = pipeline.export_colmap(
                config.export.output_dir,
                n_pointcloud_samples=config.export.pointcloud_samples,
                filename_pattern=config.export.filename_pattern
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
