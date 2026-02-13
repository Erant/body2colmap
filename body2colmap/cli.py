"""
Command-line interface for body2colmap.

This module provides the main entry point for the CLI tool.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .config import create_argument_parser, Config
from .face import FaceLandmarkIngest
from .pipeline import OrbitPipeline
from .scene import Scene


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

        # --- Debug: render from original SAM-3D-Body viewpoint ---
        # When --use-original-camera is also active, skip the standalone debug
        # path; the compositing will happen on orbit frame 0 instead.
        if args.debug_original_view and not config.path.use_original_camera:
            if is_splat:
                raise ValueError("--debug-original-view is only supported for .npz input")

            auto_frame = not args.no_auto_frame

            if args.verbose:
                mode_label = "auto-framed" if auto_frame else "raw"
                print(f"\n[DEBUG] Original-view rendering mode ({mode_label})")

            # Load metadata from .npz to get original focal length
            metadata = Scene.load_npz_metadata(config.input_file)
            if args.verbose:
                print(f"  .npz keys: {metadata['_all_keys']}")
                for k, v in metadata.items():
                    if k != '_all_keys':
                        val_repr = f"shape={v.shape}" if hasattr(v, 'shape') else repr(v)
                        print(f"  {k}: {val_repr}")

            if 'focal_length' not in metadata:
                raise ValueError(
                    "--debug-original-view requires 'focal_length' in .npz file, "
                    f"but it only contains: {metadata['_all_keys']}"
                )

            original_fl = float(metadata['focal_length'])
            if args.verbose:
                print(f"\n  Original focal length: {original_fl:.2f} px")
                print(f"  Render resolution: {config.render.resolution[0]}x{config.render.resolution[1]}")
                if not auto_frame:
                    print(f"  NOTE: focal_length is from SAM-3D-Body's internal crop.")
                    print(f"  For exact overlay, render resolution must match the crop size.")

            # DO NOT auto-orient — the original view requires the mesh
            # in its post-conversion position (no additional rotations).

            # Determine render modes
            modes = config.render.modes

            # Build render kwargs
            render_kwargs = {
                'mesh_color': config.render.mesh_color,
                'bg_color': config.render.bg_color,
                'joint_radius': config.skeleton.joint_radius,
                'bone_radius': config.skeleton.bone_radius,
            }

            # Load face landmarks if needed
            if config.skeleton.face_landmarks:
                fl_70 = FaceLandmarkIngest.from_json(config.skeleton.face_landmarks)
                render_kwargs['face_landmarks'] = fl_70
            if config.skeleton.face_mode:
                render_kwargs['face_mode'] = config.skeleton.face_mode

            # Render from original viewpoint
            if args.verbose:
                print(f"\n  Rendering modes: {modes}")
                if auto_frame:
                    print(f"  Auto-framing: fill_ratio={config.camera.fill_ratio}")

            rendered, framing_info = pipeline.render_original_view(
                original_focal_length=original_fl,
                modes=modes,
                auto_frame=auto_frame,
                fill_ratio=config.camera.fill_ratio,
                **render_kwargs
            )

            if auto_frame and args.verbose:
                s = framing_info['scale_factor']
                f_new = framing_info['framed_focal_length']
                cx, cy = framing_info['framed_principal_point']
                bbox = framing_info['original_2d_bbox']
                print(f"\n  Auto-framing results:")
                print(f"    Scale factor: {s:.4f} ({'zoom in' if s > 1 else 'zoom out'})")
                print(f"    Framed focal length: {f_new:.2f} px")
                print(f"    Framed principal point: ({cx:.1f}, {cy:.1f})")
                print(f"    Original 2D bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")

            # Save images
            output_dir = Path(config.export.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            import cv2
            for mode, image in rendered.items():
                safe_mode = mode.replace('+', '_')
                filename = f"original_view_{safe_mode}.png"
                filepath = output_dir / filename
                if image.shape[2] == 4:
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
                else:
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(filepath), image_bgr)
                print(f"  Saved: {filepath}")

            # --- Warp original image if provided ---
            original_image_path = args.original_image
            if original_image_path is not None:
                if not auto_frame:
                    print("  WARNING: --original-image requires auto-framing; ignoring --no-auto-frame")
                    # Re-render with auto-framing enabled
                    rendered, framing_info = pipeline.render_original_view(
                        original_focal_length=original_fl,
                        modes=modes,
                        auto_frame=True,
                        fill_ratio=config.camera.fill_ratio,
                        **render_kwargs
                    )
                    # Re-save the auto-framed renders
                    for mode, image in rendered.items():
                        safe_mode = mode.replace('+', '_')
                        filename = f"original_view_{safe_mode}.png"
                        filepath = output_dir / filename
                        if image.shape[2] == 4:
                            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
                        else:
                            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(filepath), image_bgr)

                orig_img = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
                if orig_img is None:
                    raise FileNotFoundError(f"Could not read image: {original_image_path}")

                h_img, w_img = orig_img.shape[:2]
                w_r, h_r = config.render.resolution

                if args.verbose:
                    print(f"\n  Original image: {original_image_path} ({w_img}x{h_img})")
                    if (w_img, h_img) != (w_r, h_r):
                        print(f"  Image size differs from render size ({w_r}x{h_r}); "
                              "adjusting affine to compensate.")

                # Background color for padding (convert 0-1 float RGB → 0-255 int BGR)
                bg_r, bg_g, bg_b = config.render.bg_color
                border_bgr = (int(bg_b * 255), int(bg_g * 255), int(bg_r * 255))

                warped = pipeline.renderer.warp_original_image(
                    orig_img,
                    camera=framing_info['camera'],
                    original_focal_length=original_fl,
                    border_color=border_bgr,
                )

                warped_path = output_dir / "original_view_warped.png"
                cv2.imwrite(str(warped_path), warped)
                print(f"  Saved: {warped_path}")

                # Create overlay composites (render on top of warped original)
                warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
                for mode, render_rgba in rendered.items():
                    alpha = render_rgba[:, :, 3:4].astype(np.float32) / 255.0
                    render_rgb = render_rgba[:, :, :3].astype(np.float32)
                    base_rgb = warped_rgb.astype(np.float32)
                    composite = (alpha * render_rgb + (1.0 - alpha) * base_rgb)
                    composite = np.clip(composite, 0, 255).astype(np.uint8)

                    safe_mode = mode.replace('+', '_')
                    overlay_path = output_dir / f"original_view_overlay_{safe_mode}.png"
                    overlay_bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(overlay_path), overlay_bgr)
                    print(f"  Saved: {overlay_path}")

                framing_info['image_size'] = [w_img, h_img]

            # Save framing metadata (exclude non-serializable camera object)
            import json
            serializable = {k: v for k, v in framing_info.items() if k != 'camera'}
            framing_path = output_dir / "original_view_framing.json"
            with open(framing_path, 'w') as f:
                json.dump(serializable, f, indent=2)
            print(f"  Saved: {framing_path}")

            # Save 2D keypoints if available
            if 'pred_keypoints_2d' in metadata:
                import json
                kp2d = metadata['pred_keypoints_2d']
                kp_path = output_dir / "original_view_keypoints_2d.json"
                kp_list = kp2d.tolist() if hasattr(kp2d, 'tolist') else kp2d
                with open(kp_path, 'w') as f:
                    json.dump(kp_list, f, indent=2)
                print(f"  Saved 2D keypoints: {kp_path}")

            if args.verbose:
                print("\n[DEBUG] Done.")

            return 0

        # Load face landmarks if provided
        face_landmarks_70 = None
        if config.skeleton.face_landmarks:
            if args.verbose:
                print(f"  Loading face landmarks: {config.skeleton.face_landmarks}")
            face_landmarks_70 = FaceLandmarkIngest.from_json(config.skeleton.face_landmarks)
            if args.verbose:
                print(f"  Converted to OpenPose Face 70 ({face_landmarks_70.shape})")

        # Determine original_focal_length for use-original-camera mode
        original_fl_for_orbit = None
        if config.path.use_original_camera and not is_splat:
            metadata = Scene.load_npz_metadata(config.input_file)
            if 'focal_length' not in metadata:
                raise ValueError(
                    "--use-original-camera requires 'focal_length' in .npz file, "
                    f"but it only contains: {metadata['_all_keys']}"
                )
            original_fl_for_orbit = float(metadata['focal_length'])
            if args.verbose:
                print(f"  Original-camera mode: focal_length={original_fl_for_orbit:.2f} px")
                print(f"  Skipping auto-orient (mesh stays in original position)")

        # Auto-orient: rotate body to face camera, plus any user offset
        # Skip when using original camera (mesh must stay in its original position)
        if not is_splat and original_fl_for_orbit is None:
            pipeline.auto_orient(rotation_offset_deg=config.path.initial_rotation)
            if args.verbose:
                print(f"  Auto-oriented body (offset: {config.path.initial_rotation}°)")

        # Generate orbit
        if args.verbose:
            print("\n[2/4] Generating orbit path...")

        # Build orbit kwargs based on pattern
        orbit_kwargs = {
            'framing': config.path.framing,
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
            original_focal_length=original_fl_for_orbit,
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

        # --- Debug: composite frame 0 with warped original image ---
        if (args.debug_original_view
                and config.path.use_original_camera
                and args.original_image is not None
                and pipeline.orbit_params is not None):
            import cv2

            if args.verbose:
                print("\n[DEBUG] Compositing orbit frame 0 with warped original image...")

            orig_img = cv2.imread(args.original_image, cv2.IMREAD_COLOR)
            if orig_img is None:
                raise FileNotFoundError(f"Could not read image: {args.original_image}")

            frame0_camera = pipeline.orbit_params['frame0_camera']

            # Background color for padding
            bg_r, bg_g, bg_b = config.render.bg_color
            border_bgr = (int(bg_b * 255), int(bg_g * 255), int(bg_r * 255))

            warped = pipeline.renderer.warp_original_image(
                orig_img,
                camera=frame0_camera,
                original_focal_length=pipeline.orbit_params['original_focal_length'],
                border_color=border_bgr,
            )

            output_dir = Path(config.export.output_dir)

            warped_path = output_dir / "frame0_warped.png"
            cv2.imwrite(str(warped_path), warped)
            print(f"  Saved: {warped_path}")

            # Composite each rendered mode's frame 0 on top of warped image
            warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            for mode, images in rendered.items():
                if not images:
                    continue
                frame0_rgba = images[0]
                alpha = frame0_rgba[:, :, 3:4].astype(np.float32) / 255.0
                render_rgb = frame0_rgba[:, :, :3].astype(np.float32)
                base_rgb = warped_rgb.astype(np.float32)
                composite = (alpha * render_rgb + (1.0 - alpha) * base_rgb)
                composite = np.clip(composite, 0, 255).astype(np.uint8)

                safe_mode = mode.replace('+', '_')
                overlay_path = output_dir / f"frame0_overlay_{safe_mode}.png"
                overlay_bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(overlay_path), overlay_bgr)
                print(f"  Saved: {overlay_path}")

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
