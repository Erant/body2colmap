"""
Command-line interface for texture projection workflow.

Provides subcommands for the two-phase circular→helical Canny projection pipeline:

  Phase 1 (circular): Generate reference views for diffusion
    body2colmap-projection circular input.npz -o ./circular_output

  (External: User runs diffusion model on circular_output/*.png)

  Phase 2 (helical): Project diffusion output and render guidance
    body2colmap-projection helical input.npz --diffusion-dir ./circular_output -o ./helical_output
"""

import argparse
import sys
from pathlib import Path
from typing import Optional


def cmd_circular(args: argparse.Namespace) -> int:
    """Execute Phase 1: Generate circular orbit for diffusion input."""
    from .projection_pipeline import ProjectionPipeline
    from .renderer import Renderer

    print(f"Phase 1: Generating circular orbit views")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output_dir}")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Frames: {args.n_frames}")
    print(f"  Elevation: {args.elevation}°")
    print(f"  Mode: {args.render_mode}")

    # Create pipeline
    pipeline = ProjectionPipeline.from_npz_file(
        args.input,
        render_size=(args.width, args.height),
        include_skeleton=True,  # Always load skeleton for depth+skeleton mode
        atlas_size=(args.atlas_size, args.atlas_size)
    )
    print(f"  Loaded scene: {pipeline.scene}")

    # Setup circular orbit
    pipeline.setup_circular_orbit(
        n_frames=args.n_frames,
        elevation_deg=args.elevation,
        fill_ratio=args.fill_ratio
    )
    print(f"  Generated {len(pipeline.circular_cameras)} cameras")

    # Render based on mode
    print(f"  Rendering...")
    if args.render_mode == "mesh":
        images = pipeline.render_circular_orbit(
            mesh_color=tuple(args.mesh_color) if args.mesh_color else None,
            bg_color=tuple(args.bg_color)
        )
    else:
        # depth+skeleton mode - use renderer directly
        renderer = Renderer(pipeline.scene, (args.width, args.height))
        images = []
        for camera in pipeline.circular_cameras:
            modes = {"depth": {"normalize": True}}
            if pipeline.scene.skeleton_joints is not None:
                modes["skeleton"] = {
                    "joint_radius": 0.006,
                    "bone_radius": 0.003,
                    "use_openpose_colors": True
                }
            image = renderer.render_composite(camera, modes)
            images.append(image)
    print(f"  Rendered {len(images)} images")

    # Export
    print(f"  Exporting...")
    pipeline.export_circular_orbit(
        args.output_dir,
        images,
        filename_pattern=args.filename_pattern
    )
    print(f"  Exported to {args.output_dir}")

    print()
    print("Phase 1 complete!")
    print(f"Next steps:")
    print(f"  1. Process images in {args.output_dir}/ through your diffusion model")
    print(f"  2. Save processed images back to {args.output_dir}/ (or another directory)")
    print(f"  3. Run: body2colmap-projection helical {args.input} --diffusion-dir {args.output_dir} -o ./helical_output")

    return 0


def cmd_helical(args: argparse.Namespace) -> int:
    """Execute Phase 2: Project Canny and render helical guidance."""
    from .projection_pipeline import ProjectionPipeline

    print(f"Phase 2: Generating helical orbit with Canny projection")
    print(f"  Input: {args.input}")
    print(f"  Diffusion dir: {args.diffusion_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Frames: {args.n_frames}")
    print(f"  Loops: {args.n_loops}")

    # Create pipeline
    pipeline = ProjectionPipeline.from_npz_file(
        args.input,
        render_size=(args.width, args.height),
        include_skeleton=args.skeleton,
        atlas_size=(args.atlas_size, args.atlas_size)
    )
    print(f"  Loaded scene: {pipeline.scene}")

    # Setup circular orbit (needed to match camera count)
    pipeline.setup_circular_orbit(
        n_frames=args.circular_frames,
        elevation_deg=args.circular_elevation,
        fill_ratio=args.fill_ratio
    )
    print(f"  Set up {len(pipeline.circular_cameras)} circular cameras")

    # Load diffusion images
    print(f"  Loading diffusion images from {args.diffusion_dir}...")
    images = pipeline.load_diffusion_images(
        args.diffusion_dir,
        filename_pattern=args.filename_pattern
    )
    print(f"  Loaded {len(images)} images")

    # Select subset of frames for projection if requested
    projection_images = images
    projection_cameras = pipeline.circular_cameras
    if args.projection_frames is not None and args.projection_frames < len(images):
        # Evenly sample frames around the orbit
        n_proj = args.projection_frames
        indices = [int(i * len(images) / n_proj) for i in range(n_proj)]
        projection_images = [images[i] for i in indices]
        projection_cameras = [pipeline.circular_cameras[i] for i in indices]
        print(f"  Using {len(projection_images)} frames for projection (indices: {indices})")
        # Temporarily override circular_cameras for atlas generation
        original_cameras = pipeline.circular_cameras
        pipeline.circular_cameras = projection_cameras

    # Generate texture (atlas or vertex colors)
    blend_mode = args.blend_mode
    use_vertex_colors = getattr(args, 'use_vertex_colors', False)

    if use_vertex_colors:
        print(f"  Generating vertex colors (blend: {blend_mode})...")
        vertex_colors = pipeline.generate_vertex_colors_from_images(
            images=projection_images,
            blend_mode=blend_mode
        )
        print(f"  Generated vertex colors: {vertex_colors.shape}")
    else:
        print(f"  Generating texture atlas (mode: {args.texture_mode}, blend: {blend_mode})...")
        atlas = pipeline.generate_texture_atlas_from_images(
            images=projection_images,
            mode=args.texture_mode,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            canny_blur=args.canny_blur,
            uv_method=args.uv_method,
            blend_mode=blend_mode
        )
        print(f"  Generated atlas: {atlas.shape}")

    # Restore original cameras if we overrode them
    if args.projection_frames is not None and args.projection_frames < len(images):
        pipeline.circular_cameras = original_cameras

    # Save atlas if requested (only for texture atlas mode)
    if args.save_atlas and not use_vertex_colors:
        atlas_path = Path(args.output_dir) / f"texture_atlas_{args.texture_mode}.png"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        from PIL import Image
        Image.fromarray(atlas).save(atlas_path)
        print(f"  Saved atlas to {atlas_path}")

    # Setup helical orbit
    pipeline.setup_helical_orbit(
        n_frames=args.n_frames,
        n_loops=args.n_loops,
        amplitude_deg=args.amplitude,
        lead_in_deg=args.lead_in,
        lead_out_deg=args.lead_out,
        fill_ratio=args.fill_ratio
    )
    print(f"  Set up {len(pipeline.helical_cameras)} helical cameras")

    # Render with texture
    mode_desc = "vertex colors" if use_vertex_colors else "texture atlas"
    skel_desc = " + skeleton" if args.skeleton else ""
    print(f"  Rendering composites ({args.base_mode} + {mode_desc}{skel_desc})...")
    guidance_images = pipeline.render_helical_with_texture(
        base_mode=args.base_mode,
        include_texture=True,
        include_skeleton=args.skeleton,
        use_vertex_colors=use_vertex_colors
    )
    print(f"  Rendered {len(guidance_images)} images")

    # Export
    print(f"  Exporting...")
    pipeline.export_helical_output(
        args.output_dir,
        guidance_images,
        filename_pattern=args.filename_pattern
    )
    print(f"  Exported to {args.output_dir}")

    print()
    print("Phase 2 complete!")
    print(f"Guidance images saved to {args.output_dir}/")
    print(f"COLMAP files saved to {args.output_dir}/sparse/0/")

    return 0


def cmd_test(args: argparse.Namespace) -> int:
    """Quick test of pipeline components without full rendering."""
    print("Testing projection pipeline components...")
    print()

    # Test imports
    print("[1/6] Testing imports...")
    try:
        from . import edges as edge_module
        from . import texture_projection as tex_proj
        from .projection_pipeline import ProjectionPipeline
        from .scene import Scene
        from .renderer import Renderer
        print("  OK: All modules imported")
    except ImportError as e:
        print(f"  FAIL: Import error: {e}")
        return 1

    # Test edge detection
    print("[2/6] Testing edge detection...")
    try:
        import numpy as np
        test_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        edges = edge_module.canny(test_img, low_threshold=50, high_threshold=150)
        rgba = edge_module.edges_to_rgba(edges, color=(1.0, 1.0, 1.0))
        print(f"  OK: Canny output shape: {edges.shape}, RGBA shape: {rgba.shape}")
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1

    # Test UV generation
    print("[3/6] Testing UV generation...")
    try:
        import numpy as np
        # Simple cube vertices
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ], dtype=np.float32)
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # front
            [4, 6, 5], [4, 7, 6],  # back
        ], dtype=np.int32)

        uvs_cyl = tex_proj.generate_cylindrical_uvs(vertices)
        uvs_sph = tex_proj.generate_spherical_uvs(vertices)
        print(f"  OK: Cylindrical UVs shape: {uvs_cyl.shape}, Spherical: {uvs_sph.shape}")
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1

    # Test TextureProjector
    print("[4/6] Testing TextureProjector...")
    try:
        projector = tex_proj.TextureProjector(vertices, faces, uvs_cyl, atlas_size=(128, 128))
        atlas = projector.get_atlas()
        print(f"  OK: Atlas shape: {atlas.shape}")
    except Exception as e:
        print(f"  FAIL: {e}")
        return 1

    # Test with actual file if provided
    if args.input:
        print(f"[5/6] Testing Scene loading from {args.input}...")
        try:
            scene = Scene.from_npz_file(args.input, include_skeleton=True)
            print(f"  OK: Loaded scene with {len(scene.vertices)} vertices, {len(scene.faces)} faces")
            if scene.skeleton_joints is not None:
                print(f"      Skeleton: {len(scene.skeleton_joints)} joints")
        except Exception as e:
            print(f"  FAIL: {e}")
            return 1

        print(f"[6/6] Testing Renderer...")
        try:
            renderer = Renderer(scene, render_size=(128, 128))
            from .camera import Camera
            from .path import OrbitPath
            from .utils import compute_default_focal_length

            # Create a test camera
            focal = compute_default_focal_length(128)
            camera = Camera(focal_length=(focal, focal), image_size=(128, 128))
            target = scene.get_bbox_center()
            camera.look_at(eye=target + np.array([0, 0, 2]), target=target)

            # Test render_face_ids
            face_ids = renderer.render_face_ids(camera)
            n_visible = np.sum(face_ids >= 0)
            print(f"  OK: Face IDs rendered, {n_visible} pixels have visible faces")

            # Test render_mesh
            mesh_img = renderer.render_mesh(camera)
            print(f"  OK: Mesh rendered, shape: {mesh_img.shape}")

        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print("[5/6] Skipping Scene loading (no input file provided)")
        print("[6/6] Skipping Renderer test (no input file)")

    print()
    print("All tests passed!")
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="body2colmap-projection",
        description="Texture projection workflow for circular→helical Canny guidance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phase 1: Generate circular orbit for diffusion
  body2colmap-projection circular input.npz -o ./circular_output

  # Phase 2: Load diffusion output and generate helical guidance
  body2colmap-projection helical input.npz --diffusion-dir ./circular_output -o ./helical_output

  # Quick test of components
  body2colmap-projection test
  body2colmap-projection test input.npz  # with actual mesh
"""
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- circular subcommand ---
    circular = subparsers.add_parser(
        "circular",
        help="Phase 1: Generate circular orbit views for diffusion input",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    circular.add_argument("input", help="Path to SAM-3D-Body .npz file")
    circular.add_argument("-o", "--output-dir", required=True, help="Output directory")
    circular.add_argument("--width", type=int, default=720, help="Render width")
    circular.add_argument("--height", type=int, default=1280, help="Render height")
    circular.add_argument("--n-frames", type=int, default=81, help="Number of views")
    circular.add_argument("--elevation", type=float, default=0.0, help="Elevation angle (degrees)")
    circular.add_argument("--fill-ratio", type=float, default=0.8, help="Frame fill ratio")
    circular.add_argument("--render-mode", choices=["mesh", "depth+skeleton"], default="depth+skeleton", help="Render mode")
    circular.add_argument("--atlas-size", type=int, default=1024, help="UV atlas size")
    circular.add_argument("--mesh-color", type=float, nargs=3, help="Mesh RGB color (0-1, for mesh mode)")
    circular.add_argument("--bg-color", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Background RGB")
    circular.add_argument("--filename-pattern", default="frame_{:04d}.png", help="Output filename pattern")
    circular.set_defaults(func=cmd_circular)

    # --- helical subcommand ---
    helical = subparsers.add_parser(
        "helical",
        help="Phase 2: Project Canny and render helical guidance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    helical.add_argument("input", help="Path to SAM-3D-Body .npz file")
    helical.add_argument("--diffusion-dir", required=True, help="Directory with diffusion-processed images")
    helical.add_argument("-o", "--output-dir", required=True, help="Output directory")
    helical.add_argument("--width", type=int, default=720, help="Render width")
    helical.add_argument("--height", type=int, default=1280, help="Render height")

    # Circular orbit params (must match Phase 1)
    helical.add_argument("--circular-frames", type=int, default=81, help="Circular orbit frame count (must match Phase 1)")
    helical.add_argument("--circular-elevation", type=float, default=0.0, help="Circular elevation (must match Phase 1)")

    # Helical orbit params
    helical.add_argument("--n-frames", type=int, default=81, help="Helical orbit frame count")
    helical.add_argument("--n-loops", type=int, default=2, help="Number of full rotations")
    helical.add_argument("--amplitude", type=float, default=30.0, help="Elevation amplitude (degrees)")
    helical.add_argument("--lead-in", type=float, default=45.0, help="Lead-in azimuth range")
    helical.add_argument("--lead-out", type=float, default=45.0, help="Lead-out azimuth range")
    helical.add_argument("--fill-ratio", type=float, default=0.8, help="Frame fill ratio")

    # Texture projection params
    helical.add_argument("--texture-mode", choices=["canny", "color", "both"], default="color", help="Texture projection mode")
    helical.add_argument("--blend-mode", choices=["max", "average", "best_angle"], default="best_angle",
                        help="Blending mode: max (preserve all), average (blend), best_angle (use best view per texel)")
    helical.add_argument("--canny-low", type=int, default=50, help="Canny low threshold (for canny/both modes)")
    helical.add_argument("--canny-high", type=int, default=150, help="Canny high threshold (for canny/both modes)")
    helical.add_argument("--canny-blur", type=int, default=5, help="Canny blur kernel size (for canny/both modes)")
    helical.add_argument("--uv-method", choices=["cylindrical", "spherical"], default="cylindrical", help="UV generation method")
    helical.add_argument("--projection-frames", type=int, default=None,
                        help="Number of frames to use for projection (default: all circular frames). Use 4-8 for sharper results.")
    helical.add_argument("--use-vertex-colors", action="store_true",
                        help="Use vertex colors instead of texture atlas (avoids UV overlap issues)")

    # Output options - defaults: mesh base + skeleton
    helical.add_argument("--skeleton", action="store_true", default=True, help="Include skeleton overlay (default)")
    helical.add_argument("--no-skeleton", dest="skeleton", action="store_false", help="Exclude skeleton overlay")
    helical.add_argument("--base-mode", choices=["mesh", "depth", "normals"], default="mesh",
                        help="Base layer type (default: mesh)")
    helical.add_argument("--atlas-size", type=int, default=1024, help="UV atlas size")
    helical.add_argument("--save-atlas", action="store_true", help="Save texture atlas image")
    helical.add_argument("--filename-pattern", default="frame_{:04d}.png", help="Output filename pattern")
    helical.set_defaults(func=cmd_helical)

    # --- test subcommand ---
    test = subparsers.add_parser(
        "test",
        help="Quick test of pipeline components",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    test.add_argument("input", nargs="?", help="Optional .npz file for full test")
    test.set_defaults(func=cmd_test)

    return parser


def main(argv: Optional[list] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    try:
        return args.func(args)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
