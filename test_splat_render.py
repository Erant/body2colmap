#!/usr/bin/env python3
"""
Test script for Gaussian Splat rendering.

Usage:
    python test_splat_render.py <path_to_ply_file> [output_dir]

This will:
1. Load the PLY file
2. Generate a few test views (circular orbit)
3. Render and save images to output_dir (default: ./splat_test_output)

Requirements:
    pip install body2colmap[splat]
"""

import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Test Gaussian Splat rendering")
    parser.add_argument("ply_file", help="Path to Gaussian splat PLY file")
    parser.add_argument("--output", "-o", default="./splat_test_output",
                       help="Output directory (default: ./splat_test_output)")
    parser.add_argument("--n-frames", "-n", type=int, default=8,
                       help="Number of test frames (default: 8)")
    parser.add_argument("--width", type=int, default=512,
                       help="Render width (default: 512)")
    parser.add_argument("--height", type=int, default=512,
                       help="Render height (default: 512)")
    parser.add_argument("--pattern", choices=["circular", "helical"], default="circular",
                       help="Camera path pattern (default: circular)")
    args = parser.parse_args()

    # Check PLY file exists
    ply_path = Path(args.ply_file)
    if not ply_path.exists():
        print(f"Error: PLY file not found: {ply_path}")
        return 1

    print(f"Loading PLY file: {ply_path}")

    # Import here to get better error messages
    try:
        from body2colmap.pipeline import OrbitPipeline
    except ImportError as e:
        print(f"Error importing body2colmap: {e}")
        print("Make sure body2colmap is installed: pip install -e .")
        return 1

    try:
        from body2colmap.splat_scene import SplatScene
    except ImportError as e:
        print(f"Error importing splat support: {e}")
        print("Make sure splat dependencies are installed: pip install body2colmap[splat]")
        return 1

    # Load the PLY file
    try:
        pipeline = OrbitPipeline.from_ply_file(
            str(ply_path),
            render_size=(args.width, args.height)
        )
        print(f"Loaded: {pipeline.scene}")
    except Exception as e:
        print(f"Error loading PLY file: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Show scene bounds
    min_corner, max_corner = pipeline.scene.get_bounds()
    center = pipeline.scene.get_bbox_center()
    print(f"Scene bounds: min={min_corner}, max={max_corner}")
    print(f"Scene center: {center}")

    # Generate camera path
    print(f"\nGenerating {args.n_frames} camera positions ({args.pattern} path)...")
    try:
        pipeline.set_orbit_params(
            pattern=args.pattern,
            n_frames=args.n_frames,
            fill_ratio=0.8
        )
        print(f"Orbit radius: {pipeline.orbit_params['radius']:.3f}")
    except Exception as e:
        print(f"Error generating camera path: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Render frames
    print(f"\nRendering {args.n_frames} frames...")
    try:
        images = pipeline.render_all(modes=["splat"], bg_color=(1.0, 1.0, 1.0))
        print(f"Rendered {len(images['splat'])} images")
    except Exception as e:
        print(f"Error rendering: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save images
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving images to: {output_dir}")

    try:
        saved = pipeline.export_images(
            str(output_dir),
            images["splat"],
            filename_pattern="frame_{:04d}.png"
        )
        print(f"Saved {len(saved)} images:")
        for p in saved[:3]:
            print(f"  {p}")
        if len(saved) > 3:
            print(f"  ... and {len(saved) - 3} more")
    except Exception as e:
        print(f"Error saving images: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\nDone! Check the output directory for rendered images.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
