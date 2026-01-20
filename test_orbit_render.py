#!/usr/bin/env python3
"""
Test script for orbit rendering.

Tests:
1. Loading .npz file
2. Generating helical orbit path (3 loops, 120 frames)
3. Rendering all frames
4. Saving to directory

Usage:
    python test_orbit_render.py <input.npz> <output_dir> [--frames N] [--pattern circular|helical]
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from body2colmap.scene import Scene
from body2colmap.camera import Camera
from body2colmap.path import OrbitPath
from body2colmap.renderer import Renderer
from body2colmap.exporter import ImageExporter


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test orbit rendering")
    parser.add_argument("input", help="Path to .npz file from SAM-3D-Body")
    parser.add_argument("output_dir", help="Directory to save rendered frames")
    parser.add_argument("--frames", type=int, default=120, help="Number of frames")
    parser.add_argument("--pattern", choices=["circular", "helical"], default="helical",
                       help="Orbit pattern")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--elevation", type=float, default=0.0,
                       help="Elevation for circular pattern (degrees)")

    args = parser.parse_args()

    print("=" * 70)
    print("Orbit Rendering Test")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")
    print(f"Frames: {args.frames}")
    print(f"Pattern: {args.pattern}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print("=" * 70)

    # Step 1: Load scene
    print(f"\n[1/5] Loading scene...")
    try:
        scene = Scene.from_npz_file(args.input, include_skeleton=False)
        print(f"  ✓ Loaded: {scene}")
    except Exception as e:
        print(f"  ✗ Error loading scene: {e}")
        return 1

    # Step 2: Analyze scene and compute orbit parameters
    print("\n[2/5] Computing orbit parameters...")

    centroid = scene.get_centroid()
    bounds = scene.get_bounds()

    print(f"  Centroid: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")

    # Auto-compute radius with good framing
    radius = OrbitPath.auto_compute_radius(
        scene_bounds=bounds,
        fill_ratio=0.7,  # Leave some margin
        fov_deg=45.0
    )

    print(f"  Orbit radius: {radius:.3f}")
    print(f"  Pattern: {args.pattern}")

    # Step 3: Generate orbit path
    print(f"\n[3/5] Generating {args.pattern} orbit path...")

    orbit = OrbitPath(target=centroid, radius=radius)

    # Create camera template
    render_size = (args.resolution, args.resolution)
    camera_template = Camera.from_fov(
        fov_deg=45.0,
        image_size=render_size
    )

    # Generate cameras based on pattern
    if args.pattern == "circular":
        cameras = orbit.circular(
            n_frames=args.frames,
            elevation_deg=args.elevation,
            camera_template=camera_template
        )
        print(f"  ✓ Generated {len(cameras)} cameras")
        print(f"    Elevation: {args.elevation}°")
        print(f"    Azimuth: 0° to 360°")

    elif args.pattern == "helical":
        cameras = orbit.helical(
            n_frames=args.frames,
            n_loops=3,
            amplitude_deg=30.0,
            lead_in_deg=45.0,
            lead_out_deg=45.0,
            camera_template=camera_template
        )
        print(f"  ✓ Generated {len(cameras)} cameras")
        print(f"    Loops: 3")
        print(f"    Elevation: -30° to +30°")
        print(f"    Total rotation: {45 + 3*360 + 45}°")

    # Verify first and last camera positions
    first_cam = cameras[0]
    last_cam = cameras[-1]
    print(f"\n  First camera: [{first_cam.position[0]:.2f}, {first_cam.position[1]:.2f}, {first_cam.position[2]:.2f}]")
    print(f"  Last camera:  [{last_cam.position[0]:.2f}, {last_cam.position[1]:.2f}, {last_cam.position[2]:.2f}]")

    # Step 4: Render all frames
    print(f"\n[4/5] Rendering {len(cameras)} frames...")
    print("  This may take a while...")

    renderer = Renderer(scene, render_size=render_size)
    images = []

    try:
        for i, camera in enumerate(cameras):
            # Render frame
            image = renderer.render_mesh(
                camera,
                mesh_color=(0.65, 0.74, 0.86),
                bg_color=(1.0, 1.0, 1.0)
            )
            images.append(image)

            # Progress update every 10 frames
            if (i + 1) % 10 == 0 or (i + 1) == len(cameras):
                percent = 100 * (i + 1) / len(cameras)
                print(f"    Rendered {i + 1}/{len(cameras)} frames ({percent:.1f}%)")

        print(f"  ✓ Rendered {len(images)} frames")

    except Exception as e:
        print(f"  ✗ Error during rendering: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 5: Save images
    print(f"\n[5/5] Saving images to {args.output_dir}...")

    try:
        # Generate filenames
        filenames = ImageExporter.generate_filenames(
            n_frames=len(images),
            pattern="frame_{:04d}.png"
        )

        # Create exporter and save
        exporter = ImageExporter(images, filenames)
        saved_paths = exporter.export(Path(args.output_dir))

        print(f"  ✓ Saved {len(saved_paths)} images")
        print(f"    {saved_paths[0]}")
        if len(saved_paths) > 1:
            print(f"    ...")
            print(f"    {saved_paths[-1]}")

    except Exception as e:
        print(f"  ✗ Error saving images: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Summary
    print("\n" + "=" * 70)
    print("✓ Orbit Rendering Complete!")
    print("=" * 70)
    print(f"\nSaved {len(images)} frames to: {args.output_dir}")
    print("\nNext steps:")
    print("  • View individual frames to verify rendering")
    print("  • Create video/GIF to see orbit motion:")
    print(f"      ffmpeg -r 30 -i {args.output_dir}/frame_%04d.png -vcodec libx264 -pix_fmt yuv420p orbit.mp4")
    print("  • Visually check that:")
    print("    - Mesh stays centered throughout orbit")
    print("    - Camera rotates smoothly around mesh")
    print("    - Elevation changes correctly (for helical)")
    print("    - No sudden jumps or flips")

    return 0


if __name__ == "__main__":
    sys.exit(main())
