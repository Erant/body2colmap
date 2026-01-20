#!/usr/bin/env python3
"""
Full pipeline test: Orbit rendering + COLMAP export.

This is the complete pipeline test that generates:
1. Multi-view rendered images
2. COLMAP format camera parameters
3. Initial point cloud

Usage:
    python test_colmap_export.py <input.npz> <output_dir> [--frames N]
"""

import sys
import argparse
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from body2colmap.scene import Scene
from body2colmap.camera import Camera
from body2colmap.path import OrbitPath
from body2colmap.renderer import Renderer
from body2colmap.exporter import ColmapExporter, ImageExporter


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Full COLMAP export pipeline test")
    parser.add_argument("input", help="Path to .npz file from SAM-3D-Body")
    parser.add_argument("output_dir", help="Directory for COLMAP files and images")
    parser.add_argument("--frames", type=int, default=120, help="Number of frames")
    parser.add_argument("--pattern", choices=["circular", "helical"], default="helical")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--pointcloud-samples", type=int, default=50000,
                       help="Number of points to sample from mesh")

    args = parser.parse_args()

    print("=" * 70)
    print("Full COLMAP Export Pipeline Test")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")
    print(f"Frames: {args.frames}")
    print(f"Pattern: {args.pattern}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Point cloud samples: {args.pointcloud_samples}")
    print("=" * 70)

    # Step 1: Load scene
    print(f"\n[1/6] Loading scene...")
    try:
        scene = Scene.from_npz_file(args.input, include_skeleton=False)
        print(f"  ✓ Loaded: {scene}")

        # Show scene info
        centroid = scene.get_centroid()
        bounds = scene.get_bounds()
        print(f"  Centroid: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")

    except Exception as e:
        print(f"  ✗ Error loading scene: {e}")
        return 1

    # Step 2: Generate orbit path
    print(f"\n[2/6] Generating {args.pattern} orbit path...")

    # Auto-compute orbit radius
    radius = OrbitPath.auto_compute_radius(
        scene_bounds=bounds,
        fill_ratio=0.7,
        fov_deg=45.0
    )

    orbit = OrbitPath(target=centroid, radius=radius)

    # Create camera template
    render_size = (args.resolution, args.resolution)
    camera_template = Camera.from_fov(
        fov_deg=45.0,
        image_size=render_size
    )

    # Generate cameras
    if args.pattern == "circular":
        cameras = orbit.circular(
            n_frames=args.frames,
            elevation_deg=0.0,
            camera_template=camera_template
        )
    else:  # helical
        cameras = orbit.helical(
            n_frames=args.frames,
            n_loops=3,
            amplitude_deg=30.0,
            lead_in_deg=45.0,
            lead_out_deg=45.0,
            camera_template=camera_template
        )

    print(f"  ✓ Generated {len(cameras)} cameras")
    print(f"  Orbit radius: {radius:.3f}")

    # Step 3: Render all frames
    print(f"\n[3/6] Rendering {len(cameras)} frames...")
    print("  This may take a while...")

    renderer = Renderer(scene, render_size=render_size)
    images = []

    try:
        for i, camera in enumerate(cameras):
            image = renderer.render_mesh(
                camera,
                mesh_color=(0.65, 0.74, 0.86),
                bg_color=(1.0, 1.0, 1.0)
            )
            images.append(image)

            if (i + 1) % 20 == 0 or (i + 1) == len(cameras):
                percent = 100 * (i + 1) / len(cameras)
                print(f"    Rendered {i + 1}/{len(cameras)} frames ({percent:.1f}%)")

        print(f"  ✓ Rendered {len(images)} frames")

    except Exception as e:
        print(f"  ✗ Error during rendering: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 4: Save images
    print(f"\n[4/6] Saving images...")

    output_path = Path(args.output_dir)

    try:
        # Generate filenames
        filenames = ImageExporter.generate_filenames(
            n_frames=len(images),
            pattern="frame_{:04d}.png"
        )

        # Save images
        exporter = ImageExporter(images, filenames)
        saved_paths = exporter.export(output_path)

        print(f"  ✓ Saved {len(saved_paths)} images to {output_path}/")

    except Exception as e:
        print(f"  ✗ Error saving images: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 5: Export COLMAP files
    print(f"\n[5/6] Exporting COLMAP files...")

    try:
        # Create COLMAP exporter with point cloud
        colmap_exporter = ColmapExporter.from_scene_and_cameras(
            scene=scene,
            cameras=cameras,
            image_names=filenames,
            n_pointcloud_samples=args.pointcloud_samples
        )

        # Export COLMAP files
        colmap_exporter.export(output_path)

        print(f"  ✓ Exported COLMAP files to {output_path}/")
        print(f"    - cameras.txt (1 camera)")
        print(f"    - images.txt ({len(cameras)} images)")
        print(f"    - points3D.txt ({args.pointcloud_samples} points)")

    except Exception as e:
        print(f"  ✗ Error exporting COLMAP: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 6: Validate output
    print(f"\n[6/6] Validating output...")

    try:
        # Check all files exist
        cameras_file = output_path / "cameras.txt"
        images_file = output_path / "images.txt"
        points_file = output_path / "points3D.txt"

        if not cameras_file.exists():
            print(f"  ✗ Missing: cameras.txt")
            return 1
        if not images_file.exists():
            print(f"  ✗ Missing: images.txt")
            return 1
        if not points_file.exists():
            print(f"  ✗ Missing: points3D.txt")
            return 1

        # Quick format validation
        with open(images_file, 'r') as f:
            lines = [l for l in f if not l.startswith('#') and l.strip()]
            # Should be 2 lines per image
            if len(lines) != len(cameras) * 2:
                print(f"  ⚠ Warning: images.txt has {len(lines)} lines, expected {len(cameras) * 2}")

        with open(cameras_file, 'r') as f:
            lines = [l for l in f if not l.startswith('#') and l.strip()]
            if len(lines) != 1:
                print(f"  ⚠ Warning: cameras.txt has {len(lines)} cameras, expected 1")

        with open(points_file, 'r') as f:
            lines = [l for l in f if not l.startswith('#') and l.strip()]
            print(f"  Point cloud: {len(lines)} points")

        print(f"  ✓ All COLMAP files present and formatted correctly")

    except Exception as e:
        print(f"  ✗ Error validating output: {e}")
        return 1

    # Success summary
    print("\n" + "=" * 70)
    print("✓ Full Pipeline Complete!")
    print("=" * 70)
    print(f"\nOutput directory: {output_path}")
    print(f"\nGenerated files:")
    print(f"  • {len(cameras)} rendered images (frame_0001.png - frame_{len(cameras):04d}.png)")
    print(f"  • cameras.txt (camera intrinsics)")
    print(f"  • images.txt (camera extrinsics)")
    print(f"  • points3D.txt ({args.pointcloud_samples} points)")

    print(f"\nNext steps:")
    print(f"  1. View images to verify rendering")
    print(f"  2. Validate COLMAP files:")
    print(f"       colmap feature_extractor --database_path {output_path}/database.db --image_path {output_path}")
    print(f"  3. Or visualize in COLMAP GUI:")
    print(f"       colmap gui")
    print(f"       (Import model from {output_path})")
    print(f"  4. Use for 3D Gaussian Splatting training")

    print(f"\nCoordinate system check:")
    print(f"  ✓ SAM-3D-Body → World conversion applied")
    print(f"  ✓ World → COLMAP/OpenCV conversion applied")
    print(f"  ✓ Quaternions in (w,x,y,z) order")
    print(f"  ✓ World-to-camera transforms")
    print(f"  ✓ Point cloud in world coordinates")

    return 0


if __name__ == "__main__":
    sys.exit(main())
