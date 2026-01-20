#!/usr/bin/env python3
"""
Test script for skeleton and depth rendering.

Tests:
1. Loading skeleton from SAM-3D-Body output
2. Depth rendering
3. Skeleton-only rendering
4. Composite mesh+skeleton rendering

Usage:
    python test_skeleton_render.py <input.npz> <output_dir>
"""

import sys
import argparse
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from body2colmap.scene import Scene
from body2colmap.camera import Camera
from body2colmap.renderer import Renderer
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Test skeleton and depth rendering")
    parser.add_argument("input", help="Path to .npz file from SAM-3D-Body")
    parser.add_argument("output_dir", help="Directory to save rendered images")
    parser.add_argument("--resolution", type=int, default=512)

    args = parser.parse_args()

    print("=" * 70)
    print("Skeleton & Depth Rendering Test")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Load scene WITH skeleton
    print("\n[1/5] Loading scene with skeleton...")
    try:
        scene = Scene.from_npz_file(args.input, include_skeleton=True)
        print(f"  ✓ Loaded: {scene}")

        if scene.skeleton_joints is None:
            print("  ⚠ Warning: No skeleton data found in input file")
            print("    Skeleton rendering will be skipped")
            has_skeleton = False
        else:
            print(f"  Skeleton format: {scene.skeleton_format}")
            print(f"  Number of joints: {len(scene.skeleton_joints)}")
            has_skeleton = True

    except Exception as e:
        print(f"  ✗ Error loading scene: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 2: Set up camera
    print("\n[2/5] Setting up camera...")

    centroid = scene.get_centroid()
    radius = scene.get_bounding_sphere_radius()
    camera_distance = radius * 2.5

    camera_position = centroid + np.array([0, 0, camera_distance], dtype=np.float32)

    render_size = (args.resolution, args.resolution)
    camera = Camera.from_fov(
        fov_deg=45.0,
        image_size=render_size,
        position=camera_position
    )
    camera.look_at(centroid)

    print(f"  Camera position: [{camera.position[0]:.2f}, {camera.position[1]:.2f}, {camera.position[2]:.2f}]")
    print(f"  Looking at: [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}]")

    # Step 3: Create renderer
    print("\n[3/5] Creating renderer...")
    renderer = Renderer(scene, render_size=render_size)
    print("  ✓ Renderer ready")

    # Step 4: Render different modes
    print("\n[4/5] Rendering images...")

    renders = {}

    try:
        # Mesh only
        print("  Rendering mesh...")
        renders["mesh"] = renderer.render_mesh(
            camera,
            mesh_color=(0.65, 0.74, 0.86),
            bg_color=(1.0, 1.0, 1.0)
        )
        print("    ✓ Mesh rendered")

        # Depth (grayscale)
        print("  Rendering depth (grayscale)...")
        renders["depth_gray"] = renderer.render_depth(
            camera,
            normalize=True,
            colormap=None
        )
        print("    ✓ Depth (grayscale) rendered")

        # Depth (viridis colormap)
        print("  Rendering depth (viridis)...")
        renders["depth_viridis"] = renderer.render_depth(
            camera,
            normalize=True,
            colormap="viridis"
        )
        print("    ✓ Depth (viridis) rendered")

        # Skeleton only (if available)
        if has_skeleton:
            print("  Rendering skeleton...")
            try:
                renders["skeleton"] = renderer.render_skeleton(
                    camera,
                    joint_radius=0.015,
                    bone_radius=0.008,
                    joint_color=(1.0, 0.0, 0.0),  # Red joints
                    bone_color=(0.0, 1.0, 0.0)    # Green bones
                )
                print("    ✓ Skeleton rendered")
            except Exception as e:
                print(f"    ⚠ Skeleton rendering failed: {e}")
                has_skeleton = False

            # Composite: mesh + skeleton
            if has_skeleton:
                print("  Rendering mesh + skeleton...")
                try:
                    renders["mesh_skeleton"] = renderer.render_composite(
                        camera,
                        modes={
                            "mesh": {"color": (0.65, 0.74, 0.86)},
                            "skeleton": {
                                "joint_radius": 0.015,
                                "bone_radius": 0.008,
                                "joint_color": (1.0, 0.0, 0.0),
                                "bone_color": (0.0, 1.0, 0.0)
                            }
                        }
                    )
                    print("    ✓ Mesh + skeleton rendered")
                except Exception as e:
                    print(f"    ⚠ Composite rendering failed: {e}")

                # Composite: depth + skeleton
                print("  Rendering depth + skeleton...")
                try:
                    renders["depth_skeleton"] = renderer.render_composite(
                        camera,
                        modes={
                            "depth": {"normalize": True, "colormap": "viridis"},
                            "skeleton": {
                                "joint_radius": 0.015,
                                "bone_radius": 0.008,
                                "joint_color": (1.0, 0.0, 0.0),
                                "bone_color": (1.0, 1.0, 0.0)  # Yellow bones
                            }
                        }
                    )
                    print("    ✓ Depth + skeleton rendered")
                except Exception as e:
                    print(f"    ⚠ Depth + skeleton composite failed: {e}")

    except Exception as e:
        print(f"  ✗ Error during rendering: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 5: Save images
    print("\n[5/5] Saving images...")

    try:
        import cv2

        saved_files = []
        for name, image in renders.items():
            filepath = output_path / f"{name}.png"

            # Convert RGBA to BGRA for OpenCV
            if image.shape[2] == 4:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
            else:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cv2.imwrite(str(filepath), image_bgr)
            saved_files.append(filepath)
            print(f"    ✓ Saved: {filepath.name}")

        print(f"\n  Saved {len(saved_files)} images to {output_path}/")

    except Exception as e:
        print(f"  ✗ Error saving images: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Summary
    print("\n" + "=" * 70)
    print("✓ Test Complete!")
    print("=" * 70)
    print(f"\nGenerated renders:")
    for name in renders.keys():
        print(f"  • {name}.png")

    print(f"\nRender modes tested:")
    print(f"  ✓ Mesh rendering")
    print(f"  ✓ Depth rendering (grayscale)")
    print(f"  ✓ Depth rendering (viridis colormap)")
    if has_skeleton:
        print(f"  ✓ Skeleton rendering")
        print(f"  ✓ Composite mesh + skeleton")
        print(f"  ✓ Composite depth + skeleton")
    else:
        print(f"  ⚠ Skeleton rendering skipped (no skeleton data)")

    print(f"\nOpen the images to verify:")
    print(f"  • Depth maps show distance correctly")
    print(f"  • Skeleton joints (red spheres) at correct positions")
    print(f"  • Skeleton bones (green/yellow cylinders) connecting joints")
    print(f"  • Composite renders show skeleton overlay on mesh/depth")

    return 0


if __name__ == "__main__":
    sys.exit(main())
