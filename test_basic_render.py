#!/usr/bin/env python3
"""
Simple test script for basic rendering.

Tests:
1. Loading .npz file
2. Coordinate conversion
3. Camera setup
4. Basic mesh rendering
5. Image saving

Usage:
    python test_basic_render.py <input.npz> [output.png]
"""

import sys
import numpy as np
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from body2colmap.scene import Scene
from body2colmap.camera import Camera
from body2colmap.renderer import Renderer


def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python test_basic_render.py <input.npz> [output.png]")
        print("\nThis script tests basic rendering by:")
        print("  1. Loading SAM-3D-Body .npz file")
        print("  2. Converting coordinates")
        print("  3. Setting up a camera looking at the front")
        print("  4. Rendering the mesh")
        print("  5. Saving the result")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "test_render.png"

    print("=" * 60)
    print("Basic Render Test")
    print("=" * 60)

    # Step 1: Load scene
    print(f"\n[1/5] Loading scene from {input_path}...")
    try:
        scene = Scene.from_npz_file(input_path, include_skeleton=False)
        print(f"  ✓ Loaded: {scene}")
        print(f"    Vertices: {len(scene.vertices)}")
        print(f"    Faces: {len(scene.faces)}")
    except FileNotFoundError:
        print(f"  ✗ Error: File not found: {input_path}")
        sys.exit(1)
    except KeyError as e:
        print(f"  ✗ Error: Missing required field in .npz: {e}")
        print("\n  Required fields:")
        print("    - pred_vertices")
        print("    - pred_cam_t")
        print("    - faces")
        sys.exit(1)
    except Exception as e:
        print(f"  ✗ Error loading scene: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 2: Get scene info
    print("\n[2/5] Analyzing scene geometry...")
    centroid = scene.get_centroid()
    bounds_min, bounds_max = scene.get_bounds()
    radius = scene.get_bounding_sphere_radius()

    print(f"  Centroid: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")
    print(f"  Bounds: [{bounds_min[0]:.3f}, {bounds_min[1]:.3f}, {bounds_min[2]:.3f}] to "
          f"[{bounds_max[0]:.3f}, {bounds_max[1]:.3f}, {bounds_max[2]:.3f}]")
    print(f"  Bounding sphere radius: {radius:.3f}")

    # Step 3: Set up camera
    print("\n[3/5] Setting up camera...")

    # Camera should be in front of the mesh, looking at it
    # Position camera at a reasonable distance based on bounding sphere
    camera_distance = radius * 2.5  # 2.5x the bounding sphere radius
    camera_position = centroid + np.array([0, 0, camera_distance], dtype=np.float32)

    # Create camera with reasonable FOV
    render_size = (512, 512)
    camera = Camera.from_fov(
        fov_deg=45.0,
        image_size=render_size,
        position=camera_position
    )

    # Point camera at centroid
    camera.look_at(centroid)

    print(f"  Camera position: [{camera.position[0]:.3f}, {camera.position[1]:.3f}, {camera.position[2]:.3f}]")
    print(f"  Looking at: [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")
    print(f"  Distance: {camera_distance:.3f}")
    print(f"  FOV: 45°")
    print(f"  Resolution: {render_size[0]}x{render_size[1]}")

    # Step 4: Render
    print("\n[4/5] Rendering...")
    try:
        renderer = Renderer(scene, render_size=render_size)

        # Render with light blue-gray mesh, white background
        image = renderer.render_mesh(
            camera,
            mesh_color=(0.65, 0.74, 0.86),  # Light blue-gray
            bg_color=(1.0, 1.0, 1.0)  # White
        )

        print(f"  ✓ Rendered image: {image.shape}, dtype={image.dtype}")
        print(f"    Min pixel value: {image.min()}")
        print(f"    Max pixel value: {image.max()}")

        # Check alpha channel
        alpha_mask = image[:, :, 3] > 0
        n_mesh_pixels = alpha_mask.sum()
        n_bg_pixels = (~alpha_mask).sum()
        print(f"    Mesh pixels: {n_mesh_pixels} ({100*n_mesh_pixels/image[:,:,3].size:.1f}%)")
        print(f"    Background pixels: {n_bg_pixels} ({100*n_bg_pixels/image[:,:,3].size:.1f}%)")

    except ImportError as e:
        print(f"  ✗ Error: Missing dependency: {e}")
        print("\n  Required packages:")
        print("    pip install pyrender trimesh")
        sys.exit(1)
    except Exception as e:
        print(f"  ✗ Error during rendering: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 5: Save image
    print(f"\n[5/5] Saving to {output_path}...")
    try:
        import cv2

        # Convert RGBA to BGRA for OpenCV
        image_bgra = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)

        # Save
        success = cv2.imwrite(output_path, image_bgra)

        if success:
            print(f"  ✓ Saved: {output_path}")
        else:
            print(f"  ✗ Error: Failed to save image")
            sys.exit(1)

    except ImportError:
        print("  ✗ Error: opencv-python not installed")
        print("  Install with: pip install opencv-python")
        sys.exit(1)
    except Exception as e:
        print(f"  ✗ Error saving image: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✓ Test Complete!")
    print("=" * 60)
    print(f"\nRendered image saved to: {output_path}")
    print("\nThis confirms:")
    print("  ✓ File loading works")
    print("  ✓ Coordinate conversion works")
    print("  ✓ Camera setup works")
    print("  ✓ Rendering works")
    print("  ✓ Image export works")


if __name__ == "__main__":
    main()
