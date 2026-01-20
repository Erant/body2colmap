#!/usr/bin/env python3
"""
Debug skeleton joint indices.

This script renders each joint with its index number overlaid,
helping to identify which joints are which in the skeleton format.

Usage:
    python debug_skeleton_joints.py <input.npz> <output_dir>
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from body2colmap.scene import Scene
from body2colmap.camera import Camera
from body2colmap.renderer import Renderer


def main():
    parser = argparse.ArgumentParser(description="Debug skeleton joint indices")
    parser.add_argument("input", help="Path to .npz file from SAM-3D-Body")
    parser.add_argument("output_dir", help="Directory to save renders")
    parser.add_argument("--resolution", type=int, default=1024, help="Higher res for reading numbers")

    args = parser.parse_args()

    print("=" * 70)
    print("Skeleton Joint Index Debugger")
    print("=" * 70)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load scene with skeleton
    print("\n[1/4] Loading scene with skeleton...")
    scene = Scene.from_npz_file(args.input, include_skeleton=True)

    if scene.skeleton_joints is None:
        print("  ✗ No skeleton data found!")
        return 1

    print(f"  ✓ Loaded {len(scene.skeleton_joints)} joints")
    print(f"  Format: {scene.skeleton_format}")

    # Set up camera
    print("\n[2/4] Setting up camera...")
    centroid = scene.get_centroid()
    radius = scene.get_bounding_sphere_radius()
    camera_distance = radius * 2.5

    camera_position = centroid + np.array([0, 0, camera_distance], dtype=np.float32)
    render_size = (args.resolution, args.resolution)
    camera = Camera.from_fov(fov_deg=45.0, image_size=render_size, position=camera_position)
    camera.look_at(centroid)

    # Render skeleton with joints only (no bones)
    print("\n[3/4] Rendering skeleton (joints only)...")
    renderer = Renderer(scene, render_size=render_size)

    # Create a custom render with larger joints
    try:
        import pyrender
        import trimesh

        pr_scene = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[1.0, 1.0, 1.0])

        # Render mesh lightly in background
        mesh_tm = trimesh.Trimesh(vertices=scene.vertices, faces=scene.faces, process=False)
        mesh_tm.visual.vertex_colors = np.array([200, 200, 200, 100], dtype=np.uint8)
        pr_mesh = pyrender.Mesh.from_trimesh(mesh_tm, smooth=True)
        pr_scene.add(pr_mesh)

        # Add large colored spheres for each joint
        for i, joint_pos in enumerate(scene.skeleton_joints):
            # Color based on index (rainbow)
            hue = (i / len(scene.skeleton_joints)) * 360
            # Simple HSV to RGB (hue only, full saturation and value)
            h_sector = int(hue / 60)
            f = (hue / 60) - h_sector
            v = 1.0
            s = 1.0
            p = v * (1 - s)
            q = v * (1 - f * s)
            t = v * (1 - (1 - f) * s)

            if h_sector == 0:
                r, g, b = v, t, p
            elif h_sector == 1:
                r, g, b = q, v, p
            elif h_sector == 2:
                r, g, b = p, v, t
            elif h_sector == 3:
                r, g, b = p, q, v
            elif h_sector == 4:
                r, g, b = t, p, v
            else:
                r, g, b = v, p, q

            sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.025)
            sphere.vertices += joint_pos
            sphere.visual.vertex_colors = np.array([int(r*255), int(g*255), int(b*255), 255], dtype=np.uint8)

            pr_scene.add(pyrender.Mesh.from_trimesh(sphere, smooth=True))

        # Add camera
        pr_camera = pyrender.PerspectiveCamera(
            yfov=2 * np.arctan(render_size[1] / (2 * camera.fy)),
            aspectRatio=render_size[0] / render_size[1]
        )
        pr_scene.add(pr_camera, pose=camera.get_c2w())

        # Render
        r = pyrender.OffscreenRenderer(viewport_width=render_size[0], viewport_height=render_size[1])
        color, depth = r.render(pr_scene, flags=pyrender.RenderFlags.RGBA)
        r.delete()

        print(f"  ✓ Rendered skeleton with {len(scene.skeleton_joints)} joints")

    except Exception as e:
        print(f"  ✗ Error rendering: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save image
    print("\n[4/4] Saving image and joint list...")
    try:
        import cv2

        output_img = output_path / "skeleton_joints_numbered.png"
        image_bgra = cv2.cvtColor(color, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(output_img), image_bgra)
        print(f"  ✓ Saved: {output_img}")

        # Project joints to 2D and save coordinates
        joint_list_file = output_path / "joint_indices.txt"
        with open(joint_list_file, 'w') as f:
            f.write("Joint Index | 3D Position (world coords)\n")
            f.write("-" * 50 + "\n")

            for i, joint_pos in enumerate(scene.skeleton_joints):
                f.write(f"Joint {i:2d}  | [{joint_pos[0]:7.3f}, {joint_pos[1]:7.3f}, {joint_pos[2]:7.3f}]\n")

        print(f"  ✓ Saved: {joint_list_file}")

        # Print some key joint positions for manual inspection
        print("\n" + "=" * 70)
        print("Joint positions saved!")
        print("=" * 70)
        print("\nTo identify joints:")
        print("1. Open skeleton_joints_numbered.png")
        print("2. Each joint is a colored sphere (rainbow colors)")
        print("3. Compare with joint_indices.txt to see which is which")
        print("\nLook for anatomical landmarks:")
        print("  - Lowest Y value → feet/ankles")
        print("  - Highest Y value → head")
        print("  - Center mass → pelvis/torso")
        print("  - Extreme X values → hands, elbows")
        print("  - Clusters of joints → hands (20 joints each)")

    except Exception as e:
        print(f"  ✗ Error saving: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
