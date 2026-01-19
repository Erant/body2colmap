#!/usr/bin/env python3
"""
Create a simple test .npz file for testing the pipeline.

This generates a minimal valid SAM-3D-Body output with a simple cube mesh.
Useful for testing the pipeline without needing actual SAM-3D-Body output.

Usage:
    python create_test_data.py [output.npz]
"""

import sys
import numpy as np


def create_simple_cube_mesh():
    """
    Create a simple cube mesh for testing.

    Returns vertices, faces as if from SAM-3D-Body.
    """
    # Cube vertices (centered at origin)
    vertices = np.array([
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ], dtype=np.float32)

    # Cube faces (12 triangles, 2 per face)
    faces = np.array([
        # Front
        [0, 1, 2], [0, 2, 3],
        # Back
        [4, 6, 5], [4, 7, 6],
        # Left
        [0, 3, 7], [0, 7, 4],
        # Right
        [1, 5, 6], [1, 6, 2],
        # Bottom
        [0, 4, 5], [0, 5, 1],
        # Top
        [3, 2, 6], [3, 6, 7],
    ], dtype=np.int32)

    return vertices, faces


def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else "test_cube.npz"

    print("Creating test .npz file...")

    # Create simple cube mesh
    vertices, faces = create_simple_cube_mesh()

    # SAM-3D-Body typically positions mesh in front of camera
    # cam_t is the translation that positions the mesh
    cam_t = np.array([0.0, 0.0, 5.0], dtype=np.float32)

    # Save in SAM-3D-Body output format
    np.savez(
        output_path,
        pred_vertices=vertices,
        pred_cam_t=cam_t,
        faces=faces,
        # Optional fields
        focal_length=np.float32(1000.0),
        bbox=np.array([100, 100, 400, 400], dtype=np.float32)
    )

    print(f"✓ Created: {output_path}")
    print("\nFile contents:")
    print(f"  pred_vertices: {vertices.shape} {vertices.dtype}")
    print(f"  pred_cam_t: {cam_t.shape} {cam_t.dtype}")
    print(f"  faces: {faces.shape} {faces.dtype}")
    print("\nTest with:")
    print(f"  python test_basic_render.py {output_path}")


if __name__ == "__main__":
    main()
