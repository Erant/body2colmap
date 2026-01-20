#!/usr/bin/env python3
"""
Inspect and validate COLMAP format files.

Usage:
    python inspect_colmap.py <colmap_dir>

Reads cameras.txt, images.txt, and points3D.txt and displays:
- Summary statistics
- Format validation
- First few entries for manual inspection
"""

import sys
from pathlib import Path


def inspect_cameras(filepath):
    """Inspect cameras.txt file."""
    print("\n" + "=" * 70)
    print("CAMERAS.TXT")
    print("=" * 70)

    if not filepath.exists():
        print("  ✗ File not found")
        return False

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse cameras
    cameras = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        cameras.append(line)

    print(f"Number of cameras: {len(cameras)}")

    if cameras:
        print(f"\nFirst camera:")
        parts = cameras[0].split()
        print(f"  CAMERA_ID: {parts[0]}")
        print(f"  MODEL: {parts[1]}")
        print(f"  WIDTH: {parts[2]}")
        print(f"  HEIGHT: {parts[3]}")
        print(f"  PARAMS: {' '.join(parts[4:])}")

        if parts[1] == "PINHOLE":
            fx, fy, cx, cy = parts[4:8]
            print(f"\n  Intrinsics (PINHOLE):")
            print(f"    fx = {fx}")
            print(f"    fy = {fy}")
            print(f"    cx = {cx}")
            print(f"    cy = {cy}")

    return True


def inspect_images(filepath):
    """Inspect images.txt file."""
    print("\n" + "=" * 70)
    print("IMAGES.TXT")
    print("=" * 70)

    if not filepath.exists():
        print("  ✗ File not found")
        return False

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse images (2 lines per image)
    images = []
    i = 0
    while i < len(lines):
        line1 = lines[i].strip()
        i += 1

        if not line1 or line1.startswith('#'):
            continue

        # Get second line (points2D)
        line2 = lines[i].strip() if i < len(lines) else ""
        i += 1

        images.append((line1, line2))

    print(f"Number of images: {len(images)}")

    if images:
        print(f"\nFirst image:")
        parts = images[0][0].split()
        print(f"  IMAGE_ID: {parts[0]}")
        print(f"  QW QX QY QZ: {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
        print(f"  TX TY TZ: {parts[5]} {parts[6]} {parts[7]}")
        print(f"  CAMERA_ID: {parts[8]}")
        print(f"  NAME: {parts[9]}")
        print(f"  POINTS2D: {images[0][1] if images[0][1] else '(empty)'}")

        # Validate quaternion
        qw, qx, qy, qz = map(float, parts[1:5])
        quat_norm = (qw**2 + qx**2 + qy**2 + qz**2)**0.5
        print(f"\n  Quaternion norm: {quat_norm:.6f} (should be ~1.0)")
        if abs(quat_norm - 1.0) > 0.01:
            print(f"  ⚠ Warning: Quaternion not normalized!")

        print(f"\nLast image:")
        parts = images[-1][0].split()
        print(f"  IMAGE_ID: {parts[0]}")
        print(f"  NAME: {parts[9]}")
        print(f"  TX TY TZ: {parts[5]} {parts[6]} {parts[7]}")

    # Check line count
    non_comment_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
    expected_lines = len(images) * 2
    if len(non_comment_lines) != expected_lines:
        print(f"\n  ⚠ Warning: Expected {expected_lines} non-comment lines, found {len(non_comment_lines)}")

    return True


def inspect_points3d(filepath):
    """Inspect points3D.txt file."""
    print("\n" + "=" * 70)
    print("POINTS3D.TXT")
    print("=" * 70)

    if not filepath.exists():
        print("  ✗ File not found")
        return False

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse points
    points = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        points.append(line)

    print(f"Number of points: {len(points)}")

    if points:
        print(f"\nFirst point:")
        parts = points[0].split()
        print(f"  POINT3D_ID: {parts[0]}")
        print(f"  XYZ: {parts[1]} {parts[2]} {parts[3]}")
        print(f"  RGB: {parts[4]} {parts[5]} {parts[6]}")
        print(f"  ERROR: {parts[7]}")
        print(f"  TRACK: {' '.join(parts[8:]) if len(parts) > 8 else '(empty)'}")

        # Compute point cloud bounds
        import numpy as np
        xyz = []
        for point_line in points[:min(1000, len(points))]:  # Sample first 1000
            parts = point_line.split()
            xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
        xyz = np.array(xyz)

        print(f"\nPoint cloud bounds (sampled from first {len(xyz)} points):")
        print(f"  X: [{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}]")
        print(f"  Y: [{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}]")
        print(f"  Z: [{xyz[:, 2].min():.3f}, {xyz[:, 2].max():.3f}]")
        print(f"  Centroid: [{xyz[:, 0].mean():.3f}, {xyz[:, 1].mean():.3f}, {xyz[:, 2].mean():.3f}]")

    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_colmap.py <colmap_dir>")
        print("\nInspects COLMAP format files and displays summary statistics.")
        sys.exit(1)

    colmap_dir = Path(sys.argv[1])

    if not colmap_dir.exists():
        print(f"Error: Directory not found: {colmap_dir}")
        sys.exit(1)

    print("=" * 70)
    print("COLMAP File Inspector")
    print("=" * 70)
    print(f"Directory: {colmap_dir}")

    # Inspect each file
    cameras_ok = inspect_cameras(colmap_dir / "cameras.txt")
    images_ok = inspect_images(colmap_dir / "images.txt")
    points_ok = inspect_points3d(colmap_dir / "points3D.txt")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    status = []
    if cameras_ok:
        status.append("✓ cameras.txt")
    else:
        status.append("✗ cameras.txt")

    if images_ok:
        status.append("✓ images.txt")
    else:
        status.append("✗ images.txt")

    if points_ok:
        status.append("✓ points3D.txt")
    else:
        status.append("✗ points3D.txt")

    for s in status:
        print(f"  {s}")

    if all([cameras_ok, images_ok, points_ok]):
        print("\n✓ All COLMAP files present and readable")
    else:
        print("\n✗ Some files missing or unreadable")


if __name__ == "__main__":
    main()
