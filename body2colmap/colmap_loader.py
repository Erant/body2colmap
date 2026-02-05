"""
Lightweight COLMAP text file parser.

Parses cameras.txt, images.txt, and points3D.txt without requiring the
pycolmap package. Designed to load training data produced by our own
exporter, but handles the general COLMAP text format.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from numpy.typing import NDArray


@dataclass
class ColmapDataset:
    """Parsed COLMAP dataset ready for training.

    Attributes:
        camtoworlds: (N, 4, 4) camera-to-world matrices (OpenCV convention).
        Ks: (N, 3, 3) per-image intrinsic matrices.
        image_paths: Absolute paths to training images.
        image_sizes: (N, 2) per-image (width, height).
        points: (M, 3) initial 3D point cloud.
        points_rgb: (M, 3) point colours 0-255.
        scene_scale: Scalar – max camera distance from centroid.
    """

    camtoworlds: NDArray[np.float32]
    Ks: NDArray[np.float32]
    image_paths: List[Path]
    image_sizes: NDArray[np.int32]
    points: NDArray[np.float32]
    points_rgb: NDArray[np.uint8]
    scene_scale: float


def _quaternion_wxyz_to_rotation(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert COLMAP quaternion (w,x,y,z) to 3x3 rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z,  2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,       1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def _parse_cameras_txt(path: Path) -> dict:
    """Parse cameras.txt → {camera_id: (model, width, height, params)}."""
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            cameras[cam_id] = (model, width, height, params)
    return cameras


def _parse_images_txt(path: Path) -> list:
    """Parse images.txt → list of (image_id, qw,qx,qy,qz, tx,ty,tz, cam_id, name).

    images.txt alternates: pose line, then points2D line (which we skip).
    """
    images = []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
    # Every other line is the pose, the alternate is 2D points
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        image_id = int(parts[0])
        qw, qx, qy, qz = (float(parts[j]) for j in range(1, 5))
        tx, ty, tz = (float(parts[j]) for j in range(5, 8))
        cam_id = int(parts[8])
        name = parts[9]
        images.append((image_id, np.array([qw, qx, qy, qz]),
                        np.array([tx, ty, tz]), cam_id, name))
    return images


def _parse_points3d_txt(path: Path) -> Tuple[NDArray, NDArray]:
    """Parse points3D.txt → (points (M,3), colors (M,3))."""
    pts, cols = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            cols.append([int(parts[4]), int(parts[5]), int(parts[6])])
    if not pts:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    return np.array(pts, dtype=np.float32), np.array(cols, dtype=np.uint8)


def _intrinsic_matrix(model: str, params: list, width: int, height: int) -> NDArray:
    """Build 3x3 intrinsic matrix from COLMAP camera model + params."""
    if model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    elif model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        fx = fy = f
    elif model in ("RADIAL", "OPENCV", "OPENCV_FISHEYE"):
        fx, fy, cx, cy = params[:4]
    elif model == "SIMPLE_RADIAL":
        f, cx, cy = params[:3]
        fx = fy = f
    else:
        # Best effort: assume first params are fx, fy, cx, cy
        fx, fy, cx, cy = params[:4]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def load_colmap(data_dir: str, images_dir: Optional[str] = None) -> ColmapDataset:
    """Load a COLMAP text-format dataset.

    Looks for ``sparse/0/`` (or ``sparse/``) for the reconstruction and
    ``images/`` for the training images, following the conventional layout::

        data_dir/
        ├── sparse/0/          (cameras.txt, images.txt, points3D.txt)
        └── images/            (*.png / *.jpg)

    Args:
        data_dir: Root of the dataset.
        images_dir: Explicit image directory.  If *None*, auto-detected
            as ``<data_dir>/images`` or the parent of the sparse dir.

    Returns:
        A :class:`ColmapDataset` with all fields populated.
    """
    root = Path(data_dir)

    # Locate sparse reconstruction
    sparse_dir = None
    for candidate in [root / "sparse" / "0", root / "sparse", root]:
        if (candidate / "cameras.txt").exists():
            sparse_dir = candidate
            break
    if sparse_dir is None:
        raise FileNotFoundError(
            f"Cannot find cameras.txt in {root}/sparse/0/, {root}/sparse/, or {root}/"
        )

    # Locate image directory
    if images_dir is not None:
        img_dir = Path(images_dir)
    elif (root / "images").is_dir():
        img_dir = root / "images"
    else:
        img_dir = root

    # Parse COLMAP text files
    cameras = _parse_cameras_txt(sparse_dir / "cameras.txt")
    images = _parse_images_txt(sparse_dir / "images.txt")
    points, points_rgb = _parse_points3d_txt(sparse_dir / "points3D.txt")

    # Sort images by id for deterministic order
    images.sort(key=lambda x: x[0])

    # Build per-image arrays
    camtoworlds_list = []
    Ks_list = []
    paths_list = []
    sizes_list = []

    for _img_id, qvec, tvec, cam_id, name in images:
        # w2c rotation and translation from COLMAP
        R_w2c = _quaternion_wxyz_to_rotation(qvec)
        t_w2c = tvec

        # Invert to camera-to-world
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R_c2w.astype(np.float32)
        c2w[:3, 3] = t_c2w.astype(np.float32)
        camtoworlds_list.append(c2w)

        model, w, h, params = cameras[cam_id]
        Ks_list.append(_intrinsic_matrix(model, params, w, h))
        sizes_list.append([w, h])

        img_path = img_dir / name
        paths_list.append(img_path)

    camtoworlds = np.stack(camtoworlds_list)  # (N, 4, 4)
    Ks = np.stack(Ks_list)                    # (N, 3, 3)
    image_sizes = np.array(sizes_list, dtype=np.int32)

    # Scene scale: max distance of any camera from the mean camera position
    cam_positions = camtoworlds[:, :3, 3]
    center = cam_positions.mean(axis=0)
    dists = np.linalg.norm(cam_positions - center, axis=1)
    scene_scale = float(dists.max()) if len(dists) > 0 else 1.0

    return ColmapDataset(
        camtoworlds=camtoworlds,
        Ks=Ks,
        image_paths=paths_list,
        image_sizes=image_sizes,
        points=points,
        points_rgb=points_rgb,
        scene_scale=scene_scale,
    )
