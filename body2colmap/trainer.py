"""
3D Gaussian Splat trainer using gsplat.

Reads a standard COLMAP dataset (cameras.txt, images.txt, points3D.txt + images/)
and trains 3D Gaussians using gsplat's rasterization() and DefaultStrategy.

Supports RGBA images with alpha-masked loss (transparent background).
Auto-scales K matrix when actual image size differs from COLMAP's recorded size.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ColmapCamera:
    """Parsed COLMAP camera intrinsics."""

    camera_id: int
    model: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class ColmapImage:
    """Parsed COLMAP image extrinsics."""

    image_id: int
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    name: str


@dataclass
class ColmapDataset:
    """Full COLMAP dataset ready for training."""

    # Per-image data
    c2w_matrices: Tensor  # (N, 4, 4) camera-to-world
    K_matrices: Tensor  # (N, 3, 3) intrinsics (scaled to actual image size)
    images: List[Tensor]  # list of (H, W, 3) float32 RGB in [0,1]
    alphas: List[Optional[Tensor]]  # list of (H, W, 1) float32 or None
    image_names: List[str]
    widths: List[int]
    heights: List[int]

    # Point cloud
    points: Tensor  # (M, 3) positions
    colors: Tensor  # (M, 3) RGB in [0, 1]

    @property
    def n_images(self) -> int:
        return len(self.images)

    @property
    def n_points(self) -> int:
        return len(self.points)


@dataclass
class TrainResult:
    """Result of training."""

    # Trained Gaussian parameters (all on CPU)
    means: Tensor  # (N, 3)
    scales: Tensor  # (N, 3) log-space
    quats: Tensor  # (N, 4) wxyz
    opacities: Tensor  # (N,) logit-space
    sh0: Tensor  # (N, 1, 3) DC spherical harmonic
    sh_rest: Tensor  # (N, K, 3) higher-order SH

    # Per-Gaussian statistics from DefaultStrategy
    grad2d: Tensor  # gradient accumulator
    view_count: Tensor  # how many views each Gaussian was seen in

    # Training metrics
    final_loss: float
    n_gaussians: int
    sh_degree: int


# ---------------------------------------------------------------------------
# COLMAP parsing
# ---------------------------------------------------------------------------


def _parse_cameras_txt(filepath: Path) -> Dict[int, ColmapCamera]:
    """Parse cameras.txt → dict of camera_id → ColmapCamera."""
    cameras = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            if model == "PINHOLE":
                fx, fy, cx, cy = (float(x) for x in parts[4:8])
            elif model == "SIMPLE_PINHOLE":
                f_val = float(parts[4])
                cx, cy = float(parts[5]), float(parts[6])
                fx = fy = f_val
            else:
                raise ValueError(f"Unsupported camera model: {model}")
            cameras[camera_id] = ColmapCamera(
                camera_id=camera_id,
                model=model,
                width=width,
                height=height,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
            )
    return cameras


def _parse_images_txt(filepath: Path) -> List[ColmapImage]:
    """Parse images.txt → list of ColmapImage.

    COLMAP format: two lines per image — an extrinsic line (10+ fields) followed
    by a points2D line (which may be empty/blank). We identify extrinsic lines by
    checking that they have at least 10 whitespace-separated fields.
    """
    images = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Extrinsic lines have at least 10 fields:
            # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            # Points2D lines have triplets (X Y POINT3D_ID) or are empty
            if len(parts) >= 10:
                images.append(
                    ColmapImage(
                        image_id=int(parts[0]),
                        qw=float(parts[1]),
                        qx=float(parts[2]),
                        qy=float(parts[3]),
                        qz=float(parts[4]),
                        tx=float(parts[5]),
                        ty=float(parts[6]),
                        tz=float(parts[7]),
                        camera_id=int(parts[8]),
                        name=parts[9],
                    )
                )
            # else: points2D line — skip
    return images


def _parse_points3d_txt(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse points3D.txt → (positions (M,3), colors (M,3))."""
    positions = []
    colors = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            colors.append([int(parts[4]), int(parts[5]), int(parts[6])])
    if not positions:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    return (
        np.array(positions, dtype=np.float32),
        np.array(colors, dtype=np.float32) / 255.0,
    )


def _quat_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Quaternion (w,x,y,z) → 3x3 rotation matrix."""
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_colmap_dataset(
    colmap_dir: str,
    device: str = "cuda",
) -> ColmapDataset:
    """
    Load a full COLMAP dataset for training.

    Expects directory structure:
        colmap_dir/
        ├── sparse/0/
        │   ├── cameras.txt
        │   ├── images.txt
        │   └── points3D.txt
        └── images/
            └── *.png

    Auto-scales K matrix if actual image dimensions differ from COLMAP's recorded
    dimensions (e.g. if images were resized after COLMAP processing).

    Args:
        colmap_dir: Path to COLMAP dataset root
        device: torch device

    Returns:
        ColmapDataset with all data loaded as tensors
    """
    import cv2

    colmap_dir = Path(colmap_dir)
    sparse_dir = colmap_dir / "sparse" / "0"
    images_dir = colmap_dir / "images"

    # If sparse/0 doesn't exist, check if files are directly in colmap_dir
    if not sparse_dir.exists():
        if (colmap_dir / "cameras.txt").exists():
            sparse_dir = colmap_dir
            # images might be in same dir or a sibling "images" dir
            if not images_dir.exists():
                images_dir = colmap_dir
        else:
            raise FileNotFoundError(
                f"Cannot find COLMAP files. Looked in {sparse_dir} "
                f"and {colmap_dir}"
            )

    # Parse COLMAP text files
    cameras = _parse_cameras_txt(sparse_dir / "cameras.txt")
    colmap_images = _parse_images_txt(sparse_dir / "images.txt")
    points_np, colors_np = _parse_points3d_txt(sparse_dir / "points3D.txt")

    # Sort images by name for deterministic ordering
    colmap_images.sort(key=lambda img: img.name)

    # Load images and build tensors
    c2w_list = []
    K_list = []
    rgb_list = []
    alpha_list = []
    name_list = []
    width_list = []
    height_list = []

    for cimg in colmap_images:
        cam = cameras[cimg.camera_id]

        # Load image
        img_path = images_dir / cimg.name
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")

        actual_h, actual_w = img.shape[:2]

        # Separate RGB and alpha
        if img.shape[2] == 4:
            # BGRA → RGBA
            rgb_np = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            alpha_np = img[:, :, 3:4].astype(np.float32) / 255.0
        else:
            # BGR → RGB
            rgb_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            alpha_np = None

        # Build K matrix, auto-scale if image size differs from COLMAP record
        scale_x = actual_w / cam.width
        scale_y = actual_h / cam.height
        K = np.array(
            [
                [cam.fx * scale_x, 0.0, cam.cx * scale_x],
                [0.0, cam.fy * scale_y, cam.cy * scale_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        # Build c2w from COLMAP's w2c quaternion+translation
        # COLMAP stores w2c: R_w2c, t_w2c such that x_cam = R_w2c @ x_world + t_w2c
        R_w2c = _quat_to_rotation(cimg.qw, cimg.qx, cimg.qy, cimg.qz)
        t_w2c = np.array([cimg.tx, cimg.ty, cimg.tz], dtype=np.float64)

        # Invert to get c2w
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R_c2w.astype(np.float32)
        c2w[:3, 3] = t_c2w.astype(np.float32)

        c2w_list.append(torch.from_numpy(c2w))
        K_list.append(torch.from_numpy(K))
        rgb_list.append(torch.from_numpy(rgb_np))
        alpha_list.append(
            torch.from_numpy(alpha_np) if alpha_np is not None else None
        )
        name_list.append(cimg.name)
        width_list.append(actual_w)
        height_list.append(actual_h)

    c2w_matrices = torch.stack(c2w_list).to(device)
    K_matrices = torch.stack(K_list).to(device)
    points = torch.from_numpy(points_np).to(device)
    colors = torch.from_numpy(colors_np).to(device)

    # Print diagnostics (once at start)
    print(f"Loaded {len(rgb_list)} images, {len(points)} initial points")
    if rgb_list:
        print(f"Image size: {width_list[0]}x{height_list[0]}")
        if width_list[0] != cameras[colmap_images[0].camera_id].width:
            print(
                f"  K scaled: COLMAP records "
                f"{cameras[colmap_images[0].camera_id].width}x"
                f"{cameras[colmap_images[0].camera_id].height}"
                f" → actual {width_list[0]}x{height_list[0]}"
            )
    if alpha_list[0] is not None:
        a = alpha_list[0]
        print(f"Alpha: min={a.min().item():.3f} max={a.max().item():.3f} mean={a.mean().item():.3f}")
    sample_K = K_list[0]
    print(
        f"K: fx={sample_K[0,0].item():.1f} fy={sample_K[1,1].item():.1f} "
        f"cx={sample_K[0,2].item():.1f} cy={sample_K[1,2].item():.1f}"
    )

    return ColmapDataset(
        c2w_matrices=c2w_matrices,
        K_matrices=K_matrices,
        images=rgb_list,
        alphas=alpha_list,
        image_names=name_list,
        widths=width_list,
        heights=height_list,
        points=points,
        colors=colors,
    )


# ---------------------------------------------------------------------------
# Gaussian initialization
# ---------------------------------------------------------------------------


def _knn_distances(points: Tensor, k: int = 3) -> Tensor:
    """Compute average distance to k nearest neighbors for each point."""
    # Use cdist for pairwise distances (works for reasonable point counts)
    dists = torch.cdist(points[None], points[None])[0]  # (N, N)
    # Set diagonal to large value so a point doesn't match itself
    dists.fill_diagonal_(float("inf"))
    # Get k smallest distances per point
    knn_dists, _ = dists.topk(k, largest=False, dim=-1)  # (N, k)
    return knn_dists.mean(dim=-1)  # (N,)


def init_gaussians(
    points: Tensor,
    colors: Tensor,
    sh_degree: int = 3,
    device: str = "cuda",
) -> Dict[str, Tensor]:
    """
    Initialize Gaussian parameters from a point cloud.

    Args:
        points: (M, 3) initial positions
        colors: (M, 3) RGB in [0, 1]
        sh_degree: max SH degree
        device: torch device

    Returns:
        Dict of parameter name → tensor (all require grad)
    """
    N = len(points)
    n_sh_coeffs = (sh_degree + 1) ** 2

    means = points.clone().float().to(device)

    # Scale from KNN distances (or a reasonable default)
    if N > 1:
        avg_dist = _knn_distances(means, k=min(3, N - 1))
        scales = torch.log(avg_dist.clamp(min=1e-7)).unsqueeze(-1).expand(-1, 3).clone()
    else:
        scales = torch.zeros(N, 3, device=device)

    quats = torch.zeros(N, 4, device=device)
    quats[:, 0] = 1.0  # identity rotation (w=1, x=y=z=0)

    opacities = torch.full((N,), fill_value=-2.197, device=device)  # logit(0.1)

    # SH coefficients: DC from colors, rest zeroed
    C0 = 0.28209479177387814
    sh0 = ((colors.float().to(device) - 0.5) / C0).unsqueeze(1)  # (N, 1, 3)
    sh_rest = torch.zeros(N, n_sh_coeffs - 1, 3, device=device)  # (N, K, 3)

    params = {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "sh0": sh0,
        "sh_rest": sh_rest,
    }

    # Enable gradients
    for p in params.values():
        p.requires_grad_(True)

    return params


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _compute_scene_scale(c2w_matrices: Tensor) -> float:
    """Compute scene scale from camera positions for LR adjustment."""
    cam_positions = c2w_matrices[:, :3, 3]  # (N, 3)
    center = cam_positions.mean(dim=0)
    dists = torch.norm(cam_positions - center, dim=-1)
    return max(dists.max().item(), 1e-6)


def train(
    colmap_dir: str,
    output_dir: str,
    max_steps: int = 30000,
    sh_degree: int = 3,
    device: str = "cuda",
    bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ssim_lambda: float = 0.2,
    lr_means_base: float = 1.6e-4,
    lr_scales: float = 5e-3,
    lr_quats: float = 1e-3,
    lr_opacities: float = 5e-2,
    lr_sh0: float = 0.25,
    lr_sh_rest_factor: float = 0.05,
    sh_degree_interval: int = 1000,
    refine_start: int = 500,
    refine_stop: int = 15000,
    refine_every: int = 100,
    grow_grad2d: float = 0.0002,
    prune_opacity: float = 0.005,
    verbose: bool = True,
) -> TrainResult:
    """
    Train 3D Gaussians from a COLMAP dataset.

    Args:
        colmap_dir: Path to COLMAP dataset
        output_dir: Path to save output PLY
        max_steps: Number of training iterations
        sh_degree: Maximum spherical harmonics degree
        device: Torch device
        bg_color: Background color (R, G, B) in [0, 1]
        ssim_lambda: Weight for SSIM loss (L1 weight = 1 - ssim_lambda)
        lr_means_base: Base learning rate for positions (scaled by scene_scale)
        lr_scales: Learning rate for scales
        lr_quats: Learning rate for quaternions
        lr_opacities: Learning rate for opacities
        lr_sh0: Learning rate for DC spherical harmonics
        lr_sh_rest_factor: Factor for higher-order SH LR (lr_sh0 * factor)
        sh_degree_interval: Steps between SH degree increases
        refine_start: Step to start densification
        refine_stop: Step to stop densification
        refine_every: Densification interval
        grow_grad2d: Gradient threshold for densification
        prune_opacity: Opacity threshold for pruning
        verbose: Print training progress

    Returns:
        TrainResult with trained parameters and statistics
    """
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy

    # Load data
    dataset = load_colmap_dataset(colmap_dir, device=device)

    if dataset.n_points == 0:
        raise ValueError("No initial points in points3D.txt — cannot initialize Gaussians")

    # Scene scale for LR
    scene_scale = _compute_scene_scale(dataset.c2w_matrices)
    if verbose:
        print(f"Scene scale: {scene_scale:.4f}")

    # Initialize Gaussians
    params = init_gaussians(
        dataset.points, dataset.colors, sh_degree=sh_degree, device=device
    )

    if verbose:
        print(f"Initialized {len(params['means'])} Gaussians")

    # Background color tensor
    bg = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # Optimizers — one per parameter group for DefaultStrategy compatibility
    lr_means = lr_means_base * scene_scale
    lr_sh_rest = lr_sh0 * lr_sh_rest_factor

    optimizers = {
        "means": torch.optim.Adam([params["means"]], lr=lr_means, eps=1e-15),
        "scales": torch.optim.Adam([params["scales"]], lr=lr_scales, eps=1e-15),
        "quats": torch.optim.Adam([params["quats"]], lr=lr_quats, eps=1e-15),
        "opacities": torch.optim.Adam(
            [params["opacities"]], lr=lr_opacities, eps=1e-15
        ),
        "sh0": torch.optim.Adam([params["sh0"]], lr=lr_sh0, eps=1e-15),
        "sh_rest": torch.optim.Adam(
            [params["sh_rest"]], lr=lr_sh_rest, eps=1e-15
        ),
    }

    # DefaultStrategy for densification
    strategy = DefaultStrategy(
        refine_start_iter=refine_start,
        refine_stop_iter=refine_stop,
        refine_every=refine_every,
        grow_grad2d=grow_grad2d,
        prune_opa=prune_opacity,
        verbose=verbose,
    )
    strategy_state = strategy.initialize_state()

    # Current SH degree (progressively increased)
    current_sh_degree = 0

    # Training loop
    final_loss_val = 0.0
    for step in range(max_steps):
        # Increase SH degree on schedule
        if sh_degree_interval > 0 and step > 0 and step % sh_degree_interval == 0:
            if current_sh_degree < sh_degree:
                current_sh_degree += 1
                if verbose:
                    print(f"Step {step}: SH degree → {current_sh_degree}")

        # Sample random view
        idx = torch.randint(0, dataset.n_images, (1,)).item()
        gt_rgb = dataset.images[idx].to(device)  # (H, W, 3)
        gt_alpha = dataset.alphas[idx]  # (H, W, 1) or None
        if gt_alpha is not None:
            gt_alpha = gt_alpha.to(device)
        c2w = dataset.c2w_matrices[idx]  # (4, 4)
        K = dataset.K_matrices[idx]  # (3, 3)
        width = dataset.widths[idx]
        height = dataset.heights[idx]

        # viewmat = inverse(c2w) = w2c
        viewmat = torch.linalg.inv(c2w)

        # Composite GT over background (same as gsplat does internally)
        if gt_alpha is not None:
            gt_composited = gt_rgb * gt_alpha + bg * (1.0 - gt_alpha)
        else:
            gt_composited = gt_rgb

        # Rasterize
        colors_sh = torch.cat([params["sh0"], params["sh_rest"]], dim=1)  # (N, K, 3)

        renders, render_alphas, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=colors_sh,
            viewmats=viewmat.unsqueeze(0),  # (1, 4, 4)
            Ks=K.unsqueeze(0),  # (1, 3, 3)
            width=width,
            height=height,
            sh_degree=current_sh_degree,
            backgrounds=bg,  # (3,) — no batch dim needed
        )
        # renders: (1, H, W, 3), render_alphas: (1, H, W, 1)
        rendered = renders[0]  # (H, W, 3)

        # Loss
        if gt_alpha is not None:
            # Alpha-weighted L1: expand alpha to 3 channels for correct normalization
            alpha_rgb = gt_alpha.expand_as(rendered)  # (H, W, 3)
            l1_loss = (
                (torch.abs(rendered - gt_composited) * alpha_rgb).sum()
                / alpha_rgb.sum().clamp(min=1.0)
            )
        else:
            l1_loss = F.l1_loss(rendered, gt_composited)

        # SSIM loss (if weight > 0)
        if ssim_lambda > 0:
            # Convert to (1, 3, H, W) for SSIM
            rendered_nchw = rendered.permute(2, 0, 1).unsqueeze(0)
            gt_nchw = gt_composited.permute(2, 0, 1).unsqueeze(0)
            ssim_val = _ssim(rendered_nchw, gt_nchw)
            loss = (1.0 - ssim_lambda) * l1_loss + ssim_lambda * (1.0 - ssim_val)
        else:
            loss = l1_loss

        # Backward
        loss.backward()

        final_loss_val = loss.item()

        # Densification strategy
        strategy.step_pre_backward(
            params=params,
            optimizers=optimizers,
            state=strategy_state,
            step=step,
            info=info,
        )

        strategy.step_post_backward(
            params=params,
            optimizers=optimizers,
            state=strategy_state,
            step=step,
            info=info,
        )

        # Optimizer step
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

        # Logging
        if verbose and (step % 1000 == 0 or step == max_steps - 1):
            n_gs = len(params["means"])
            print(
                f"Step {step:>6d}/{max_steps} | "
                f"loss={final_loss_val:.5f} | "
                f"gaussians={n_gs}"
            )

    # Collect results
    grad2d = strategy_state.get("grad2d", torch.zeros(len(params["means"])))
    view_count = strategy_state.get(
        "count", torch.zeros(len(params["means"]), dtype=torch.int32)
    )

    # Save PLY
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ply_path = output_path / "point_cloud.ply"
    save_ply(
        ply_path,
        means=params["means"].detach(),
        scales=params["scales"].detach(),
        quats=params["quats"].detach(),
        opacities=params["opacities"].detach(),
        sh0=params["sh0"].detach(),
        sh_rest=params["sh_rest"].detach(),
        grad2d=grad2d if isinstance(grad2d, Tensor) else None,
        view_count=view_count if isinstance(view_count, Tensor) else None,
    )
    if verbose:
        print(f"Saved {len(params['means'])} Gaussians to {ply_path}")

    return TrainResult(
        means=params["means"].detach().cpu(),
        scales=params["scales"].detach().cpu(),
        quats=params["quats"].detach().cpu(),
        opacities=params["opacities"].detach().cpu(),
        sh0=params["sh0"].detach().cpu(),
        sh_rest=params["sh_rest"].detach().cpu(),
        grad2d=grad2d.detach().cpu() if isinstance(grad2d, Tensor) else torch.tensor(grad2d),
        view_count=view_count.detach().cpu() if isinstance(view_count, Tensor) else torch.tensor(view_count),
        final_loss=final_loss_val,
        n_gaussians=len(params["means"]),
        sh_degree=sh_degree,
    )


# ---------------------------------------------------------------------------
# SSIM (simplified, window_size=11)
# ---------------------------------------------------------------------------


def _ssim(
    img1: Tensor, img2: Tensor, window_size: int = 11
) -> Tensor:
    """Compute SSIM between two images (N, C, H, W)."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    C = img1.shape[1]

    # Create Gaussian window
    gauss = torch.exp(
        -torch.arange(window_size, dtype=torch.float32, device=img1.device)
        .sub(window_size // 2)
        .pow(2)
        / (2 * 1.5 ** 2)
    )
    gauss = gauss / gauss.sum()
    window_1d = gauss.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    window = window_2d.expand(C, 1, window_size, window_size).contiguous()

    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=C)
    mu2 = F.conv2d(img2, window, padding=pad, groups=C)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


# ---------------------------------------------------------------------------
# PLY export
# ---------------------------------------------------------------------------


def save_ply(
    filepath: Path,
    means: Tensor,
    scales: Tensor,
    quats: Tensor,
    opacities: Tensor,
    sh0: Tensor,
    sh_rest: Tensor,
    grad2d: Optional[Tensor] = None,
    view_count: Optional[Tensor] = None,
) -> None:
    """
    Save trained Gaussians as PLY file in standard 3DGS format.

    Properties:
        x, y, z             - Position
        scale_0, scale_1, scale_2  - Log-space scales
        rot_0, rot_1, rot_2, rot_3 - Quaternion (wxyz)
        opacity             - Logit-space opacity
        f_dc_0, f_dc_1, f_dc_2    - DC spherical harmonic (RGB)
        f_rest_*            - Higher-order SH coefficients
        grad2d              - Accumulated 2D gradient (from DefaultStrategy)
        view_count          - Number of views each Gaussian was observed in

    Args:
        filepath: Output path
        means: (N, 3)
        scales: (N, 3) log-space
        quats: (N, 4) wxyz
        opacities: (N,) logit-space
        sh0: (N, 1, 3) DC SH
        sh_rest: (N, K, 3) higher-order SH
        grad2d: (N,) gradient accumulator from DefaultStrategy
        view_count: (N,) per-Gaussian observation count
    """
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        raise ImportError(
            "plyfile is required for PLY export. Install with: pip install plyfile"
        )

    N = len(means)
    means_np = means.cpu().numpy()
    scales_np = scales.cpu().numpy()
    quats_np = quats.cpu().numpy()
    opacities_np = opacities.cpu().numpy()
    sh0_np = sh0.cpu().numpy().squeeze(1)  # (N, 3)
    sh_rest_np = sh_rest.cpu().numpy()  # (N, K, 3)

    n_sh_rest = sh_rest_np.shape[1]

    # Build structured array
    dtype_fields = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
    ]
    # Normals (placeholder for compat)
    dtype_fields += [("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    # DC SH
    dtype_fields += [("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")]
    # Rest SH — interleaved per-channel as expected by standard 3DGS readers
    for i in range(n_sh_rest):
        dtype_fields.append((f"f_rest_{i}", "f4"))
    for i in range(n_sh_rest):
        dtype_fields.append((f"f_rest_{i + n_sh_rest}", "f4"))
    for i in range(n_sh_rest):
        dtype_fields.append((f"f_rest_{i + 2 * n_sh_rest}", "f4"))
    # Opacity
    dtype_fields.append(("opacity", "f4"))
    # Scales
    dtype_fields += [("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4")]
    # Rotations
    dtype_fields += [
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ]
    # Training statistics (for renderer use)
    dtype_fields.append(("grad2d", "f4"))
    dtype_fields.append(("view_count", "u4"))

    arr = np.zeros(N, dtype=dtype_fields)
    arr["x"] = means_np[:, 0]
    arr["y"] = means_np[:, 1]
    arr["z"] = means_np[:, 2]
    arr["nx"] = 0.0
    arr["ny"] = 0.0
    arr["nz"] = 0.0
    arr["f_dc_0"] = sh0_np[:, 0]
    arr["f_dc_1"] = sh0_np[:, 1]
    arr["f_dc_2"] = sh0_np[:, 2]

    # SH rest: channel-interleaved
    # sh_rest_np is (N, K, 3) — K coefficients, 3 channels
    for i in range(n_sh_rest):
        arr[f"f_rest_{i}"] = sh_rest_np[:, i, 0]  # R channel
        arr[f"f_rest_{i + n_sh_rest}"] = sh_rest_np[:, i, 1]  # G channel
        arr[f"f_rest_{i + 2 * n_sh_rest}"] = sh_rest_np[:, i, 2]  # B channel

    arr["opacity"] = opacities_np
    arr["scale_0"] = scales_np[:, 0]
    arr["scale_1"] = scales_np[:, 1]
    arr["scale_2"] = scales_np[:, 2]
    arr["rot_0"] = quats_np[:, 0]
    arr["rot_1"] = quats_np[:, 1]
    arr["rot_2"] = quats_np[:, 2]
    arr["rot_3"] = quats_np[:, 3]

    # Training statistics
    if grad2d is not None:
        arr["grad2d"] = grad2d.cpu().numpy()
    if view_count is not None:
        arr["view_count"] = view_count.cpu().numpy().astype(np.uint32)

    el = PlyElement.describe(arr, "vertex")
    PlyData([el]).write(str(filepath))
