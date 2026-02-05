"""
3D Gaussian Splat trainer using gsplat.

Wraps gsplat's rasterization and densification into a self-contained
training loop that reads a COLMAP dataset and produces a trained splat
(PLY or checkpoint) plus per-Gaussian statistics (gradient norms, view
counts).

Typical usage::

    from body2colmap.trainer import SplatTrainer, TrainConfig

    trainer = SplatTrainer("./colmap_output", TrainConfig(max_steps=30_000))
    result = trainer.train()
    result.save_ply("output/point_cloud.ply")
    print(result.avg_grad_norm)   # per-Gaussian average gradient norm
    print(result.view_count)      # per-Gaussian view count

The implementation intentionally keeps things simple: single-GPU, no
distributed training, no appearance embedding, no pose optimisation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .colmap_loader import ColmapDataset, load_colmap


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Training hyper-parameters with sensible defaults."""

    # General
    max_steps: int = 30_000
    batch_size: int = 1
    sh_degree: int = 3
    sh_degree_interval: int = 1000  # increase SH degree every N steps

    # Loss
    ssim_lambda: float = 0.2

    # Learning rates  (scaled by sqrt(batch_size) internally)
    lr_means: float = 1.6e-4
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: Optional[float] = None   # auto from scene_scale if None
    lr_shN: Optional[float] = None   # auto: lr_sh0 / 20

    # Densification (DefaultStrategy)
    strategy: Literal["default", "mcmc"] = "default"
    refine_start: int = 500
    refine_stop: int = 15_000
    refine_every: int = 100
    reset_opacity_every: int = 3000
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    prune_opacity: float = 0.005
    prune_scale3d: float = 0.1
    absgrad: bool = True

    # MCMC-specific (only used when strategy == "mcmc")
    mcmc_cap_max: int = 1_000_000

    # Rasterisation
    packed: bool = False
    antialiased: bool = False
    near_plane: float = 0.01
    far_plane: float = 1e10
    bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # background for alpha compositing
    ignore_alpha: bool = False  # if True, treat all images as opaque (for debugging)
    invert_alpha: bool = False  # if True, use (1 - alpha) instead of alpha

    # Checkpointing / evaluation
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # Device
    device: str = "cuda"

    # Debug
    debug_save_images: bool = False  # save GT vs rendered images during training
    debug_save_every: int = 500  # save debug images every N steps


# ---------------------------------------------------------------------------
# Training result
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    """Result returned after training completes.

    Holds the trained Gaussian parameters as well as accumulated
    per-Gaussian statistics from the densification strategy.
    """

    # Trained parameters (all torch tensors on *cpu*)
    means: Tensor        # [N, 3]
    scales: Tensor       # [N, 3]  (log-space)
    quats: Tensor        # [N, 4]  (wxyz)
    opacities: Tensor    # [N]     (logit-space)
    sh0: Tensor          # [N, 1, 3]
    shN: Tensor          # [N, K, 3]
    sh_degree: int

    # Per-Gaussian statistics accumulated during training
    grad2d: Tensor       # [N]  cumulative 2D gradient norms
    view_count: Tensor   # [N]  number of views each Gaussian was visible in

    # Overall training loss (final step)
    final_loss: float = 0.0

    # Convenience -----------------------------------------------------------

    @property
    def avg_grad_norm(self) -> Tensor:
        """Average 2D gradient norm per Gaussian (grad2d / view_count)."""
        return self.grad2d / self.view_count.clamp(min=1)

    @property
    def n_gaussians(self) -> int:
        return self.means.shape[0]

    # I/O -------------------------------------------------------------------

    def save_ply(self, path: str) -> Path:
        """Export trained splat as a standard 3DGS PLY file."""
        try:
            from plyfile import PlyData, PlyElement
        except ImportError:
            raise ImportError("plyfile is required for PLY export: pip install plyfile")

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        n = self.n_gaussians
        sh_coeffs = torch.cat([self.sh0, self.shN], dim=1)  # [N, K, 3]
        n_sh = sh_coeffs.shape[1]

        # Build structured array
        attrs = ["x", "y", "z"]
        attrs += [f"scale_{i}" for i in range(3)]
        attrs += [f"rot_{i}" for i in range(4)]
        attrs += ["opacity"]
        attrs += [f"f_dc_{i}" for i in range(3)]
        n_rest = n_sh - 1
        for i in range(n_rest):
            for c in range(3):
                attrs += [f"f_rest_{i + c * n_rest}"]

        dtype_list = [(a, "f4") for a in attrs]
        arr = np.empty(n, dtype=dtype_list)

        means_np = self.means.detach().numpy()
        arr["x"] = means_np[:, 0]
        arr["y"] = means_np[:, 1]
        arr["z"] = means_np[:, 2]

        scales_np = self.scales.detach().numpy()
        for i in range(3):
            arr[f"scale_{i}"] = scales_np[:, i]

        quats_np = self.quats.detach().numpy()
        for i in range(4):
            arr[f"rot_{i}"] = quats_np[:, i]

        arr["opacity"] = self.opacities.detach().numpy()

        sh_np = sh_coeffs.detach().numpy()
        for i in range(3):
            arr[f"f_dc_{i}"] = sh_np[:, 0, i]

        if n_rest > 0:
            for i in range(n_rest):
                for c in range(3):
                    arr[f"f_rest_{i + c * n_rest}"] = sh_np[:, 1 + i, c]

        el = PlyElement.describe(arr, "vertex")
        PlyData([el]).write(str(out))
        return out

    def save_checkpoint(self, path: str) -> Path:
        """Save a torch checkpoint with parameters and statistics."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "means": self.means,
            "scales": self.scales,
            "quats": self.quats,
            "opacities": self.opacities,
            "sh0": self.sh0,
            "shN": self.shN,
            "sh_degree": self.sh_degree,
            "grad2d": self.grad2d,
            "view_count": self.view_count,
        }
        torch.save(data, str(out))
        return out


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SplatTrainer:
    """Train 3D Gaussian Splats from a COLMAP dataset.

    Args:
        data_dir: Path to the COLMAP dataset root (with ``sparse/0/`` and
            ``images/``).
        config: Training hyper-parameters.  Uses defaults when *None*.
        images_dir: Explicit image directory override.
    """

    def __init__(
        self,
        data_dir: str,
        config: Optional[TrainConfig] = None,
        images_dir: Optional[str] = None,
    ):
        self.cfg = config or TrainConfig()
        self.device = torch.device(self.cfg.device)
        self.dataset: ColmapDataset = load_colmap(data_dir, images_dir=images_dir)

        # Will be initialised in _setup()
        self.splats: Dict[str, torch.nn.Parameter] = {}
        self.optimizers: Dict[str, torch.optim.Adam] = {}
        self.strategy: Any = None
        self.strategy_state: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_splats(self) -> None:
        """Initialise Gaussian parameters from the SfM point cloud."""
        points = torch.tensor(self.dataset.points, dtype=torch.float32, device=self.device)
        colors_uint8 = self.dataset.points_rgb  # (M, 3) 0-255
        colors_f = torch.tensor(colors_uint8 / 255.0, dtype=torch.float32, device=self.device)

        n = len(points)
        if n == 0:
            raise ValueError("Point cloud is empty – cannot initialise Gaussians.")

        # Scales: initialise from average nearest-neighbour distance
        # Compute K=4 nearest neighbours via brute-force cdist (only runs
        # once at init, so O(N^2) is fine for typical point cloud sizes).
        with torch.no_grad():
            dists = torch.cdist(points, points)  # (N, N) pairwise L2
            # K+1 smallest (includes self at dist 0)
            topk_dists, _ = dists.topk(4, dim=-1, largest=False)
            avg_dist = topk_dists[:, 1:].mean(dim=-1)  # skip self
        scales = torch.log(avg_dist.unsqueeze(-1).expand(-1, 3)).float()

        # Quaternions: identity
        quats = torch.zeros(n, 4, device=self.device)
        quats[:, 0] = 1.0

        # Opacities: inverse sigmoid(0.1) ≈ -2.197
        opacities = torch.full((n,), fill_value=torch.logit(torch.tensor(0.1)).item(),
                               device=self.device)

        # SH coefficients from point colours
        C0 = 0.28209479177387814
        sh0 = ((colors_f - 0.5) / C0).unsqueeze(1)  # (N, 1, 3)

        n_sh_rest = (self.cfg.sh_degree + 1) ** 2 - 1
        shN = torch.zeros(n, n_sh_rest, 3, device=self.device)

        self.splats = {
            "means": torch.nn.Parameter(points),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(quats),
            "opacities": torch.nn.Parameter(opacities),
            "sh0": torch.nn.Parameter(sh0),
            "shN": torch.nn.Parameter(shN),
        }

    def _init_optimizers(self) -> None:
        """Create per-parameter Adam optimisers."""
        BS = self.cfg.batch_size
        scene_scale = self.dataset.scene_scale

        lr_sh0 = self.cfg.lr_sh0 or (0.25 / (20.0 if scene_scale > 10 else 1.0))
        lr_shN = self.cfg.lr_shN or (lr_sh0 / 20.0)

        param_lr = [
            ("means",     self.cfg.lr_means * scene_scale),
            ("scales",    self.cfg.lr_scales),
            ("quats",     self.cfg.lr_quats),
            ("opacities", self.cfg.lr_opacities),
            ("sh0",       lr_sh0),
            ("shN",       lr_shN),
        ]

        self.optimizers = {}
        for name, lr in param_lr:
            lr_scaled = lr * math.sqrt(BS)
            eps = 1e-15 / math.sqrt(BS)
            betas = (1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999))
            self.optimizers[name] = torch.optim.Adam(
                [self.splats[name]],
                lr=lr_scaled,
                eps=eps,
                betas=betas,
            )

    def _init_strategy(self) -> None:
        """Initialise the densification strategy."""
        if self.cfg.strategy == "mcmc":
            from gsplat.strategy import MCMCStrategy  # type: ignore
            self.strategy = MCMCStrategy(
                cap_max=self.cfg.mcmc_cap_max,
                refine_start_iter=self.cfg.refine_start,
                refine_stop_iter=self.cfg.refine_stop,
                refine_every=self.cfg.refine_every,
                min_opacity=self.cfg.prune_opacity,
            )
        else:
            from gsplat.strategy import DefaultStrategy  # type: ignore
            self.strategy = DefaultStrategy(
                refine_start_iter=self.cfg.refine_start,
                refine_stop_iter=self.cfg.refine_stop,
                refine_every=self.cfg.refine_every,
                reset_every=self.cfg.reset_opacity_every,
                grow_grad2d=self.cfg.grow_grad2d,
                grow_scale3d=self.cfg.grow_scale3d,
                prune_opa=self.cfg.prune_opacity,
                prune_scale3d=self.cfg.prune_scale3d,
                absgrad=self.cfg.absgrad,
            )

        self.strategy_state = self.strategy.initialize_state(
            scene_scale=self.dataset.scene_scale,
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_image(self, idx: int) -> Tuple[Tensor, Optional[Tensor]]:
        """Load a training image.

        Returns:
            rgb: (H, W, 3) float32 in [0, 1]
            alpha: (H, W, 1) float32 in [0, 1], or *None* if opaque.
        """
        import cv2  # type: ignore
        path = str(self.dataset.image_paths[idx])
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")

        if img.ndim == 3 and img.shape[2] == 4 and not self.cfg.ignore_alpha:
            # BGRA → RGBA
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            rgba = torch.tensor(img, dtype=torch.float32, device=self.device) / 255.0
            rgb = rgba[:, :, :3]
            alpha = rgba[:, :, 3:4]
            if self.cfg.invert_alpha:
                alpha = 1.0 - alpha
            return rgb, alpha
        else:
            # Drop alpha channel if present
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb = torch.tensor(img, dtype=torch.float32, device=self.device) / 255.0
            return rgb, None

    def _sample_batch(
        self, step: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], int, int]:
        """Sample a random batch of training views.

        Returns:
            viewmats: (B, 4, 4)
            Ks: (B, 3, 3)
            pixels: (B, H, W, 3) — RGB composited over bg_color
            alphas: (B, H, W, 1) or *None* if all images are opaque
            width: int
            height: int
        """
        n_images = len(self.dataset.image_paths)
        indices = np.random.choice(n_images, self.cfg.batch_size, replace=False)

        imgs, masks, vmats, ks = [], [], [], []
        has_alpha = False
        for idx in indices:
            rgb, alpha = self._load_image(idx)
            if alpha is not None:
                has_alpha = True
                # Composite ground-truth over bg_color so it matches the
                # rasteriser output (which also composites over bg_color).
                bg = torch.tensor(self.cfg.bg_color, dtype=torch.float32,
                                  device=self.device)
                rgb = rgb * alpha + bg * (1.0 - alpha)
                masks.append(alpha)
            else:
                masks.append(torch.ones(
                    rgb.shape[0], rgb.shape[1], 1,
                    dtype=torch.float32, device=self.device,
                ))
            imgs.append(rgb)
            c2w = torch.tensor(
                self.dataset.camtoworlds[idx], dtype=torch.float32, device=self.device
            )
            vmats.append(torch.linalg.inv(c2w))
            ks.append(
                torch.tensor(self.dataset.Ks[idx], dtype=torch.float32, device=self.device)
            )

        pixels = torch.stack(imgs)       # (B, H, W, 3)
        viewmats = torch.stack(vmats)    # (B, 4, 4)
        Ks = torch.stack(ks)             # (B, 3, 3)
        alphas = torch.stack(masks) if has_alpha else None  # (B, H, W, 1) or None
        height, width = pixels.shape[1], pixels.shape[2]
        return viewmats, Ks, pixels, alphas, width, height

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _rasterize(
        self,
        viewmats: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        sh_degree: int,
        backgrounds: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """Run gsplat rasterisation.

        Args:
            backgrounds: (C, 3) per-camera background colour.  When
                *None* the configured ``bg_color`` is used for every
                camera in the batch.
        """
        from gsplat import rasterization  # type: ignore

        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)  # (N, K, 3)

        if backgrounds is None:
            n_cams = viewmats.shape[-3]  # C
            backgrounds = torch.tensor(
                self.cfg.bg_color, dtype=torch.float32, device=self.device,
            ).expand(n_cams, -1)  # (C, 3)

        render_mode = "RGB"
        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"

        renders, alphas, info = rasterization(
            means=self.splats["means"],
            quats=self.splats["quats"],
            scales=torch.exp(self.splats["scales"]),
            opacities=torch.sigmoid(self.splats["opacities"]),
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            sh_degree=sh_degree,
            packed=self.cfg.packed,
            absgrad=self.cfg.absgrad,
            backgrounds=backgrounds,
            render_mode=render_mode,
            rasterize_mode=rasterize_mode,
        )
        return renders, alphas, info

    def _save_debug_images(
        self,
        step: int,
        renders: Tensor,
        pixels: Tensor,
        gt_alphas: Optional[Tensor],
        output_dir: Optional[Path] = None,
    ) -> None:
        """Save GT vs rendered comparison images for debugging."""
        import cv2  # type: ignore

        if output_dir is not None:
            debug_dir = output_dir / "debug"
        else:
            debug_dir = Path("debug_images")
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Take first image in batch
        render_np = (renders[0].detach().cpu().numpy() * 255).astype(np.uint8)
        gt_np = (pixels[0].detach().cpu().numpy() * 255).astype(np.uint8)

        # Convert RGB to BGR for cv2
        render_bgr = cv2.cvtColor(render_np, cv2.COLOR_RGB2BGR)
        gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR)

        # Create side-by-side comparison
        h, w = render_bgr.shape[:2]
        comparison = np.zeros((h, w * 2 + 10, 3), dtype=np.uint8)
        comparison[:, :w] = gt_bgr
        comparison[:, w + 10:] = render_bgr

        # Add labels
        cv2.putText(comparison, "GT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Rendered", (w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Save
        cv2.imwrite(str(debug_dir / f"step_{step:06d}.png"), comparison)

        # Also save alpha mask if present
        if gt_alphas is not None:
            alpha_np = (gt_alphas[0, :, :, 0].detach().cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(debug_dir / f"step_{step:06d}_alpha.png"), alpha_np)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, progress_cb=None, output_dir: Optional[Path] = None) -> TrainResult:
        """Run the full training loop.

        Args:
            progress_cb: Optional callback
                ``fn(step, max_steps, loss, *, it_per_sec, n_gaussians)``
                called every 100 steps for progress reporting.

        Returns:
            :class:`TrainResult` with trained parameters and statistics.
        """
        try:
            from fused_ssim import fused_ssim  # type: ignore
        except ImportError:
            fused_ssim = None

        # Initialise everything
        self._init_splats()
        self._init_optimizers()
        self._init_strategy()

        # Debug: print K matrix and image dimensions
        K0 = self.dataset.Ks[0]
        img_size = self.dataset.image_sizes[0]
        rgb_test, _ = self._load_image(0)
        actual_h, actual_w = rgb_test.shape[:2]
        print(f"  [intrinsics] K matrix for image 0:")
        print(f"    fx={K0[0,0]:.1f}  fy={K0[1,1]:.1f}")
        print(f"    cx={K0[0,2]:.1f}  cy={K0[1,2]:.1f}")
        print(f"  [intrinsics] COLMAP image size: {img_size[0]}x{img_size[1]} (WxH)")
        print(f"  [intrinsics] Actual loaded size: {actual_w}x{actual_h} (WxH)")
        if img_size[0] != actual_w or img_size[1] != actual_h:
            print(f"  [intrinsics] WARNING: Size mismatch! K matrix may be wrong.")

        # Check alpha values on first image (diagnostic)
        rgb, alpha = self._load_image(0)
        if alpha is not None:
            alpha_mean = alpha.mean().item()
            alpha_min = alpha.min().item()
            alpha_max = alpha.max().item()
            opaque_frac = (alpha > 0.5).float().mean().item()
            transparent_frac = (alpha < 0.5).float().mean().item()
            print(f"  [alpha] mean={alpha_mean:.3f} min={alpha_min:.3f} "
                  f"max={alpha_max:.3f}")
            print(f"  [alpha] opaque (>0.5): {opaque_frac*100:.1f}%  "
                  f"transparent (<0.5): {transparent_frac*100:.1f}%")

            # Check RGB values in transparent vs opaque regions
            opaque_mask = alpha > 0.5
            transp_mask = alpha < 0.5
            if opaque_mask.any():
                rgb_opaque_mean = rgb[opaque_mask.expand_as(rgb)].mean().item()
                print(f"  [alpha] RGB mean in opaque regions: {rgb_opaque_mean:.3f}")
            if transp_mask.any():
                rgb_transp_mean = rgb[transp_mask.expand_as(rgb)].mean().item()
                print(f"  [alpha] RGB mean in transparent regions: {rgb_transp_mean:.3f}")

            if alpha_mean < 0.05:
                print("  [alpha] WARNING: alpha is mostly zero - "
                      "images may be inverted! Try --no-alpha")
            elif alpha_mean > 0.95:
                print("  [alpha] NOTE: alpha is mostly one - "
                      "images have minimal transparency")
            elif opaque_frac < 0.3:
                # Small figure in large transparent background - check if alpha might be inverted
                print("  [alpha] NOTE: small opaque region - "
                      "if training fails, try --no-alpha")
        else:
            print("  [alpha] Images are opaque (no alpha channel or --no-alpha)")

        max_steps = self.cfg.max_steps
        t0 = time.time()

        for step in range(max_steps):
            # Current SH degree (ramp up over training)
            sh_degree = min(step // self.cfg.sh_degree_interval, self.cfg.sh_degree)

            # Sample training batch
            viewmats, Ks, pixels, gt_alphas, width, height = self._sample_batch(step)

            # Forward
            renders, alphas, info = self._rasterize(viewmats, Ks, width, height, sh_degree)

            # Pre-backward: let strategy retain grads on means2d
            self.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # Loss — mask by ground-truth alpha so transparent regions
            # don't contribute to the reconstruction objective.
            if gt_alphas is not None:
                # Expand alpha to RGB channels for proper normalization
                # Without this, the loss would be 3× too large (sum over 3 channels
                # but normalized by sum over 1 channel)
                alpha_rgb = gt_alphas.expand_as(renders)  # (B, H, W, 3)
                l1 = (torch.abs(renders - pixels) * alpha_rgb).sum() / alpha_rgb.sum().clamp(min=1)
            else:
                l1 = F.l1_loss(renders, pixels)

            if fused_ssim is not None:
                ssim_val = fused_ssim(
                    renders.permute(0, 3, 1, 2),
                    pixels.permute(0, 3, 1, 2),
                    padding="valid",
                )
                loss = l1 * (1.0 - self.cfg.ssim_lambda) + (1.0 - ssim_val) * self.cfg.ssim_lambda
            else:
                loss = l1

            # Backward
            loss.backward()

            # Post-backward: densification (grow / prune)
            self.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                packed=self.cfg.packed,
            )

            # Optimiser step
            for opt in self.optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)

            last_loss = loss.item()

            # Debug: save GT vs rendered comparison images
            if self.cfg.debug_save_images and step % self.cfg.debug_save_every == 0:
                self._save_debug_images(step, renders, pixels, gt_alphas, output_dir)

            # Progress
            if progress_cb and step % 100 == 0:
                elapsed = time.time() - t0
                it_s = (step + 1) / elapsed if elapsed > 0 else 0.0
                n_gs = len(self.splats["means"])
                progress_cb(step, max_steps, last_loss,
                            it_per_sec=it_s, n_gaussians=n_gs)

        elapsed = time.time() - t0

        # Gather final statistics
        grad2d = self.strategy_state.get("grad2d")
        count = self.strategy_state.get("count")

        n = len(self.splats["means"])
        if grad2d is None:
            grad2d = torch.zeros(n, device=self.device)
        if count is None:
            count = torch.zeros(n, device=self.device)

        return TrainResult(
            means=self.splats["means"].detach().cpu(),
            scales=self.splats["scales"].detach().cpu(),
            quats=self.splats["quats"].detach().cpu(),
            opacities=self.splats["opacities"].detach().cpu(),
            sh0=self.splats["sh0"].detach().cpu(),
            shN=self.splats["shN"].detach().cpu(),
            sh_degree=self.cfg.sh_degree,
            grad2d=grad2d.detach().cpu(),
            view_count=count.detach().cpu(),
            final_loss=last_loss,
        )

    # ------------------------------------------------------------------
    # Post-training rendering
    # ------------------------------------------------------------------

    def render(
        self,
        c2w: np.ndarray,
        K: np.ndarray,
        width: int,
        height: int,
        sh_degree: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        """Render a single view from the trained splats.

        This is useful for post-training inspection: the returned ``info``
        dict contains ``means2d`` (with ``.grad`` / ``.absgrad`` after a
        backward pass), ``radii``, ``depths``, etc.

        Args:
            c2w: (4, 4) camera-to-world matrix (OpenCV convention).
            K: (3, 3) intrinsic matrix.
            width: Image width.
            height: Image height.
            sh_degree: SH degree to use (defaults to trained degree).

        Returns:
            (render_image, render_alpha, info) – same as
            ``gsplat.rasterization``.
        """
        if sh_degree is None:
            sh_degree = self.cfg.sh_degree

        c2w_t = torch.tensor(c2w, dtype=torch.float32, device=self.device)
        viewmat = torch.linalg.inv(c2w_t).unsqueeze(0)  # (1, 4, 4)
        K_t = torch.tensor(K, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            renders, alphas, info = self._rasterize(viewmat, K_t, width, height, sh_degree)

        return renders.squeeze(0), alphas.squeeze(0), info

    def render_stat_map(
        self,
        stat: Tensor,
        c2w: np.ndarray,
        K: np.ndarray,
        width: int,
        height: int,
        log_scale: bool = False,
    ) -> np.ndarray:
        """Render a per-Gaussian scalar statistic as a grayscale image.

        Splatts each Gaussian with a grayscale intensity proportional to
        its statistic value instead of its SH colour.  Useful for
        visualising gradient norms, view counts, or any other per-Gaussian
        scalar.

        Args:
            stat: (N,) per-Gaussian scalar values (e.g.
                ``result.avg_grad_norm`` or ``result.view_count``).
            c2w: (4, 4) camera-to-world matrix.
            K: (3, 3) intrinsic matrix.
            width: Image width in pixels.
            height: Image height in pixels.
            log_scale: If *True*, apply ``log(1 + x)`` before normalising.
                Useful when the value range spans several orders of
                magnitude (common for raw gradient norms).

        Returns:
            (H, W) ``np.float32`` array in [0, 1] – a grayscale image
            where brighter pixels correspond to higher statistic values.
        """
        from gsplat import rasterization  # type: ignore

        s = stat.to(self.device).float()
        if log_scale:
            s = torch.log1p(s)

        s_min, s_max = s.min(), s.max()
        if s_max > s_min:
            s_norm = (s - s_min) / (s_max - s_min)
        else:
            s_norm = torch.zeros_like(s)

        # Expand scalar to (N, 3) so rasterisation produces an RGB image
        colors = s_norm.unsqueeze(-1).expand(-1, 3)  # (N, 3)

        c2w_t = torch.tensor(c2w, dtype=torch.float32, device=self.device)
        viewmat = torch.linalg.inv(c2w_t).unsqueeze(0)
        K_t = torch.tensor(K, dtype=torch.float32, device=self.device).unsqueeze(0)

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"

        with torch.no_grad():
            renders, _alphas, _info = rasterization(
                means=self.splats["means"],
                quats=self.splats["quats"],
                scales=torch.exp(self.splats["scales"]),
                opacities=torch.sigmoid(self.splats["opacities"]),
                colors=colors,
                viewmats=viewmat,
                Ks=K_t,
                width=width,
                height=height,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                sh_degree=None,  # pre-computed colours, not SH
                packed=self.cfg.packed,
                render_mode="RGB",
                rasterize_mode=rasterize_mode,
            )

        # Take first channel (all three are identical)
        gray = renders.squeeze(0)[:, :, 0].cpu().numpy()
        return gray
