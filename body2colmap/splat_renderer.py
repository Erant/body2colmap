"""
Renderer for Gaussian Splats using gsplat.

This module provides SplatRenderer which wraps gsplat's rasterization
functions. It takes Camera objects (same as mesh Renderer) and produces
RGBA images.

Coordinate System:
    gsplat uses OpenGL convention (Y-up, camera looks down -Z).
    This matches our world coordinates exactly.
    We use Camera.get_w2c() directly - NO coordinate conversion needed.

    This maintains the principle: coordinate conversion only at I/O boundaries.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from numpy.typing import NDArray

from .splat_scene import SplatScene
from .camera import Camera


class SplatRenderer:
    """
    Render Gaussian splats using gsplat.

    Unlike the mesh Renderer which has multiple modes (mesh, depth, skeleton),
    SplatRenderer renders the splat directly with view-dependent colors from
    spherical harmonics.
    """

    def __init__(
        self,
        scene: SplatScene,
        render_size: Tuple[int, int],
        device: str = "cuda"
    ):
        """
        Initialize renderer.

        Args:
            scene: SplatScene to render
            render_size: (width, height) in pixels
            device: torch device ("cuda" recommended, "cpu" much slower)
        """
        self.scene = scene
        self.width, self.height = render_size
        self.device = device

        # Cached torch tensors (lazily initialized)
        self._tensors: Optional[Dict[str, Any]] = None
        self._torch = None  # torch module reference

    def _ensure_tensors(self):
        """Lazily convert numpy arrays to torch tensors on device."""
        if self._tensors is not None:
            return

        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is required for splat rendering. "
                "Install with: pip install torch"
            )

        self._torch = torch

        # Convert scene data to torch tensors
        self._tensors = {
            'means': torch.from_numpy(self.scene.means).to(self.device),
            'scales': torch.from_numpy(self.scene.scales).to(self.device),
            'quats': torch.from_numpy(self.scene.quats).to(self.device),
            'opacities': torch.from_numpy(self.scene.opacities).to(self.device),
            'sh_coeffs': torch.from_numpy(self.scene.sh_coeffs).to(self.device),
        }

    def render(
        self,
        camera: Camera,
        bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> NDArray[np.uint8]:
        """
        Render the splat from given camera viewpoint.

        Args:
            camera: Camera object (same interface as mesh Renderer)
            bg_color: Background RGB color (0-1 range)

        Returns:
            RGBA image (height, width, 4), dtype uint8
            Alpha comes from accumulated opacity during rasterization.

        Note:
            Uses Camera.get_w2c() directly since gsplat uses OpenGL convention
            which matches our world coordinate system. No conversion needed.
        """
        self._ensure_tensors()
        torch = self._torch

        try:
            from gsplat import rasterization
        except ImportError:
            raise ImportError(
                "gsplat is required for splat rendering. "
                "Install with: pip install gsplat"
            )

        # Get camera matrices
        # CRITICAL: use get_w2c() NOT get_colmap_extrinsics()
        # gsplat uses OpenGL convention, same as our world coords
        viewmat = torch.from_numpy(
            camera.get_w2c().astype(np.float32)
        ).to(self.device).unsqueeze(0)  # (1, 4, 4)

        K = torch.tensor([
            [camera.fx, 0.0, camera.cx],
            [0.0, camera.fy, camera.cy],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, 3, 3)

        # Prepare Gaussian parameters
        means = self._tensors['means']  # (N, 3)
        quats = self._tensors['quats']  # (N, 4) wxyz

        # Scales are stored in log-space, gsplat wants actual scales
        scales = torch.exp(self._tensors['scales'])  # (N, 3)

        # Opacities are stored as logits, gsplat wants 0-1
        opacities = torch.sigmoid(self._tensors['opacities'])  # (N,)

        # Compute view-dependent colors from spherical harmonics
        sh_coeffs = self._tensors['sh_coeffs']  # (N, K, 3)
        colors = self._eval_sh(
            sh_coeffs,
            means,
            viewmat,
            self.scene.sh_degree
        )  # (N, 3)

        # Background color tensor
        backgrounds = torch.tensor(
            [bg_color],
            dtype=torch.float32,
            device=self.device
        )  # (1, 3)

        # Call gsplat rasterization
        # Returns: (colors, alphas, meta) or similar depending on version
        render_colors, render_alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=self.width,
            height=self.height,
            backgrounds=backgrounds,
            sh_degree=None,  # We pre-compute colors, don't pass SH to gsplat
        )

        # Convert to numpy RGBA
        # render_colors: (1, H, W, 3), render_alphas: (1, H, W, 1)
        rgb = render_colors[0].detach().cpu().numpy()  # (H, W, 3)
        alpha = render_alphas[0, ..., 0].detach().cpu().numpy()  # (H, W)

        # Clamp and convert to uint8
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        alpha = np.clip(alpha * 255, 0, 255).astype(np.uint8)

        # Combine into RGBA
        rgba = np.dstack([rgb, alpha])

        return rgba

    def _eval_sh(
        self,
        sh_coeffs,  # (N, K, 3)
        means,      # (N, 3)
        viewmat,    # (1, 4, 4)
        sh_degree: int
    ):
        """
        Evaluate spherical harmonics to get view-dependent colors.

        Args:
            sh_coeffs: SH coefficients (N, K, 3) where K = (degree+1)^2
            means: Gaussian centers (N, 3)
            viewmat: World-to-camera matrix (1, 4, 4)
            sh_degree: Maximum SH degree

        Returns:
            colors: RGB colors (N, 3) in range [0, 1]
        """
        torch = self._torch

        # Extract camera position from viewmat
        # viewmat is w2c, so camera position in world = -R^T @ t
        R = viewmat[0, :3, :3]  # (3, 3)
        t = viewmat[0, :3, 3]   # (3,)
        cam_pos = -R.T @ t      # (3,)

        # Compute view directions (from Gaussian to camera)
        view_dirs = cam_pos.unsqueeze(0) - means  # (N, 3)
        view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-8)

        # Evaluate SH
        # SH basis functions for degree 0, 1, 2, 3
        C0 = 0.28209479177387814
        C1 = 0.4886025119029199
        C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ]
        C3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        ]

        x, y, z = view_dirs[:, 0], view_dirs[:, 1], view_dirs[:, 2]
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z

        # Start with DC term
        result = C0 * sh_coeffs[:, 0, :]  # (N, 3)

        if sh_degree >= 1 and sh_coeffs.shape[1] >= 4:
            result = result + \
                C1 * y.unsqueeze(-1) * sh_coeffs[:, 1, :] + \
                C1 * z.unsqueeze(-1) * sh_coeffs[:, 2, :] + \
                C1 * x.unsqueeze(-1) * sh_coeffs[:, 3, :]

        if sh_degree >= 2 and sh_coeffs.shape[1] >= 9:
            result = result + \
                C2[0] * xy.unsqueeze(-1) * sh_coeffs[:, 4, :] + \
                C2[1] * yz.unsqueeze(-1) * sh_coeffs[:, 5, :] + \
                C2[2] * (2.0 * zz - xx - yy).unsqueeze(-1) * sh_coeffs[:, 6, :] + \
                C2[3] * xz.unsqueeze(-1) * sh_coeffs[:, 7, :] + \
                C2[4] * (xx - yy).unsqueeze(-1) * sh_coeffs[:, 8, :]

        if sh_degree >= 3 and sh_coeffs.shape[1] >= 16:
            # IMPORTANT: Compute scalar terms first, then unsqueeze to avoid
            # broadcasting (N,) * (N, 1) -> (N, N) which explodes memory
            result = result + \
                C3[0] * (y * (3.0 * xx - yy)).unsqueeze(-1) * sh_coeffs[:, 9, :] + \
                C3[1] * (xy * z).unsqueeze(-1) * sh_coeffs[:, 10, :] + \
                C3[2] * (y * (4.0 * zz - xx - yy)).unsqueeze(-1) * sh_coeffs[:, 11, :] + \
                C3[3] * (z * (2.0 * zz - 3.0 * xx - 3.0 * yy)).unsqueeze(-1) * sh_coeffs[:, 12, :] + \
                C3[4] * (x * (4.0 * zz - xx - yy)).unsqueeze(-1) * sh_coeffs[:, 13, :] + \
                C3[5] * (z * (xx - yy)).unsqueeze(-1) * sh_coeffs[:, 14, :] + \
                C3[6] * (x * (xx - 3.0 * yy)).unsqueeze(-1) * sh_coeffs[:, 15, :]

        # Add 0.5 bias and clamp to valid color range
        result = result + 0.5
        result = torch.clamp(result, 0.0, 1.0)

        return result

    def __del__(self):
        """Clean up tensors."""
        self._tensors = None
