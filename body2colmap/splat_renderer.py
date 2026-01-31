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

        # SH coefficients - gsplat handles evaluation internally
        # Shape: (N, K, 3) where K = (sh_degree + 1)^2
        sh_coeffs = self._tensors['sh_coeffs']  # (N, K, 3)

        # Call gsplat rasterization
        # gsplat evaluates SH internally when sh_degree is provided
        # Don't pass backgrounds - we'll composite with alpha afterwards
        render_colors, render_alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=sh_coeffs,  # Pass SH coeffs, gsplat evaluates them
            viewmats=viewmat,
            Ks=K,
            width=self.width,
            height=self.height,
            sh_degree=self.scene.sh_degree,  # Let gsplat handle SH evaluation
        )

        # Convert to numpy
        # render_colors: (1, H, W, 3), render_alphas: (1, H, W, 1)
        rgb = render_colors[0].detach().cpu().numpy()  # (H, W, 3)
        alpha = render_alphas[0, ..., 0].detach().cpu().numpy()  # (H, W)

        # Composite with background color using alpha
        bg = np.array(bg_color, dtype=np.float32)
        rgb = rgb * alpha[..., np.newaxis] + bg * (1.0 - alpha[..., np.newaxis])

        # Clamp and convert to uint8
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        alpha = np.clip(alpha * 255, 0, 255).astype(np.uint8)

        # Combine into RGBA
        rgba = np.dstack([rgb, alpha])

        return rgba

    def __del__(self):
        """Clean up tensors."""
        self._tensors = None
