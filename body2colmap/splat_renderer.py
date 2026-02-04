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
        # gsplat uses OpenCV convention (camera looks down +Z, Y down)
        # Our world coords are OpenGL (camera looks down -Z, Y up)
        # Apply 180° X rotation to convert: multiply w2c by opengl_to_opencv
        w2c = camera.get_w2c()

        # OpenGL to OpenCV conversion (180° rotation around X)
        opengl_to_opencv = np.array([
            [1.0,  0.0,  0.0, 0.0],
            [0.0, -1.0,  0.0, 0.0],
            [0.0,  0.0, -1.0, 0.0],
            [0.0,  0.0,  0.0, 1.0],
        ], dtype=np.float32)

        # Apply conversion: new_w2c = opengl_to_opencv @ w2c
        viewmat_cv = opengl_to_opencv @ w2c
        viewmat = torch.from_numpy(viewmat_cv).to(self.device).unsqueeze(0)  # (1, 4, 4)

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

    def render_depth(
        self,
        camera: Camera,
        normalize: bool = True,
        colormap: Optional[str] = None,
        expected_depth: bool = True
    ) -> NDArray[np.uint8]:
        """
        Render depth map from Gaussian splats.

        Uses gsplat's native depth rendering mode. Unlike mesh depth (single surface),
        splat depth is volume-based, considering all Gaussians along each ray weighted
        by their opacity.

        Args:
            camera: Camera object (same interface as mesh Renderer)
            normalize: If True, normalize depth to 0-1 range for visualization
            colormap: Optional colormap name ("viridis", "plasma", etc.)
                     If None, returns grayscale depth
            expected_depth: If True, use expected depth (ED): sum(w*z)/sum(w)
                           If False, use accumulated depth (D): sum(w*z)
                           Expected depth is usually more intuitive.

        Returns:
            RGBA image (height, width, 4), dtype uint8
            Alpha comes from accumulated opacity during rasterization.
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

        # Get camera matrices (same setup as render())
        w2c = camera.get_w2c()

        # OpenGL to OpenCV conversion (180° rotation around X)
        opengl_to_opencv = np.array([
            [1.0,  0.0,  0.0, 0.0],
            [0.0, -1.0,  0.0, 0.0],
            [0.0,  0.0, -1.0, 0.0],
            [0.0,  0.0,  0.0, 1.0],
        ], dtype=np.float32)

        viewmat_cv = opengl_to_opencv @ w2c
        viewmat = torch.from_numpy(viewmat_cv).to(self.device).unsqueeze(0)

        K = torch.tensor([
            [camera.fx, 0.0, camera.cx],
            [0.0, camera.fy, camera.cy],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

        # Prepare Gaussian parameters
        means = self._tensors['means']
        quats = self._tensors['quats']
        scales = torch.exp(self._tensors['scales'])
        opacities = torch.sigmoid(self._tensors['opacities'])
        sh_coeffs = self._tensors['sh_coeffs']

        # Select depth mode
        render_mode = "ED" if expected_depth else "D"

        # Render depth using gsplat
        render_depths, render_alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=sh_coeffs,
            viewmats=viewmat,
            Ks=K,
            width=self.width,
            height=self.height,
            sh_degree=self.scene.sh_degree,
            render_mode=render_mode,
        )

        # Convert to numpy
        # render_depths: (1, H, W, 1), render_alphas: (1, H, W, 1)
        depth = render_depths[0, ..., 0].detach().cpu().numpy()  # (H, W)
        alpha = render_alphas[0, ..., 0].detach().cpu().numpy()  # (H, W)

        # Create alpha mask (convert 0-1 opacity to 0-255)
        alpha_mask = (alpha * 255).astype(np.uint8)

        # Normalize depth if requested
        if normalize:
            # Use alpha > threshold to identify valid pixels
            valid_mask = alpha > 0.01
            valid_depth = depth[valid_mask]
            if len(valid_depth) > 0:
                min_depth = valid_depth.min()
                max_depth = valid_depth.max()
                depth_range = max_depth - min_depth
                if depth_range > 1e-6:
                    depth_normalized = np.zeros_like(depth)
                    # Invert so closer = white (1.0), farther = black (0.0)
                    depth_normalized[valid_mask] = 1.0 - (depth[valid_mask] - min_depth) / depth_range
                else:
                    depth_normalized = np.ones_like(depth) * 0.5
            else:
                depth_normalized = np.zeros_like(depth)
        else:
            depth_normalized = depth

        # Apply colormap if requested
        if colormap is not None:
            try:
                import matplotlib.cm as cm
            except ImportError:
                raise ImportError("matplotlib is required for colormaps")

            cmap = cm.get_cmap(colormap)
            depth_colored = cmap(depth_normalized)[:, :, :3]  # RGB
            depth_colored = (depth_colored * 255).astype(np.uint8)
        else:
            # Grayscale
            depth_gray = (depth_normalized * 255).astype(np.uint8)
            depth_colored = np.stack([depth_gray] * 3, axis=-1)

        # Combine with alpha
        rgba = np.dstack([depth_colored, alpha_mask])

        return rgba

    def __del__(self):
        """Clean up tensors."""
        self._tensors = None
