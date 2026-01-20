"""Utilities for ComfyUI integration: image format conversion and rendering setup."""

import os
import numpy as np
import torch
from typing import List
from numpy.typing import NDArray


def setup_headless_rendering():
    """Configure OpenGL for headless rendering.

    ComfyUI may run without a display. Try EGL first (GPU-accelerated),
    fall back to OSMesa (software rendering) if EGL is unavailable.
    """
    if "PYOPENGL_PLATFORM" in os.environ:
        # Already configured
        return

    try:
        # Try EGL first (NVIDIA GPUs, faster)
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        import pyrender
        # Quick test to see if EGL works
        r = pyrender.OffscreenRenderer(64, 64)
        r.delete()
        print("[Body2COLMAP] Using EGL for rendering")
    except Exception as e:
        # Fall back to OSMesa (software rendering, slower but more compatible)
        try:
            os.environ["PYOPENGL_PLATFORM"] = "osmesa"
            import pyrender
            r = pyrender.OffscreenRenderer(64, 64)
            r.delete()
            print("[Body2COLMAP] Using OSMesa for rendering")
        except Exception as e2:
            # If both fail, just continue - renderer will be created on first use
            print(f"[Body2COLMAP] Warning: Could not initialize renderer ({e2})")


def rendered_to_comfy(images: List[NDArray]) -> torch.Tensor:
    """
    Convert list of rendered RGBA images to ComfyUI IMAGE format.

    Args:
        images: List of [H, W, 4] uint8 arrays in [0, 255] (RGBA)

    Returns:
        Tensor of shape [B, H, W, 3] in [0, 1] float32 (RGB)

    Note:
        Alpha channel is currently dropped because ComfyUI's IMAGE type expects RGB.
        The renderer produces RGBA where alpha represents mesh coverage/opacity.
        TODO: Consider adding separate MASK output type or compositing against bg_color.
    """
    if not images:
        raise ValueError("Empty image list")

    # Stack into batch
    batch = np.stack(images, axis=0)  # [B, H, W, 4]

    # Drop alpha channel (ComfyUI IMAGE is RGB only)
    # TODO: Should we composite against background color using alpha instead?
    batch = batch[..., :3]  # [B, H, W, 3]

    # Convert to float [0, 1]
    batch = batch.astype(np.float32) / 255.0

    # Convert to torch tensor
    return torch.from_numpy(batch)


def comfy_to_cv2(images: torch.Tensor) -> List[NDArray]:
    """
    Convert ComfyUI IMAGE to list of OpenCV BGR images for saving.

    Args:
        images: Tensor of shape [B, H, W, 3] in [0, 1] (RGB)

    Returns:
        List of [H, W, 3] uint8 BGR arrays
    """
    # To numpy
    batch = images.cpu().numpy()  # [B, H, W, 3]

    # Scale to [0, 255]
    batch = (batch * 255).astype(np.uint8)

    # RGB to BGR for OpenCV
    batch = batch[..., ::-1]

    return [batch[i] for i in range(batch.shape[0])]


def comfy_to_rgb(images: torch.Tensor) -> List[NDArray]:
    """
    Convert ComfyUI IMAGE to list of RGB numpy arrays.

    Args:
        images: Tensor of shape [B, H, W, 3] in [0, 1] (RGB)

    Returns:
        List of [H, W, 3] uint8 RGB arrays
    """
    # To numpy
    batch = images.cpu().numpy()  # [B, H, W, 3]

    # Scale to [0, 255]
    batch = (batch * 255).astype(np.uint8)

    return [batch[i] for i in range(batch.shape[0])]
