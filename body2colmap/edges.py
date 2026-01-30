"""
Edge detection utilities for texture projection and overlay rendering.

This module provides edge extraction functions that can be applied to rendered
images for Canny edge guidance generation.

Supported methods:
- Canny: Classic Canny edge detection with configurable thresholds
- Sobel: Gradient-based edge detection
"""

import numpy as np
from typing import Tuple, Optional, Callable
from numpy.typing import NDArray


def _ensure_grayscale(image: NDArray) -> NDArray[np.uint8]:
    """
    Convert image to grayscale if needed.

    Uses cv2.cvtColor for standard cases, with fallback for edge cases.

    Args:
        image: Input image, shape (H, W), (H, W, 3), or (H, W, 4)

    Returns:
        Grayscale image, shape (H, W), dtype uint8
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV (cv2) is required for grayscale conversion")

    if image.ndim == 2:
        # Already grayscale
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        return image

    if image.ndim == 3:
        # Handle float images first
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Use cv2.cvtColor for conversion
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        elif image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    raise ValueError(f"Unsupported image shape: {image.shape}")


def canny(
    image: NDArray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    blur_kernel: int = 5
) -> NDArray[np.uint8]:
    """
    Apply Canny edge detection.

    Args:
        image: Input image, any format (will be converted to grayscale)
        low_threshold: Lower threshold for hysteresis (0-255)
        high_threshold: Upper threshold for hysteresis (0-255)
        blur_kernel: Gaussian blur kernel size (must be odd). Set to 0 to disable blur.

    Returns:
        Edge image, shape (H, W), dtype uint8, values 0 or 255

    Note:
        Uses OpenCV's Canny implementation. Applies Gaussian blur before
        edge detection to reduce noise.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV (cv2) is required for edge detection. "
            "Install with: pip install opencv-python"
        )

    # Convert to grayscale
    gray = _ensure_grayscale(image)

    # Apply Gaussian blur to reduce noise
    if blur_kernel > 0:
        if blur_kernel % 2 == 0:
            blur_kernel += 1  # Must be odd
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    return edges


def sobel(
    image: NDArray,
    kernel_size: int = 3,
    threshold: Optional[int] = None
) -> NDArray[np.uint8]:
    """
    Apply Sobel edge detection.

    Args:
        image: Input image, any format (will be converted to grayscale)
        kernel_size: Sobel kernel size (1, 3, 5, or 7)
        threshold: Optional threshold to binarize output (0-255).
                  If None, returns gradient magnitude normalized to 0-255.

    Returns:
        Edge image, shape (H, W), dtype uint8

    Note:
        Computes gradient magnitude from X and Y Sobel filters.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV (cv2) is required for edge detection. "
            "Install with: pip install opencv-python"
        )

    # Convert to grayscale
    gray = _ensure_grayscale(image)

    # Compute Sobel gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # Compute gradient magnitude
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalize to 0-255
    if magnitude.max() > 0:
        magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
    else:
        magnitude = magnitude.astype(np.uint8)

    # Apply threshold if specified
    if threshold is not None:
        magnitude = ((magnitude >= threshold) * 255).astype(np.uint8)

    return magnitude


def laplacian(
    image: NDArray,
    kernel_size: int = 3,
    threshold: Optional[int] = None
) -> NDArray[np.uint8]:
    """
    Apply Laplacian edge detection.

    Args:
        image: Input image, any format (will be converted to grayscale)
        kernel_size: Laplacian kernel size (must be positive and odd)
        threshold: Optional threshold to binarize output (0-255).
                  If None, returns absolute Laplacian normalized to 0-255.

    Returns:
        Edge image, shape (H, W), dtype uint8
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV (cv2) is required for edge detection. "
            "Install with: pip install opencv-python"
        )

    # Convert to grayscale
    gray = _ensure_grayscale(image)

    # Apply Laplacian
    laplacian_result = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)

    # Take absolute value (edges can be positive or negative)
    laplacian_abs = np.abs(laplacian_result)

    # Normalize to 0-255
    if laplacian_abs.max() > 0:
        laplacian_norm = (laplacian_abs / laplacian_abs.max() * 255).astype(np.uint8)
    else:
        laplacian_norm = laplacian_abs.astype(np.uint8)

    # Apply threshold if specified
    if threshold is not None:
        laplacian_norm = ((laplacian_norm >= threshold) * 255).astype(np.uint8)

    return laplacian_norm


# Registry of available edge detection methods
EDGE_METHODS = {
    "canny": canny,
    "sobel": sobel,
    "laplacian": laplacian,
}


def get_edge_detector(method: str) -> Callable:
    """
    Get edge detection function by name.

    Args:
        method: Method name ("canny", "sobel", "laplacian")

    Returns:
        Edge detection function

    Raises:
        ValueError: If method is not recognized
    """
    if method not in EDGE_METHODS:
        available = ", ".join(EDGE_METHODS.keys())
        raise ValueError(f"Unknown edge method '{method}'. Available: {available}")

    return EDGE_METHODS[method]


def detect_edges(
    image: NDArray,
    method: str = "canny",
    **kwargs
) -> NDArray[np.uint8]:
    """
    Detect edges using specified method.

    This is the main entry point for edge detection.

    Args:
        image: Input image, any format
        method: Detection method ("canny", "sobel", "laplacian")
        **kwargs: Method-specific parameters

    Returns:
        Edge image, shape (H, W), dtype uint8

    Example:
        edges = detect_edges(rgb_image, method="canny", low_threshold=50, high_threshold=150)
    """
    detector = get_edge_detector(method)
    return detector(image, **kwargs)


def edges_to_rgba(
    edges: NDArray[np.uint8],
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    background_alpha: int = 0
) -> NDArray[np.uint8]:
    """
    Convert edge image to RGBA with specified color.

    Args:
        edges: Edge image, shape (H, W), values 0-255
        color: RGB color (0-1 range) for edge pixels
        background_alpha: Alpha value for non-edge pixels (0 = transparent)

    Returns:
        RGBA image, shape (H, W, 4), dtype uint8
        - Edge pixels: specified color with alpha=255
        - Non-edge pixels: black with alpha=background_alpha
    """
    h, w = edges.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Create mask where edges exist
    edge_mask = edges > 0

    # Set edge color
    rgba[edge_mask, 0] = int(color[0] * 255)
    rgba[edge_mask, 1] = int(color[1] * 255)
    rgba[edge_mask, 2] = int(color[2] * 255)
    rgba[edge_mask, 3] = 255

    # Set background alpha
    rgba[~edge_mask, 3] = background_alpha

    return rgba
