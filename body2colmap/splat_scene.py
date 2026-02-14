"""
Gaussian Splat scene loaded from PLY file.

This module provides SplatScene, which is analogous to Scene but holds
Gaussian splat data instead of mesh data. It implements the same interface
methods (get_bounds, get_bbox_center, get_point_cloud) so pipelines can
use either interchangeably.

No coordinate conversion happens here - PLY files from our pipeline are
already in world coordinates (Y-up, Z-out).
"""

import numpy as np
from typing import Tuple, Optional
from numpy.typing import NDArray


class SplatScene:
    """
    Gaussian Splat scene from PLY file.

    Attributes:
        means: (N, 3) Gaussian centers in world coords
        scales: (N, 3) scale parameters (log-space from training)
        quats: (N, 4) rotation quaternions (wxyz convention)
        opacities: (N,) opacity values (logit-space from training)
        sh_coeffs: (N, K, 3) spherical harmonics coefficients
        sh_degree: int, SH degree (typically 3, giving 16 coeffs)

    The interface mirrors Scene:
        - get_bounds() -> (min_corner, max_corner)
        - get_bbox_center() -> center point
        - get_point_cloud() -> (points, colors) for COLMAP export
    """

    def __init__(
        self,
        means: NDArray[np.float32],
        scales: NDArray[np.float32],
        quats: NDArray[np.float32],
        opacities: NDArray[np.float32],
        sh_coeffs: NDArray[np.float32],
        sh_degree: int = 3
    ):
        """
        Initialize SplatScene.

        Args:
            means: (N, 3) Gaussian centers in world coordinates
            scales: (N, 3) scale parameters
            quats: (N, 4) rotation quaternions (wxyz)
            opacities: (N,) opacity values
            sh_coeffs: (N, K, 3) spherical harmonics coefficients
            sh_degree: SH degree (default 3)
        """
        self.means = np.asarray(means, dtype=np.float32)
        self.scales = np.asarray(scales, dtype=np.float32)
        self.quats = np.asarray(quats, dtype=np.float32)
        self.opacities = np.asarray(opacities, dtype=np.float32)
        self.sh_coeffs = np.asarray(sh_coeffs, dtype=np.float32)
        self.sh_degree = sh_degree

        # Cached computations
        self._bounds: Optional[Tuple[NDArray[np.float32], NDArray[np.float32]]] = None

    @classmethod
    def from_ply(cls, filepath: str) -> "SplatScene":
        """
        Load from PLY file.

        Supports standard 3DGS PLY format with properties:
        - x, y, z: Gaussian centers
        - scale_0, scale_1, scale_2: log-scale parameters
        - rot_0, rot_1, rot_2, rot_3: quaternion (wxyz)
        - opacity: opacity logit
        - f_dc_0, f_dc_1, f_dc_2: DC spherical harmonics
        - f_rest_*: higher-order SH coefficients

        Args:
            filepath: Path to .ply file

        Returns:
            SplatScene instance

        Note:
            No coordinate conversion - PLY is assumed to be in world coords.
            This is true for splats trained on our COLMAP output.
        """
        try:
            from plyfile import PlyData
        except ImportError:
            raise ImportError(
                "plyfile is required for loading PLY files. "
                "Install with: pip install plyfile"
            )

        plydata = PlyData.read(filepath)
        vertex = plydata['vertex']

        # Extract positions
        means = np.stack([
            vertex['x'],
            vertex['y'],
            vertex['z']
        ], axis=-1).astype(np.float32)

        n_gaussians = len(means)

        # Extract scales (log-space)
        scales = np.stack([
            vertex['scale_0'],
            vertex['scale_1'],
            vertex['scale_2']
        ], axis=-1).astype(np.float32)

        # Extract rotations (quaternion wxyz)
        # Note: 3DGS stores as rot_0=w, rot_1=x, rot_2=y, rot_3=z
        quats = np.stack([
            vertex['rot_0'],
            vertex['rot_1'],
            vertex['rot_2'],
            vertex['rot_3']
        ], axis=-1).astype(np.float32)

        # Normalize quaternions
        quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)

        # Extract opacities (logit-space)
        opacities = np.asarray(vertex['opacity'], dtype=np.float32)

        # Extract spherical harmonics
        # DC component (degree 0)
        sh_dc = np.stack([
            vertex['f_dc_0'],
            vertex['f_dc_1'],
            vertex['f_dc_2']
        ], axis=-1).astype(np.float32)  # (N, 3)

        # Count higher-order SH coefficients
        # For degree 3: 16 total coeffs per channel, 1 DC + 15 rest
        sh_rest_names = [name for name in vertex.data.dtype.names if name.startswith('f_rest_')]
        n_sh_rest = len(sh_rest_names) // 3  # 3 channels (RGB)

        if n_sh_rest > 0:
            # Extract rest coefficients
            sh_rest = np.zeros((n_gaussians, n_sh_rest, 3), dtype=np.float32)
            for i in range(n_sh_rest):
                sh_rest[:, i, 0] = vertex[f'f_rest_{i}']
                sh_rest[:, i, 1] = vertex[f'f_rest_{i + n_sh_rest}']
                sh_rest[:, i, 2] = vertex[f'f_rest_{i + 2 * n_sh_rest}']

            # Combine DC and rest: (N, 1+n_sh_rest, 3)
            sh_coeffs = np.concatenate([
                sh_dc[:, np.newaxis, :],  # (N, 1, 3)
                sh_rest                    # (N, n_sh_rest, 3)
            ], axis=1)
        else:
            # Only DC component
            sh_coeffs = sh_dc[:, np.newaxis, :]  # (N, 1, 3)

        # Infer SH degree from coefficient count
        n_coeffs = sh_coeffs.shape[1]
        if n_coeffs == 1:
            sh_degree = 0
        elif n_coeffs == 4:
            sh_degree = 1
        elif n_coeffs == 9:
            sh_degree = 2
        elif n_coeffs == 16:
            sh_degree = 3
        else:
            # Default to 3, will use what we have
            sh_degree = 3

        return cls(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            sh_coeffs=sh_coeffs,
            sh_degree=sh_degree
        )

    def get_bounds(self) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Get axis-aligned bounding box from Gaussian means.

        Returns:
            min_corner: Minimum (x, y, z) coordinates
            max_corner: Maximum (x, y, z) coordinates

        Note:
            This uses only Gaussian centers, ignoring their scales/extents.
            For most scenes this is sufficient for camera framing.
        """
        if self._bounds is None:
            min_corner = np.min(self.means, axis=0)
            max_corner = np.max(self.means, axis=0)
            self._bounds = (min_corner, max_corner)

        return self._bounds

    def get_bbox_center(self) -> NDArray[np.float32]:
        """
        Get center of bounding box.

        Used for camera look-at target, same as Scene.get_bbox_center().

        Returns:
            Bounding box center in world coordinates
        """
        min_corner, max_corner = self.get_bounds()
        return (min_corner + max_corner) / 2.0

    def get_centroid(self) -> NDArray[np.float32]:
        """
        Get centroid (mean of all Gaussian centers).

        Returns:
            Centroid position in world coordinates
        """
        return np.mean(self.means, axis=0)

    def get_point_cloud(
        self,
        n_samples: int = 50000,
        color: Tuple[int, int, int] = (128, 128, 128)
    ) -> Tuple[NDArray[np.float32], NDArray[np.uint8]]:
        """
        Get point cloud for COLMAP export.

        For splats, we use the Gaussian centers as points.
        Colors come from the DC spherical harmonics component.

        Args:
            n_samples: Maximum number of points to return
            color: Fallback RGB color if SH extraction fails

        Returns:
            points: (N, 3) positions in world coordinates
            colors: (N, 3) RGB colors (0-255)
        """
        n_gaussians = len(self.means)

        # Subsample if needed
        if n_samples < n_gaussians:
            indices = np.random.choice(n_gaussians, n_samples, replace=False)
            points = self.means[indices]
            sh_dc = self.sh_coeffs[indices, 0, :]
        else:
            points = self.means
            sh_dc = self.sh_coeffs[:, 0, :]

        # Convert SH DC to RGB
        # The DC component represents color, but needs SH normalization
        # SH DC coefficient: C0 = 0.28209479177387814 (1 / (2 * sqrt(pi)))
        C0 = 0.28209479177387814
        rgb = sh_dc * C0 + 0.5  # Map from SH space to [0, 1]
        rgb = np.clip(rgb, 0.0, 1.0)
        colors = (rgb * 255).astype(np.uint8)

        return points, colors

    def to_ply(self, filepath: str) -> None:
        """
        Write to PLY file in standard 3DGS format.

        Produces a file compatible with from_ply() and standard 3DGS tools.
        All values are written in their native spaces (log-scale, logit-opacity,
        raw SH coefficients).

        Args:
            filepath: Path to output .ply file
        """
        try:
            from plyfile import PlyData, PlyElement
        except ImportError:
            raise ImportError(
                "plyfile is required for writing PLY files. "
                "Install with: pip install plyfile"
            )

        n_gaussians = len(self.means)
        n_sh_rest = self.sh_coeffs.shape[1] - 1  # Total coeffs minus DC

        # Build structured dtype
        props = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
            ('opacity', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ]
        for i in range(n_sh_rest * 3):
            props.append((f'f_rest_{i}', 'f4'))

        vertex_data = np.empty(n_gaussians, dtype=props)

        # Positions
        vertex_data['x'] = self.means[:, 0]
        vertex_data['y'] = self.means[:, 1]
        vertex_data['z'] = self.means[:, 2]

        # Scales (log-space)
        vertex_data['scale_0'] = self.scales[:, 0]
        vertex_data['scale_1'] = self.scales[:, 1]
        vertex_data['scale_2'] = self.scales[:, 2]

        # Rotations (quaternion wxyz)
        vertex_data['rot_0'] = self.quats[:, 0]
        vertex_data['rot_1'] = self.quats[:, 1]
        vertex_data['rot_2'] = self.quats[:, 2]
        vertex_data['rot_3'] = self.quats[:, 3]

        # Opacity (logit-space)
        vertex_data['opacity'] = self.opacities

        # SH DC component
        vertex_data['f_dc_0'] = self.sh_coeffs[:, 0, 0]
        vertex_data['f_dc_1'] = self.sh_coeffs[:, 0, 1]
        vertex_data['f_dc_2'] = self.sh_coeffs[:, 0, 2]

        # SH rest coefficients (interleaved: all ch0, then all ch1, then all ch2)
        for i in range(n_sh_rest):
            vertex_data[f'f_rest_{i}'] = self.sh_coeffs[:, i + 1, 0]
            vertex_data[f'f_rest_{i + n_sh_rest}'] = self.sh_coeffs[:, i + 1, 1]
            vertex_data[f'f_rest_{i + 2 * n_sh_rest}'] = self.sh_coeffs[:, i + 1, 2]

        element = PlyElement.describe(vertex_data, 'vertex')
        PlyData([element]).write(filepath)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"SplatScene(gaussians={len(self.means)}, "
            f"sh_degree={self.sh_degree})"
        )

    def __len__(self) -> int:
        """Return number of Gaussians."""
        return len(self.means)
