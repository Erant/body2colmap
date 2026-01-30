"""
Texture projection utilities for baking rendered views onto mesh UV space.

This module provides functionality to:
1. Generate UV coordinates for meshes (auto-unwrapping)
2. Project rendered images onto UV atlas using face visibility
3. Blend multiple projections from different views

Primary use case: Project Canny edges from circular orbit onto mesh,
then render from helical orbit for diffusion model guidance.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from numpy.typing import NDArray

from .camera import Camera


def generate_cylindrical_uvs(
    vertices: NDArray[np.float32],
    axis: str = "y"
) -> NDArray[np.float32]:
    """
    Generate UV coordinates using cylindrical projection.

    Good for humanoid meshes where the subject is roughly cylindrical
    around the vertical axis.

    Args:
        vertices: Mesh vertices, shape (N, 3)
        axis: Axis of the cylinder ("x", "y", or "z")

    Returns:
        UV coordinates, shape (N, 2), values in [0, 1]
    """
    # Get axis indices
    axis_map = {"x": 0, "y": 1, "z": 2}
    up_axis = axis_map[axis.lower()]
    # The other two axes form the circular cross-section
    other_axes = [i for i in range(3) if i != up_axis]

    # Get coordinates in the cross-section plane
    x_raw = vertices[:, other_axes[0]]
    z_raw = vertices[:, other_axes[1]]

    # CENTER the mesh before computing angles!
    # This is critical - the mesh may have a large offset from origin
    x_center = (x_raw.min() + x_raw.max()) / 2
    z_center = (z_raw.min() + z_raw.max()) / 2
    x = x_raw - x_center
    z = z_raw - z_center

    # Debug: print vertex extents
    print(f"  [UV Debug] Raw X range: [{x_raw.min():.3f}, {x_raw.max():.3f}], centered at {x_center:.3f}")
    print(f"  [UV Debug] Raw Z range: [{z_raw.min():.3f}, {z_raw.max():.3f}], centered at {z_center:.3f}")
    print(f"  [UV Debug] Centered X range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"  [UV Debug] Centered Z range: [{z.min():.3f}, {z.max():.3f}]")

    angles = np.arctan2(z, x)  # Range: [-pi, pi]

    # Normalize to [0, 1]
    u = (angles + np.pi) / (2 * np.pi)

    # V coordinate is height along axis, normalized
    heights = vertices[:, up_axis]
    v_min, v_max = heights.min(), heights.max()
    if v_max - v_min > 1e-6:
        v = (heights - v_min) / (v_max - v_min)
    else:
        v = np.zeros_like(heights)

    # Debug: print UV range
    print(f"  [UV Debug] U range: [{u.min():.3f}, {u.max():.3f}]")
    print(f"  [UV Debug] V range: [{v.min():.3f}, {v.max():.3f}]")

    return np.stack([u, v], axis=1).astype(np.float32)


def generate_spherical_uvs(
    vertices: NDArray[np.float32],
    center: Optional[NDArray[np.float32]] = None
) -> NDArray[np.float32]:
    """
    Generate UV coordinates using spherical projection.

    Args:
        vertices: Mesh vertices, shape (N, 3)
        center: Center point for projection. If None, uses centroid.

    Returns:
        UV coordinates, shape (N, 2), values in [0, 1]
    """
    if center is None:
        center = vertices.mean(axis=0)

    # Direction from center to each vertex
    directions = vertices - center
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-8)  # Avoid division by zero
    normalized = directions / lengths

    # Spherical coordinates
    # theta: azimuth angle (around Y axis)
    # phi: elevation angle (from XZ plane)
    theta = np.arctan2(normalized[:, 2], normalized[:, 0])  # [-pi, pi]
    phi = np.arcsin(np.clip(normalized[:, 1], -1, 1))  # [-pi/2, pi/2]

    # Normalize to [0, 1]
    u = (theta + np.pi) / (2 * np.pi)
    v = (phi + np.pi / 2) / np.pi

    return np.stack([u, v], axis=1).astype(np.float32)


def generate_uvs(
    vertices: NDArray[np.float32],
    faces: NDArray[np.int32],
    method: str = "cylindrical",
    **kwargs
) -> NDArray[np.float32]:
    """
    Generate UV coordinates for a mesh.

    Args:
        vertices: Mesh vertices, shape (N, 3)
        faces: Mesh faces, shape (M, 3)
        method: UV generation method:
            - "cylindrical": Cylindrical projection (good for standing humans)
            - "spherical": Spherical projection
            - "xatlas": Use xatlas for proper unwrapping (if available)
        **kwargs: Method-specific parameters

    Returns:
        UV coordinates, shape (N, 2), values in [0, 1]

    Raises:
        ValueError: If method is not recognized
        ImportError: If xatlas is requested but not installed
    """
    if method == "cylindrical":
        return generate_cylindrical_uvs(vertices, axis=kwargs.get("axis", "y"))
    elif method == "spherical":
        return generate_spherical_uvs(vertices, center=kwargs.get("center"))
    elif method == "xatlas":
        # Use trimesh's built-in unwrap() which uses xatlas internally
        try:
            import trimesh
        except ImportError:
            raise ImportError(
                "trimesh is required for UV unwrapping. "
                "Install with: pip install trimesh"
            )

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        try:
            # trimesh.unwrap() uses xatlas internally
            unwrapped = mesh.unwrap()
            uvs = unwrapped.visual.uv

            # xatlas may create new vertices at seams
            if len(uvs) != len(vertices):
                print(f"Warning: xatlas changed vertex count ({len(vertices)} -> {len(uvs)}), "
                      f"falling back to cylindrical UVs")
                return generate_cylindrical_uvs(vertices)
            return uvs.astype(np.float32)
        except Exception as e:
            # xatlas not installed or failed
            print(f"Warning: UV unwrapping failed ({e}), falling back to cylindrical UVs")
            return generate_cylindrical_uvs(vertices)
    else:
        raise ValueError(f"Unknown UV method: {method}. Use 'cylindrical', 'spherical', or 'xatlas'")


class TextureProjector:
    """
    Projects rendered images onto mesh UV atlas.

    This class accumulates edge/texture information from multiple camera views
    onto a UV texture atlas, handling visibility and blending.

    Example usage:
        projector = TextureProjector(vertices, faces, uv_coords, atlas_size=(1024, 1024))

        for camera, edge_image, face_ids in zip(cameras, edge_images, face_id_buffers):
            projector.project_view(edge_image, camera, face_ids)

        atlas = projector.get_atlas()
    """

    def __init__(
        self,
        vertices: NDArray[np.float32],
        faces: NDArray[np.int32],
        uv_coords: NDArray[np.float32],
        atlas_size: Tuple[int, int] = (1024, 1024)
    ):
        """
        Initialize TextureProjector.

        Args:
            vertices: Mesh vertices, shape (N, 3)
            faces: Mesh face indices, shape (M, 3)
            uv_coords: UV coordinates per vertex, shape (N, 2), values in [0, 1]
            atlas_size: (width, height) of output atlas in pixels
        """
        self.vertices = vertices
        self.faces = faces
        self.uv_coords = uv_coords
        self.atlas_width, self.atlas_height = atlas_size

        # Pre-compute face UVs (3 UV coords per face)
        self.face_uvs = uv_coords[faces]  # Shape: (M, 3, 2)

        # Pre-compute face normals for view-angle weighting
        self._precompute_face_normals()

        # Initialize atlas accumulator (RGBA + weight for blending)
        # Using float for accumulation, will convert to uint8 at end
        self._atlas_accum = np.zeros(
            (self.atlas_height, self.atlas_width, 4),
            dtype=np.float32
        )
        self._atlas_weights = np.zeros(
            (self.atlas_height, self.atlas_width),
            dtype=np.float32
        )
        # For best_angle mode: track best view angle per pixel
        self._atlas_best_angle = np.full(
            (self.atlas_height, self.atlas_width),
            -1.0,  # -1 means no data yet (valid angles are 0 to 1)
            dtype=np.float32
        )

        # Pre-compute UV bounds per face for faster lookup
        self._precompute_face_uv_bounds()

    def _precompute_face_normals(self):
        """Pre-compute face normals for view-angle calculations."""
        # Get vertices for each face
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]

        # Compute face normals via cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        normals = np.cross(edge1, edge2)

        # Normalize
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths = np.maximum(lengths, 1e-8)  # Avoid division by zero
        self.face_normals = normals / lengths  # Shape: (M, 3)

    def _precompute_face_uv_bounds(self):
        """Pre-compute axis-aligned bounding boxes in UV space for each face."""
        self.face_uv_min = self.face_uvs.min(axis=1)  # Shape: (M, 2)
        self.face_uv_max = self.face_uvs.max(axis=1)  # Shape: (M, 2)

    def _barycentric_coords(
        self,
        p: NDArray[np.float32],
        v0: NDArray[np.float32],
        v1: NDArray[np.float32],
        v2: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """
        Compute barycentric coordinates for points in triangles.

        Args:
            p: Points to test, shape (K, 2)
            v0, v1, v2: Triangle vertices, shape (K, 2) each

        Returns:
            (u, v, w): Barycentric coordinates, each shape (K,)
        """
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        v0p = p - v0

        dot00 = np.sum(v0v1 * v0v1, axis=1)
        dot01 = np.sum(v0v1 * v0v2, axis=1)
        dot02 = np.sum(v0v1 * v0p, axis=1)
        dot11 = np.sum(v0v2 * v0v2, axis=1)
        dot12 = np.sum(v0v2 * v0p, axis=1)

        denom = dot00 * dot11 - dot01 * dot01
        # Avoid division by zero for degenerate triangles
        denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)

        v = (dot11 * dot02 - dot01 * dot12) / denom
        w = (dot00 * dot12 - dot01 * dot02) / denom
        u = 1.0 - v - w

        return u, v, w

    def project_view(
        self,
        image: NDArray[np.uint8],
        camera: Camera,
        face_ids: NDArray[np.int32],
        blend_mode: str = "max"
    ) -> None:
        """
        Project a single view onto the UV atlas.

        Uses face-center UVs for robustness (avoids coordinate system mismatches
        between pyrender and camera projection). For best quality, use best_angle
        blend mode which selects the view with most perpendicular viewing angle.

        Args:
            image: Source image to project, shape (H, W, C), RGBA uint8
            camera: Camera that captured the image (used for view-angle weighting)
            face_ids: Face ID buffer from render_face_ids(), shape (H, W)
                     Values are face indices or -1 for background
            blend_mode: How to blend with existing atlas values:
                - "max": Take maximum value (preserves all edges)
                - "replace": Overwrite existing values
                - "average": Running average
                - "best_angle": Keep color from best viewing angle

        Note:
            This method modifies internal state. Call get_atlas() to retrieve result.
        """
        h, w = face_ids.shape
        img_h, img_w = image.shape[:2]

        if (h, w) != (img_h, img_w):
            raise ValueError(f"face_ids shape {(h, w)} doesn't match image shape {(img_h, img_w)}")

        num_faces = len(self.faces)

        # Get all pixels with valid face IDs
        valid_mask = (face_ids >= 0) & (face_ids < num_faces)
        valid_y, valid_x = np.where(valid_mask)
        valid_face_ids = face_ids[valid_mask]

        if len(valid_face_ids) == 0:
            return  # No visible faces

        # Get pixel colors from source image
        pixel_colors = image[valid_y, valid_x].astype(np.float32)  # Shape: (K, 4)

        # Use face center UVs (robust, doesn't depend on projection matching)
        face_center_uvs = self.face_uvs.mean(axis=1)  # Shape: (M, 2)
        pixel_uvs = face_center_uvs[valid_face_ids]  # Shape: (K, 2)

        # Handle UV seam for cylindrical mapping
        # Check if any face crosses the seam by looking at UV range
        face_u_values = self.face_uvs[:, :, 0]  # Shape: (M, 3)
        face_u_range = face_u_values.max(axis=1) - face_u_values.min(axis=1)
        seam_faces = face_u_range > 0.5

        # For seam faces, adjust the center UV
        # If center U < 0.5 but face spans seam, it should be on the "high" side
        for i, face_id in enumerate(valid_face_ids):
            if seam_faces[face_id]:
                # Recompute center with seam handling
                uvs = self.face_uvs[face_id]  # Shape: (3, 2)
                u_vals = uvs[:, 0].copy()
                # Add 1 to small U values before averaging
                u_vals[u_vals < 0.5] += 1.0
                center_u = u_vals.mean() % 1.0
                pixel_uvs[i, 0] = center_u

        # Convert to atlas coordinates
        atlas_x = (pixel_uvs[:, 0] * (self.atlas_width - 1)).astype(np.int32)
        atlas_y = ((1 - pixel_uvs[:, 1]) * (self.atlas_height - 1)).astype(np.int32)

        atlas_x = np.clip(atlas_x, 0, self.atlas_width - 1)
        atlas_y = np.clip(atlas_y, 0, self.atlas_height - 1)

        # For best_angle mode, compute view angles
        if blend_mode == "best_angle":
            # Get face centers in 3D
            face_v0 = self.vertices[self.faces[valid_face_ids, 0]]
            face_v1 = self.vertices[self.faces[valid_face_ids, 1]]
            face_v2 = self.vertices[self.faces[valid_face_ids, 2]]
            face_centers = (face_v0 + face_v1 + face_v2) / 3.0

            # Compute view direction
            view_dirs = camera.position - face_centers
            view_dirs_norm = view_dirs / np.linalg.norm(view_dirs, axis=1, keepdims=True)

            # Get face normals
            face_norms = self.face_normals[valid_face_ids]

            # Dot product = view angle quality (1.0 = head-on)
            dot_products = np.abs(np.sum(view_dirs_norm * face_norms, axis=1))

        # Apply blending
        if blend_mode == "max":
            np.maximum.at(self._atlas_accum, (atlas_y, atlas_x), pixel_colors)
        elif blend_mode == "replace":
            self._atlas_accum[atlas_y, atlas_x] = pixel_colors
        elif blend_mode == "average":
            np.add.at(self._atlas_accum, (atlas_y, atlas_x), pixel_colors)
            np.add.at(self._atlas_weights, (atlas_y, atlas_x), 1)
        elif blend_mode == "best_angle":
            # Keep only the color from the view with best viewing angle
            for i in range(len(atlas_x)):
                ax, ay = atlas_x[i], atlas_y[i]
                if dot_products[i] > self._atlas_best_angle[ay, ax]:
                    self._atlas_best_angle[ay, ax] = dot_products[i]
                    self._atlas_accum[ay, ax] = pixel_colors[i]
        else:
            raise ValueError(f"Unknown blend mode: {blend_mode}")

    def project_view_fast(
        self,
        image: NDArray[np.uint8],
        face_ids: NDArray[np.int32],
        blend_mode: str = "max"
    ) -> None:
        """
        Fast projection using direct UV lookup per face (no barycentric interpolation).

        This is faster but less accurate - uses face center UVs instead of
        per-pixel interpolation. Good for edge maps where exact positioning
        is less critical.

        Args:
            image: Source image to project, shape (H, W, C), RGBA uint8
            face_ids: Face ID buffer, shape (H, W)
            blend_mode: Blending mode ("max", "replace", "average")
        """
        h, w = face_ids.shape
        num_faces = len(self.faces)

        # Get all pixels with valid face IDs
        # Filter out negative IDs (background) AND out-of-bounds IDs
        # (can occur due to anti-aliasing blending the encoded RGB colors)
        valid_mask = (face_ids >= 0) & (face_ids < num_faces)
        valid_y, valid_x = np.where(valid_mask)
        valid_face_ids = face_ids[valid_mask]

        if len(valid_face_ids) == 0:
            return

        # Get pixel colors
        pixel_colors = image[valid_y, valid_x].astype(np.float32)

        # Use face center UV (average of 3 vertices)
        face_center_uvs = self.face_uvs.mean(axis=1)  # Shape: (M, 2)
        pixel_uvs = face_center_uvs[valid_face_ids]  # Shape: (K, 2)

        # Convert to atlas coordinates
        atlas_x = (pixel_uvs[:, 0] * (self.atlas_width - 1)).astype(np.int32)
        atlas_y = ((1 - pixel_uvs[:, 1]) * (self.atlas_height - 1)).astype(np.int32)

        atlas_x = np.clip(atlas_x, 0, self.atlas_width - 1)
        atlas_y = np.clip(atlas_y, 0, self.atlas_height - 1)

        # Apply blending (vectorized for speed)
        if blend_mode == "max":
            # Use numpy's maximum.at for atomic max updates
            np.maximum.at(self._atlas_accum, (atlas_y, atlas_x), pixel_colors)
        elif blend_mode == "replace":
            self._atlas_accum[atlas_y, atlas_x] = pixel_colors
        elif blend_mode == "average":
            np.add.at(self._atlas_accum, (atlas_y, atlas_x), pixel_colors)
            np.add.at(self._atlas_weights, (atlas_y, atlas_x), 1)

    def get_atlas(self) -> NDArray[np.uint8]:
        """
        Get the accumulated texture atlas.

        Returns:
            RGBA texture atlas, shape (atlas_height, atlas_width, 4), dtype uint8
        """
        atlas = self._atlas_accum.copy()

        # Handle averaging if weights were accumulated
        if self._atlas_weights.max() > 1:
            mask = self._atlas_weights > 0
            for c in range(4):
                atlas[:, :, c][mask] /= self._atlas_weights[mask]

        return np.clip(atlas, 0, 255).astype(np.uint8)

    def reset(self) -> None:
        """Reset the atlas accumulator for a new projection pass."""
        self._atlas_accum.fill(0)
        self._atlas_weights.fill(0)
        self._atlas_best_angle.fill(-1.0)


def project_edges_to_atlas(
    vertices: NDArray[np.float32],
    faces: NDArray[np.int32],
    edge_images: List[NDArray[np.uint8]],
    face_id_buffers: List[NDArray[np.int32]],
    cameras: List[Camera],
    atlas_size: Tuple[int, int] = (1024, 1024),
    uv_method: str = "cylindrical",
    blend_mode: str = "max",
    fast_mode: bool = True
) -> Tuple[NDArray[np.uint8], NDArray[np.float32]]:
    """
    Convenience function to project edge images onto UV atlas.

    Args:
        vertices: Mesh vertices, shape (N, 3)
        faces: Mesh faces, shape (M, 3)
        edge_images: List of edge images (RGBA), one per view
        face_id_buffers: List of face ID buffers, one per view
        cameras: List of cameras corresponding to each view
        atlas_size: (width, height) of output atlas
        uv_method: UV generation method ("cylindrical", "spherical", "xatlas")
        blend_mode: Blending mode ("max", "replace", "average")
        fast_mode: Use fast projection (less accurate but faster)

    Returns:
        atlas: RGBA texture atlas, shape (atlas_height, atlas_width, 4)
        uv_coords: Generated UV coordinates, shape (N, 2)
    """
    # Generate UVs
    uv_coords = generate_uvs(vertices, faces, method=uv_method)

    # Create projector
    projector = TextureProjector(vertices, faces, uv_coords, atlas_size)

    # Project all views
    for edge_img, face_ids, camera in zip(edge_images, face_id_buffers, cameras):
        if fast_mode:
            projector.project_view_fast(edge_img, face_ids, blend_mode)
        else:
            projector.project_view(edge_img, camera, face_ids, blend_mode)

    return projector.get_atlas(), uv_coords
