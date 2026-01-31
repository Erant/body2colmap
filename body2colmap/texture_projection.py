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
        Project a single view onto the UV atlas by rasterizing UV triangles.

        For each visible face, computes the average color from the source image
        and fills the corresponding UV triangle in the atlas.

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
        num_faces = len(self.faces)

        # Get all pixels with valid face IDs
        valid_mask = (face_ids >= 0) & (face_ids < num_faces)

        if not np.any(valid_mask):
            return

        # Group pixels by face and compute average color per face
        valid_face_ids = face_ids[valid_mask]
        valid_colors = image[valid_mask].astype(np.float32)

        # Find unique visible faces
        unique_faces = np.unique(valid_face_ids)

        # For best_angle mode, compute view angles per face
        if blend_mode == "best_angle":
            face_centers_3d = self.vertices[self.faces].mean(axis=1)  # (M, 3)
            view_dirs = camera.position - face_centers_3d
            view_dirs = view_dirs / np.linalg.norm(view_dirs, axis=1, keepdims=True)
            face_dot_products = np.abs(np.sum(view_dirs * self.face_normals, axis=1))

        # Process each visible face
        for face_id in unique_faces:
            # Get average color for this face
            face_mask = valid_face_ids == face_id
            face_color = valid_colors[face_mask].mean(axis=0)

            # Get UV coordinates for this face's vertices
            uv0, uv1, uv2 = self.face_uvs[face_id]  # Each is (2,)

            # Handle UV seam
            u_vals = np.array([uv0[0], uv1[0], uv2[0]])
            if u_vals.max() - u_vals.min() > 0.5:
                # Face crosses seam - adjust small values
                if uv0[0] < 0.5: uv0 = (uv0[0] + 1.0, uv0[1])
                if uv1[0] < 0.5: uv1 = (uv1[0] + 1.0, uv1[1])
                if uv2[0] < 0.5: uv2 = (uv2[0] + 1.0, uv2[1])

            # Convert UV to atlas coordinates (no modulo - allow extended coords for seam)
            def uv_to_atlas(uv):
                ax = int(uv[0] * (self.atlas_width - 1))
                ay = int((1 - uv[1]) * (self.atlas_height - 1))
                return ax, np.clip(ay, 0, self.atlas_height - 1)

            p0 = uv_to_atlas(uv0)
            p1 = uv_to_atlas(uv1)
            p2 = uv_to_atlas(uv2)

            # Rasterize triangle in atlas (handles x wrapping for seam triangles)
            atlas_pixels = self._rasterize_triangle_with_wrap(p0, p1, p2)

            # Apply color to atlas pixels
            if blend_mode == "best_angle":
                dot = face_dot_products[face_id]
                for ax, ay in atlas_pixels:
                    if dot > self._atlas_best_angle[ay, ax]:
                        self._atlas_best_angle[ay, ax] = dot
                        self._atlas_accum[ay, ax] = face_color
            elif blend_mode == "max":
                for ax, ay in atlas_pixels:
                    self._atlas_accum[ay, ax] = np.maximum(self._atlas_accum[ay, ax], face_color)
            elif blend_mode == "replace":
                for ax, ay in atlas_pixels:
                    self._atlas_accum[ay, ax] = face_color
            elif blend_mode == "average":
                for ax, ay in atlas_pixels:
                    self._atlas_accum[ay, ax] += face_color
                    self._atlas_weights[ay, ax] += 1

    def _rasterize_triangle_with_wrap(self, p0, p1, p2):
        """
        Rasterize a triangle with x-coordinate wrapping for UV seam handling.

        Uses scanline algorithm. X coordinates may extend beyond atlas width
        for seam-crossing triangles; they are wrapped using modulo.
        """
        # Sort vertices by y coordinate
        pts = sorted([p0, p1, p2], key=lambda p: p[1])
        (x0, y0), (x1, y1), (x2, y2) = pts

        pixels = []

        def add_scanline(y, x_start, x_end):
            if x_start > x_end:
                x_start, x_end = x_end, x_start
            # Don't clamp x range - iterate and wrap each x coordinate
            for x in range(x_start, x_end + 1):
                if 0 <= y < self.atlas_height:
                    # Wrap x coordinate for seam handling
                    wrapped_x = x % self.atlas_width
                    pixels.append((wrapped_x, y))

        # Handle degenerate triangles
        if y0 == y2:
            # Horizontal line
            x_min = min(x0, x1, x2)
            x_max = max(x0, x1, x2)
            add_scanline(y0, x_min, x_max)
            return pixels

        for y in range(max(0, y0), min(self.atlas_height, y2 + 1)):
            # Compute x intersections with triangle edges
            if y < y1:
                # Upper part: edges (p0, p1) and (p0, p2)
                if y1 != y0:
                    xa = x0 + (x1 - x0) * (y - y0) // max(1, y1 - y0)
                else:
                    xa = x0
                xb = x0 + (x2 - x0) * (y - y0) // max(1, y2 - y0)
            else:
                # Lower part: edges (p1, p2) and (p0, p2)
                if y2 != y1:
                    xa = x1 + (x2 - x1) * (y - y1) // max(1, y2 - y1)
                else:
                    xa = x1
                xb = x0 + (x2 - x0) * (y - y0) // max(1, y2 - y0)

            add_scanline(y, xa, xb)

        return pixels

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
