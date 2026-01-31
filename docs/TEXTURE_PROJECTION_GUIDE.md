# Texture Projection Implementation Guide

**Created**: 2026-01-31
**Purpose**: Document lessons learned and provide a clean implementation guide for projecting colors from 2D images onto 3D mesh vertices.

## Overview

This feature enables a two-phase workflow:
1. **Phase 1 (Circular)**: Render mesh from circular orbit → feed to diffusion model
2. **Phase 2 (Helical)**: Project diffusion output colors back onto mesh → render from helical orbit for ControlNet guidance

The key insight is that we store colors **per-vertex** rather than in a UV texture atlas, avoiding UV overlap issues entirely.

---

## Critical Lessons Learned

### 1. Coordinate System Mismatch: The X-Flip

**Problem**: Textures appeared "facing backwards" - front of head on back of mesh.

**Root Cause**: The source images (from diffusion/rendering) have a horizontal mirror relative to our projection math.

**Solution**: Flip X when sampling from the **source image**, but NOT when checking the depth buffer.

```python
# Source images need X-flip
px_flipped = w - 1 - px_int
color = image[py_int, px_flipped]

# Depth buffer uses original coordinates (rendered with our projection)
rendered_depth = depth_buffer[py_int, px_int]
```

**Why this works**:
- The depth buffer is rendered by our code using the same camera/projection
- The source images have different coordinate conventions
- Only the image sampling needs the flip, not the visibility check

### 2. Why Vertex Colors Beat UV Atlas

**Failed approach**: Cylindrical UV mapping + texture atlas

**Problems encountered**:
1. **UV Overlap**: Arms and torso at same angle from Y-axis → same U coordinate → colors bleed between body parts
2. **Face ID Anti-aliasing**: RGB-encoded face IDs blend at triangle edges, producing random valid face IDs that map to wrong triangles
3. **UV Seam Handling**: Triangles crossing the U=0/U=1 boundary require special handling

**Working approach**: Store colors directly on vertices

**Benefits**:
- No UV coordinate issues
- No face ID encoding/decoding
- Simpler projection math
- Colors interpolate naturally during rendering

### 3. Depth Buffer Visibility vs Face IDs

**Failed approach**: Render face IDs (encode face index in RGB), decode to determine visibility

**Problem**: Anti-aliasing blends RGB values at edges, producing invalid or random face IDs

**Working approach**: Direct depth buffer comparison

```python
# Project vertex to screen
projected = camera.project(vertex)
px, py = int(round(projected[0])), int(round(projected[1]))

# Check depth buffer at projected location
rendered_depth = depth_buffer[py, px]
vertex_depth = compute_vertex_depth(vertex, camera)

# Visible if vertex is at or in front of rendered surface (2cm tolerance)
if vertex_depth <= rendered_depth + 0.02:  # Absolute tolerance in meters
    # Vertex is visible from this camera
```

### 4. Depth Tolerance and Backface Culling

**Lesson**: Use absolute tolerance for depth, not relative

- **Depth tolerance**: Use **absolute** 2cm (`vertex_depth <= rendered_depth + 0.02`)
  - Relative 10% at 2.5m = 25cm tolerance → arm/torso color bleed
  - Absolute 2cm handles mesh discretization without allowing nearby body parts to bleed
- **Backface culling**: Allow `dot > -0.1` (slightly negative) for grazing angles

Too strict = missing colors on many vertices (gray patches)

### 5. Background Pixel Filtering

Diffusion output may have white backgrounds or transparent regions that shouldn't be projected.

```python
# Skip background pixels
is_white = color[:3].min() > 240
is_transparent = color[3] < 10 if len(color) > 3 else False
if is_white or is_transparent:
    continue
```

### 6. View-Angle Weighted Blending

For vertices visible from multiple cameras, use the view with the best angle (most perpendicular to surface).

```python
# Compute dot product between view direction and vertex normal
view_dir = normalize(camera.position - vertex)
dot = dot_product(view_dir, vertex_normal)

# Higher dot = better viewing angle
if dot > vertex_best_angle[vi]:
    vertex_best_angle[vi] = dot
    vertex_colors[vi] = sampled_color
```

---

## Clean Implementation Architecture

Following the project's design principles: coordinate conversions at boundaries only, camera moves not mesh, explicit transforms.

### Module Structure

```
body2colmap/
├── texture_projection.py    # Core projection logic
│   ├── VertexColorProjector  # Main class
│   └── (remove TextureProjector, UV generation - unused)
├── projection_pipeline.py   # High-level orchestration
└── renderer.py              # Add render_vertex_colors() mode
```

### VertexColorProjector - Clean Implementation

```python
class VertexColorProjector:
    """
    Projects colors from 2D images directly onto mesh vertices.

    Coordinate Convention:
    - All 3D coordinates are in World space (Y-up, camera looks down -Z)
    - Image coordinates: (row, col) where row 0 is top
    - Projection output: (x, y) where x is column, y is row

    CRITICAL: Source images require X-flip when sampling due to
    coordinate convention mismatch. Depth buffer does NOT need flip.
    """

    def __init__(self, vertices: NDArray, faces: NDArray):
        self.vertices = vertices
        self.faces = faces
        self.num_vertices = len(vertices)

        # Pre-compute face normals, then vertex normals
        self.face_normals = self._compute_face_normals()
        self.vertex_normals = self._compute_vertex_normals()

        # Accumulator state
        self._colors = np.zeros((self.num_vertices, 4), dtype=np.float32)
        self._best_angles = np.full(self.num_vertices, -1.0, dtype=np.float32)

    def project_view(
        self,
        image: NDArray[np.uint8],      # Source image (H, W, 4) RGBA
        depth_buffer: NDArray[np.float32],  # Depth from same camera (H, W)
        camera: Camera,
        depth_tolerance: float = 0.02
    ) -> None:
        """
        Project colors from a single view onto visible vertices.

        Args:
            image: Source image to sample colors from
            depth_buffer: Rendered depth buffer for visibility testing
            camera: Camera used to render both image and depth buffer
            depth_tolerance: Absolute tolerance in meters (default 2cm)
        """
        h, w = depth_buffer.shape

        # Project all vertices to screen coordinates
        projected = camera.project(self.vertices)  # (N, 2) as (x, y)

        # Compute vertex depths in camera space
        vertex_depths = self._compute_vertex_depths(camera)

        # Compute view quality (dot product with normal)
        view_dirs = camera.position - self.vertices
        view_dirs /= np.linalg.norm(view_dirs, axis=1, keepdims=True)
        dots = np.sum(view_dirs * self.vertex_normals, axis=1)

        for vi in range(self.num_vertices):
            px, py = int(round(projected[vi, 0])), int(round(projected[vi, 1]))

            # Bounds check
            if not (0 <= px < w and 0 <= py < h):
                continue

            # Backface culling (relaxed for grazing angles)
            if dots[vi] < -0.1:
                continue

            # Visibility check using depth buffer (ORIGINAL coordinates)
            rendered_depth = depth_buffer[py, px]
            if rendered_depth > 0:
                if vertex_depths[vi] > rendered_depth + depth_tolerance:
                    continue  # Occluded (absolute tolerance, default 2cm)

            # Sample color with X-FLIP (source image coordinate mismatch)
            px_flipped = w - 1 - px
            color = image[py, px_flipped].astype(np.float32)

            # Skip background pixels
            if color[:3].min() > 240 or color[3] < 10:
                continue

            # Update if this view has better angle
            if dots[vi] > self._best_angles[vi]:
                self._best_angles[vi] = dots[vi]
                self._colors[vi] = color

    def get_vertex_colors(self) -> NDArray[np.uint8]:
        """Return accumulated vertex colors as RGBA uint8."""
        return np.clip(self._colors, 0, 255).astype(np.uint8)

    def _compute_vertex_depths(self, camera: Camera) -> NDArray[np.float32]:
        """Compute depth of each vertex in camera space."""
        w2c = camera.get_w2c()
        vertices_h = np.hstack([self.vertices, np.ones((self.num_vertices, 1))])
        vertices_cam = (w2c @ vertices_h.T).T[:, :3]
        return -vertices_cam[:, 2]  # Negate: camera looks down -Z
```

### Pipeline Integration

```python
def generate_vertex_colors_from_images(
    pipeline: ProjectionPipeline,
    images: List[NDArray[np.uint8]]
) -> NDArray[np.uint8]:
    """
    Project colors from circular orbit images onto mesh vertices.

    Args:
        pipeline: Initialized pipeline with circular cameras set up
        images: Diffusion-processed images, one per circular camera

    Returns:
        Vertex colors (N, 4) RGBA uint8
    """
    projector = VertexColorProjector(
        pipeline.scene.vertices,
        pipeline.scene.faces
    )

    renderer = Renderer(pipeline.scene, pipeline.render_size)

    for camera, image in zip(pipeline.circular_cameras, images):
        # Render depth buffer for visibility testing
        depth_buffer = renderer.render_depth_raw(camera)

        # Project this view's colors
        projector.project_view(image, depth_buffer, camera)

    return projector.get_vertex_colors()
```

### Rendering with Vertex Colors

Add to `Renderer` class:

```python
def render_vertex_colors(
    self,
    camera: Camera,
    vertex_colors: NDArray[np.uint8],
    bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> NDArray[np.uint8]:
    """
    Render mesh with per-vertex colors.

    Args:
        camera: Camera to render from
        vertex_colors: RGBA colors per vertex (N, 4)
        bg_color: Background color

    Returns:
        Rendered RGBA image
    """
    # Create trimesh with vertex colors
    mesh = trimesh.Trimesh(
        vertices=self.scene.vertices,
        faces=self.scene.faces,
        vertex_colors=vertex_colors,
        process=False
    )

    # Convert to pyrender mesh and render
    # ... standard pyrender setup ...
```

---

## Pitfalls to Avoid

### 1. Don't Use Face ID Buffers for Visibility
Anti-aliasing corrupts RGB-encoded face indices. Use depth buffer instead.

### 2. Don't Apply X-Flip to Depth Buffer
The depth buffer is rendered by our code with our projection. Only the source images need the flip.

### 3. Don't Use Cylindrical UVs for Human Bodies
Arms and torso overlap in UV space. Vertex colors avoid this entirely.

### 4. Don't Be Too Strict with Depth/Angle Tolerances
- Depth: 10% tolerance handles mesh discretization
- Backface: Allow slightly negative dots for edges

### 5. Don't Forget Background Filtering
White (>240) and transparent (<10 alpha) pixels are background, not texture.

### 6. Don't Mix Coordinate Systems Mid-Pipeline
All 3D is in World coords. Conversion to COLMAP/OpenCV happens only at export boundary.

---

## Configuration Defaults

For the projection pipeline CLI:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--n-frames` | 81 | Match circular orbit frame count |
| `--n-loops` | 2 | Sufficient coverage without redundancy |
| `--base-mode` | mesh | Better for 3DGS than depth |
| `--skeleton` | true | Provides pose guidance |
| `--joint-radius` | 0.006 | Subtle overlay |
| `--bone-radius` | 0.003 | Subtle overlay |
| `--blend-mode` | best_angle | Use best view per vertex |
| `--depth-tolerance` | 0.02 | 2cm absolute (meters) |

---

## Testing Checklist

1. **Orientation test**: Render front view, verify face texture is on front of mesh
2. **Occlusion test**: Arm in front of body should have arm color, not body color
3. **Coverage test**: All visible vertices should have color (no gray patches)
4. **Seam test**: No visible seams or discontinuities at body part boundaries

---

## Summary

The key insights for clean implementation:

1. **Vertex colors, not UV atlas** - avoids overlap and seam issues
2. **Depth buffer visibility, not face IDs** - robust to anti-aliasing
3. **X-flip for source images only** - depth buffer uses original projection
4. **Generous tolerances** - 10% depth, relaxed backface culling
5. **Best-angle blending** - pick color from most perpendicular view
6. **Background filtering** - skip white/transparent pixels

Following these principles, the implementation is straightforward: project each vertex, check visibility with depth buffer, sample color with X-flip, keep best-angle result.
