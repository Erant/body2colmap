# Body2COLMAP Package Implementation Notes

## Module Overview

This directory contains the core implementation of the body2colmap tool.

### Module Dependencies

```
coordinates.py  (no dependencies - pure math)
    ↓
camera.py  (depends on: coordinates)
    ↓
scene.py  (depends on: coordinates)
    ↓
face.py  (no dependencies - pure math + embedded data)
    ↓
path.py  (depends on: camera, coordinates)
    ↓
splat_scene.py  (no dependencies - pure data + PLY I/O)
    ↓
splat_anchor.py  (depends on: camera, splat_scene)
    ↓
renderer.py  (depends on: scene, camera, face, skeleton)
    ↓
exporter.py  (depends on: camera, scene)
    ↓
pipeline.py  (depends on: all of the above)
    ↓
cli.py  (depends on: pipeline, config)
```

### Coordinate System Convention

**ALL modules use World/Renderer coordinates internally:**
- Origin: At mesh position (after applying cam_t from SAM-3D-Body)
- +Y: Up (toward head)
- +Z: Out of screen (toward viewer)
- +X: Right
- Camera looks down -Z axis

**Coordinate conversions happen ONLY in:**
1. `scene.py::Scene.from_sam3d_output()` - SAM-3D → World
2. `splat_anchor.py::compute_anchor_transform()` - external splat → World
3. `exporter.py::ColmapExporter.export()` - World → COLMAP/OpenCV

### Module Descriptions

#### coordinates.py
**Purpose**: Define coordinate systems and provide conversion functions

**Key components**:
- `WorldCoordinates`: Class documenting the canonical coordinate system
- `sam3d_to_world()`: Convert SAM-3D-Body vertices to world coords
- `world_to_colmap_camera()`: Convert camera pose for COLMAP export
- `rotation_to_quaternion_wxyz()`: Rotation matrix → quaternion (w,x,y,z)
- `look_at_matrix()`: Build camera-to-world matrix from eye, target, up

**Testing priority**: HIGH - all other modules depend on this being correct

#### camera.py
**Purpose**: Camera representation with intrinsics and extrinsics

**Key components**:
- `Camera` class:
  - Intrinsics: focal_length, principal_point, image_size
  - Extrinsics: position (world coords), rotation (c2w)
  - Methods: look_at(), get_c2w(), get_w2c(), get_colmap_quat_tvec()

**Testing priority**: HIGH - used everywhere

#### scene.py
**Purpose**: Manage 3D scene (mesh, skeleton, lighting)

**Key components**:
- `Scene` class:
  - Mesh: trimesh.Trimesh in world coordinates
  - Skeleton: optional joint positions and bone connections
  - Methods: from_sam3d_output(), get_point_cloud(), get_bounds()

**Testing priority**: MEDIUM - mainly data container

#### path.py
**Purpose**: Generate camera orbit paths

**Key components**:
- `OrbitPath` class:
  - `circular()`: Fixed elevation, rotating azimuth
  - `sinusoidal()`: Oscillating elevation
  - `helical()`: Multiple loops with linear elevation change, plus a uniform
    `elevation_offset_deg` shift used by anchored orbits
- Module-level helpers (usable outside `OrbitPipeline`):
  - `helical_elevation_deg()`: The helix elevation ramp as a pure function of
    fractional progress. Single source of truth shared by `helical()` and the
    anchor solver — do NOT re-inline this logic.
  - `compute_original_camera_orbit_params()`: Radius/azimuth/elevation that put
    a circular orbit's frame 0 on a given camera (default: the origin)
  - `compute_helical_anchor_params()`: Solves `start_azimuth_deg`,
    `elevation_offset_deg` and `anchor_frame_index` so a helix passes through a
    given camera. Raises `ValueError` rather than emitting a degenerate path.

**Returns**: List[Camera] with positions and orientations set

**Testing priority**: HIGH - path patterns must be correct for good 3DGS training

#### renderer.py
**Purpose**: Render images using pyrender

**Key components**:
- `Renderer` class:
  - `render_mesh()`: Color mesh with lighting
  - `render_depth()`: Depth buffer
  - `render_mask()`: Boolean silhouette coverage from the depth buffer
  - `render_outline()`: Flat two-tone silhouette (`filled` or `stroke`)
  - `render_skeleton()`: Joints and bones
  - `render_composite()`: Base layer (`mesh`/`depth`/`outline`) + overlays,
    plus an optional pre-rendered `splat_layer` composited last
  - `_render_depth_buffer()`: The raw pyrender depth buffer. Single source of
    truth for both `render_depth()` and `render_mask()` — and for the splat
    overlay's depth-gauge fit. Do NOT re-open a pyrender scene to read depth.
  - Alpha channel handling
- Module-level `parse_composite_modes()`: the ONE place mode strings are split
  and validated, with `BASE_LAYERS` / `OVERLAY_LAYERS` naming the vocabulary.
  Used by `cli.py` and `pipeline.render_original_view()`; both used to parse
  independently, so a new layer had to be added to each by hand.

**Testing priority**: MEDIUM - mainly wraps pyrender

#### exporter.py
**Purpose**: Export to COLMAP and other formats

**Key components**:
- `ColmapExporter`: Write cameras.txt, images.txt, points3D.txt
- `ImageExporter`: Save PNG/JPG images
- Coordinate conversion: World → COLMAP (OpenCV convention)

**Testing priority**: HIGH - output must match COLMAP spec exactly

#### pipeline.py
**Purpose**: High-level API orchestrating all components

**Key components**:
- `OrbitPipeline` class:
  - Load SAM-3D-Body output
  - Generate orbit path
  - Render frames
  - Export COLMAP and images

**`orbit_params` contract** (populated by `set_orbit_params()`; the
original-camera keys only appear when `original_focal_length` is set):
- `anchor_frame_index`: Index of the frame sitting at the original camera.
  0 for circular and sinusoidal, solved for on helical — **always read this
  key, never assume 0.**
- `anchor_camera`: The `Camera` at that index (was `frame0_camera` before
  helical anchoring landed).
- `anchor_elevation_offset_deg`: Uniform tilt applied to the helix to make the
  anchor exact. 0.0 for non-helical patterns.
- `warp_homography`: 3x3 matrix aligning the original image with the anchor
  frame's view, for `cv2.warpPerspective()`.
- Also: `pattern`, `n_frames`, `radius`, `original_focal_length`,
  `framed_focal_length`, `start_azimuth_deg`, `derived_elevation_deg`,
  `framing_info`.

**Testing priority**: MEDIUM - integration tests cover this

#### config.py
**Purpose**: Configuration management

**Key components**:
- Parse CLI arguments
- Load/validate YAML config files
- Merge configs with sensible defaults

**Testing priority**: LOW - mostly validation logic

#### cli.py
**Purpose**: Command-line interface

**Key components**:
- Argument parser setup
- Main entry point
- Progress reporting

**Testing priority**: LOW - thin wrapper around pipeline

## Implementation Order

1. **coordinates.py** - Foundation for everything else
2. **camera.py** - Needed by path and renderer
3. **scene.py** - Loads input data
4. **path.py** - Generates camera positions
5. **renderer.py** - Produces images
6. **exporter.py** - Writes output files
7. **pipeline.py** - Ties it all together
8. **config.py** + **cli.py** - User interface

## Testing Strategy

### Unit Tests
Each module has isolated tests:
- `coordinates.py`: Test conversions with known values
- `camera.py`: Test look_at, c2w/w2c, quaternions
- `path.py`: Test orbit patterns produce expected positions
- `scene.py`: Test loading SAM-3D data
- `renderer.py`: Test rendering modes (may need fixtures)
- `exporter.py`: Test COLMAP format output

### Integration Tests
- Load real SAM-3D-Body output → render → export → verify in COLMAP viewer
- End-to-end pipeline test

### Validation Tests
- Feed COLMAP output into 3DGS training
- Verify point cloud and cameras align visually

## Common Pitfalls

### Coordinate System Confusion
- **Always document**: What coordinate system is this in?
- **Name variables clearly**: `pos_world`, `rot_c2w`, `quat_w2c_wxyz`
- **Test conversions**: Forward + reverse should be identity

### Quaternion Conventions
- COLMAP uses (w, x, y, z) order
- scipy.spatial.transform.Rotation uses (x, y, z, w)
- **Always specify**: quaternion_wxyz or quaternion_xyzw

### Camera Pose Conventions
- c2w (camera-to-world): Transforms camera-local points → world
  - Camera position = c2w[:3, 3]
  - Camera forward = -c2w[:3, 2] (looks down -Z)
- w2c (world-to-camera): Transforms world points → camera
  - w2c = inverse(c2w)
  - For rotation only: R_w2c = R_c2w.T

### Alpha Channel
- Mesh/depth/outline = 1.0 where surface exists, 0.0 background
- Skeleton does NOT contribute to alpha (for masking)
- Use RGBA throughout, convert to RGB if needed for export

## Performance Notes

### Potential Bottlenecks
1. **Rendering**: pyrender is CPU-based, can be slow
   - Consider: parallel rendering if multiple cores available
   - Consider: optional GPU backend (pyrender + EGL)

2. **Point cloud sampling**: trimesh.sample_surface can be slow for dense sampling
   - Default 50k points is reasonable
   - May need progress reporting for >100k points

3. **Image I/O**: Writing many PNG files can be slow
   - Consider: batch writes
   - Consider: optional video output

### Optimization Later
- Don't optimize prematurely
- First: get it working correctly
- Then: profile with real data
- Only optimize bottlenecks

## External Dependencies

**Core rendering**:
- `pyrender`: OpenGL-based mesh rendering
- `trimesh`: Mesh operations, surface sampling
- `numpy`: All math operations
- `opencv-python` (cv2): Image I/O and color conversion

**Utilities**:
- `scipy`: Rotation utilities (quaternion conversions)
- `pyyaml`: Config file parsing (optional)
- `click` or `argparse`: CLI argument parsing

**Optional**:
- `matplotlib`: Colormaps for depth visualization
- `tqdm`: Progress bars
- `pillow`: Alternative image I/O

## Production Lessons Learned (2026-01-20)

### ✅ Implementation Complete!
All modules are functional and tested with real SAM-3D-Body output. Successfully generates training data for 3D Gaussian Splatting.

### Critical Bugs Fixed

#### 1. Portrait Auto-Framing (`pipeline.py:141-172`)
**The Problem**: Figure appeared tiny in portrait orientations (e.g., 720x1280).

**Root Causes**:
1. Used 3D bounding box diagonal for scene size (dominated by height)
2. Only considered X dimension for horizontal extent (ignores Z)

**The Fix**:
```python
# For width: use max(X, Z) because camera orbits in XZ plane
scene_width = max(
    max_corner[0] - min_corner[0],  # Front/back views see X
    max_corner[2] - min_corner[2]   # Side views see Z
)
scene_height = max_corner[1] - min_corner[1]

# Compute radius needed for each dimension
radius_h = (scene_width / 2.0) / np.tan(horizontal_fov_rad * fill_ratio / 2.0)
radius_v = (scene_height / 2.0) / np.tan(vertical_fov_rad * fill_ratio / 2.0)
radius = max(radius_h, radius_v)  # Use larger to fit both
```

**Key Insight**: Orbiting cameras see different projections at different angles. Horizontal extent must consider BOTH X and Z dimensions.

#### 2. Camera Look-At Target (`scene.py:201-209`, `pipeline.py:176`)
**The Problem**: Camera pointed too high on human meshes.

**Root Cause**: `get_centroid()` computes mean of vertices, which is biased by vertex density. Human meshes have more vertices in face/hands, pulling centroid upward.

**The Fix**: Added `get_bbox_center()` which returns geometric center of bounding box:
```python
def get_bbox_center(self) -> NDArray[np.float32]:
    """Geometric center, unaffected by vertex density."""
    min_corner, max_corner = self.get_bounds()
    return (min_corner + max_corner) / 2.0
```

**Key Insight**: For camera framing, use geometric center (bbox), not statistical center (centroid).

#### 3. COLMAP Filename Mismatch (`pipeline.py:329-370`, `cli.py:157-163`)
**The Problem**: Custom `filename_pattern` used for images but not COLMAP metadata.

**Root Cause**: `export_colmap()` hardcoded default pattern, `export_images()` used config pattern.

**The Fix**: Added `filename_pattern` parameter to `export_colmap()` and passed through consistently:
```python
def export_colmap(
    self,
    output_dir: str,
    n_pointcloud_samples: int = 50000,
    filename_pattern: str = "frame_{:04d}.png"  # NEW
) -> Path:
    image_names = ImageExporter.generate_filenames(
        n_frames=len(self.cameras),
        pattern=filename_pattern  # Use provided pattern
    )
```

**Key Insight**: When multiple export functions reference same files, ensure pattern consistency.

#### 4. Config Override Bypass (`config.py:200-350`)
**The Problem**: Config file `n_frames: 81` ignored, always produced 120 frames.

**Root Cause**: CLI argument had `default=120`, so argparse always provided a value even when user didn't specify it.

**The Fix**: Remove defaults from override arguments, check for None:
```python
# In argument parser - NO DEFAULT
parser.add_argument("--n-frames", type=int)  # Not: default=120

# In override logic - CHECK FOR NONE
if args.n_frames is not None:  # Not: if args.n_frames:
    config.path.n_frames = args.n_frames
```

**Key Insight**: For CLI overrides of config file values, arguments must have NO default and use None-checking.

### Face Landmark Rendering (`face.py`)

**Purpose**: Generate OpenPose Face 70 keypoints anchored to skeleton head joints.

**Architecture**:
- No MediaPipe dependency. Canonical face geometry embedded as constant array.
- 70 keypoints extracted from MediaPipe's canonical_face_model.obj via MaixPy mapping.
- Pupils (indices 68-69) synthesized as centroids of 6-point eye contours.
- 5 anchor points (nose, eyes, ears) shared between face model and skeleton.
- Procrustes alignment (SVD-based) computes scale + rotation + translation.

**Face Visibility**: Hemisphere test using dot product of face normal and view direction.
Face rendered only when facing camera (frontal 180 degrees). This prevents ControlNet/3DGS
from seeing facial features on the back of the head.

**Face Normal Computation**: cross(left_eye - right_eye, nose_bridge_top - chin).
These two vectors span the face plane horizontally and vertically; their cross product
gives the outward-facing normal.

**Rendering**: White points (icospheres) + optional white cylinders for the OpenPose
face bone connections. Smaller geometry than body skeleton (~35% of body joint/bone radii).
The eyes are the exception — see below.

**Eye Rendering**: The eyes are drawn as filled two-tone shapes, not landmark dots.
`build_eye_geometry()` turns each 6-point eye contour (36-41, 42-47) into a flat
sclera surface with a pupil disc centered on the pupil landmark (68/69).
`get_face_draw_lists(eye_style)` is what the renderer iterates, so the dots and the
eye outline segments the shapes replace are never drawn on top of them.

`eye_style="dots"` is the escape hatch back to the original rendering: it makes
`get_face_draw_lists()` return every point and bone, and `_render_face()` returns
before building any eye geometry. There is no duplicated draw loop.

Three properties matter:
1. **The contour is flattened into its own plane.** The eye opening curves around
   the eyeball, and that curvature bulges in front of the (flat) pupil disc and
   clips it into a bowtie. Both shapes read as flat areas anyway, so the
   projection costs nothing visually. Do not remove it.
2. **Eye height is measured at the pupil**, as twice the distance from the pupil
   center to the nearest lid segment — not as the height of the whole contour. The
   pupil sits off-center, so a disc sized from the tallest part of the opening
   would poke through a lid near the corners. With this definition,
   `pupil_scale = 1.0` touches both lids exactly and `pupil_scale <= 1.0`
   (asserted) is what guarantees the pupil never spills out.
3. **Winding is decided from the geometry**, not assumed: the left and right
   contours run in opposite directions, and pyrender backface-culls by default.

The 6 contour points are Catmull-Rom resampled to `EYE_CONTOUR_SEGMENTS` before
filling — a raw hexagon reads as angular and its straight edges cut the corners
off the opening.

**CLI**: `--face-mode full|points|none`, `--eye-style shape|dots`, `--eye-color`,
`--pupil-color`, `--pupil-scale`, composite mode `skeleton+face`.

**Standalone utility**: `tools/extract_face_landmarks.py` runs MediaPipe Face Mesh on
an image and outputs JSON with 70 OpenPose-format keypoints. Only dependency on
mediapipe is in this utility, not the main package.

### Skeleton Rendering Details (`skeleton.py`)

#### Official Bone Connectivity
Source: https://github.com/facebookresearch/sam-3d-body/blob/main/sam_3d_body/metadata/mhr70.py

**MHR70 format**: 70 joints total
- 11 leg bones
- 7 torso bones
- 7 head bones
- 20 left hand bones
- 20 right hand bones

**OpenPose Body25+Hands**: 65 joints (MHR70 converted)
- 25 body joints (Body25)
- 20 left hand joints
- 20 right hand joints
- MidHip computed as average of left/right hips

#### Official Color Palette
Source: OpenPose `poseParametersRender.hpp`

**Body25 colors**: 25-color rainbow gradient (pink-red → purple)
- **Gotcha**: Official source has duplicate red at index 8 (same as index 1)
- **Our fix**: Changed index 8 to cyan-green `(0, 255, 85)` for proper gradient

**Hand colors**: Per-finger colors for visual distinction
- Thumb: Pink-red
- Index: Orange
- Middle: Green
- Ring: Cyan
- Pinky: Purple

### Auto-Framing Algorithm

**Complete algorithm** for proper framing across all aspect ratios:

```python
# 1. Get scene bounds
min_corner, max_corner = scene.get_bounds()

# 2. Compute per-dimension extents
#    For width: max of X and Z (camera orbits in XZ plane)
#    For height: just Y (up dimension)
scene_width = max(
    max_corner[0] - min_corner[0],  # X extent
    max_corner[2] - min_corner[2]   # Z extent
)
scene_height = max_corner[1] - min_corner[1]  # Y extent

# 3. Compute FOVs based on image dimensions and focal length
width, height = render_size
horizontal_fov_rad = 2 * np.arctan(width / (2 * focal_length))
vertical_fov_rad = 2 * np.arctan(height / (2 * focal_length))

# 4. Compute radius needed for each dimension
desired_h_angle = horizontal_fov_rad * fill_ratio
radius_h = (scene_width / 2.0) / np.tan(desired_h_angle / 2.0)

desired_v_angle = vertical_fov_rad * fill_ratio
radius_v = (scene_height / 2.0) / np.tan(desired_v_angle / 2.0)

# 5. Use max to ensure fit in both dimensions
radius = max(radius_h, radius_v)

# 6. Look at bbox center (not centroid)
target = scene.get_bbox_center()
```

**Why this works**:
- Portrait (720x1280): Vertical FOV is narrow, `radius_v` dominates
- Landscape (1280x720): Horizontal FOV is narrow, `radius_h` dominates
- Square (1024x1024): Whichever dimension is larger in scene dominates
- Orbit visibility: Using `max(X, Z)` ensures figure fits from all angles

### Configuration Management Pattern (`config.py`)

**Three-tier precedence** (highest to lowest):
1. CLI arguments (explicitly set by user)
2. Config file values (YAML)
3. Hardcoded defaults (in dataclass definitions)

**Implementation pattern**:
```python
# 1. Load config from YAML (or create default)
config = Config.from_yaml(args.config) if args.config else Config()

# 2. Override with CLI args ONLY if explicitly set (not None)
if args.output_dir is not None:
    config.export.output_dir = args.output_dir

if args.n_frames is not None:
    config.path.n_frames = args.n_frames

# 3. Handle compound overrides (e.g., resolution)
if args.resolution is not None:
    width, height = parse_resolution(args.resolution)
    config.render.resolution = (width, height)
elif args.width is not None or args.height is not None:
    # Allow individual width/height to override just one dimension
    width = args.width if args.width is not None else config.render.resolution[0]
    height = args.height if args.height is not None else config.render.resolution[1]
    config.render.resolution = (width, height)
```

### Composite Rendering Pattern (`renderer.py`, `pipeline.py`)

**Composite modes** (e.g., "depth+skeleton", "outline+skeleton", "depth+skeleton+face"):

Recognized base layers are `mesh`, `depth` and `outline` (checked in that
order in `render_composite()`); recognized overlays are `skeleton` and `face`.

1. Parse composite string: `"depth+skeleton+face"` → base="depth", overlays=["skeleton", "face"]
2. Render base mode to RGBA
3. For each overlay:
   - Render overlay mode to RGBA
   - Alpha-blend overlay onto base using OpenCV:
   ```python
   alpha = overlay[:, :, 3:4] / 255.0
   composite = composite * (1 - alpha) + overlay[:, :, :3] * alpha
   composite_alpha = np.maximum(composite_alpha, overlay[:, :, 3])
   ```
4. Return composite RGBA

**Key**: Skeleton renders with transparent background, so it only appears over mesh/depth.

## Outline Mode (2026-08)

### What it is
`outline` renders the mesh as a **flat two-tone image**: every pixel the mesh
covers gets `fg_color`, everything else gets `bg_color`. There is no lighting,
no shading and no gradient, so the only information in the image is the shape
of the silhouette. Intended as a control/conditioning image, and combinable
with the skeleton overlay as `outline+skeleton`.

### Key design: silhouette comes from the depth buffer
`render_mask()` derives coverage from `depth > 0`, **not** from the color
buffer's alpha. The depth buffer is unaffected by mesh color, lighting or
anti-aliasing, so the mask is exact and binary. `render_outline()` and any
future mask-consuming code should go through `render_mask()` rather than
re-deriving coverage from a color render.

### Key design: two styles from one mask
- `filled` (default): the whole silhouette gets `fg_color`.
- `stroke`: only a band along the silhouette boundary gets `fg_color`; the
  interior gets `bg_color`. The band is `dilate(mask) & ~erode(mask)` with an
  elliptical kernel of radius `round(thickness / 2)`, so it straddles the
  boundary and comes out ~`thickness` px across. Disconnected components and
  interior holes are stroked correctly for free.

OpenCV's default morphology border handling treats outside-the-image as
neutral, so a silhouette running off the edge of the frame is **not** given a
spurious stroke along the image border. Do not "fix" this by passing an
explicit border value.

### Key design: blur lives in `render_outline()`, not `render_composite()`
`blur` (radius in px, default 4, `0` disables) applies a Gaussian to color and
alpha together so the two edges stay in step. It is deliberately applied at the
end of `render_outline()` rather than to the finished composite: overlays are
blended on top of the already-blurred base, so the skeleton in
`outline+skeleton` stays pixel-sharp. Do not move this to the composite stage —
that would smear the skeleton too.

Kernel size is `2 * blur + 1` with OpenCV's auto-derived sigma.

### Key design: alpha tracks mesh coverage, not the drawn foreground
Alpha is `mask | fg_mask`, i.e. the silhouette plus (for `stroke`) the outward
half of the band. Two reasons:
1. **Consistency**: `mesh` and `depth` set alpha to mesh coverage, so `outline`
   drops into `render_composite()` as a base layer and works as a 3DGS training
   mask, with the same semantics.
2. **No clipping**: half a stroke band lies *outside* the silhouette. If alpha
   were just the drawn foreground, `outline+skeleton` in `stroke` style would
   render the skeleton into pixels with alpha=0 — invisible in the saved RGBA.

For `filled` the union is exactly the silhouette, so the two rules coincide;
the distinction only matters for `stroke`, where it costs a ~`thickness/2` px
dilation of the mask.

### Configuration
`RenderConfig.outline_color` (foreground), `outline_bg_color`,
`outline_style`, `outline_thickness`, `outline_blur` — plumbed through YAML,
`--outline-*` CLI flags, `pipeline.render_all()`,
`pipeline.render_original_view()` and `renderer.render_composite()` (where the
keys are `fg_color`, `bg_color`, `style`, `thickness`, `blur`).


## Gaussian-Splat Overlay (2026-08)

See the parent `CLAUDE.md` for the geometry and the measured constants. Notes
that matter when touching the code:

### `splat_anchor.py` is a boundary module
It is the third coordinate conversion point. The whole transform is
`P_world = M @ (p_ply + centroid)` with `M` upper-triangular — no rotation is
solved for, because both frames put their source camera at identity rotation by
construction. Do not add a registration/ICP step: it would break the exact 2-D
reprojection the anchor rests on, which is verified by the (0,0) best-fit-shift
gate.

### `transform_splat_scene()` re-decomposes covariances, and must stay exact
Gaussians carry orientation, so a non-uniform `M` cannot just scale them:
`Sigma' = M Sigma M^T`, then `eigh` back to a rotation and three axis scales.
Three details are load-bearing:

1. **Eigenvalues are sorted descending** so the smallest axis stays in column 2.
   That preserves the upstream "surface normal in the shortest-scale axis"
   convention, which 3DGS normal-supervision trainers assume.
2. **`det(R)` is forced to +1.** `eigh` returns an orthonormal basis, not
   necessarily a right-handed one, and a left-handed one is not a rotation.
3. **Quaternion conversion is branchless Shepperd.** The naive trace-only
   formula loses sign and precision as the trace approaches -1, which is common
   here since many surface normals point almost straight down -Z in world space.

There is a uniform-scale fast path (`M == diag(s,s,s)`): scales shift by
`log(s)` and quaternions are untouched. It exists because that case is common
(matching intrinsics, or `reconcile_intrinsics: false`) and re-decomposing
20k+ rotations for nothing is wasteful.

**SH degree 0 is asserted, not assumed.** A general linear map reorients every
Gaussian, which would also require rotating the SH bands. The upstream pipeline
only ever emits degree 0 — one view constrains nothing view-dependent — so this
raises rather than silently producing wrong view-dependent colour.

### The splat layer is passed in pre-rendered, not named in `modes`
`Renderer` is pyrender-backed and cannot rasterize Gaussians; `SplatRenderer` is
gsplat-backed and knows nothing about meshes. Rather than couple them, the
pipeline renders the splat and hands `render_composite()` the finished RGBA via
`splat_layer=`. `render_splat_layer()` returns `None` on a culled frame, and
`_composite_splat()` treats `None` as a no-op, so the cull needs no branch at
any call site.

### The splat layer must be rendered with straight alpha
`SplatRenderer.render()` normally composites RGB over `bg_color` and returns
alpha alongside. Blending *that* over another layer blends toward the background
twice, which shows as a halo around the silhouette. `bg_color=None` returns
un-premultiplied colour instead, and the overlay path always uses it.

### The splat contributes to alpha; the skeleton does not
`_composite_splat()` does `alpha = max(base_alpha, splat_alpha)`, matching how
the face overlay unions alpha into the skeleton render. The splat is real
subject coverage, exactly as the mesh silhouette is, so a mask derived from the
composite has to include it. The skeleton stays out of alpha because it is an
annotation, not geometry.

### `attach_splat_overlay()` refuses an auto-oriented scene
`auto_orient()` calls `scene.rotate_around_y()`, which rotates about the bbox
centre and therefore moves the subject out of the original camera's frame. The
anchor is defined relative to that camera at the origin, so the two are
incompatible. `OrbitPipeline._auto_oriented` records that it ran; the check is
there rather than in a docstring because the failure would otherwise be a
silently misplaced face.

## Next Steps

See parent CLAUDE.md for overall project status and potential future enhancements.
