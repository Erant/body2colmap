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
splat_renderer.py  (depends on: camera, splat_scene + the
                    external brush-splat-render binary)
    ↓
splat_anchor.py  (depends on: camera, splat_scene)
    ↓
fade.py  (depends on: camera)
    ↓
background.py  (depends on: camera, coordinates, fade)
    ↓
renderer.py  (depends on: scene, camera, face, skeleton, background)
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
  - `composite_over_background()`: draws the optional `background` behind a
    base layer. A no-op when none is set, so callers need no branch. Applied
    to the BASE only, before overlays — see below.
  - `_render_depth_buffer()`: The raw pyrender depth buffer. Single source of
    truth for both `render_depth()` and `render_mask()` — and for the splat
    overlay's depth-gauge fit. Do NOT re-open a pyrender scene to read depth.
  - Alpha channel handling
- Module-level `parse_composite_modes()`: the ONE place mode strings are split
  and validated, with `BASE_LAYERS` / `OVERLAY_LAYERS` naming the vocabulary.
  Used by `cli.py` and `pipeline.render_original_view()`; both used to parse
  independently, so a new layer had to be added to each by hand.

**Testing priority**: MEDIUM - mainly wraps pyrender

#### background.py
**Purpose**: Draw a world-fixed environment behind the base layer, so an orbit
reads as the camera moving rather than the subject spinning

**Key components**:
- `Background` class:
  - `Background.create()`: the single entry point, dispatching one `texture`
    field between a built-in generator and a path
  - `render(camera)`: the environment as RGB, by per-pixel direction lookup
  - `composite(image, camera)`: alpha-blend an RGBA base layer over it
  - `_surface_vectors()`: ray → lookup vector. The whole infinite/finite
    distinction lives here and nowhere else
- Parameterizations: `equirect_uv()` / `equirect_directions()` and
  `cube_face_uv()` / `cube_face_directions()`, each pair mutual inverses
- Generators in `TEXTURE_GENERATORS`: `grid`, `checker`, `gradient`,
  `blender_sky`. Each takes `flat=True` for its pattern-free form, which is
  what the subject fade dissolves the pattern into
- `_sampling_maps()` / `_sample()`: texture lookup split from map construction,
  so the textured and plain backdrops are sampled identically
- `DEFAULT_RADIUS_SCALE`: the default backdrop radius, as a multiple of the
  orbit radius. Lives here rather than in `config.py` because `config.py` and
  `pipeline.py` both default to it and must not drift
- Loaders: `load_texture()` and `equirect_to_cube()`

**Testing priority**: HIGH - every convention here is silently plausible when
wrong (see the marker test)

#### fade.py
**Purpose**: Fade the backdrop out around the subject, so an `outline` frame
does not present the silhouette as a hard occlusion boundary

**Key components**:
- `SubjectFade.target`: `"plain"` (default) reveals the pattern-free backdrop,
  `"color"` a flat colour, `"blur"` an average of the backdrop into itself.
  Only `"plain"` actually removes a line; see below
- `Ellipsoid`: stored as the linear map to the unit sphere, not as
  (axes, rotation). That is what makes the per-pixel test two lines
  - `Ellipsoid.fit()`: minimum-volume enclosing ellipsoid (Khachiyan), with
    enclosure imposed exactly afterwards against the *full* point set
  - `ray_distance()`: the one scalar the whole feature is built on
- `SubjectFade`: profile + falloff + target, applied to a finished backdrop
- `DECAY_PROFILES`: the seven decay shapes, all `w(0) = 1` and monotone

**Testing priority**: HIGH - a fade that merely looks right on one frame can
still let the grid run into the silhouette halfway round the orbit

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
- `target`: The orbit's look-at point, in world coords. Used by
  `configure_background()` to centre a finite backdrop on the subject rather
  than on the world origin — which for SAM-3D-Body output is the original
  camera, and can be metres away.
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

**Gaussian splats** (optional, `pip install body2colmap[splat]`):
- `plyfile`: Reading and writing 3DGS `.ply` files
- `brush-splat-render`: **an external binary, not a Python package.** Build it
  from the brush repo with `cargo build --release -p brush-splat-render`, then
  point `$BRUSH_SPLAT_RENDER` at it (or pass `--splat-renderer`). It is
  wgpu/Vulkan, so it needs a GPU but no CUDA toolchain. There is deliberately
  no Python rasterizer fallback -- see the root `CLAUDE.md`.

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
`Renderer` is pyrender-backed and cannot rasterize Gaussians; `SplatRenderer`
shells out to `brush-splat-render` and knows nothing about meshes. Rather than
couple them, the pipeline renders the splat and hands `render_composite()` the
finished RGBA via `splat_layer=`. `render_splat_layer()` returns `None` on a
culled frame, and `_composite_splat()` treats `None` as a no-op, so the cull
needs no branch at any call site.

The second reason is batching, and it is why the *plural*
`pipeline.render_splat_layers(cameras)` exists: the binary initializes wgpu and
loads the ply once per invocation, then loops the camera list. Rendering an
81-frame orbit one frame at a time would pay that setup 81 times, so
`render_composite_all()` renders every layer up front in a single invocation and
indexes into the result. Culled cameras are never sent, and their slots come
back `None`, so the list still lines up index-for-index with `cameras`.
`render_splat_layer()` (singular) survives for genuine one-off frames —
`render_original_view()` and the anchor-verification recipe.

### The splat layer must be rendered with straight alpha
`SplatRenderer.render*()` normally composites RGB over `bg_color` and returns
alpha alongside. Blending *that* over another layer blends toward the background
twice, which shows as a halo around the silhouette. `bg_color=None` returns
un-premultiplied colour instead, and the overlay path always uses it.

This is why the binary is always invoked with `--background 0,0,0`, whatever
`bg_color` says. brush composites as `rgb*alpha + bg*(1-alpha)`, so a black
background makes its RGB output *premultiplied* — the same intermediate gsplat
produced, and the one both conventions derive from in Python: `bg_color=None`
divides by alpha, a real `bg_color` adds `bg*(1-alpha)`. One Rust path, one
Python path. The 8-bit round trip costs at most 1/255 in the final composite,
because the quantized quantity *is* the premultiplied contribution.

### Success is decided by the output files, not the exit code
`brush-splat-render` intermittently dies from a signal (SIGSEGV) *after* having
written every frame it was asked for. The cause is not yet tracked down, and
treating a non-zero exit as failure would throw away complete, correct renders
whenever it fires.

So `render_many()` checks that every expected file exists and is non-empty, and
consults the exit status only to *explain* a shortfall. A crash that produced
everything is logged as a warning and the frames are used; a crash that lost
frames raises, naming what is missing and decoding the signal number (an
operator seeing "exit -11" should not have to look up SIGSEGV).

Truncation is covered by the read path rather than the file check: a partial
write is non-empty, so `_read_frame()` treats an image OpenCV cannot decode, or
whose dimensions are wrong, as a partial write and raises. Do not "simplify"
this back to `check=True` or `returncode != 0` -- and do not weaken the
per-file check to a directory-not-empty test, because the failure being guarded
against is a *partial* sequence.

### A crashed render is deliverable exactly once, through `on_fault`
Everything an invocation touches lives in a temp directory `render_many()`
deletes in a `finally` -- success, exception, either way. So by the time the
caller catches the exception the `cameras.json` naming the views, and the frames
that did land, are already gone. That is not hypothetical: a real
`brush-splat-render` crash on a rented GPU pod left nothing but an exit code,
and the pod did not outlive the investigation.

`on_fault` is called while the directory is still there, with a `RenderFault`
carrying the live paths (`run_dir`, `cameras_path`, `frames_dir`, `expected`,
`missing`, `written`) plus the argv, the decoded exit status and whatever output
was captured. A caller that wants any of it must copy it out **inside the hook,
synchronously** -- the class docstring says so, and the b2crunner crashlog is
the consumer this was built for.

Three properties the implementation has to keep:

- **It fires for a tolerated crash too.** A run that wrote everything and then
  died is still worth recording; `RenderFault.complete` is what tells the two
  apart. Do not move the call inside the `if missing:` branch.
- **It fires at most once**, which is why the tolerated case calls it
  explicitly and every raising path goes through the `except`. The lost-file
  branch does *not* call it directly -- its `raise` carries it there.
- **A hook that throws is logged and swallowed.** A broken crash reporter
  replacing the render's own error is the worst possible outcome: the operator
  then spends the afternoon on the reporter.

`tests/test_splat_renderer.py` drives all of this against a stub binary, since
the cases worth testing are crashes and the real binary crashes when it feels
like it.

### Two smaller seams, for the same reason
Both exist so a downstream caller does not have to reimplement the whole
invocation to get one behaviour -- which is exactly what b2crunner had done,
carrying a parallel copy of this method for the crash report alone.

`on_output` is handed each line as it arrives. `render_many()` therefore drives
the binary with `Popen` and a line loop instead of `subprocess.run`: on an
81-frame render, output that only arrives at the end is a blank log for the
duration. Both streams are merged, since their interleaving is what makes a
crash readable, and the output is captured in every case -- `verbose` and
`on_output` decide who sees it live, not whether it is kept.

`ply_path` renders a file the caller already has rather than serializing
`scene` into the temp directory. A trained splat is hundreds of megabytes and
the caller usually loaded the scene *from* that file. It is deliberately not
validated against `scene`: checking would mean reading the file back, which is
the cost being avoided. `close()` does not delete a file it was handed.

### Confidence gating is base-render only
`brush-splat-render --confidence` scores each Gaussian by how well the training
views constrained it. An overlay splat is masktest's 2.5-D shell reconstructed
from one photograph: no training views, no `ev_*` block, so the binary would
warn and silently degenerate the gate to plain alpha. It also composites over
`cull_color` and writes the gate as alpha, leaving nothing to un-premultiply by.
`attach_splat_overlay()` therefore raises when confidence is configured, the
same way it raises after `auto_orient()` — a silent degradation here would look
exactly like a working feature.

**It changes what alpha means.** Without it, alpha is accumulated opacity. With
it, alpha is the gate. Anything downstream using that alpha as a 3DGS training
mask is then masking on evidence rather than coverage, which is the point.

**It also moves compositing into the binary.** The gated path is the one case
where `--background 0,0,0` does not apply: `--cull-color` is what the binary
composites over *and* what culled pixels resolve to, so it is the entire
background of the frame. `ConfidenceOptions.cull_color` therefore defaults to
`None`, meaning "use the render's `bg_color`" — resolved in `to_args()`, where
both values are in scope. Without that, `bg_color` would be a silently ignored
argument whenever gating was on, and `--bg-color` would mean two different
things depending on a flag elsewhere. Set `cull_color` explicitly only to make
culled regions visible against the background.

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

## Environment Backdrop (2026-08)

`background.py` plus a hook in `render_composite()`. Draws the inside of a
sphere or an axis-aligned cube behind the base layer so an orbit reads as the
camera moving, not the subject spinning. See the parent CLAUDE.md for the
overview; what follows is the detail that is easy to get wrong.

### Why it is not scene geometry
A surrounding sphere added to the pyrender scene would put geometry behind
every pixel. `render_mask()` derives mesh coverage from `depth > 0` — that is
its whole reason for existing, in preference to the colour buffer's alpha — so
the silhouette would become the full frame, and `outline` mode along with the
alpha channel would collapse.

Even setting that aside, geometry costs more than it buys here: pyrender's
`IntrinsicsCamera` defaults to `zfar=100` and would clip a large sphere, an
unlit textured material has to fight the shader gamma documented at
`_flat_color_rgba8()`, and a tessellated sphere pinches at the poles. The
remap has none of those problems and is about forty lines.

### The three lookup conventions, and how they are pinned
Three separate conventions have to agree, and all three are silently plausible
when wrong — a mirrored sky still looks like a sky:

1. **Ray generation** inverts `Camera.project()`, which converts OpenGL camera
   space to OpenCV by `* [1, -1, -1]` before applying `K`. So a pixel's OpenCV
   ray maps back by negating Y and Z. Get this wrong and the render is
   vertically mirrored.
2. **Equirect** uses the project's spherical convention (azimuth from +Z toward
   +X), with `v` running zenith to nadir. Get this wrong and the sky is upside
   down or a quarter-turn off.
3. **Cubemap** follows the OpenGL face/UV table, so standard assets load
   without surprises.

`TestMarkerLandsWhereProjectionSaysItShould` is what actually holds them: a
texture with one bright texel, rendered from an off-axis camera, with the
resulting centroid compared against `camera.project(position + direction)`.
That reference comes from `Camera` itself, so it is independent of everything
inside `background.py`. Round-trip tests alone would not catch a convention
that is self-consistently wrong.

### `_ray_grid_cam` caches camera space, not world space
Every camera on an orbit shares one set of intrinsics (the project assumes
this throughout), so the camera-space ray grid is built once and `_world_rays()`
applies only the per-frame rotation. Caching *world* directions instead would
be a per-frame bug that looks like a frozen backdrop — which is exactly what a
missing rotation looks like, and how it was caught during development.

### Rotation is applied to the ray, not to the lookup
`rotation_deg` rotates both the ray direction and the camera's offset from
centre, before intersection. Rotating the texture coordinate after intersection
would be cheaper and wrong for a cube: the texture would slide across walls
that stayed put. `test_a_cube_rotates_with_its_texture` pins it by exploiting a
quarter turn being a symmetry of the cube itself.

### Padding, because `borderMode` cannot express what is needed
An equirect wraps horizontally but not vertically: at the seam the correct
neighbour is the opposite column, at the poles it is the same row. `cv2.remap`
applies one `borderMode` to both axes, so `_pad_equirect()` builds the two
borders by hand and the sample coordinates are offset by one (`u * W + 0.5`,
not `- 0.5`). Cube faces get plain edge replication — true cubemap edge
filtering would pull the neighbouring face's border row, which is off by at
most half a texel and needs an adjacency table nothing else would use.

### The defaulted radius needs one rule, in three places
The defaults are a `grid` cube at `DEFAULT_RADIUS_SCALE` times the orbit
radius, and they are a set: a cube at infinity has no corners and no parallax,
so defaulting the texture and geometry without the radius would ship the weak
version of the feature. See the parent CLAUDE.md for the comparison that
settled it.

That makes `radius_scale` a defaulted half of a mutually exclusive pair, so
every entry point has to answer "the caller set only `radius`" the same way —
**the explicit radius supersedes the default**, and naming both is still an
error:

- `Config.from_yaml()` keys off *presence* in the mapping, not the value, so a
  file writing `radius_scale: null` still gets infinity.
- `Config.apply_cli_overrides()` already worked this way for a config-file
  scale; `--background-infinite` is the CLI's way to say null.
- `configure_background()` needs the `_UNSET` sentinel for it, because an
  explicit `radius_scale=None` (asking for infinity) and an unmentioned
  argument are different requests and `None` cannot express both.

`BackgroundConfig.validate()` checks each radius value before checking the two
against each other, so `BackgroundConfig(radius=0.0)` complains about the zero
rather than about a conflict with a default the caller never set.

### `opaque` is a real choice, not a cosmetic one
With `opaque=True` (the default) alpha becomes 255 everywhere and the frame is
a flat conditioning image. That destroys the silhouette mask — deliberately,
because these frames feed a video model that flattens alpha anyway. Anything
that needs the mask sets `opaque=False`, which fills only RGB behind the
subject and leaves alpha exactly as the base layer produced it.

### Composite order: under the base, before the overlays
`render_composite()` applies the backdrop the moment the base layer exists, in
both branches (the `mesh`/`depth`/`outline` chain and the skeleton-as-base
early return). It cannot go later: the skeleton overlay blends into RGB
*without* touching alpha, so a backdrop composited afterwards would use the
mesh silhouette's alpha and blend the skeleton away everywhere outside it.

For single-mode renders there is no overlay to order against, so
`pipeline.render_all()` calls `composite_over_background()` itself.
`render_original_view()` does the same, but only for non-composite modes —
`render_composite()` has already handled the rest.

### Resolution fitting exists to stop temporal flicker
Not for speed. A 4K panorama minified into a small frame point-samples
differently every frame, and the shimmer reads as motion to a video model. One
`INTER_AREA` resize down to roughly twice the render's angular resolution
removes it. It runs once, lazily, on the first render — which is why
`_padded` is invalidated there and `_fitted` guards the second call.

### Splat scenes are excluded at the CLI, not silently
A `.ply` input is rendered by `brush-splat-render`, which composites against a
flat colour of its own; there is no pyrender base layer to draw behind.
`cli.py` raises rather than ignoring `--background`, and `pipeline.renderer`
only assigns the backdrop for a mesh scene.


## Backdrop Fade (2026-09)

### The failure it fixes is downstream of the one the backdrop fixed
The backdrop exists so a video model reads an orbit as camera motion. But in
`outline` modes it also draws a grid right up to the silhouette, and VACE reads
that as a hard occlusion boundary: it will not paint outside the outline, so
bulky clothing and hair come out squashed onto the shape of the bare mesh. The
fade clears a shell around the subject, keeping the rotation cue in the far
field and leaving structure-free room next to the silhouette.

So this is not a cosmetic softening. Both settings are load-bearing in
opposite directions, which is why `falloff` is the knob most worth sweeping.

### The shell is a fitted ellipsoid, not a dilated outline
The obvious implementation — dilate the silhouette mask per frame — fails three
ways at once, and `Ellipsoid` avoids all three:

1. **It would swim.** A screen-space dilation of a per-frame mask is a
   different world-space region every frame. A fitted ellipsoid is one fixed
   object the camera moves around, so the clear zone is temporally consistent
   for free — which is the whole point for a video model.
2. **It has no depth.** The mask says nothing about how far the clear zone
   should extend *behind* the subject, so parallax against a finite cube would
   be wrong.
3. **It costs a distance transform per frame.** The ellipsoid test is a
   closest-approach computation in the space where the ellipsoid is the unit
   sphere — one `einsum` over the ray grid.

The enclosure property is what makes it safe: an ellipsoid containing the mesh
contains its silhouette from *every* viewpoint, so the clear zone can never
fall inside the outline on some frame. `TestClearZoneCoversTheSilhouette` pins
exactly that, over a full orbit.

### `fit()` guarantees enclosure independently of the solver
Khachiyan is iterative and is given a loose tolerance and an iteration cap, so
its answer is a good *shape*, not a proof. Enclosure is then imposed by
scaling the result until the outermost point of the **full** vertex set sits on
the surface.

That is what lets the fit run on a stride of the vertices (`max_points`,
default 4000) with no risk at all: the check is O(N) and runs on everything.
`test_encloses_every_point_even_when_subsampled` drives `max_points` down to 20
so the solver's answer is useless and only the guarantee is left.

### `falloff` is in subject radii, and that is the point
`ray_distance()` returns the closest approach *in units of the ellipsoid
radius in that direction*, so `u = (m - 1) / falloff` is dimensionless. One
`falloff` therefore works at any subject size and any orbit radius — which
matters because auto-framing means neither is known when the value is chosen.

A consequence worth knowing: the band is anisotropic in world units. It is as
wide as the subject is in each direction, so a standing figure gets a tall
clear zone and a narrow one. That is usually what you want; it is also why
`falloff = 1.0` is not as aggressive as it sounds (the clear zone is the
ellipsoid at 2x, and the subject only fills ~0.8 of frame).

### The lines fade to the wall, through a second texture
`target="plain"` samples the backdrop **twice through one set of maps**: once
normally, once from the same generator with `flat=True`. At full weight the
pixel *is* the plain render, so the line colour becomes the wall colour that
was behind it while the shading, the corners and the floor/ceiling split stay
sharp.

`_sampling_maps()` exists for this: computing the maps once and passing them to
`_sample()` twice is what guarantees the two renders line up to the texel.
`_fit_resolution()` and `_padded_texture()` carry the plain texture in lockstep
for the same reason — a plain texture resized on its own detail budget would
not index the same way. The test asserts bit-identity against a `flat=True`
backdrop rendered separately.

**`target="blur"` is the earlier mistake, kept honestly named.** The first
implementation area-averaged the render into itself and called it `"local"`,
on the reasoning that a box average below the texture's frequency removes the
lines. An average removes their *frequency*, not their brightness: the energy
spreads into a wide grey band, so the clear zone is a smear of the grid rather
than a grid-free wall. It stays only because a **loaded** texture has no plain
variant and this is the only thing available there.
`test_blur_target_smears_the_lines_instead_of_removing_them` pins the
difference so the two cannot quietly converge.

**`flat` is a generator kwarg, not a parameter override.** A table of
"parameters that suppress this generator's pattern" cannot express `grid`:
setting `line_color = base_color` still leaves lines on the floor and ceiling,
which draw over their own base colours. Only the generator knows what its
pattern is, so each one says so.

### The fade runs inside `Background.render()`, so it cannot touch the subject
`composite()` lays the base layer over an already-faded backdrop, so an opaque
subject pixel is untouched by construction — there is no ordering to get wrong
and no mask to intersect. The corollary is that `render()` now needs the
*unrotated* world rays as well as the surface vectors: the ellipsoid lives in
world space, `_surface_vectors()` rotates into the environment's frame. Hence
the local `dirs` rather than a nested call.

### `_texture_mean()` weights an equirect by solid angle
An unweighted mean over an equirectangular image is dominated by the poles,
which it oversamples enormously. Since the value is a *fallback background
colour*, getting it wrong shows up directly as a patch of the wrong tone.

## Next Steps

See parent CLAUDE.md for overall project status and potential future enhancements.
