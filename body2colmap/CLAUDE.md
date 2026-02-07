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
2. `exporter.py::ColmapExporter.export()` - World → COLMAP/OpenCV

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
  - `helical()`: Multiple loops with linear elevation change

**Returns**: List[Camera] with positions and orientations set

**Testing priority**: HIGH - path patterns must be correct for good 3DGS training

#### renderer.py
**Purpose**: Render images using pyrender

**Key components**:
- `Renderer` class:
  - `render_mesh()`: Color mesh with lighting
  - `render_depth()`: Depth buffer
  - `render_skeleton()`: Joints and bones
  - Alpha channel handling

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
- Mesh/depth = 1.0 where surface exists, 0.0 background
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

**Rendering**: White points (icospheres) + optional white cylinders for 63 OpenPose
face bone connections. Smaller geometry than body skeleton (~35% of body joint/bone radii).

**CLI**: `--face-mode full|points|none`, composite mode `skeleton+face`.

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

**Composite modes** (e.g., "mesh+skeleton", "depth+skeleton"):

1. Parse composite string: `"mesh+skeleton"` → base="mesh", overlays=["skeleton"]
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

## Next Steps

See parent CLAUDE.md for overall project status and potential future enhancements.
