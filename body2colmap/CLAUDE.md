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
path.py  (depends on: camera, coordinates)
    ↓
renderer.py  (depends on: scene, camera)
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

## Next Steps

See parent CLAUDE.md for overall project status and roadmap.
