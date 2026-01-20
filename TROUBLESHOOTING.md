# Troubleshooting Guide

Common issues and solutions for body2colmap.

## Installation Issues

### PyRender / OpenGL Errors

**Problem**: `RuntimeError: Unable to initialize OpenGL`

**Solutions**:
```bash
# Option 1: Use EGL backend
export PYOPENGL_PLATFORM=egl

# Option 2: Use OSMesa (software rendering)
pip install pyrender[osmesa]

# Option 3: Check X11 display (if using SSH)
export DISPLAY=:0
```

### Missing Dependencies

**Problem**: `ModuleNotFoundError: No module named 'trimesh'`

**Solution**:
```bash
pip install trimesh opencv-python pyrender numpy scipy pyyaml
```

## Rendering Issues

### Figure Appears Too Small

**Problem**: Figure only fills small portion of frame

**Possible Causes**:

1. **fill_ratio too low**
   ```yaml
   camera:
     fill_ratio: 0.8  # Try increasing to 0.9 or 0.95
   ```

2. **Manual radius too large**
   ```yaml
   path:
     radius: null  # Set to null for auto-computation
   ```

3. **Scene bounds incorrect**
   - Check: `scene.get_bounds()` in Python
   - Verify mesh loaded correctly from .npz file

### Figure Appears Too Large / Clipped

**Problem**: Figure extends beyond frame edges

**Solutions**:

1. **Lower fill_ratio**
   ```yaml
   camera:
     fill_ratio: 0.7  # Default is 0.8
   ```

2. **Check aspect ratio**
   - Ensure resolution matches expected dimensions
   - Portrait: width < height (e.g., 720x1280)
   - Landscape: width > height (e.g., 1280x720)

### Camera Pointing Wrong Direction

**Problem**: Camera looking at chest, feet, or off to the side

**Solutions**:

1. **Verify using bbox center** (not centroid)
   - Code uses `scene.get_bbox_center()` in pipeline.py:176
   - If you modified this, revert to bbox center

2. **Check elevation angle** (for circular path)
   ```yaml
   path:
     pattern: "circular"
     elevation_deg: 0.0  # 0 = eye level, >0 = looking down
   ```

3. **Check scene bounds**
   ```python
   from body2colmap.scene import Scene
   scene = Scene.from_npz_file("estimate.npz")
   print("Bounds:", scene.get_bounds())
   print("Bbox center:", scene.get_bbox_center())
   print("Centroid:", scene.get_centroid())
   ```

### Mesh Appears Upside Down or Backwards

**Problem**: Figure inverted or facing wrong way

**This should NOT happen** - it means coordinate conversion is broken.

**Debug**:
1. Check `sam3d_to_world()` in coordinates.py
2. Verify Z negation: SAM-3D-Body uses Z-in, we use Z-out
3. Ensure no hidden rotations in renderer.py

**Expected**: Figure upright, facing camera at azimuth=0°

## Skeleton Issues

### Bones Connecting Wrong Joints

**Problem**: Lines between random joints (e.g., wrist to knee)

**Solution**: Verify using official MHR70 bone connectivity

```python
from body2colmap import skeleton
print(f"Number of bones: {len(skeleton.MHR70_BODY_BONES)}")  # Should be 65
```

If incorrect, check skeleton.py has official definitions from:
https://github.com/facebookresearch/sam-3d-body/blob/main/sam_3d_body/metadata/mhr70.py

### Wrong Skeleton Colors

**Problem**: Random colors, duplicate colors, or incorrect rainbow gradient

**Solution**:
1. Verify `OPENPOSE_BODY25_COLORS` has 25 colors
2. Check color index 8 is NOT red (should be cyan-green)
3. Ensure `use_openpose_colors: true` in config

```yaml
skeleton:
  enabled: true
  format: "openpose_body25_hands"
  # Colors are hardcoded in skeleton.py
```

### Skeleton Not Visible

**Problem**: Skeleton mode renders but shows only background

**Possible Causes**:

1. **Skeleton not loaded**
   ```python
   scene = Scene.from_npz_file("estimate.npz", include_skeleton=True)
   ```

2. **Joint/bone radius too small**
   ```yaml
   skeleton:
     joint_radius: 0.02  # Increase if not visible
     bone_radius: 0.012
   ```

3. **Skeleton joints out of frame**
   - Check bounds: `scene.skeleton_joints.min()`, `scene.skeleton_joints.max()`

## Export Issues

### COLMAP Import Fails

**Problem**: `colmap feature_extractor` or `colmap mapper` fails to load data

**Possible Causes**:

1. **Filename mismatch**
   - Check images.txt lists same filenames as actual image files
   - Both should use same `filename_pattern`

2. **Wrong directory structure**
   ```
   output/
   ├── cameras.txt
   ├── images.txt
   ├── points3D.txt
   └── *.png  (image files)
   ```

3. **Invalid quaternions in images.txt**
   - Quaternions must be normalized: w² + x² + y² + z² = 1
   - Check for NaN or Inf values

4. **Image files don't match dimensions**
   - cameras.txt WIDTH HEIGHT must match actual image size
   - Check image dimensions: `cv2.imread("frame_0001.png").shape`

### Images.txt Has Wrong Filenames

**Problem**: images.txt references frame_0001.png but files are named IMG_0001.png

**Solution**: Ensure same pattern used for both exports

```python
pattern = "IMG_{:04d}.png"
pipeline.export_images(output_dir, images, filename_pattern=pattern)
pipeline.export_colmap(output_dir, filename_pattern=pattern)  # Must match!
```

Or in YAML:
```yaml
export:
  filename_pattern: "IMG_{:04d}.png"
```

### Point Cloud Empty or Wrong

**Problem**: points3D.txt has no points or points in wrong location

**Solutions**:

1. **Increase sample count**
   ```yaml
   export:
     pointcloud_samples: 100000  # Default is 50000
   ```

2. **Check mesh has surface area**
   ```python
   mesh = scene.get_trimesh()
   print(f"Faces: {len(mesh.faces)}, Area: {mesh.area}")
   ```

3. **Verify coordinate system**
   - Points should be in same world coords as cameras
   - Check first few lines of points3D.txt are near mesh bounds

## Configuration Issues

### Config File Values Ignored

**Problem**: Set n_frames: 81 in config but get 120 frames

**Cause**: CLI argument overriding config file

**Solution**: Don't use CLI argument if you want config file value

```bash
# ❌ Wrong - CLI arg overrides config
python -m body2colmap.cli --config my.yaml --n-frames 120 input.npz

# ✅ Right - Let config file value be used
python -m body2colmap.cli --config my.yaml input.npz
```

### Resolution Not Applied

**Problem**: Set resolution in config but output is different size

**Possible Causes**:

1. **CLI override**
   ```bash
   --resolution 1024x1024  # This overrides config file
   ```

2. **Wrong format in config**
   ```yaml
   # ❌ Wrong
   render:
     resolution: "1024x1024"  # String, not list

   # ✅ Right
   render:
     resolution: [1024, 1024]  # List [width, height]
   ```

### Composite Mode Not Working

**Problem**: mesh+skeleton mode fails or renders only one layer

**Debug**:
```yaml
render:
  modes: ["mesh+skeleton"]  # Note: string with +, not list
```

Supported composite modes:
- `mesh+skeleton`
- `depth+skeleton`

NOT supported:
- `skeleton+mesh` (order matters - base first)
- `mesh+depth` (no alpha channel to blend)

## Performance Issues

### Rendering Very Slow

**Typical speeds**:
- Mesh: ~0.5 sec/frame (GPU), ~2 sec/frame (CPU)
- Depth: ~0.3 sec/frame
- Skeleton: ~1 sec/frame
- Composite: ~1.5 sec/frame

**If slower**:

1. **Check PyRender backend**
   ```python
   import pyrender
   print(pyrender.constants.RenderFlags)
   ```

2. **Reduce resolution**
   ```yaml
   render:
     resolution: [512, 512]  # Lower res = faster
   ```

3. **Reduce frame count**
   ```yaml
   path:
     n_frames: 60  # Fewer frames = faster
   ```

### Memory Error

**Problem**: `MemoryError` or system freezes during rendering

**Solutions**:

1. **Reduce pointcloud samples**
   ```yaml
   export:
     pointcloud_samples: 10000  # Down from 50000
   ```

2. **Reduce resolution**
3. **Render fewer frames at a time** (split into batches)
4. **Close other applications**

## Debugging Tips

### Enable Verbose Output

```bash
python -m body2colmap.cli --config my.yaml --verbose input.npz
```

### Check Intermediate Values

```python
from body2colmap import Scene, OrbitPipeline

scene = Scene.from_npz_file("estimate.npz", include_skeleton=True)
print(f"Vertices: {scene.vertices.shape}")
print(f"Faces: {scene.faces.shape}")
print(f"Bounds: {scene.get_bounds()}")
print(f"Bbox center: {scene.get_bbox_center()}")
print(f"Skeleton joints: {scene.skeleton_joints.shape if scene.skeleton_joints is not None else None}")

pipeline = OrbitPipeline(scene, render_size=(1024, 1024))
pipeline.set_orbit_params(pattern="circular", n_frames=10, radius=None)
print(f"Cameras: {len(pipeline.cameras)}")
print(f"First camera position: {pipeline.cameras[0].position}")
```

### Visualize Single Frame

Test rendering without full pipeline:

```python
from body2colmap import Scene, Renderer, Camera

scene = Scene.from_npz_file("estimate.npz")
renderer = Renderer(scene, render_size=(512, 512))

# Create camera looking at scene
camera = Camera(
    focal_length=(500, 500),
    image_size=(512, 512)
)
camera.look_at(
    eye=scene.get_bbox_center() + [0, 0, 2],  # 2 units in front
    target=scene.get_bbox_center(),
    up=[0, 1, 0]
)

# Render
image = renderer.render_mesh(camera)

# Save
import cv2
cv2.imwrite("test.png", cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2BGR))
```

## Still Having Issues?

1. **Check existing issues**: https://github.com/anthropics/claude-code/issues
2. **Provide details**:
   - Full error message and stack trace
   - Config file (YAML)
   - Command used
   - System info (OS, Python version, GPU)
   - Input file characteristics (mesh size, skeleton present?)
3. **Include debug output**:
   ```python
   print(f"body2colmap version: {body2colmap.__version__}")
   print(f"numpy: {np.__version__}")
   print(f"cv2: {cv2.__version__}")
   ```
