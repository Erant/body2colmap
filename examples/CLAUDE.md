# Examples

This directory contains example usage of body2colmap.

## Basic Usage

### Command Line

```bash
# Basic usage with default settings
body2colmap --input estimation.npz --output-dir ./output

# Custom orbit pattern
body2colmap \
  --input estimation.npz \
  --output-dir ./output \
  --orbit-mode helical \
  --n-frames 120 \
  --helical-loops 3

# Adjust rendering
body2colmap \
  --input estimation.npz \
  --output-dir ./output \
  --width 1024 \
  --height 1024 \
  --mode mesh
```

### Python API

```python
from body2colmap import OrbitPipeline

# Load and render
pipeline = OrbitPipeline.from_npz_file("estimation.npz")

# Configure orbit
pipeline.set_orbit_params(
    pattern="helical",
    n_frames=120,
    n_loops=3,
    amplitude_deg=30.0
)

# Render
images = pipeline.render_all(modes=["mesh", "depth"])

# Export
pipeline.export_colmap("./output")
pipeline.export_images("./output", images["mesh"])
```

## Advanced Examples

### Custom Camera Setup

```python
from body2colmap import OrbitPipeline, Camera

pipeline = OrbitPipeline.from_npz_file(
    "estimation.npz",
    render_size=(1024, 1024),
    focal_length=1200.0  # Custom focal length
)
```

### Multiple Render Modes

```python
# Render both mesh and depth
images = pipeline.render_all(
    modes=["mesh", "depth"],
    mesh_color=(0.8, 0.8, 0.8),
    bg_color=(0.0, 0.0, 0.0),
    normalize_depth=True
)

# Export both
pipeline.export_images("./output/mesh", images["mesh"])
pipeline.export_images("./output/depth", images["depth"])
```

### Custom Orbit Path

```python
from body2colmap import Scene, Camera, OrbitPath

# Load scene
scene = Scene.from_npz_file("estimation.npz")

# Create custom orbit
target = scene.get_centroid()
orbit = OrbitPath(target=target, radius=3.0)

# Generate cameras
cameras = orbit.helical(
    n_frames=200,
    n_loops=5,
    amplitude_deg=45.0
)

# Manually render and export
# ...
```

## Example Data

TODO: Add sample SAM-3D-Body output for testing

## Integration with 3DGS

After running body2colmap, the output can be used with 3D Gaussian Splatting:

```bash
# Run body2colmap
body2colmap --input estimation.npz --output-dir ./data/person

# Train 3DGS (example with gaussian-splatting)
python train.py -s ./data/person

# Output directory structure expected by 3DGS:
# data/person/
# ├── cameras.txt
# ├── images.txt
# ├── points3D.txt
# └── frame_0001.png
# └── frame_0002.png
# └── ...
```

## Tips

### Choosing Orbit Parameters

**For best 3DGS results:**
- Use `helical` pattern with 3-5 loops
- 100-200 frames total
- Amplitude 30-45 degrees
- This provides good coverage from multiple angles

**For turntable visualization:**
- Use `circular` pattern
- 36-72 frames (10° or 5° increments)
- Elevation 0-15 degrees

**For dynamic views:**
- Use `sinusoidal` pattern
- 2-3 cycles
- Creates nice up-down motion

### Adjusting Framing

If the subject is too small or too large in frame:

```python
# Auto-framing (default)
pipeline.set_orbit_params(
    pattern="helical",
    n_frames=120,
    fill_ratio=0.8  # 80% of viewport
)

# Manual radius
pipeline.set_orbit_params(
    pattern="helical",
    n_frames=120,
    radius=3.5  # Fixed distance
)
```

### Performance

Rendering can be slow for many frames. Consider:
- Start with fewer frames (36) for testing
- Reduce resolution (256x256) for quick iterations
- Use final settings (512x512, 120 frames) only when ready
