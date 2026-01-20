# Quick Start Guide

Get started with body2colmap in 5 minutes.

## Installation

```bash
pip install -r requirements.txt
# or
pip install numpy opencv-python pyrender trimesh scipy pyyaml
```

## Basic Usage

### 1. Generate Default Config

```bash
python -m body2colmap.cli --save-config my_config.yaml
```

This creates a template with all options documented.

### 2. Edit Config for Your Needs

```yaml
# my_config.yaml
input_file: "~/data/sam3d_output.npz"

render:
  resolution: [1024, 1024]  # Square aspect ratio
  modes: ["mesh", "depth", "mesh+skeleton"]

path:
  pattern: "helical"
  n_frames: 120
  helical_loops: 3
  helical_amplitude_deg: 30.0

export:
  output_dir: "./output"
  colmap: true
```

### 3. Run

```bash
python -m body2colmap.cli --config my_config.yaml ~/data/sam3d_output.npz
```

That's it! You'll get:
- `output/frame_*.png` - Rendered images
- `output/cameras.txt` - Camera intrinsics
- `output/images.txt` - Camera poses
- `output/points3D.txt` - Initial point cloud

## Common Use Cases

### Portrait Videos (TikTok, Instagram Stories)

```yaml
render:
  resolution: [720, 1280]  # 9:16 aspect ratio

camera:
  auto_frame: true
  fill_ratio: 0.85  # Fill most of frame
```

### Landscape Videos (YouTube, Standard)

```yaml
render:
  resolution: [1920, 1080]  # 16:9 aspect ratio

camera:
  auto_frame: true
  fill_ratio: 0.8
```

### High-Quality 3DGS Training

```yaml
render:
  resolution: [2048, 2048]  # High resolution
  modes: ["mesh"]  # Just mesh, no skeleton

path:
  pattern: "helical"
  n_frames: 240  # More views = better reconstruction
  helical_loops: 5
  helical_amplitude_deg: 45.0  # See from more angles

export:
  pointcloud_samples: 100000  # Dense point cloud
```

### Quick Preview (Fast Iteration)

```yaml
render:
  resolution: [512, 512]  # Lower res = faster

path:
  n_frames: 30  # Fewer frames

export:
  colmap: false  # Skip COLMAP if not needed
```

### Debug Skeleton Pose

```yaml
render:
  resolution: [1024, 1024]
  modes: ["skeleton", "mesh+skeleton"]
  bg_color: [0.2, 0.2, 0.2]  # Dark background for visibility

skeleton:
  joint_radius: 0.025  # Larger for visibility
  bone_radius: 0.015
```

## Command-Line Overrides

You can override any config setting from command line:

```bash
# Override resolution
python -m body2colmap.cli --config my.yaml \
  --width 1920 --height 1080 \
  input.npz

# Override frame count
python -m body2colmap.cli --config my.yaml \
  --n-frames 200 \
  input.npz

# Override output directory
python -m body2colmap.cli --config my.yaml \
  --output-dir ~/results/experiment1 \
  input.npz

# Combine multiple overrides
python -m body2colmap.cli --config my.yaml \
  --width 2048 \
  --height 2048 \
  --n-frames 240 \
  --output-dir ~/hq_results \
  input.npz
```

## Programmatic Usage (Python API)

If you want more control:

```python
from body2colmap import Scene, OrbitPipeline

# 1. Load scene
scene = Scene.from_npz_file("estimate.npz", include_skeleton=True)

# 2. Create pipeline
pipeline = OrbitPipeline(
    scene=scene,
    render_size=(1024, 1024),
    focal_length=None  # Auto-computed
)

# 3. Set orbit
pipeline.set_orbit_params(
    pattern="helical",
    n_frames=120,
    radius=None,  # Auto-framed
    fill_ratio=0.8,
    helical_loops=3,
    helical_amplitude_deg=30.0
)

# 4. Render
mesh_images = pipeline.render_all(modes=["mesh"])["mesh"]
composite_images = pipeline.render_composite_all({
    "mesh": {"color": (0.65, 0.74, 0.86), "bg_color": (1, 1, 1)},
    "skeleton": {"use_openpose_colors": True}
})

# 5. Export
pipeline.export_images("./output", mesh_images, filename_pattern="mesh_{:04d}.png")
pipeline.export_images("./output", composite_images, filename_pattern="composite_{:04d}.png")
pipeline.export_colmap("./output", filename_pattern="mesh_{:04d}.png")
```

## Output Structure

After running, your output directory will look like:

```
output/
├── cameras.txt           # Camera intrinsics (COLMAP format)
├── images.txt            # Camera poses (COLMAP format)
├── points3D.txt          # Initial point cloud (COLMAP format)
├── frame_0001.png        # Rendered images
├── frame_0002.png
├── ...
└── frame_0120.png
```

## Next Steps

### Use with 3D Gaussian Splatting

```bash
# 1. Run body2colmap
python -m body2colmap.cli --config my.yaml input.npz

# 2. Extract features with COLMAP (optional - most 3DGS impls skip this)
colmap feature_extractor --database_path output/database.db --image_path output/

# 3. Train 3DGS
cd gaussian-splatting
python train.py -s /path/to/output --eval
```

### Customize Rendering

See example configs in `examples/` directory:
- `examples/portrait.yaml` - Portrait orientation
- `examples/landscape.yaml` - Landscape orientation
- `examples/high_quality.yaml` - High-res for final output
- `examples/debug.yaml` - Debug skeleton and framing

### Troubleshooting

If something doesn't look right, see `TROUBLESHOOTING.md` for common issues:
- Figure too small/large
- Camera pointing wrong direction
- Skeleton bones wrong
- COLMAP import fails

## Tips

1. **Start with default config**: Generate with `--save-config`, then modify
2. **Test with low resolution first**: 512x512, 30 frames for quick iteration
3. **Use auto-framing**: Set `auto_frame: true` and adjust `fill_ratio` as needed
4. **Portrait needs special attention**: Use `max(X,Z)` for scene width (already implemented)
5. **Composite modes are powerful**: "mesh+skeleton" shows both simultaneously
6. **CLI overrides are your friend**: Keep base config, override for experiments

## Examples

### Minimal Working Example

```bash
# 1. Create minimal config
cat > minimal.yaml << EOF
render:
  resolution: [1024, 1024]
path:
  pattern: "circular"
  n_frames: 60
export:
  output_dir: "./output"
  colmap: true
EOF

# 2. Run
python -m body2colmap.cli --config minimal.yaml ~/sam3d_output.npz

# 3. Check output
ls -lh output/
```

### Full-Featured Example

```bash
# 1. Generate template and edit
python -m body2colmap.cli --save-config full.yaml
# Edit full.yaml with your preferences

# 2. Run with verbose output
python -m body2colmap.cli --config full.yaml --verbose ~/sam3d_output.npz

# 3. View results
eog output/frame_*.png  # Image viewer
# Or import into COLMAP GUI
```

## Documentation

- `CLAUDE.md` - Architecture and design decisions
- `body2colmap/CLAUDE.md` - Implementation details and gotchas
- `CHANGELOG.md` - Version history
- `TROUBLESHOOTING.md` - Common problems and solutions
- `README.md` - Full documentation (if exists)

## Get Help

```bash
# Show all CLI options
python -m body2colmap.cli --help

# Generate example config with all options
python -m body2colmap.cli --save-config template.yaml
```
