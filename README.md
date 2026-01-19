# Body2COLMAP

Generate synthetic multi-view training data for 3D Gaussian Splatting from SAM-3D-Body output.

## Overview

Body2COLMAP is a command-line tool and Python library that converts single-image 3D body reconstructions into multi-view synthetic datasets suitable for 3D Gaussian Splatting training.

**Input**: 3D mesh from SAM-3D-Body (`.npz` file)
**Output**: Multi-view images + COLMAP camera parameters

### What it does

1. **Loads** 3D mesh reconstruction from SAM-3D-Body
2. **Generates** camera orbit paths (circular, sinusoidal, or helical)
3. **Renders** multi-view images from these cameras
4. **Exports** COLMAP format camera parameters and point cloud
5. **Outputs** data ready for 3D Gaussian Splatting training

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/body2colmap.git
cd body2colmap

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Requirements

- Python 3.8+
- numpy
- pyrender
- trimesh
- opencv-python
- scipy

## Quick Start

### Command Line

```bash
# Basic usage
body2colmap --input estimation.npz --output-dir ./output

# With custom settings
body2colmap \
  --input estimation.npz \
  --output-dir ./output \
  --orbit-mode helical \
  --n-frames 120 \
  --width 512 \
  --height 512
```

### Python API

```python
from body2colmap import OrbitPipeline

# Create pipeline
pipeline = OrbitPipeline.from_npz_file("estimation.npz")

# Configure orbit
pipeline.set_orbit_params(pattern="helical", n_frames=120)

# Render frames
images = pipeline.render_all(modes=["mesh"])

# Export
pipeline.export_colmap("./output")
pipeline.export_images("./output", images["mesh"])
```

## Features

### Orbit Patterns

- **Circular**: Fixed elevation, rotating azimuth (turntable)
- **Sinusoidal**: Oscillating elevation for dynamic views
- **Helical**: Multiple loops with elevation change (best for 3DGS)

### Render Modes

- **Mesh**: Colored mesh with lighting
- **Depth**: Depth maps (with optional colormaps)
- **Skeleton**: 3D skeleton visualization (coming soon)

### Export Formats

- **COLMAP**: Standard sparse reconstruction format
  - `cameras.txt`: Camera intrinsics
  - `images.txt`: Camera extrinsics
  - `points3D.txt`: Initial point cloud
- **Images**: PNG with alpha channel

## Documentation

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for detailed specification and architecture notes.

For development notes, see [CLAUDE.md](CLAUDE.md).

## Architecture

Body2COLMAP uses a clean, modular architecture:

```
body2colmap/
├── coordinates.py   # Coordinate system definitions
├── camera.py        # Camera class with intrinsics/extrinsics
├── scene.py         # 3D scene management
├── path.py          # Orbit path generation
├── renderer.py      # Image rendering
├── exporter.py      # COLMAP export
├── pipeline.py      # High-level API
└── cli.py           # Command-line interface
```

### Key Design Principles

1. **Single coordinate system**: All internal computation in world/renderer coords
2. **Camera movement**: Mesh stays stationary, cameras orbit
3. **Explicit transforms**: Coordinate conversions only at system boundaries
4. **Separation of concerns**: Each module has single responsibility

## Usage with 3D Gaussian Splatting

Body2COLMAP output is compatible with standard 3DGS training pipelines:

```bash
# Generate training data
body2colmap --input person.npz --output-dir ./data/person

# Train with gaussian-splatting
cd gaussian-splatting
python train.py -s ../data/person
```

## Development

### Running Tests

```bash
# All tests
pytest

# Specific module
pytest tests/test_coordinates.py

# With coverage
pytest --cov=body2colmap
```

### Project Status

This is a skeleton architecture implementation. Core modules are defined but not fully implemented.

**Completed**:
- ✅ Architecture design
- ✅ Module interfaces
- ✅ Coordinate system specification
- ✅ Documentation

**In Progress**:
- ⏳ Core implementations
- ⏳ Unit tests
- ⏳ Integration tests

**TODO**:
- ❌ Skeleton rendering
- ❌ Video export
- ❌ YAML config support

See [CLAUDE.md](CLAUDE.md) for implementation roadmap.

## Contributing

Contributions welcome! Please see development notes in `CLAUDE.md` for architecture details and design principles.

## License

[Add license here]

## Citation

If you use this tool in your research, please cite:

```
[Add citation here]
```

## Acknowledgments

- SAM-3D-Body for 3D body reconstruction
- COLMAP for camera parameter format
- 3D Gaussian Splatting for the underlying technique
