# Body2COLMAP

Generate synthetic multi-view training data for 3D Gaussian Splatting from SAM-3D-Body output.

## Overview

Body2COLMAP is a command-line tool and Python library that converts single-image 3D body reconstructions into multi-view synthetic datasets suitable for 3D Gaussian Splatting training.

**Input**: 3D mesh from SAM-3D-Body (`.npz` file) or Gaussian Splat (`.ply` file)
**Output**: Multi-view images + COLMAP camera parameters

### What it does

1. **Loads** 3D mesh reconstruction from SAM-3D-Body
2. **Generates** camera orbit paths (circular, sinusoidal, or helical)
3. **Renders** multi-view images (mesh, depth, skeleton, face landmarks)
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

For face landmark extraction (optional):
- mediapipe (`pip install mediapipe`)

## Quick Start

### Command Line

```bash
# Basic usage — mesh rendering with COLMAP export
body2colmap --input estimation.npz --output-dir ./output

# With skeleton overlay
body2colmap --input estimation.npz --output-dir ./output \
  --skeleton --render-modes depth+skeleton

# With face landmarks from a photo of the subject
python tools/extract_face_landmarks.py photo.jpg -o face.json
body2colmap --input estimation.npz --output-dir ./output \
  --face-landmarks face.json --render-modes skeleton+face
```

### Python API

```python
from body2colmap import OrbitPipeline

# Create pipeline
pipeline = OrbitPipeline.from_npz_file("estimation.npz")

# Auto-orient body to face camera (default, no offset)
pipeline.auto_orient()

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

Single modes:
- **mesh**: Colored mesh with lighting
- **depth**: Depth maps (with optional colormaps)

Composite modes (overlays combined via `+`):
- **depth+skeleton**: Depth map with skeleton overlay
- **skeleton+face**: Skeleton with face landmark overlay
- **depth+skeleton+face**: All three combined

### Auto-Orient

By default, the body is automatically rotated to face the camera at frame 0 of the orbit. The facing direction is computed from the skeleton's shoulder and hip joints (averaged torso normal, projected to the horizontal XZ plane).

Use `--initial-rotation` to add an offset from the auto-facing position:

```bash
# Default: body faces camera
body2colmap --input estimation.npz --output-dir ./output

# Body's right side toward camera
body2colmap --input estimation.npz --output-dir ./output --initial-rotation 90

# Back toward camera
body2colmap --input estimation.npz --output-dir ./output --initial-rotation 180
```

This ensures consistent starting orientation regardless of how the subject was posed in the source image, giving predictable control over when features appear and disappear during the orbit.

### Face Landmark Rendering

Face landmarks render the OpenPose Face 70 keypoint topology (jawline, eyebrows, nose, eyes, lips, pupils) as white points and connecting lines on top of the skeleton.

Two modes of operation:

1. **Canonical face model** (no external data): Uses a generic face shape derived from MediaPipe's canonical face geometry. Good for testing; not subject-specific.

2. **Subject-specific face landmarks** (recommended): Extract landmarks from a photo of the subject using the included `tools/extract_face_landmarks.py`, then pass the resulting JSON file via `--face-landmarks`. The landmarks are automatically aligned to the skeleton via Procrustes fitting.

Face landmarks are only rendered when the face is pointing toward the camera. By default this is the frontal 180-degree hemisphere; use `--face-max-angle` to narrow the range (e.g. `--face-max-angle 45` for only near-frontal views).

See [Face Landmarks](#face-landmarks) below for the full workflow.

### Framing Presets

- **full**: Entire body visible (default)
- **torso**: Waist up (requires skeleton data)
- **bust**: Shoulders and head (requires skeleton data)
- **head**: Head only (requires skeleton data)

### Export Formats

- **COLMAP**: Standard sparse reconstruction format
  - `cameras.txt`: Camera intrinsics
  - `images.txt`: Camera extrinsics
  - `points3D.txt`: Initial point cloud
- **Images**: PNG with alpha channel

## Face Landmarks

### Overview

Body2COLMAP can render face landmarks on top of the skeleton. This is a two-step process:

1. **Extract** face landmarks from a reference photo using `tools/extract_face_landmarks.py`
2. **Render** by passing the JSON file to `body2colmap --face-landmarks`

The extraction tool uses MediaPipe FaceLandmarker to detect 478 facial landmarks, which are then converted to the OpenPose Face 70 keypoint format internally.

### Step 1: Extract Face Landmarks

```bash
# Install mediapipe (one-time)
pip install mediapipe

# Extract landmarks from a photo
python tools/extract_face_landmarks.py photo.jpg -o face_landmarks.json

# With options
python tools/extract_face_landmarks.py photo.jpg \
  -o face_landmarks.json \
  --min-confidence 0.3 \
  --save-crop face_crop.jpg  # saves the detected face region for verification
```

On first run, the tool downloads two small model files (~4MB total) to `~/.cache/body2colmap/`.

The tool uses a two-stage detection pipeline:
- First tries MediaPipe FaceLandmarker on the full image (works for selfies/headshots)
- If no face is found, falls back to MediaPipe FaceDetector to locate the face bounding box, crops to it, then re-runs FaceLandmarker on the crop
- If multiple faces are detected, selects the most frontal one

### Step 2: Render with Face Landmarks

```bash
# Skeleton + face overlay
body2colmap --input estimation.npz --output-dir ./output \
  --face-landmarks face_landmarks.json \
  --render-modes skeleton+face

# Depth + skeleton + face
body2colmap --input estimation.npz --output-dir ./output \
  --face-landmarks face_landmarks.json \
  --render-modes depth+skeleton+face

# Face points only (no connecting lines)
body2colmap --input estimation.npz --output-dir ./output \
  --face-landmarks face_landmarks.json \
  --face-mode points \
  --render-modes skeleton+face
```

Providing `--face-landmarks` automatically enables face rendering (`--face-mode full`). You can override this with `--face-mode points` or `--face-mode none`.

### Face Landmarks JSON Format

The JSON file produced by `extract_face_landmarks.py` has the following structure:

```json
{
  "version": "1.0",
  "source": "mediapipe",
  "source_image": "photo.jpg",
  "image_size": [1536, 2048],
  "n_landmarks": 478,
  "refined": true,
  "landmarks": [
    [0.432100, 0.215300, -0.023400],
    [0.445200, 0.218100, -0.031200],
    ...
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | Format version (`"1.0"`) |
| `source` | string | Landmark source, must be `"mediapipe"` |
| `source_image` | string | Original image filename (informational) |
| `image_size` | [int, int] | Source image `[width, height]` in pixels. Required for correct aspect ratio during Procrustes alignment. |
| `n_landmarks` | int | Number of landmarks (468 or 478) |
| `refined` | bool | `true` if iris landmarks are present (478 points) |
| `landmarks` | [[float, float, float], ...] | Normalized landmark coordinates `[x, y, z]` |

**Landmark coordinates**:
- `x`: Normalized to image width (0.0 = left edge, 1.0 = right edge)
- `y`: Normalized to image height (0.0 = top edge, 1.0 = bottom edge)
- `z`: Relative depth estimate (roughly same scale as x; calibrated automatically during fitting)

The `image_size` field is important: it allows `body2colmap` to denormalize coordinates correctly so that portrait and landscape images produce proper face proportions.

### CLI Options Reference

| Option | Default | Description |
|--------|---------|-------------|
| `--face-landmarks PATH` | None | Path to face landmarks JSON. Implies `--face-mode full`. |
| `--face-mode {full,points,none}` | None | `full`: points + lines, `points`: points only, `none`: disabled |
| `--face-max-angle DEGREES` | 90 | Max degrees off face normal to render. 90 = full hemisphere, 45 = only within 45 degrees of straight-on. |
| `--skeleton` | off | Enable skeleton rendering (required for face) |
| `--render-modes MODES` | `mesh` | Comma-separated list, e.g. `skeleton+face,depth+skeleton+face` |

### Config File

Face options can also be set in the YAML config file:

```yaml
skeleton:
  enabled: true
  face_mode: "full"               # "full", "points", or null
  face_landmarks: "face.json"     # path to landmarks JSON
  face_max_angle: 90.0            # degrees off face normal to render
```

### Writing a Custom Client

If you want to produce face landmark JSON from your own detection pipeline (not using the included tool), the minimum required fields are:

```json
{
  "source": "mediapipe",
  "image_size": [WIDTH, HEIGHT],
  "landmarks": [
    [x0, y0, z0],
    [x1, y1, z1],
    ...
  ]
}
```

Requirements:
- `source` must be `"mediapipe"` (the only format currently supported)
- `landmarks` must contain at least 468 entries in MediaPipe Face Mesh vertex order
- If 478 entries are provided, indices 468-477 are treated as iris landmarks (468 = right iris center, 473 = left iris center)
- `image_size` should be `[width, height]` of the image used for detection, so coordinates can be denormalized correctly
- Coordinates should be normalized: `x` in [0, 1] relative to width, `y` in [0, 1] relative to height, `z` as relative depth

The conversion pipeline internally:
1. Maps MediaPipe 468 indices to OpenPose Face 68 keypoints (MaixPy convention)
2. Adds pupils (indices 68-69) from iris centers or eye contour centroids
3. Denormalizes coordinates using `image_size` (x * width, y * height, z * width)
4. Calibrates z depth to match the skeleton's head geometry
5. Fits to the skeleton via Procrustes alignment (5 anchor points: nose, eyes, ears)

## Configuration

### YAML Config File

Generate a default config template:

```bash
body2colmap --save-config config.yaml
```

Use it:

```bash
body2colmap --config config.yaml --input estimation.npz
```

CLI arguments override config file values.

## Usage with 3D Gaussian Splatting

Body2COLMAP output is compatible with standard 3DGS training pipelines:

```bash
# Generate training data
body2colmap --input person.npz --output-dir ./data/person

# Train with gaussian-splatting
cd gaussian-splatting
python train.py -s ../data/person
```

## Architecture

```
body2colmap/
├── coordinates.py   # Coordinate system definitions
├── camera.py        # Camera class with intrinsics/extrinsics
├── scene.py         # 3D scene management
├── path.py          # Orbit path generation
├── skeleton.py      # Skeleton format conversion and rendering data
├── face.py          # Face landmarks, Procrustes alignment, visibility
├── renderer.py      # Image rendering (mesh, depth, skeleton, face)
├── exporter.py      # COLMAP export
├── pipeline.py      # High-level API
├── config.py        # Configuration management (CLI + YAML)
└── cli.py           # Command-line interface
tools/
└── extract_face_landmarks.py  # Standalone MediaPipe face extraction utility
```

### Key Design Principles

1. **Single coordinate system**: All internal computation in Y-up OpenGL/renderer coords
2. **Camera movement**: Mesh stays stationary, cameras orbit
3. **Explicit transforms**: Coordinate conversions only at system boundaries
4. **Separation of concerns**: Each module has single responsibility

## Development

### Running Tests

```bash
# All tests
pytest

# Specific module
pytest tests/test_face.py

# With coverage
pytest --cov=body2colmap
```

## Documentation

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for detailed specification and architecture notes.

For development notes, see [CLAUDE.md](CLAUDE.md).

## Contributing

Contributions welcome! Please see development notes in `CLAUDE.md` for architecture details and design principles.

## License

[Add license here]

## Acknowledgments

- SAM-3D-Body for 3D body reconstruction
- MediaPipe for face landmark detection
- COLMAP for camera parameter format
- 3D Gaussian Splatting for the underlying technique
