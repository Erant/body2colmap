# Testing Guide

## First Time Setup

### Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

**Required packages**:
- numpy
- pyrender
- trimesh
- opencv-python
- scipy

### Install System Dependencies for pyrender

pyrender requires OpenGL. On Linux:

```bash
# Ubuntu/Debian
sudo apt-get install libosmesa6-dev freeglut3-dev

# Or for headless rendering (recommended)
pip install pyrender
export PYOPENGL_PLATFORM=osmesa
```

## Basic Render Test

This tests the core pipeline: loading → coordinate conversion → rendering → saving.

### Option 1: With Real SAM-3D-Body Output

If you have a `.npz` file from SAM-3D-Body:

```bash
python test_basic_render.py <your_file.npz> output.png
```

### Option 2: With Test Cube Data

Generate a simple test cube:

```bash
# Create test data
python create_test_data.py test_cube.npz

# Render it
python test_basic_render.py test_cube.npz test_render.png
```

## What the Test Does

The `test_basic_render.py` script:

1. **Loads** the .npz file and validates required fields
2. **Converts** coordinates from SAM-3D-Body to world coords
3. **Analyzes** scene geometry (bounds, centroid)
4. **Sets up** a camera looking at the front of the mesh
5. **Renders** the mesh with lighting
6. **Saves** the result as PNG with alpha channel

## Expected Output

```
============================================================
Basic Render Test
============================================================

[1/5] Loading scene from test_cube.npz...
  ✓ Loaded: Scene(vertices=8, faces=12)
    Vertices: 8
    Faces: 12

[2/5] Analyzing scene geometry...
  Centroid: [0.000, 0.000, 5.000]
  Bounds: [-0.500, -0.500, 4.500] to [0.500, 0.500, 5.500]
  Bounding sphere radius: 0.866

[3/5] Setting up camera...
  Camera position: [0.000, 0.000, 7.165]
  Looking at: [0.000, 0.000, 5.000]
  Distance: 2.165
  FOV: 45°
  Resolution: 512x512

[4/5] Rendering...
  ✓ Rendered image: (512, 512, 4), dtype=uint8
    Min pixel value: 0
    Max pixel value: 255
    Mesh pixels: 45231 (17.3%)
    Background pixels: 216737 (82.7%)

[5/5] Saving to test_render.png...
  ✓ Saved: test_render.png

============================================================
✓ Test Complete!
============================================================

Rendered image saved to: test_render.png

This confirms:
  ✓ File loading works
  ✓ Coordinate conversion works
  ✓ Camera setup works
  ✓ Rendering works
  ✓ Image export works
```

## Troubleshooting

### "No module named 'pyrender'"

Install dependencies:
```bash
pip install -r requirements.txt
```

### "RuntimeError: Unable to initialize OpenGL"

Install OpenGL libraries or use OSMesa for headless rendering:
```bash
# Ubuntu
sudo apt-get install libosmesa6-dev

# Set environment variable
export PYOPENGL_PLATFORM=osmesa
```

### "Missing required field in .npz"

Your .npz file must contain:
- `pred_vertices`: (N, 3) mesh vertices
- `pred_cam_t`: (3,) camera translation
- `faces`: (M, 3) triangle indices

These are standard SAM-3D-Body outputs. If you're using custom data,
ensure these fields are present.

### "File not found"

Check the path to your .npz file. Use absolute paths if needed:
```bash
python test_basic_render.py /full/path/to/file.npz
```

## Next Steps

Once the basic render test passes:

1. **Visual inspection**: Open the rendered PNG and verify it looks correct
2. **Coordinate system**: Mesh should be right-side up, facing forward
3. **Alpha channel**: Background should be transparent/white
4. **Continue development**: Implement orbit paths, COLMAP export, etc.

## Unit Tests

Run the unit tests:

```bash
# All tests
pytest

# Specific module
pytest tests/test_coordinates.py

# With verbose output
pytest -v

# With coverage
pytest --cov=body2colmap
```

Note: Some tests may be incomplete as implementation continues.
