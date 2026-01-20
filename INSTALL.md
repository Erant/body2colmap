# Installation Guide

## Quick Install (Recommended)

Use the provided install script which handles dependency conflicts automatically:

```bash
# Option 1: Using Python script (cross-platform)
python install.py

# Option 2: Using shell script (Linux/macOS)
./install.sh
```

## Manual Installation

If you prefer to install manually or need to troubleshoot:

### Step 1: Install core dependencies

```bash
pip install -r install_requirements.txt
```

### Step 2: Install pyrender (without PyOpenGL 3.1.0)

```bash
pip install pyrender --no-deps
```

### Step 3: Install body2colmap

```bash
pip install -e . --no-deps
```

### Step 4: Verify installation

```bash
python -c "import body2colmap; print(body2colmap.__version__)"
```

## Why This Process?

The `pyrender` package has a hard dependency on `PyOpenGL==3.1.0` (exact version), which fails to build on modern Python versions. We work around this by:

1. Installing a newer PyOpenGL version (3.1.10+) first
2. Installing pyrender with `--no-deps` to skip its old PyOpenGL requirement
3. All other dependencies are compatible and install normally

## For ComfyUI Integration

When using body2colmap as a module in ComfyUI:

1. Navigate to your ComfyUI custom_nodes directory
2. Clone or link the body2colmap repository
3. Run the install script:
   ```bash
   cd custom_nodes/ComfyUI-Body2COLMAP
   python install.py
   ```

Alternatively, install system-wide:
```bash
cd /path/to/body2colmap
python install.py
```

Then import in your ComfyUI nodes:
```python
from body2colmap import Scene, OrbitPipeline, Camera
```

## Troubleshooting

### "No module named 'body2colmap'"

Make sure you're using the same Python environment where you installed the package:

```bash
which python
python -c "import sys; print(sys.path)"
pip show body2colmap
```

### PyOpenGL build errors

If you see build errors for PyOpenGL during installation, install it separately first:

```bash
pip install PyOpenGL PyOpenGL-accelerate
```

Then retry the installation.

### Import errors for other dependencies

Ensure all dependencies are installed:

```bash
pip install numpy scipy opencv-python trimesh Pillow imageio networkx pyglet freetype-py
```

## Dependencies

- Python >= 3.8
- numpy >= 1.20.0
- PyOpenGL >= 3.1.10
- pyrender >= 0.1.45
- trimesh >= 3.9.0
- opencv-python >= 4.5.0
- scipy >= 1.7.0
- Pillow >= 8.0.0
- imageio >= 2.9.0
- networkx >= 2.5
- pyglet >= 1.4.10
- freetype-py >= 2.2.0
