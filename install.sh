#!/bin/bash
# Installation script for body2colmap that handles PyOpenGL version conflict

set -e

echo "Installing body2colmap and dependencies..."

# Install all dependencies except pyrender
echo "Step 1/3: Installing core dependencies..."
pip install -r install_requirements.txt

# Install pyrender without its dependencies (to avoid PyOpenGL 3.1.0)
echo "Step 2/3: Installing pyrender (skipping conflicting dependencies)..."
pip install pyrender --no-deps

# Install body2colmap in editable mode
echo "Step 3/3: Installing body2colmap..."
pip install -e . --no-deps

echo ""
echo "✓ Installation complete!"
echo ""
echo "Verify installation with:"
echo "  python -c 'import body2colmap; print(body2colmap.__version__)'"
