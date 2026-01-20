#!/usr/bin/env python3
"""
Installation script for body2colmap that handles PyOpenGL version conflict.

Usage:
    python install.py
"""

import subprocess
import sys
from pathlib import Path


def run_pip(*args):
    """Run pip command and check for errors."""
    cmd = [sys.executable, "-m", "pip"] + list(args)
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    return result.stdout


def main():
    """Install body2colmap with proper dependency handling."""
    script_dir = Path(__file__).parent
    requirements_file = script_dir / "install_requirements.txt"

    print("Installing body2colmap and dependencies...")
    print("=" * 60)

    # Step 1: Install all dependencies except pyrender
    print("\nStep 1/3: Installing core dependencies...")
    run_pip("install", "-r", str(requirements_file))

    # Step 2: Install pyrender without its dependencies
    print("\nStep 2/3: Installing pyrender (skipping conflicting dependencies)...")
    run_pip("install", "pyrender", "--no-deps")

    # Step 3: Install body2colmap in editable mode
    print("\nStep 3/3: Installing body2colmap...")
    run_pip("install", "-e", str(script_dir), "--no-deps")

    print("\n" + "=" * 60)
    print("✓ Installation complete!")
    print("\nVerify installation with:")
    print("  python -c 'import body2colmap; print(body2colmap.__version__)'")


if __name__ == "__main__":
    main()
