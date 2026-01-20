# ComfyUI-Body2COLMAP Implementation Guide

**Created**: 2026-01-20
**Purpose**: Implementation guide for creating a ComfyUI node pack that integrates body2colmap with ComfyUI-SAM3DBody

## Overview

This document describes how to implement **ComfyUI-Body2COLMAP**, a node pack that generates multi-view training data for 3D Gaussian Splatting from SAM-3D-Body mesh reconstructions.

### Design Philosophy

1. **Leverage existing nodes** - Use ComfyUI's built-in `SaveImage`, `PreviewImage`, `SaveAnimatedWEBP` instead of custom export nodes
2. **Minimize node count** - Only create nodes for functionality that doesn't exist
3. **Direct SAM3DBody integration** - Accept `SAM3D_OUTPUT` type directly from ComfyUI-SAM3DBody
4. **Modular but not fragmented** - Path generators are separate for flexibility, but render settings are inline

### Integration Point

The **ComfyUI-SAM3DBody** node pack (https://github.com/PozzettiAndrea/ComfyUI-SAM3DBody) outputs:

```python
SAM3D_OUTPUT = {
    "vertices": tensor,        # (N, 3) mesh vertices
    "faces": tensor,           # (F, 3) face indices
    "joints": tensor,          # (127, 3) MHR joint positions
    "joint_coords": tensor,    # Joint coordinates
    "joint_rotations": tensor, # Joint rotation parameters
    "camera": tensor,          # Camera translation
    "focal_length": float,     # Estimated focal length
    "bbox": tuple,             # Bounding box
    "pose_params": dict,       # Body/hand pose parameters
}
```

Our nodes accept this type directly, enabling seamless workflow integration.

---

## Node Architecture

### Final Node List (5 nodes)

| Node | Category | Purpose |
|------|----------|---------|
| `Body2COLMAP_CircularPath` | Path | Generate circular orbit camera path |
| `Body2COLMAP_SinusoidalPath` | Path | Generate sinusoidal elevation path |
| `Body2COLMAP_HelicalPath` | Path | Generate helical path (best for 3DGS) |
| `Body2COLMAP_Render` | Render | Render multi-view images from mesh |
| `Body2COLMAP_ExportCOLMAP` | Export | Export COLMAP sparse format |

### Nodes We DON'T Need (use built-ins instead)

| Don't Create | Use Instead | Reason |
|--------------|-------------|--------|
| ExportImages | `SaveImage` | Built-in handles batch images, saves each frame separately |
| PreviewFrame | `PreviewImage` + `ImageFromBatch` | Built-in preview works with batches |
| LoadNPZ | (none) | Debugging only, not for release |
| FromSAM3D | Integrated into `Body2COLMAP_Render` | Reduces node count |
| AnimationExport | `SaveAnimatedWEBP` / `SaveAnimatedPNG` | Built-in handles image sequences |

---

## Example Workflows

### Workflow 1: Basic 3DGS Training Data Generation

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────────┐
│ Load Image  │────▶│ SAM3DBodyProcess│────▶│ Body2COLMAP_        │
│ (built-in)  │     │ (SAM3DBody pack)│     │ HelicalPath         │
└─────────────┘     └────────┬────────┘     │ n_frames: 120       │
                             │              │ n_loops: 3          │
                             │              │ amplitude: 30°      │
                    SAM3D_OUTPUT            └──────────┬──────────┘
                             │                         │
                             │                    B2C_PATH
                             │                         │
                             ▼                         ▼
                    ┌────────────────────────────────────────────┐
                    │           Body2COLMAP_Render               │
                    │  resolution: 720x1280                      │
                    │  mode: mesh                                │
                    │  mesh_color: (0.65, 0.74, 0.86)            │
                    └──────────┬─────────────────────┬───────────┘
                               │                     │
                          IMAGE batch           B2C_RENDER_DATA
                               │                     │
                ┌──────────────┴──────────┐          │
                ▼                         ▼          ▼
        ┌─────────────┐          ┌──────────────────────────────┐
        │ SaveImage   │          │ Body2COLMAP_ExportCOLMAP     │
        │ (built-in)  │          │ output_dir: ./colmap_output  │
        │ prefix:     │          │ pointcloud_samples: 50000    │
        │ frame_      │          └──────────────────────────────┘
        └─────────────┘
```

**Output:**
- `output/frame_00001.png` through `frame_00120.png` (via SaveImage)
- `colmap_output/sparse/0/cameras.txt`, `images.txt`, `points3D.txt` (via ExportCOLMAP)

### Workflow 2: Quick Preview with Different Path Types

```
                    ┌─────────────────┐
                    │ SAM3DBodyProcess│
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │ CircularPath    │ │ SinusoidalPath  │ │ HelicalPath     │
    │ elevation: 15°  │ │ amplitude: 45°  │ │ loops: 2        │
    │ n_frames: 36    │ │ cycles: 3       │ │ amplitude: 20°  │
    └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
             │                   │                   │
             ▼                   ▼                   ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │ Render          │ │ Render          │ │ Render          │
    │ mode: mesh      │ │ mode: depth     │ │ mode: skeleton  │
    └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
             │                   │                   │
             ▼                   ▼                   ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │ SaveAnimatedWEBP│ │ SaveAnimatedWEBP│ │ SaveAnimatedWEBP│
    │ fps: 30         │ │ fps: 30         │ │ fps: 30         │
    └─────────────────┘ └─────────────────┘ └─────────────────┘
```

**Output:** Three animated WEBP files showing different orbit patterns and render modes.

### Workflow 3: Skeleton Overlay for Debugging

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Load Image  │────▶│ SAM3DBodyProcess│────▶│ HelicalPath     │
└─────────────┘     └────────┬────────┘     └────────┬────────┘
                             │                       │
                             ▼                       ▼
                    ┌────────────────────────────────────────┐
                    │         Body2COLMAP_Render             │
                    │  mode: mesh+skeleton                   │
                    │  skeleton_format: openpose_body25_hands│
                    │  joint_radius: 0.015                   │
                    │  bone_radius: 0.008                    │
                    └──────────────────┬─────────────────────┘
                                       │
                                  IMAGE batch
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
            ┌─────────────┐                      ┌─────────────────┐
            │ ImageFrom   │                      │ SaveImage       │
            │ Batch       │                      │ prefix: skel_   │
            │ index: 0    │                      └─────────────────┘
            └──────┬──────┘
                   │
                   ▼
            ┌─────────────┐
            │ PreviewImage│
            │ (built-in)  │
            └─────────────┘
```

---

## Detailed Node Specifications

### 1. Body2COLMAP_CircularPath

Generates a simple circular orbit at a fixed elevation angle.

```python
class Body2COLMAP_CircularPath:
    """Generate circular camera orbit path around mesh."""

    CATEGORY = "Body2COLMAP/Path"
    FUNCTION = "generate_path"
    RETURN_TYPES = ("B2C_PATH",)
    RETURN_NAMES = ("camera_path",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_data": ("SAM3D_OUTPUT",),
                "n_frames": ("INT", {
                    "default": 36,
                    "min": 4,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Number of camera positions around the orbit"
                }),
                "elevation_deg": ("FLOAT", {
                    "default": 0.0,
                    "min": -89.0,
                    "max": 89.0,
                    "step": 1.0,
                    "tooltip": "Camera elevation angle (0=eye level, positive=above)"
                }),
            },
            "optional": {
                "radius": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Orbit radius in meters (0=auto-compute from mesh bounds)"
                }),
                "fill_ratio": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "How much of viewport should contain mesh (for auto-radius)"
                }),
                "start_azimuth_deg": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 360.0,
                    "step": 5.0,
                    "tooltip": "Starting azimuth angle (0=front)"
                }),
            }
        }

    def generate_path(self, mesh_data, n_frames, elevation_deg,
                      radius=0.0, fill_ratio=0.8, start_azimuth_deg=0.0):
        # 1. Extract mesh bounds from SAM3D_OUTPUT
        # 2. Auto-compute radius if radius=0
        # 3. Generate circular camera positions
        # 4. Return B2C_PATH dict with cameras and metadata
        ...
```

**Use case:** Simple turntable-style views, product photography aesthetic.

---

### 2. Body2COLMAP_SinusoidalPath

Generates orbit with oscillating elevation for better surface coverage.

```python
class Body2COLMAP_SinusoidalPath:
    """Generate sinusoidal camera orbit with oscillating elevation."""

    CATEGORY = "Body2COLMAP/Path"
    FUNCTION = "generate_path"
    RETURN_TYPES = ("B2C_PATH",)
    RETURN_NAMES = ("camera_path",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_data": ("SAM3D_OUTPUT",),
                "n_frames": ("INT", {
                    "default": 60,
                    "min": 4,
                    "max": 1000,
                    "step": 1,
                }),
                "amplitude_deg": ("FLOAT", {
                    "default": 30.0,
                    "min": 0.0,
                    "max": 89.0,
                    "step": 1.0,
                    "tooltip": "Maximum elevation deviation from center"
                }),
                "n_cycles": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of up/down oscillations per full rotation"
                }),
            },
            "optional": {
                "radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0}),
                "fill_ratio": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0}),
                "start_azimuth_deg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0}),
            }
        }
```

**Use case:** Better coverage for meshes with interesting top/bottom details.

---

### 3. Body2COLMAP_HelicalPath

Generates multi-loop helical path - the recommended pattern for 3DGS training.

```python
class Body2COLMAP_HelicalPath:
    """Generate helical camera path - best for 3D Gaussian Splatting training."""

    CATEGORY = "Body2COLMAP/Path"
    FUNCTION = "generate_path"
    RETURN_TYPES = ("B2C_PATH",)
    RETURN_NAMES = ("camera_path",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_data": ("SAM3D_OUTPUT",),
                "n_frames": ("INT", {
                    "default": 120,
                    "min": 4,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Total number of frames across all loops"
                }),
                "n_loops": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Number of full 360° rotations"
                }),
                "amplitude_deg": ("FLOAT", {
                    "default": 30.0,
                    "min": 0.0,
                    "max": 89.0,
                    "step": 1.0,
                    "tooltip": "Elevation range: camera goes from -amplitude to +amplitude"
                }),
            },
            "optional": {
                "lead_in_deg": ("FLOAT", {
                    "default": 45.0,
                    "min": 0.0,
                    "max": 180.0,
                    "step": 5.0,
                    "tooltip": "Degrees of rotation at bottom before ascending"
                }),
                "lead_out_deg": ("FLOAT", {
                    "default": 45.0,
                    "min": 0.0,
                    "max": 180.0,
                    "step": 5.0,
                    "tooltip": "Degrees of rotation at top after ascending"
                }),
                "radius": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0}),
                "fill_ratio": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0}),
                "start_azimuth_deg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0}),
            }
        }
```

**Use case:** 3D Gaussian Splatting training - provides excellent multi-view coverage with smooth camera motion.

---

### 4. Body2COLMAP_Render

Main rendering node that generates multi-view images from the mesh.

```python
class Body2COLMAP_Render:
    """Render multi-view images of mesh from camera path."""

    CATEGORY = "Body2COLMAP"
    FUNCTION = "render"
    RETURN_TYPES = ("IMAGE", "B2C_RENDER_DATA")
    RETURN_NAMES = ("images", "render_data")
    OUTPUT_TOOLTIPS = (
        "Batch of rendered images (connect to SaveImage or PreviewImage)",
        "Render metadata for COLMAP export (connect to ExportCOLMAP)"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_data": ("SAM3D_OUTPUT",),
                "camera_path": ("B2C_PATH",),
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                }),
                "render_mode": ([
                    "mesh",
                    "depth",
                    "skeleton",
                    "mesh+skeleton",
                    "depth+skeleton"
                ], {
                    "default": "mesh",
                    "tooltip": "What to render: mesh surface, depth map, skeleton, or composites"
                }),
            },
            "optional": {
                # Mesh rendering options
                "mesh_color_r": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mesh_color_g": ("FLOAT", {"default": 0.74, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mesh_color_b": ("FLOAT", {"default": 0.86, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bg_color_r": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bg_color_g": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "bg_color_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),

                # Skeleton rendering options (for skeleton modes)
                "skeleton_format": ([
                    "openpose_body25_hands",
                    "mhr70"
                ], {"default": "openpose_body25_hands"}),
                "joint_radius": ("FLOAT", {
                    "default": 0.015,
                    "min": 0.001,
                    "max": 0.1,
                    "step": 0.001,
                    "tooltip": "Sphere radius for skeleton joints (meters)"
                }),
                "bone_radius": ("FLOAT", {
                    "default": 0.008,
                    "min": 0.001,
                    "max": 0.05,
                    "step": 0.001,
                    "tooltip": "Cylinder radius for skeleton bones (meters)"
                }),

                # Depth rendering options
                "depth_colormap": ([
                    "grayscale",
                    "viridis",
                    "plasma",
                    "inferno",
                    "magma"
                ], {"default": "grayscale"}),

                # Camera options
                "focal_length": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10000.0,
                    "tooltip": "Focal length in pixels (0=auto for ~47° FOV)"
                }),
            }
        }

    def render(self, mesh_data, camera_path, width, height, render_mode, **kwargs):
        """
        Render all camera positions and return batch of images.

        Returns:
            images: Tensor of shape [N, H, W, 3] in [0,1] range (ComfyUI IMAGE format)
            render_data: Dict containing cameras and scene for COLMAP export
        """
        # 1. Convert SAM3D_OUTPUT to body2colmap Scene
        # 2. Create Renderer with resolution
        # 3. For each camera in path:
        #    - Render frame based on mode
        #    - Convert RGBA [0,255] to RGB [0,1]
        # 4. Stack into batch tensor
        # 5. Package render_data with cameras + scene info
        ...
```

**Output `images`:** Connect to `SaveImage` (saves each frame), `PreviewImage` (preview batch), or `SaveAnimatedWEBP` (create animation).

**Output `render_data`:** Connect to `Body2COLMAP_ExportCOLMAP` for COLMAP format export.

---

### 5. Body2COLMAP_ExportCOLMAP

Exports camera parameters and point cloud in COLMAP sparse reconstruction format.

```python
class Body2COLMAP_ExportCOLMAP:
    """Export COLMAP sparse reconstruction format for 3DGS training."""

    CATEGORY = "Body2COLMAP"
    FUNCTION = "export"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    OUTPUT_NODE = True  # Terminal node - produces file output

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "render_data": ("B2C_RENDER_DATA",),
                "images": ("IMAGE",),
                "output_directory": ("STRING", {
                    "default": "output/colmap",
                    "tooltip": "Directory for COLMAP files (creates sparse/0/ subdirectory)"
                }),
            },
            "optional": {
                "filename_pattern": ("STRING", {
                    "default": "frame_{:04d}.png",
                    "tooltip": "Filename pattern for images.txt (must match SaveImage output)"
                }),
                "pointcloud_samples": ("INT", {
                    "default": 50000,
                    "min": 1000,
                    "max": 500000,
                    "step": 1000,
                    "tooltip": "Number of points to sample from mesh surface"
                }),
                "save_images": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also save images to output_directory/images/"
                }),
            }
        }

    def export(self, render_data, images, output_directory,
               filename_pattern="frame_{:04d}.png",
               pointcloud_samples=50000,
               save_images=True):
        """
        Export COLMAP format files.

        Creates:
            output_directory/
            ├── images/           (if save_images=True)
            │   ├── frame_0000.png
            │   └── ...
            └── sparse/0/
                ├── cameras.txt   (camera intrinsics)
                ├── images.txt    (camera extrinsics per image)
                └── points3D.txt  (initial point cloud)
        """
        # 1. Create directory structure
        # 2. Export cameras.txt (shared intrinsics)
        # 3. Export images.txt (per-frame extrinsics, world-to-camera)
        # 4. Sample point cloud from mesh, export points3D.txt
        # 5. Optionally save images
        ...
```

**Important:** The `filename_pattern` must match what's used in `SaveImage` prefix if using separate image saving.

---

## Custom Data Types

### B2C_PATH

Contains camera positions and orbit metadata.

```python
B2C_PATH = {
    "cameras": List[Camera],      # List of Camera objects with intrinsics/extrinsics
    "orbit_center": np.ndarray,   # (3,) center point cameras look at
    "orbit_radius": float,        # Distance from center
    "pattern": str,               # "circular", "sinusoidal", or "helical"
    "n_frames": int,              # Number of camera positions
}
```

### B2C_RENDER_DATA

Contains everything needed for COLMAP export.

```python
B2C_RENDER_DATA = {
    "cameras": List[Camera],      # Same cameras used for rendering
    "scene": Scene,               # body2colmap Scene object
    "resolution": Tuple[int,int], # (width, height)
    "focal_length": float,        # Focal length used
}
```

---

## Directory Structure

```
ComfyUI-Body2COLMAP/
├── __init__.py                 # ComfyUI registration
├── pyproject.toml              # Package metadata
├── requirements.txt            # Dependencies
├── README.md                   # User documentation
│
├── nodes/
│   ├── __init__.py             # Node exports
│   ├── path_nodes.py           # CircularPath, SinusoidalPath, HelicalPath
│   ├── render_node.py          # Render node
│   └── export_node.py          # ExportCOLMAP node
│
├── core/
│   ├── __init__.py
│   ├── types.py                # B2C_PATH, B2C_RENDER_DATA definitions
│   ├── sam3d_adapter.py        # SAM3D_OUTPUT → body2colmap conversion
│   └── comfy_utils.py          # Image format conversion utilities
│
├── body2colmap/                # Existing body2colmap package (symlink or copy)
│   ├── coordinates.py
│   ├── camera.py
│   ├── path.py
│   ├── scene.py
│   ├── renderer.py
│   ├── exporter.py
│   └── ...
│
└── workflows/                  # Example workflow JSON files
    ├── basic_helical_3dgs.json
    ├── multi_path_comparison.json
    └── skeleton_debug.json
```

---

## Implementation Details

### SAM3D_OUTPUT to Scene Conversion

The adapter converts SAM3DBody output to body2colmap's internal format:

```python
# core/sam3d_adapter.py

import numpy as np
import torch
from body2colmap.scene import Scene
from body2colmap.coordinates import sam3d_to_world

def sam3d_output_to_scene(mesh_data: dict, include_skeleton: bool = True) -> Scene:
    """
    Convert SAM3D_OUTPUT dict to body2colmap Scene.

    SAM3D uses MHR coordinates (different from our world coords).
    """
    # Extract and convert tensors to numpy
    vertices = _to_numpy(mesh_data["vertices"])
    faces = _to_numpy(mesh_data["faces"])

    # Convert from SAM3D/MHR coordinates to world coordinates
    # MHR: Y-down, Z-forward → World: Y-up, Z-backward
    vertices_world = sam3d_to_world(vertices)

    # Create trimesh
    import trimesh
    mesh = trimesh.Trimesh(vertices=vertices_world, faces=faces)

    # Extract skeleton if requested
    skeleton = None
    if include_skeleton and "joints" in mesh_data:
        joints = _to_numpy(mesh_data["joints"])
        # SAM3D uses 127 MHR joints, we need 70 MHR70 joints
        skeleton = convert_mhr_to_mhr70(joints)
        skeleton = sam3d_to_world(skeleton)

    return Scene(mesh=mesh, skeleton=skeleton)


def _to_numpy(tensor_or_array):
    """Convert torch tensor or numpy array to numpy."""
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.detach().cpu().numpy()
    return np.asarray(tensor_or_array)
```

### Image Format Conversion

ComfyUI uses `[B, H, W, C]` tensors with values in `[0, 1]`. Our renderer outputs `[H, W, 4]` numpy arrays with values in `[0, 255]`.

```python
# core/comfy_utils.py

import numpy as np
import torch

def rendered_to_comfy(images: list[np.ndarray]) -> torch.Tensor:
    """
    Convert list of rendered RGBA images to ComfyUI IMAGE format.

    Args:
        images: List of [H, W, 4] uint8 arrays in [0, 255]

    Returns:
        Tensor of shape [B, H, W, 3] in [0, 1] float32
    """
    # Stack into batch
    batch = np.stack(images, axis=0)  # [B, H, W, 4]

    # Drop alpha channel (ComfyUI IMAGE is RGB)
    batch = batch[..., :3]  # [B, H, W, 3]

    # Convert to float [0, 1]
    batch = batch.astype(np.float32) / 255.0

    # Convert to torch tensor
    return torch.from_numpy(batch)


def comfy_to_cv2(images: torch.Tensor) -> list[np.ndarray]:
    """
    Convert ComfyUI IMAGE to list of OpenCV BGR images for saving.

    Args:
        images: Tensor of shape [B, H, W, 3] in [0, 1]

    Returns:
        List of [H, W, 3] uint8 BGR arrays
    """
    # To numpy
    batch = images.cpu().numpy()  # [B, H, W, 3]

    # Scale to [0, 255]
    batch = (batch * 255).astype(np.uint8)

    # RGB to BGR for OpenCV
    batch = batch[..., ::-1]

    return [batch[i] for i in range(batch.shape[0])]
```

### OpenGL Context for Headless Rendering

ComfyUI may run headless (no display). Configure pyrender accordingly:

```python
# core/comfy_utils.py

import os

def setup_headless_rendering():
    """Configure OpenGL for headless rendering."""
    # Try EGL first (NVIDIA), fall back to OSMesa (software)
    if "PYOPENGL_PLATFORM" not in os.environ:
        try:
            # Test if EGL works
            os.environ["PYOPENGL_PLATFORM"] = "egl"
            import pyrender
            pyrender.OffscreenRenderer(64, 64)
        except Exception:
            # Fall back to OSMesa
            os.environ["PYOPENGL_PLATFORM"] = "osmesa"
```

Call this at module load time in `__init__.py`.

---

## Node Registration

```python
# __init__.py

from .core.comfy_utils import setup_headless_rendering

# Configure headless rendering before importing nodes
setup_headless_rendering()

from .nodes.path_nodes import (
    Body2COLMAP_CircularPath,
    Body2COLMAP_SinusoidalPath,
    Body2COLMAP_HelicalPath,
)
from .nodes.render_node import Body2COLMAP_Render
from .nodes.export_node import Body2COLMAP_ExportCOLMAP

NODE_CLASS_MAPPINGS = {
    "Body2COLMAP_CircularPath": Body2COLMAP_CircularPath,
    "Body2COLMAP_SinusoidalPath": Body2COLMAP_SinusoidalPath,
    "Body2COLMAP_HelicalPath": Body2COLMAP_HelicalPath,
    "Body2COLMAP_Render": Body2COLMAP_Render,
    "Body2COLMAP_ExportCOLMAP": Body2COLMAP_ExportCOLMAP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Body2COLMAP_CircularPath": "Circular Path",
    "Body2COLMAP_SinusoidalPath": "Sinusoidal Path",
    "Body2COLMAP_HelicalPath": "Helical Path (3DGS)",
    "Body2COLMAP_Render": "Render Multi-View",
    "Body2COLMAP_ExportCOLMAP": "Export COLMAP",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

---

## Implementation Phases

### Phase 1: Foundation
1. Create package structure
2. Implement `sam3d_adapter.py` - SAM3D → body2colmap conversion
3. Implement `comfy_utils.py` - image format conversion, headless setup
4. Test conversion with sample SAM3D output

### Phase 2: Path Nodes
1. Implement `Body2COLMAP_HelicalPath` first (most important)
2. Implement `Body2COLMAP_CircularPath`
3. Implement `Body2COLMAP_SinusoidalPath`
4. Test all path patterns produce valid camera positions

### Phase 3: Render Node
1. Implement `Body2COLMAP_Render` with mesh mode
2. Add depth rendering mode
3. Add skeleton rendering modes
4. Add composite modes (mesh+skeleton, depth+skeleton)
5. Test all render modes produce correct output

### Phase 4: Export Node
1. Implement `Body2COLMAP_ExportCOLMAP`
2. Verify COLMAP output works with 3DGS training
3. Test coordinate conversion (world → COLMAP)

### Phase 5: Polish
1. Create example workflows
2. Write user-facing README
3. Add comprehensive error messages
4. Test with ComfyUI Manager installation
5. Performance optimization for large frame counts

---

## Testing Strategy

### Unit Tests

Test body2colmap integration independently:

```python
def test_sam3d_to_scene_conversion():
    """Test SAM3D_OUTPUT converts to valid Scene."""
    mock_sam3d = {
        "vertices": torch.randn(6890, 3),
        "faces": torch.randint(0, 6890, (13776, 3)),
        "joints": torch.randn(127, 3),
    }
    scene = sam3d_output_to_scene(mock_sam3d)
    assert scene.mesh is not None
    assert len(scene.mesh.vertices) == 6890
```

### Integration Tests

Test full workflow in ComfyUI:

1. Load test image → SAM3DBodyProcess → HelicalPath → Render → SaveImage
2. Verify output images exist and have correct resolution
3. Verify COLMAP files are valid format

### Visual Validation

Create test workflow that renders all modes side-by-side for visual inspection.

---

## Dependencies

```txt
# requirements.txt
trimesh>=4.0.0
pyrender>=0.1.45
numpy>=1.24.0
opencv-python>=4.8.0
PyOpenGL>=3.1.0
torch>=2.0.0  # For tensor conversion (already in ComfyUI)
```

Note: `torch` is already available in ComfyUI environment.

---

## Known Limitations & Future Work

### Current Limitations

1. **Single mesh per render** - Cannot combine multiple SAM3D outputs
2. **No texture support** - Renders solid color mesh only
3. **Fixed skeleton format** - Only MHR70 → OpenPose conversion
4. **Memory for large batches** - 1000+ frames may need chunking

### Potential Future Nodes

| Node | Purpose | Priority |
|------|---------|----------|
| `Body2COLMAP_CombineMeshes` | Combine multiple SAM3D outputs | Low |
| `Body2COLMAP_CustomPath` | User-defined camera keyframes | Medium |
| `Body2COLMAP_TexturedRender` | Render with texture from input image | High |
| `Body2COLMAP_BatchProcess` | Process multiple images in batch | Medium |

---

## Coordinate System Reference

Understanding coordinate transforms is critical for correct COLMAP export:

```
SAM3D/MHR Coordinates          World/Renderer Coordinates       COLMAP Coordinates
(input from SAM3DBody)         (internal processing)            (output for 3DGS)

    Y (down)                       Y (up)                          Y (down)
    |                              |                               |
    |                              |                               |
    +--- X (right)                 +--- X (right)                  +--- X (right)
   /                              /                               /
  Z (forward)                    Z (backward)                    Z (forward)

Conversion: Flip Y, Flip Z      Conversion: 180° rotation around X axis
```

The conversions happen at two points:
1. **Input**: `sam3d_adapter.py` converts SAM3D → World
2. **Output**: `exporter.py` converts World → COLMAP

---

## Quick Reference: Connecting to Built-in Nodes

| Our Output | Connect To | Result |
|------------|------------|--------|
| `images` (IMAGE) | `SaveImage` | Saves each frame as separate file |
| `images` (IMAGE) | `PreviewImage` | Preview all frames in UI |
| `images` (IMAGE) | `SaveAnimatedWEBP` | Create animated preview |
| `images` (IMAGE) | `ImageFromBatch` → `PreviewImage` | Preview single frame |
| `render_data` | `Body2COLMAP_ExportCOLMAP` | Export COLMAP format |
