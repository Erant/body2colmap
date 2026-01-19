# Body2COLMAP Architecture Notes

**Created**: 2026-01-19
**Purpose**: CLI tool for generating synthetic multi-view training data for Gaussian Splatting

## Project Overview

This tool takes SAM-3D-Body output (3D mesh reconstruction from a single image) and generates:
1. Multi-view rendered images from orbit camera paths
2. COLMAP format camera parameters (intrinsics + extrinsics)
3. Initial point cloud for 3D Gaussian Splatting training

## Core Design Principles

### 1. Single Canonical Coordinate System
- **All internal computation uses Renderer/OpenGL coordinates** (Y-up, camera looks down -Z)
- Coordinate conversions happen ONLY at system boundaries:
  - **Input boundary**: SAM-3D-Body → Renderer coords (in `scene.py`)
  - **Output boundary**: Renderer → COLMAP/OpenCV coords (in `exporter.py`)
- NO hidden transforms buried in rendering functions

### 2. Camera Movement, Not Mesh Movement
- Mesh remains stationary in world space
- Camera orbits around the mesh
- This matches COLMAP/3DGS expectations: static scene, moving cameras

### 3. Separation of Concerns
Each module has a single, clear responsibility:
- `coordinates.py`: Coordinate system definitions and conversions
- `camera.py`: Camera intrinsics/extrinsics representation
- `path.py`: Orbit path pattern generation
- `scene.py`: 3D scene management (mesh, skeleton, lighting)
- `renderer.py`: Image rendering (mesh, depth, skeleton modes)
- `exporter.py`: Export to COLMAP and other formats
- `pipeline.py`: High-level orchestration
- `cli.py`: Command-line interface

### 4. Explicit Over Implicit
- All transforms are documented with input/output coordinate systems
- No "magic" rotations or translations
- Clear function signatures with type hints
- Comprehensive docstrings

## Critical Lessons from Previous Implementation

### ❌ Mistakes to Avoid
1. **Hidden mesh transforms**: Previous impl applied 180° X rotation inside rendering
2. **Rotating mesh instead of camera**: Led to inverted transform confusion
3. **Multiple transform points**: Transforms scattered across functions
4. **Mixed c2w/w2c conventions**: Easy to get backwards
5. **Unclear coordinate systems**: No single source of truth

### ✅ How We Fix These
1. **No hidden transforms**: Mesh vertices are in world coords, period
2. **Stationary mesh**: Only camera moves
3. **Boundary transforms**: Convert coordinates at input/output only
4. **Consistent conventions**: Camera stores c2w, COLMAP export handles w2c
5. **Documented systems**: WorldCoordinates class defines canonical system

## Directory Structure

```
body2colmap/
├── CLAUDE.md                    # This file - top-level architecture notes
├── IMPLEMENTATION.md            # Original specification document
├── README.md                    # User-facing documentation
├── pyproject.toml               # Package configuration (Poetry/setuptools)
├── body2colmap/                 # Main package
│   ├── CLAUDE.md                # Package-level implementation notes
│   ├── __init__.py
│   ├── coordinates.py           # Coordinate systems and conversions
│   ├── camera.py                # Camera class
│   ├── path.py                  # Orbit path generators
│   ├── scene.py                 # Scene management
│   ├── renderer.py              # Rendering engine
│   ├── exporter.py              # Export to COLMAP/other formats
│   ├── pipeline.py              # High-level API
│   ├── config.py                # Configuration management
│   └── cli.py                   # Command-line interface
├── tests/                       # Unit tests
│   ├── CLAUDE.md                # Testing strategy notes
│   ├── test_coordinates.py
│   ├── test_camera.py
│   ├── test_path.py
│   ├── test_scene.py
│   ├── test_renderer.py
│   └── test_exporter.py
└── examples/                    # Example usage
    └── CLAUDE.md                # Example documentation
```

## Data Flow

```
SAM-3D-Body .npz file
        │
        ▼
┌─────────────────────────────────────────────┐
│  Scene.from_sam3d_output()                  │
│  • Load mesh vertices, faces                │
│  • Convert: SAM-3D coords → World coords    │  ← CONVERSION POINT #1
│  • Load skeleton (optional)                 │
└─────────────────────────────────────────────┘
        │
        ▼
    Scene object (world coords)
        │
        ├──────────────────────┬───────────────────┐
        ▼                      ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  OrbitPath      │  │  Camera         │  │  Renderer       │
│  • Generate     │  │  • Intrinsics   │  │  • Setup        │
│    camera       │  │  • look_at()    │  │    pyrender     │
│    positions    │  │                 │  │  • Render       │
└─────────────────┘  └─────────────────┘  │    modes        │
        │                      │           └─────────────────┘
        └──────────┬───────────┘                   │
                   ▼                               ▼
            List[Camera]                    List[RGBA images]
                   │                               │
                   ▼                               │
┌──────────────────────────────────────┐           │
│  ColmapExporter                      │           │
│  • Convert: World → COLMAP coords    │  ← CONVERSION POINT #2
│  • Write cameras.txt, images.txt     │           │
│  • Sample point cloud → points3D.txt │           │
└──────────────────────────────────────┘           │
                   │                               │
                   ▼                               ▼
          COLMAP sparse/0/              output_dir/*.png
```

## Key Design Decisions

### Why Renderer/OpenGL as Canonical Coords?
- Minimizes transforms (pyrender already uses OpenGL)
- Makes rendering straightforward (no conversion needed)
- Camera orbit math is intuitive in Y-up system
- COLMAP conversion is well-defined (180° X rotation)

### Why Separate Camera Class?
- Encapsulates both intrinsics and extrinsics
- Provides methods for common operations (look_at, project)
- Handles c2w ↔ w2c conversions internally
- Makes testing easier (can test camera math independently)

### Why Path Generators Return Camera Objects?
- Clean separation: path generation vs rendering
- Easy to test path patterns (check camera positions)
- Flexible: can mix multiple path types
- Reusable: same Camera class for all patterns

## Implementation Status

- [ ] Phase 1: Core infrastructure (coordinates, camera)
- [ ] Phase 2: Scene and path generation
- [ ] Phase 3: Rendering
- [ ] Phase 4: Export
- [ ] Phase 5: Pipeline and CLI
- [ ] Phase 6: Validation and testing

## Next Steps

1. Create package structure and skeleton files
2. Implement `coordinates.py` with WorldCoordinates and conversion functions
3. Implement `Camera` class with unit tests
4. Continue with scene, path, renderer, exporter
5. Build high-level pipeline and CLI
6. Integration testing with real SAM-3D-Body output
