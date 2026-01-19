# Architecture Summary

This document provides a quick overview of the body2colmap architecture.

## Directory Structure

```
body2colmap/
├── CLAUDE.md                      # Top-level architecture and design notes
├── IMPLEMENTATION.md              # Original detailed specification
├── ARCHITECTURE.md                # This file - architecture overview
├── README.md                      # User documentation
├── pyproject.toml                 # Package configuration
├── setup.py                       # Backwards compatibility
├── requirements.txt               # Core dependencies
├── requirements-dev.txt           # Development dependencies
├── .gitignore                     # Git ignore patterns
│
├── body2colmap/                   # Main package
│   ├── CLAUDE.md                  # Package implementation notes
│   ├── __init__.py                # Package exports
│   ├── coordinates.py             # Coordinate systems (208 lines)
│   ├── camera.py                  # Camera class (246 lines)
│   ├── scene.py                   # Scene management (241 lines)
│   ├── path.py                    # Orbit paths (245 lines)
│   ├── renderer.py                # Rendering engine (226 lines)
│   ├── exporter.py                # COLMAP export (301 lines)
│   ├── pipeline.py                # High-level API (247 lines)
│   ├── config.py                  # Configuration (178 lines)
│   └── cli.py                     # CLI entry point (127 lines)
│
├── tests/                         # Unit tests
│   ├── CLAUDE.md                  # Testing strategy notes
│   ├── __init__.py
│   ├── test_coordinates.py        # Coordinate tests (177 lines)
│   └── test_camera.py             # Camera tests (141 lines)
│
└── examples/                      # Usage examples
    └── CLAUDE.md                  # Example documentation
```

## Module Graph

```
User CLI / Python API
        ├─> cli.py
        └─> pipeline.py
                ├─> scene.py ──> coordinates.py
                ├─> path.py ───> camera.py ──> coordinates.py
                ├─> renderer.py
                └─> exporter.py

Dependencies:
- coordinates.py: No internal deps (pure math)
- camera.py: coordinates
- scene.py: coordinates
- path.py: camera, coordinates
- renderer.py: scene, camera
- exporter.py: camera, scene
- pipeline.py: all of the above
```

## Data Flow

```
SAM-3D-Body .npz
        │
        ▼
[Scene.from_sam3d_output]
        │ (Coordinate conversion #1: SAM-3D → World)
        ▼
    Scene (world coords)
        │
        ├──────────────────┬─────────────┐
        ▼                  ▼             ▼
   OrbitPath          Camera       Renderer
   (generate          (look_at)    (render)
    positions)             │             │
        │                  │             │
        └──────┬───────────┘             │
               ▼                         ▼
         List[Camera]            List[RGBA images]
               │                         │
               ▼                         │
    [ColmapExporter.export]              │
               │ (Conversion #2: World → COLMAP)
               ▼                         ▼
         COLMAP files              PNG images
```

## Key Design Decisions

### 1. Single Canonical Coordinate System
- **World/Renderer coords** (Y-up, Z-out) used throughout
- Conversions ONLY at boundaries (input, output)
- No hidden transforms in rendering

### 2. Stationary Mesh, Moving Camera
- Mesh fixed in world space
- Cameras orbit around mesh
- Matches COLMAP/3DGS expectations

### 3. Clean Module Separation
Each module has single responsibility:
- **coordinates.py**: Define system, provide conversions
- **camera.py**: Intrinsics + extrinsics representation
- **scene.py**: Mesh and geometry management
- **path.py**: Generate camera trajectories
- **renderer.py**: Produce images
- **exporter.py**: Write output files
- **pipeline.py**: Orchestrate everything
- **cli.py**: User interface

### 4. Camera Pose Convention
- Internally: c2w (camera-to-world) matrices
- COLMAP export: w2c (world-to-camera) quaternion + translation
- Conversion handled in Camera.get_colmap_extrinsics()

### 5. Explicit Over Implicit
- All functions document input/output coordinate systems
- Type hints throughout
- Comprehensive docstrings
- No "magic" behavior

## Implementation Status

### ✅ Complete
- Architecture design
- Module interfaces
- Coordinate system specification
- Documentation structure
- Skeleton implementations
- Basic test suite structure

### ⏳ In Progress (Next Steps)
1. Complete coordinate conversion implementations
2. Finish Camera class methods
3. Implement orbit path generators
4. Complete rendering integration
5. Finish COLMAP exporter
6. Add comprehensive tests

### ❌ Future Work
- Skeleton rendering
- Video export
- YAML config support
- Performance optimization
- Additional orbit patterns

## Critical Components

### Coordinate Conversions
Two conversion points:
1. **Input** (scene.py): SAM-3D-Body → World coords
2. **Output** (exporter.py): World → COLMAP coords

### Camera Math
- c2w ↔ w2c conversions
- Rotation matrix ↔ quaternion (w,x,y,z order!)
- look_at matrix construction
- Spherical ↔ Cartesian for orbit paths

### COLMAP Format
- cameras.txt: PINHOLE model, fx fy cx cy
- images.txt: Two lines per image, quaternion (w,x,y,z), world-to-camera
- points3D.txt: World coordinates, RGB 0-255

## Testing Strategy

### High Priority Tests
1. **Coordinate conversions** with known ground truth
2. **Camera transformations** (c2w, w2c, quaternions)
3. **Orbit paths** produce expected positions
4. **COLMAP export** format compliance

### Integration Tests
- Load SAM-3D-Body → render → export → verify in COLMAP viewer
- End-to-end: input .npz → output compatible with 3DGS

### Validation Tests
- Visual: Check in COLMAP viewer
- Functional: Train 3DGS and verify results

## Development Workflow

1. **Implement** core functionality module by module
2. **Test** each module in isolation
3. **Integrate** modules via pipeline
4. **Validate** with real data
5. **Optimize** if needed

## File Size Summary

Total lines of Python code (excluding tests): ~2,019 lines
- Core modules: 1,892 lines
- CLI/Config: 127 lines
- Tests: 318 lines (so far)

The codebase is intentionally verbose with comprehensive documentation.
Each function has detailed docstrings explaining:
- What coordinate system it uses
- What it does
- What it returns
- Important notes and gotchas

## Notes for Implementation

See [CLAUDE.md](CLAUDE.md) for detailed implementation notes.
See [tests/CLAUDE.md](tests/CLAUDE.md) for testing strategy.
See [IMPLEMENTATION.md](IMPLEMENTATION.md) for the original specification.
