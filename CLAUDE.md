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
- `coordinates.py`: Coordinate system definitions, conversions, spherical ↔ Cartesian
- `camera.py`: Camera intrinsics/extrinsics representation
- `path.py`: Orbit path pattern generation
- `scene.py`: 3D scene management (mesh, skeleton, lighting)
- `renderer.py`: Image rendering (mesh, depth, skeleton modes)
- `exporter.py`: Export to COLMAP and other formats
- `utils.py`: Auto-framing, homography warp, focal length utilities
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
│   ├── utils.py                 # Auto-framing, homography warp, focal length
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

- [x] Phase 1: Core infrastructure (coordinates, camera)
- [x] Phase 2: Scene and path generation
- [x] Phase 3: Rendering (mesh, depth, skeleton, composites)
- [x] Phase 4: Export (COLMAP, images)
- [x] Phase 5: Pipeline and CLI
- [x] Phase 6: Validation and testing

**Status**: ✅ FUNCTIONAL - Successfully generates 3DGS training data from SAM-3D-Body output

## Recent Fixes and Lessons Learned (2026-01-20)

### Portrait Orientation Auto-Framing
**Problem**: Figure appeared very small in portrait mode (720x1280) despite auto-framing.

**Root Cause**: Auto-framing computed orbit radius using 3D bounding box diagonal, which for standing humans is dominated by height. This caused the camera to be placed too far away.

**Solution**: Compute radius separately for horizontal and vertical dimensions:
```python
# Scene extents - for width, use max of X and Z since camera orbits
scene_width = max(
    max_corner[0] - min_corner[0],  # X extent
    max_corner[2] - min_corner[2]   # Z extent (depth)
)
scene_height = max_corner[1] - min_corner[1]  # Y extent

# Compute radius needed for each dimension, use max
radius_h = (scene_width / 2.0) / np.tan(horizontal_fov_rad * fill_ratio / 2.0)
radius_v = (scene_height / 2.0) / np.tan(vertical_fov_rad * fill_ratio / 2.0)
radius = max(radius_h, radius_v)
```

**Key Insight**: For orbiting cameras, horizontal extent must consider BOTH X and Z dimensions since the camera sees different projections at different orbit angles.

### Camera Look-At Target
**Problem**: Camera pointed too high, especially for meshes with higher vertex density in head/upper body.

**Root Cause**: Using `get_centroid()` (mean of all vertices) which is biased by vertex distribution.

**Solution**: Added `get_bbox_center()` which returns geometric center of bounding box:
```python
def get_bbox_center(self) -> NDArray[np.float32]:
    """Get center of axis-aligned bounding box (unaffected by vertex density)."""
    min_corner, max_corner = self.get_bounds()
    return (min_corner + max_corner) / 2.0
```

**Lesson**: For camera framing, use geometric center (bbox center), not vertex-weighted centroid.

### COLMAP Filename Pattern Mismatch
**Problem**: Custom `filename_pattern` used for saving images but not reflected in COLMAP `images.txt`.

**Root Cause**: `export_colmap()` hardcoded default pattern while `export_images()` used custom pattern.

**Solution**: Added `filename_pattern` parameter to `export_colmap()` and passed through from CLI config.

**Lesson**: When multiple export functions reference the same files, ensure pattern is consistent across all.

### Configuration Override Issues
**Problem**: Config file values for `n_frames` ignored, always using 120 frames.

**Root Cause**: CLI argument `--n-frames` had `default=120`, so argparse always provided a value even when user didn't specify it.

**Solution**: Remove default from argument definition, check `if args.n_frames is not None:` before overriding config.

**Lesson**: For CLI overrides of config file values, arguments should have NO default and use None-checking.

## Original-Camera Orbit Mode (2026-02)

### Overview
`--use-original-camera` / `original_focal_length` mode generates an orbit where frame 0 matches the SAM-3D-Body camera viewpoint (at the origin, looking toward the mesh). This enables diffusion-based pipelines to use the input image as a conditioning frame.

### Key Design: Geometric Radius
**Problem**: How to ensure frame 0 lands exactly at the origin (the original camera position)?

**Solution**: Use the geometric distance `||target||` (distance from origin to mesh bbox center) as the orbit radius. Since the orbit uses spherical coordinates centered on the target, and the original camera at the origin has a specific (azimuth, elevation) relative to the target, the spherical-to-Cartesian roundtrip reproduces the origin exactly.

This avoids any position discontinuity between frame 0 and frame 1 — they are just adjacent positions on the same smooth orbit. The auto-framed focal length was computed for this exact distance, so framing is correct by construction.

**Rejected alternative**: "Pinning" frame 0 to identity rotation + shifted principal point. This caused a rotation discontinuity at frame 0→1 because all other frames used `look_at()`. The approach was removed entirely.

### Key Design: Homography Warp for Input Image
**Problem**: Frame 0's camera has a `look_at()` rotation (not identity) and a different focal length than the original. How to warp the input image to match?

**Solution**: `compute_warp_to_camera()` in `utils.py` computes a 3×3 homography:
```
H = K_target @ R_cv @ K_orig^{-1}
```
where `R_cv = flip @ R_c2w^T @ flip` with `flip = diag(1, -1, -1)` to account for the OpenGL→OpenCV convention used in the projection pipeline.

This is used with `cv2.warpPerspective()` to align the original image with frame 0's rendered view.

### Key Design: Elevation Override
In original-camera mode, the `elevation_deg` parameter for circular orbits is **not user-tunable** — it is geometrically determined by the mesh position. The pipeline forces the derived elevation to ensure frame 0 lands at the origin. This follows the same pattern as the "Configuration Override Issues" lesson below.

### Spherical Coordinate Convention
Used by `coordinates.cartesian_to_spherical()` / `spherical_to_cartesian()`:
- **Y-up** convention (matches world coordinates)
- **Azimuth**: angle in XZ plane from +Z axis. 0° = +Z (toward viewer), 90° = +X (right), 180° = -Z (behind)
- **Elevation**: angle above XZ plane. 0° = eye level, +45° = above, -45° = below
- **Radius**: distance from origin

These are the inverse of each other: `spherical_to_cartesian(cartesian_to_spherical(v)) ≈ v`.

## Critical Implementation Details

### Skeleton Rendering
- **Format**: MHR70 (70 joints) → OpenPose Body25+Hands (65 joints)
- **Bone connectivity**: Official 65 bones from SAM-3D-Body repository
- **Colors**: OpenPose Body25 rainbow gradient + per-finger hand colors
  - Body: 25-color rainbow (pink-red → purple)
  - Hands: Per-finger colors (thumb=pink, index=orange, middle=green, ring=cyan, pinky=purple)
- **Gotcha**: OpenPose official colors have duplicate red at index 8 (was thigh) - changed to cyan-green for proper gradient

### Auto-Framing Strategy
For proper framing across all aspect ratios:
1. Compute scene extents per dimension (not diagonal)
2. For width: use `max(X_extent, Z_extent)` to account for orbit
3. For height: use `Y_extent`
4. Compute FOV for each dimension based on image dimensions
5. Calculate radius needed for each dimension separately
6. Use `max(radius_h, radius_v)` to ensure fit in both dimensions
7. Look at bbox center (not centroid) for consistent framing

### Configuration Management
- **Format**: YAML for human readability
- **Override precedence**: CLI args > config file > defaults
- **Pattern**: Load config from YAML, then selectively override with CLI args that are `not None`
- **Composite modes**: Support "depth+skeleton", "skeleton+face", "depth+skeleton+face" rendering

## Known Limitations

1. **Single mesh per scene**: Only supports one mesh at a time
2. **Static scenes only**: No animation/deformation support
3. **Fixed intrinsics**: All cameras share same intrinsics (typical for orbit rendering)
4. **Skeleton format**: Only MHR70 input supported (though converts to multiple formats)

## Future Enhancements

- [ ] Support for multiple meshes in scene
- [ ] Animation/temporal sequences
- [ ] Custom camera path patterns (Lissajous curves, etc.)
- [ ] Texture preservation from input image
- [ ] Normal map rendering mode
- [ ] Segmentation mask export
- [ ] Batch processing of multiple inputs
