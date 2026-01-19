# Testing Strategy

## Overview

Tests are organized by module, with each test file corresponding to a source module.

## Test Structure

```
tests/
├── test_coordinates.py   # Coordinate system conversions
├── test_camera.py        # Camera class
├── test_path.py          # Orbit path generation
├── test_scene.py         # Scene loading and management
├── test_renderer.py      # Rendering (may need fixtures)
├── test_exporter.py      # COLMAP export
└── test_pipeline.py      # Integration tests
```

## Testing Priorities

### HIGH PRIORITY (Must be tested thoroughly)

1. **coordinates.py**
   - SAM-3D to world conversion
   - World to COLMAP conversion
   - Rotation to quaternion (with known values)
   - Look-at matrix construction
   - Spherical to Cartesian conversion
   - **Critical**: Round-trip conversions should be identity

2. **camera.py**
   - c2w ↔ w2c conversions
   - COLMAP extrinsics output
   - Quaternion output (verify w,x,y,z order)
   - look_at() method
   - Projection of known 3D points

3. **path.py**
   - Orbit patterns produce expected positions
   - Cameras actually look at target
   - Circular: constant elevation
   - Helical: correct number of loops
   - Auto radius computation

4. **exporter.py**
   - COLMAP file format compliance
   - Correct quaternion order (w,x,y,z)
   - Two lines per image in images.txt
   - Point cloud coordinates

### MEDIUM PRIORITY

5. **scene.py**
   - Loading from .npz files
   - Bounds and centroid computation
   - Point cloud sampling

6. **renderer.py**
   - May require fixtures (sample mesh, cameras)
   - Alpha channel correctness
   - Depth rendering

7. **pipeline.py**
   - Integration tests (end-to-end)
   - May be slow, run separately

### LOW PRIORITY

8. **config.py** - Validation logic only
9. **cli.py** - Thin wrapper, tested manually

## Test Data

### Required Fixtures

1. **Simple mesh**: Cube or sphere with known geometry
2. **Sample SAM-3D-Body output**: Small .npz file for testing
3. **Known camera poses**: Hand-calculated for verification

### Generating Test Data

Create fixtures in `tests/fixtures/`:
- `simple_mesh.npz`: Minimal mesh for quick tests
- `known_transforms.json`: Ground truth for coordinate conversions

## Running Tests

```bash
# All tests
pytest

# Specific module
pytest tests/test_coordinates.py

# With coverage
pytest --cov=body2colmap

# Verbose
pytest -v
```

## Critical Test Cases

### Coordinate System Consistency

**Test**: Create camera, export to COLMAP, verify point projects correctly
```python
def test_coordinate_consistency():
    # 1. Create world point
    point_world = np.array([0, 0, 5])

    # 2. Create camera at known position
    camera = Camera(...)
    camera.position = np.array([2, 1, 8])
    camera.look_at(point_world)

    # 3. Export to COLMAP
    quat, tvec = camera.get_colmap_extrinsics()

    # 4. Manually transform point using COLMAP params
    # Should match camera.project(point_world)
```

### Quaternion Convention

**Test**: Verify COLMAP quaternion is (w,x,y,z) not (x,y,z,w)
```python
def test_quaternion_order():
    # Identity rotation
    R = np.eye(3)
    quat = rotation_to_quaternion_wxyz(R)

    # Identity quaternion is [1, 0, 0, 0] (w=1)
    assert quat[0] == 1.0  # w comes first
```

### Camera w2c Convention

**Test**: COLMAP stores world-to-camera, not camera-to-world
```python
def test_colmap_uses_w2c():
    # Camera at [0, 0, 5] looking at origin
    camera = Camera(...)
    camera.position = np.array([0, 0, 5])
    camera.look_at(np.array([0, 0, 0]))

    quat, tvec = camera.get_colmap_extrinsics()

    # Reconstruct w2c from COLMAP output
    R_w2c = quaternion_to_matrix(quat)

    # Apply to camera position
    # Should transform to origin (camera sees itself at origin in its frame)
    pos_in_cam = R_w2c @ camera.position + tvec
    assert np.allclose(pos_in_cam, [0, 0, 0])
```

## Common Issues to Test For

### Off-by-One Errors
- Frame numbering (1-based vs 0-based)
- Camera ID in COLMAP (starts at 1)
- Image ID in COLMAP (starts at 1)

### Sign Errors
- Camera forward is -Z, not +Z
- Elevation angle sign
- Azimuth rotation direction

### Coordinate Confusion
- Mixing c2w and w2c
- Forgetting to transpose rotation
- Wrong quaternion order

## Validation Tests

Beyond unit tests, we need integration validation:

### Visual Validation
1. Export to COLMAP
2. Open in COLMAP GUI
3. Verify:
   - Cameras are positioned correctly
   - Camera frustums point at mesh
   - Point cloud aligns with cameras

### 3DGS Validation
1. Run complete pipeline
2. Feed output to 3DGS training
3. Verify:
   - Training converges
   - Rendered novel views look reasonable
   - No strange artifacts

## Test Fixtures Location

Store test data in `tests/fixtures/`:
- Keep files small (<1MB)
- Document what each file contains
- Version control simple fixtures
- Gitignore large test outputs
