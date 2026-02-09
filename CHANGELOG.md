# Changelog

All notable changes to body2colmap will be documented in this file.

## [0.2.0] - 2026-02-09

### Changed
- Bump version to 0.2.0 for public release
- Update Development Status classifier from Alpha to Beta
- Remove unused `scipy` dependency
- Remove `setuptools_scm` from build-system requirements (not used)
- Add MIT LICENSE file
- Clean up README for public release

## [0.1.0] - 2026-01-20

### Production Ready
First fully functional release. Successfully generates 3D Gaussian Splatting training data from SAM-3D-Body output.

### Added
- Complete CLI tool with YAML configuration support
- Multiple rendering modes: mesh, depth, skeleton, and composites (mesh+skeleton, depth+skeleton)
- Auto-framing for proper figure scaling across all aspect ratios (portrait, landscape, square)
- Helical and circular orbit path patterns
- COLMAP format export (cameras.txt, images.txt, points3D.txt)
- Official MHR70 skeleton support with OpenPose Body25+Hands visualization
- Composite rendering with alpha-blending (e.g., mesh+skeleton overlay)
- Configuration file system with command-line overrides
- Separate width/height configuration options

### Fixed
- **Portrait auto-framing**: Figure no longer appears tiny in portrait orientations (e.g., 720x1280)
  - Root cause: Used 3D diagonal for scene size, only considered X for horizontal extent
  - Solution: Use per-dimension extents with max(X,Z) for width to account for orbit geometry
  - Commits: 660c969, e4dda2e

- **Camera look-at target**: Camera no longer points too high
  - Root cause: Used vertex-weighted centroid which is biased by mesh density
  - Solution: Use geometric bounding box center instead of centroid
  - Commit: c43412a

- **COLMAP filename mismatch**: images.txt now reflects custom filename patterns
  - Root cause: export_colmap() hardcoded default pattern while export_images() used custom pattern
  - Solution: Added filename_pattern parameter to export_colmap()
  - Commit: 263c081

- **Config override bypass**: Config file n_frames value no longer ignored
  - Root cause: CLI argument had default=120, always overriding config file
  - Solution: Remove default from argument, check for None before overriding
  - Commit: 12c9491

- **Skeleton rendering**:
  - Fixed bone connectivity using official MHR70 definitions (65 bones total)
  - Fixed color palette duplicate red at index 8 (changed to cyan-green for proper gradient)
  - Added per-finger hand colors (thumb, index, middle, ring, pinky)
  - Implemented MHR70 → OpenPose Body25+Hands format conversion

### Technical Details

#### Auto-Framing Algorithm
```python
# Key insight: For orbiting cameras, horizontal extent must consider BOTH X and Z
scene_width = max(
    max_corner[0] - min_corner[0],  # Front/back views see X
    max_corner[2] - min_corner[2]   # Side views see Z
)
scene_height = max_corner[1] - min_corner[1]

# Compute radius for each dimension separately, use max to ensure fit in both
radius_h = (scene_width / 2.0) / np.tan(horizontal_fov_rad * fill_ratio / 2.0)
radius_v = (scene_height / 2.0) / np.tan(vertical_fov_rad * fill_ratio / 2.0)
radius = max(radius_h, radius_v)
```

#### Skeleton Format Conversion
- Input: MHR70 (70 joints from SAM-3D-Body)
- Output: OpenPose Body25+Hands (65 joints)
- MidHip joint computed as average of left/right hips
- Official bone connectivity from SAM-3D-Body repository
- Custom color palette fix: index 8 changed from red to cyan-green

#### Configuration Management
- YAML-based configuration files
- Three-tier precedence: CLI args > config file > defaults
- CLI arguments must have NO default to allow config file values through
- Separate --width and --height options for flexible resolution control

### Known Limitations
- Single mesh per scene only
- Static scenes (no animation support)
- All cameras share same intrinsics
- MHR70 skeleton input only (though converts to other formats)

## [0.1.0] - 2026-01-19

### Added
- Initial implementation of core modules
- Basic coordinate system conversions
- Camera class with look_at functionality
- Scene loading from SAM-3D-Body .npz files
- Basic mesh and depth rendering
- COLMAP export infrastructure

---

## Version History Context

This project went through multiple iterations before reaching production quality:

1. **Early implementation (pre-January 2026)**: Had coordinate system confusion with hidden transforms
2. **Refactor (2026-01-19)**: Clean architecture with single canonical coordinate system
3. **Production release (2026-01-20)**: All critical bugs fixed, ready for real-world use

### Migration from Previous Versions

If migrating from earlier implementations:
- Update coordinate conversion calls (SAM-3D → World now happens in scene.py)
- Replace centroid with bbox_center for camera framing
- Update skeleton bone definitions to official MHR70 list
- Use Config.from_yaml() for configuration management
