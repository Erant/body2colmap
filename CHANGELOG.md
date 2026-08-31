# Changelog

All notable changes to body2colmap will be documented in this file.

## [Unreleased]

### Added
- **Confidence gating for splat renders** (`--splat-confidence`): gates every pixel by
  how well the training views actually constrained the Gaussians covering it, instead of
  leaving the decision to a downstream threshold on rendered alpha. Drops low-confidence
  fringes at the source.
  - **Changes what the alpha channel means**: alpha becomes the confidence gate, not
    accumulated opacity. A mask taken from such a frame masks on evidence, not coverage.
  - Needs evidence: `ev_*` properties in the `.ply` (from `brush --export-evidence`) or
    `--splat-confidence-dataset` to measure it at render time
  - `.ply` input only. An overlay splat is reconstructed from a single photograph, so it
    has no training views to score against; `--splat-overlay` combined with
    `--splat-confidence` is rejected rather than silently degraded to plain alpha
  - `splat.cull_color` defaults to `render.bg_color`: the renderer resolves culled
    pixels and the background to one colour, so this is the whole background of a
    gated render. Set it only to make culled regions stand out.
  - Config: `splat.confidence`, `splat.cull_color`, `splat.gate_lo`, `splat.gate_hi`,
    `splat.confidence_sidecar`, `splat.confidence_dataset`, `splat.confidence_extra_args`
  - CLI: `--splat-confidence`, `--splat-cull-color`, `--splat-gate-lo`, `--splat-gate-hi`,
    `--splat-confidence-sidecar`, `--splat-confidence-dataset`
- `OrbitPipeline.render_splat_layers(cameras)` and `SplatRenderer.render_many(cameras)`:
  batched splat rendering. The renderer binary initializes its GPU context and loads the
  ply once per invocation, so a per-frame loop paid that setup once per frame.
  `render_composite_all()` and `render_all(modes=["splat"])` now batch internally.
- `OrbitPipeline.configure_splat_renderer()` to select the renderer binary, confidence
  options and verbosity for both the base and overlay splat paths
- `SplatRenderer(on_fault=...)` / `OrbitPipeline.configure_splat_renderer(on_fault=...)`:
  a hook called with a `RenderFault` when a splat render goes wrong, **while that
  invocation's temp directory still exists**. `render_many()` deletes it on the way out
  whatever happens, so this is the only chance to save the `cameras.json`, the frames
  that did land, or anything else from a crash — without it a failure on a machine that
  does not outlive the investigation leaves nothing but an exit code. Fires for a run
  that lost files and for one that wrote everything and died anyway (`RenderFault.complete`
  separates them), at most once per invocation; an exception out of the hook is logged
  and swallowed so a broken reporter cannot replace the render's own error.
- `SplatRenderer(on_output=...)`: a callback given each line the binary writes as it
  arrives, so a caller can relay progress into its own log. `render_many()` drives the
  binary through `Popen` and a line loop rather than `subprocess.run` to make this
  possible; output is now captured in every case, including under `verbose`, so a
  `RenderFault` always carries it.
- `SplatRenderer(ply_path=...)`: render an existing `.ply` instead of serializing
  `scene` to a temp file. For a caller that loaded the scene from a file and did not
  modify it, writing a hundreds-of-megabytes splat back out to render it is pure cost.
  `close()` leaves a caller-supplied file alone.
- Splat renders are judged by the files the renderer produced, not by its exit status.
  `brush-splat-render` intermittently dies from a signal after writing every frame it
  was asked for; such a run now succeeds with a logged warning, while one that actually
  lost frames raises, naming the missing files and decoding the signal. Partial writes
  are caught on read, since a truncated file is still non-empty.
- `SplatScene.get_framing_bounds()`, so a `.ply` input reaches
  `set_orbit_params()` at all — it previously raised `AttributeError`. Only the `"full"`
  preset is supported; the partial presets derive their threshold from skeleton joints,
  which a `.ply` does not carry.
- **Eyes rendered as shapes, not dots**: face landmark rendering now draws each eye as
  a filled two-tone shape — a flat sclera with a pupil disc centered on the pupil
  landmark — instead of the 6-point contour dots and their outline. Dots carry almost
  no gaze information at video-diffusion resolutions; an eye with a visible pupil does.
  - `face.build_eye_geometry()` builds both eyes from the fitted OpenPose Face 70
    landmarks; the contour is Catmull-Rom smoothed and flattened into its own plane
  - `pupil_scale` sets the pupil diameter as a fraction of the eye height measured at
    the pupil, so `1.0` is a disc touching the upper and lower lid. Values above 1.0
    are rejected rather than clipped.
  - The eye contour points, the pupil points and the 12 eye-loop bones are no longer
    drawn on top of the shapes
  - `skeleton.eye_style: "dots"` (`--eye-style dots`) restores the previous
    landmark-dot rendering
  - Config: `skeleton.eye_style`, `skeleton.eye_color`, `skeleton.pupil_color`,
    `skeleton.pupil_scale`
  - CLI: `--eye-style`, `--eye-color`, `--pupil-color`, `--pupil-scale`
- **`outline` render mode**: renders the mesh as a flat two-tone silhouette — one
  color for every pixel the mesh covers, another for the background, with no
  lighting or shading. Combinable with the skeleton overlay as `outline+skeleton`.
  - `Renderer.render_outline()` and `Renderer.render_mask()` (boolean silhouette
    coverage taken from the depth buffer, so it is exact and unaffected by
    lighting, mesh color or anti-aliasing)
  - `renderer.render_composite()` accepts `outline` as a base layer alongside
    `mesh` and `depth`
  - `style="stroke"` draws only a band along the silhouette boundary instead of a
    solid fill, with a configurable pixel width
  - Alpha marks mesh coverage (as in `mesh`/`depth`), so outline renders work as
    3DGS training masks and as composite base layers
  - `blur` softens the outline edge (color and alpha together), defaulting to a
    4 px radius; `0` restores hard two-tone edges. Applied inside
    `render_outline()`, so a skeleton overlay composited on top stays sharp.
  - Config: `render.outline_color` (foreground), `render.outline_bg_color`,
    `render.outline_style`, `render.outline_thickness`, `render.outline_blur`
  - CLI: `--outline-color`, `--outline-bg-color`, `--outline-style`,
    `--outline-thickness`, `--outline-blur`
- **Helical anchoring for original-camera mode**: `--use-original-camera` now works with
  `pattern: helical`, not just `circular`. One frame of the helix lands exactly on the
  SAM-3D-Body camera so the input image can be injected as a conditioning frame for
  diffusion pipelines.
  - `path.compute_helical_anchor_params()` solves the start azimuth, a sub-degree uniform
    elevation offset, and the resulting anchor frame index
  - `path.helical_elevation_deg()` exposes the elevation ramp as a pure function, shared
    by the generator and the solver
  - `OrbitPath.helical()` gained `elevation_offset_deg` (defaults to 0.0, backwards compatible)
  - Raises `ValueError` with actionable messages instead of emitting a degenerate path when
    the anchor is outside the helix elevation band, when there is no ramp to solve on, or
    when the helix is sampled too coarsely to reach the anchor
- First test coverage for `OrbitPath.helical()` (previously untested): elevation ramp,
  offset uniformity, anchor round-trips, smoothness, and all error paths

### Changed
- **Breaking**: Gaussian splats are rasterized by the external `brush-splat-render`
  binary instead of `gsplat`. gsplat publishes no wheel past torch 2.4 / cu124, so on a
  modern stack it JIT-compiles CUDA kernels on first use and needs `nvcc` at runtime,
  forcing a CUDA *devel* base image downstream. The replacement is wgpu/Vulkan.
  - `gsplat` is dropped from the `splat` extra (`plyfile` stays), and `torch` — which
    only ever existed to feed gsplat — is no longer imported anywhere in the package
  - Build the binary with `cargo build --release -p brush-splat-render` in a brush
    checkout. body2colmap finds it via `--splat-renderer` / `splat.renderer_binary`,
    then `$BRUSH_SPLAT_RENDER`, then `PATH`
  - Rendering is unchanged: measured against a captured gsplat oracle across both
    contracts (straight-alpha overlay at SH degree 0, composited base at SH degree 3),
    MAE 0.00013-0.00042 on RGB and 0.00011-0.00020 on alpha against a 1/255 = 0.0039
    bar, with best-fit integer shift (0, 0) on every frame
- **Breaking**: `splat.device` is removed, and a config still carrying it now raises
  rather than silently ignoring it. The binary picks its own wgpu adapter. Use
  `splat.renderer_binary` to point at a specific build.
- **Breaking**: `OrbitPipeline.attach_splat_overlay()` lost its `device` parameter; call
  `configure_splat_renderer()` instead.
- **Breaking**: `orbit_params['frame0_camera']` renamed to `orbit_params['anchor_camera']`.
  Read `orbit_params['anchor_frame_index']` for the conditioning frame rather than assuming 0
  — it is 0 for circular but solved for on helical.
- `--debug-original-view` writes `frame{index}_warped.png` / `frame{index}_overlay_{mode}.png`
  for the anchor frame (unchanged filenames when the index is 0)

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
