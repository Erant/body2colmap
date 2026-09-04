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
  - **Input boundary**: external Gaussian splat → Renderer coords (in `splat_anchor.py`)
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
- `renderer.py`: Image rendering (mesh, depth, outline, skeleton modes)
- `background.py`: Environment backdrop (sphere/cube) drawn behind the base layer
- `fade.py`: Bounding ellipsoid + decay profiles, fading the backdrop out
  around the subject
- `splat_scene.py` / `splat_renderer.py`: Gaussian splat storage, and
  rasterization via the external `brush-splat-render` binary
- `splat_anchor.py`: Place an externally-built splat in world coords
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
│   ├── background.py            # Environment backdrop (sphere/cube)
│   ├── fade.py                  # Backdrop fade around the subject
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
│   ├── test_background.py
│   ├── test_fade.py
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
In original-camera mode, the `elevation_deg` parameter for circular orbits is **not user-tunable** — it is geometrically determined by the mesh position. The pipeline forces the derived elevation to ensure frame 0 lands at the origin. The same applies to `start_azimuth_deg` for helical orbits (see below). This follows the same pattern as the "Configuration Override Issues" lesson below.

### Helical Anchoring (2026-08)

**Problem**: The frame-0 guarantee only works for circular orbits. A helix sweeps elevation monotonically, so the original camera's elevation dictates *where in the sequence* the anchor can occur — it is generally not frame 0. Previously `pattern: helical` + `use_original_camera` silently produced an orbit where frame 0 had the right azimuth but the wrong elevation, and `Renderer.warp_original_image()` would reject it (it asserts the camera is at the origin).

**Solution**: `path.compute_helical_anchor_params()` solves for the frame that can land on the anchor, using three steps:

1. **Elevation → frame index.** The elevation ramp (`path.helical_elevation_deg()`) is monotone piecewise-linear in progress `i / n_frames`. Invert it at the anchor's elevation, then round to the nearest frame `k`.
2. **Azimuth → `start_azimuth_deg`.** Azimuth is a pure linear ramp, so `start_azimuth = azimuth_anchor - (k / n_frames) * total_deg` makes frame `k` hit the anchor azimuth exactly.
3. **Residual → `elevation_offset_deg`.** Rounding in step 1 leaves a residual `δ`, applied to *every* frame. The path stays a perfect helix (its band shifts from `[-A, +A]` to `[-A+δ, +A+δ]`) while frame `k` lands exactly on the anchor. For the shipped `helical.yaml` (81 frames, 2 loops, A=40°, lead 30/90) the elevation step is ~1.15°/frame, so `|δ| ≤ ~0.58°`.

**Key contract**: `pipeline.orbit_params['anchor_frame_index']` — always read it rather than assuming 0. It is 0 for circular (and for sinusoidal, which is *not* anchored) and solved for on helical. The matching camera is `orbit_params['anchor_camera']` and the tilt applied is `orbit_params['anchor_elevation_offset_deg']`.

**Errors instead of degenerate paths**: the solver raises `ValueError` when the anchor's elevation is outside `±amplitude_deg` (fix: raise `helical_amplitude_deg`), when there is no ramp to solve on (`n_loops < 1` or `amplitude_deg <= 0`), or when the helix is sampled too coarsely to reach the anchor within `max_elevation_error_deg` (default 2°).

**Shared ramp function**: `helical()` and the solver both call `helical_elevation_deg()` so the generator and its inverse cannot drift apart.

### Spherical Coordinate Convention
Used by `coordinates.cartesian_to_spherical()` / `spherical_to_cartesian()`:
- **Y-up** convention (matches world coordinates)
- **Azimuth**: angle in XZ plane from +Z axis. 0° = +Z (toward viewer), 90° = +X (right), 180° = -Z (behind)
- **Elevation**: angle above XZ plane. 0° = eye level, +45° = above, -45° = below
- **Radius**: distance from origin

These are the inverse of each other: `spherical_to_cartesian(cartesian_to_spherical(v)) ≈ v`.

## Outline Rendering Mode (2026-08)

### Overview
`outline` renders the mesh as a **flat two-tone image** — one color for every
pixel the mesh covers, another for the background — with no lighting, shading
or gradient. The only information in the frame is the silhouette shape, which
makes it useful as a control/conditioning image. It composites with the
skeleton overlay as `outline+skeleton`.

Configurable via `render.outline_color` (foreground), `render.outline_bg_color`,
`render.outline_style`, `render.outline_thickness` and `render.outline_blur`,
or the matching `--outline-*` CLI flags.

### Key Design: Coverage From the Depth Buffer
`Renderer.render_mask()` derives the silhouette from `depth > 0` rather than
from the color buffer's alpha. The depth buffer does not depend on mesh color,
lighting or anti-aliasing, so the mask is exact and strictly binary — which is
what a flat two-tone render needs. Anything else needing mesh coverage should
call `render_mask()` instead of re-deriving it from a color render.

### Key Design: One Mask, Two Styles
`style="filled"` (default) fills the whole silhouette. `style="stroke"` draws
only a band along the boundary, computed morphologically as
`dilate(mask) & ~erode(mask)`, which handles disconnected components and
interior holes without any contour bookkeeping.

### Key Design: Blur Is Applied Inside `render_outline()`
`outline_blur` (radius in px, default 4) Gaussian-blurs color and alpha
together. It runs at the end of `render_outline()`, **not** on the finished
composite, so overlays land on top of the already-blurred base and stay sharp —
the skeleton in `outline+skeleton` is never blurred. Moving this to the
composite stage would smear the skeleton as well.

### Key Design: Alpha Means Mesh Coverage
Alpha is the silhouette (union the outward half of a stroke band), **not** the
drawn foreground. This keeps `outline` interchangeable with `mesh` and `depth`
as a composite base layer and as a 3DGS training mask, and it stops
`outline+skeleton` in stroke style from writing the skeleton into
fully-transparent pixels. See `body2colmap/CLAUDE.md` for the full rationale.

## Eye Rendering (2026-08)

### Overview
Face landmark rendering draws the eyes as **filled two-tone shapes** rather than
as the MediaPipe/OpenPose dots: each 6-point eye contour is filled flat in
`eye_color`, with a `pupil_color` disc centered on the pupil landmark (68/69).
A ring of dots carries almost no gaze information at video-diffusion
resolutions; a sclera with a dark pupil does, which is the point — these frames
condition gaze in the video model.

Configurable via `skeleton.eye_color`, `skeleton.pupil_color` and
`skeleton.pupil_scale`, or the matching `--eye-color` / `--pupil-color` /
`--pupil-scale` CLI flags. `skeleton.eye_style: dots` / `--eye-style dots`
restores the original landmark-dot rendering.

### Key Design: `pupil_scale` Is Capped At 1.0
`pupil_scale` is the pupil diameter as a fraction of the eye height, and the
eye height is measured **at the pupil** — twice the distance from the pupil
center to the nearest lid segment. The pupil landmark sits off-center, so
sizing the disc from the tallest part of the opening would let it poke through
a lid near the corners. Under this definition 1.0 is exactly a disc touching
the upper and lower lid, so capping the value there (asserted in
`build_eye_geometry()`, validated in `config.py`) removes any need to clip the
pupil against the eye.

### Key Design: The Eye Is Flattened Into Its Own Plane
The eye contour curves around the eyeball. Left alone, that curvature bulges in
front of the flat pupil disc and clips it into a bowtie shape. Both shapes are
meant to read as flat areas, so `_build_single_eye()` projects the contour onto
its own least-squares plane and lifts the pupil a few percent of the eye height
along the normal. See `body2colmap/CLAUDE.md` for the full rationale.

### Key Design: Eye Landmarks Are Not Also Drawn As Dots
`face.get_face_draw_lists(eye_style)` returns the points and bones the renderer
iterates. Under `"shape"` it withholds the 12 contour points, the 2 pupil points
and the 12 eye-loop bones: drawing them as well would speckle the sclera and
redraw an outline the filled shape already provides. Under `"dots"` it returns
everything, which reproduces the original rendering exactly — that is the whole
escape hatch, so there is no second code path to keep in sync.


## Gaussian-Splat Overlay (2026-08)

### Overview
`skeleton+splat` composites a real Gaussian splat of the subject's face on top
of the skeleton, in place of the synthetic `skeleton+face` landmarks. The splat
is built externally by `~/Projects/masktest` (Sapiens2 seg + pointmap + normal →
normal-integrated depth → one oriented Gaussian per masked pixel) from **the
same photograph that feeds SAM-3D-Body**. Nothing from that pipeline is
implemented here; it is fed in as a `.ply` plus its `splat_meta.json`.

The point is conditioning: a ring of landmark dots carries almost no identity,
and the canonical face model carries none at all. A splat of the actual face
carries both identity and gaze.

Configurable via the `splat:` config section or the `--splat-*` CLI flags.

### Key Design: Both Frames Already Agree, Up To Three Scalars
This is why the feature is small. Both worlds are Y-up, +Z toward the viewer,
and in both the source camera has **identity rotation**:

- masktest defines its world as `F @ p_cam - centroid`, `F = diag(1,-1,-1)`, so
  its camera sits at `-centroid` with identity OpenGL rotation — by
  construction, since no pose was ever estimated from one view.
- `sam3d_to_world()` leaves the SAM-3D-Body camera at the origin with identity
  rotation.

So `P_world = M @ (p_ply + centroid)`, where `M` is one upper-triangular 3×3
reconciling the two recovered focal lengths plus a scale gauge. There is no
rotation to solve for and no registration step.

### Key Design: `M` Is Re-Unprojection, Not A Fudge
With the gauge `s = 1`, `M` is exactly "unproject the splat's own depth map
using SAM-3D-Body's focal length instead of the one the pointmap network
assumed". The network commits to an implicit focal inferred from image content,
and on a tight face crop that is not the real camera's — measured 1122 px
against SAM-3D-Body's 1509 px on the same photo, a 26% disagreement.

`M` is therefore anisotropic (depth scales by `s`, lateral extent by
`s·f_s/f_m`) *because* the focals disagree. That is the correction, not a
distortion introduced by it. Exact 2-D alignment and a linear depth map cannot
both hold with a pure similarity unless the focals already match.

`splat.reconcile_intrinsics: false` forces a uniform scale instead: it preserves
the splat's shape exactly, at the cost of rendering the face 34% off-size
against the skeleton on this data.

### Key Design: The Scale Gauge Is Fitted Against Depth, Not Size
Scaling about the camera centre leaves every projection unchanged, so `s` is
invisible at the anchor frame. What it sets is how far along each ray the splat
sits — and therefore whether the face and the skeleton stay together as the
orbit turns away. A wrong `s` shows up as drift growing with view angle, not as
a misalignment you can see head-on.

`estimate_depth_scale()` takes the **median ratio of the mesh's depth buffer to
the splat's nearest-surface depth** over the pixels both cover. Median, not
least squares: the mesh is itself a fit to the photo, so some edge pixels pair
face against background. Comparing bounding-box centres instead would be biased
by centimetres — the mesh head's centre is inside the skull, the splat is a
surface.

Measured on the test pair: 1.3726, putting the splat at 1.61–1.83 m inside the
mesh head's 1.60–1.90 m.

### Key Design: The Cull Threshold Is Measured
The splat is a 2.5-D shell — one view, nothing behind the subject — so as the
camera turns away, the open rim swings into view as a flare of
grazing-incidence splats. Frames past `splat.max_angle_deg` off the source view
direction drop the layer entirely.

The default of **45°** comes from a sweep, not a guess: the face reads cleanly
to ~30°, the rim starts flaring by 45°, and by 60° the shell is mostly edge.

### Key Design: `splat` Names A Base *Or* An Overlay, By Scene Type
A `.ply` input is a `SplatScene` and `splat` is its **base** layer, rendered
alone. An `.npz` input with `--splat-overlay` is a mesh `Scene` and `splat` is
an **overlay** on top of the skeleton. The two cannot co-occur — a `SplatScene`
has no mesh or skeleton to composite against — and `attach_splat_overlay()`
raises if you try.

### Validation
The gate is masktest's own: render the anchored splat from the original camera
and compare against the photograph. The **best-fit integer shift over a ±4 px
search must be (0, 0)** — PSNR also falls from ordinary splat blur, but a wrong
quaternion order, a scale paired with the wrong rotation column or a flipped
axis all *displace* the render. Measured on the test pair: shift (0, 0),
30.98 dB over 23412 mask pixels.

### Gotcha: SAM-3D-Body's Focal Is Full-Image, Not Crop
`cli.py` used to warn that `focal_length` "is from SAM-3D-Body's internal crop".
It is not, at least with the MoGe FoV estimator wired in: `camera_head.py` sets
`focal_length = cam_int[0, 0]` straight from the estimator's intrinsics for the
**whole image**, and `pred_cam_t` places the mesh so that
`pred_vertices + pred_cam_t` projects with that `K` into full-image pixels
(verified: reprojected keypoints match `pred_keypoints_2d` to 1e-4 px). MoGe
returns a centred principal point, so `(cx, cy) = (W/2, H/2)`.

This is what lets a *crop* of the photo be related to SAM-3D-Body's frame
analytically, via `splat.crop_box`.

## Rasterizing Splats: brush, Not gsplat (2026-08)

### Overview
Gaussian splats are rasterized by **`brush-splat-render`**, a standalone binary
in the brush repo (`crates/brush-splat-render`, documented in brush's
`docs/splat-render.md`). `splat_renderer.py` writes a `.ply` and a
`cameras.json`, invokes it, and reads RGBA PNGs back.

It replaced a `gsplat.rasterization` call, which was also the only use of torch
in the package. gsplat publishes no wheel past torch 2.4 / cu124, so on a modern
stack it JIT-compiles its CUDA kernels on first use and needs `nvcc` at runtime
-- forcing a CUDA *devel* base image on anything that packages this. brush is
wgpu/Vulkan and is already present wherever this pipeline runs, so the swap
drops the CUDA toolchain entirely and leaves one graphics API instead of two.

Find the binary via `--splat-renderer` / `splat.renderer_binary`, then
`$BRUSH_SPLAT_RENDER`, then `PATH`.

### Key Design: The Camera Convention Is the Binary's Job
`Camera.rotation` is serialized row-major into `cameras.json` **untouched**. The
binary's `to_brush_camera()` does the OpenGL -> OpenCV conversion
(`R_cv = R_gl @ diag(1, -1, -1)`, i.e. negate the Y and Z *columns*). Converting
on both sides cancels into a vertically mirrored render that looks almost right,
which is exactly the failure this note exists to prevent.

Note the pre-swap `splat_renderer.py` docstring claimed "no conversion needed"
directly above the lines that converted. That stale comment was corrected, not
carried across.

### Key Design: One Invocation Per Sequence
The binary initializes wgpu and loads the ply once per invocation, then loops
the camera list. So the batch call is the primary API -- `render_many()`, and
`pipeline.render_splat_layers(cameras)` for the overlay -- with the singular
forms reserved for genuine one-off frames. `render_composite_all()` renders
every splat layer up front rather than inside its frame loop.

### Key Design: Always Render on Black
The binary always runs with `--background 0,0,0`, whatever `bg_color` says, so
its RGB comes back premultiplied. Both alpha conventions are then derived in
Python. See `body2colmap/CLAUDE.md` for the full rationale.

### Key Design: Artifacts Decide Success, Not the Exit Code
The binary can be killed by SIGSEGV *after* writing every frame. `render_many()`
therefore verifies the expected files exist and are non-empty, and reads the
exit status only to explain a shortfall: a crash that produced everything warns
and proceeds, a crash that lost frames raises. See `body2colmap/CLAUDE.md`.

### Key Design: A Crashed Render Gets One Chance To Be Kept
`render_many()` works in a temp directory it deletes on the way out, success or
failure -- so by the time a caller sees the exception, the `cameras.json` and
the frames that did land are gone. `on_fault` is called *before* that, with a
`RenderFault` holding the live paths, for a caller that wants to save a crash
report. Downstream (b2crunner) does exactly this. See `body2colmap/CLAUDE.md`.

### Confidence Gating
`--confidence` gates each pixel by per-splat multi-view evidence rather than a
downstream alpha threshold, exposed as `splat.confidence` and the
`--splat-confidence*` flags. **It makes the alpha channel the gate, not
accumulated opacity**, and it works only for a `.ply` base render -- an overlay
splat is built from a single photograph and has no training views to measure
against, so `attach_splat_overlay()` refuses it.

### Validation
The swap was gated against a captured gsplat oracle: the same splats and the
same cameras, rendered once through gsplat and once through brush, on both
contracts (straight-alpha overlay at SH degree 0, and composited base render at
SH degree 3). Measured MAE 0.00013-0.00042 on RGB and 0.00011-0.00020 on alpha,
against a bar of 1/255 = 0.0039, with best-fit integer shift (0, 0) everywhere
-- structure matters more than the mean here, since a mirrored or offset render
is a convention bug however small its MAE. End-to-end, 14 of 16 composited
`skeleton+splat` frames came out bit-identical and the 2 splat-bearing frames
differed by MAE 7.3e-5.

## Environment Backdrop (2026-08)

### Overview
`background.py` draws a static environment — the inside of a sphere or an
axis-aligned cube — behind the base render, so an orbiting camera sees the
world sweep past. It exists to break one specific failure: with a blank
background a video diffusion model reads an orbit as **the subject rotating**,
and prompt conditioning is not strong enough to correct it. A world-fixed
backdrop is the cue that says otherwise.

Configurable via the `background:` config section or the `--background-*` CLI
flags. Off by default; when on, it defaults to a **`grid` cube at 3x the orbit
radius** (`DEFAULT_RADIUS_SCALE`). Textures are either generated (`grid`,
`checker`, `gradient`, `blender_sky`) or loaded from disk (equirect image,
packed cubemap, or a directory of six faces).

### Key Design: A Remap, Not Scene Geometry
The obvious implementation — a giant inverted sphere added to the pyrender
scene — breaks `outline` mode. `Renderer.render_mask()` derives mesh coverage
from `depth > 0`, and a surrounding sphere puts geometry behind **every**
pixel, so the silhouette becomes the whole frame and the alpha channel with it.

Instead each pixel's ray is intersected with the surface analytically and the
hit point looked up in the texture, as one `cv2.remap`. That also sidesteps
pyrender's `zfar`, an unlit-textured-material setup, and pole pinching on a
tessellated sphere. Intrinsics are fixed across an orbit, so the camera-space
ray grid is computed once and only rotated per frame.

### Key Design: The Cue Is Azimuthal Structure, Which the Blender Sky Lacks
A Nishita sky is **azimuthally symmetric apart from its sun**. Rotating the
camera about Y changes nothing else in frame — which is precisely the motion
the backdrop is supposed to make legible. As a rotation cue the sun is doing
all of the work.

This is measured, not asserted: `tests/test_background.py` pins the per-latitude
standard deviation of each generated texture. With the sun suppressed the sky
scores 0.00 8-bit levels; `checker` scores 82 and `grid` 27. `blender_sky` is
shipped, but `grid` and `checker` are what actually carry the signal.

### Key Design: The Default Is a Grid Cube, and the Three Settings Are a Set
`grid` + `cube` + `radius_scale: 3.0` is the default because it is the
arrangement that carries the cue most strongly. Compared side by side on one
orbit against a checker sphere, a grid sphere and a checker cube: `checker`
scores higher on raw azimuthal variance but its cells are self-similar, so it
says the view turned without saying how far; a sphere has no corners to pass at
all. The cube's wall seams and its floor/ceiling split are the landmarks that
make the rotation legible.

The three settings stand or fall together. `radius_scale` is defaulted rather
than left at infinity **because a cube at infinity is not a room** — the ray
intersection drops out, the corners with it, and the render is a plain ruled
field with no parallax, which is the weak version of the feature. Defaulting
the texture and the geometry while leaving the radius alone would have shipped
exactly that.

The cost is that `radius` and `radius_scale`, being mutually exclusive, now
need a rule for "the caller set only the other one". There is one rule, applied
at all three entry points: **an explicitly set `radius` supersedes the
defaulted `radius_scale`**; naming both is still an error. Infinity stays
reachable — `--background-infinite`, `radius_scale: null` in YAML, or
`radius_scale=None` in `configure_background()`, which distinguishes an
explicit `None` from an unmentioned argument via a `_UNSET` sentinel.

### Key Design: Radius Decides Whether Sphere vs Cube Means Anything
With both radius forms null the surface is at infinity and the lookup depends
only on ray *direction*. The backdrop then tracks camera rotation but not translation —
correct for a distant sky, and the point at which sphere and cube differ only
in how the texture is parameterized, not in what is rendered.

A finite radius intersects the ray properly (forward root of the sphere; the
nearest slab exit for the box) and gives real parallax between subject and
backdrop. That is what makes a cube read as a room. `radius_scale` sizes it as
a multiple of the orbit radius, for auto-framed orbits whose scale is not known
up front; it must exceed 1.0, and a camera that ends up outside the surface is
an error rather than a garbage render.

### Key Design: The Backdrop Goes Under the *Base* Layer
`render_composite()` draws it immediately after the base and before any
overlay. The skeleton overlay writes RGB without touching alpha, so
compositing the backdrop after it would blend the skeleton away everywhere
outside the mesh silhouette.

`opaque` (default true) then forces alpha to 255, which is right for
conditioning frames. `--background-keep-alpha` fills only RGB and leaves the
silhouette alpha intact as a training mask.

### Key Design: Oversized Textures Are Area-Averaged Down
A 4K panorama point-sampled into a 720p frame resamples differently every
frame, and the shimmer reads as motion to a video model — the exact opposite of
what a *static* backdrop is for. `_fit_resolution()` downsamples once, to about
twice the render's own angular resolution (`2·pi·fx` texels around a sphere,
`(pi/2)·fx` across a cube face).

### Scope
Conditioning frames only. The backdrop is not exported to COLMAP, adds no
points to the point cloud, and never enters the depth buffer or the silhouette
mask. It is also mesh-scene only: a `.ply` input is rasterized by
`brush-splat-render`, which composites against a flat colour of its own, so
there is no pyrender base layer to draw behind and the CLI rejects the
combination.

### Validation
The load-bearing test renders a texture holding one bright texel and checks
where it lands. A mirrored, transposed or half-turned lookup all still produce
a plausible-looking sky; only comparing against `Camera.project()` catches
them. Measured: sub-pixel agreement (0.75 px tolerance) across four marker
directions, on both sphere and cube.


## Backdrop Fade (2026-09)

### Overview
`fade.py` fades the backdrop toward a flat tone in a shell around the subject.
It exists because the backdrop broke something while fixing something else: in
`outline` modes a grid running right up to the silhouette reads to VACE as a
**hard occlusion boundary**, so it refuses to paint outside the outline and
bulky clothing and hair come out squashed onto the shape of the bare mesh. A
structure-free zone next to the silhouette gives the model room to expand,
while the far field keeps the rotation cue the backdrop was added for.

Configurable via the `background.fade:` config section or the
`--background-fade*` CLI flags. Off by default; enabling it with
`--background-fade PROFILE` names the decay shape in the same flag, matching
`--background`'s own shape.

### Key Design: The Shell Is a Fitted Ellipsoid, Not a Dilated Outline
The clear zone is the projection of a minimum-volume ellipsoid fitted to the
**mesh vertices**, and only incidentally looks like a fattened silhouette.
Three reasons, none of which a per-frame mask dilation can supply:

- **An ellipsoid enclosing the mesh encloses its silhouette from every
  viewpoint.** The clear zone therefore cannot fall inside the outline partway
  round the orbit — a failure that would be invisible in the frame you checked.
- **It is one fixed world-space object**, so the clear zone is a region the
  camera moves around rather than a screen effect that swims frame to frame.
  For a video model that temporal consistency is the whole game.
- **The per-pixel test is two lines.** Transform the ray into the space where
  the ellipsoid is the unit sphere and take its closest approach to the origin;
  no distance transform, no silhouette rasterization.

A capsule was the other candidate — the user's own framing — and hugs a
standing figure slightly better along the spine. It was not taken because it
does not linearize: the "how far outside am I" scalar stops being a norm and
the ray test grows a segment case. `margin` covers the same ground.

### Key Design: Enclosure Is Imposed, Not Solved For
Khachiyan's algorithm is iterative, given a loose tolerance and an iteration
cap. Its answer is a good *shape*, not a proof of enclosure. `Ellipsoid.fit()`
therefore scales the result until the outermost point of the **full** vertex
set sits exactly on the surface.

That is what lets the fit run on a stride of the vertices — 4000 by default,
~50 ms on an SMPL-X mesh — with no risk of clipping a stray vertex. The
guarantee is O(N) and runs on everything.

### Key Design: `falloff` Is In Subject Radii, Not Pixels
`ray_distance()` returns the closest approach in units of the ellipsoid's own
radius in that direction, so the band width is dimensionless. One setting holds
across an auto-framed orbit at any subject size or orbit radius — neither of
which is known when the value is chosen.

The band is consequently anisotropic in world units: as wide as the subject is
in each direction, so a standing figure gets a tall clear zone and a narrow
one. This is also why `falloff: 1.0` is milder than it sounds — the clear zone
is the ellipsoid at 2x, against a subject filling ~0.8 of the frame.

### Key Design: Seven Profiles, Split Into Compact and Tailed
`step`, `linear`, `smoothstep` (default) and `cosine` reach zero exactly at the
band edge. `exponential`, `gaussian` and `inverse_square` have tails that never
quite do, and take `rate` to tighten them. `inverse_square`'s tail is the heavy
one: still 2.7% faded at three band widths out at the default rate, which reads
as a faint wash over the whole frame. That is the profile's character and it is
pinned in `tests/test_fade.py`; it is also why a compact profile is the default.

`step` is the control condition — a plain hole in the backdrop — and is what
the coverage tests use, since it makes the clear zone a clean boolean.

### Key Design: The Lines Fade To The Wall, Via A Second Texture
`target: plain` (the default) renders the backdrop **twice through one set of
sampling maps**: once normally, and once from the same generator with
`flat=True`, which suppresses its pattern — a grid's walls without their lines,
a checker's mean tone. At full fade weight the pixel *is* the plain render, so
the line colour goes to the wall colour that was behind it and everything else
— the wall shading, the corners, the floor/ceiling split — stays exactly where
it was, at full sharpness.

The two textures are fitted, padded and sampled in lockstep precisely so this
identity holds: `tests/test_fade.py` asserts the clear zone is *bit-identical*
to a `flat=True` backdrop rendered on its own.

**Rejected: blurring the backdrop into itself.** The first implementation
area-averaged the render down to ~24 px and back up and called it `local`,
reasoning that a box average below the texture's own frequency removes the
lines. It does not. An average removes a line's *frequency*, not its
brightness: the line's energy is spread into a wide grey band, so the clear
zone comes out as a smear of the grid rather than a grid-free wall — visibly
so, and measurably (the cleared region reads brighter than the wall it should
have become). It survives as `target: blur`, honestly named, because a
**loaded** texture has no plain variant to reveal and it is the only option
there.

**Rejected: one flat colour.** `target: color` erases the backdrop's shading
along with its lines, so the clear zone reads as a patch wherever it crosses a
floor/wall seam — a soft blob is still a shape. Kept as the honest control and
for deliberately matching `render.bg_color`.

The `flat` kwarg lives on each generator rather than in a table of
pattern-suppressing parameter overrides, because for `grid` there is no such
override: setting `line_color = base_color` still leaves lines visible on the
floor and ceiling, which use their own base colours. Only the generator knows
what its pattern is.

### Scope
The fade runs inside `Background.render()`, so `composite()` lays the base
layer over an already-faded backdrop and the subject is untouched by
construction. Everything the backdrop is excluded from, the fade is excluded
from too: no COLMAP export, no point cloud, no depth buffer, no silhouette
mask. It needs a backdrop to fade — `--background-fade` alone is rejected
rather than ignored — and a mesh to fit an ellipsoid to, so splat scenes are
out for the same reason they have no backdrop. `target: plain` additionally
needs a *generated* texture, and is rejected rather than silently downgraded
when given a loaded one.

### Validation
The load-bearing test is `TestClearZoneCoversTheSilhouette`: every projected
vertex must be fully faded, from fifteen viewpoints spanning a full orbit at
three elevations. A fade that is merely centred on the subject passes a
single-frame eyeball check and fails this. The far-field half of the same test
pins that the cue survives — a fade over the whole frame is the original
failure with extra steps.

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
- **Composite modes**: Support "depth+skeleton", "outline+skeleton", "skeleton+face", "depth+skeleton+face" rendering

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
