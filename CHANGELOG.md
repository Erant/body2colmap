# Changelog

All notable changes to body2colmap will be documented in this file.

## [Unreleased]

### Fixed
- **`SplatScene` no longer strips extra vertex properties on a load-and-save round
  trip.** `from_ply` collects every column beyond the standard 3DGS set into
  `scene.extras` (name -> per-Gaussian array, file order, dtype intact) and `to_ply`
  writes them back after the standard columns. brush's `ev_*` multi-view evidence
  block was the casualty: a scene staged through `to_ply` for `brush-splat-render
  --confidence` lost it, the renderer warned "--confidence without evidence", and the
  gate silently became plain alpha. `transform_splat_scene` copies extras through by
  row (unreoriented; it refuses the SH-degree-3 splats evidence rides on anyway)

### Added
- **An outline drawn from a supplied mask.** `Renderer.render_outline` takes
  `mask=`, a boolean (height, width) array that replaces the mesh silhouette, and
  `render_composite` forwards `modes["outline"]["mask"]` to it. The drawing itself
  moved to a module-level `outline_from_mask(mask, ...)` — the same fill, stroke,
  colours and blur, on any coverage mask, with no GL and no mesh
  - **Why**: the mesh silhouette is wrong wherever hair and clothing leave the body
    model. A matte of the subject — of a photograph, or of a frame a video model
    generated from the mesh drawing — is a better opinion of where the subject is,
    and the outline is the one layer of a `outline+skeleton+splat` frame that can
    take it without touching the skeleton or the splat
  - The mask must be on the renderer's pixel grid; a wrong shape or a non-boolean
    dtype is a `ValueError`, never a silent resample. Alpha then tracks the mask
    rather than mesh coverage
  - `render_mask` is not called when a mask is given, so a mesh-free Renderer can
    draw the outline
- **Inactive-region mask for `*+splat` composites** (`--splat-inactive-mask`):
  replaces each frame's alpha with the inactive/reactive mask a conditioned video
  model takes — **0 over the splat** ("already real, keep it"), **255 elsewhere**
  ("synthetic, generate it"), the convention Wan 2.2 VACE uses. Off by default.
  - **Why**: the splat is the one part of a `skeleton+splat` frame that is real
    photographic content, and without a mask nothing says so. Marking it inactive
    is what stops the model repainting the face the overlay exists to carry
  - In the alpha channel rather than a parallel `masks/` sequence, because that is
    where the downstream already reads it (b2crunner's `wan22_vace_denoise` takes
    the frame's own alpha as `control_masks`). RGB is untouched — an inactive pixel
    is marked, not erased
  - The cost is that alpha stops meaning subject coverage, so one run yields either
    the conditioning mask or a 3DGS training mask, not both. Hence opt-in
  - `--splat-mask-threshold` (default `0.9`) is the splat alpha at or above which a
    pixel is marked inactive. High on purpose: below full coverage the frame is a
    blend of the splat with the layers under it, and preserving such a pixel
    preserves the synthetic half too
  - `--splat-mask-grow PX` grows the inactive region, or shrinks it when negative —
    the useful direction, pulling the boundary clear of the splat's antialiased edge
  - A frame past `--splat-max-angle` carries no splat and comes out wholly reactive,
    rather than carrying no mask: a sequence with gaps no longer lines up with the
    frames it describes
  - Needs `--splat-overlay` and a `*+splat` render mode; either missing is an error
    rather than a silently uniform mask
  - Config: `splat.inactive_mask`, `splat.inactive_mask_threshold`,
    `splat.inactive_mask_grow`. API:
    `OrbitPipeline.render_composite_all(inactive_mask=...)` and
    `Renderer.render_composite(inactive_mask=...)`, taking a
    `splat_renderer.InactiveMaskOptions`
- **Environment backdrop** (`--background`): draws a world-fixed sphere or cube behind
  the render, so an orbit reads as the camera moving rather than the subject spinning on
  a turntable. Off by default.
  - Textures are generated or loaded: `grid`, `checker`, `gradient`, `blender_sky` (an
    approximation of Blender's default Sky Texture), or a path to an equirectangular
    image, a packed cubemap (4:3 cross, 6:1 strip, 1:6 column) or a directory of six
    face images
  - **Defaults to a `grid` cube at 3x the orbit radius** — walls meeting at corners over
    a floor and ceiling that read apart. Compared side by side against a checker sphere,
    a grid sphere and a checker cube on the same orbit, this is the arrangement that
    makes the rotation legible; the other three either repeat (checker) or give the
    camera nothing to pass (a sphere has no corners). The three settings are a set: a
    cube is only a room at a finite radius
  - `--background-geometry sphere|cube`. At an infinite radius the two differ only in
    how the texture is parameterized; the geometric difference needs a finite radius
  - `--background-radius` / `--background-radius-scale`: a finite surface gives real
    parallax between subject and backdrop, which is what makes a cube read as a room.
    A radius supersedes the defaulted scale; `--background-infinite` puts the surface at
    infinity instead, where it tracks camera rotation but not translation
  - **Caveat worth knowing before picking a texture**: a Nishita-style sky is
    azimuthally symmetric apart from its sun, so it barely changes as the camera orbits
    and supplies almost none of the cue this feature exists for. `grid` and `checker`
    carry roughly 40x and 120x more azimuthal signal respectively (measured as
    per-latitude standard deviation, pinned in `tests/test_background.py`)
  - Alpha is forced opaque by default, suiting conditioning frames.
    `--background-keep-alpha` fills only RGB and leaves the silhouette alpha usable as
    a training mask
  - Conditioning frames only: the backdrop is not exported to COLMAP, adds no points to
    the point cloud, and never enters the depth buffer or the silhouette mask.
    `.npz` input only — a `.ply` is rasterized by brush against a flat colour of its
    own, and the combination is rejected rather than ignored
  - Config: the `background:` section. CLI: `--background`, `--no-background`,
    `--background-geometry`, `--background-resolution`, `--background-radius`,
    `--background-radius-scale`, `--background-infinite`, `--background-rotation`,
    `--background-keep-alpha`
  - API: `OrbitPipeline.configure_background()`, `clear_background()`, and the
    `body2colmap.background` module
- **DWPose skeleton style** (`Renderer.render_skeleton(style="dwpose")`, or
  `{"skeleton": {"style": "dwpose"}}` in a composite): reproduces the convention Wan 2.2
  VACE's pose maps are actually drawn in, rather than approximating it. Renderer-level
  for now — not yet exposed on the CLI.
  - **Why**: measured against a real DWPose render of the same MHR70 skeleton through
    one camera at 720x1280, ours agreed on almost nothing — limbs ~4-5 px against
    DWPose's ~7, full brightness against its `canvas * 0.6` (mean lit luminance 122 vs
    66), and 10 of 17 limbs the wrong hue
  - The hues were wrong **structurally**, not by typo: BODY_25 routes the torso through
    MidHip and BODY_18 does not, so our palette spent two colour slots DWPose never
    spends. The upper body matched exactly and everything from the hips down, plus the
    whole head, was one step out
  - The style picks **connectivity, not just colour**: `get_skeleton_bones_dwpose()`
    builds its own 57-bone list rather than filtering the Body25 one, because two of
    its bones are not in that list. `limbSeq[:17]` means no feet (DWPose detects them
    and draws none) and no MidHip, so neck->RHip and neck->LHip run whole across the
    chest
  - Body limbs dimmed to 60% with the joint dots left **undimmed** — the order
    `draw_bodypose` does it in, and the only thing making the dots read. Hands at a
    quarter width under `hsv_to_rgb([ie / 20, 1, 1])`, undimmed because `draw_handpose`
    runs after the 0.6 pass
  - `get_joint_colors_dwpose()` returns None for the 7 of 65 joints DWPose has no
    keypoint for, and the renderer skips the sphere it would otherwise draw there — a
    dot with no bone on it reads as a speck, not a keypoint
  - **Gotcha worth knowing separately**: pyrender's fragment shader ends on
    `pow(color.xyz, vec3(1.0/2.2))`, so every vertex colour handed to it comes back
    lifted — a nominal 85 renders as 155, and this project's skeletons have always
    carried that. Asking for 0.6 got 0.79. `_pyrender_rgba(linearize=True)` cancels it,
    but only when a style asks: the older `openpose` style keeps its lift deliberately,
    so it stays an unchanged "before" to compare against
- **Backdrop fade around the subject** (`--background-fade`): fades the backdrop toward
  a flat tone in a shell around the subject, so an `outline` frame keeps its rotation
  cue in the far field without presenting the silhouette as a hard boundary. Off by
  default.
  - **Why**: the backdrop that fixes one failure causes another. With a grid running
    right up to the silhouette, VACE reads the outline as a hard occlusion boundary and
    refuses to paint past it — bulky clothing and hair get squashed back onto the shape
    of the bare mesh. A structure-free zone next to the silhouette gives it room
  - The clear zone is the projection of a **minimum-volume ellipsoid fitted to the mesh
    vertices**, not of any one frame's outline: an ellipsoid enclosing the mesh encloses
    its silhouette from every viewpoint, so the zone can never fall inside the outline
    partway round the orbit. Enclosure is imposed exactly after the solve, so the fit
    can run on a subsample without risking a clipped vertex
  - Seven decay profiles: `step` (the hard-edged control), `linear`, `smoothstep`
    (default), `cosine`, `exponential`, `gaussian`, `inverse_square`. The first four
    reach zero at the band edge; the last three have tails and take
    `--background-fade-rate`
  - `--background-fade-falloff` is a **multiple of the subject's own radius**, not a
    pixel count, so one setting holds across an auto-framed orbit
  - **The lines fade to the wall, not into a blur.** `--background-fade-target plain`
    (the default) renders the backdrop twice through one set of sampling maps — once
    normally, once from the same generator with its pattern suppressed — so inside the
    clear zone the line colour becomes the wall colour that was behind it, while the
    shading, the corners and the floor/ceiling split stay sharp. `color` uses one flat
    colour; `blur` averages the backdrop into itself, which spreads each line into a
    grey band rather than removing it, and exists only because a loaded texture has no
    pattern-free variant
  - Each built-in generator gained a `flat=True` form (grid without lines, checker as
    its mean tone, sky without its sun), which is what the fade dissolves into
  - `--background-fade-margin` inflates the fitted ellipsoid, for when the mesh is a
    bare body and the subject to be generated is not
  - Needs a backdrop; `--background-fade` on its own is rejected rather than ignored
  - Config: the `background.fade:` section. CLI: `--background-fade`,
    `--no-background-fade`, `--background-fade-falloff`, `--background-fade-rate`,
    `--background-fade-margin`, `--background-fade-target`, `--background-fade-color`,
    `--background-fade-detail`
  - API: `OrbitPipeline.configure_background_fade()`, `clear_background_fade()`, and the
    `body2colmap.fade` module (`Ellipsoid`, `SubjectFade`, `DECAY_PROFILES`)
- `orbit_params['target']`: the orbit's look-at point, so a finite backdrop can be
  centred on the subject rather than on the world origin
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
