# Body2COLMAP

Generate synthetic multi-view training data for 3D Gaussian Splatting from SAM-3D-Body output.

## Overview

Body2COLMAP is a command-line tool and Python library that converts single-image 3D body reconstructions into multi-view synthetic datasets suitable for 3D Gaussian Splatting training.

**Input**: 3D mesh from SAM-3D-Body (`.npz` file) or Gaussian Splat (`.ply` file)
**Output**: Multi-view images + COLMAP camera parameters

### What it does

1. **Loads** 3D mesh reconstruction from SAM-3D-Body
2. **Generates** camera orbit paths (circular, sinusoidal, or helical)
3. **Renders** multi-view images (mesh, depth, skeleton, face landmarks)
4. **Exports** COLMAP format camera parameters and point cloud
5. **Outputs** data ready for 3D Gaussian Splatting training

## Installation

```bash
# Clone repository
git clone https://github.com/Erant/body2colmap.git
cd body2colmap

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Requirements

- Python 3.10+
- numpy
- pyrender
- trimesh
- opencv-python

For face landmark extraction (optional):
- mediapipe (`pip install mediapipe`)

For Gaussian splats — a `.ply` input, or `--splat-overlay` (optional):
- plyfile (`pip install body2colmap[splat]`)
- the `brush-splat-render` binary, built from the
  [brush](https://github.com/ArthurBrussee/brush) repo. It is wgpu/Vulkan, so
  it needs a GPU but no CUDA toolchain. See
  [Rendering splats](#rendering-splats).

## Quick Start

### Command Line

```bash
# Basic usage — mesh rendering with COLMAP export
body2colmap estimation.npz --output-dir ./output

# With skeleton overlay
body2colmap estimation.npz --output-dir ./output \
  --skeleton --render-modes depth+skeleton

# Flat outline of the mesh with a skeleton overlay
body2colmap estimation.npz --output-dir ./output \
  --skeleton --render-modes outline+skeleton \
  --outline-color 1,1,1 --outline-bg-color 0,0,0

# With face landmarks from a photo of the subject
python tools/extract_face_landmarks.py photo.jpg -o face.json
body2colmap estimation.npz --output-dir ./output \
  --face-landmarks face.json --render-modes skeleton+face

# With a real Gaussian-splat face instead of synthetic landmarks
body2colmap estimation.npz --output-dir ./output \
  --config circular-splat.yaml \
  --splat-overlay face_splat.ply --splat-crop 141,0,594,477
```

### Python API

```python
from body2colmap import OrbitPipeline

# Create pipeline
pipeline = OrbitPipeline.from_npz_file("estimation.npz")

# Auto-orient body to face camera (default, no offset)
pipeline.auto_orient()

# Configure orbit
pipeline.set_orbit_params(pattern="helical", n_frames=120)

# Render frames
images = pipeline.render_all(modes=["mesh"])

# Outline mode takes its options as render_kwargs
outlines = pipeline.render_all(
    modes=["outline"],
    outline_color=(0.0, 0.0, 0.0),      # foreground (the mesh)
    outline_bg_color=(1.0, 1.0, 1.0),   # background
    outline_style="filled",             # or "stroke"
    outline_thickness=3,                # stroke width in px
    outline_blur=4,                     # edge blur radius in px, 0 = hard
)["outline"]

# Composites take them under an "outline" base layer, with shorter keys
composites = pipeline.render_composite_all({
    "outline": {"fg_color": (1, 1, 1), "bg_color": (0, 0, 0), "blur": 4},
    "skeleton": {"joint_radius": 0.015, "bone_radius": 0.008},
})

# Export
pipeline.export_colmap("./output")
pipeline.export_images("./output", images["mesh"])
```

Attaching a Gaussian-splat face overlay (see
[Gaussian-Splat Overlay](#gaussian-splat-overlay)). The splat must come from the
same photo as the `.npz`, and the scene must NOT be auto-oriented:

```python
from body2colmap import OrbitPipeline, Scene

meta = Scene.load_npz_metadata("estimation.npz")
focal = float(meta["focal_length"])
h, w = meta["img_shape"]                     # note: (height, width)

pipeline = OrbitPipeline.from_npz_file(
    "estimation.npz", render_size=(720, 1280), include_skeleton=True
)
# original_focal_length puts one orbit frame on the photo's own viewpoint
pipeline.set_orbit_params(
    pattern="circular", n_frames=72, original_focal_length=focal
)
pipeline.attach_splat_overlay(
    ply_path="face_splat.ply",
    meta_path="splat_meta.json",
    original_focal_length=focal,
    original_image_size=(int(w), int(h)),
    crop_box=(141, 0, 594, 477),   # region of the photo the splat was built from
    scale=None,                    # None = fit the depth gauge against the mesh
    max_angle_deg=45.0,
)

frames = pipeline.render_composite_all({
    "skeleton": {"joint_radius": 0.006, "bone_radius": 0.003},
})
pipeline.export_images("./output", frames)

print(pipeline.splat_overlay_params["scale"])       # the fitted depth gauge
print(pipeline.splat_view_angle_deg(pipeline.cameras[0]))  # 0.0 at the anchor
```

`render_composite_all()` and `render_original_view()` pick the splat layer up
automatically once it is attached; no `"splat"` key is needed in the modes dict.

## Features

### Orbit Patterns

- **Circular**: Fixed elevation, rotating azimuth (turntable)
- **Sinusoidal**: Oscillating elevation for dynamic views
- **Helical**: Multiple loops with elevation change (best for 3DGS)

### Render Modes

Single modes:
- **mesh**: Colored mesh with lighting
- **depth**: Depth maps (with optional colormaps)
- **outline**: Flat two-tone silhouette of the mesh (no shading)
- **skeleton**: Skeleton joints and bones
- **splat**: The Gaussian splat itself. Only for a `.ply` input, where it is
  the whole scene and the only valid mode; see
  [Rendering splats](#rendering-splats).

Composite modes (overlays combined via `+`):
- **depth+skeleton**: Depth map with skeleton overlay
- **outline+skeleton**: Flat silhouette with skeleton overlay
- **skeleton+face**: Skeleton with face landmark overlay
- **depth+skeleton+face**: All three combined
- **skeleton+splat**: Skeleton with a real Gaussian-splat face on top
  (requires `--splat-overlay`; see [Gaussian-Splat Overlay](#gaussian-splat-overlay))

The first layer is the base and the rest are overlays, drawn in the order
given. Base layers are `mesh`, `depth`, `outline` and `skeleton`; overlays are
`skeleton`, `face` and `splat`. Unknown, repeated or misplaced layer names are
rejected up front rather than failing during rendering.

#### Outline Mode

`outline` renders the mesh as a single flat color with no lighting or shading,
so the only information in the image is the shape of the silhouette. Both
colors are configurable, and the silhouette can be drawn solid or as a
boundary stroke:

```bash
# Solid black figure on white (default)
body2colmap estimation.npz --output-dir ./out --render-modes outline

# White figure on dark blue, with a skeleton overlay
body2colmap estimation.npz --output-dir ./out \
  --skeleton --render-modes outline+skeleton \
  --outline-color 1,1,1 --outline-bg-color 0.08,0.08,0.16

# Hard-edged silhouette (blur off)
body2colmap estimation.npz --output-dir ./out \
  --render-modes outline --outline-blur 0

# Line-art contour instead of a solid fill
body2colmap estimation.npz --output-dir ./out \
  --render-modes outline --outline-style stroke --outline-thickness 4
```

| Option | Default | Description |
|--------|---------|-------------|
| `--outline-color R,G,B` | `0,0,0` | Foreground (mesh) color, floats 0-1 |
| `--outline-bg-color R,G,B` | `1,1,1` | Background color, floats 0-1 |
| `--outline-style {filled,stroke}` | `filled` | Solid silhouette, or boundary band only |
| `--outline-thickness PIXELS` | `3` | Stroke width; only used with `--outline-style stroke` |
| `--outline-blur PIXELS` | `4` | Blur radius softening the outline edge; `0` for hard edges |

The alpha channel marks mesh coverage (as in `mesh` and `depth` modes), so
outline renders can be used as training masks and as composite base layers.

The blur softens both color and alpha together, and is applied to the outline
only — in `outline+skeleton` the skeleton is composited on top afterwards and
stays sharp.

### Environment Backdrop

`--background` draws a **world-fixed environment** — the inside of a sphere or
a cube — behind the render. It exists to break a specific failure: with a blank
background, a video diffusion model tends to read an orbit as *the subject
rotating* rather than the camera moving around it, and prompt conditioning is
not strong enough to correct it. A backdrop that sweeps past as the camera
moves supplies the missing cue.

```bash
# The default: a room, with ruled walls on a cube three times the orbit radius
body2colmap estimation.npz --output-dir ./out \
  --skeleton --render-modes outline+skeleton --background grid

# Blender's default sky, on a sphere at infinity
body2colmap estimation.npz --output-dir ./out \
  --skeleton --render-modes outline+skeleton \
  --background blender_sky --background-geometry sphere --background-infinite

# Your own panorama
body2colmap estimation.npz --output-dir ./out \
  --render-modes outline --background ~/hdri/studio_4k.jpg \
  --background-geometry sphere

# A directory of six cube faces (px/nx/py/ny/pz/nz)
body2colmap estimation.npz --output-dir ./out \
  --render-modes outline --background ./skybox
```

#### Picking a texture

**A Nishita-style sky is azimuthally symmetric apart from its sun.** Rotating
the camera about the vertical axis changes nothing else in frame — which is
exactly the motion the backdrop is meant to make legible. The sun is doing all
of the work there, and a smooth gradient does none at all. `grid` is the
default for the opposite reason: ruled walls, corners and a floor and ceiling
that read apart give the camera something to pass at every azimuth.

Measured as per-latitude standard deviation in 8-bit levels (pinned in
`tests/test_background.py`):

| Texture | Azimuthal signal | Notes |
|---------|-----------------|-------|
| `checker` | 82 | Ugly on purpose. If an orbit does not read as an orbit over a checker, the problem is the camera path, not the backdrop |
| `grid` | 27 | Ruled walls, darker floor, lighter ceiling. The strongest realistic cue, especially on a cube |
| `blender_sky` | 0.7 avg (11 at the sun) | Approximation of Blender's default Sky Texture |
| `gradient` | 0 | A control with no rotation cue whatsoever |

Loaded textures are whatever you give them; a panorama with landmarks around
the horizon behaves like `grid`, a clear sky behaves like `blender_sky`.

#### Sphere or cube, and how far away

The default surface is a **cube at 3x the orbit radius**. A finite radius means
the ray is intersected properly, which yields real parallax between the subject
and the backdrop — and that is what makes a cube read as an actual room, with
corners and wall perspective that move correctly.
`--background-radius-scale` sizes it as a multiple of the orbit radius, so it
still fits when the orbit is auto-framed; the camera must end up inside, so it
has to exceed 1.0. `--background-radius` gives it in world units instead, and
supersedes the scale.

`--background-infinite` puts the surface at **infinity**: the lookup then
depends only on ray direction, so the backdrop tracks camera rotation but not
camera translation. That is correct for a distant sky, and it is also the case
where `sphere` and `cube` differ only in how the texture is laid out — not in
what gets rendered. A cube at infinity has no corners, so pair it with
`--background-geometry sphere` and a sky.

| Option | Default | Description |
|--------|---------|-------------|
| `--background TEXTURE` | off | Generator name (`grid`, `checker`, `blender_sky`, `gradient`) or a path |
| `--background-geometry {sphere,cube}` | `cube` | Surface the texture is mapped onto |
| `--background-radius UNITS` | — | Surface radius in world units; supersedes the scale below |
| `--background-radius-scale FACTOR` | `3.0` | Radius as a multiple of the orbit radius; must be > 1 |
| `--background-infinite` | off | Put the surface at infinity: rotation but no parallax, and no corners on a cube |
| `--background-rotation DEGREES` | `0` | Turn the environment about the vertical axis, e.g. to aim the sun |
| `--background-resolution PIXELS` | `1024` | Generated texture size; ignored for a loaded one |
| `--background-keep-alpha` | off | Fill only RGB, leaving the silhouette alpha as a mask |
| `--no-background` | — | Disable a backdrop enabled by a config file |

Accepted texture files: for a sphere, a 2:1 equirectangular image. For a cube,
a directory of six faces (`px`/`nx`/`py`/`ny`/`pz`/`nz`, `posx`/`negx`/..., or
`right`/`left`/`top`/`bottom`/`front`/`back`), a 4:3 horizontal cross, a 6:1
strip, a 1:6 column, or a 2:1 equirectangular image resampled onto the cube.

#### Fading the backdrop around the subject

The backdrop that fixes one failure causes another. In `outline` modes a grid
that runs right up to the silhouette reads to a video model as a **hard
occlusion boundary**: it will not paint outside the outline, so bulky clothing
and hair get squashed back onto the shape of the bare mesh.

`--background-fade` clears the backdrop in a shell around the subject. The
rotation cue survives in the far field, and there is structure-free room next
to the silhouette to expand into.

```bash
# The default shape, on the default backdrop
body2colmap estimation.npz --output-dir ./out \
  --skeleton --render-modes outline+skeleton \
  --background grid --background-fade smoothstep

# A tighter halo, and room for clothing the bare mesh does not have
body2colmap estimation.npz --output-dir ./out \
  --render-modes outline --background grid \
  --background-fade gaussian --background-fade-falloff 0.4 \
  --background-fade-margin 1.2
```

The clear zone is the projection of an **ellipsoid fitted to the mesh**, not of
any one frame's outline. An ellipsoid that encloses the mesh encloses its
silhouette from every viewpoint, so the clear zone can never fall inside the
outline partway round the orbit — and being one fixed world-space object, it is
a region the camera moves around rather than a screen effect that swims.
`--background-fade-margin` inflates it, which is the knob to reach for when the
mesh is a bare body and the subject you want generated is not.

`--background-fade-falloff` sets how wide the fade band is, as a **multiple of
the subject's own radius** rather than a pixel count, so one setting holds
across an auto-framed orbit. 1.0 means the backdrop is fully back by twice the
subject's extent.

| Profile | Shape |
|---------|-------|
| `step` | Hard cut at the band edge. The control condition — a plain hole |
| `linear` | Straight ramp, with a visible slope break at each end |
| `smoothstep` | Hermite ramp, flat at both ends. The default |
| `cosine` | Raised cosine; like smoothstep but steeper through the middle |
| `exponential` | Steepest right at the silhouette, then a long thin tail |
| `gaussian` | Flat at the silhouette, then falls away |
| `inverse_square` | The heaviest tail: still 2.7% faded at three band widths out, which reads as a faint wash over the whole frame |

The first four reach zero exactly at the band edge; the last three have tails
that never quite do, and take `--background-fade-rate` to tighten them.

By default the backdrop fades to its own **local tone** — averaged down below
the texture's own frequency, so the lines disappear but the wall/floor/ceiling
shading carries through and the clear zone has no edge against its surroundings.
`--background-fade-target color` uses one flat colour instead (the texture's
mean, or `--background-fade-color`), which is easier to reason about but leaves
a visible patch wherever it crosses a cube's floor/wall seam.

| Option | Default | Description |
|--------|---------|-------------|
| `--background-fade PROFILE` | off | Enable the fade with this decay profile |
| `--background-fade-falloff FACTOR` | `1.0` | Band width, as a multiple of the subject's radius |
| `--background-fade-rate K` | `4.0` | Shape constant for `exponential`, `gaussian`, `inverse_square` |
| `--background-fade-margin FACTOR` | `1.0` | Inflate the fitted ellipsoid before measuring |
| `--background-fade-target {local,color}` | `local` | Fade to the backdrop's own tone, or to one flat colour |
| `--background-fade-color R,G,B` | texture mean | Flat colour; implies `--background-fade-target color` |
| `--background-fade-detail PIXELS` | `24` | Resolution the backdrop is averaged down to for `local` |
| `--no-background-fade` | — | Disable a fade enabled by a config file |

Generator parameters go in the config file, since they vary per texture:

```yaml
background:
  enabled: true
  texture: "blender_sky"
  geometry: "sphere"
  radius_scale: null            # a sky belongs at infinity
  params: {sun_azimuth_deg: 40, sun_elevation_deg: 25, sun_size_deg: 6}
```

Writing `radius` on its own is enough to override the defaulted `radius_scale`;
setting both is an error. Writing either as `null` asks for a backdrop at
infinity.

#### Python API

`configure_background()` carries the same defaults, and stores the settings
rather than building the backdrop immediately — a finite radius is measured
against the orbit, which `set_orbit_params()` has not established yet. Call
either one first.

```python
pipeline = OrbitPipeline.from_npz_file("estimation.npz", include_skeleton=True)
pipeline.set_orbit_params(pattern="circular", n_frames=72)

# The default: a grid cube at 3x the orbit radius
pipeline.configure_background()

# A sky at infinity instead. radius_scale=None is the explicit opt-out; leaving
# it unmentioned would apply the default finite radius.
pipeline.configure_background(
    texture="blender_sky",
    geometry="sphere",
    radius_scale=None,
    params={"sun_azimuth_deg": 40, "sun_elevation_deg": 25},
)

# A radius in world units. Passing it supersedes the defaulted scale; passing
# both raises.
pipeline.configure_background(texture="grid", radius=8.5)

pipeline.clear_background()   # back to no backdrop at all
```

Every render path picks the backdrop up on its own — `render_all()`,
`render_composite_all()` and `render_original_view()` — so nothing else in the
call sequence changes.

#### Scope

These are **conditioning frames**. The backdrop is not exported to COLMAP, adds
no points to the point cloud, and never enters the depth buffer or the
silhouette mask — so an `outline` render still measures the mesh, not the sky.
By default alpha is forced opaque; `--background-keep-alpha` keeps the
silhouette usable as a training mask instead.

`.npz` input only. A `.ply` is rasterized by `brush-splat-render`, which
composites against a flat colour of its own, so there is no base layer to draw
behind; the combination is rejected rather than silently ignored.

### Gaussian-Splat Overlay

`skeleton+splat` composites a **real Gaussian splat of the subject's face** onto
the skeleton, in place of the synthetic landmarks of `skeleton+face`. Landmark
dots carry almost no identity and the canonical face model carries none at all;
a splat of the actual face carries both identity and gaze, which is the point
when these frames condition a video model.

The splat is produced **externally** and fed in as a standard 3DGS `.ply` plus
its `splat_meta.json`. Nothing about building it is implemented here.

#### What you need

| Input | Notes |
|-------|-------|
| `splat.ply` | Standard 3DGS binary PLY, **SH degree 0**. Higher orders are rejected. |
| `splat_meta.json` | Must contain `intrinsics.{f,cx,cy}`, `centroid`, `width`, `height`. |
| The SAM-3D-Body `.npz` | Must contain `focal_length`, and `img_shape` unless you pass `--splat-image-size`. |

Both artifacts must come from **the same photograph**. The splat's producer must
define its world as the source camera's frame flipped by `diag(1,-1,-1)` and
recentred on `centroid`, i.e. the source camera sits at `-centroid` with
identity OpenGL rotation. That is what `~/Projects/masktest` emits.

Crop the photo to the head before building the splat — the upstream models run
at a fixed resolution, so cropping spends that budget on the face — then tell
body2colmap where the crop came from with `--splat-crop x0,y0,x1,y1` in
full-image pixels. Omit it if the splat used the whole photo.

#### How it anchors

There is no registration step. Both coordinate systems are Y-up with +Z toward
the viewer, and in both the source camera has identity rotation *by
construction*, so placing the splat is one linear map:

```
P_world = M @ (p_ply + centroid)
```

`M` is upper-triangular and reconciles the two independently recovered focal
lengths, plus one scale gauge. Consequences worth knowing:

- **2-D alignment at the anchor frame is exact by construction**, not fitted.
- The two focals usually *disagree* — the splat's is inferred from image content
  by its own network, and on a tight face crop that is not the real camera's
  (measured: 1122 px vs SAM-3D-Body's 1509 px, a 26% gap). Reconciling them is
  therefore load-bearing, not cosmetic. Set `reconcile_intrinsics: false` to
  place the splat with a uniform scale instead, preserving its shape exactly at
  the cost of rendering the face off-size.
- **The scale gauge is invisible at the anchor frame.** Scaling about the camera
  centre leaves every projection unchanged. What it sets is how far along the
  view rays the splat sits, and therefore whether the face and the skeleton stay
  together as the orbit turns away. Leave `scale: null` to fit it against the
  mesh's depth buffer; a wrong value shows up as drift growing with view angle.

#### The angle cull

The splat is a 2.5-D shell — one view, nothing behind the subject — so as the
camera turns away the open rim swings into view. Frames more than
`--splat-max-angle` degrees off the splat's source view drop the layer entirely.

The default of **45°** is measured: the face reads cleanly to about 30°, the rim
starts flaring by 45°, and by 60° the shell is mostly edge. On a 72-frame full
circle that keeps the face on roughly 20 frames, centred on the anchor. Raise it
if you would rather have coverage than a clean silhouette.

#### Constraints

- Use with `--use-original-camera` (or `original_focal_length=` in the Python
  API). Otherwise no orbit frame sits at the photo's viewpoint and the anchor,
  though still geometrically correct, buys you nothing.
- **Incompatible with auto-orient.** `auto_orient()` rotates the scene about its
  bbox centre, moving the subject out of the original camera's frame that the
  anchor is defined against. `attach_splat_overlay()` raises rather than
  silently misplacing the face. `--use-original-camera` already skips
  auto-orient.
- The splat contributes to the composite's alpha channel (it is real subject
  coverage, like the mesh silhouette). The skeleton does not — it is an
  annotation.
- A `.ply` **input** is a different feature: there `splat` is the base layer and
  the whole scene is rendered as a splat. That cannot be combined with an
  overlay, and body2colmap raises if you try.

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--splat-overlay PLY` | None | The 3DGS `.ply`. Enables the `splat` overlay layer. |
| `--splat-meta JSON` | `splat_meta.json` beside the PLY | Fitted intrinsics and centroid |
| `--splat-crop X0,Y0,X1,Y1` | None | Region of the photo the splat's input was cut from, in full-image pixels |
| `--splat-image-size WxH` | from `img_shape` in the `.npz` | Size of the full photo SAM-3D-Body saw |
| `--splat-scale S` | fitted | Depth gauge; omit to fit it against the mesh |
| `--splat-max-angle DEGREES` | `45` | Cull past this far off the splat's source view |
| `--splat-no-reconcile` | off | Place with a uniform scale, ignoring the splat's own focal |
| `--splat-renderer PATH` | `$BRUSH_SPLAT_RENDER`, then `PATH` | The `brush-splat-render` binary |

```yaml
splat:
  overlay_ply: "face_splat.ply"
  meta_json: null                 # null = splat_meta.json beside the PLY
  original_image_size: null       # null = img_shape from the .npz
  crop_box: [141, 0, 594, 477]    # null if the splat used the whole photo
  scale: null                     # null = fit against the mesh
  reconcile_intrinsics: true
  max_angle_deg: 45.0
  renderer_binary: null           # null = $BRUSH_SPLAT_RENDER, then PATH
```

A ready-to-run example is in [`circular-splat.yaml`](circular-splat.yaml):

```bash
body2colmap estimation.npz -o ./out --config circular-splat.yaml \
  --splat-overlay face_splat.ply --splat-crop 141,0,594,477
```

#### Verifying an anchor

The splat, rendered from the original camera, must land on the photograph. The
test is the **best-fit integer shift**, not PSNR: ordinary splat blur costs
PSNR, but a wrong quaternion order, a flipped axis or a bad crop box all
*displace* the render.

```python
camera = Camera(focal_length=(focal, focal), image_size=(w, h))
layer = pipeline.render_splat_layer(camera)   # pipeline render_size must be (w, h)
# compare layer's RGB against the photo inside layer's alpha,
# searching integer shifts over +/-4 px -- the optimum must be (0, 0)
```

Requires `pip install body2colmap[splat]` (plyfile) and the
`brush-splat-render` binary — see [Rendering splats](#rendering-splats).

### Rendering splats

Both splat paths — a `.ply` input rendered as the base layer, and the
`--splat-overlay` face composite — rasterize through **`brush-splat-render`**,
a standalone binary from the [brush](https://github.com/ArthurBrussee/brush)
repo. It is wgpu/Vulkan, so it needs a GPU but no CUDA toolchain, and there is
deliberately no Python fallback.

```bash
# in your brush checkout
cargo build --release -p brush-splat-render
export BRUSH_SPLAT_RENDER=$PWD/target/release/brush-splat-render
```

body2colmap looks for it in `--splat-renderer` / `splat.renderer_binary`, then
`$BRUSH_SPLAT_RENDER`, then `PATH`.

The renderer currently has an intermittent crash that can kill it *after* it has
written every frame. body2colmap judges a run by the files it produced rather
than by its exit status, so such a run succeeds and logs a warning; a run that
actually lost frames fails, naming the missing ones.

The binary renders a whole camera list per invocation, so a sequence is
rendered in one call — `render_all(modes=["splat"])` and
`render_composite_all()` both batch internally. Use
`pipeline.render_splat_layers(cameras)` rather than a loop over
`render_splat_layer()` if you drive the API yourself.

#### Confidence gating

`--splat-confidence` gates each pixel by how well the training views actually
constrained the Gaussians covering it, instead of leaving the decision to a
threshold on rendered alpha. It drops low-confidence fringes at the source.

**It changes what the alpha channel means:** alpha becomes the confidence gate,
not accumulated opacity. A mask taken from such a frame masks on evidence
rather than coverage.

The background also comes from the renderer rather than being composited
afterwards, because culled pixels and the background resolve to the same
colour. `--bg-color` still sets it; pass `--splat-cull-color` only when you
want culled regions to stand out against the background for inspection.

It needs evidence — either `ev_*` properties baked into the `.ply` by
`brush ... --export-evidence`, or `--splat-confidence-dataset` pointing at the
training set so it can be measured at render time. It is available only for a
`.ply` input: an overlay splat is reconstructed from a single photograph, so
there are no training views to score against, and `--splat-overlay` combined
with `--splat-confidence` is rejected rather than silently degraded.

| Option | Default | Description |
|--------|---------|-------------|
| `--splat-confidence` | off | Enable confidence gating |
| `--splat-confidence-dataset DIR` | None | Measure evidence here if the `.ply` carries none |
| `--splat-gate-lo C` | `0.45` | At or below this confidence, a pixel is fully culled |
| `--splat-gate-hi C` | `0.65` | At or above this, fully kept (equal to `--splat-gate-lo` = hard cut) |
| `--splat-cull-color R,G,B` | follows `--bg-color` | What culled pixels resolve to — and, since the renderer uses one colour for both, the whole background of a gated render |
| `--splat-confidence-sidecar` | off | Also write `<frame>.conf.png`, the raw confidence before gating |

```bash
body2colmap scene.ply -o ./out --render-modes splat \
  --splat-confidence --splat-confidence-dataset ./colmap_dataset
```

### Auto-Orient

By default, the body is automatically rotated to face the camera at frame 0 of the orbit. The facing direction is computed from the skeleton's shoulder and hip joints (averaged torso normal, projected to the horizontal XZ plane).

Use `--initial-rotation` to add an offset from the auto-facing position:

```bash
# Default: body faces camera
body2colmap estimation.npz --output-dir ./output

# Body's right side toward camera
body2colmap estimation.npz --output-dir ./output --initial-rotation 90

# Back toward camera
body2colmap estimation.npz --output-dir ./output --initial-rotation 180
```

This ensures consistent starting orientation regardless of how the subject was posed in the source image, giving predictable control over when features appear and disappear during the orbit.

### Original-Camera Orbit (`--use-original-camera`)

When you have the original input image, `--use-original-camera` generates an orbit where **one frame matches the SAM-3D-Body camera viewpoint**. This is useful for diffusion-based pipelines where the input image serves as a conditioning frame anchoring the generation to a known-good view.

For circular orbits that frame is frame 0. For **helical** orbits the elevation ramp determines where in the sequence the original viewpoint can occur, so the index is solved for — read it from `orbit_params['anchor_frame_index']` rather than assuming 0 (the CLI names its debug output `frame{index}_warped.png` accordingly). Sinusoidal orbits are not anchored.

```bash
# Original-camera orbit with the source image composited at frame 0
body2colmap estimation.npz --output-dir ./output \
  --use-original-camera --original-image photo.jpg \
  --render-modes skeleton

# Adjust how much of the frame the subject fills (default: 0.8)
body2colmap estimation.npz --output-dir ./output \
  --use-original-camera --original-image photo.jpg \
  --fill-ratio 0.6 --render-modes mesh
```

How it works:

1. The orbit radius is derived from the geometric distance between the origin (SAM-3D-Body camera) and the mesh bbox center, so the anchor frame lands exactly at the original camera position via a spherical-coordinate roundtrip
2. All cameras use `look_at()` with a centered principal point and an auto-framed focal length that zooms the subject to fill the viewport
3. For helical orbits, the start azimuth is solved so the anchor frame hits the original azimuth exactly, and a sub-degree uniform elevation offset is applied to the whole helix so it also hits the original elevation. The path stays a smooth helix with no discontinuity at the anchor
4. The original image is warped via a homography to align with the anchor frame's look-at view, accounting for both the focal-length zoom and the slight rotation correction

If the original camera sits outside the helix's elevation band (`±helical_amplitude_deg`), the run fails with an error telling you how far to raise the amplitude, rather than producing a degenerate path.

**Important**: Do not use `--initial-rotation` / `auto_orient()` with `--use-original-camera`. The orbit geometry depends on the unrotated mesh position to place frame 0 at the origin.

#### Python API

```python
import numpy as np
from body2colmap import OrbitPipeline

pipeline = OrbitPipeline.from_npz_file("estimation.npz")
# Do NOT call pipeline.auto_orient() in original-camera mode

# original_focal_length triggers original-camera orbit
pipeline.set_orbit_params(
    pattern="helical",
    n_frames=81,
    n_loops=2,
    amplitude_deg=40.0,
    original_focal_length=500.0,  # from .npz file
    fill_ratio=0.8,
)

# Which rendered frame corresponds to the original camera
k = pipeline.orbit_params['anchor_frame_index']

# The orbit_params dict contains the warp homography for the input image
H = pipeline.orbit_params['warp_homography']  # 3x3 numpy array
# Use with: cv2.warpPerspective(original_image, H, (w, h))
# The result aligns with rendered frame k, ready to inject as a conditioning frame

images = pipeline.render_all(modes=["mesh"])
pipeline.export_colmap("./output")
pipeline.export_images("./output", images["mesh"])
```

### Face Landmark Rendering

Face landmarks render the OpenPose Face 70 keypoint topology (jawline, eyebrows, nose, eyes, lips, pupils) as white points and connecting lines on top of the skeleton.

The eyes are the exception: instead of a ring of dots, each eye is drawn as a filled shape with a pupil disc inside it, which conditions gaze far more strongly in a video diffusion model. Use `--eye-color` and `--pupil-color` to set the two colors, and `--pupil-scale` to set the pupil size as a fraction of the eye height — `1.0` is a pupil that touches the upper and lower lid, and values above that are rejected.

Pass `--eye-style dots` to render the eyes as plain landmark dots and an outline instead, which is what earlier versions did.

Two modes of operation:

1. **Canonical face model** (no external data): Uses a generic face shape derived from MediaPipe's canonical face geometry. Good for testing; not subject-specific.

2. **Subject-specific face landmarks** (recommended): Extract landmarks from a photo of the subject using the included `tools/extract_face_landmarks.py`, then pass the resulting JSON file via `--face-landmarks`. The landmarks are automatically aligned to the skeleton via Procrustes fitting.

Face landmarks are only rendered when the face is pointing toward the camera. By default this is the frontal 180-degree hemisphere; use `--face-max-angle` to narrow the range (e.g. `--face-max-angle 45` for only near-frontal views).

See [Face Landmarks](#face-landmarks) below for the full workflow.

### Framing Presets

- **full**: Entire body visible (default)
- **torso**: Waist up (requires skeleton data)
- **bust**: Shoulders and head (requires skeleton data)
- **head**: Head only (requires skeleton data)

### Export Formats

- **COLMAP**: Standard sparse reconstruction format
  - `cameras.txt`: Camera intrinsics
  - `images.txt`: Camera extrinsics
  - `points3D.txt`: Initial point cloud
- **Images**: PNG with alpha channel

## Face Landmarks

### Overview

Body2COLMAP can render face landmarks on top of the skeleton. This is a two-step process:

1. **Extract** face landmarks from a reference photo using `tools/extract_face_landmarks.py`
2. **Render** by passing the JSON file to `body2colmap --face-landmarks`

The extraction tool uses MediaPipe FaceLandmarker to detect 478 facial landmarks, which are then converted to the OpenPose Face 70 keypoint format internally.

### Step 1: Extract Face Landmarks

```bash
# Install mediapipe (one-time)
pip install mediapipe

# Extract landmarks from a photo
python tools/extract_face_landmarks.py photo.jpg -o face_landmarks.json

# With options
python tools/extract_face_landmarks.py photo.jpg \
  -o face_landmarks.json \
  --min-confidence 0.3 \
  --save-crop face_crop.jpg  # saves the detected face region for verification
```

On first run, the tool downloads two small model files (~4MB total) to `~/.cache/body2colmap/`.

The tool uses a two-stage detection pipeline:
- First tries MediaPipe FaceLandmarker on the full image (works for selfies/headshots)
- If no face is found, falls back to MediaPipe FaceDetector to locate the face bounding box, crops to it, then re-runs FaceLandmarker on the crop
- If multiple faces are detected, selects the most frontal one

### Step 2: Render with Face Landmarks

```bash
# Skeleton + face overlay
body2colmap estimation.npz --output-dir ./output \
  --face-landmarks face_landmarks.json \
  --render-modes skeleton+face

# Depth + skeleton + face
body2colmap estimation.npz --output-dir ./output \
  --face-landmarks face_landmarks.json \
  --render-modes depth+skeleton+face

# Face points only (no connecting lines)
body2colmap estimation.npz --output-dir ./output \
  --face-landmarks face_landmarks.json \
  --face-mode points \
  --render-modes skeleton+face
```

Providing `--face-landmarks` automatically enables face rendering (`--face-mode full`). You can override this with `--face-mode points` or `--face-mode none`.

### Face Landmarks JSON Format

The JSON file produced by `extract_face_landmarks.py` has the following structure:

```json
{
  "version": "1.0",
  "source": "mediapipe",
  "source_image": "photo.jpg",
  "image_size": [1536, 2048],
  "n_landmarks": 478,
  "refined": true,
  "landmarks": [
    [0.432100, 0.215300, -0.023400],
    [0.445200, 0.218100, -0.031200],
    ...
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | Format version (`"1.0"`) |
| `source` | string | Landmark source, must be `"mediapipe"` |
| `source_image` | string | Original image filename (informational) |
| `image_size` | [int, int] | Source image `[width, height]` in pixels. Required for correct aspect ratio during Procrustes alignment. |
| `n_landmarks` | int | Number of landmarks (468 or 478) |
| `refined` | bool | `true` if iris landmarks are present (478 points) |
| `landmarks` | [[float, float, float], ...] | Normalized landmark coordinates `[x, y, z]` |

**Landmark coordinates**:
- `x`: Normalized to image width (0.0 = left edge, 1.0 = right edge)
- `y`: Normalized to image height (0.0 = top edge, 1.0 = bottom edge)
- `z`: Relative depth estimate (roughly same scale as x; calibrated automatically during fitting)

The `image_size` field is important: it allows `body2colmap` to denormalize coordinates correctly so that portrait and landscape images produce proper face proportions.

### CLI Options Reference

| Option | Default | Description |
|--------|---------|-------------|
| `--face-landmarks PATH` | None | Path to face landmarks JSON. Implies `--face-mode full`. |
| `--face-mode {full,points,none}` | None | `full`: points + lines, `points`: points only, `none`: disabled |
| `--face-max-angle DEGREES` | 90 | Max degrees off face normal to render. 90 = full hemisphere, 45 = only within 45 degrees of straight-on. |
| `--eye-style {shape,dots}` | `shape` | `shape`: filled eye with a pupil disc, `dots`: plain landmark dots |
| `--eye-color R,G,B` | `1,1,1` | Color of the filled eye shape |
| `--pupil-color R,G,B` | `0,0,0` | Color of the pupil disc |
| `--pupil-scale SCALE` | `0.75` | Pupil diameter as a fraction of eye height, in (0, 1]. `1.0` touches both lids. |
| `--skeleton` | off | Enable skeleton rendering (required for face) |
| `--render-modes MODES` | `mesh` | Comma-separated list, e.g. `skeleton+face,depth+skeleton+face` |
| `--outline-color R,G,B` | `0,0,0` | Outline foreground color (see [Outline Mode](#outline-mode)) |
| `--outline-bg-color R,G,B` | `1,1,1` | Outline background color |
| `--outline-style {filled,stroke}` | `filled` | Outline fill style |
| `--outline-thickness PIXELS` | `3` | Outline stroke width (`stroke` style only) |
| `--outline-blur PIXELS` | `4` | Outline blur radius; `0` disables. Never blurs the skeleton. |
| `--splat-overlay PLY` | None | Gaussian-splat face to composite on the skeleton (see [Gaussian-Splat Overlay](#gaussian-splat-overlay)) |
| `--splat-meta JSON` | beside the PLY | Splat metadata (intrinsics, centroid) |
| `--splat-crop X0,Y0,X1,Y1` | None | Region of the photo the splat's input was cut from |
| `--splat-image-size WxH` | from `.npz` | Size of the full photo SAM-3D-Body saw |
| `--splat-scale S` | fitted | Splat depth gauge; omit to fit against the mesh |
| `--splat-max-angle DEGREES` | `45` | Cull the splat past this far off its source view |
| `--splat-no-reconcile` | off | Place the splat with a uniform scale |
| `--background TEXTURE` | off | Environment backdrop: generator name or path (see [Environment Backdrop](#environment-backdrop)) |
| `--background-geometry {sphere,cube}` | `cube` | Surface the backdrop texture is mapped onto |
| `--background-radius UNITS` | — | Backdrop radius in world units; supersedes the scale below |
| `--background-radius-scale FACTOR` | `3.0` | Backdrop radius as a multiple of the orbit radius; must be > 1 |
| `--background-infinite` | off | Put the backdrop at infinity instead of the default finite radius |
| `--background-rotation DEGREES` | `0` | Turn the environment about the vertical axis |
| `--background-resolution PIXELS` | `1024` | Generated backdrop texture size |
| `--background-keep-alpha` | off | Backdrop fills RGB only, leaving the silhouette alpha as a mask |

### Config File

Face options can also be set in the YAML config file:

```yaml
skeleton:
  enabled: true
  face_mode: "full"               # "full", "points", or null
  face_landmarks: "face.json"     # path to landmarks JSON
  face_max_angle: 90.0            # degrees off face normal to render
  eye_style: "shape"              # "shape" (filled eye + pupil) or "dots"
  eye_color: [1.0, 1.0, 1.0]      # filled eye shape
  pupil_color: [0.0, 0.0, 0.0]    # pupil disc
  pupil_scale: 0.75               # pupil diameter as a fraction of eye height, (0, 1]

background:
  enabled: true
  geometry: "cube"                # "sphere" or "cube"
  texture: "grid"                 # generator name, or a path to an image/dir
  radius_scale: 3.0               # multiple of the orbit radius; null = infinite
  rotation_deg: 0.0               # turn the environment about the vertical axis
  opaque: true                    # false keeps the silhouette alpha as a mask
  params: {}                      # generator arguments, e.g. {n_per_face: 8}

splat:
  overlay_ply: "face_splat.ply"   # 3DGS .ply; null disables the overlay
  crop_box: [141, 0, 594, 477]    # region of the photo the splat came from
  scale: null                     # null = fit the depth gauge against the mesh
  max_angle_deg: 45.0             # cull past this far off the source view
```

### Writing a Custom Client

If you want to produce face landmark JSON from your own detection pipeline (not using the included tool), the minimum required fields are:

```json
{
  "source": "mediapipe",
  "image_size": [WIDTH, HEIGHT],
  "landmarks": [
    [x0, y0, z0],
    [x1, y1, z1],
    ...
  ]
}
```

Requirements:
- `source` must be `"mediapipe"` (the only format currently supported)
- `landmarks` must contain at least 468 entries in MediaPipe Face Mesh vertex order
- If 478 entries are provided, indices 468-477 are treated as iris landmarks (468 = right iris center, 473 = left iris center)
- `image_size` should be `[width, height]` of the image used for detection, so coordinates can be denormalized correctly
- Coordinates should be normalized: `x` in [0, 1] relative to width, `y` in [0, 1] relative to height, `z` as relative depth

The conversion pipeline internally:
1. Maps MediaPipe 468 indices to OpenPose Face 68 keypoints (MaixPy convention)
2. Adds pupils (indices 68-69) from iris centers or eye contour centroids
3. Denormalizes coordinates using `image_size` (x * width, y * height, z * width)
4. Calibrates z depth to match the skeleton's head geometry
5. Fits to the skeleton via Procrustes alignment (5 anchor points: nose, eyes, ears)

## Configuration

### YAML Config File

Generate a default config template:

```bash
body2colmap --save-config config.yaml
```

Use it:

```bash
body2colmap estimation.npz --config config.yaml
```

CLI arguments override config file values.

## Usage with 3D Gaussian Splatting

Body2COLMAP output is compatible with standard 3DGS training pipelines:

```bash
# Generate training data
body2colmap person.npz --output-dir ./data/person

# Train with gaussian-splatting
cd gaussian-splatting
python train.py -s ../data/person
```

## Architecture

```
body2colmap/
├── coordinates.py   # Coordinate system definitions
├── camera.py        # Camera class with intrinsics/extrinsics
├── scene.py         # 3D scene management
├── path.py          # Orbit path generation
├── skeleton.py      # Skeleton format conversion and rendering data
├── face.py          # Face landmarks, Procrustes alignment, visibility
├── renderer.py      # Image rendering (mesh, depth, skeleton, face)
├── splat_scene.py   # Gaussian splat storage and PLY I/O
├── splat_renderer.py # rasterization via brush-splat-render
├── splat_anchor.py  # Place an external splat in world coords
├── exporter.py      # COLMAP export
├── utils.py         # Auto-framing, homography warp, focal length utilities
├── pipeline.py      # High-level API
├── config.py        # Configuration management (CLI + YAML)
└── cli.py           # Command-line interface
tools/
└── extract_face_landmarks.py  # Standalone MediaPipe face extraction utility
```

### Key Design Principles

1. **Single coordinate system**: All internal computation in Y-up OpenGL/renderer coords
2. **Camera movement**: Mesh stays stationary, cameras orbit
3. **Explicit transforms**: Coordinate conversions only at system boundaries
4. **Separation of concerns**: Each module has single responsibility

## Development

### Running Tests

```bash
# All tests
pytest

# Specific module
pytest tests/test_face.py

# With coverage
pytest --cov=body2colmap
```

## Documentation

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for detailed specification and architecture notes.

For development notes, see [CLAUDE.md](CLAUDE.md).

## Contributing

Contributions welcome! Please see development notes in `CLAUDE.md` for architecture details and design principles.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- SAM-3D-Body for 3D body reconstruction
- MediaPipe for face landmark detection
- COLMAP for camera parameter format
- 3D Gaussian Splatting for the underlying technique
