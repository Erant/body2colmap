"""
Environment background rendering.

Draws a static environment — the inside of a sphere or a cube surrounding the
scene — behind a render, so that an orbiting camera sees the world sweep past.
This is the cue that tells a downstream video-diffusion model that the *camera*
is moving rather than the subject rotating on a turntable.

The background is a per-pixel direction lookup (one ``cv2.remap``), not scene
geometry.  Adding a giant inverted sphere to the pyrender scene would make
``Renderer.render_mask()`` — which derives mesh coverage from ``depth > 0`` —
report every pixel as covered, breaking ``outline`` mode and the alpha channel
along with it.  See ``body2colmap/CLAUDE.md``.

All directions are in world/renderer coordinates (Y-up, camera looks down -Z).
"""

import inspect
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .camera import Camera
from .coordinates import spherical_to_cartesian
from .fade import SubjectFade


#: Default backdrop radius, as a multiple of the orbit radius.
#:
#: The default backdrop is a grid cube, and a cube is only a *room* at a finite
#: radius -- at infinity the ray intersection drops out, the corners with it,
#: and what is left is a plain ruled field with no parallax.  3.0 puts the
#: walls comfortably clear of an auto-framed subject while keeping the wall
#: perspective legible as the camera swings around.
DEFAULT_RADIUS_SCALE: float = 3.0


#: Cube faces in canonical order, named by the axis they face along.
CUBE_FACES: Tuple[str, ...] = ("px", "nx", "py", "ny", "pz", "nz")

#: Alternative face names accepted when loading a directory of six images.
_FACE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "px": ("px", "posx", "right"),
    "nx": ("nx", "negx", "left"),
    "py": ("py", "posy", "top", "up"),
    "ny": ("ny", "negy", "bottom", "down"),
    "pz": ("pz", "posz", "front"),
    "nz": ("nz", "negz", "back"),
}

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".exr", ".hdr", ".tif", ".tiff", ".bmp")


# ---------------------------------------------------------------------------
# Ray generation
# ---------------------------------------------------------------------------

def camera_ray_directions(camera: Camera) -> NDArray[np.float32]:
    """
    Unit ray direction per pixel, in world coordinates.

    Inverts the projection in :meth:`Camera.project`, which converts OpenGL
    camera space to OpenCV (``* [1, -1, -1]``) before applying ``K``.  So a
    pixel's OpenCV ray ``((px - cx) / fx, (py - cy) / fy, 1)`` maps back to the
    OpenGL camera ray by negating Y and Z, and the camera-to-world rotation
    takes it to world space.

    Args:
        camera: Camera to generate rays for.

    Returns:
        Float32 array, shape (height, width, 3), unit length.
    """
    px = np.arange(camera.width, dtype=np.float32) + 0.5
    py = np.arange(camera.height, dtype=np.float32) + 0.5

    x_cv = (px - camera.cx) / camera.fx           # (W,)
    y_cv = (py - camera.cy) / camera.fy           # (H,)

    grid_x, grid_y = np.meshgrid(x_cv, y_cv)      # (H, W)

    # OpenCV ray -> OpenGL camera ray: negate Y and Z.
    dirs_cam = np.stack(
        [grid_x, -grid_y, -np.ones_like(grid_x)], axis=-1
    ).astype(np.float32)

    dirs_world = dirs_cam @ np.asarray(camera.rotation, dtype=np.float32).T

    norms = np.linalg.norm(dirs_world, axis=-1, keepdims=True)
    return (dirs_world / norms).astype(np.float32)


def _rotation_y(angle_deg: float) -> NDArray[np.float32]:
    """Rotation matrix about +Y by ``angle_deg`` degrees."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array(
        [[c, 0.0, s],
         [0.0, 1.0, 0.0],
         [-s, 0.0, c]],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Parameterizations: direction -> texture coordinate
# ---------------------------------------------------------------------------

def equirect_uv(vectors: NDArray[np.float32]) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Equirectangular (lat/long) texture coordinates for a set of directions.

    Uses the project's spherical convention: azimuth measured in the XZ plane
    from +Z toward +X, elevation above the XZ plane.  ``u`` wraps at the -Z
    meridian; ``v`` runs 0 at +Y (zenith) to 1 at -Y (nadir), matching the
    row order of a standard equirectangular image.

    Args:
        vectors: (..., 3) directions. Need not be unit length.

    Returns:
        ``(u, v)``, each (...) in [0, 1].
    """
    v = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]

    azimuth = np.arctan2(x, z)                       # 0 at +Z, +pi/2 at +X
    u = (azimuth / (2.0 * np.pi)) + 0.5              # 0 at -Z, wraps there
    v_coord = np.arccos(np.clip(y, -1.0, 1.0)) / np.pi

    return u.astype(np.float32), v_coord.astype(np.float32)


def cube_face_uv(
    vectors: NDArray[np.float32]
) -> Tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.float32]]:
    """
    Cubemap face index and face-local texture coordinates for directions.

    Follows the OpenGL cubemap convention so that standard cubemap assets load
    without surprises.

    Args:
        vectors: (..., 3) directions. Need not be unit length.

    Returns:
        ``(face_index, s, t)`` where ``face_index`` indexes :data:`CUBE_FACES`
        and ``s``, ``t`` are in [0, 1] with ``t`` increasing downward.
    """
    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)

    # Dominant axis picks the face.  Ties are broken X > Y > Z, which only
    # matters exactly on an edge.
    x_major = (ax >= ay) & (ax >= az)
    y_major = (~x_major) & (ay >= az)
    z_major = (~x_major) & (~y_major)

    face = np.zeros(x.shape, dtype=np.int32)
    ma = np.empty(x.shape, dtype=np.float32)
    sc = np.empty(x.shape, dtype=np.float32)
    tc = np.empty(x.shape, dtype=np.float32)

    def _assign(sel, index, major, s_val, t_val):
        face[sel] = index
        ma[sel] = major[sel]
        sc[sel] = s_val[sel]
        tc[sel] = t_val[sel]

    _assign(x_major & (x > 0), CUBE_FACES.index("px"), ax, -z, -y)
    _assign(x_major & (x <= 0), CUBE_FACES.index("nx"), ax, z, -y)
    _assign(y_major & (y > 0), CUBE_FACES.index("py"), ay, x, z)
    _assign(y_major & (y <= 0), CUBE_FACES.index("ny"), ay, x, -z)
    _assign(z_major & (z > 0), CUBE_FACES.index("pz"), az, x, -y)
    _assign(z_major & (z <= 0), CUBE_FACES.index("nz"), az, -x, -y)

    # A direction of exactly zero cannot arise from a normalized ray, but guard
    # the division anyway so a degenerate input yields the face centre.
    safe = np.where(ma > 0, ma, 1.0)
    s = (sc / safe + 1.0) * 0.5
    t = (tc / safe + 1.0) * 0.5

    return face, s.astype(np.float32), t.astype(np.float32)


# ---------------------------------------------------------------------------
# Procedural texture generators
# ---------------------------------------------------------------------------
#
# A generator is called once per texture plane -- once for an equirect sphere
# texture, six times for a cube -- and returns float RGB in [0, 1].
#
#   gen(dirs, uv=(s, t) or None, face=name or None, **params) -> (..., 3)
#
# ``dirs`` are the unit world directions of the plane's texels.  ``uv`` and
# ``face`` are supplied only when rasterizing a cube face, so a generator can
# align its pattern to the walls instead of to lat/long.  Generators that want
# a purely directional pattern (the sky ones) just ignore both.

def _as_rgb(color: Sequence[float]) -> NDArray[np.float32]:
    """Coerce an RGB triple to a float32 array, validating range and length."""
    arr = np.asarray(color, dtype=np.float32)
    if arr.shape != (3,):
        raise ValueError(f"Expected an RGB triple, got {color!r}")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"RGB components must be in [0, 1], got {color!r}")
    return arr


def _smoothstep(edge0: float, edge1: float, x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Hermite smoothstep, clamped outside ``[edge0, edge1]``."""
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def blender_sky(
    dirs: NDArray[np.float32],
    uv: Optional[Tuple[NDArray, NDArray]] = None,
    face: Optional[str] = None,
    sun_elevation_deg: float = 15.0,
    sun_azimuth_deg: float = 0.0,
    sun_size_deg: float = 3.0,
    sun_intensity: float = 1.0,
    zenith_color: Sequence[float] = (0.13, 0.28, 0.62),
    horizon_color: Sequence[float] = (0.78, 0.86, 0.95),
    ground_color: Sequence[float] = (0.045, 0.045, 0.045),
    flat: bool = False,
) -> NDArray[np.float32]:
    """
    An approximation of Blender's default Sky Texture (Nishita) world.

    Blue at the zenith fading to a pale horizon, dark below it, with a sun disc
    and halo at the default 15 degrees elevation.

    Be aware of what this texture can and cannot signal.  A Nishita sky is
    **azimuthally symmetric apart from the sun**: rotating the camera about Y
    changes nothing except where the sun sits in frame.  As a cue that the
    camera is orbiting rather than the subject spinning, the sun is doing all
    of the work.  :func:`checker` and :func:`grid` carry far more azimuthal
    structure and are the better test of whether the cue lands at all.

    Args:
        dirs: (..., 3) unit world directions.
        uv: Unused (this pattern is purely directional).
        face: Unused.
        sun_elevation_deg: Sun height above the horizon.
        sun_azimuth_deg: Sun azimuth, measured from +Z toward +X.
        sun_size_deg: Angular diameter of the sun disc.  Blender's default is
            0.545 degrees, which is only a handful of pixels at typical
            render resolutions; the default here is widened so the disc
            actually reads in a conditioning frame.
        sun_intensity: Scales the disc and halo. 0 removes the sun, and with
            it the only azimuthal feature in the texture.
        zenith_color: Sky color straight up.
        horizon_color: Sky color at the horizon.
        ground_color: Color below the horizon.
        flat: Drop the sun. The sky is already a smooth gradient, so the disc
            and its halo are the only pattern there is to remove.

    Returns:
        Float RGB in [0, 1], shape ``dirs.shape``.
    """
    zenith = _as_rgb(zenith_color)
    horizon = _as_rgb(horizon_color)
    ground = _as_rgb(ground_color)

    if sun_size_deg <= 0.0:
        raise ValueError(f"sun_size_deg must be > 0, got {sun_size_deg}")

    height = dirs[..., 1]

    # Gradient above the horizon.  The exponent is a fit to the shape of
    # Nishita's falloff, which is much faster near the horizon than linear.
    t = np.clip(height, 0.0, 1.0) ** 0.42
    sky = horizon * (1.0 - t[..., None]) + zenith * t[..., None]

    # A hard horizon aliases into a shimmering line once the texture is
    # minified, so soften it over a fraction of a degree.
    below = _smoothstep(0.004, -0.004, height)[..., None]
    color = sky * (1.0 - below) + ground * below

    if sun_intensity > 0.0 and not flat:
        sun_dir = spherical_to_cartesian(
            azimuth_deg=sun_azimuth_deg,
            elevation_deg=sun_elevation_deg,
            radius=1.0,
        ).astype(np.float32)
        cos_angle = np.clip(dirs @ sun_dir, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        radius_rad = math.radians(sun_size_deg) / 2.0
        disc = _smoothstep(radius_rad, radius_rad * 0.75, angle)
        halo = np.exp(-angle / 0.09) * 0.55

        glow = np.clip(disc + halo, 0.0, 1.0)[..., None] * sun_intensity
        color = color + glow * (1.0 - color)

    return np.clip(color, 0.0, 1.0)


def gradient(
    dirs: NDArray[np.float32],
    uv: Optional[Tuple[NDArray, NDArray]] = None,
    face: Optional[str] = None,
    top_color: Sequence[float] = (0.55, 0.60, 0.68),
    bottom_color: Sequence[float] = (0.10, 0.11, 0.13),
    power: float = 1.0,
    flat: bool = False,
) -> NDArray[np.float32]:
    """
    A plain vertical two-stop gradient.

    Carries no azimuthal information whatsoever — it is a neutral backdrop, not
    a rotation cue.  Useful as a control when measuring what the background is
    actually contributing.

    Args:
        dirs: (..., 3) unit world directions.
        uv: Unused.
        face: Unused.
        top_color: Color at the zenith.
        bottom_color: Color at the nadir.
        power: Shaping exponent on the blend; > 1 pushes the mix downward.
        flat: Ignored. A gradient has no pattern to remove — it is already
            the smooth backdrop that a fade would reveal.

    Returns:
        Float RGB in [0, 1], shape ``dirs.shape``.
    """
    if power <= 0.0:
        raise ValueError(f"power must be > 0, got {power}")

    top = _as_rgb(top_color)
    bottom = _as_rgb(bottom_color)

    t = (np.clip(dirs[..., 1], -1.0, 1.0) * 0.5 + 0.5) ** power
    return bottom * (1.0 - t[..., None]) + top * t[..., None]


def checker(
    dirs: NDArray[np.float32],
    uv: Optional[Tuple[NDArray, NDArray]] = None,
    face: Optional[str] = None,
    n_azimuth: int = 16,
    n_elevation: int = 8,
    n_per_face: int = 4,
    color_a: Sequence[float] = (0.85, 0.85, 0.85),
    color_b: Sequence[float] = (0.18, 0.20, 0.24),
    flat: bool = False,
) -> NDArray[np.float32]:
    """
    A two-tone checker — the maximum-azimuthal-signal texture.

    Ugly, and deliberately so: this is the texture to validate against.  If an
    orbit does not read as an orbit over a checker, no amount of photographic
    sky is going to rescue it, and the problem is in the camera path rather
    than in the backdrop.

    On a sphere the tiles follow lat/long, so they pinch at the poles.  On a
    cube they follow the face UVs, giving square tiles that align with the
    walls, floor and ceiling.

    Args:
        dirs: (..., 3) unit world directions.
        uv: Face-local ``(s, t)`` when rasterizing a cube face.
        face: Face name when rasterizing a cube face.
        n_azimuth: Tiles around the full 360 degrees (sphere only).
        n_elevation: Tiles from zenith to nadir (sphere only).
        n_per_face: Tiles across one cube face, per axis (cube only).
        color_a: First tile color.
        color_b: Second tile color.
        flat: Drop the tiling and return the mean of the two colors. A checker
            is pure pattern with no underlying shading, so its pattern-free
            form is a single tone.

    Returns:
        Float RGB in [0, 1], shape ``dirs.shape``.
    """
    first = _as_rgb(color_a)
    second = _as_rgb(color_b)

    if flat:
        return np.broadcast_to((first + second) * 0.5, dirs.shape).copy()

    if uv is not None:
        if n_per_face < 1:
            raise ValueError(f"n_per_face must be >= 1, got {n_per_face}")
        s, t = uv
        cell = np.floor(s * n_per_face) + np.floor(t * n_per_face)
    else:
        if n_azimuth < 1 or n_elevation < 1:
            raise ValueError(
                f"n_azimuth and n_elevation must be >= 1, got "
                f"{n_azimuth} and {n_elevation}"
            )
        u, v = equirect_uv(dirs)
        cell = np.floor(u * n_azimuth) + np.floor(v * n_elevation)

    parity = (cell.astype(np.int64) % 2).astype(bool)
    return np.where(parity[..., None], first, second)


def grid(
    dirs: NDArray[np.float32],
    uv: Optional[Tuple[NDArray, NDArray]] = None,
    face: Optional[str] = None,
    n_azimuth: int = 24,
    n_elevation: int = 12,
    n_per_face: int = 6,
    line_width: float = 0.035,
    base_color: Sequence[float] = (0.42, 0.44, 0.48),
    line_color: Sequence[float] = (0.88, 0.89, 0.92),
    floor_color: Sequence[float] = (0.24, 0.25, 0.27),
    ceiling_color: Sequence[float] = (0.58, 0.60, 0.64),
    flat: bool = False,
) -> NDArray[np.float32]:
    """
    A ruled room: grid lines on flat walls, with a darker floor and lighter
    ceiling.

    This is the strongest cue of the built-in generators.  The lines give
    azimuthal structure that sweeps with the camera, and the floor/ceiling
    tinting resolves the up/down ambiguity that a symmetric texture leaves
    open.  Paired with ``geometry="cube"`` and a finite radius it reads as an
    actual room, with corners and wall perspective that move correctly under
    camera translation.

    Args:
        dirs: (..., 3) unit world directions.
        uv: Face-local ``(s, t)`` when rasterizing a cube face.
        face: Face name when rasterizing a cube face.
        n_azimuth: Meridians around the full 360 degrees (sphere only).
        n_elevation: Parallels from zenith to nadir (sphere only).
        n_per_face: Grid divisions across one cube face, per axis (cube only).
        line_width: Line thickness as a fraction of one cell.
        base_color: Wall color.
        line_color: Grid line color.
        floor_color: Base color for the downward face/hemisphere.
        ceiling_color: Base color for the upward face/hemisphere.
        flat: Draw the room without its grid lines — walls, floor and ceiling
            in their own colors and nothing else. This is what the lines fade
            *to* under the subject fade; see :class:`~body2colmap.fade.SubjectFade`.

    Returns:
        Float RGB in [0, 1], shape ``dirs.shape``.
    """
    if not 0.0 < line_width < 1.0:
        raise ValueError(f"line_width must be in (0, 1), got {line_width}")

    base = _as_rgb(base_color)
    line = _as_rgb(line_color)
    floor = _as_rgb(floor_color)
    ceiling = _as_rgb(ceiling_color)

    if uv is not None:
        if n_per_face < 1:
            raise ValueError(f"n_per_face must be >= 1, got {n_per_face}")
        s, t = uv
        su, sv = s * n_per_face, t * n_per_face
        if face == "ny":
            ground = np.ones(dirs.shape[:-1], dtype=np.float32)
        elif face == "py":
            ground = -np.ones(dirs.shape[:-1], dtype=np.float32)
        else:
            ground = np.zeros(dirs.shape[:-1], dtype=np.float32)
    else:
        if n_azimuth < 1 or n_elevation < 1:
            raise ValueError(
                f"n_azimuth and n_elevation must be >= 1, got "
                f"{n_azimuth} and {n_elevation}"
            )
        u, v = equirect_uv(dirs)
        su, sv = u * n_azimuth, v * n_elevation
        # -1 at the zenith, +1 at the nadir, matching the cube's face test.
        ground = -np.clip(dirs[..., 1], -1.0, 1.0)

    up_mix = np.clip(-ground, 0.0, 1.0)[..., None]
    down_mix = np.clip(ground, 0.0, 1.0)[..., None]
    color = base * (1.0 - up_mix - down_mix) + ceiling * up_mix + floor * down_mix

    if flat:
        return color

    # Distance to the nearest cell boundary, in cells.
    du = np.minimum(su % 1.0, 1.0 - (su % 1.0))
    dv = np.minimum(sv % 1.0, 1.0 - (sv % 1.0))
    on_line = np.minimum(du, dv) < (line_width * 0.5)

    return np.where(on_line[..., None], line, color)


#: Built-in procedural textures, by the name used in config and on the CLI.
TEXTURE_GENERATORS: Dict[str, Callable[..., NDArray[np.float32]]] = {
    "blender_sky": blender_sky,
    "gradient": gradient,
    "checker": checker,
    "grid": grid,
}


def _call_generator(
    name: str,
    dirs: NDArray[np.float32],
    uv: Optional[Tuple[NDArray, NDArray]],
    face: Optional[str],
    params: Dict[str, Any],
) -> NDArray[np.uint8]:
    """
    Evaluate a named generator and quantize to 8-bit RGB.

    Args:
        name: Generator name, a key of :data:`TEXTURE_GENERATORS`.
        dirs: (..., 3) unit world directions for each texel.
        uv: Face-local coordinates, or None for an equirect plane.
        face: Face name, or None for an equirect plane.
        params: Extra keyword arguments for the generator.

    Returns:
        uint8 RGB, shape ``dirs.shape``.

    Raises:
        ValueError: If the generator is unknown, or ``params`` names an
            argument it does not accept.
    """
    try:
        fn = TEXTURE_GENERATORS[name]
    except KeyError:
        raise ValueError(
            f"Unknown background texture generator {name!r}. "
            f"Built-ins: {', '.join(sorted(TEXTURE_GENERATORS))}. "
            f"A path to an image or a directory of cube faces also works."
        ) from None

    accepted = set(inspect.signature(fn).parameters) - {"dirs", "uv", "face"}
    unknown = sorted(set(params) - accepted)
    if unknown:
        raise ValueError(
            f"Background texture {name!r} does not accept "
            f"{', '.join(repr(u) for u in unknown)}. "
            f"Accepted parameters: {', '.join(sorted(accepted))}"
        )

    rgb = fn(dirs, uv=uv, face=face, **params)
    return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Texture rasterization
# ---------------------------------------------------------------------------

def equirect_directions(height: int) -> NDArray[np.float32]:
    """
    World directions for every texel of an equirectangular image.

    Args:
        height: Image height. Width is ``2 * height``.

    Returns:
        Float32 array, shape (height, 2 * height, 3), unit length. The inverse
        of :func:`equirect_uv` evaluated at texel centres.
    """
    width = 2 * height

    u = (np.arange(width, dtype=np.float32) + 0.5) / width
    v = (np.arange(height, dtype=np.float32) + 0.5) / height
    grid_u, grid_v = np.meshgrid(u, v)

    azimuth = (grid_u - 0.5) * 2.0 * np.pi
    polar = grid_v * np.pi

    sin_polar = np.sin(polar)
    return np.stack(
        [sin_polar * np.sin(azimuth), np.cos(polar), sin_polar * np.cos(azimuth)],
        axis=-1,
    ).astype(np.float32)


def cube_face_directions(
    face: str,
    size: int
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """
    World directions and face-local coordinates for every texel of a cube face.

    The inverse of :func:`cube_face_uv` evaluated at texel centres.

    Args:
        face: One of :data:`CUBE_FACES`.
        size: Face resolution in pixels (square).

    Returns:
        ``(dirs, s, t)``: unit directions of shape (size, size, 3), and the
        face-local coordinates of shape (size, size) in [0, 1].

    Raises:
        ValueError: If ``face`` is not a known face name.
    """
    if face not in CUBE_FACES:
        raise ValueError(
            f"Unknown cube face {face!r}. Faces: {', '.join(CUBE_FACES)}"
        )

    coord = (np.arange(size, dtype=np.float32) + 0.5) / size
    s, t = np.meshgrid(coord, coord)

    sc = s * 2.0 - 1.0
    tc = t * 2.0 - 1.0
    one = np.ones_like(sc)

    axes = {
        "px": (one, -tc, -sc),
        "nx": (-one, -tc, sc),
        "py": (sc, one, tc),
        "ny": (sc, -one, -tc),
        "pz": (sc, -tc, one),
        "nz": (-sc, -tc, -one),
    }[face]

    dirs = np.stack(axes, axis=-1).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    return dirs, s, t


def generate_texture(
    name: str,
    geometry: str = "sphere",
    resolution: int = 1024,
    params: Optional[Dict[str, Any]] = None,
) -> Union[NDArray[np.uint8], List[NDArray[np.uint8]]]:
    """
    Rasterize a built-in procedural texture.

    Args:
        name: Generator name, a key of :data:`TEXTURE_GENERATORS`.
        geometry: "sphere" (one equirectangular plane) or "cube" (six faces).
        resolution: Equirect height, or cube face size.
        params: Extra keyword arguments for the generator.

    Returns:
        For a sphere, a ``(resolution, 2 * resolution, 3)`` uint8 image. For a
        cube, a list of six ``(resolution, resolution, 3)`` uint8 images in
        :data:`CUBE_FACES` order.

    Raises:
        ValueError: On an unknown generator, geometry or parameter.
    """
    params = dict(params or {})

    if resolution < 8:
        raise ValueError(f"Background resolution must be >= 8, got {resolution}")

    if geometry == "sphere":
        return _call_generator(name, equirect_directions(resolution), None, None, params)

    if geometry == "cube":
        faces = []
        for face in CUBE_FACES:
            dirs, s, t = cube_face_directions(face, resolution)
            faces.append(_call_generator(name, dirs, (s, t), face, params))
        return faces

    raise ValueError(
        f"Unknown background geometry {geometry!r}. Use 'sphere' or 'cube'."
    )


# ---------------------------------------------------------------------------
# Texture loading
# ---------------------------------------------------------------------------

def _read_image(path: Path) -> NDArray[np.uint8]:
    """Read an image file as 8-bit RGB, dropping any alpha."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        hint = ""
        if path.suffix.lower() in (".exr", ".hdr"):
            hint = (
                " OpenCV gates HDR formats behind an environment variable; try "
                "OPENCV_IO_ENABLE_OPENEXR=1, or convert the panorama to PNG."
            )
        raise ValueError(f"Could not read background texture: {path}.{hint}")

    if image.dtype != np.uint8:
        # HDR sources (.exr/.hdr) come back float. Reinhard-tone-map rather
        # than clip: these are backdrops for conditioning frames, so keeping
        # the sky from blowing out matters more than radiometric fidelity.
        image = image.astype(np.float32)
        image = np.clip(image, 0.0, None)
        image = image / (1.0 + image)
        image = (image ** (1.0 / 2.2) * 255.0 + 0.5).astype(np.uint8)

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    elif image.shape[2] != 3:
        raise ValueError(
            f"Background texture {path} has {image.shape[2]} channels; "
            f"expected 1, 3 or 4"
        )

    return np.ascontiguousarray(image[:, :, ::-1])


def _find_face_files(directory: Path) -> List[Path]:
    """
    Locate six cube-face images in a directory, by stem.

    Accepts ``px``/``nx``/... , ``posx``/``negx``/... and
    ``right``/``left``/``top``/``bottom``/``front``/``back``.

    Args:
        directory: Directory to search.

    Returns:
        Six paths in :data:`CUBE_FACES` order.

    Raises:
        ValueError: If any face is missing or ambiguous.
    """
    by_stem: Dict[str, List[Path]] = {}
    for entry in sorted(directory.iterdir()):
        if entry.is_file() and entry.suffix.lower() in _IMAGE_SUFFIXES:
            by_stem.setdefault(entry.stem.lower(), []).append(entry)

    paths: List[Path] = []
    for face in CUBE_FACES:
        matches = [p for a in _FACE_ALIASES[face] for p in by_stem.get(a, [])]
        if not matches:
            raise ValueError(
                f"No image for cube face {face!r} in {directory}. Expected a "
                f"file named one of: "
                f"{', '.join(a + '.*' for a in _FACE_ALIASES[face])}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Multiple images for cube face {face!r} in {directory}: "
                f"{', '.join(p.name for p in matches)}"
            )
        paths.append(matches[0])

    return paths


def equirect_to_cube(
    equirect: NDArray[np.uint8],
    size: int
) -> List[NDArray[np.uint8]]:
    """
    Resample an equirectangular image into six cube faces.

    Args:
        equirect: uint8 RGB, ideally 2:1.
        size: Output face resolution.

    Returns:
        Six ``(size, size, 3)`` uint8 images in :data:`CUBE_FACES` order.
    """
    import cv2

    height, width = equirect.shape[:2]
    padded = _pad_equirect(equirect)

    faces = []
    for face in CUBE_FACES:
        dirs, _, _ = cube_face_directions(face, size)
        u, v = equirect_uv(dirs)
        map_x = (u * width + 0.5).astype(np.float32)
        map_y = (v * height + 0.5).astype(np.float32)
        faces.append(
            cv2.remap(padded, map_x, map_y, cv2.INTER_LINEAR)
        )

    return faces


def _split_cube_image(image: NDArray[np.uint8], path: Path) -> List[NDArray[np.uint8]]:
    """
    Split a packed cubemap image into six faces.

    Recognized layouts, chosen by aspect ratio:

    - **4:3 horizontal cross** — the common layout, with ``+Y`` above ``+Z``
      and ``-Y`` below it, and the row ``-X +Z +X -Z``.
    - **6:1 strip** / **1:6 column** — faces in :data:`CUBE_FACES` order.
    - **2:1** — treated as equirectangular and resampled onto the cube.

    The 3:4 vertical cross is deliberately not accepted: its ``-Z`` face is
    rotated 180 degrees in some exports and not others, so guessing would
    silently produce a backdrop with one wall upside down.

    Args:
        image: uint8 RGB.
        path: Source path, for error messages.

    Returns:
        Six square uint8 images in :data:`CUBE_FACES` order.

    Raises:
        ValueError: If the aspect ratio matches no known layout, or the
            implied face size is not integral.
    """
    height, width = image.shape[:2]
    aspect = width / height

    def _face_size(divisor_w: int, divisor_h: int) -> int:
        if width % divisor_w or height % divisor_h:
            raise ValueError(
                f"Cubemap {path} is {width}x{height}, which does not divide "
                f"evenly into {divisor_w}x{divisor_h} faces"
            )
        size_w, size_h = width // divisor_w, height // divisor_h
        if size_w != size_h:
            raise ValueError(
                f"Cubemap {path} implies non-square faces ({size_w}x{size_h})"
            )
        return size_w

    if abs(aspect - 4.0 / 3.0) < 0.01:
        size = _face_size(4, 3)

        def _cell(row: int, col: int) -> NDArray[np.uint8]:
            return np.ascontiguousarray(
                image[row * size:(row + 1) * size, col * size:(col + 1) * size]
            )

        # row 1 is  -X  +Z  +X  -Z ; +Y sits above +Z and -Y below it.
        return [
            _cell(1, 2),  # px
            _cell(1, 0),  # nx
            _cell(0, 1),  # py
            _cell(2, 1),  # ny
            _cell(1, 1),  # pz
            _cell(1, 3),  # nz
        ]

    if abs(aspect - 6.0) < 0.01:
        size = _face_size(6, 1)
        return [
            np.ascontiguousarray(image[:, i * size:(i + 1) * size])
            for i in range(6)
        ]

    if abs(aspect - 1.0 / 6.0) < 0.01:
        size = _face_size(1, 6)
        return [
            np.ascontiguousarray(image[i * size:(i + 1) * size, :])
            for i in range(6)
        ]

    if abs(aspect - 2.0) < 0.05:
        return equirect_to_cube(image, max(64, height // 2))

    raise ValueError(
        f"Cannot infer a cubemap layout for {path} ({width}x{height}, aspect "
        f"{aspect:.3f}). Supported: a 4:3 horizontal cross, a 6:1 strip, a 1:6 "
        f"column, a 2:1 equirectangular image, or a directory of six face "
        f"images."
    )


def load_texture(
    path: Union[str, Path],
    geometry: str = "sphere",
) -> Union[NDArray[np.uint8], List[NDArray[np.uint8]]]:
    """
    Load a background texture from disk.

    For ``geometry="sphere"`` the file is an equirectangular image. For
    ``geometry="cube"`` it may be a directory of six face images, a packed
    cubemap, or an equirectangular image that is resampled onto the cube --
    see :func:`_split_cube_image`.

    Args:
        path: Image file, or (cube only) a directory of six face images.
        geometry: "sphere" or "cube".

    Returns:
        An equirect image, or a list of six face images.

    Raises:
        ValueError: On a missing path, an unusable file, or an unknown
            geometry.
    """
    path = Path(path).expanduser()

    if not path.exists():
        raise ValueError(f"Background texture not found: {path}")

    if geometry == "sphere":
        if path.is_dir():
            raise ValueError(
                f"Background texture {path} is a directory; sphere geometry "
                f"takes a single equirectangular image. Use geometry='cube' "
                f"for a directory of six faces."
            )
        equirect = _read_image(path)
        aspect = equirect.shape[1] / equirect.shape[0]
        if abs(aspect - 2.0) > 0.05:
            raise ValueError(
                f"Equirectangular texture {path} is {equirect.shape[1]}x"
                f"{equirect.shape[0]} (aspect {aspect:.3f}); expected 2:1"
            )
        return equirect

    if geometry == "cube":
        if path.is_dir():
            return [_read_image(p) for p in _find_face_files(path)]
        return _split_cube_image(_read_image(path), path)

    raise ValueError(
        f"Unknown background geometry {geometry!r}. Use 'sphere' or 'cube'."
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _pad_equirect(equirect: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Pad an equirect image by one texel for artifact-free bilinear sampling.

    Horizontally the image wraps, so the padding is the opposite edge -- this
    is what keeps the -Z meridian from showing a seam. Vertically it replicates
    the pole rows, where wrapping would sample the *other* pole.

    ``cv2.remap``'s ``borderMode`` cannot express that split, which is why the
    padding is done by hand and the sample coordinates are offset by one.

    Args:
        equirect: uint8 RGB, shape (H, W, 3).

    Returns:
        uint8 RGB, shape (H + 2, W + 2, 3).
    """
    wrapped = np.concatenate(
        [equirect[:, -1:], equirect, equirect[:, :1]], axis=1
    )
    return np.concatenate(
        [wrapped[:1], wrapped, wrapped[-1:]], axis=0
    )


def _pad_face(face: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Pad a cube face by one texel, replicating its edges.

    True cubemap edge filtering would pull the neighbouring face's border row.
    Replication is off by at most half a texel along the seam and needs no
    adjacency table; at the face resolutions used here that is invisible.

    Args:
        face: uint8 RGB, shape (S, S, 3).

    Returns:
        uint8 RGB, shape (S + 2, S + 2, 3).
    """
    padded = np.concatenate([face[:, :1], face, face[:, -1:]], axis=1)
    return np.concatenate([padded[:1], padded, padded[-1:]], axis=0)


def _check_texture(
    texture: Union[NDArray[np.uint8], Sequence[NDArray[np.uint8]]],
    geometry: str,
    what: str = "texture",
) -> Any:
    """
    Validate and normalize a texture for a geometry.

    Args:
        texture: An equirectangular image, or six cube faces.
        geometry: "sphere" or "cube".
        what: Noun for the error messages, so a plain texture's complaint does
            not read as if the main one were wrong.

    Returns:
        A contiguous uint8 equirect array, or a list of six such faces.

    Raises:
        ValueError: On a malformed texture.
    """
    if geometry == "sphere":
        equirect = np.asarray(texture)
        if equirect.ndim != 3 or equirect.shape[2] != 3:
            raise ValueError(
                f"Sphere background {what} needs an (H, W, 3) image, got shape "
                f"{equirect.shape}"
            )
        return np.ascontiguousarray(equirect.astype(np.uint8))

    faces = list(texture)
    if len(faces) != 6:
        raise ValueError(f"Cube background {what} needs 6 faces, got {len(faces)}")
    shapes = {np.asarray(f).shape for f in faces}
    if len(shapes) != 1:
        raise ValueError(
            f"Cube {what} faces must all be the same size, got {sorted(shapes)}"
        )
    shape = shapes.pop()
    if len(shape) != 3 or shape[2] != 3 or shape[0] != shape[1]:
        raise ValueError(f"Cube {what} faces must be square (S, S, 3), got {shape}")
    return [np.ascontiguousarray(np.asarray(f).astype(np.uint8)) for f in faces]


class Background:
    """
    A static environment drawn behind a render.

    The environment is the inside of a sphere or an axis-aligned cube centred
    on ``center``.  Each pixel's ray is intersected with that surface and the
    hit point looked up in the texture, so the backdrop is fixed in world space
    and sweeps past as the camera orbits.

    **Radius matters more than it looks.**  With ``radius=None`` the surface is
    at infinity and the lookup depends only on ray *direction*: the backdrop
    responds to camera rotation but not to camera translation, and sphere and
    cube then differ only in how the texture is parameterized, not in what is
    rendered.  A finite radius adds real parallax between the subject and the
    backdrop, and is what makes the cube read as a room with corners.

    These are conditioning frames, so the environment is deliberately absent
    from everything else: it is not exported to COLMAP, contributes no points
    to the point cloud, and does not participate in the depth buffer or the
    silhouette mask.

    Attributes:
        geometry: "sphere" or "cube".
        center: World-space centre of the surface, shape (3,).
        radius: Sphere radius or cube half-extent, or None for infinity.
        rotation_deg: Rotation of the environment about +Y, in degrees.
        opaque: Whether :meth:`composite` forces alpha to 255.
    """

    def __init__(
        self,
        texture: Union[NDArray[np.uint8], Sequence[NDArray[np.uint8]]],
        geometry: str = "sphere",
        center: Optional[NDArray[np.float32]] = None,
        radius: Optional[float] = None,
        rotation_deg: float = 0.0,
        opaque: bool = True,
        fade: Optional[SubjectFade] = None,
        plain_texture: Optional[
            Union[NDArray[np.uint8], Sequence[NDArray[np.uint8]]]
        ] = None,
    ):
        """
        Args:
            texture: An equirectangular uint8 RGB image for ``geometry="sphere"``,
                or six square uint8 RGB faces in :data:`CUBE_FACES` order for
                ``geometry="cube"``.
            geometry: "sphere" or "cube".
            center: World-space centre. Defaults to the origin. Only meaningful
                with a finite ``radius``; set it to the orbit target.
            radius: Sphere radius, or cube half-extent, in world units. None
                places the surface at infinity.
            rotation_deg: Rotate the environment about +Y. Aims the sun, or
                turns a cube's walls relative to the subject.
            opaque: If True, :meth:`composite` sets alpha to 255 everywhere,
                producing a flat conditioning frame. If False the base layer's
                alpha is preserved and the background only fills RGB, which
                keeps the silhouette usable as a mask.
            fade: Optional :class:`~body2colmap.fade.SubjectFade`, which
                fades the backdrop out around the subject so an ``outline``
                frame does not read as a hard occlusion boundary. Applied to
                :meth:`render`'s output, so it never touches the subject.
            plain_texture: The same environment with its *pattern* removed —
                a grid's walls without their lines, a checker's mean tone —
                and the same size as ``texture``. This is what the fade's
                default ``target="plain"`` reveals: the lines fade out and the
                wall behind them stays, rather than being smeared into it.
                Only a generated texture has one; a loaded image cannot be
                decomposed this way.

        Raises:
            ValueError: On an unknown geometry, a malformed texture, or a
                non-positive radius.
        """
        if geometry not in ("sphere", "cube"):
            raise ValueError(
                f"Unknown background geometry {geometry!r}. Use 'sphere' or 'cube'."
            )

        checked = _check_texture(texture, geometry)
        if geometry == "sphere":
            self._equirect, self._faces = checked, None
        else:
            self._equirect, self._faces = None, checked

        if plain_texture is None:
            self._plain_equirect = self._plain_faces = None
        else:
            plain = _check_texture(plain_texture, geometry, what="plain texture")
            if geometry == "sphere":
                if plain.shape != self._equirect.shape:
                    raise ValueError(
                        f"The plain texture is {plain.shape} but the texture "
                        f"it stands in for is {self._equirect.shape}; they are "
                        f"sampled with one set of maps and must match"
                    )
                self._plain_equirect, self._plain_faces = plain, None
            else:
                if plain[0].shape != self._faces[0].shape:
                    raise ValueError(
                        f"The plain texture's faces are {plain[0].shape} but "
                        f"the texture's are {self._faces[0].shape}; they are "
                        f"sampled with one set of maps and must match"
                    )
                self._plain_equirect, self._plain_faces = None, plain

        if radius is not None and radius <= 0.0:
            raise ValueError(f"Background radius must be > 0, got {radius}")

        self.geometry = geometry
        self.center = (
            np.zeros(3, dtype=np.float32) if center is None
            else np.asarray(center, dtype=np.float32).reshape(3)
        )
        self.radius = None if radius is None else float(radius)
        self.rotation_deg = float(rotation_deg)
        self.opaque = bool(opaque)
        self.fade = fade

        if (fade is not None and fade.target == "plain"
                and not self.has_plain_texture()):
            raise ValueError(
                "The fade's target='plain' dissolves the backdrop's pattern "
                "into the shading behind it, which needs that shading as a "
                "separate texture. Only a generated texture has one -- a "
                "loaded image cannot be split into pattern and shading. Use "
                "target='color' for a flat clear zone, or target='blur' to "
                "average the image into itself."
            )

        # Lazily built, keyed on camera intrinsics (fixed across an orbit).
        self._ray_key: Optional[Tuple[float, ...]] = None
        self._ray_cache: Optional[NDArray[np.float32]] = None
        self._padded: Optional[Any] = None

        # Resolution fitting. The source is kept so that a later camera with a
        # longer focal length can be re-fitted from it rather than from an
        # already-downsampled copy -- render_original_view() typically zooms in
        # relative to the orbit, and detail thrown away cannot be recovered.
        self._source: Optional[Any] = None
        self._plain_source: Optional[Any] = None
        self._fitted_fx: Optional[float] = None
        self._padded_plain: Optional[Any] = None

        # Mean texture colour, the fallback fade colour. Cached because it is
        # a whole-texture reduction and does not change with resolution
        # fitting -- INTER_AREA preserves the mean.
        self._mean_color: Optional[NDArray[np.float32]] = None

    # -- construction -------------------------------------------------------

    @classmethod
    def create(
        cls,
        texture: str = "grid",
        geometry: str = "cube",
        resolution: int = 1024,
        center: Optional[NDArray[np.float32]] = None,
        radius: Optional[float] = None,
        rotation_deg: float = 0.0,
        opaque: bool = True,
        params: Optional[Dict[str, Any]] = None,
        fade: Optional[SubjectFade] = None,
    ) -> "Background":
        """
        Build a Background from a generator name or a path.

        This is the single entry point used by config and the CLI, so that
        ``texture`` can be either kind of thing in one field.

        Args:
            texture: A built-in generator name (a key of
                :data:`TEXTURE_GENERATORS`) or a path to an image, or -- for
                ``geometry="cube"`` -- a directory of six face images.
            geometry: "sphere" or "cube".
            resolution: Equirect height / cube face size, for generated
                textures only. Loaded textures keep their own resolution.
            center: World-space centre of the surface.
            radius: Sphere radius or cube half-extent; None for infinity.
                Note that the default ``geometry`` only reads as a room with a
                finite radius; see :data:`DEFAULT_RADIUS_SCALE`.
            rotation_deg: Rotate the environment about +Y.
            opaque: See :meth:`__init__`.
            params: Extra keyword arguments for a generator. Rejected for a
                loaded texture, where they would silently do nothing.
            fade: Optional :class:`~body2colmap.fade.SubjectFade`; see
                :meth:`__init__`.

        Returns:
            A Background.

        Raises:
            ValueError: On an unknown generator name that is also not an
                existing path, or ``params`` given alongside a file.
        """
        if texture in TEXTURE_GENERATORS:
            data = generate_texture(texture, geometry, resolution, params)
            # The same texture with its pattern suppressed, so a fade can
            # dissolve the pattern into the shading underneath instead of
            # blurring the two together. Cheap: one extra rasterization, once.
            plain = generate_texture(
                texture, geometry, resolution, {**(params or {}), "flat": True}
            )
        else:
            plain = None
            if not Path(texture).expanduser().exists():
                # A near-miss on a generator name is far more likely than a
                # genuine missing file, so name the built-ins rather than
                # reporting it as a bad path.
                raise ValueError(
                    f"Background texture {texture!r} is neither a built-in "
                    f"generator nor an existing path. Generators: "
                    f"{', '.join(sorted(TEXTURE_GENERATORS))}"
                )
            if params:
                raise ValueError(
                    f"background params {sorted(params)} were given, but the "
                    f"texture {texture!r} is a file, not a generator. "
                    f"Generators: {', '.join(sorted(TEXTURE_GENERATORS))}"
                )
            data = load_texture(texture, geometry)

        return cls(
            data,
            geometry=geometry,
            center=center,
            radius=radius,
            rotation_deg=rotation_deg,
            opaque=opaque,
            fade=fade,
            plain_texture=plain,
        )

    # -- rendering ----------------------------------------------------------

    def _ray_grid_cam(self, camera: Camera) -> NDArray[np.float32]:
        """
        Unit ray directions in *camera* space, cached across frames.

        Every camera on an orbit shares one set of intrinsics, so only the
        rotation changes frame to frame. Caching the camera-space grid turns
        the per-frame cost into a single matrix multiply -- see
        :meth:`_world_rays`.

        Args:
            camera: Camera to generate rays for.

        Returns:
            Float32 array, shape (height, width, 3), unit length.
        """
        key = (camera.fx, camera.fy, camera.cx, camera.cy,
               camera.width, camera.height)

        if self._ray_key != key:
            px = np.arange(camera.width, dtype=np.float32) + 0.5
            py = np.arange(camera.height, dtype=np.float32) + 0.5
            grid_x, grid_y = np.meshgrid(
                (px - camera.cx) / camera.fx,
                (py - camera.cy) / camera.fy,
            )
            # OpenCV ray -> OpenGL camera ray: negate Y and Z. See
            # camera_ray_directions().
            dirs = np.stack(
                [grid_x, -grid_y, -np.ones_like(grid_x)], axis=-1
            ).astype(np.float32)
            dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

            self._ray_key = key
            self._ray_cache = dirs

        return self._ray_cache

    def _world_rays(self, camera: Camera) -> NDArray[np.float32]:
        """
        Unit ray directions in world coordinates.

        Args:
            camera: Camera to generate rays for.

        Returns:
            Float32 array, shape (height, width, 3), unit length. Equivalent
            to :func:`camera_ray_directions`, but reusing the cached
            camera-space grid.
        """
        rotation = np.asarray(camera.rotation, dtype=np.float32)
        return self._ray_grid_cam(camera) @ rotation.T

    def _fit_resolution(self, camera: Camera) -> None:
        """
        Downsample an over-detailed texture to roughly the render's own
        angular resolution.

        A 4K panorama minified to a 720p frame is point-sampled differently
        every frame, and the resulting shimmer reads as motion to a video
        model -- the opposite of what a *static* backdrop is for. Area-
        averaging down to about twice the needed detail removes it, and costs
        one resize per run.

        The camera's angular pixel size is ``1 / fx`` radians, so a sphere
        needs about ``2 * pi * fx`` texels around and a cube face, spanning 90
        degrees, about ``(pi / 2) * fx`` across.

        Args:
            camera: The camera the background will be rendered for.
        """
        if self._fitted_fx is not None and camera.fx <= self._fitted_fx:
            return

        import cv2

        if self._source is None:
            self._source = self._current_texture()
            self._plain_source = self._current_texture(plain=True)

        if self.geometry == "sphere":
            ideal_w = max(256, int(round(2.0 * np.pi * camera.fx)))

            def fit(source):
                # The plain texture is sampled with the *same* maps as the
                # main one, so it has to be resized in lockstep even when its
                # own detail would not have needed it.
                if source is None:
                    return None
                return (
                    cv2.resize(source, (ideal_w, ideal_w // 2),
                               interpolation=cv2.INTER_AREA)
                    if self._source.shape[1] > 2 * ideal_w else source
                )

            self._equirect = fit(self._source)
            self._plain_equirect = fit(self._plain_source)
        else:
            ideal_s = max(64, int(round(0.5 * np.pi * camera.fx)))

            def fit(source):
                if source is None:
                    return None
                return (
                    [cv2.resize(f, (ideal_s, ideal_s),
                                interpolation=cv2.INTER_AREA) for f in source]
                    if self._source[0].shape[0] > 2 * ideal_s else list(source)
                )

            self._faces = fit(self._source)
            self._plain_faces = fit(self._plain_source)

        self._fitted_fx = float(camera.fx)
        self._padded = None
        self._padded_plain = None

    def _current_texture(self, plain: bool = False):
        """
        The texture in use, whichever geometry this is.

        Args:
            plain: Return the pattern-free variant instead. None when the
                texture was loaded from a file rather than generated.

        Returns:
            An equirect array, a list of six faces, or None.
        """
        if self.geometry == "sphere":
            return self._plain_equirect if plain else self._equirect
        return list(self._plain_faces) if plain and self._plain_faces else (
            None if plain else self._faces
        )

    def has_plain_texture(self) -> bool:
        """
        Whether a pattern-free variant of this backdrop exists.

        Only generated textures have one — a loaded image cannot be split into
        pattern and shading — so this gates the fade's ``target="plain"``.

        Returns:
            True if :meth:`render` can reveal a plain backdrop under the fade.
        """
        return (self._plain_equirect if self.geometry == "sphere"
                else self._plain_faces) is not None

    def _padded_texture(self, plain: bool = False):
        """
        Get the border-padded texture, building it on first use.

        Args:
            plain: Pad the pattern-free variant instead.

        Returns:
            A padded equirect array, or a list of six padded faces.
        """
        if plain:
            if self._padded_plain is None:
                source = self._current_texture(plain=True)
                self._padded_plain = (
                    _pad_equirect(source) if self.geometry == "sphere"
                    else [_pad_face(f) for f in source]
                )
            return self._padded_plain

        if self._padded is None:
            if self.geometry == "sphere":
                self._padded = _pad_equirect(self._equirect)
            else:
                self._padded = [_pad_face(f) for f in self._faces]
        return self._padded

    def _surface_vectors(
        self,
        camera: Camera,
        dirs: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Vectors from the surface centre to where each ray meets the surface.

        At infinite radius this is just the ray direction. At a finite radius
        it is the hit point relative to ``center``, which is what introduces
        parallax: the same direction lands on a different part of the texture
        depending on where the camera stands.

        The environment rotation is applied to the whole ray -- origin and
        direction both -- rather than to the texture lookup, so that a cube's
        walls turn with its texture instead of the texture sliding across
        fixed walls.

        Args:
            camera: Camera being rendered.
            dirs: (H, W, 3) unit world ray directions.

        Returns:
            (H, W, 3) lookup vectors. Not normalized.

        Raises:
            ValueError: If a finite surface does not enclose the camera.
        """
        if self.rotation_deg:
            rot = _rotation_y(-self.rotation_deg)
            dirs = dirs @ rot.T

        if self.radius is None:
            return dirs

        origin = np.asarray(camera.position, dtype=np.float32) - self.center
        if self.rotation_deg:
            origin = rot @ origin

        if self.geometry == "sphere":
            distance = float(np.linalg.norm(origin))
            if distance >= self.radius:
                raise ValueError(
                    f"Camera is {distance:.3f} from the background centre but "
                    f"the sphere radius is {self.radius:.3f}; the camera must "
                    f"be inside it. Increase background radius, or set it to "
                    f"None for an infinite backdrop."
                )
            # |origin + t * dirs|^2 = radius^2, taking the forward root.
            b = np.einsum("...i,...i->...", dirs, origin)
            c = distance * distance - self.radius * self.radius
            t = -b + np.sqrt(np.maximum(b * b - c, 0.0))
        else:
            extent = float(np.max(np.abs(origin)))
            if extent >= self.radius:
                raise ValueError(
                    f"Camera is {extent:.3f} from the background centre along "
                    f"its dominant axis but the cube half-extent is "
                    f"{self.radius:.3f}; the camera must be inside it. "
                    f"Increase background radius, or set it to None for an "
                    f"infinite backdrop."
                )
            # Exit distance through an axis-aligned box from inside it: the
            # nearest of the three slab exits.
            with np.errstate(divide="ignore", invalid="ignore"):
                t_axis = (np.sign(dirs) * self.radius - origin) / dirs
            t_axis = np.where(np.abs(dirs) < 1e-12, np.inf, t_axis)
            t = np.min(t_axis, axis=-1)

        return origin + t[..., None] * dirs

    def _texture_mean(self) -> NDArray[np.float32]:
        """
        Mean colour of the texture, as RGB floats in [0, 1].

        The fallback fade colour: it is the one flat colour that leaves the
        frame's overall tone unchanged, which matters because the alternative
        -- a guess -- reads as a patch.

        Returns:
            Shape (3,) float32 in [0, 1].
        """
        if self._mean_color is None:
            if self.geometry == "sphere":
                # Weight rows by their solid angle: an equirect image
                # oversamples the poles badly, and an unweighted mean would be
                # pulled toward whatever is at the zenith.
                height = self._equirect.shape[0]
                rows = (np.arange(height, dtype=np.float64) + 0.5) / height
                weights = np.sin(rows * np.pi)
                mean = (
                    (self._equirect.astype(np.float64).mean(axis=1)
                     * weights[:, None]).sum(axis=0) / weights.sum()
                )
            else:
                mean = np.mean(
                    [f.astype(np.float64).mean(axis=(0, 1)) for f in self._faces],
                    axis=0,
                )
            self._mean_color = (mean / 255.0).astype(np.float32)
        return self._mean_color

    def render(self, camera: Camera) -> NDArray[np.uint8]:
        """
        Render the environment as seen from a camera.

        Args:
            camera: Camera to render from.

        Returns:
            uint8 RGB, shape (camera.height, camera.width, 3).

        Raises:
            ValueError: If a finite surface does not enclose the camera.
        """
        self._fit_resolution(camera)

        # Kept in a local, not folded into the call: the fade is measured
        # against world-space rays, whereas _surface_vectors() rotates them
        # into the environment's own frame.
        dirs = self._world_rays(camera)
        maps = self._sampling_maps(camera, self._surface_vectors(camera, dirs))

        image = self._sample(camera, maps)

        if self.fade is None:
            return image

        # The plain backdrop is sampled with the *same* maps, so the fade
        # dissolves the pattern into the shading that was behind it rather
        # than smearing the two together.
        plain = (
            self._sample(camera, maps, plain=True)
            if self.fade.target == "plain" else None
        )
        return self.fade.apply(
            image, camera, dirs,
            default_color=self._texture_mean(),
            plain=plain,
        )

    def _sampling_maps(
        self,
        camera: Camera,
        vectors: NDArray[np.float32],
    ) -> Tuple[Any, ...]:
        """
        Texture-space sampling coordinates for every pixel.

        Computed once and reused for both the textured and the plain backdrop,
        which is what guarantees the two line up exactly.

        Args:
            camera: Camera being rendered.
            vectors: (H, W, 3) surface lookup vectors.

        Returns:
            ``(map_x, map_y)`` for a sphere, or ``(map_x, map_y, face_index)``
            for a cube.
        """
        if self.geometry == "sphere":
            height, width = self._equirect.shape[:2]
            u, v = equirect_uv(vectors)
            # +0.5 rather than -0.5: the one-texel pad shifts every index by 1.
            return ((u * width + 0.5).astype(np.float32),
                    (v * height + 0.5).astype(np.float32))

        face_index, s, t = cube_face_uv(vectors)
        size = self._faces[0].shape[0]
        return ((s * size + 0.5).astype(np.float32),
                (t * size + 0.5).astype(np.float32),
                face_index)

    def _sample(
        self,
        camera: Camera,
        maps: Tuple[Any, ...],
        plain: bool = False,
    ) -> NDArray[np.uint8]:
        """
        Look up one texture through pre-computed sampling maps.

        Args:
            camera: Camera being rendered, for the output size.
            maps: From :meth:`_sampling_maps`.
            plain: Sample the pattern-free variant instead.

        Returns:
            uint8 RGB, shape (camera.height, camera.width, 3).
        """
        import cv2

        padded = self._padded_texture(plain=plain)

        if self.geometry == "sphere":
            map_x, map_y = maps
            return cv2.remap(padded, map_x, map_y, cv2.INTER_LINEAR)

        map_x, map_y, face_index = maps
        image = np.empty((camera.height, camera.width, 3), dtype=np.uint8)
        for i, face_texture in enumerate(padded):
            mask = face_index == i
            if not mask.any():
                continue
            # Sampling the whole frame per face and masking beats gathering
            # scattered pixels: remap is vectorized and there are only six.
            image[mask] = cv2.remap(
                face_texture, map_x, map_y, cv2.INTER_LINEAR
            )[mask]
        return image

    def composite(
        self,
        image: NDArray[np.uint8],
        camera: Camera,
    ) -> NDArray[np.uint8]:
        """
        Alpha-blend an RGBA layer over the environment.

        Must be applied to the *base* layer, before any overlay is drawn. The
        skeleton overlay writes RGB without touching alpha, so compositing the
        background afterwards would blend the skeleton away everywhere outside
        the silhouette.

        Args:
            image: RGBA uint8, shape (H, W, 4). Modified in place.
            camera: Camera the layer was rendered from.

        Returns:
            ``image``.

        Raises:
            ValueError: If ``image`` is not RGBA, if it disagrees with the
                camera's size, or if a finite surface excludes the camera.
        """
        if image.ndim != 3 or image.shape[2] != 4:
            raise ValueError(
                f"Background compositing needs an RGBA image, got shape "
                f"{image.shape}"
            )
        if image.shape[:2] != (camera.height, camera.width):
            raise ValueError(
                f"Image is {image.shape[1]}x{image.shape[0]} but the camera is "
                f"{camera.width}x{camera.height}"
            )

        env = self.render(camera)

        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        image[:, :, :3] = (
            image[:, :, :3] * alpha + env * (1.0 - alpha)
        ).astype(np.uint8)

        if self.opaque:
            image[:, :, 3] = 255

        return image

    def describe(self) -> str:
        """One-line summary, for verbose CLI output."""
        if self.geometry == "sphere":
            size = f"{self._equirect.shape[1]}x{self._equirect.shape[0]} equirect"
        else:
            size = f"6 x {self._faces[0].shape[0]}px faces"
        extent = "infinite" if self.radius is None else f"radius {self.radius:.3f}"
        fade = "" if self.fade is None else f", {self.fade.describe()}"
        return (
            f"{self.geometry} ({extent}, {size}, "
            f"rotation {self.rotation_deg:g} deg, "
            f"{'opaque' if self.opaque else 'alpha preserved'}{fade})"
        )

    def __repr__(self) -> str:
        return f"Background({self.describe()})"
