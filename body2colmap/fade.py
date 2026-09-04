"""
Fading the backdrop out around the subject.

A world-fixed backdrop tells a video diffusion model that the *camera* is
moving (see :mod:`body2colmap.background`).  But in ``outline`` modes it says
something else as well: the grid runs right up to the silhouette, so the model
reads the silhouette as a hard occlusion boundary and refuses to paint outside
it.  Bulky clothing and hair get squashed back inside the outline of the bare
mesh.

This module fades the backdrop toward a flat colour in a shell around the
subject, so the frame keeps its rotation cue in the far field while leaving a
structure-free zone next to the silhouette for the model to expand into.

The shell is defined by a **bounding ellipsoid** of the mesh, not by the
silhouette itself:

* An ellipsoid that encloses the mesh encloses its silhouette from *every*
  viewpoint, so the clear zone never falls inside the outline on some frame of
  the orbit.
* It is one fixed world-space object, so the clear zone is a stable 3-D region
  the camera moves around, not a per-frame screen effect that would swim.
* Its projection is a conic, so the per-pixel test is a single closest-approach
  computation in the space where the ellipsoid is the unit sphere -- no
  distance transform, no silhouette rasterization.

A capsule (segment plus radius) was the other candidate shape.  It hugs a
standing figure slightly better along the spine, but it does not linearize:
the "how far outside am I" scalar stops being a norm and the ray test grows a
segment-to-line case.  The ellipsoid's ``margin`` knob covers the same ground.

All geometry is in world/renderer coordinates (Y-up, camera looks down -Z).
"""

import math
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .camera import Camera


#: Fade profile used when none is named.  Smoothstep is C1 at both ends, so
#: neither the onset of the fade nor its outer edge leaves a visible ring.
DEFAULT_PROFILE: str = "smoothstep"

#: Width of the fade band, as a multiple of the ellipsoid's own radius in the
#: direction concerned.  1.0 means the fade runs out at twice the subject's
#: extent, which is scale-free: it tracks the subject's size in frame.
DEFAULT_FALLOFF: float = 1.0

#: Shape constant for the profiles that take one (exponential, gaussian,
#: inverse_square).  Ignored by the rest.
DEFAULT_RATE: float = 4.0

#: Ellipsoid inflation before the fade is measured.  1.0 is the fitted hull.
DEFAULT_MARGIN: float = 1.0

#: Long-side resolution, in pixels, that the backdrop is area-averaged down to
#: for ``target="local"``.  Well below the grid's own frequency, so the lines
#: disappear and only the wall/floor/ceiling tone survives.
DEFAULT_DETAIL: int = 24

#: Points sampled from the mesh for the ellipsoid fit.  The result is checked
#: and, if need be, inflated against *all* points afterwards, so this only
#: trades tightness for speed -- never enclosure.
DEFAULT_FIT_POINTS: int = 4000


# ---------------------------------------------------------------------------
# Decay profiles
# ---------------------------------------------------------------------------
#
# Each takes the normalized distance beyond the ellipsoid surface,
#
#     u = (m - 1) / falloff,   clamped at 0,
#
# where m is the ray's closest approach to the ellipsoid centre measured in
# units of the ellipsoid radius (so m = 1 is the surface), and returns the
# fade weight: 1 = fully replaced by the fade colour, 0 = backdrop untouched.
#
# All satisfy w(0) = 1 and decrease monotonically.  They differ in where they
# reach zero -- linear/smoothstep/cosine/step do so exactly at u = 1, while
# exponential, gaussian and inverse_square have tails that never quite reach
# it.  inverse_square's tail is the heavy one: at the default rate it is still
# at 2.7% of full fade at u = 3, which shows up as a faint wash over the whole
# frame.  That is the profile's character, not a bug, and it is why the
# compact profiles are the ones on by default.


def _step(u: NDArray[np.float32], rate: float) -> NDArray[np.float32]:
    """Hard cut: full fade inside ``u < 1``, nothing outside. The control."""
    return (u < 1.0).astype(np.float32)


def _linear(u: NDArray[np.float32], rate: float) -> NDArray[np.float32]:
    """Straight ramp to zero at ``u = 1``. Leaves a slope discontinuity."""
    return np.clip(1.0 - u, 0.0, 1.0)


def _smoothstep(u: NDArray[np.float32], rate: float) -> NDArray[np.float32]:
    """Hermite ramp, flat at both ends. The default."""
    x = np.clip(1.0 - u, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _cosine(u: NDArray[np.float32], rate: float) -> NDArray[np.float32]:
    """Raised cosine. Like smoothstep but steeper through the middle."""
    x = np.clip(u, 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * x))


def _exponential(u: NDArray[np.float32], rate: float) -> NDArray[np.float32]:
    """``exp(-rate * u)``: steepest at the silhouette, long thin tail."""
    return np.exp(-rate * u)


def _gaussian(u: NDArray[np.float32], rate: float) -> NDArray[np.float32]:
    """``exp(-rate * u^2)``: flat right at the silhouette, then falls away."""
    return np.exp(-rate * u * u)


def _inverse_square(u: NDArray[np.float32], rate: float) -> NDArray[np.float32]:
    """``1 / (1 + rate * u^2)``: the heaviest tail of the set."""
    return 1.0 / (1.0 + rate * u * u)


#: Fade profiles, by the name used in config and on the CLI.
DECAY_PROFILES: Dict[str, Callable[[NDArray[np.float32], float], NDArray[np.float32]]] = {
    "step": _step,
    "linear": _linear,
    "smoothstep": _smoothstep,
    "cosine": _cosine,
    "exponential": _exponential,
    "gaussian": _gaussian,
    "inverse_square": _inverse_square,
}

#: How the faded region is coloured.
FADE_TARGETS: Tuple[str, ...] = ("local", "color")


def decay_weight(
    u: NDArray[np.float32],
    profile: str = DEFAULT_PROFILE,
    rate: float = DEFAULT_RATE,
) -> NDArray[np.float32]:
    """
    Evaluate a named decay profile.

    Args:
        u: Normalized distance beyond the ellipsoid surface, >= 0.
        profile: A key of :data:`DECAY_PROFILES`.
        rate: Shape constant, used by exponential / gaussian / inverse_square.

    Returns:
        Fade weights in [0, 1], same shape as ``u``.

    Raises:
        ValueError: If the profile is unknown or ``rate`` is not positive.
    """
    try:
        fn = DECAY_PROFILES[profile]
    except KeyError:
        raise ValueError(
            f"Unknown fade profile {profile!r}. Choose from: "
            f"{', '.join(sorted(DECAY_PROFILES))}"
        ) from None

    if rate <= 0.0:
        raise ValueError(f"Fade rate must be > 0, got {rate}")

    weight = fn(np.asarray(u, dtype=np.float32), float(rate))
    return np.clip(weight, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Bounding ellipsoid
# ---------------------------------------------------------------------------

class Ellipsoid:
    """
    An ellipsoid, stored as the linear map that takes it to the unit sphere.

    A point ``p`` is inside when ``|transform @ (p - center)| <= 1``.  Keeping
    the map rather than (axes, rotation) is what makes the ray test cheap:
    transform the ray, and the ellipsoid problem becomes a sphere problem.

    Attributes:
        center: World-space centre, shape (3,).
        transform: World -> normalized space, shape (3, 3).
    """

    def __init__(
        self,
        center: NDArray[np.float32],
        transform: NDArray[np.float32],
    ):
        """
        Args:
            center: World-space centre, shape (3,).
            transform: Invertible 3x3 mapping world offsets to a space in
                which the ellipsoid is the unit sphere.

        Raises:
            ValueError: If ``transform`` is singular or misshapen.
        """
        self.center = np.asarray(center, dtype=np.float64).reshape(3)
        self.transform = np.asarray(transform, dtype=np.float64).reshape(3, 3)

        singular = np.linalg.svd(self.transform, compute_uv=False)
        if not np.all(singular > 1e-12):
            raise ValueError(
                f"Ellipsoid transform is singular (singular values "
                f"{singular}); the fitted points are probably degenerate."
            )
        #: Semi-axis lengths, longest first.
        self.axes = (1.0 / singular).astype(np.float64)

    # -- construction -------------------------------------------------------

    @classmethod
    def fit(
        cls,
        points: NDArray[np.float32],
        margin: float = DEFAULT_MARGIN,
        tol: float = 1e-3,
        max_iter: int = 500,
        max_points: int = DEFAULT_FIT_POINTS,
    ) -> "Ellipsoid":
        """
        Minimum-volume enclosing ellipsoid of a point set (Khachiyan).

        Every returned ellipsoid encloses **all** of ``points``, whatever the
        solver did.  The iteration is only asked for a good shape; enclosure
        is then imposed exactly, by scaling the result until the outermost
        point sits on the surface.  That is what lets the fit run on a
        subsample without any risk of clipping a stray vertex -- the check
        is O(N) and runs on the full set.

        Args:
            points: (N, 3) points to enclose.
            margin: Inflate the fitted semi-axes by this factor. 1.0 is the
                tight hull; > 1.0 pushes the fade further out.
            tol: Khachiyan convergence tolerance on the relative excess.
            max_iter: Iteration cap.
            max_points: Subsample the fit to at most this many points, by a
                deterministic stride. Enclosure is still checked against all.

        Returns:
            An Ellipsoid containing every point in ``points``.

        Raises:
            ValueError: If fewer than four points are given, if they are
                coplanar (no ellipsoid of positive volume encloses them
                tightly), or if ``margin`` is not positive.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] < 4:
            raise ValueError(
                f"Fitting an ellipsoid needs at least 4 points, got "
                f"{pts.shape[0]}"
            )
        if margin <= 0.0:
            raise ValueError(f"Ellipsoid margin must be > 0, got {margin}")

        sample = pts
        if pts.shape[0] > max_points:
            sample = pts[:: max(1, pts.shape[0] // max_points)]

        center, shape = _khachiyan(sample, tol=tol, max_iter=max_iter)

        # Impose enclosure on the full set. `shape` defines the ellipsoid as
        # (p - c)' shape (p - c) <= 1, so dividing it by the worst point's
        # value puts that point exactly on the surface.
        offsets = pts - center
        worst = float(np.max(np.einsum("ij,jk,ik->i", offsets, shape, offsets)))
        if worst > 1.0:
            shape = shape / worst

        # transform = shape^(1/2), via the eigendecomposition of a symmetric
        # positive-definite matrix: shape = V diag(w) V', so |diag(sqrt w) V' x|^2
        # is exactly x' shape x.
        shape = 0.5 * (shape + shape.T)
        eigenvalues, eigenvectors = np.linalg.eigh(shape)
        if np.any(eigenvalues <= 0.0):
            raise ValueError(
                "Ellipsoid fit collapsed to zero volume; the points are "
                "coplanar or nearly so."
            )
        transform = (np.sqrt(eigenvalues)[:, None] * eigenvectors.T) / margin

        return cls(center, transform)

    @classmethod
    def from_bounds(
        cls,
        min_corner: NDArray[np.float32],
        max_corner: NDArray[np.float32],
        margin: float = DEFAULT_MARGIN,
    ) -> "Ellipsoid":
        """
        The axis-aligned ellipsoid circumscribing a bounding box.

        A fallback for callers with bounds but no vertices. Noticeably looser
        than :meth:`fit` -- the box's corners force each semi-axis out by
        sqrt(3) -- so prefer the fit where the points are available.

        Args:
            min_corner: Box minimum, shape (3,).
            max_corner: Box maximum, shape (3,).
            margin: Extra inflation on top.

        Returns:
            An Ellipsoid containing the box.
        """
        lo = np.asarray(min_corner, dtype=np.float64).reshape(3)
        hi = np.asarray(max_corner, dtype=np.float64).reshape(3)
        half = np.maximum((hi - lo) / 2.0, 1e-9) * math.sqrt(3.0) * margin
        return cls((lo + hi) / 2.0, np.diag(1.0 / half))

    def scaled(self, margin: float) -> "Ellipsoid":
        """
        A copy inflated by ``margin`` about the same centre.

        Args:
            margin: Multiplier on every semi-axis. Must be > 0.

        Returns:
            A new Ellipsoid.

        Raises:
            ValueError: If ``margin`` is not positive.
        """
        if margin <= 0.0:
            raise ValueError(f"Ellipsoid margin must be > 0, got {margin}")
        return Ellipsoid(self.center, self.transform / margin)

    # -- queries ------------------------------------------------------------

    def normalized_distance(
        self, points: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """
        Distance of points from the centre, in units of the ellipsoid radius.

        Args:
            points: (..., 3) world positions.

        Returns:
            (...) values; 1.0 exactly on the surface, < 1 inside.
        """
        offsets = np.asarray(points, dtype=np.float64) - self.center
        return np.linalg.norm(offsets @ self.transform.T, axis=-1).astype(np.float32)

    def contains(self, points: NDArray[np.float32], tol: float = 1e-6) -> NDArray[np.bool_]:
        """
        Whether points lie inside the ellipsoid.

        Args:
            points: (..., 3) world positions.
            tol: Slack on the surface test.

        Returns:
            (...) booleans.
        """
        return self.normalized_distance(points) <= 1.0 + tol

    def ray_distance(
        self,
        origin: NDArray[np.float32],
        directions: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Closest approach of each ray to the ellipsoid, in radius units.

        This is the whole screen-space test.  In the normalized space the
        ellipsoid is the unit sphere, so a ray's closest approach to the
        centre is a two-line projection, and the value it returns is exactly
        "how far outside the silhouette is this pixel", measured in a way that
        does not depend on which direction the camera is looking from.

        The ray is treated as forward-only: a direction pointing away from the
        subject reports the distance at the camera itself, not the distance to
        the line extended backwards.

        Args:
            origin: Ray origin (the camera centre), shape (3,).
            directions: (..., 3) ray directions. Need not be unit length.

        Returns:
            (...) closest approach; <= 1 means the ray meets the ellipsoid, so
            the pixel is inside the subject's silhouette.
        """
        o = (np.asarray(origin, dtype=np.float64).reshape(3) - self.center) @ self.transform.T
        d = np.asarray(directions, dtype=np.float64) @ self.transform.T

        dd = np.einsum("...i,...i->...", d, d)
        od = np.einsum("...i,...i->...", d, o)

        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(dd > 1e-30, -od / dd, 0.0)
        t = np.maximum(t, 0.0)

        closest = o + t[..., None] * d
        return np.linalg.norm(closest, axis=-1).astype(np.float32)

    def describe(self) -> str:
        """One-line summary, for verbose CLI output."""
        axes = ", ".join(f"{a:.3f}" for a in sorted(self.axes, reverse=True))
        center = ", ".join(f"{c:.3f}" for c in self.center)
        return f"ellipsoid axes ({axes}) at ({center})"

    def __repr__(self) -> str:
        return f"Ellipsoid({self.describe()})"


def _khachiyan(
    points: NDArray[np.float64],
    tol: float,
    max_iter: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Khachiyan's barycentric-coordinate descent for the MVEE.

    Args:
        points: (N, 3) float64 points.
        tol: Stop when the worst point's excess falls below this.
        max_iter: Iteration cap. Hitting it is not an error -- the caller
            imposes enclosure afterwards regardless.

    Returns:
        ``(center, shape)`` with the ellipsoid ``(p - c)' shape (p - c) <= 1``.

    Raises:
        ValueError: If the point set is degenerate enough that the dual
            problem is singular.
    """
    n, d = points.shape
    # Lift to homogeneous coordinates: the MVEE of `points` is the minimum
    # enclosing *cone* section of these, which is what makes the update below
    # a simple reweighting.
    q = np.concatenate([points, np.ones((n, 1))], axis=1).T  # (d+1, N)
    weights = np.full(n, 1.0 / n)

    for _ in range(max_iter):
        moment = (q * weights) @ q.T                          # (d+1, d+1)
        try:
            solved = np.linalg.solve(moment, q)               # (d+1, N)
        except np.linalg.LinAlgError:
            raise ValueError(
                "Ellipsoid fit failed: the point set is degenerate (coplanar "
                "or collinear), so no enclosing ellipsoid of positive volume "
                "exists."
            ) from None

        excess = np.einsum("ij,ij->j", q, solved)             # (N,)
        worst = int(np.argmax(excess))
        peak = float(excess[worst])

        step = (peak - d - 1.0) / ((d + 1.0) * (peak - 1.0))
        if not np.isfinite(step) or step <= 0.0:
            break

        weights *= 1.0 - step
        weights[worst] += step

        if peak / (d + 1.0) - 1.0 < tol:
            break

    center = points.T @ weights
    covariance = (points.T * weights) @ points - np.outer(center, center)
    try:
        shape = np.linalg.inv(covariance) / d
    except np.linalg.LinAlgError:
        raise ValueError(
            "Ellipsoid fit failed: the point set is degenerate (coplanar or "
            "collinear), so no enclosing ellipsoid of positive volume exists."
        ) from None

    return center, shape


# ---------------------------------------------------------------------------
# The fade itself
# ---------------------------------------------------------------------------

class SubjectFade:
    """
    Fades a rendered backdrop toward a flat colour around the subject.

    Applied by :class:`~body2colmap.background.Background` to its own render,
    before the base layer is composited over it -- so the fade only ever
    touches the backdrop, never the subject.

    The clear zone is the projection of an :class:`Ellipsoid` enclosing the
    mesh, widened by a decay band whose width scales with the subject rather
    than with the frame, so it holds up across an auto-framed orbit.

    Attributes:
        ellipsoid: The enclosing ellipsoid, already inflated by any margin.
        profile: Decay profile name, a key of :data:`DECAY_PROFILES`.
        falloff: Band width, in multiples of the ellipsoid radius.
        rate: Shape constant for the profiles that take one.
        target: "local" or "color"; see :meth:`__init__`.
        color: Explicit fade colour as RGB floats in [0, 1], or None.
        detail: Long-side resolution for the "local" average, in pixels.
    """

    def __init__(
        self,
        ellipsoid: Ellipsoid,
        profile: str = DEFAULT_PROFILE,
        falloff: float = DEFAULT_FALLOFF,
        rate: float = DEFAULT_RATE,
        target: str = "local",
        color: Optional[Sequence[float]] = None,
        detail: int = DEFAULT_DETAIL,
    ):
        """
        Args:
            ellipsoid: Ellipsoid enclosing the subject, in world coordinates.
            profile: Decay profile, a key of :data:`DECAY_PROFILES`.
            falloff: Width of the fade band, as a multiple of the ellipsoid's
                radius. Scale-free: the band tracks the subject's size in
                frame rather than a pixel count.
            rate: Shape constant for exponential / gaussian / inverse_square.
            target: What the backdrop fades *to*. ``"local"`` uses the
                backdrop's own colour with its detail averaged away, so the
                lines vanish and the wall/floor/ceiling tone carries through
                with no visible patch. ``"color"`` uses one flat colour for
                the whole clear zone.
            color: RGB floats in [0, 1] for ``target="color"``. None means
                the caller supplies a default -- in practice the texture's
                own mean colour.
            detail: Long-side resolution the backdrop is area-averaged down
                to for ``target="local"``. Must be >= 1, and wants to be well
                below the texture's own frequency.

        Raises:
            ValueError: On an unknown profile or target, a non-positive
                falloff, rate or detail, or a malformed colour.
        """
        if profile not in DECAY_PROFILES:
            raise ValueError(
                f"Unknown fade profile {profile!r}. Choose from: "
                f"{', '.join(sorted(DECAY_PROFILES))}"
            )
        if target not in FADE_TARGETS:
            raise ValueError(
                f"Unknown fade target {target!r}. Use "
                f"{' or '.join(repr(t) for t in FADE_TARGETS)}."
            )
        if falloff <= 0.0:
            raise ValueError(
                f"Fade falloff must be > 0, got {falloff}. Use "
                f"profile='step' for a hard-edged clear zone."
            )
        if rate <= 0.0:
            raise ValueError(f"Fade rate must be > 0, got {rate}")
        if detail < 1:
            raise ValueError(f"Fade detail must be >= 1 pixel, got {detail}")

        self.ellipsoid = ellipsoid
        self.profile = profile
        self.falloff = float(falloff)
        self.rate = float(rate)
        self.target = target
        self.detail = int(detail)

        if color is None:
            self.color = None
        else:
            rgb = np.asarray(color, dtype=np.float32).reshape(-1)
            if rgb.size != 3:
                raise ValueError(
                    f"Fade colour must be 3 RGB floats, got {rgb.size} values"
                )
            if np.any(rgb < 0.0) or np.any(rgb > 1.0):
                raise ValueError(
                    f"Fade colour components must be in [0, 1], got {color}"
                )
            self.color = rgb

    # -- evaluation ---------------------------------------------------------

    def weights(
        self,
        camera: Camera,
        directions: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Per-pixel fade weight for one view.

        Args:
            camera: Camera being rendered. Only its position is used; the
                directions carry the rest.
            directions: (H, W, 3) world-space ray directions, *before* any
                environment rotation -- the ellipsoid lives in world space,
                not in the backdrop's rotated frame.

        Returns:
            (H, W) float32 in [0, 1]. 1 inside the subject's silhouette,
            falling to 0 across the band.
        """
        m = self.ellipsoid.ray_distance(camera.position, directions)
        u = np.maximum(m - 1.0, 0.0) / self.falloff
        return decay_weight(u, self.profile, self.rate)

    def apply(
        self,
        image: NDArray[np.uint8],
        camera: Camera,
        directions: NDArray[np.float32],
        default_color: Optional[Sequence[float]] = None,
    ) -> NDArray[np.uint8]:
        """
        Fade a rendered backdrop around the subject.

        Args:
            image: uint8 RGB backdrop, shape (H, W, 3).
            camera: Camera the backdrop was rendered from.
            directions: (H, W, 3) world ray directions, as for :meth:`weights`.
            default_color: RGB floats in [0, 1] used when ``target="color"``
                and no explicit colour was set. None falls back to the mean
                of ``image``.

        Returns:
            A new uint8 RGB image.
        """
        weight = self.weights(camera, directions)[..., None]
        if not np.any(weight > 0.0):
            return image

        base = image.astype(np.float32)
        target = self._target_image(base, default_color)
        faded = base * (1.0 - weight) + target * weight
        return np.clip(faded + 0.5, 0, 255).astype(np.uint8)

    def _target_image(
        self,
        base: NDArray[np.float32],
        default_color: Optional[Sequence[float]],
    ) -> NDArray[np.float32]:
        """
        The colour the backdrop fades to, as a float image or a broadcastable
        triple.

        Args:
            base: The backdrop render as float32 RGB in [0, 255].
            default_color: Fallback colour in [0, 1], or None.

        Returns:
            Either (H, W, 3) or (3,), in [0, 255].
        """
        if self.target == "local":
            import cv2

            height, width = base.shape[:2]
            long_side = max(height, width)
            # Area-average down and bilinearly back up. A box average below
            # the texture's own frequency erases the lines while preserving
            # local tone exactly, so the clear zone has no edge against the
            # surrounding backdrop -- which a single flat colour cannot manage
            # across a cube's floor/wall/ceiling split.
            small_w = max(1, int(round(width * self.detail / long_side)))
            small_h = max(1, int(round(height * self.detail / long_side)))
            small = cv2.resize(base, (small_w, small_h), interpolation=cv2.INTER_AREA)
            return cv2.resize(
                small, (width, height), interpolation=cv2.INTER_LINEAR
            ).reshape(height, width, 3)

        if self.color is not None:
            return self.color * 255.0
        if default_color is not None:
            return np.asarray(default_color, dtype=np.float32).reshape(3) * 255.0
        return base.reshape(-1, 3).mean(axis=0)

    def describe(self) -> str:
        """One-line summary, for verbose CLI output."""
        if self.target == "local":
            to = f"local tone ({self.detail}px average)"
        elif self.color is not None:
            to = "colour (" + ", ".join(f"{c:.2f}" for c in self.color) + ")"
        else:
            to = "mean texture colour"
        shaped = self.profile in ("exponential", "gaussian", "inverse_square")
        rate = f", rate {self.rate:g}" if shaped else ""
        return (
            f"{self.profile} fade to {to} over {self.falloff:g}x"
            f"{rate}, around {self.ellipsoid.describe()}"
        )

    def __repr__(self) -> str:
        return f"SubjectFade({self.describe()})"
