"""
Anchor an externally-produced Gaussian splat to the body2colmap world frame.

This is COORDINATE CONVERSION POINT #3 (the others are
``Scene.from_sam3d_output()`` and ``ColmapExporter.export()``).  Like those, it
lives at a system boundary and is the only place the incoming splat's
coordinates are touched.

The splat is produced externally from the *same photograph* that feeds
SAM-3D-Body -- see ``~/Projects/masktest`` (Sapiens2 seg + pointmap + normal ->
normal-integrated depth -> one oriented Gaussian per masked pixel).  That
pipeline emits a standard 3DGS ``.ply`` plus a ``splat_meta.json``.

Why this is tractable at all
----------------------------
Both frames are already the same convention -- Y up, +Z toward the viewer --
and in both the source camera has *identity* rotation:

* **splat frame**: ``p_world = F @ p_cam - centroid`` with ``F = diag(1,-1,-1)``,
  so the source camera sits at ``-centroid`` with identity OpenGL c2w rotation.
  It is identity *by construction*: the world is defined as the pointmap frame
  flipped and recentred, so no pose was ever estimated.
* **body2colmap frame**: ``sam3d_to_world()`` leaves the SAM-3D-Body camera at
  the origin with identity rotation.

So the two differ only by the centroid recentring, the two independently
recovered focal lengths, and one scale gauge.

The transform
-------------
``P_world = M @ (p_ply + centroid)``

Step 1 undoes the recentring, putting the splat in the source-camera frame with
the camera at the origin.  Step 2 is a single 3x3 upper-triangular ``M`` that
makes every Gaussian reproject onto the pixel it actually came from, now under
SAM-3D-Body's intrinsics:

    a  = s * f_s / f_m
    bx = s * (cx_s - cx_m) / f_m
    by = s * (cy_s - cy_m) / f_m

            [ a   0  -bx ]
    M   =   [ 0   a   by ]
            [ 0   0   s  ]

Derived in OpenCV coordinates and mapped back through ``(X, -Y, -Z)``, matching
the flip ``Camera.project()`` applies.

**``M`` is not an arbitrary distortion.**  With ``s = 1`` it is exactly
"re-unproject the splat's depth map using the correct focal length":

    P' = ((u - cx_m) * z / f_m,  (v - cy_m) * z / f_m,  z)

The pointmap network commits to its own implicit focal, inferred from image
content, and on a tight face crop that focal is not the real camera's.  Keeping
the predicted *depths* and re-unprojecting them under SAM-3D-Body's focal is the
geometrically correct correction, not a fudge.  It is anisotropic (depth scales
by ``s``, lateral extent by ``a``) precisely because the two focals disagree.

``s`` is then a pure metric-scale gauge.  Scaling about the camera centre leaves
every projection unchanged, so ``s`` is invisible at the anchor frame -- but it
sets how far along each ray the splat sits, which is what governs whether the
splat and the skeleton stay together as the orbit moves away from the anchor.
:func:`estimate_depth_scale` fits it by comparing the two front surfaces.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .camera import Camera
from .splat_scene import SplatScene

__all__ = [
    "load_splat_meta",
    "compute_anchor_transform",
    "estimate_depth_scale",
    "transform_splat_scene",
    "anchor_splat_to_world",
]


def load_splat_meta(filepath: str) -> Dict[str, Any]:
    """
    Load and validate a ``splat_meta.json`` written alongside the splat PLY.

    Args:
        filepath: Path to the JSON file.

    Returns:
        The parsed dictionary.

    Raises:
        ValueError: If a required key is missing.
    """
    meta = json.loads(Path(filepath).read_text())

    missing = [k for k in ("intrinsics", "centroid", "width", "height") if k not in meta]
    if missing:
        raise ValueError(
            f"{filepath} is missing required key(s): {', '.join(missing)}. "
            "Expected a splat_meta.json as written by masktest's face_to_splat.py."
        )
    missing = [k for k in ("f", "cx", "cy") if k not in meta["intrinsics"]]
    if missing:
        raise ValueError(
            f"{filepath}: 'intrinsics' is missing {', '.join(missing)}."
        )
    return meta


def compute_anchor_transform(
    splat_meta: Dict[str, Any],
    original_focal_length: float,
    original_image_size: Tuple[int, int],
    crop_box: Optional[Tuple[int, int, int, int]] = None,
    scale: float = 1.0,
    reconcile_intrinsics: bool = True,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Build the linear map that carries splat coordinates into world coordinates.

    Args:
        splat_meta: Parsed ``splat_meta.json`` (see :func:`load_splat_meta`).
        original_focal_length: SAM-3D-Body's ``focal_length`` from the .npz, in
            pixels of the *full* original image.  Its principal point is the
            image centre (SAM-3D-Body takes both from the MoGe FoV estimator,
            which returns a centred principal point, and ``pred_cam_t`` places
            the mesh so it projects into full-image pixels).
        original_image_size: ``(width, height)`` of that full original image.
        crop_box: ``(x0, y0, x1, y1)`` in full-image pixels, giving the region
            the splat's input image was cut from.  ``None`` means the splat was
            built from the full image, in which case its recorded size must
            match ``original_image_size``.  A crop that was also resized is
            handled: the resize factor is derived from the box against
            ``splat_meta['width']``.
        scale: The depth gauge ``s``.  See module docstring;
            :func:`estimate_depth_scale` fits it.
        reconcile_intrinsics: When False, ignore the splat's fitted intrinsics
            and use a pure uniform scale ``diag(s, s, s)``.  This preserves the
            splat's shape exactly at the cost of a size mismatch against the
            skeleton at the anchor frame.  Only useful when you have reason to
            trust the splat's focal over SAM-3D-Body's.

    Returns:
        ``(M, centroid)`` -- a 3x3 float64 matrix and the 3-vector to add to the
        PLY coordinates before applying it.

    Raises:
        ValueError: If ``crop_box`` is absent and the sizes disagree, or if the
            crop box is degenerate or non-uniformly scaled.
    """
    w_o, h_o = int(original_image_size[0]), int(original_image_size[1])
    w_s = int(splat_meta["width"])
    h_s = int(splat_meta["height"])

    f_s = float(splat_meta["intrinsics"]["f"])
    cx_s = float(splat_meta["intrinsics"]["cx"])
    cy_s = float(splat_meta["intrinsics"]["cy"])

    # --- Express the splat's intrinsics in the full original image's pixel grid ---
    if crop_box is None:
        if (w_s, h_s) != (w_o, h_o):
            raise ValueError(
                f"The splat was built from a {w_s}x{h_s} image but SAM-3D-Body saw "
                f"{w_o}x{h_o}. Pass the crop box the splat's input was cut from "
                "(--splat-crop x0,y0,x1,y1 in full-image pixels), or rebuild the "
                "splat from the full image."
            )
        x0 = y0 = 0.0
        r_x = r_y = 1.0
    else:
        x0, y0, x1, y1 = (float(v) for v in crop_box)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Degenerate crop_box: {crop_box}")
        r_x = (x1 - x0) / w_s   # full-image px per splat-image px
        r_y = (y1 - y0) / h_s
        if not np.isclose(r_x, r_y, rtol=1e-3):
            raise ValueError(
                f"crop_box {crop_box} implies a non-uniform resize "
                f"({r_x:.4f} horizontally vs {r_y:.4f} vertically). The splat's "
                "input must keep the original aspect ratio."
            )

    r = 0.5 * (r_x + r_y)
    f_s_o = f_s * r
    cx_s_o = cx_s * r + x0
    cy_s_o = cy_s * r + y0

    # --- SAM-3D-Body's intrinsics: focal from the .npz, principal point centred ---
    f_m = float(original_focal_length)
    cx_m, cy_m = w_o / 2.0, h_o / 2.0

    if not reconcile_intrinsics:
        f_s_o, cx_s_o, cy_s_o = f_m, cx_m, cy_m

    s = float(scale)
    a = s * f_s_o / f_m
    bx = s * (cx_s_o - cx_m) / f_m
    by = s * (cy_s_o - cy_m) / f_m

    M = np.array([
        [a,   0.0, -bx],
        [0.0, a,    by],
        [0.0, 0.0,  s ],
    ], dtype=np.float64)

    centroid = np.asarray(splat_meta["centroid"], dtype=np.float64).reshape(3)
    return M, centroid


def _quats_to_matrices(quats_wxyz: NDArray[np.floating]) -> NDArray[np.float64]:
    """Batch (N, 4) wxyz unit quaternions -> (N, 3, 3) rotation matrices."""
    q = np.asarray(quats_wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = np.empty((len(q), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _matrices_to_quats(R: NDArray[np.floating]) -> NDArray[np.float64]:
    """
    Batch (N, 3, 3) rotation matrices -> (N, 4) wxyz unit quaternions.

    Uses branchless Shepperd: build all four candidates and take the one with
    the largest denominator per matrix.  The naive trace-only formula loses
    precision and sign as the trace approaches -1, which is common here --
    many surface normals point almost straight down -Z in world space.
    """
    R = np.asarray(R, dtype=np.float64)
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]

    # Four candidate squared-denominators, one per branch.
    t = np.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], axis=1)

    cand = np.stack([
        np.stack([t[:, 0],      m21 - m12,  m02 - m20,  m10 - m01], axis=1),
        np.stack([m21 - m12,    t[:, 1],    m01 + m10,  m02 + m20], axis=1),
        np.stack([m02 - m20,    m01 + m10,  t[:, 2],    m12 + m21], axis=1),
        np.stack([m10 - m01,    m02 + m20,  m12 + m21,  t[:, 3]  ], axis=1),
    ], axis=1)  # (N, 4 branches, 4 components)

    best = np.argmax(t, axis=1)
    q = cand[np.arange(len(R)), best]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    # Canonical sign: w >= 0.
    q = np.where(q[:, :1] < 0, -q, q)
    return q


def transform_splat_scene(
    scene: SplatScene,
    M: NDArray[np.floating],
    centroid: NDArray[np.floating],
) -> SplatScene:
    """
    Apply ``P -> M @ (P + centroid)`` to every Gaussian.

    Means transform directly.  Covariances transform as ``S' = M S M^T`` and are
    re-decomposed into a rotation and three axis scales.

    Args:
        scene: The splat as loaded from the PLY (splat-frame coordinates).
        M: 3x3 linear map from :func:`compute_anchor_transform`.
        centroid: 3-vector added before applying ``M``.

    Returns:
        A new :class:`SplatScene` in world coordinates. ``extras`` are copied
        through row for row and NOT reoriented: they are opaque per-Gaussian
        scalars to this module, so a directional extra (brush's ``ev_dir_*``)
        would come out in the source frame. No pipeline anchors an
        evidence-bearing splat -- those are SH degree 3 and refused below.

    Raises:
        ValueError: If the scene carries higher-order spherical harmonics.
            A general linear map reorients every Gaussian, which would require
            rotating the SH bands too.  The upstream pipeline only ever emits
            degree 0 (one view constrains nothing view-dependent), so this
            raises rather than silently producing wrong view-dependent colour.
    """
    if scene.sh_degree != 0:
        raise ValueError(
            f"Anchoring requires an SH degree 0 splat, got degree {scene.sh_degree}. "
            "The transform reorients every Gaussian, which would also require "
            "rotating the SH bands (not implemented)."
        )

    M = np.asarray(M, dtype=np.float64)
    centroid = np.asarray(centroid, dtype=np.float64).reshape(3)

    means = (np.asarray(scene.means, np.float64) + centroid) @ M.T

    # Uniform-scale fast path: covariances only change size, not orientation,
    # so the quaternions are untouched and the log-scales just shift.
    off_diag = M - np.diag(np.diag(M))
    diag = np.diag(M)
    if np.allclose(off_diag, 0.0, atol=1e-12) and np.allclose(diag, diag[0], rtol=1e-9):
        s = float(diag[0])
        if s <= 0.0:
            raise ValueError(f"Anchor scale must be positive, got {s}")
        return SplatScene(
            means=means.astype(np.float32),
            scales=(np.asarray(scene.scales, np.float64) + np.log(s)).astype(np.float32),
            quats=np.asarray(scene.quats, np.float32).copy(),
            opacities=np.asarray(scene.opacities, np.float32).copy(),
            sh_coeffs=np.asarray(scene.sh_coeffs, np.float32).copy(),
            sh_degree=scene.sh_degree,
            extras={k: v.copy() for k, v in scene.extras.items()},
        )

    R = _quats_to_matrices(scene.quats)                       # (N, 3, 3)
    sigma = np.exp(np.asarray(scene.scales, np.float64))      # (N, 3)

    # Sigma = R diag(sigma^2) R^T, then Sigma' = M Sigma M^T.
    RS = R * sigma[:, None, :]                                # scale each column
    cov = RS @ np.transpose(RS, (0, 2, 1))
    cov = M @ cov @ M.T

    # Symmetrise to kill the asymmetry float error leaves behind, then
    # re-decompose.  eigh returns ascending eigenvalues; reverse to descending
    # so the smallest axis stays in column 2 -- that keeps the upstream
    # "surface normal in column 2" convention that 3DGS normal-supervision
    # trainers assume.
    cov = 0.5 * (cov + np.transpose(cov, (0, 2, 1)))
    evals, evecs = np.linalg.eigh(cov)
    evals = evals[:, ::-1]
    evecs = evecs[:, :, ::-1]

    # eigh gives an orthonormal basis but not necessarily a right-handed one.
    dets = np.linalg.det(evecs)
    evecs[dets < 0, :, 0] *= -1.0

    scales = 0.5 * np.log(np.maximum(evals, 1e-30))
    quats = _matrices_to_quats(evecs)

    return SplatScene(
        means=means.astype(np.float32),
        scales=scales.astype(np.float32),
        quats=quats.astype(np.float32),
        opacities=np.asarray(scene.opacities, np.float32).copy(),
        sh_coeffs=np.asarray(scene.sh_coeffs, np.float32).copy(),
        sh_degree=scene.sh_degree,
        extras={k: v.copy() for k, v in scene.extras.items()},
    )


def estimate_depth_scale(
    means_world: NDArray[np.floating],
    camera: Camera,
    mesh_depth: NDArray[np.floating],
    min_overlap_px: int = 200,
) -> float:
    """
    Fit the depth gauge ``s`` by matching two front surfaces pixel for pixel.

    The splat and the SAM-3D-Body mesh describe the same face seen from the same
    camera, so where both cover a pixel their distances should agree.  Because
    ``s`` scales every splat depth uniformly, the fit is a single robust ratio.

    A median over the overlap is used rather than a least-squares fit: the mesh
    is itself a parametric fit to the photo and does not line up with it
    perfectly, so a handful of pixels straddle a silhouette edge and pair the
    face against the background.  The median ignores them.

    Note this compares like with like.  Comparing bounding-box centres instead
    would be biased by several centimetres, because the mesh head's centre sits
    inside the skull while the splat is a surface.

    Args:
        means_world: Splat means already mapped to world coordinates *at gauge
            1* (i.e. via :func:`compute_anchor_transform` with ``scale=1.0``).
        camera: Camera to compare from -- normally the original SAM-3D-Body
            camera (at the origin, identity rotation, ``focal_length`` from the
            .npz, full image size).
        mesh_depth: The mesh's raw depth buffer from that same camera, as
            returned by ``Renderer._render_depth_buffer()``.  Must match the
            camera's image size.  0 means "no mesh here".
        min_overlap_px: Fail rather than fit on too little evidence.

    Returns:
        The fitted scale ``s``.

    Raises:
        ValueError: If the camera and depth buffer disagree on size, or the two
            surfaces overlap in fewer than ``min_overlap_px`` pixels.
    """
    mesh_depth = np.asarray(mesh_depth)
    h, w = mesh_depth.shape[:2]
    if (w, h) != (camera.width, camera.height):
        raise ValueError(
            f"mesh_depth is {w}x{h} but the camera is "
            f"{camera.width}x{camera.height}"
        )

    means_world = np.asarray(means_world, dtype=np.float64)

    # Splat depth along the view axis, matching the depth buffer's convention.
    w2c = camera.get_w2c()
    pts_cam = (w2c[:3, :3] @ means_world.T).T + w2c[:3, 3]
    depth = -pts_cam[:, 2]                      # camera looks down -Z

    uv = camera.project(means_world.astype(np.float32))
    px = np.round(uv[:, 0]).astype(np.int64)
    py = np.round(uv[:, 1]).astype(np.int64)

    keep = (depth > 0) & (px >= 0) & (px < w) & (py >= 0) & (py < h)
    if not np.any(keep):
        raise ValueError("No splat Gaussians project inside the camera's frame.")

    # Nearest splat surface per pixel. np.minimum.at handles the duplicates that
    # many Gaussians landing on one pixel produce.
    splat_depth = np.full((h, w), np.inf, dtype=np.float64)
    np.minimum.at(splat_depth, (py[keep], px[keep]), depth[keep])

    overlap = np.isfinite(splat_depth) & (mesh_depth > 0)
    n = int(np.count_nonzero(overlap))
    if n < min_overlap_px:
        raise ValueError(
            f"The splat and the mesh overlap in only {n} pixels "
            f"(need {min_overlap_px}). Are they from the same photo, and was "
            "the scene left un-oriented?"
        )

    return float(np.median(mesh_depth[overlap] / splat_depth[overlap]))


def anchor_splat_to_world(
    ply_path: str,
    meta_path: str,
    original_focal_length: float,
    original_image_size: Tuple[int, int],
    crop_box: Optional[Tuple[int, int, int, int]] = None,
    scale: Optional[float] = None,
    reconcile_intrinsics: bool = True,
    depth_probe: Optional[Any] = None,
) -> Tuple[SplatScene, Dict[str, Any]]:
    """
    Load a splat PLY and place it in world coordinates.

    Args:
        ply_path: The 3DGS ``.ply``.
        meta_path: Its ``splat_meta.json``.
        original_focal_length: SAM-3D-Body's ``focal_length`` from the .npz.
        original_image_size: ``(width, height)`` of the full original photo.
        crop_box: ``(x0, y0, x1, y1)`` the splat's input was cut from, in
            full-image pixels. ``None`` if the splat used the whole photo.
        scale: The depth gauge. ``None`` fits it via :func:`estimate_depth_scale`,
            which requires ``depth_probe``.
        reconcile_intrinsics: See :func:`compute_anchor_transform`.
        depth_probe: Callable ``(camera) -> depth buffer`` used to fit the gauge
            when ``scale`` is None. The pipeline passes a renderer bound to the
            original image size.

    Returns:
        ``(splat_scene_in_world_coords, info)``. ``info`` carries ``scale``,
        ``scale_source`` ("fitted" or "given"), ``M``, ``centroid`` and
        ``source_view_dir`` -- the unit direction from the origin camera to the
        splat, which is the front of the 2.5-D shell.

    Raises:
        ValueError: If ``scale`` is None and no ``depth_probe`` is supplied.
    """
    meta = load_splat_meta(meta_path)
    scene = SplatScene.from_ply(ply_path)

    if scale is None:
        if depth_probe is None:
            raise ValueError(
                "Fitting the depth gauge needs a depth_probe; pass an explicit "
                "scale instead."
            )
        M1, centroid = compute_anchor_transform(
            meta, original_focal_length, original_image_size,
            crop_box=crop_box, scale=1.0,
            reconcile_intrinsics=reconcile_intrinsics,
        )
        w, h = original_image_size
        probe_camera = Camera(
            focal_length=(original_focal_length, original_focal_length),
            image_size=(w, h),
        )
        means_gauge1 = (np.asarray(scene.means, np.float64) + centroid) @ M1.T
        scale = estimate_depth_scale(
            means_gauge1, probe_camera, depth_probe(probe_camera)
        )
        scale_source = "fitted"
    else:
        scale = float(scale)
        scale_source = "given"

    M, centroid = compute_anchor_transform(
        meta, original_focal_length, original_image_size,
        crop_box=crop_box, scale=scale,
        reconcile_intrinsics=reconcile_intrinsics,
    )
    world = transform_splat_scene(scene, M, centroid)

    center = world.get_bbox_center().astype(np.float64)
    norm = float(np.linalg.norm(center))
    if norm < 1e-9:
        raise ValueError(
            "The anchored splat sits on the source camera itself; its view "
            "direction is undefined."
        )

    info = {
        "scale": float(scale),
        "scale_source": scale_source,
        "M": M,
        "centroid": centroid,
        "source_view_dir": center / norm,
        "splat_meta": meta,
        "n_gaussians": len(world),
    }
    return world, info
