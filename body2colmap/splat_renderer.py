"""
Renderer for Gaussian Splats, backed by ``brush-splat-render``.

This module provides :class:`SplatRenderer`, which shells out to the
``brush-splat-render`` binary from the brush repository. It takes Camera
objects (same interface as the mesh :class:`~body2colmap.renderer.Renderer`)
and produces RGBA images.

Why a subprocess rather than a Python rasterizer:
    The previous implementation called ``gsplat.rasterization`` on torch CUDA
    tensors. gsplat publishes no wheel past torch 2.4 / cu124, so on a modern
    stack it JIT-compiles its CUDA kernels on first use and needs ``nvcc`` at
    runtime -- which forces a CUDA *devel* base image on anything packaging
    this. brush is wgpu/Vulkan and is already present wherever this pipeline
    runs, so routing through it drops the CUDA toolchain entirely and leaves
    one graphics API instead of two.

Coordinate System:
    ``brush-splat-render``'s ``cameras.json`` takes an **OpenGL** convention
    camera-to-world pose -- exactly what :class:`~body2colmap.camera.Camera`
    stores -- so ``camera.rotation`` is serialized row-major and untouched.
    The OpenGL -> OpenCV conversion (``R_cv = R_gl @ diag(1, -1, -1)``, i.e.
    negate the Y and Z *columns*) happens inside the binary, in
    ``to_brush_camera()``.

    This maintains the principle: coordinate conversion only at I/O boundaries.

Batching:
    The binary initializes wgpu and loads the ply once per invocation, then
    loops the whole camera list. Rendering a sequence therefore goes through
    :meth:`SplatRenderer.render_many`, not a per-frame loop over
    :meth:`SplatRenderer.render` -- the latter is a one-camera convenience and
    pays a full process startup.
"""

import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from .camera import Camera
from .splat_scene import SplatScene

logger = logging.getLogger(__name__)

BINARY_NAME = "brush-splat-render"
BINARY_ENV_VAR = "BRUSH_SPLAT_RENDER"

__all__ = ["ConfidenceOptions", "SplatRenderer", "resolve_binary"]


def resolve_binary(explicit: Optional[str] = None) -> str:
    """
    Locate the ``brush-splat-render`` binary.

    Resolution order, first hit wins: an explicit path (from
    ``--splat-renderer`` / ``splat.renderer_binary``), ``$BRUSH_SPLAT_RENDER``,
    then ``PATH``.

    Args:
        explicit: Path given by the user, or None.

    Returns:
        Path to an executable.

    Raises:
        RuntimeError: If no binary can be found, or an explicitly given path
            is not executable.
    """
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
        # Also accept a bare command name, so --splat-renderer can name
        # something on PATH rather than only a full path.
        found = shutil.which(explicit)
        if found:
            return found
        raise RuntimeError(
            f"{explicit!r} is not an executable {BINARY_NAME} binary, and is "
            "not on PATH."
        )

    from_env = os.environ.get(BINARY_ENV_VAR)
    if from_env:
        path = Path(from_env).expanduser()
        if not (path.is_file() and os.access(path, os.X_OK)):
            raise RuntimeError(
                f"${BINARY_ENV_VAR} is set to {from_env!r}, which is not an "
                f"executable {BINARY_NAME} binary."
            )
        return str(path)

    found = shutil.which(BINARY_NAME)
    if found:
        return found

    raise RuntimeError(
        f"{BINARY_NAME} not found. Build it with\n"
        f"  cargo build --release -p {BINARY_NAME}\n"
        f"in your brush checkout, then set ${BINARY_ENV_VAR} to the built\n"
        f"binary (target/release/{BINARY_NAME}) or pass --splat-renderer."
    )


def _describe_exit(returncode: int) -> str:
    """
    Describe a subprocess exit status in a sentence fragment.

    POSIX reports death-by-signal as a negative return code, which is worth
    naming: an operator seeing "exit -11" has to look it up, and the fact that
    it was SIGSEGV rather than a non-zero exit is the whole difference between
    a crash and a rejected input.
    """
    if returncode == 0:
        return "exited cleanly"
    if returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except ValueError:
            name = "unknown signal"
        return f"was killed by signal {-returncode} ({name})"
    return f"exited with status {returncode}"


@dataclass
class ConfidenceOptions:
    """
    Options for ``brush-splat-render --confidence``.

    Confidence mode gates every pixel by a per-splat multi-view confidence
    measured against the training views, instead of leaving the decision to a
    downstream threshold on rendered alpha.

    **It changes what the alpha channel means.** Without it, alpha is
    accumulated splat opacity. With it, alpha is the *gate*: how much the
    training views actually constrained the splats covering that pixel. A mask
    derived from a confidence render therefore masks on evidence, not coverage
    -- which is the point, and why it replaces a downstream
    threshold-and-dilate stage.

    It needs evidence: either ``ev_*`` properties baked into the ply by
    ``brush ... --export-evidence``, or ``dataset`` pointing at the training
    set so it can be measured here. Given neither, the binary warns and treats
    every splat as fully trusted, which silently degenerates to plain alpha.

    Attributes:
        cull_color: RGB (0-1) that culled pixels resolve to, or ``None`` to
            follow the render's ``bg_color``.

            The binary uses one colour for both roles -- culled pixels *and*
            the background composited under the splat -- so this is the whole
            background of a confidence render, not just the culled parts.
            Defaulting it to ``bg_color`` is what keeps ``--bg-color`` meaning
            the same thing whether or not gating is on; set it explicitly only
            to make culled regions stand out against the background for
            inspection.
        gate_lo: Confidence at or below which a pixel is fully culled.
        gate_hi: Confidence at or above which a pixel is fully kept. Equal to
            ``gate_lo`` gives a hard cut.
        sidecar: Also capture the raw per-pixel confidence map alongside each
            frame (see :attr:`SplatRenderer.last_confidence_maps`).
        dataset: Training dataset directory (COLMAP / nerfstudio /
            RealityCapture) to measure evidence against when the ply carries
            none. Load options must match the training run -- pass them
            through ``extra_args``.
        extra_args: Verbatim passthrough for the binary's tuning flags
            (``--conf-tau``, ``--conf-min-views``, ``--conf-facing``, the
            dataset option group, ...). These are mirrored rather than
            duplicated so their defaults stay defined in exactly one place.
    """

    cull_color: Optional[Tuple[float, float, float]] = None
    gate_lo: float = 0.45
    gate_hi: float = 0.65
    sidecar: bool = False
    dataset: Optional[str] = None
    extra_args: Sequence[str] = field(default_factory=tuple)

    def to_args(self, bg_color: Tuple[float, float, float]) -> List[str]:
        """
        Build the binary's confidence flags.

        Args:
            bg_color: The render's background, used as the cull colour when
                :attr:`cull_color` is None.
        """
        cull = self.cull_color if self.cull_color is not None else bg_color
        args = [
            "--confidence",
            "--cull-color", ",".join(f"{c:g}" for c in cull),
            "--gate-lo", f"{self.gate_lo:g}",
            "--gate-hi", f"{self.gate_hi:g}",
        ]
        if self.sidecar:
            args.append("--confidence-sidecar")
        if self.dataset:
            args += ["--dataset", str(self.dataset)]
        args += list(self.extra_args)
        return args


class SplatRenderer:
    """
    Render Gaussian splats via the ``brush-splat-render`` binary.

    Unlike the mesh Renderer which has multiple modes (mesh, depth, skeleton),
    SplatRenderer renders the splat directly with view-dependent colors from
    spherical harmonics.
    """

    def __init__(
        self,
        scene: SplatScene,
        render_size: Tuple[int, int],
        binary: Optional[str] = None,
        confidence: Optional[ConfidenceOptions] = None,
        verbose: bool = False,
    ):
        """
        Initialize renderer.

        Args:
            scene: SplatScene to render
            render_size: (width, height) in pixels
            binary: Path to ``brush-splat-render``. None resolves it from
                ``$BRUSH_SPLAT_RENDER`` then ``PATH`` -- see
                :func:`resolve_binary`.
            confidence: Enable confidence gating. See
                :class:`ConfidenceOptions` -- note that it changes the meaning
                of the alpha channel.
            verbose: Pass ``RUST_LOG=info`` so the binary reports per-frame
                progress on stderr.
        """
        self.scene = scene
        self.width, self.height = render_size
        self.binary = resolve_binary(binary)
        self.confidence = confidence
        self.verbose = verbose

        #: Raw per-pixel confidence maps (uint8, HxW) from the most recent
        #: :meth:`render_many`, or None. Only populated with
        #: ``ConfidenceOptions.sidecar``.
        self.last_confidence_maps: Optional[List[NDArray[np.uint8]]] = None

        self._workdir: Optional[tempfile.TemporaryDirectory] = None
        self._ply_path: Optional[Path] = None

    # -- ply staging ------------------------------------------------------

    def _ensure_ply(self) -> Path:
        """
        Serialize the scene to a ply the binary can load, once.

        The scene is often not a file on disk -- the overlay is built in
        memory by :func:`~body2colmap.splat_anchor.anchor_splat_to_world` --
        so it has to be written out. Done lazily and cached, so an 81-frame
        orbit writes it once rather than per invocation.
        """
        if self._ply_path is not None:
            return self._ply_path

        self._workdir = tempfile.TemporaryDirectory(prefix="body2colmap-splat-")
        path = Path(self._workdir.name) / "scene.ply"
        self.scene.to_ply(str(path))
        self._ply_path = path
        return path

    # -- camera serialization ---------------------------------------------

    def _cameras_json(self, cameras: Sequence[Camera]) -> Dict[str, Any]:
        """
        Build the binary's ``cameras.json`` payload.

        ``rotation`` is ``camera.rotation`` -- the OpenGL camera-to-world
        matrix -- serialized row-major and unmodified. Do NOT convert it here:
        the binary's ``to_brush_camera()`` owns that conversion, and doing it
        on both sides would cancel out into a vertically mirrored render.
        """
        entries = []
        for i, cam in enumerate(cameras):
            if (cam.width, cam.height) != (self.width, self.height):
                raise ValueError(
                    f"camera {i} is {cam.width}x{cam.height} but this renderer "
                    f"is {self.width}x{self.height}. All cameras must share the "
                    "renderer's resolution -- cx/cy are pixel coordinates and "
                    "are interpreted against it."
                )
            entries.append({
                "name": f"f{i:05d}.png",
                "fx": float(cam.fx),
                "fy": float(cam.fy),
                "cx": float(cam.cx),
                "cy": float(cam.cy),
                "position": [float(v) for v in cam.position],
                "rotation": [[float(v) for v in row] for row in cam.rotation],
            })
        return {"width": int(self.width), "height": int(self.height), "cameras": entries}

    # -- rendering ---------------------------------------------------------

    def render(
        self,
        camera: Camera,
        bg_color: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0)
    ) -> NDArray[np.uint8]:
        """
        Render the splat from given camera viewpoint.

        Convenience wrapper around :meth:`render_many` for a single camera.
        It costs a full binary invocation, so render sequences with
        :meth:`render_many` instead.

        Args:
            camera: Camera object (same interface as mesh Renderer)
            bg_color: See :meth:`render_many`.

        Returns:
            RGBA image (height, width, 4), dtype uint8
        """
        return self.render_many([camera], bg_color=bg_color)[0]

    def render_many(
        self,
        cameras: Sequence[Camera],
        bg_color: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0)
    ) -> List[NDArray[np.uint8]]:
        """
        Render the splat from many viewpoints in one binary invocation.

        Args:
            cameras: Cameras to render, in order.
            bg_color: Background RGB color (0-1 range), composited under the
                splat using its accumulated alpha. Pass ``None`` to get the
                raw rasterized colour instead, i.e. straight (un-premultiplied)
                RGB alongside alpha.

                Use ``None`` whenever the result is going to be composited over
                something else. Blending an already-background-composited image
                over another layer blends toward that background twice, which
                shows up as a halo around the silhouette.

                Under :class:`ConfidenceOptions` this is applied by the binary
                rather than here, as the colour it composites over and resolves
                culled pixels to -- the binary uses one colour for both. Set
                ``ConfidenceOptions.cull_color`` to separate them. ``None`` is
                rejected there, since a gated render has no accumulated opacity
                to un-premultiply by.

        Returns:
            List of RGBA images (height, width, 4), dtype uint8, one per
            camera. Alpha comes from accumulated opacity during rasterization
            -- or, with confidence gating, from the confidence gate.

        Note:
            The binary always runs with ``--background 0,0,0``, which makes its
            RGB output premultiplied by alpha -- the same intermediate gsplat
            produced. Both ``bg_color`` behaviours are then derived here, so
            there is one Rust path and one Python path. The 8-bit round trip
            costs at most 1/255 in the final composite, because the quantized
            quantity *is* the premultiplied contribution.
        """
        if not cameras:
            return []

        if self.confidence is not None and bg_color is None:
            raise ValueError(
                "bg_color=None (straight alpha) is incompatible with confidence "
                "gating: the binary composites RGB over cull_color and writes "
                "the gate as alpha, so there is no accumulated opacity to "
                "un-premultiply by."
            )

        ply = self._ensure_ply()
        run_dir = Path(tempfile.mkdtemp(dir=self._workdir.name, prefix="run-"))
        try:
            cameras_path = run_dir / "cameras.json"
            cameras_path.write_text(json.dumps(self._cameras_json(cameras)))
            frames_dir = run_dir / "frames"

            cmd = [
                self.binary,
                "--splat", str(ply),
                "--cameras", str(cameras_path),
                "--output-dir", str(frames_dir),
                "--background", "0,0,0",
            ]
            if self.confidence is not None:
                cmd += self.confidence.to_args(bg_color)

            env = dict(os.environ)
            if self.verbose:
                env.setdefault("RUST_LOG", "info")

            # Under `verbose` the binary's stderr is left inherited so its
            # progress log streams live; capturing it would mean the caller
            # sees nothing until the whole sequence is done, which defeats the
            # point on an 81-frame render.
            result = subprocess.run(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=None if self.verbose else subprocess.PIPE,
                text=True,
            )

            # Success is decided by the artifacts, not the exit status.
            # brush-splat-render intermittently dies from a signal (SIGSEGV)
            # *after* writing every frame it was asked for; treating that as a
            # failure would throw away a complete, correct render. So check
            # what it produced, and fall back to the exit status only to
            # explain a genuine shortfall. Truncation is caught downstream:
            # _read_frame() rejects a file OpenCV cannot decode or whose
            # dimensions are wrong.
            expected = [frames_dir / f"f{i:05d}.png" for i in range(len(cameras))]
            if self.confidence is not None and self.confidence.sidecar:
                expected += [
                    frames_dir / f"f{i:05d}.conf.png" for i in range(len(cameras))
                ]
            missing = [
                path.name for path in expected
                if not (path.is_file() and path.stat().st_size > 0)
            ]

            if missing:
                shown = ", ".join(missing[:5])
                if len(missing) > 5:
                    shown += f", ... ({len(missing)} total)"
                detail = (
                    "see its output above"
                    if result.stderr is None
                    else result.stderr.strip()
                )
                raise RuntimeError(
                    f"{BINARY_NAME} did not produce "
                    f"{len(missing)} of {len(expected)} expected output files "
                    f"({shown}). It {_describe_exit(result.returncode)}.\n{detail}"
                )

            if result.returncode != 0:
                logger.warning(
                    "%s %s, but wrote all %d expected output files; using them. "
                    "This is a known intermittent fault in the renderer, not a "
                    "bad render.",
                    BINARY_NAME, _describe_exit(result.returncode), len(expected),
                )

            images = [
                self._read_frame(frames_dir / f"f{i:05d}.png", bg_color)
                for i in range(len(cameras))
            ]
            self.last_confidence_maps = (
                [self._read_gray(frames_dir / f"f{i:05d}.conf.png")
                 for i in range(len(cameras))]
                if self.confidence is not None and self.confidence.sidecar
                else None
            )
            return images
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def _read_frame(
        self,
        path: Path,
        bg_color: Optional[Tuple[float, float, float]]
    ) -> NDArray[np.uint8]:
        """Read one rendered frame and apply the requested alpha convention."""
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            # The file passed the exists-and-non-empty check, so this is a
            # partial write -- the renderer was killed part-way through it.
            raise RuntimeError(
                f"{path.name} could not be decoded. {BINARY_NAME} wrote it "
                "only partially, most likely by dying mid-write."
            )
        if image.shape[:2] != (self.height, self.width):
            raise RuntimeError(
                f"{path.name} is {image.shape[1]}x{image.shape[0]}, expected "
                f"{self.width}x{self.height}"
            )
        if image.shape[2] != 4:
            raise RuntimeError(f"{path.name} has {image.shape[2]} channels, expected RGBA")

        rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

        if self.confidence is not None:
            # Already composited over cull_color, alpha is the gate. Nothing
            # to recover.
            return rgba

        rgb = rgba[:, :, :3].astype(np.float32) / 255.0
        alpha = rgba[:, :, 3].astype(np.float32) / 255.0

        if bg_color is not None:
            # rgb is premultiplied (rendered on black), so `over` is an add.
            rgb = rgb + np.array(bg_color, dtype=np.float32) * (1.0 - alpha[..., None])
        else:
            # Straight alpha: recover un-premultiplied colour so an `over`
            # blend onto another layer is correct.
            safe = np.maximum(alpha, 1e-6)[..., np.newaxis]
            rgb = np.where(alpha[..., np.newaxis] > 1e-6, rgb / safe, 0.0)

        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        return np.dstack([rgb, rgba[:, :, 3]])

    @staticmethod
    def _read_gray(path: Path) -> NDArray[np.uint8]:
        """Read one confidence sidecar."""
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(
                f"{path.name} could not be decoded. {BINARY_NAME} wrote it "
                "only partially, most likely by dying mid-write."
            )
        return image

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Drop the staged ply and its temp directory."""
        if self._workdir is not None:
            self._workdir.cleanup()
            self._workdir = None
        self._ply_path = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
