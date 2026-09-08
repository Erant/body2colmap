"""
Rendering engine using pyrender.

This module provides the Renderer class which handles actual image generation
using pyrender. Supports multiple rendering modes: mesh, depth, outline,
skeleton, and composite modes.

All rendering happens in world coordinates (no coordinate conversion needed
since pyrender uses OpenGL convention matching our world coords).
"""

import numpy as np
from typing import TYPE_CHECKING, Optional, Tuple, Dict, Any, List
from numpy.typing import NDArray

from .scene import Scene
from .camera import Camera

if TYPE_CHECKING:
    # Annotations only. Splat support is optional at runtime; the layer and
    # its mask options both arrive from the pipeline already built.
    from .splat_renderer import InactiveMaskOptions


def _flat_color_rgba8(color: Tuple[float, float, float]) -> NDArray[np.uint8]:
    """
    Convert an RGB float color (0-1) to an opaque uint8 RGBA vertex color.

    The color is gamma pre-compensated so the rendered pixel matches the
    requested value. pyrender's shader gamma-encodes its output
    (``pow(color, 1/2.2)``) without ever decoding vertex colors, so an
    uncompensated mid-tone comes back visibly washed out — e.g. a 0.35
    channel renders as 0.62.

    This only holds for a pass lit by ambient light alone, where the shader
    reduces to ``pow(albedo * ambient, 1/2.2)``. It is not valid for passes
    with directional or point lights.

    Args:
        color: RGB floats in 0-1

    Returns:
        Opaque RGBA uint8 array, shape (4,)
    """
    encoded = np.clip(np.asarray(color, dtype=np.float64), 0.0, 1.0) ** 2.2
    return np.array(
        [int(round(c * 255)) for c in encoded] + [255],
        dtype=np.uint8
    )


#: Layers that can start a composite mode string (the base, drawn first).
BASE_LAYERS = ("mesh", "depth", "outline", "skeleton")

#: Layers that can follow a "+" in a composite mode string, in draw order.
#: "skeleton" appears in both: it is a base on its own ("skeleton+face") and an
#: overlay on top of geometry ("depth+skeleton").
OVERLAY_LAYERS = ("skeleton", "face", "splat")

#: Every recognized layer name.
LAYER_NAMES = tuple(dict.fromkeys(BASE_LAYERS + OVERLAY_LAYERS))


def parse_composite_modes(mode_str: str) -> Tuple[str, List[str]]:
    """
    Split a render mode string into its base layer and overlays.

    ``"depth+skeleton+face"`` -> ``("depth", ["skeleton", "face"])``.
    A plain ``"mesh"`` -> ``("mesh", [])``.

    This is the single place mode strings are interpreted. Both the CLI and
    :meth:`OrbitPipeline.render_original_view` used to split and validate them
    independently, which meant a new layer had to be added to each by hand.

    Note ``"splat"`` is context-dependent: it names a *base* layer when the
    input is a ``.ply`` (the whole scene is a splat, rendered by
    ``SplatRenderer``), and an *overlay* here when the input is an ``.npz``
    with a splat attached alongside via
    :meth:`OrbitPipeline.attach_splat_overlay`. The two cannot co-occur — a
    ``SplatScene`` has no mesh or skeleton to composite against — so the scene
    type disambiguates. A bare ``"splat"`` never reaches this function as a
    composite.

    Args:
        mode_str: A mode string, with or without "+".

    Returns:
        ``(base, overlays)``.

    Raises:
        ValueError: On an empty component, an unknown layer name, a base that
            can only be an overlay, or an overlay that can only be a base.
    """
    parts = [p.strip() for p in mode_str.split("+")]

    if any(not p for p in parts):
        raise ValueError(f"Empty layer in render mode {mode_str!r}")

    unknown = [p for p in parts if p not in LAYER_NAMES]
    if unknown:
        raise ValueError(
            f"Unknown render layer(s) {', '.join(repr(u) for u in unknown)} in "
            f"{mode_str!r}. Known layers: {', '.join(LAYER_NAMES)}"
        )

    base, overlays = parts[0], parts[1:]

    if base not in BASE_LAYERS:
        raise ValueError(
            f"{base!r} cannot be the base layer of {mode_str!r}; it is an "
            f"overlay. Base layers: {', '.join(BASE_LAYERS)}"
        )

    bad = [o for o in overlays if o not in OVERLAY_LAYERS]
    if bad:
        raise ValueError(
            f"{', '.join(repr(b) for b in bad)} cannot be an overlay in "
            f"{mode_str!r}. Overlays: {', '.join(OVERLAY_LAYERS)}"
        )

    dupes = [p for p in set(parts) if parts.count(p) > 1]
    if dupes:
        raise ValueError(
            f"Layer(s) {', '.join(repr(d) for d in dupes)} repeated in {mode_str!r}"
        )

    return base, overlays


# pyrender's fragment shader ends on ``pow(color.xyz, vec3(1.0/2.2))``
# (shaders/mesh.frag), so a vertex colour handed to it comes back lifted: ask
# for 0.6 and the render is 0.79. Raising the request by the same exponent
# first cancels it, which is what a style that has to reproduce another
# renderer's exact bytes needs. The older skeleton style deliberately does NOT
# do this — its renders have always carried the lift, and a colour-managed
# version of them would be a different picture, not a fixed one.
_PYRENDER_OUTPUT_GAMMA = 2.2


def _pyrender_rgba(
    color: Tuple[float, float, float],
    linearize: bool = False,
) -> NDArray[np.uint8]:
    """The opaque vertex colour to hand pyrender for a requested RGB.

    Args:
        color: Requested RGB, 0-1 range.
        linearize: Pre-compensate for the shader's output gamma, so the
            rendered pixel matches ``color`` rather than a lifted version of it.

    Returns:
        RGBA uint8, shape (4,).
    """
    if linearize:
        color = tuple(max(c, 0.0) ** _PYRENDER_OUTPUT_GAMMA for c in color)
    return np.array(
        [int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), 255],
        dtype=np.uint8,
    )


def outline_from_mask(
    mask: NDArray[np.bool_],
    fg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    bg_color: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0),
    style: str = "filled",
    thickness: int = 3,
    blur: int = 4,
) -> NDArray[np.uint8]:
    """
    Draw a flat two-tone outline from a boolean coverage mask.

    This is the whole of what :meth:`Renderer.render_outline` does once it
    has a silhouette: it is a module-level function so the silhouette can
    come from somewhere other than the mesh. `render_outline` hands it the
    depth-buffer mask by default; a caller with a better opinion of where
    the subject is — a matte of a photograph or of a generated frame, say —
    hands its own through `render_outline(mask=...)`.

    No GL, no mesh, no gamma: the colours are written into the buffer as
    plain ``int(c * 255)`` bytes, exactly as they always were, because
    nothing here passes through pyrender's shader.

    Args:
        mask: Boolean array, shape (height, width). True where the subject
            covers the pixel.
        fg_color: RGB color (0-1 range) for the subject
        bg_color: RGB color (0-1 range) for the background. If None, the
            background RGB is left black (alpha is 0 there either way).
        style: "filled" fills the whole silhouette; "stroke" draws only a
            band along its boundary and gives the interior bg_color.
        thickness: Stroke width in pixels. Only used when style="stroke".
        blur: Blur radius in pixels, applied to color and alpha together.
            0 leaves hard two-tone edges.

    Returns:
        RGBA image, shape (height, width, 4), dtype uint8. Alpha is 255
        over the mask (unioned with the outer half of a stroke), so the
        image works as a composite base layer and as a training mask.

    Raises:
        ValueError: If style is not "filled" or "stroke", if blur is
            negative, or if the mask is not a 2-D boolean array.
    """
    if style not in ("filled", "stroke"):
        raise ValueError(
            f"Unknown outline style: {style!r}. Use 'filled' or 'stroke'."
        )
    if blur < 0:
        raise ValueError(f"Outline blur must be >= 0, got {blur}")
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError(
            "Outline mask must be a 2-D boolean array, got "
            f"shape {mask.shape} dtype {mask.dtype}"
        )
    height, width = mask.shape

    if style == "stroke":
        import cv2

        # Kernel radius is half the requested width so the band straddles
        # the silhouette boundary and ends up ~thickness px across.
        radius = max(1, int(round(thickness / 2.0)))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        mask_u8 = mask.astype(np.uint8)
        # Default morphology border handling treats outside-the-image as
        # neutral, so a silhouette running off the edge is not given a
        # spurious stroke along the image border.
        outer = cv2.dilate(mask_u8, kernel)
        inner = cv2.erode(mask_u8, kernel)
        fg_mask = (outer > 0) & (inner == 0)
    else:
        fg_mask = mask

    fg_rgb = np.array([int(c * 255) for c in fg_color], dtype=np.uint8)
    if bg_color is None:
        bg_rgb = np.zeros(3, dtype=np.uint8)
    else:
        bg_rgb = np.array([int(c * 255) for c in bg_color], dtype=np.uint8)

    # Alpha tracks coverage (not the drawn foreground) so "outline" behaves
    # like "mesh"/"depth" as a composite base and as a mask. The union
    # keeps the outward half of a stroke from being clipped.
    alpha_mask = mask | fg_mask

    image = np.empty((height, width, 4), dtype=np.uint8)
    image[:, :, :3] = np.where(fg_mask[:, :, None], fg_rgb, bg_rgb)
    image[:, :, 3] = alpha_mask.astype(np.uint8) * 255

    if blur > 0:
        import cv2

        # Blur color and alpha together so the two edges stay in step.
        # Applied here rather than in render_composite() so that overlays
        # (e.g. the skeleton in "outline+skeleton") stay sharp: they are
        # composited on top of the already-blurred base.
        ksize = 2 * int(blur) + 1
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)

    return image


class Renderer:
    """
    Render images from Scene using pyrender.

    Supports multiple rendering modes:
    - mesh: Colored mesh with lighting
    - depth: Depth buffer
    - outline: Flat two-tone silhouette (no shading)
    - skeleton: 3D skeleton as spheres (joints) and cylinders (bones)
    - composite: Combinations of the above

    All modes support alpha channel output.
    """

    def __init__(
        self,
        scene: Scene,
        render_size: Tuple[int, int] = (512, 512),
        background=None,
    ):
        """
        Initialize Renderer.

        Args:
            scene: Scene to render
            render_size: (width, height) in pixels
            background: Optional
                :class:`~body2colmap.background.Background` drawn behind the
                base layer. Deliberately not part of the pyrender scene: a
                surrounding sphere would cover every pixel of the depth buffer,
                and :meth:`render_mask` — and through it ``outline`` mode and
                the alpha channel — reads mesh coverage from exactly that.
                May be reassigned between renders.
        """
        self.scene = scene
        self.width, self.height = render_size
        self.background = background

        # Will be lazily initialized
        self._pyrender_scene = None
        self._renderer = None

    def _get_pyrender_renderer(self):
        """
        Get or create pyrender OffscreenRenderer.

        Lazily creates the renderer to avoid OpenGL initialization
        until actually needed.

        Returns:
            pyrender.OffscreenRenderer instance
        """
        if self._renderer is None:
            try:
                import pyrender
            except ImportError:
                raise ImportError(
                    "pyrender is required for rendering. "
                    "Install with: pip install pyrender"
                )

            self._renderer = pyrender.OffscreenRenderer(
                viewport_width=self.width,
                viewport_height=self.height
            )

        return self._renderer

    def _create_pyrender_scene(
        self,
        mesh_color: Optional[Tuple[float, float, float]] = None,
        include_skeleton: bool = False
    ):
        """
        Create pyrender.Scene with mesh and optional skeleton.

        Args:
            mesh_color: RGB color (0-1 range) for mesh
                       If None, uses default light gray
            include_skeleton: Whether to add skeleton to scene

        Returns:
            pyrender.Scene instance

        Note:
            The scene geometry is in world coordinates.
            No coordinate conversion needed here.
        """
        try:
            import pyrender
            import trimesh
        except ImportError:
            raise ImportError(
                "pyrender and trimesh are required. "
                "Install with: pip install pyrender trimesh"
            )

        # Create pyrender scene with strong ambient light
        pr_scene = pyrender.Scene(
            bg_color=[0, 0, 0, 0],  # Transparent background
            ambient_light=[0.7, 0.7, 0.7]  # Strong ambient for even lighting
        )

        # Add mesh
        if mesh_color is None:
            mesh_color = (0.65, 0.74, 0.86)  # Light gray-blue

        # Create trimesh with vertex colors
        mesh_tm = trimesh.Trimesh(
            vertices=self.scene.vertices,
            faces=self.scene.faces,
            process=False
        )

        # Set vertex colors (RGBA, 0-255)
        color_rgba = np.array([
            int(mesh_color[0] * 255),
            int(mesh_color[1] * 255),
            int(mesh_color[2] * 255),
            255
        ], dtype=np.uint8)
        mesh_tm.visual.vertex_colors = np.tile(color_rgba, (len(mesh_tm.vertices), 1))

        # Create pyrender mesh
        pr_mesh = pyrender.Mesh.from_trimesh(mesh_tm, smooth=True)
        pr_scene.add(pr_mesh)

        # Add multiple directional lights for even illumination from all sides
        # Front light (positive Z)
        light_front = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
        pr_scene.add(light_front, pose=np.eye(4))

        # Back light (negative Z)
        light_back = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
        back_pose = np.eye(4)
        back_pose[:3, :3] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])  # Rotate 180° around Y
        pr_scene.add(light_back, pose=back_pose)

        # Left light (positive X)
        light_left = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.5)
        left_pose = np.eye(4)
        left_pose[:3, :3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])  # Rotate 90° around Y
        pr_scene.add(light_left, pose=left_pose)

        # Right light (negative X)
        light_right = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.5)
        right_pose = np.eye(4)
        right_pose[:3, :3] = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])  # Rotate -90° around Y
        pr_scene.add(light_right, pose=right_pose)

        return pr_scene

    def render_mesh(
        self,
        camera: Camera,
        mesh_color: Optional[Tuple[float, float, float]] = None,
        bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> NDArray[np.uint8]:
        """
        Render mesh with lighting.

        Args:
            camera: Camera to render from
            mesh_color: RGB color (0-1 range) for mesh
            bg_color: RGB color (0-1 range) for background

        Returns:
            RGBA image, shape (height, width, 4), dtype uint8
            Alpha = 255 where mesh, 0 where background
        """
        try:
            import pyrender
        except ImportError:
            raise ImportError("pyrender is required")

        # Create scene
        pr_scene = self._create_pyrender_scene(mesh_color=mesh_color)

        # Add camera to scene
        pr_camera = pyrender.IntrinsicsCamera(
            fx=camera.fx, fy=camera.fy,
            cx=camera.cx, cy=camera.cy
        )
        camera_pose = camera.get_c2w()
        pr_scene.add(pr_camera, pose=camera_pose)

        # Render
        renderer = self._get_pyrender_renderer()
        color, depth = renderer.render(pr_scene, flags=pyrender.RenderFlags.RGBA)

        # Make color array writable (pyrender may return read-only array on some platforms)
        if not color.flags.writeable:
            color = np.array(color, copy=True)

        # Replace background color (where alpha=0)
        bg_rgba = np.array([
            int(bg_color[0] * 255),
            int(bg_color[1] * 255),
            int(bg_color[2] * 255),
            0
        ], dtype=np.uint8)

        mask = color[:, :, 3] == 0
        color[mask] = bg_rgba

        return color

    def _render_depth_buffer(self, camera: Camera) -> NDArray[np.float32]:
        """
        Render the raw pyrender depth buffer for a camera.

        This is the single source of truth for mesh coverage and mesh depth:
        the depth buffer does not depend on mesh colour, lighting or
        anti-aliasing, so anything derived from it is exact.

        Args:
            camera: Camera to render from

        Returns:
            Float array, shape (height, width). Metric distance along the view
            axis where the mesh covers the pixel, 0.0 where it does not.
        """
        try:
            import pyrender
        except ImportError:
            raise ImportError("pyrender is required")

        pr_scene = self._create_pyrender_scene()

        pr_camera = pyrender.IntrinsicsCamera(
            fx=camera.fx, fy=camera.fy,
            cx=camera.cx, cy=camera.cy
        )
        pr_scene.add(pr_camera, pose=camera.get_c2w())

        renderer = self._get_pyrender_renderer()
        _, depth = renderer.render(pr_scene)

        return depth

    def render_depth(
        self,
        camera: Camera,
        normalize: bool = True,
        colormap: Optional[str] = None
    ) -> NDArray[np.uint8]:
        """
        Render depth buffer.

        Args:
            camera: Camera to render from
            normalize: If True, normalize depth to 0-1 range for visualization
            colormap: Optional colormap name ("viridis", "plasma", etc.)
                     If None, returns grayscale depth

        Returns:
            RGBA image, shape (height, width, 4), dtype uint8
            Alpha = 255 where depth exists, 0 where no geometry
        """
        depth = self._render_depth_buffer(camera)

        # Create alpha mask (1 where depth exists)
        alpha = (depth > 0).astype(np.uint8) * 255

        # Normalize depth if requested
        if normalize:
            valid_depth = depth[depth > 0]
            if len(valid_depth) > 0:
                min_depth = valid_depth.min()
                max_depth = valid_depth.max()
                depth_normalized = np.zeros_like(depth)
                mask = depth > 0
                # Invert so closer = white (1.0), farther = black (0.0)
                depth_normalized[mask] = 1.0 - (depth[mask] - min_depth) / (max_depth - min_depth)
            else:
                depth_normalized = depth
        else:
            depth_normalized = depth

        # Apply colormap if requested
        if colormap is not None:
            try:
                import matplotlib.cm as cm
            except ImportError:
                raise ImportError("matplotlib is required for colormaps")

            cmap = cm.get_cmap(colormap)
            depth_colored = cmap(depth_normalized)[:, :, :3]  # RGB
            depth_colored = (depth_colored * 255).astype(np.uint8)
        else:
            # Grayscale
            depth_gray = (depth_normalized * 255).astype(np.uint8)
            depth_colored = np.stack([depth_gray] * 3, axis=-1)

        # Combine with alpha
        rgba = np.dstack([depth_colored, alpha])

        return rgba

    def render_mask(
        self,
        camera: Camera
    ) -> NDArray[np.bool_]:
        """
        Render the mesh silhouette as a boolean coverage mask.

        Uses the depth buffer rather than the color buffer, so the result is
        independent of lighting, mesh color and anti-aliasing.

        Args:
            camera: Camera to render from

        Returns:
            Boolean array, shape (height, width). True where the mesh covers
            the pixel.
        """
        return self._render_depth_buffer(camera) > 0

    def render_outline(
        self,
        camera: Camera,
        fg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        bg_color: Optional[Tuple[float, float, float]] = (1.0, 1.0, 1.0),
        style: str = "filled",
        thickness: int = 3,
        blur: int = 4,
        mask: Optional[NDArray[np.bool_]] = None,
    ) -> NDArray[np.uint8]:
        """
        Render the mesh as a flat two-tone outline.

        The mesh carries no shading at all: every covered pixel gets the same
        foreground color, so the only information in the image is the shape of
        the silhouette. Useful as a control/conditioning image.

        The silhouette is the mesh's by default (:meth:`render_mask`). Pass
        ``mask`` to draw a silhouette from somewhere else — a matte of a
        photograph or of a generated frame — through the same fill, colours
        and blur; the drawing is then :func:`outline_from_mask`'s, and the
        mesh is not rasterized at all. The mask must already be on this
        renderer's pixel grid: a caller who resampled it knows how, this
        method does not.

        Args:
            camera: Camera to render from (unused when ``mask`` is given)
            fg_color: RGB color (0-1 range) for the mesh
            bg_color: RGB color (0-1 range) for the background. If None, the
                background RGB is left black (alpha is 0 there either way).
            style: Outline style:
                - "filled": the whole silhouette is filled with fg_color
                - "stroke": only a band along the silhouette boundary is drawn
                  in fg_color; the interior gets bg_color
            thickness: Stroke width in pixels. Only used when style="stroke".
            blur: Blur radius in pixels, applied to both color and alpha so
                the edge softens consistently. 0 disables blurring, leaving
                hard two-tone edges.
            mask: Optional boolean array, shape (height, width), True where
                the subject covers the pixel. Replaces the mesh silhouette.

        Returns:
            RGBA image, shape (height, width, 4), dtype uint8.
            Alpha = 255 over the silhouette (the mesh's, or ``mask``),
            matching mesh/depth mode so
            the result works as a composite base layer and as a training mask.
            For style="stroke" the alpha additionally covers the outer half of
            the boundary band (a ~thickness/2 px dilation of the silhouette),
            so no part of the stroke is clipped by the alpha channel.
            With blur > 0 both edges become gradients rather than hard steps.

        Raises:
            ValueError: If style is not "filled" or "stroke", if blur is
                negative, or if ``mask`` is not a boolean array of this
                renderer's (height, width).
        """
        if mask is None:
            mask = self.render_mask(camera)
        else:
            mask = np.asarray(mask)
            if mask.shape != (self.height, self.width):
                raise ValueError(
                    "Outline mask must match the render size: expected "
                    f"({self.height}, {self.width}), got {mask.shape}"
                )
        return outline_from_mask(
            mask,
            fg_color=fg_color,
            bg_color=bg_color,
            style=style,
            thickness=thickness,
            blur=blur,
        )

    def render_skeleton(
        self,
        camera: Camera,
        joint_radius: float = 0.02,
        bone_radius: float = 0.012,
        joint_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        bone_color: Tuple[float, float, float] = None,
        use_openpose_colors: bool = True,
        render_bones: bool = True,
        target_format: str = "openpose_body25_hands",
        style: str = "openpose",
        face_mode: str = None,
        face_landmarks: Optional[NDArray[np.float32]] = None,
        face_max_angle: float = 90.0,
        eye_style: str = None,
        eye_color: Tuple[float, float, float] = None,
        pupil_color: Tuple[float, float, float] = None,
        pupil_scale: float = None,
        bg_color: Optional[Tuple[float, float, float]] = None,
    ) -> NDArray[np.uint8]:
        """
        Render 3D skeleton as spheres (joints) and cylinders (bones).

        Args:
            camera: Camera to render from
            joint_radius: Radius of joint spheres in meters
            bone_radius: Radius of bone cylinders in meters
            joint_color: RGB color (0-1) for joints
            bone_color: RGB color (0-1) for bones (if not using OpenPose colors)
            use_openpose_colors: If True, use OpenPose color scheme for bones
            render_bones: If False, only render joints (no bone cylinders)
            target_format: Skeleton format to render ("openpose_body25_hands", "mhr70", etc.)
            style: Which drawing convention to colour and size the skeleton by:
                - "openpose": this project's own scheme (default)
                - "dwpose": the convention Wan/VACE pose maps are drawn in —
                  dimmed body limbs, undimmed joint dots, hands a quarter as
                  thick under a full hue sweep, blue hand keypoints. See the
                  DWPose section of :mod:`body2colmap.skeleton`. Only defined
                  for target_format="openpose_body25_hands".
                Ignored when use_openpose_colors is False.
            face_mode: Face landmark rendering mode:
                - None: No face landmarks (default)
                - "full": Points + connectivity lines
                - "points": Points only, no connecting lines
            face_landmarks: Optional custom face landmarks in OpenPose Face 70
                format, shape (70, 3). If provided, used instead of the
                canonical face model for Procrustes fitting.
            face_max_angle: Maximum angle (degrees) off the face normal at which
                face landmarks are rendered. 90 = full frontal hemisphere.
            eye_style: How to render the eyes:
                - "shape": filled eye with a pupil disc (default)
                - "dots": the original OpenPose landmark dots and eye outline
            eye_color: RGB color (0-1) for the filled eye shape (sclera)
            pupil_color: RGB color (0-1) for the pupil disc
            pupil_scale: Pupil diameter as a fraction of the eye height,
                in (0, 1]. 1.0 = a disc touching the upper and lower lid.
            bg_color: RGB color (0-1 range) for background. If None,
                background remains transparent (alpha=0).

        Returns:
            RGBA image, shape (height, width, 4), dtype uint8

        Note:
            When bg_color is None, skeleton does NOT contribute to alpha
            for masking purposes. This is intentional - skeleton is
            overlay only. When bg_color is set, background pixels are
            filled with the specified color (alpha remains 0).

        Raises:
            ValueError: If scene has no skeleton data
        """
        if self.scene.skeleton_joints is None:
            raise ValueError("Scene has no skeleton data")

        try:
            import pyrender
            import trimesh
            from . import skeleton as skel_module
        except ImportError as e:
            raise ImportError(f"Required dependencies not available: {e}")

        # Convert skeleton format if needed
        skeleton_joints = self.scene.skeleton_joints
        skeleton_format = target_format

        if self.scene.skeleton_format == "mhr70" and target_format == "openpose_body25_hands":
            # Convert MHR70 → OpenPose Body25+Hands
            skeleton_joints = skel_module.convert_mhr70_to_openpose_body25_hands(self.scene.skeleton_joints)
        elif self.scene.skeleton_format != target_format:
            raise ValueError(
                f"Conversion from {self.scene.skeleton_format} to {target_format} not implemented"
            )

        # Create pyrender scene
        pr_scene = pyrender.Scene(
            bg_color=[0, 0, 0, 0],  # Transparent background
            ambient_light=[1.0, 1.0, 1.0]  # Full bright ambient for clear skeleton
        )

        # Get bone connectivity and colors (needed for both bones and joints)
        bones = skel_module.get_skeleton_bones(skeleton_format)

        # Get bone colors (OpenPose style or single color), and with them any
        # per-bone width the style asks for and any joint colours it does not
        # want derived from the bones.
        bone_radius_scales = {}
        joint_colors_list = None
        linearize_colors = False
        if use_openpose_colors:
            if style == "dwpose":
                if skeleton_format != "openpose_body25_hands":
                    raise ValueError(
                        f"style='dwpose' is only defined for "
                        f"target_format='openpose_body25_hands', got "
                        f"'{skeleton_format}'"
                    )
                # The style picks the connectivity too: DWPose draws no
                # feet, so the toe and heel bones go and their joints with
                # them (get_joint_colors_dwpose returns None for those).
                bones = skel_module.get_skeleton_bones_dwpose()
                bone_colors = skel_module.get_bone_colors_dwpose()
                bone_radius_scales = skel_module.get_bone_radius_scales_dwpose()
                # DWPose's numbers are the bytes in its output PNG, so they
                # have to survive the shader unchanged.
                linearize_colors = True
                joint_colors_list = skel_module.get_joint_colors_dwpose(
                    len(skeleton_joints)
                )
            elif style == "openpose":
                bone_colors = skel_module.get_bone_colors_openpose_style(skeleton_format)
            else:
                raise ValueError(
                    f"Unknown skeleton style: {style!r}. "
                    f"Valid options: 'openpose', 'dwpose'"
                )
        else:
            default_bone_color = bone_color if bone_color is not None else (0.0, 1.0, 0.0)
            bone_colors = {bone: default_bone_color for bone in bones}

        # Add bones as cylinders FIRST (so joints render on top)
        if render_bones:

            for start_idx, end_idx in bones:
                if start_idx >= len(skeleton_joints) or end_idx >= len(skeleton_joints):
                    continue  # Skip invalid bone indices

                start_pos = skeleton_joints[start_idx]
                end_pos = skeleton_joints[end_idx]

                # Create cylinder connecting start to end
                direction = end_pos - start_pos
                length = np.linalg.norm(direction)

                if length < 1e-6:
                    continue  # Skip zero-length bones

                # Create cylinder along Z axis
                cylinder = trimesh.creation.cylinder(
                    radius=bone_radius * bone_radius_scales.get((start_idx, end_idx), 1.0),
                    height=length,
                    sections=8
                )

                # Rotate and translate to connect joints
                # Cylinder default: along Z axis, centered at origin
                # We need: from start_pos to end_pos

                # Compute rotation to align Z axis with bone direction
                z_axis = np.array([0, 0, 1], dtype=np.float32)
                bone_dir = direction / length

                # Rotation axis: cross product
                rot_axis = np.cross(z_axis, bone_dir)
                rot_axis_len = np.linalg.norm(rot_axis)

                if rot_axis_len > 1e-6:
                    rot_axis = rot_axis / rot_axis_len
                    # Rotation angle
                    cos_angle = np.dot(z_axis, bone_dir)
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

                    # Build rotation matrix
                    rot_matrix = trimesh.transformations.rotation_matrix(angle, rot_axis)
                else:
                    # Parallel or anti-parallel
                    if np.dot(z_axis, bone_dir) > 0:
                        rot_matrix = np.eye(4)
                    else:
                        # 180 degree rotation
                        rot_matrix = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])

                # Translate to center between joints
                center = (start_pos + end_pos) / 2
                rot_matrix[:3, 3] = center

                cylinder.apply_transform(rot_matrix)

                # Get color for this bone
                this_bone_color = bone_colors.get((start_idx, end_idx), (0.0, 1.0, 0.0))
                cylinder.visual.vertex_colors = _pyrender_rgba(
                    this_bone_color, linearize=linearize_colors
                )

                mesh = pyrender.Mesh.from_trimesh(cylinder, smooth=False)
                pr_scene.add(mesh)

        # Compute per-joint colors from bone colors, unless the style above
        # already supplied its own table (DWPose colours a dot by its own joint
        # index, and undimmed, so it cannot be read off the bones).
        if joint_colors_list is None:
            if use_openpose_colors:
                joint_colors_list = skel_module.get_joint_colors_from_bones(bone_colors, len(skeleton_joints))
            else:
                # Use single color for all joints
                joint_colors_list = [joint_color] * len(skeleton_joints)

        # Add joints as spheres LAST (render on top of bones)
        for joint_idx, joint_pos in enumerate(skeleton_joints):
            # Use per-joint color. None is a style saying this joint is not
            # drawn at all — a dot left behind by a bone the style dropped
            # would read as a speck of noise, not as a keypoint.
            this_joint_color = joint_colors_list[joint_idx]
            if this_joint_color is None:
                continue

            sphere = trimesh.creation.icosphere(subdivisions=2, radius=joint_radius)
            sphere.vertices += joint_pos
            sphere.visual.vertex_colors = _pyrender_rgba(
                this_joint_color, linearize=linearize_colors
            )

            mesh = pyrender.Mesh.from_trimesh(sphere, smooth=False)
            pr_scene.add(mesh)

        # Add camera
        pr_camera = pyrender.IntrinsicsCamera(
            fx=camera.fx, fy=camera.fy,
            cx=camera.cx, cy=camera.cy
        )
        pr_scene.add(pr_camera, pose=camera.get_c2w())

        # Render skeleton
        renderer = self._get_pyrender_renderer()
        skel_color, _ = renderer.render(pr_scene, flags=pyrender.RenderFlags.RGBA)

        # Render face landmarks as separate pass and composite on top
        if face_mode is not None:
            face_image = self._render_face(
                skeleton_joints, camera, face_mode,
                face_joint_radius=joint_radius * 0.35,
                face_bone_radius=bone_radius * 0.35,
                face_landmarks=face_landmarks,
                face_max_angle=face_max_angle,
                eye_style=eye_style,
                eye_color=eye_color,
                pupil_color=pupil_color,
                pupil_scale=pupil_scale,
            )
            if face_image is not None:
                skel_color = np.array(skel_color, copy=True)
                face_alpha = face_image[:, :, 3:4] / 255.0
                skel_color[:, :, :3] = (
                    face_image[:, :, :3] * face_alpha +
                    skel_color[:, :, :3] * (1 - face_alpha)
                ).astype(np.uint8)
                skel_color[:, :, 3] = np.maximum(
                    skel_color[:, :, 3], face_image[:, :, 3]
                )

        # Fill background with specified color (where alpha=0)
        if bg_color is not None:
            if not skel_color.flags.writeable:
                skel_color = np.array(skel_color, copy=True)
            bg_rgba = np.array([
                int(bg_color[0] * 255),
                int(bg_color[1] * 255),
                int(bg_color[2] * 255),
                0
            ], dtype=np.uint8)
            mask = skel_color[:, :, 3] == 0
            skel_color[mask] = bg_rgba

        return skel_color

    def _render_face(
        self,
        skeleton_joints: NDArray[np.float32],
        camera: Camera,
        face_mode: str,
        face_joint_radius: float = 0.005,
        face_bone_radius: float = 0.003,
        face_landmarks: Optional[NDArray[np.float32]] = None,
        face_max_angle: float = 90.0,
        eye_style: str = None,
        eye_color: Tuple[float, float, float] = None,
        pupil_color: Tuple[float, float, float] = None,
        pupil_scale: float = None,
    ) -> Optional[NDArray[np.uint8]]:
        """
        Render face landmarks as a separate RGBA image.

        Rendered in its own pyrender scene so it can be composited on top
        of the body skeleton without depth-test occlusion from the larger
        skeleton head joints.

        Args:
            skeleton_joints: Skeleton joints in world coords (OpenPose Body25+)
            camera: Camera for visibility test and rendering
            face_mode: "full" (points + connectivity) or "points" (points only)
            face_joint_radius: Radius for face keypoint spheres
            face_bone_radius: Radius for face connection cylinders
            face_landmarks: Optional custom face landmarks in OpenPose Face 70
                format, shape (70, 3). Passed to fit_face_to_skeleton() to use
                instead of the canonical face model.
            face_max_angle: Maximum angle (degrees) off the face normal at which
                face landmarks are rendered. 90 = full frontal hemisphere.
            eye_style: "shape" for a filled eye with a pupil disc, "dots"
                for the original landmark dots. Defaults to
                face.DEFAULT_EYE_STYLE. The eye color and scale options below
                apply to "shape" only.
            eye_color: RGB color (0-1) for the filled eye shape (sclera).
                Defaults to face.DEFAULT_EYE_COLOR.
            pupil_color: RGB color (0-1) for the pupil disc.
                Defaults to face.DEFAULT_PUPIL_COLOR.
            pupil_scale: Pupil diameter as a fraction of the eye height,
                in (0, 1]. Defaults to face.DEFAULT_PUPIL_SCALE.

        Returns:
            RGBA image with face landmarks, or None if face is not visible
        """
        import trimesh
        import pyrender
        from . import face as face_module

        # Check we have enough joints for the 5 anchor points
        max_anchor = max(face_module.SKELETON_ANCHOR_JOINT_INDICES)
        if len(skeleton_joints) <= max_anchor:
            return None

        # Fit face to skeleton (custom landmarks override canonical model)
        fitted_landmarks, residual = face_module.fit_face_to_skeleton(
            skeleton_joints, face_landmarks=face_landmarks
        )

        # Check visibility (angle threshold from face normal)
        if not face_module.is_face_visible(
            fitted_landmarks, camera.position, max_angle_deg=face_max_angle
        ):
            return None

        # Create separate scene for face
        pr_scene = pyrender.Scene(
            bg_color=[0, 0, 0, 0],
            ambient_light=[1.0, 1.0, 1.0]
        )

        face_color_uint8 = _flat_color_rgba8(face_module.FACE_COLOR)

        if eye_style is None:
            eye_style = face_module.DEFAULT_EYE_STYLE
        # Under eye_style="shape" the eye landmarks and eye contour bones are
        # withheld here and drawn as filled eye shapes below instead.
        point_indices, bones = face_module.get_face_draw_lists(eye_style)

        # Add face bones as cylinders (if full mode)
        if face_mode == "full":
            for start_idx, end_idx in bones:
                start_pos = fitted_landmarks[start_idx]
                end_pos = fitted_landmarks[end_idx]

                direction = end_pos - start_pos
                length = np.linalg.norm(direction)

                if length < 1e-6:
                    continue

                cylinder = trimesh.creation.cylinder(
                    radius=face_bone_radius,
                    height=length,
                    sections=6
                )

                z_axis = np.array([0, 0, 1], dtype=np.float32)
                bone_dir = direction / length

                rot_axis = np.cross(z_axis, bone_dir)
                rot_axis_len = np.linalg.norm(rot_axis)

                if rot_axis_len > 1e-6:
                    rot_axis = rot_axis / rot_axis_len
                    cos_angle = np.dot(z_axis, bone_dir)
                    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                    rot_matrix = trimesh.transformations.rotation_matrix(angle, rot_axis)
                else:
                    if np.dot(z_axis, bone_dir) > 0:
                        rot_matrix = np.eye(4)
                    else:
                        rot_matrix = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])

                center = (start_pos + end_pos) / 2
                rot_matrix[:3, 3] = center
                cylinder.apply_transform(rot_matrix)

                cylinder.visual.vertex_colors = face_color_uint8
                mesh = pyrender.Mesh.from_trimesh(cylinder, smooth=False)
                pr_scene.add(mesh)

        # Add face joints as spheres
        for idx in point_indices:
            sphere = trimesh.creation.icosphere(subdivisions=1, radius=face_joint_radius)
            sphere.vertices += fitted_landmarks[idx]
            sphere.visual.vertex_colors = face_color_uint8
            mesh = pyrender.Mesh.from_trimesh(sphere, smooth=False)
            pr_scene.add(mesh)

        # Add the eyes as flat filled shapes with a pupil disc on top
        if eye_style != "shape":
            return self._render_face_scene(pr_scene, camera)

        eye_color_uint8 = _flat_color_rgba8(
            eye_color if eye_color is not None else face_module.DEFAULT_EYE_COLOR
        )
        pupil_color_uint8 = _flat_color_rgba8(
            pupil_color if pupil_color is not None else face_module.DEFAULT_PUPIL_COLOR
        )
        if pupil_scale is None:
            pupil_scale = face_module.DEFAULT_PUPIL_SCALE

        for eye in face_module.build_eye_geometry(
            fitted_landmarks, pupil_scale=pupil_scale
        ):
            for vertices, faces, color in (
                (eye.eye_vertices, eye.eye_faces, eye_color_uint8),
                (eye.pupil_vertices, eye.pupil_faces, pupil_color_uint8),
            ):
                tri = trimesh.Trimesh(
                    vertices=vertices, faces=faces, process=False
                )
                tri.visual.vertex_colors = color
                pr_scene.add(pyrender.Mesh.from_trimesh(tri, smooth=False))

        return self._render_face_scene(pr_scene, camera)

    def _render_face_scene(self, pr_scene, camera: Camera) -> NDArray[np.uint8]:
        """
        Add a camera to a prepared face scene and render it.

        Args:
            pr_scene: pyrender Scene holding the face geometry
            camera: Camera to render from

        Returns:
            RGBA image, shape (height, width, 4), dtype uint8
        """
        import pyrender

        pr_camera = pyrender.IntrinsicsCamera(
            fx=camera.fx, fy=camera.fy,
            cx=camera.cx, cy=camera.cy
        )
        pr_scene.add(pr_camera, pose=camera.get_c2w())

        renderer = self._get_pyrender_renderer()
        color, _ = renderer.render(pr_scene, flags=pyrender.RenderFlags.RGBA)
        return color

    def composite_over_background(
        self,
        image: NDArray[np.uint8],
        camera: Camera,
    ) -> NDArray[np.uint8]:
        """
        Draw the environment backdrop behind a rendered base layer.

        A no-op when no background is configured, so callers need no branch.

        Apply this to the *base* layer only, before any overlay. The skeleton
        overlay writes RGB without touching alpha, so compositing the backdrop
        after it would blend the skeleton away everywhere outside the mesh
        silhouette.

        Args:
            image: RGBA base layer. Modified in place when a background is set.
            camera: Camera the layer was rendered from.

        Returns:
            ``image``.
        """
        if self.background is None:
            return image
        return self.background.composite(image, camera)

    def render_composite(
        self,
        camera: Camera,
        modes: Dict[str, Any],
        splat_layer: Optional[NDArray[np.uint8]] = None,
        inactive_mask: Optional["InactiveMaskOptions"] = None,
    ) -> NDArray[np.uint8]:
        """
        Render composite of multiple modes (e.g., mesh + skeleton overlay).

        Args:
            camera: Camera to render from
            modes: Dictionary specifying which modes to render and their options
                  Example: {
                      "mesh": {"color": (0.65, 0.74, 0.86)},
                      "skeleton": {"joint_radius": 0.015}
                  }

                  Recognized base layers: "mesh", "depth", "outline"
                  (checked in that order). Recognized overlays: "skeleton",
                  "face". An "outline" entry may carry ``"mask"``, a boolean
                  (height, width) array that replaces the mesh silhouette —
                  see :meth:`render_outline`.
            splat_layer: Optional pre-rendered RGBA Gaussian-splat layer with
                **straight** alpha, composited last (on top of everything).
                It is passed in already rendered rather than named in ``modes``
                for two reasons. Splats are rasterized by an external binary
                via :class:`~body2colmap.splat_renderer.SplatRenderer`, which
                this pyrender-backed class knows nothing about; and that binary
                renders a whole camera list per invocation, so the pipeline
                batches every frame's layer up front and hands them down one at
                a time. ``None`` for a frame where the splat is culled.
            inactive_mask: Replace the finished frame's alpha with a
                conditioning mask marking the splat as the region a video model
                should preserve — see
                :class:`~body2colmap.splat_renderer.InactiveMaskOptions`. It is
                applied last, so it overrides every other rule about what alpha
                means here.

        Returns:
            RGBA image with composited modes

        Note:
            Modes are composited in order:
            0. background, if the Renderer has one (behind everything)
            1. mesh, depth or outline (base layer)
            2. skeleton (overlay)
            3. splat (overlay)

            Alpha is the base layer's, unioned with the splat's where present —
            unless an opaque background has already forced it to 255, or
            ``inactive_mask`` has replaced it outright.
        """
        # Render base layer
        base_image = None

        if "mesh" in modes:
            mesh_opts = modes["mesh"] if isinstance(modes["mesh"], dict) else {}
            base_image = self.render_mesh(
                camera,
                mesh_color=mesh_opts.get("color"),
                bg_color=mesh_opts.get("bg_color", (1.0, 1.0, 1.0))
            )
        elif "depth" in modes:
            depth_opts = modes["depth"] if isinstance(modes["depth"], dict) else {}
            base_image = self.render_depth(
                camera,
                normalize=depth_opts.get("normalize", True),
                colormap=depth_opts.get("colormap")
            )
        elif "outline" in modes:
            outline_opts = modes["outline"] if isinstance(modes["outline"], dict) else {}
            base_image = self.render_outline(
                camera,
                fg_color=outline_opts.get("fg_color", (0.0, 0.0, 0.0)),
                bg_color=outline_opts.get("bg_color", (1.0, 1.0, 1.0)),
                style=outline_opts.get("style", "filled"),
                thickness=outline_opts.get("thickness", 3),
                blur=outline_opts.get("blur", 4),
                mask=outline_opts.get("mask"),
            )

        if base_image is not None:
            base_image = self.composite_over_background(base_image, camera)

        # Determine face mode and custom landmarks from composite modes
        face_mode = None
        custom_face_landmarks = None
        face_max_angle = 90.0
        eye_style = None
        eye_color = None
        pupil_color = None
        pupil_scale = None
        if "face" in modes and self.scene.skeleton_joints is not None:
            face_opts = modes["face"] if isinstance(modes["face"], dict) else {}
            face_mode = face_opts.get("face_mode", "full")
            custom_face_landmarks = face_opts.get("face_landmarks")
            face_max_angle = face_opts.get("face_max_angle", 90.0)
            eye_style = face_opts.get("eye_style")
            eye_color = face_opts.get("eye_color")
            pupil_color = face_opts.get("pupil_color")
            pupil_scale = face_opts.get("pupil_scale")

        # If skeleton is present but no mesh/depth base, render skeleton directly
        if base_image is None:
            if "skeleton" not in modes:
                raise ValueError(
                    "Must specify 'mesh', 'depth', 'outline', or 'skeleton' "
                    "as base layer"
                )

            skel_opts = modes["skeleton"] if isinstance(modes["skeleton"], dict) else {}
            base_image = self.render_skeleton(
                camera,
                joint_radius=skel_opts.get("joint_radius", 0.015),
                bone_radius=skel_opts.get("bone_radius", 0.008),
                joint_color=skel_opts.get("joint_color", (1.0, 0.0, 0.0)),
                bone_color=skel_opts.get("bone_color", (0.0, 1.0, 0.0)),
                target_format=skel_opts.get("target_format", "openpose_body25_hands"),
                style=skel_opts.get("style", "openpose"),
                face_mode=face_mode,
                face_landmarks=custom_face_landmarks,
                face_max_angle=face_max_angle,
                eye_style=eye_style,
                eye_color=eye_color,
                pupil_color=pupil_color,
                pupil_scale=pupil_scale,
                bg_color=skel_opts.get("bg_color"),
            )
            base_image = self.composite_over_background(base_image, camera)
            return self._composite_splat(base_image, splat_layer, inactive_mask)

        # Overlay skeleton if requested
        if "skeleton" in modes and self.scene.skeleton_joints is not None:
            skel_opts = modes["skeleton"] if isinstance(modes["skeleton"], dict) else {}

            # Render skeleton (with optional face landmarks)
            skel_image = self.render_skeleton(
                camera,
                joint_radius=skel_opts.get("joint_radius", 0.015),
                bone_radius=skel_opts.get("bone_radius", 0.008),
                joint_color=skel_opts.get("joint_color", (1.0, 0.0, 0.0)),
                bone_color=skel_opts.get("bone_color", (0.0, 1.0, 0.0)),
                target_format=skel_opts.get("target_format", "openpose_body25_hands"),
                style=skel_opts.get("style", "openpose"),
                face_mode=face_mode,
                face_landmarks=custom_face_landmarks,
                face_max_angle=face_max_angle,
                eye_style=eye_style,
                eye_color=eye_color,
                pupil_color=pupil_color,
                pupil_scale=pupil_scale,
            )

            # Composite skeleton over base using alpha blending
            # Skeleton alpha determines blending
            skel_alpha = skel_image[:, :, 3:4] / 255.0
            base_image[:, :, :3] = (
                skel_image[:, :, :3] * skel_alpha +
                base_image[:, :, :3] * (1 - skel_alpha)
            ).astype(np.uint8)

            # Keep base layer's alpha (skeleton doesn't affect masking)

        return self._composite_splat(base_image, splat_layer, inactive_mask)

    @staticmethod
    def _composite_splat(
        base_image: NDArray[np.uint8],
        splat_layer: Optional[NDArray[np.uint8]],
        inactive_mask: Optional["InactiveMaskOptions"] = None,
    ) -> NDArray[np.uint8]:
        """
        Alpha-blend a straight-alpha splat layer on top of a finished composite.

        Unlike the skeleton overlay, the splat *does* contribute to alpha. It is
        real subject coverage, exactly as the mesh silhouette is, so a mask
        derived from the result has to include it — the same reason the face
        overlay unions alpha into the skeleton render. The skeleton stays out of
        alpha because it is an annotation, not geometry.

        ``inactive_mask`` opts out of all of that: alpha stops describing
        coverage and becomes the conditioning mask instead. It is applied here,
        after the blend and on every path through ``render_composite()``,
        including the culled frames where there is no layer to blend — those
        still need a mask, saying the whole frame is reactive.

        Args:
            base_image: RGBA composite to draw onto. Modified in place.
            splat_layer: RGBA with straight alpha, or None to do nothing.
            inactive_mask: Replace alpha with a conditioning mask over the
                splat. None leaves alpha meaning coverage.

        Returns:
            ``base_image``.

        Raises:
            ValueError: If the two layers disagree on size.
        """
        if splat_layer is None:
            if inactive_mask is not None:
                inactive_mask.apply(base_image, None)
            return base_image

        if splat_layer.shape[:2] != base_image.shape[:2]:
            raise ValueError(
                f"splat_layer is {splat_layer.shape[1]}x{splat_layer.shape[0]} "
                f"but the composite is {base_image.shape[1]}x{base_image.shape[0]}"
            )

        alpha = splat_layer[:, :, 3:4].astype(np.float32) / 255.0
        base_image[:, :, :3] = (
            splat_layer[:, :, :3] * alpha +
            base_image[:, :, :3] * (1.0 - alpha)
        ).astype(np.uint8)
        base_image[:, :, 3] = np.maximum(base_image[:, :, 3], splat_layer[:, :, 3])

        if inactive_mask is not None:
            inactive_mask.apply(base_image, splat_layer)

        return base_image

    def warp_original_image(
        self,
        image: NDArray[np.uint8],
        camera: Camera,
        original_focal_length: float,
        border_color: Tuple[int, ...] = (255, 255, 255),
    ) -> NDArray[np.uint8]:
        """
        Warp an original image to align with renders from the given camera.

        Computes and applies a transform that maps pixels from the original
        image (taken at the SAM-3D-Body viewpoint) into the render frame
        defined by ``camera``.  The camera must be at the origin but may
        have a non-identity rotation (e.g. from look_at) as well as a
        scaled focal length and/or shifted principal point.

        When the camera has identity rotation, the transform is a pure
        affine (scale + translate).  When the camera has a non-identity
        rotation (e.g. orbit frame 0 using look_at), a 3x3 homography is
        used instead to account for the perspective change.

        Args:
            image: Original image, shape (H_img, W_img, C) where C is 3 or 4.
            camera: Camera used for rendering.  Must be at the origin.
                May have identity or non-identity rotation.
            original_focal_length: Focal length stored in the SAM-3D-Body
                .npz, corresponding to the original image resolution.
            border_color: Fill color for pixels outside the original image.
                Length must match the channel count of ``image``.

        Returns:
            Warped image at ``(self.width, self.height)`` with the same number
            of channels as the input.

        Raises:
            ValueError: If the camera is not at the origin.
        """
        import cv2
        from .utils import compute_warp_to_camera

        # --- Assert origin position ---
        # Tolerance is generous (1e-3) because the orbit's spherical-coordinate
        # roundtrip can introduce small floating-point drift.
        if not np.allclose(camera.position, 0.0, atol=1e-3):
            raise ValueError(
                "warp_original_image requires a camera at the origin. "
                f"Got position={camera.position.tolist()}"
            )

        h_img, w_img = image.shape[:2]

        is_identity_rotation = np.allclose(
            camera.rotation, np.eye(3), atol=1e-5
        )

        if is_identity_rotation:
            # Pure affine: scale + translate (fast path, exact)
            s = float(camera.fx / original_focal_length)
            tx = camera.cx - s * w_img / 2.0
            ty = camera.cy - s * h_img / 2.0
            M = np.array([[s, 0.0, tx],
                           [0.0, s, ty]], dtype=np.float64)
            warped = cv2.warpAffine(
                image, M, (self.width, self.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_color,
            )
        else:
            # Homography: accounts for rotation + intrinsic change
            H = compute_warp_to_camera(
                original_focal_length=original_focal_length,
                original_image_size=(w_img, h_img),
                target_camera=camera,
            )
            warped = cv2.warpPerspective(
                image, H, (self.width, self.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_color,
            )

        return warped

    def delete(self) -> None:
        """
        Release the OpenGL context held by the pyrender renderer.

        Safe to call more than once. Prefer this over waiting for garbage
        collection whenever a Renderer is short-lived -- notably the throwaway
        one ``OrbitPipeline.attach_splat_overlay()`` builds at the original
        photo's resolution to read a single depth buffer.
        """
        if self._renderer is not None:
            self._renderer.delete()
            self._renderer = None

    def __del__(self):
        """
        Clean up renderer resources.

        Errors are swallowed: a finalizer must not raise, and this one can run
        during interpreter shutdown, where pyrender's EGL teardown re-imports
        and fails with "sys.meta_path is None". That ordering is not
        hypothetical -- importing a large module late (torch, back when splat
        rendering went through gsplat) is enough to delay this past the import
        system's teardown.
        """
        try:
            self.delete()
        except Exception:
            pass
