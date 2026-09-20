"""
Tests for the two-tone outline, drawn from a coverage mask.

`outline_from_mask` is the whole of what `Renderer.render_outline` does once
it has a silhouette, and since 2026-09-08 that silhouette can be handed in
(`render_outline(mask=...)`) instead of rasterized from the mesh. None of it
needs GL, so the things a wrong answer would still look plausible for are
pinned here on hand-built masks: the fill and stroke geometry, the **exact
bytes** the colours land as (b2crunner's `_outline_grey` nudges by half a
level to survive `int(c * 255)` — round or gamma-correct here and every fill
it draws shifts), alpha as coverage rather than as ink, and the seam itself:
a supplied mask never touches the mesh, and a wrong one is refused rather than
resampled.
"""

import numpy as np
import pytest

from body2colmap.renderer import Renderer, outline_from_mask, relief_fill


SIZE = (16, 12)   # (width, height) -- deliberately not square


def _block(size=SIZE):
    """A boolean mask covering rows 3:9, columns 4:12."""
    width, height = size
    mask = np.zeros((height, width), dtype=bool)
    mask[3:9, 4:12] = True
    return mask


class TestFill:
    def test_the_silhouette_is_fg_and_the_rest_bg(self):
        image = outline_from_mask(_block(), fg_color=(0.2, 0.4, 0.6),
                                  bg_color=(1.0, 1.0, 1.0), blur=0)
        mask = _block()
        assert image.shape == (SIZE[1], SIZE[0], 4)
        assert image.dtype == np.uint8
        assert np.all(image[mask][:, :3] == (51, 102, 153))
        assert np.all(image[~mask][:, :3] == 255)

    def test_colours_are_truncated_bytes_not_rounded(self):
        """`int(c * 255)`, exactly. b2crunner's `_outline_grey` returns
        (value + 0.5) / 255 so that truncation recovers `value`; rounding
        here would land one level high on every fill it draws."""
        grey = (0x6F + 0.5) / 255.0
        image = outline_from_mask(_block(), fg_color=(grey,) * 3,
                                  bg_color=((0x7F + 0.5) / 255.0,) * 3, blur=0)
        assert image[5, 6, :3].tolist() == [0x6F] * 3
        assert image[0, 0, :3].tolist() == [0x7F] * 3

    def test_no_bg_color_leaves_black_rgb_and_zero_alpha(self):
        image = outline_from_mask(_block(), bg_color=None, blur=0)
        assert np.all(image[~_block()] == 0)

    def test_alpha_is_coverage(self):
        image = outline_from_mask(_block(), blur=0)
        assert np.array_equal(image[:, :, 3] == 255, _block())
        assert np.all(image[:, :, 3][~_block()] == 0)


class TestStroke:
    def test_the_band_straddles_the_boundary(self):
        """Interior pixels get bg, a band around the edge gets fg, and the
        band has pixels on both sides of the boundary."""
        mask = _block()
        image = outline_from_mask(mask, fg_color=(0, 0, 0), bg_color=(1, 1, 1),
                                  style="stroke", thickness=2, blur=0)
        ink = image[:, :, 0] == 0
        assert not ink[6, 8]            # deep interior: bg
        assert ink[3, 8] and ink[8, 8]  # the boundary rows
        assert ink[2, 8] and ink[9, 8]  # one outside each
        assert not ink[0, 0]

    def test_alpha_covers_the_outer_half_of_the_band(self):
        """`alpha = mask | band`: the stroke's outward half is not clipped."""
        mask = _block()
        image = outline_from_mask(mask, style="stroke", thickness=2, blur=0)
        alpha = image[:, :, 3] == 255
        assert np.all(alpha[mask])
        assert alpha[2, 8] and alpha[9, 8]
        assert not alpha[0, 0]


class TestBlur:
    def test_blur_softens_colour_and_alpha_together(self):
        hard = outline_from_mask(_block(), fg_color=(0, 0, 0), blur=0)
        soft = outline_from_mask(_block(), fg_color=(0, 0, 0), blur=2)
        # Intermediate levels appear in both channels, and the deep interior
        # and far exterior are untouched.
        for channel in (0, 3):
            values = np.unique(soft[:, :, channel])
            assert len(values) > 2, channel
            assert len(np.unique(hard[:, :, channel])) == 2
        assert soft[6, 8].tolist() == hard[6, 8].tolist()
        assert soft[0, 0].tolist() == hard[0, 0].tolist()


class TestRefusals:
    def test_bad_style_and_negative_blur(self):
        with pytest.raises(ValueError, match="style"):
            outline_from_mask(_block(), style="dashed")
        with pytest.raises(ValueError, match="blur"):
            outline_from_mask(_block(), blur=-1)

    def test_a_non_boolean_or_non_2d_mask_is_refused(self):
        with pytest.raises(ValueError, match="boolean"):
            outline_from_mask(_block().astype(np.uint8))
        with pytest.raises(ValueError, match="boolean"):
            outline_from_mask(_block()[:, :, None])


class _MeshlessRenderer(Renderer):
    """A Renderer that was never given a scene and refuses to rasterize."""

    def __init__(self, size):
        # Skip Renderer.__init__ (pyrender, EGL) entirely.
        self.width, self.height = size
        self.background = None

    def render_mask(self, camera):
        raise AssertionError("render_mask was called: the mesh was rasterized")


class TestSuppliedMask:
    def test_a_supplied_mask_never_touches_the_mesh(self):
        renderer = _MeshlessRenderer(SIZE)
        image = renderer.render_outline(camera=None, mask=_block(), blur=0)
        assert np.array_equal(image, outline_from_mask(_block(), blur=0))

    def test_it_goes_through_the_same_options(self):
        renderer = _MeshlessRenderer(SIZE)
        image = renderer.render_outline(
            camera=None, mask=_block(), fg_color=(0, 0, 0), bg_color=(1, 1, 1),
            style="stroke", thickness=2, blur=1,
        )
        expected = outline_from_mask(_block(), fg_color=(0, 0, 0),
                                     bg_color=(1, 1, 1), style="stroke",
                                     thickness=2, blur=1)
        assert np.array_equal(image, expected)

    def test_the_wrong_size_is_refused_not_resampled(self):
        renderer = _MeshlessRenderer(SIZE)
        with pytest.raises(ValueError, match="render size"):
            renderer.render_outline(camera=None, mask=_block((8, 6)))

    def test_render_composite_forwards_the_mask(self):
        """`modes["outline"]["mask"]` reaches render_outline, so a pipeline
        that composites through `render_composite` needs no other seam."""
        renderer = _MeshlessRenderer(SIZE)
        seen = {}

        def fake_outline(camera, **kwargs):
            seen.update(kwargs)
            return outline_from_mask(kwargs["mask"], blur=0)

        renderer.render_outline = fake_outline
        image = renderer.render_composite(
            camera=None, modes={"outline": {"mask": _block(), "blur": 0}}
        )
        assert seen["mask"] is not None and np.array_equal(seen["mask"], _block())
        assert np.array_equal(image[:, :, 3] == 255, _block())


def _grey(byte):
    return ((byte + 0.5) / 255.0,) * 3


class TestReliefFill:
    """`relief_fill`: the body model's depth as a few grey levels under the
    silhouette. A flat outline plus a 2-D skeleton reads the same from the
    front and from behind; this is the bit that says which."""

    def _ramp(self):
        """Depth 1.0 (near, left) to 2.0 (far, right) across the block."""
        width, height = SIZE
        depth = np.zeros((height, width), dtype=np.float32)
        cols = np.linspace(1.0, 2.0, 8, dtype=np.float32)
        depth[3:9, 4:12] = cols[None, :]
        return depth

    def test_near_is_near_color_and_far_is_far_color(self):
        fill = relief_fill(self._ramp(), _block(), center_depth=1.5,
                           depth_range=1.0, levels=16,
                           far_color=_grey(0x67), near_color=_grey(0x77))
        assert fill.shape == (SIZE[1], SIZE[0], 3) and fill.dtype == np.uint8
        assert fill[5, 4].tolist() == [0x77] * 3     # depth 1.0, the near end
        assert fill[5, 11].tolist() == [0x67] * 3    # depth 2.0, the far end
        # Monotone across the ramp, and every level a distinct byte.
        row = fill[5, 4:12, 0].astype(int)
        assert all(a >= b for a, b in zip(row, row[1:]))
        assert len(set(row)) > 2

    def test_levels_quantise(self):
        fill = relief_fill(self._ramp(), _block(), center_depth=1.5,
                           depth_range=1.0, levels=2,
                           far_color=_grey(0x60), near_color=_grey(0x70))
        assert set(fill[_block()][:, 0].tolist()) == {0x60, 0x70}

    def test_depth_outside_the_window_clamps(self):
        depth = self._ramp()
        fill = relief_fill(depth, _block(), center_depth=1.5,
                           depth_range=0.2, levels=16,
                           far_color=_grey(0x60), near_color=_grey(0x70))
        assert fill[5, 4].tolist() == [0x70] * 3
        assert fill[5, 11].tolist() == [0x60] * 3

    def test_outside_the_mask_is_far_color(self):
        fill = relief_fill(self._ramp(), _block(), center_depth=1.5,
                           depth_range=1.0, levels=16,
                           far_color=_grey(0x60), near_color=_grey(0x70))
        assert np.all(fill[~_block()] == 0x60)

    def test_mask_beyond_the_mesh_takes_the_nearest_depth(self):
        """A matte wider than the body model (hair, a coat) is filled at
        the depth of the nearest covered pixel, not left at some depth of
        its own."""
        depth = self._ramp()
        wide = _block()
        wide[3:9, 2:4] = True     # two columns left of the mesh, nearest = col 4
        wide[3:9, 12:14] = True   # two right, nearest = col 11
        fill = relief_fill(depth, wide, center_depth=1.5, depth_range=1.0,
                           levels=16, far_color=_grey(0x60),
                           near_color=_grey(0x70))
        assert fill[5, 2].tolist() == fill[5, 4].tolist() == [0x70] * 3
        assert fill[5, 13].tolist() == fill[5, 11].tolist() == [0x60] * 3

    def test_smooth_blurs_the_depth_within_the_mask(self):
        depth = self._ramp()
        sharp = relief_fill(depth, _block(), center_depth=1.5, depth_range=1.0,
                            levels=16, far_color=_grey(0x60),
                            near_color=_grey(0x70), smooth=0)
        soft = relief_fill(depth, _block(), center_depth=1.5, depth_range=1.0,
                           levels=16, far_color=_grey(0x60),
                           near_color=_grey(0x70), smooth=2.0)
        # The ends pull toward the middle; the edge of the mask does not
        # pull toward the (absent) background.
        assert soft[5, 4, 0] <= sharp[5, 4, 0]
        assert soft[5, 11, 0] >= sharp[5, 11, 0]
        assert np.all(soft[~_block()] == 0x60)
        assert soft[3, 4, 0] == soft[5, 4, 0]

    def test_no_coverage_is_all_far_color(self):
        fill = relief_fill(np.zeros((SIZE[1], SIZE[0]), np.float32), _block(),
                           center_depth=1.0, depth_range=1.0, levels=4,
                           far_color=_grey(0x60), near_color=_grey(0x70))
        assert np.all(fill == 0x60)

    def test_refusals(self):
        depth = self._ramp()
        with pytest.raises(ValueError, match="levels"):
            relief_fill(depth, _block(), 1.5, 1.0, 1, _grey(0), _grey(1))
        with pytest.raises(ValueError, match="depth_range"):
            relief_fill(depth, _block(), 1.5, 0.0, 4, _grey(0), _grey(1))
        with pytest.raises(ValueError, match="smooth"):
            relief_fill(depth, _block(), 1.5, 1.0, 4, _grey(0), _grey(1), smooth=-1)
        with pytest.raises(ValueError, match="shape"):
            relief_fill(depth, _block((8, 6)), 1.5, 1.0, 4, _grey(0), _grey(1))


class TestFillSeam:
    """`outline_from_mask(fill=...)`: a per-pixel foreground in place of the
    flat colour, everything else — background, alpha, blur — as before."""

    def test_the_fill_replaces_the_foreground_only(self):
        width, height = SIZE
        fill = np.full((height, width, 3), 0x42, dtype=np.uint8)
        image = outline_from_mask(_block(), fg_color=_grey(0x6F),
                                  bg_color=_grey(0x7F), blur=0, fill=fill)
        assert np.all(image[_block()][:, :3] == 0x42)
        assert np.all(image[~_block()][:, :3] == 0x7F)
        assert np.array_equal(image[:, :, 3] == 255, _block())

    def test_a_wrong_fill_is_refused(self):
        width, height = SIZE
        with pytest.raises(ValueError, match="fill"):
            outline_from_mask(_block(), fill=np.zeros((height, width), np.uint8))
        with pytest.raises(ValueError, match="fill"):
            outline_from_mask(_block(), fill=np.zeros((height, width, 3), np.float32))


class TestReliefThroughTheRenderer:
    def test_render_outline_builds_the_fill_from_the_depth_buffer(self):
        """With `relief`, render_outline rasterizes the mesh for its depth
        even when the silhouette is supplied, and hands relief_fill the
        window centred on the given world point's camera depth."""
        from body2colmap.camera import Camera

        width, height = SIZE
        renderer = _MeshlessRenderer(SIZE)
        depth = np.zeros((height, width), np.float32)
        depth[3:9, 4:8] = 1.0   # near half
        depth[3:9, 8:12] = 2.0  # far half
        renderer._render_depth_buffer = lambda camera: depth
        camera = Camera(focal_length=(10.0, 10.0), image_size=SIZE,
                        position=np.zeros(3, np.float32),
                        rotation=np.eye(3, dtype=np.float32))
        image = renderer.render_outline(
            camera, fg_color=_grey(0x6F), bg_color=_grey(0x7F), blur=0,
            mask=_block(),
            relief={"levels": 2, "depth_range": 1.0, "center": (0.0, 0.0, -1.5),
                    "near_color": _grey(0x77), "far_color": _grey(0x67)},
        )
        assert image[5, 5, :3].tolist() == [0x77] * 3
        assert image[5, 10, :3].tolist() == [0x67] * 3
        assert image[0, 0, :3].tolist() == [0x7F] * 3

    def test_relief_needs_the_filled_style(self):
        renderer = _MeshlessRenderer(SIZE)
        with pytest.raises(ValueError, match="filled"):
            renderer.render_outline(None, mask=_block(), style="stroke",
                                    relief={"levels": 2})

    def test_render_composite_forwards_relief(self):
        renderer = _MeshlessRenderer(SIZE)
        seen = {}

        def fake_outline(camera, **kwargs):
            seen.update(kwargs)
            return outline_from_mask(kwargs["mask"], blur=0)

        renderer.render_outline = fake_outline
        renderer.render_composite(
            camera=None,
            modes={"outline": {"mask": _block(), "relief": {"levels": 4}}},
        )
        assert seen["relief"] == {"levels": 4}
