"""
Tests for the conditioning mask that marks a splat overlay as inactive.

The mask is consumed by a video model, not by anything in this package, so the
things worth pinning are the ones a wrong answer would still look plausible
for: the **polarity** (alpha 0 preserves, 255 regenerates — inverted, the mask
tells the model to repaint the one real face in the frame and keep the
annotations, and it still looks like a mask), and the fact that a **culled
frame is wholly reactive rather than unmasked**, since a gap in the sequence no
longer lines up with the frames it describes.
"""

import numpy as np
import pytest

from body2colmap.pipeline import OrbitPipeline
from body2colmap.scene import Scene
from body2colmap.splat_renderer import (
    INACTIVE,
    REACTIVE,
    InactiveMaskOptions,
)


SIZE = (16, 12)   # (width, height) -- deliberately not square


def _layer(alpha, size=SIZE):
    """An RGBA straight-alpha splat layer with the given alpha plane."""
    width, height = size
    layer = np.zeros((height, width, 4), dtype=np.uint8)
    layer[:, :, :3] = 200
    layer[:, :, 3] = alpha
    return layer


def _solid_block(size=SIZE, value=255):
    """An alpha plane fully covered over rows 4:8, columns 5:11."""
    width, height = size
    alpha = np.zeros((height, width), dtype=np.uint8)
    alpha[4:8, 5:11] = value
    return alpha


class TestPolarity:
    """Black is what the model keeps; white is what it generates."""

    def test_the_splat_is_inactive_and_the_rest_reactive(self):
        """The covered region is INACTIVE, everything else REACTIVE."""
        mask = InactiveMaskOptions().build(_layer(_solid_block()), SIZE)

        covered = _solid_block() > 0
        assert np.all(mask[covered] == INACTIVE)
        assert np.all(mask[~covered] == REACTIVE)

    def test_the_two_values_are_the_vace_convention(self):
        """Inactive is 0 and reactive is 255, not the other way round."""
        assert (INACTIVE, REACTIVE) == (0, 255)

    def test_the_mask_is_single_channel_and_frame_sized(self):
        """One 8-bit plane at (height, width), ready to write as greyscale."""
        mask = InactiveMaskOptions().build(_layer(_solid_block()), SIZE)

        width, height = SIZE
        assert mask.shape == (height, width)
        assert mask.dtype == np.uint8


class TestCulledFrames:
    """A frame with no splat still gets a mask."""

    def test_a_culled_frame_is_all_reactive(self):
        """None yields a full-frame reactive mask, not an empty slot."""
        mask = InactiveMaskOptions().build(None, SIZE)

        width, height = SIZE
        assert mask.shape == (height, width)
        assert np.all(mask == REACTIVE)

    def test_an_uncovered_frame_is_all_reactive_too(self):
        """A layer that happens to be empty says the same thing."""
        width, height = SIZE
        mask = InactiveMaskOptions().build(
            _layer(np.zeros((height, width), dtype=np.uint8)), SIZE
        )
        assert np.all(mask == REACTIVE)


class TestThreshold:
    """Partial coverage is a blend, and blends are not preserved by default."""

    def test_partial_coverage_stays_reactive_at_the_default(self):
        """Alpha 0.5 is half synthetic, so the default does not freeze it."""
        alpha = _solid_block(value=128)
        mask = InactiveMaskOptions().build(_layer(alpha), SIZE)
        assert np.all(mask == REACTIVE)

    def test_lowering_it_takes_the_soft_rim_in(self):
        """A threshold under the rim's alpha marks it inactive."""
        alpha = _solid_block(value=128)
        mask = InactiveMaskOptions(threshold=0.4).build(_layer(alpha), SIZE)
        assert np.all(mask[alpha > 0] == INACTIVE)

    def test_it_is_a_floor_not_a_window(self):
        """Fully opaque pixels are covered at every valid threshold."""
        alpha = _solid_block()
        for threshold in (0.1, 0.5, 0.9, 1.0):
            mask = InactiveMaskOptions(threshold=threshold).build(
                _layer(alpha), SIZE
            )
            assert np.all(mask[alpha == 255] == INACTIVE)

    @pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5, 128])
    def test_an_out_of_range_threshold_is_rejected(self, threshold):
        """It is a fraction of alpha; 128 is the likely mistake."""
        with pytest.raises(ValueError, match="threshold"):
            InactiveMaskOptions(threshold=threshold)


class TestGrow:
    """One signed knob moves the boundary either way."""

    def test_positive_grows_the_inactive_region(self):
        """Growing by 1 px takes in the ring around the covered block."""
        alpha = _solid_block()
        plain = InactiveMaskOptions().build(_layer(alpha), SIZE)
        grown = InactiveMaskOptions(grow=1).build(_layer(alpha), SIZE)

        assert (grown == INACTIVE).sum() > (plain == INACTIVE).sum()
        # Everything the plain mask preserved is still preserved.
        assert np.all(grown[plain == INACTIVE] == INACTIVE)

    def test_negative_shrinks_it(self):
        """Shrinking pulls the boundary in off the antialiased edge."""
        alpha = _solid_block()
        plain = InactiveMaskOptions().build(_layer(alpha), SIZE)
        shrunk = InactiveMaskOptions(grow=-1).build(_layer(alpha), SIZE)

        assert (shrunk == INACTIVE).sum() < (plain == INACTIVE).sum()
        # It only ever gives area back, never takes new area.
        assert np.all(plain[shrunk == INACTIVE] == INACTIVE)

    def test_zero_leaves_the_thresholded_coverage_alone(self):
        """The default does no morphology at all."""
        alpha = _solid_block()
        mask = InactiveMaskOptions(grow=0).build(_layer(alpha), SIZE)
        assert np.array_equal(mask == INACTIVE, alpha == 255)


class TestSizeMismatch:
    """A layer from another render is an error, not a silent crop."""

    def test_a_wrongly_sized_layer_is_rejected(self):
        mask_opts = InactiveMaskOptions()
        with pytest.raises(ValueError, match="splat layer is"):
            mask_opts.build(_layer(_solid_block((8, 8)), size=(8, 8)), SIZE)


def _frame(size=SIZE):
    """A finished RGBA composite: opaque, with recognisable colour."""
    width, height = size
    frame = np.zeros((height, width, 4), dtype=np.uint8)
    frame[:, :, 0] = 10
    frame[:, :, 1] = 20
    frame[:, :, 2] = 30
    frame[:, :, 3] = 128        # some other alpha, e.g. a silhouette
    return frame


class TestApply:
    """The frame carries its own mask, in the alpha channel."""

    def test_alpha_becomes_the_mask(self):
        """The splat's pixels go to 0, the rest of the frame to 255."""
        alpha = _solid_block()
        frame = InactiveMaskOptions().apply(_frame(), _layer(alpha))

        assert np.all(frame[:, :, 3][alpha == 255] == INACTIVE)
        assert np.all(frame[:, :, 3][alpha == 0] == REACTIVE)

    def test_colour_is_untouched(self):
        """
        An inactive pixel is marked, not erased: the splat's colour is the
        real content the mask exists to preserve.
        """
        frame = InactiveMaskOptions().apply(_frame(), _layer(_solid_block()))
        assert np.array_equal(frame[:, :, :3], _frame()[:, :, :3])

    def test_it_replaces_whatever_alpha_was_there(self):
        """
        The composite's alpha means silhouette coverage; the mask means
        something else entirely, and the two cannot share the channel.
        """
        frame = InactiveMaskOptions().apply(_frame(), _layer(_solid_block()))
        assert not np.any(frame[:, :, 3] == 128)

    def test_a_culled_frame_is_wholly_reactive(self):
        """No splat in the frame means nothing in it to preserve."""
        frame = InactiveMaskOptions().apply(_frame(), None)
        assert np.all(frame[:, :, 3] == REACTIVE)


class TestPipelineGuard:
    """A mask over nothing is a mistake worth reporting."""

    @staticmethod
    def _pipeline():
        """A pipeline over a trivial mesh, with cameras but no splat."""
        vertices = np.array(
            [[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.3, 0.0]],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        pipeline = OrbitPipeline(
            Scene(vertices=vertices, faces=faces), render_size=(32, 32)
        )
        pipeline.set_orbit_params(pattern="circular", n_frames=2, radius=2.0)
        return pipeline

    def test_a_mask_without_an_overlay_is_refused(self):
        """
        With no splat the mask would come out uniformly reactive -- a valid
        image saying nothing, which is worse than an error.
        """
        with pytest.raises(ValueError, match="none is attached"):
            self._pipeline().render_composite_all(
                {"mesh": {}}, inactive_mask=InactiveMaskOptions()
            )
