"""The DWPose drawing convention, pinned against the code it reproduces.

Wan 2.2 VACE conditions on pose maps drawn by DWPose, so a skeleton meant to
read as pose has to be drawn the way DWPose draws one. The numbers asserted
here are `draw_bodypose` / `draw_handpose`'s own, from VACE's
`vace/annotators/dwpose/util.py` and the identical code in
`comfyui_controlnet_aux`; the module under test explains where each comes from.

They are pinned rather than derived because the whole point of the style is
that it is somebody else's convention: a change to one of these numbers is a
change to what the model is being shown, and should have to be typed twice.
"""

import unittest

import numpy as np

from body2colmap import skeleton as S
from body2colmap.renderer import _PYRENDER_OUTPUT_GAMMA, _pyrender_rgba


def _byte(color):
    return tuple(round(c * 255) for c in color)


class TestDWPoseBoneColors(unittest.TestCase):
    def setUp(self):
        self.colors = S.get_bone_colors_dwpose()

    def test_every_drawn_bone_has_a_color_and_no_bone_is_invented(self):
        """The renderer looks colours up by the exact (start, end) tuple it is
        drawing, so a bone keyed the other way round silently falls back to
        green."""
        self.assertEqual(set(self.colors), set(S.get_skeleton_bones_dwpose()))

    def test_body_limbs_are_dimmed_to_sixty_percent(self):
        """`canvas = (canvas * 0.6)` — the single biggest difference from the
        older style, and the reason DWPose sticks read dark."""
        self.assertEqual(_byte(self.colors[(1, 2)]), (153, 0, 0))     # neck-RSho
        self.assertEqual(_byte(self.colors[(1, 0)]), (0, 0, 153))     # neck-nose
        self.assertEqual(_byte(self.colors[(3, 4)]), (153, 153, 0))   # RElb-RWri

    def test_the_palette_is_indexed_off_dwposes_limb_order(self):
        """Where the older style went one hue step out: it spent colours on
        the MidHip bones DWPose has no equivalent for, shifting everything
        below the hips and across the whole head."""
        for bone, limb_index in (
            ((1, 2), 0), ((1, 5), 1), ((2, 3), 2), ((3, 4), 3),
            ((1, 9), 6), ((9, 10), 7), ((10, 11), 8),
            ((1, 12), 9), ((12, 13), 10), ((13, 14), 11),
            ((1, 0), 12), ((0, 15), 13), ((15, 17), 14), ((0, 16), 15),
            ((16, 18), 16),
        ):
            with self.subTest(bone=bone):
                expected = tuple(
                    c * S.DWPOSE_LIMB_DIM for c in S.DWPOSE_BODY18_COLORS[limb_index]
                )
                self.assertEqual(_byte(self.colors[bone]), _byte(expected))

    def test_the_torso_runs_neck_to_hip_with_no_midhip(self):
        """Every DWPose torso limb starts at the neck, so the two hip limbs
        cross the torso and overlap. BODY_25's Y meeting at the pelvis is a
        visibly different shape, and one no pose map has ever had."""
        bones = S.get_skeleton_bones_dwpose()
        self.assertIn((1, 9), bones)    # neck -> RHip, drawn whole
        self.assertIn((1, 12), bones)   # neck -> LHip, drawn whole
        self.assertEqual(_byte(self.colors[(1, 9)]), (0, 153, 0))
        self.assertEqual(_byte(self.colors[(1, 12)]), (0, 153, 153))
        for split in ((1, 8), (8, 9), (8, 12)):
            with self.subTest(bone=split):
                self.assertNotIn(split, bones)

    def test_nothing_reaches_a_joint_dwpose_has_no_keypoint_for(self):
        """The feet and MidHip. Both are structures that have never been in a
        VACE pose map, not stylistic differences from one."""
        for bone in S.get_skeleton_bones_dwpose():
            with self.subTest(bone=bone):
                self.assertFalse(
                    S.DWPOSE_UNDRAWN_BODY25_JOINTS.intersection(bone)
                )

    def test_the_body_is_exactly_dwposes_limbs_and_the_hands_are_extra(self):
        """57 bones: limbSeq[:17], then draw_handpose's 20 edges per hand."""
        bones = S.get_skeleton_bones_dwpose()
        self.assertEqual(len(bones), 17 + 20 + 20)
        self.assertEqual(bones[:17], list(S.DWPOSE_LIMB_SEQ_BODY25))
        self.assertIn((13, 14), self.colors)   # LKnee -> LAnkle, still there
        self.assertIn((10, 11), self.colors)   # RKnee -> RAnkle


class TestDWPoseHands(unittest.TestCase):
    def setUp(self):
        self.colors = S.get_bone_colors_dwpose()
        self.scales = S.get_bone_radius_scales_dwpose()

    def test_hand_bones_are_a_quarter_of_a_limbs_width(self):
        """`draw_handpose` strokes at thickness=2 against a limb's 8."""
        hand_bones = set(
            S.OPENPOSE_RIGHT_HAND_BONES + S.OPENPOSE_LEFT_HAND_BONES
        )
        self.assertEqual(len(self.scales), len(hand_bones))
        self.assertEqual(set(self.scales.values()), {0.25})
        for bone in self.scales:
            self.assertIn(bone, hand_bones)

    def test_no_body_bone_is_rescaled(self):
        for bone in S.DWPOSE_LIMB_SEQ_BODY25:
            with self.subTest(bone=bone):
                self.assertNotIn(bone, self.scales)

    def test_hands_are_a_full_hue_sweep_and_are_not_dimmed(self):
        """`draw_handpose` runs after the 0.6 pass, so hands keep full value —
        which is also what makes the sweep read at that width."""
        for bones in (S.OPENPOSE_RIGHT_HAND_BONES, S.OPENPOSE_LEFT_HAND_BONES):
            self.assertEqual(_byte(self.colors[bones[0]]), (255, 0, 0))
            hues = [max(self.colors[b]) for b in bones]
            self.assertEqual(set(hues), {1.0})

    def test_both_hands_get_the_same_sweep(self):
        """draw_handpose does not distinguish left from right."""
        for right, left in zip(
            S.OPENPOSE_RIGHT_HAND_BONES, S.OPENPOSE_LEFT_HAND_BONES
        ):
            with self.subTest(right=right):
                self.assertEqual(self.colors[right], self.colors[left])


class TestDWPoseJointColors(unittest.TestCase):
    def setUp(self):
        self.joints = S.get_joint_colors_dwpose(65)

    def test_dots_are_undimmed(self):
        """The dots go down after the 0.6 pass, which is the only thing making
        them visible: DWPose draws them exactly as wide as a limb."""
        self.assertEqual(_byte(self.joints[0]), (255, 0, 0))      # nose
        self.assertEqual(_byte(self.joints[1]), (255, 85, 0))     # neck

    def test_a_dot_is_colored_by_its_own_joint_index(self):
        """Not read off the bones it touches — which is why the style has to
        supply its own table rather than letting the renderer derive one."""
        for body18, body25 in enumerate(S.DWPOSE_BODY18_TO_BODY25):
            with self.subTest(body25=body25):
                if body25 in (4, 7):
                    continue  # the wrists, repainted by the hand pass below
                self.assertEqual(
                    _byte(self.joints[body25]),
                    _byte(S.DWPOSE_BODY18_COLORS[body18]),
                )

    def test_every_hand_keypoint_is_blue_wrists_included(self):
        """A hand's keypoint 0 IS the wrist, and draw_handpose goes down after
        draw_bodypose, so it lands on top of the body's own dot."""
        for index in range(S.OPENPOSE_BODY25_HANDS_N_BODY_JOINTS, 65):
            with self.subTest(joint=index):
                self.assertEqual(self.joints[index], S.DWPOSE_HAND_JOINT_COLOR)
        for wrist in (4, 7):
            self.assertEqual(self.joints[wrist], S.DWPOSE_HAND_JOINT_COLOR)

    def test_no_joint_is_left_at_the_default_white(self):
        """Every joint is either coloured or explicitly not drawn. White is
        get_joint_colors_from_bones' fallback and would mean neither."""
        self.assertNotIn((1.0, 1.0, 1.0), self.joints)

    def test_a_joint_dwpose_has_no_keypoint_for_gets_no_dot(self):
        """No bone reaches MidHip or a toe any more, so a dot there would be
        a speck with nothing attached. The renderer otherwise puts a sphere at
        every joint in the array."""
        for joint in S.DWPOSE_UNDRAWN_BODY25_JOINTS:
            with self.subTest(joint=joint):
                self.assertIsNone(self.joints[joint])
        for joint in (9, 12, 13, 14):   # the hips and ankles still get theirs
            self.assertIsNotNone(self.joints[joint])

    def test_the_dot_count_is_dwposes_own(self):
        """18 body keypoints plus 20 per hand — the wrists are shared, being
        both BODY_18 joints and each hand's keypoint 0."""
        self.assertEqual(sum(c is not None for c in self.joints), 18 + 40)


class TestPyrenderGamma(unittest.TestCase):
    """pyrender's shader ends on pow(color, 1/2.2), so a colour handed to it
    comes back lifted. DWPose's numbers are the bytes in its output PNG, so
    they have to survive it."""

    def test_linearizing_round_trips_through_the_shader(self):
        for requested in (0.6, 0.333, 1.0, 0.0):
            with self.subTest(requested=requested):
                encoded = _pyrender_rgba((requested,) * 3, linearize=True)[0]
                rendered = (encoded / 255.0) ** (1.0 / _PYRENDER_OUTPUT_GAMMA)
                self.assertAlmostEqual(rendered, requested, places=2)

    def test_the_older_style_is_left_lifted(self):
        """Its renders have always carried the lift. Correcting it would be a
        different picture, not a fixed one — and it is the "before" the DWPose
        style is judged against."""
        self.assertEqual(int(_pyrender_rgba((0.6, 0.0, 0.0))[0]), 153)
        self.assertLess(int(_pyrender_rgba((0.6, 0.0, 0.0), linearize=True)[0]), 153)


class TestStyleValidation(unittest.TestCase):
    def test_dwpose_is_only_defined_for_the_topology_it_describes(self):
        from body2colmap.renderer import Renderer
        from body2colmap.scene import Scene

        scene = Scene(
            vertices=np.zeros((3, 3), np.float32),
            faces=np.zeros((1, 3), np.int32),
            skeleton_joints=np.zeros((70, 3), np.float32),
            skeleton_format="mhr70",
        )
        renderer = Renderer(scene, render_size=(8, 8))
        with self.assertRaises(ValueError) as caught:
            renderer.render_skeleton(camera=None, style="dwpose",
                                     target_format="mhr70")
        self.assertIn("openpose_body25_hands", str(caught.exception))

    def test_an_unknown_style_is_refused(self):
        from body2colmap.renderer import Renderer
        from body2colmap.scene import Scene

        scene = Scene(
            vertices=np.zeros((3, 3), np.float32),
            faces=np.zeros((1, 3), np.int32),
            skeleton_joints=np.zeros((70, 3), np.float32),
            skeleton_format="mhr70",
        )
        renderer = Renderer(scene, render_size=(8, 8))
        with self.assertRaises(ValueError) as caught:
            renderer.render_skeleton(camera=None, style="controlnet")
        self.assertIn("controlnet", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
