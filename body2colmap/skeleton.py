"""
Skeleton format definitions and bone connectivity.

This module defines various skeleton formats (joint counts, bone connections)
and provides conversion utilities between formats.

Supported formats:
- MHR70: SAM-3D-Body's 70-joint format (default input)
- OpenPose Body25: 25-joint body-only format
- OpenPose Body25+Hands: 65-joint format (body + hands)
- COCO: 17-joint format
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from numpy.typing import NDArray


# OpenPose color palette (RGB 0-1 range)
# Colors for different body parts following OpenPose conventions
OPENPOSE_COLORS = {
    "head": (1.0, 0.0, 0.85),      # Magenta
    "torso": (0.7, 0.3, 1.0),       # Purple
    "right_arm": (1.0, 0.0, 0.0),   # Red
    "left_arm": (0.0, 1.0, 0.0),    # Green
    "right_leg": (0.0, 0.0, 1.0),   # Blue
    "left_leg": (1.0, 1.0, 0.0),    # Yellow
}


# OpenPose Body25 + Hands skeleton (65 joints total)
# Body: 25 joints (0-24)
# Left hand: 20 joints (25-44)
# Right hand: 20 joints (45-64)

OPENPOSE_BODY25_HANDS_BONES = [
    # Body bones (based on Body25)
    (0, 1),   # Nose → Neck
    (1, 2),   # Neck → RShoulder
    (2, 3),   # RShoulder → RElbow
    (3, 4),   # RElbow → RWrist
    (1, 5),   # Neck → LShoulder
    (5, 6),   # LShoulder → LElbow
    (6, 7),   # LElbow → LWrist
    (1, 8),   # Neck → MidHip
    (8, 9),   # MidHip → RHip
    (9, 10),  # RHip → RKnee
    (10, 11), # RKnee → RAnkle
    (8, 12),  # MidHip → LHip
    (12, 13), # LHip → LKnee
    (13, 14), # LKnee → LAnkle
    (0, 15),  # Nose → REye
    (15, 17), # REye → REar
    (0, 16),  # Nose → LEye
    (16, 18), # LEye → LEar
    (14, 19), # LAnkle → LBigToe
    (19, 20), # LBigToe → LSmallToe
    (14, 21), # LAnkle → LHeel
    (11, 22), # RAnkle → RBigToe
    (22, 23), # RBigToe → RSmallToe
    (11, 24), # RAnkle → RHeel
]

# Right hand bones (20 joints: 45-64)
OPENPOSE_RIGHT_HAND_BONES = [
    (4, 45),   # RWrist → RThumb1
    (45, 46),  # RThumb joints
    (46, 47),
    (47, 48),
    (4, 49),   # RWrist → RIndex1
    (49, 50),  # RIndex joints
    (50, 51),
    (51, 52),
    (4, 53),   # RWrist → RMiddle1
    (53, 54),  # RMiddle joints
    (54, 55),
    (55, 56),
    (4, 57),   # RWrist → RRing1
    (57, 58),  # RRing joints
    (58, 59),
    (59, 60),
    (4, 61),   # RWrist → RPinky1
    (61, 62),  # RPinky joints
    (62, 63),
    (63, 64),
]

# Left hand bones (20 joints: 25-44)
OPENPOSE_LEFT_HAND_BONES = [
    (7, 25),   # LWrist → LThumb1
    (25, 26),  # LThumb joints
    (26, 27),
    (27, 28),
    (7, 29),   # LWrist → LIndex1
    (29, 30),  # LIndex joints
    (30, 31),
    (31, 32),
    (7, 33),   # LWrist → LMiddle1
    (33, 34),  # LMiddle joints
    (34, 35),
    (35, 36),
    (7, 37),   # LWrist → LRing1
    (37, 38),  # RRing joints
    (38, 39),
    (39, 40),
    (7, 41),   # LWrist → LPinky1
    (41, 42),  # LPinky joints
    (42, 43),
    (43, 44),
]

# Combine all bones for full skeleton
OPENPOSE_BODY25_HANDS_ALL_BONES = (
    OPENPOSE_BODY25_HANDS_BONES +
    OPENPOSE_RIGHT_HAND_BONES +
    OPENPOSE_LEFT_HAND_BONES
)


# MHR70 bone connectivity (SAM-3D-Body / SMPL-X based)
# MHR70 follows SMPL-X topology with 70 joints:
# - Body: ~25 joints (similar to SMPL)
# - Hands: ~40 joints (20 per hand)
# - Face: ~5 joints

# Main body bones for MHR70 (estimated based on SMPL-X structure)
MHR70_BODY_BONES = [
    # Spine/torso
    (0, 3),   # Pelvis → Spine1
    (3, 6),   # Spine1 → Spine2
    (6, 9),   # Spine2 → Spine3
    (9, 12),  # Spine3 → Neck
    (12, 15), # Neck → Head

    # Left arm
    (9, 13),  # Spine3 → LShoulder
    (13, 16), # LShoulder → LElbow
    (16, 18), # LElbow → LWrist

    # Right arm
    (9, 14),  # Spine3 → RShoulder
    (14, 17), # RShoulder → RElbow
    (17, 19), # RElbow → RWrist

    # Left leg
    (0, 1),   # Pelvis → LHip
    (1, 4),   # LHip → LKnee
    (4, 7),   # LKnee → LAnkle
    (7, 10),  # LAnkle → LFoot

    # Right leg
    (0, 2),   # Pelvis → RHip
    (2, 5),   # RHip → RKnee
    (5, 8),   # RKnee → RAnkle
    (8, 11),  # RAnkle → RFoot
]


def get_skeleton_bones(format_name: str) -> List[Tuple[int, int]]:
    """
    Get bone connectivity for a skeleton format.

    Args:
        format_name: Skeleton format identifier
            - "openpose_body25_hands" (65 joints)
            - "openpose_body25" (25 joints, body only)
            - "mhr70" (70 joints, SAM-3D-Body default)

    Returns:
        List of bone connections as (start_joint_idx, end_joint_idx) tuples

    Raises:
        ValueError: If format is unknown
    """
    if format_name == "openpose_body25_hands":
        return OPENPOSE_BODY25_HANDS_ALL_BONES
    elif format_name == "openpose_body25":
        return OPENPOSE_BODY25_HANDS_BONES
    elif format_name == "mhr70":
        return MHR70_BODY_BONES
    else:
        raise ValueError(f"Unknown skeleton format: {format_name}")


def convert_skeleton(
    joints: NDArray[np.float32],
    from_format: str,
    to_format: str
) -> Optional[NDArray[np.float32]]:
    """
    Convert skeleton from one format to another.

    Args:
        joints: Joint positions, shape (N, 3)
        from_format: Source format identifier
        to_format: Target format identifier

    Returns:
        Converted joint positions, shape (M, 3)
        None if conversion not implemented

    Note:
        For now, we only support pass-through (no conversion).
        Full conversion between formats requires detailed joint mapping.
    """
    if from_format == to_format:
        return joints

    # TODO: Implement format conversions
    # For now, return None to indicate conversion not available
    return None


def validate_skeleton_format(joints: NDArray[np.float32], format_name: str) -> bool:
    """
    Validate that joint array matches expected format.

    Args:
        joints: Joint positions, shape (N, 3)
        format_name: Format identifier

    Returns:
        True if valid, False otherwise
    """
    expected_counts = {
        "mhr70": 70,
        "openpose_body25": 25,
        "openpose_body25_hands": 65,
        "coco": 17,
    }

    expected = expected_counts.get(format_name)
    if expected is None:
        return False

    return len(joints) == expected


def get_bone_colors_openpose_style(format_name: str) -> Dict[Tuple[int, int], Tuple[float, float, float]]:
    """
    Get OpenPose-style colors for each bone.

    Returns a dictionary mapping bone (start_idx, end_idx) to RGB color (0-1 range).

    Args:
        format_name: Skeleton format identifier

    Returns:
        Dictionary mapping bones to colors
    """
    if format_name == "mhr70":
        # Color each bone based on body part
        colors = {}

        # Head/neck bones
        for bone in [(12, 15)]:  # Neck → Head
            colors[bone] = OPENPOSE_COLORS["head"]

        # Torso bones
        for bone in [(0, 3), (3, 6), (6, 9), (9, 12)]:  # Spine chain
            colors[bone] = OPENPOSE_COLORS["torso"]

        # Right arm bones
        for bone in [(9, 14), (14, 17), (17, 19)]:  # Spine3 → RShoulder → RElbow → RWrist
            colors[bone] = OPENPOSE_COLORS["right_arm"]

        # Left arm bones
        for bone in [(9, 13), (13, 16), (16, 18)]:  # Spine3 → LShoulder → LElbow → LWrist
            colors[bone] = OPENPOSE_COLORS["left_arm"]

        # Right leg bones
        for bone in [(0, 2), (2, 5), (5, 8), (8, 11)]:  # Pelvis → RHip → RKnee → RAnkle → RFoot
            colors[bone] = OPENPOSE_COLORS["right_leg"]

        # Left leg bones
        for bone in [(0, 1), (1, 4), (4, 7), (7, 10)]:  # Pelvis → LHip → LKnee → LAnkle → LFoot
            colors[bone] = OPENPOSE_COLORS["left_leg"]

        return colors

    else:
        # Default: single color for all bones
        default_color = (0.0, 1.0, 0.0)  # Green
        bones = get_skeleton_bones(format_name)
        return {bone: default_color for bone in bones}
