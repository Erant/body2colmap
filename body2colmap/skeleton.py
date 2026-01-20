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


# MHR70 color palette (RGB 0-1 range)
# Colors from official SAM-3D-Body visualization
MHR70_COLORS = {
    "left": (0.0, 1.0, 0.0),           # Green [0, 255, 0]
    "right": (1.0, 0.502, 0.0),        # Orange [255, 128, 0]
    "center": (0.2, 0.6, 1.0),         # Cyan [51, 153, 255]
    "left_index": (1.0, 0.6, 1.0),     # Pink [255, 153, 255]
    "left_middle": (0.4, 0.698, 1.0),  # Light Blue [102, 178, 255]
    "left_ring": (1.0, 0.2, 0.2),      # Red [255, 51, 51]
    "right_index": (1.0, 0.6, 1.0),    # Pink [255, 153, 255]
    "right_middle": (0.4, 0.698, 1.0), # Light Blue [102, 178, 255]
    "right_ring": (1.0, 0.2, 0.2),     # Red [255, 51, 51]
}

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


# MHR70 bone connectivity (from SAM-3D-Body official definitions)
# https://github.com/facebookresearch/sam-3d-body/blob/main/sam_3d_body/metadata/mhr70.py
# 70 joints total including body, hands, and face keypoints

# MHR70 joint indices (key body joints)
MHR70_JOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_hip": 9,
    "right_hip": 10,
    "left_knee": 11,
    "right_knee": 12,
    "left_ankle": 13,
    "right_ankle": 14,
    "left_big_toe": 15,
    "left_small_toe": 16,
    "left_heel": 17,
    "right_big_toe": 18,
    "right_small_toe": 19,
    "right_heel": 20,
    "right_wrist": 41,
    "left_wrist": 62,
    "neck": 69,
}

# MHR70 bone connectivity (from SAM-3D-Body official definitions)
# https://github.com/facebookresearch/sam-3d-body/blob/main/sam_3d_body/metadata/mhr70.py
# Total: 65 bones (11 legs + 7 torso + 7 head + 20 left hand + 20 right hand)

MHR70_BODY_BONES = [
    # Legs (11 bones)
    (13, 11),  # left_ankle → left_knee
    (11, 9),   # left_knee → left_hip
    (14, 12),  # right_ankle → right_knee
    (12, 10),  # right_knee → right_hip
    (9, 10),   # left_hip → right_hip
    (13, 15),  # left_ankle → left_big_toe
    (13, 16),  # left_ankle → left_small_toe
    (13, 17),  # left_ankle → left_heel
    (14, 18),  # right_ankle → right_big_toe
    (14, 19),  # right_ankle → right_small_toe
    (14, 20),  # right_ankle → right_heel

    # Torso (7 bones)
    (5, 9),    # left_shoulder → left_hip
    (6, 10),   # right_shoulder → right_hip
    (5, 6),    # left_shoulder → right_shoulder
    (5, 7),    # left_shoulder → left_elbow
    (6, 8),    # right_shoulder → right_elbow
    (7, 62),   # left_elbow → left_wrist
    (8, 41),   # right_elbow → right_wrist

    # Head (7 bones)
    (1, 2),    # left_eye → right_eye
    (0, 1),    # nose → left_eye
    (0, 2),    # nose → right_eye
    (1, 3),    # left_eye → left_ear
    (2, 4),    # right_eye → right_ear
    (3, 5),    # left_ear → left_shoulder
    (4, 6),    # right_ear → right_shoulder

    # Left hand (20 bones) - wrist index is 62
    # Thumb
    (62, 45),  # left_wrist → thumb_base
    (45, 44),
    (44, 43),
    (43, 42),  # → thumb_tip
    # Index finger
    (62, 49),  # left_wrist → index_base
    (49, 48),
    (48, 47),
    (47, 46),  # → index_tip
    # Middle finger
    (62, 53),  # left_wrist → middle_base
    (53, 52),
    (52, 51),
    (51, 50),  # → middle_tip
    # Ring finger
    (62, 57),  # left_wrist → ring_base
    (57, 56),
    (56, 55),
    (55, 54),  # → ring_tip
    # Pinky finger
    (62, 61),  # left_wrist → pinky_base
    (61, 60),
    (60, 59),
    (59, 58),  # → pinky_tip

    # Right hand (20 bones) - wrist index is 41
    # Thumb
    (41, 24),  # right_wrist → thumb_base
    (24, 23),
    (23, 22),
    (22, 21),  # → thumb_tip
    # Index finger
    (41, 28),  # right_wrist → index_base
    (28, 27),
    (27, 26),
    (26, 25),  # → index_tip
    # Middle finger
    (41, 32),  # right_wrist → middle_base
    (32, 31),
    (31, 30),
    (30, 29),  # → middle_tip
    # Ring finger
    (41, 36),  # right_wrist → ring_base
    (36, 35),
    (35, 34),
    (34, 33),  # → ring_tip
    # Pinky finger
    (41, 40),  # right_wrist → pinky_base
    (40, 39),
    (39, 38),
    (38, 37),  # → pinky_tip
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


def get_bone_colors_mhr70() -> Dict[Tuple[int, int], Tuple[float, float, float]]:
    """
    Get official MHR70 colors for each bone.

    Returns a dictionary mapping bone (start_idx, end_idx) to RGB color (0-1 range).
    Colors follow the official SAM-3D-Body visualization scheme.

    Returns:
        Dictionary mapping bones to colors
    """
    colors = {}

    # Left leg bones (green)
    for bone in [(13, 11), (11, 9), (13, 15), (13, 16), (13, 17)]:
        colors[bone] = MHR70_COLORS["left"]

    # Right leg bones (orange)
    for bone in [(14, 12), (12, 10), (14, 18), (14, 19), (14, 20)]:
        colors[bone] = MHR70_COLORS["right"]

    # Center/torso bones (cyan)
    for bone in [(9, 10), (5, 9), (6, 10), (5, 6)]:
        colors[bone] = MHR70_COLORS["center"]

    # Left arm bones (green)
    for bone in [(5, 7), (7, 62)]:
        colors[bone] = MHR70_COLORS["left"]

    # Right arm bones (orange)
    for bone in [(6, 8), (8, 41)]:
        colors[bone] = MHR70_COLORS["right"]

    # Head bones (cyan)
    for bone in [(1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]:
        colors[bone] = MHR70_COLORS["center"]

    # Left hand bones (green with variations for fingers)
    left_thumb = [(62, 45), (45, 44), (44, 43), (43, 42)]
    left_index = [(62, 49), (49, 48), (48, 47), (47, 46)]
    left_middle = [(62, 53), (53, 52), (52, 51), (51, 50)]
    left_ring = [(62, 57), (57, 56), (56, 55), (55, 54)]
    left_pinky = [(62, 61), (61, 60), (60, 59), (59, 58)]

    for bone in left_thumb + left_pinky:
        colors[bone] = MHR70_COLORS["left"]
    for bone in left_index:
        colors[bone] = MHR70_COLORS["left_index"]
    for bone in left_middle:
        colors[bone] = MHR70_COLORS["left_middle"]
    for bone in left_ring:
        colors[bone] = MHR70_COLORS["left_ring"]

    # Right hand bones (orange with variations for fingers)
    right_thumb = [(41, 24), (24, 23), (23, 22), (22, 21)]
    right_index = [(41, 28), (28, 27), (27, 26), (26, 25)]
    right_middle = [(41, 32), (32, 31), (31, 30), (30, 29)]
    right_ring = [(41, 36), (36, 35), (35, 34), (34, 33)]
    right_pinky = [(41, 40), (40, 39), (39, 38), (38, 37)]

    for bone in right_thumb + right_pinky:
        colors[bone] = MHR70_COLORS["right"]
    for bone in right_index:
        colors[bone] = MHR70_COLORS["right_index"]
    for bone in right_middle:
        colors[bone] = MHR70_COLORS["right_middle"]
    for bone in right_ring:
        colors[bone] = MHR70_COLORS["right_ring"]

    return colors


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
        return get_bone_colors_mhr70()

    else:
        # Default: single color for all bones
        default_color = (0.0, 1.0, 0.0)  # Green
        bones = get_skeleton_bones(format_name)
        return {bone: default_color for bone in bones}
