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


# Mapping from OpenPose Body25+Hands (65 joints) to MHR70 (70 joints)
# OpenPose index → MHR70 index
MHR70_TO_OPENPOSE_BODY25_HANDS = {
    # Body (25 joints: 0-24)
    0: 0,    # Nose → nose
    1: 69,   # Neck → neck
    2: 6,    # RShoulder → right_shoulder
    3: 8,    # RElbow → right_elbow
    4: 41,   # RWrist → right_wrist
    5: 5,    # LShoulder → left_shoulder
    6: 7,    # LElbow → left_elbow
    7: 62,   # LWrist → left_wrist
    8: None, # MidHip → computed as average of left_hip (9) and right_hip (10)
    9: 10,   # RHip → right_hip
    10: 12,  # RKnee → right_knee
    11: 14,  # RAnkle → right_ankle
    12: 9,   # LHip → left_hip
    13: 11,  # LKnee → left_knee
    14: 13,  # LAnkle → left_ankle
    15: 2,   # REye → right_eye
    16: 1,   # LEye → left_eye
    17: 4,   # REar → right_ear
    18: 3,   # LEar → left_ear
    19: 15,  # LBigToe → left_big_toe
    20: 16,  # LSmallToe → left_small_toe
    21: 17,  # LHeel → left_heel
    22: 18,  # RBigToe → right_big_toe
    23: 19,  # RSmallToe → right_small_toe
    24: 20,  # RHeel → right_heel

    # Left hand (20 joints: 25-44)
    # MHR70: 42-45 (thumb base to tip), 46-49 (index), 50-53 (middle), 54-57 (ring), 58-61 (pinky)
    # OpenPose: wrist-to-tip ordering (joint 0 = base/MCP, joint 3 = tip)
    25: 45,  # LHand_Thumb0 (MCP) → left_thumb_base
    26: 44,  # LHand_Thumb1
    27: 43,  # LHand_Thumb2
    28: 42,  # LHand_Thumb3 (tip) → left_thumb_tip
    29: 49,  # LHand_Index0
    30: 48,  # LHand_Index1
    31: 47,  # LHand_Index2
    32: 46,  # LHand_Index3 (tip)
    33: 53,  # LHand_Middle0
    34: 52,  # LHand_Middle1
    35: 51,  # LHand_Middle2
    36: 50,  # LHand_Middle3 (tip)
    37: 57,  # LHand_Ring0
    38: 56,  # LHand_Ring1
    39: 55,  # LHand_Ring2
    40: 54,  # LHand_Ring3 (tip)
    41: 61,  # LHand_Pinky0
    42: 60,  # LHand_Pinky1
    43: 59,  # LHand_Pinky2
    44: 58,  # LHand_Pinky3 (tip)

    # Right hand (20 joints: 45-64)
    # MHR70: 21-24 (thumb tip to base), 25-28 (index), 29-32 (middle), 33-36 (ring), 37-40 (pinky)
    45: 24,  # RHand_Thumb0 (MCP) → right_thumb_base
    46: 23,  # RHand_Thumb1
    47: 22,  # RHand_Thumb2
    48: 21,  # RHand_Thumb3 (tip) → right_thumb_tip
    49: 28,  # RHand_Index0
    50: 27,  # RHand_Index1
    51: 26,  # RHand_Index2
    52: 25,  # RHand_Index3 (tip)
    53: 32,  # RHand_Middle0
    54: 31,  # RHand_Middle1
    55: 30,  # RHand_Middle2
    56: 29,  # RHand_Middle3 (tip)
    57: 36,  # RHand_Ring0
    58: 35,  # RHand_Ring1
    59: 34,  # RHand_Ring2
    60: 33,  # RHand_Ring3 (tip)
    61: 40,  # RHand_Pinky0
    62: 39,  # RHand_Pinky1
    63: 38,  # RHand_Pinky2
    64: 37,  # RHand_Pinky3 (tip)
}


def convert_mhr70_to_openpose_body25_hands(mhr70_joints: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Convert MHR70 skeleton (70 joints) to OpenPose Body25+Hands format (65 joints).

    Args:
        mhr70_joints: MHR70 joint positions, shape (70, 3)

    Returns:
        OpenPose Body25+Hands joint positions, shape (65, 3)

    Raises:
        ValueError: If input doesn't have 70 joints
    """
    if len(mhr70_joints) != 70:
        raise ValueError(f"Expected 70 MHR70 joints, got {len(mhr70_joints)}")

    openpose_joints = np.zeros((65, 3), dtype=np.float32)

    for openpose_idx, mhr70_idx in MHR70_TO_OPENPOSE_BODY25_HANDS.items():
        if mhr70_idx is None:
            # Special case: MidHip = average of left_hip and right_hip
            if openpose_idx == 8:
                openpose_joints[8] = (mhr70_joints[9] + mhr70_joints[10]) / 2.0
        else:
            openpose_joints[openpose_idx] = mhr70_joints[mhr70_idx]

    return openpose_joints


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


def get_bone_colors_openpose_body25_hands() -> Dict[Tuple[int, int], Tuple[float, float, float]]:
    """
    Get OpenPose-style colors for Body25+Hands skeleton.

    Returns a dictionary mapping bone (start_idx, end_idx) to RGB color (0-1 range).
    Colors follow OpenPose visualization conventions.

    Returns:
        Dictionary mapping bones to colors
    """
    colors = {}

    # Head bones (magenta)
    for bone in [(0, 15), (15, 17), (0, 16), (16, 18)]:  # Nose-eyes-ears
        colors[bone] = OPENPOSE_COLORS["head"]

    # Torso bones (purple)
    for bone in [(0, 1), (1, 8)]:  # Nose-Neck, Neck-MidHip
        colors[bone] = OPENPOSE_COLORS["torso"]

    # Right arm bones (red)
    for bone in [(1, 2), (2, 3), (3, 4)]:  # Neck-RShoulder-RElbow-RWrist
        colors[bone] = OPENPOSE_COLORS["right_arm"]

    # Left arm bones (green)
    for bone in [(1, 5), (5, 6), (6, 7)]:  # Neck-LShoulder-LElbow-LWrist
        colors[bone] = OPENPOSE_COLORS["left_arm"]

    # Right leg bones (blue)
    for bone in [(8, 9), (9, 10), (10, 11), (11, 22), (22, 23), (11, 24)]:  # MidHip-RHip-RKnee-RAnkle-toes-heel
        colors[bone] = OPENPOSE_COLORS["right_leg"]

    # Left leg bones (yellow)
    for bone in [(8, 12), (12, 13), (13, 14), (14, 19), (19, 20), (14, 21)]:  # MidHip-LHip-LKnee-LAnkle-toes-heel
        colors[bone] = OPENPOSE_COLORS["left_leg"]

    # Left hand bones (green, same as left arm)
    for bone in OPENPOSE_LEFT_HAND_BONES:
        colors[bone] = OPENPOSE_COLORS["left_arm"]

    # Right hand bones (red, same as right arm)
    for bone in OPENPOSE_RIGHT_HAND_BONES:
        colors[bone] = OPENPOSE_COLORS["right_arm"]

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
    elif format_name == "openpose_body25_hands":
        return get_bone_colors_openpose_body25_hands()
    else:
        # Default: single color for all bones
        default_color = (0.0, 1.0, 0.0)  # Green
        bones = get_skeleton_bones(format_name)
        return {bone: default_color for bone in bones}
