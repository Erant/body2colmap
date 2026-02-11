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

# OpenPose Body25 color palette (based on poseParametersRender.hpp with correction)
# RGB values in 0-1 range
# Note: Official OpenPose has index 8 as red (255,0,0) which duplicates index 1.
# This creates two red bones (shoulder and thigh). We fix this by continuing
# the rainbow gradient: index 8 should be cyan-green to bridge green→cyan.
# Source: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/pose/poseParametersRender.hpp
OPENPOSE_BODY25_COLORS = [
    (255/255, 0/255, 85/255),    # 0 - Pink-red
    (255/255, 0/255, 0/255),     # 1 - Red
    (255/255, 85/255, 0/255),    # 2 - Orange
    (255/255, 170/255, 0/255),   # 3 - Orange-yellow
    (255/255, 255/255, 0/255),   # 4 - Yellow
    (170/255, 255/255, 0/255),   # 5 - Yellow-green
    (85/255, 255/255, 0/255),    # 6 - Green-yellow
    (0/255, 255/255, 0/255),     # 7 - Green
    (0/255, 255/255, 85/255),    # 8 - Cyan-green (CORRECTED from red)
    (0/255, 255/255, 85/255),    # 9 - Cyan-green
    (0/255, 255/255, 170/255),   # 10 - Cyan
    (0/255, 255/255, 255/255),   # 11 - Cyan
    (0/255, 170/255, 255/255),   # 12 - Cyan-blue
    (0/255, 85/255, 255/255),    # 13 - Blue
    (0/255, 0/255, 255/255),     # 14 - Blue
    (255/255, 0/255, 170/255),   # 15 - Magenta-pink
    (170/255, 0/255, 255/255),   # 16 - Purple
    (255/255, 0/255, 255/255),   # 17 - Magenta
    (85/255, 0/255, 255/255),    # 18 - Purple-blue
    (0/255, 0/255, 255/255),     # 19 - Blue
    (0/255, 0/255, 255/255),     # 20 - Blue
    (0/255, 255/255, 255/255),   # 21 - Cyan
    (0/255, 255/255, 255/255),   # 22 - Cyan
    (0/255, 255/255, 255/255),   # 23 - Cyan
    (0/255, 255/255, 255/255),   # 24 - Cyan
]

# OpenPose Body25 bone-to-color mapping
# Each bone (joint pair) is assigned a specific color index from the palette above
# Format: (joint1, joint2): color_index
OPENPOSE_BODY25_BONE_COLORS = {
    (1, 8): 0,    # Neck → MidHip
    (1, 2): 1,    # Neck → RShoulder
    (1, 5): 2,    # Neck → LShoulder
    (2, 3): 3,    # RShoulder → RElbow
    (3, 4): 4,    # RElbow → RWrist
    (5, 6): 5,    # LShoulder → LElbow
    (6, 7): 6,    # LElbow → LWrist
    (8, 9): 7,    # MidHip → RHip
    (9, 10): 8,   # RHip → RKnee
    (10, 11): 9,  # RKnee → RAnkle
    (8, 12): 10,  # MidHip → LHip
    (12, 13): 11, # LHip → LKnee
    (13, 14): 12, # LKnee → LAnkle
    (1, 0): 13,   # Neck → Nose
    (0, 15): 14,  # Nose → REye
    (15, 17): 15, # REye → REar
    (0, 16): 16,  # Nose → LEye
    (16, 18): 17, # LEye → LEar
    (14, 19): 18, # LAnkle → LBigToe
    (19, 20): 19, # LBigToe → LSmallToe
    (14, 21): 20, # LAnkle → LHeel
    (11, 22): 21, # RAnkle → RBigToe
    (22, 23): 22, # RBigToe → RSmallToe
    (11, 24): 23, # RAnkle → RHeel
}

# Hand finger colors (distinct colors for each finger for visibility)
# Format: (R, G, B) in 0-1 range
OPENPOSE_HAND_FINGER_COLORS = {
    "thumb": (1.0, 0.0, 0.4),      # Pink/Red
    "index": (1.0, 0.6, 0.0),      # Orange
    "middle": (0.0, 1.0, 0.0),     # Green
    "ring": (0.0, 0.6, 1.0),       # Cyan/Blue
    "pinky": (0.6, 0.0, 1.0),      # Purple/Magenta
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
    Get official OpenPose Body25+Hands colors for each bone.

    Returns a dictionary mapping bone (start_idx, end_idx) to RGB color (0-1 range).
    Uses the official OpenPose Body25 color palette from poseParametersRender.hpp.
    Handles bidirectional bone lookup (checks both (A,B) and (B,A)).

    Returns:
        Dictionary mapping bones to colors
    """
    colors = {}

    # Helper function for bidirectional lookup
    def get_body25_color(bone: Tuple[int, int]) -> Optional[Tuple[float, float, float]]:
        """Try to find color for bone, checking both directions."""
        if bone in OPENPOSE_BODY25_BONE_COLORS:
            return OPENPOSE_BODY25_COLORS[OPENPOSE_BODY25_BONE_COLORS[bone]]
        # Try reversed
        reversed_bone = (bone[1], bone[0])
        if reversed_bone in OPENPOSE_BODY25_BONE_COLORS:
            return OPENPOSE_BODY25_COLORS[OPENPOSE_BODY25_BONE_COLORS[reversed_bone]]
        return None

    # Body bones (0-24): Use official Body25 colors with bidirectional lookup
    for bone in OPENPOSE_BODY25_HANDS_BONES:
        color = get_body25_color(bone)
        if color is not None:
            colors[bone] = color

    # Right hand bones: Per-finger colors
    # Thumb (4 bones): joints 45-48
    for bone in [(4, 45), (45, 46), (46, 47), (47, 48)]:
        colors[bone] = OPENPOSE_HAND_FINGER_COLORS["thumb"]
    # Index (4 bones): joints 49-52
    for bone in [(4, 49), (49, 50), (50, 51), (51, 52)]:
        colors[bone] = OPENPOSE_HAND_FINGER_COLORS["index"]
    # Middle (4 bones): joints 53-56
    for bone in [(4, 53), (53, 54), (54, 55), (55, 56)]:
        colors[bone] = OPENPOSE_HAND_FINGER_COLORS["middle"]
    # Ring (4 bones): joints 57-60
    for bone in [(4, 57), (57, 58), (58, 59), (59, 60)]:
        colors[bone] = OPENPOSE_HAND_FINGER_COLORS["ring"]
    # Pinky (4 bones): joints 61-64
    for bone in [(4, 61), (61, 62), (62, 63), (63, 64)]:
        colors[bone] = OPENPOSE_HAND_FINGER_COLORS["pinky"]

    # Left hand bones: Per-finger colors
    # Thumb (4 bones): joints 25-28
    for bone in [(7, 25), (25, 26), (26, 27), (27, 28)]:
        colors[bone] = OPENPOSE_HAND_FINGER_COLORS["thumb"]
    # Index (4 bones): joints 29-32
    for bone in [(7, 29), (29, 30), (30, 31), (31, 32)]:
        colors[bone] = OPENPOSE_HAND_FINGER_COLORS["index"]
    # Middle (4 bones): joints 33-36
    for bone in [(7, 33), (33, 34), (34, 35), (35, 36)]:
        colors[bone] = OPENPOSE_HAND_FINGER_COLORS["middle"]
    # Ring (4 bones): joints 37-40
    for bone in [(7, 37), (37, 38), (38, 39), (39, 40)]:
        colors[bone] = OPENPOSE_HAND_FINGER_COLORS["ring"]
    # Pinky (4 bones): joints 41-44
    for bone in [(7, 41), (41, 42), (42, 43), (43, 44)]:
        colors[bone] = OPENPOSE_HAND_FINGER_COLORS["pinky"]

    return colors


def get_joint_colors_from_bones(
    bone_colors: Dict[Tuple[int, int], Tuple[float, float, float]],
    num_joints: int
) -> List[Tuple[float, float, float]]:
    """
    Compute joint colors based on connected bone colors.

    For each joint, use the color of one of its connected bones.
    If multiple bones connect to a joint, use the first one found.

    Args:
        bone_colors: Dictionary mapping (start_idx, end_idx) to RGB color
        num_joints: Total number of joints in skeleton

    Returns:
        List of RGB colors (0-1 range), one per joint
    """
    joint_colors = [(1.0, 1.0, 1.0)] * num_joints  # Default white

    for (start_idx, end_idx), color in bone_colors.items():
        # Assign this bone's color to both joints if not already assigned
        if joint_colors[start_idx] == (1.0, 1.0, 1.0):
            joint_colors[start_idx] = color
        if joint_colors[end_idx] == (1.0, 1.0, 1.0):
            joint_colors[end_idx] = color

    return joint_colors


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


# =============================================================================
# Framing Presets - Y-coordinate thresholding for partial body framing
# =============================================================================

# Valid framing preset names
FRAMING_PRESETS = ["full", "torso", "bust", "head"]


def get_framing_y_threshold(joints: NDArray[np.float32], preset: str) -> Optional[float]:
    """
    Get Y-coordinate threshold for a framing preset.

    Uses skeleton joint positions to determine height thresholds for filtering
    mesh vertices. For partial body framing (torso, bust, head), vertices with
    Y >= threshold are included in the framing bounding box.

    Args:
        joints: MHR70 skeleton joints, shape (70, 3)
        preset: Framing preset name:
            - "full": No filtering (returns None)
            - "torso": Waist up (threshold at hip level)
            - "bust": Shoulders and head (threshold at upper chest)
            - "head": Head only (threshold at neck level)

    Returns:
        Y threshold value (include vertices where Y >= threshold)
        None for "full" preset (no filtering needed)

    Raises:
        ValueError: If preset is unknown
    """
    if preset == "full":
        return None

    if preset not in FRAMING_PRESETS:
        raise ValueError(
            f"Unknown framing preset: '{preset}'. "
            f"Valid options: {', '.join(FRAMING_PRESETS)}"
        )

    # Extract key Y coordinates from MHR70 joints
    # Using indices from MHR70_JOINTS dict
    left_hip_y = joints[9, 1]    # left_hip
    right_hip_y = joints[10, 1]  # right_hip
    left_shoulder_y = joints[5, 1]   # left_shoulder
    right_shoulder_y = joints[6, 1]  # right_shoulder
    neck_y = joints[69, 1]       # neck

    hip_y = (left_hip_y + right_hip_y) / 2.0
    shoulder_y = (left_shoulder_y + right_shoulder_y) / 2.0
    torso_height = shoulder_y - hip_y

    if preset == "torso":
        # Waist up: cut at hip level
        return float(hip_y)

    elif preset == "bust":
        # Shoulders + head: cut below shoulders to include upper chest
        # Use 25% below shoulder line (into upper chest area)
        return float(shoulder_y - 0.25 * torso_height)

    elif preset == "head":
        # Head only: cut at neck
        return float(neck_y)

    # Should not reach here due to earlier validation
    return None


# =============================================================================
# Camera Height Presets - Skeleton-based orbit center height
# =============================================================================

# Valid camera height preset names
CAMERA_HEIGHT_PRESETS = [
    "bbox_center", "feet", "knees", "waist", "chest", "shoulders", "head"
]


def get_camera_height_y(joints: NDArray[np.float32], preset: str) -> float:
    """
    Get Y-coordinate for camera orbit center height based on skeleton joints.

    Uses skeleton joint positions to determine the height at which the camera
    orbits. This sets the Y component of the orbit target point.

    Args:
        joints: MHR70 skeleton joints, shape (70, 3)
        preset: Camera height preset name:
            - "feet": Ankle height (average of left/right ankle Y)
            - "knees": Knee height (average of left/right knee Y)
            - "waist": Hip height (average of left/right hip Y)
            - "chest": Midpoint between hips and shoulders
            - "shoulders": Shoulder height (average of left/right shoulder Y)
            - "head": Nose height

    Returns:
        Y coordinate for the camera orbit center

    Raises:
        ValueError: If preset is unknown or "bbox_center" (handled elsewhere)
    """
    if preset == "bbox_center":
        raise ValueError(
            "bbox_center is not skeleton-based; handle before calling this function"
        )

    if preset not in CAMERA_HEIGHT_PRESETS:
        raise ValueError(
            f"Unknown camera height preset: '{preset}'. "
            f"Valid options: {', '.join(CAMERA_HEIGHT_PRESETS)}"
        )

    # Extract Y coordinates from MHR70 joints
    left_ankle_y = joints[13, 1]
    right_ankle_y = joints[14, 1]
    left_knee_y = joints[11, 1]
    right_knee_y = joints[12, 1]
    left_hip_y = joints[9, 1]
    right_hip_y = joints[10, 1]
    left_shoulder_y = joints[5, 1]
    right_shoulder_y = joints[6, 1]
    nose_y = joints[0, 1]

    if preset == "feet":
        return float((left_ankle_y + right_ankle_y) / 2.0)

    elif preset == "knees":
        return float((left_knee_y + right_knee_y) / 2.0)

    elif preset == "waist":
        return float((left_hip_y + right_hip_y) / 2.0)

    elif preset == "chest":
        hip_y = (left_hip_y + right_hip_y) / 2.0
        shoulder_y = (left_shoulder_y + right_shoulder_y) / 2.0
        return float((hip_y + shoulder_y) / 2.0)

    elif preset == "shoulders":
        return float((left_shoulder_y + right_shoulder_y) / 2.0)

    elif preset == "head":
        return float(nose_y)

    # Should not reach here due to earlier validation
    raise ValueError(f"Unhandled camera height preset: '{preset}'")
