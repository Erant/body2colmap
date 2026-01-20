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
from typing import List, Tuple, Optional
from numpy.typing import NDArray


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


# MHR70 to OpenPose Body25+Hands mapping
# This is a simplified mapping - may need adjustment based on actual MHR70 format
# For now, we'll map the main body joints and skip detailed hand mapping
MHR70_TO_OPENPOSE_BODY25_HANDS = {
    # Body mapping (estimated - needs verification with actual MHR70 format)
    # This assumes MHR70 follows a similar structure to SMPL-X
    0: 15,   # Pelvis → MidHip (approximate)
    # TODO: Complete this mapping once we have MHR70 documentation
    # For now, we'll just pass through joints directly if counts match
}


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
        # For MHR70, we'd need the actual bone connectivity
        # For now, return empty list (skeleton won't render)
        # TODO: Add MHR70 bone connectivity once documented
        return []
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
