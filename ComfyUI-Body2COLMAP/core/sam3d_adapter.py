"""Adapter for converting SAM3D-Body output to body2colmap Scene."""

import numpy as np
import torch
from body2colmap.scene import Scene
from body2colmap.coordinates import sam3d_to_world


def sam3d_output_to_scene(mesh_data: dict, include_skeleton: bool = True) -> Scene:
    """
    Convert SAM3D_OUTPUT dict to body2colmap Scene.

    SAM3D uses MHR coordinates (Y-down, Z-forward).
    We convert to world coordinates (Y-up, Z-backward) for internal processing.

    Args:
        mesh_data: SAM3D_OUTPUT dict from ComfyUI-SAM3DBody containing:
            - vertices: (N, 3) mesh vertices
            - faces: (F, 3) face indices
            - camera: (3,) camera translation vector
            - joints: (127, 3) MHR joint positions (optional)
        include_skeleton: Whether to include skeleton from joints

    Returns:
        Scene object in world coordinates
    """
    # Extract and convert tensors to numpy
    vertices = _to_numpy(mesh_data["vertices"])
    faces = _to_numpy(mesh_data["faces"])
    cam_t = _to_numpy(mesh_data["camera"])

    # Convert from SAM3D/MHR coordinates to world coordinates
    # MHR: Y-down, Z-forward → World: Y-up, Z-backward
    vertices_world = sam3d_to_world(vertices, cam_t)

    # Extract skeleton if requested
    skeleton_joints = None
    skeleton_format = None
    if include_skeleton and "joints" in mesh_data and mesh_data["joints"] is not None:
        joints = _to_numpy(mesh_data["joints"])

        # SAM3D uses 127 MHR joints, convert to MHR70 (70 joints)
        skeleton_joints = _convert_mhr127_to_mhr70(joints)

        # Convert to world coordinates (use same camera translation)
        skeleton_joints = sam3d_to_world(skeleton_joints, cam_t)

        # Use MHR70 format
        skeleton_format = "mhr70"

    # Create Scene with vertices and faces directly
    return Scene(
        vertices=vertices_world,
        faces=faces,
        skeleton_joints=skeleton_joints,
        skeleton_format=skeleton_format
    )


def _to_numpy(tensor_or_array) -> np.ndarray:
    """Convert torch tensor or numpy array to numpy.

    Args:
        tensor_or_array: torch.Tensor or np.ndarray

    Returns:
        numpy array
    """
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.detach().cpu().numpy().astype(np.float32)
    return np.asarray(tensor_or_array, dtype=np.float32)


def _convert_mhr127_to_mhr70(joints_127: np.ndarray) -> np.ndarray:
    """Convert MHR 127 joints to MHR70 (70 joints).

    The SAM3D-Body output has 127 joints from the full MHR model.
    We need to extract the 70 joints used in MHR70 format.

    MHR70 joint ordering (from SAM3D-Body repository):
    - 0-22: Body joints (pelvis, spine, neck, head, shoulders, elbows, wrists, hips, knees, ankles)
    - 23-67: Hand joints (21 per hand: wrist + 4x5 finger joints)

    Args:
        joints_127: (127, 3) array of MHR joint positions

    Returns:
        (70, 3) array of MHR70 joint positions
    """
    # MHR127 to MHR70 mapping (joint indices to extract)
    # Based on SAM3D-Body's SMPL-X skeleton structure

    # Body joints (0-22): pelvis to ankles
    body_indices = list(range(0, 23))  # First 23 joints

    # Left hand joints (23-43): 21 joints starting from left wrist
    left_hand_start = 25  # Offset in MHR127 for left hand
    left_hand_indices = list(range(left_hand_start, left_hand_start + 21))

    # Right hand joints (44-64): 21 joints starting from right wrist
    right_hand_start = 46  # Offset in MHR127 for right hand
    right_hand_indices = list(range(right_hand_start, right_hand_start + 21))

    # Head joint (65-69): 5 head keypoints
    head_indices = [23, 24]  # Simplified: just take 2 head joints for now
    # Pad to 5 joints by repeating the head joint
    head_indices = head_indices + [24, 24, 24]

    # Combine all indices
    mhr70_indices = body_indices + left_hand_indices + right_hand_indices + head_indices

    # Extract the 70 joints
    if len(mhr70_indices) > len(joints_127):
        # If we don't have enough joints, pad with zeros
        mhr70 = np.zeros((70, 3), dtype=np.float32)
        available_joints = min(len(joints_127), len(mhr70_indices))
        mhr70[:available_joints] = joints_127[mhr70_indices[:available_joints]]
        return mhr70

    return joints_127[mhr70_indices].astype(np.float32)
