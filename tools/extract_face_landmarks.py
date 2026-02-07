#!/usr/bin/env python3
"""
Extract face landmarks from an image using MediaPipe Face Mesh.

Outputs a JSON file containing the OpenPose Face 70 keypoints mapped from
MediaPipe's 478-landmark face mesh. This file can be consumed by body2colmap
for subject-specific face rendering anchored to the skeleton.

Requirements:
    pip install mediapipe opencv-python

Usage:
    python extract_face_landmarks.py input_image.jpg -o landmarks.json
    python extract_face_landmarks.py input_image.jpg  # prints to stdout
"""

import argparse
import json
import sys
from pathlib import Path


# MediaPipe Face Mesh 478 -> OpenPose Face 68 index mapping (MaixPy convention)
MP_TO_OPENPOSE_FACE_68 = [
    # Jawline 0-16
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,
    # Right eyebrow 17-21
    71, 63, 105, 66, 107,
    # Left eyebrow 22-26
    336, 296, 334, 293, 301,
    # Nose bridge 27-30
    168, 197, 5, 4,
    # Nose bottom 31-35
    75, 97, 2, 326, 305,
    # Right eye 36-41
    33, 160, 158, 133, 153, 144,
    # Left eye 42-47
    362, 385, 387, 263, 373, 380,
    # Outer lip 48-59
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    # Inner lip 60-67
    78, 82, 13, 312, 308, 317, 14, 87,
]

# Iris center indices (refined model only, indices 468-477)
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473

# Eye contour indices for pupil fallback (when refined landmarks unavailable)
RIGHT_EYE_CONTOUR = [33, 160, 158, 133, 153, 144]
LEFT_EYE_CONTOUR = [362, 385, 387, 263, 373, 380]


def extract_landmarks(image_path: str, refine: bool = True) -> dict:
    """
    Extract face landmarks from an image.

    Args:
        image_path: Path to input image
        refine: Whether to use refined landmarks (478 with iris)

    Returns:
        Dictionary with landmark data ready for JSON serialization
    """
    try:
        import mediapipe as mp
        import cv2
    except ImportError:
        print(
            "Error: mediapipe and opencv-python are required.\n"
            "Install with: pip install mediapipe opencv-python",
            file=sys.stderr
        )
        sys.exit(1)

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image: {image_path}", file=sys.stderr)
        sys.exit(1)

    height, width = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run face mesh
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=refine,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        print("Error: No face detected in image.", file=sys.stderr)
        sys.exit(1)

    face = results.multi_face_landmarks[0]
    n_landmarks = len(face.landmark)

    # Extract all landmarks as (x, y, z)
    all_landmarks = [
        (lm.x, lm.y, lm.z) for lm in face.landmark
    ]

    # Map to OpenPose Face 68
    openpose_68 = [all_landmarks[i] for i in MP_TO_OPENPOSE_FACE_68]

    # Add pupils (68, 69)
    if refine and n_landmarks >= 478:
        # Use iris center landmarks directly
        right_pupil = all_landmarks[RIGHT_IRIS_CENTER]
        left_pupil = all_landmarks[LEFT_IRIS_CENTER]
    else:
        # Synthesize from eye contour centroids
        right_pupil = tuple(
            sum(all_landmarks[i][j] for i in RIGHT_EYE_CONTOUR) / 6
            for j in range(3)
        )
        left_pupil = tuple(
            sum(all_landmarks[i][j] for i in LEFT_EYE_CONTOUR) / 6
            for j in range(3)
        )

    openpose_70 = openpose_68 + [right_pupil, left_pupil]

    return {
        "version": "1.0",
        "source": "mediapipe",
        "source_image": str(Path(image_path).name),
        "image_size": [width, height],
        "n_mediapipe_landmarks": n_landmarks,
        "refined": refine and n_landmarks >= 478,
        "landmarks_openpose70": [
            [round(x, 6), round(y, 6), round(z, 6)]
            for x, y, z in openpose_70
        ]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract OpenPose Face 70 landmarks from an image using MediaPipe"
    )
    parser.add_argument(
        "image",
        help="Path to input image"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output JSON file path (default: stdout)"
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Disable landmark refinement (skip iris detection)"
    )

    args = parser.parse_args()

    result = extract_landmarks(args.image, refine=not args.no_refine)

    json_str = json.dumps(result, indent=2)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(json_str)
            f.write('\n')
        print(f"Landmarks saved to: {args.output}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == "__main__":
    main()
