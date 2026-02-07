#!/usr/bin/env python3
"""
Extract face landmarks from an image using MediaPipe Face Mesh.

Outputs a JSON file containing raw MediaPipe landmarks (468 or 478 points).
The body2colmap library handles conversion to OpenPose Face 70 format via
FaceLandmarkIngest.

Requirements:
    pip install mediapipe opencv-python

Usage:
    python extract_face_landmarks.py input_image.jpg -o landmarks.json
    python extract_face_landmarks.py input_image.jpg  # prints to stdout

The output JSON can be passed to body2colmap via --face-landmarks:
    body2colmap input.npz -o output/ --face-landmarks landmarks.json
"""

import argparse
import json
import sys
from pathlib import Path


def extract_landmarks(image_path: str, refine: bool = True) -> dict:
    """
    Extract raw face landmarks from an image using MediaPipe Face Mesh.

    Args:
        image_path: Path to input image
        refine: Whether to use refined landmarks (478 with iris)

    Returns:
        Dictionary with raw MediaPipe landmark data ready for JSON serialization
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

    # Output all raw landmarks (468 or 478)
    all_landmarks = [
        [round(lm.x, 6), round(lm.y, 6), round(lm.z, 6)]
        for lm in face.landmark
    ]

    return {
        "version": "1.0",
        "source": "mediapipe",
        "source_image": str(Path(image_path).name),
        "image_size": [width, height],
        "n_landmarks": n_landmarks,
        "refined": refine and n_landmarks >= 478,
        "landmarks": all_landmarks
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract face landmarks from an image using MediaPipe Face Mesh"
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
