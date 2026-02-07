#!/usr/bin/env python3
"""
Extract face landmarks from an image using MediaPipe FaceLandmarker.

Outputs a JSON file containing raw MediaPipe landmarks (478 points).
The body2colmap library handles conversion to OpenPose Face 70 format via
FaceLandmarkIngest.

Requirements:
    pip install mediapipe opencv-python

Usage:
    python extract_face_landmarks.py input_image.jpg -o landmarks.json
    python extract_face_landmarks.py input_image.jpg  # prints to stdout

The output JSON can be passed to body2colmap via --face-landmarks:
    body2colmap input.npz -o output/ --face-landmarks landmarks.json

On first run, the FaceLandmarker model (~4MB) is downloaded automatically
to ~/.cache/body2colmap/face_landmarker.task.
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_CACHE_DIR = Path.home() / ".cache" / "body2colmap"
MODEL_PATH = MODEL_CACHE_DIR / "face_landmarker.task"


def ensure_model(model_path: Path = MODEL_PATH) -> Path:
    """Download the FaceLandmarker model if not cached."""
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading FaceLandmarker model to {model_path}...", file=sys.stderr)
    urllib.request.urlretrieve(MODEL_URL, str(model_path))
    print("Done.", file=sys.stderr)
    return model_path


def extract_landmarks(
    image_path: str,
    model_path: str = None,
    min_confidence: float = 0.3
) -> dict:
    """
    Extract raw face landmarks from an image using MediaPipe FaceLandmarker.

    Args:
        image_path: Path to input image
        model_path: Path to .task model file (auto-downloaded if None)
        min_confidence: Minimum face detection confidence (0-1)

    Returns:
        Dictionary with raw MediaPipe landmark data ready for JSON serialization
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError:
        print(
            "Error: mediapipe is required.\n"
            "Install with: pip install mediapipe",
            file=sys.stderr
        )
        sys.exit(1)

    # Ensure model is available
    if model_path is None:
        model_path = str(ensure_model())

    # Load image via OpenCV for reliable format handling, then wrap for MediaPipe
    try:
        import cv2
        bgr = cv2.imread(image_path)
        if bgr is None:
            print(f"Error: Could not read image: {image_path}", file=sys.stderr)
            sys.exit(1)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    except ImportError:
        # Fall back to MediaPipe's own loader if cv2 not available
        image = mp.Image.create_from_file(image_path)
        height = image.height
        width = image.width

    # Create FaceLandmarker
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        min_face_detection_confidence=min_confidence,
        min_face_presence_confidence=min_confidence,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    try:
        # Detect landmarks
        result = detector.detect(image)

        if not result.face_landmarks:
            print(
                f"Error: No face detected in image ({width}x{height}). "
                f"Try --min-confidence with a lower value (current: {min_confidence}).",
                file=sys.stderr
            )
            sys.exit(1)

        face = result.face_landmarks[0]
        n_landmarks = len(face)

        # Output all raw landmarks (478 with iris)
        all_landmarks = [
            [round(lm.x, 6), round(lm.y, 6), round(lm.z, 6)]
            for lm in face
        ]

        return {
            "version": "1.0",
            "source": "mediapipe",
            "source_image": str(Path(image_path).name),
            "image_size": [width, height],
            "n_landmarks": n_landmarks,
            "refined": n_landmarks >= 478,
            "landmarks": all_landmarks
        }
    finally:
        # Explicit close avoids __del__ crash during interpreter shutdown
        detector.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract face landmarks from an image using MediaPipe FaceLandmarker"
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
        "--model-path",
        help="Path to FaceLandmarker .task model (auto-downloaded if not specified)"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.3,
        help="Minimum face detection confidence 0-1 (default: 0.3)"
    )

    args = parser.parse_args()

    result = extract_landmarks(
        args.image,
        model_path=args.model_path,
        min_confidence=args.min_confidence
    )

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
