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


def _detect_with_crop(detector, rgb, crop_fraction, mp_module):
    """
    Try face detection on a crop of the upper portion of the image.

    The MediaPipe FaceLandmarker uses BlazeFace (short-range), which internally
    resizes to 128x128/256x256. For full-body images, the face may be too small
    after this downscaling. Cropping to the upper portion makes the face larger
    relative to the input.

    Args:
        detector: FaceLandmarker instance
        rgb: Full RGB image as numpy array (H, W, 3)
        crop_fraction: Fraction of image height to keep from the top (0-1)
        mp_module: mediapipe module (for mp.Image)

    Returns:
        (face_landmarks, crop_fraction) if detected, (None, crop_fraction) otherwise
    """
    import numpy as np

    full_h, full_w = rgb.shape[:2]

    if crop_fraction >= 1.0:
        # Use full image
        crop = rgb
    else:
        crop_h = int(full_h * crop_fraction)
        crop = np.ascontiguousarray(rgb[:crop_h, :, :])

    image = mp_module.Image(
        image_format=mp_module.ImageFormat.SRGB, data=crop
    )
    result = detector.detect(image)

    if result.face_landmarks:
        return result.face_landmarks[0], crop_fraction
    return None, crop_fraction


def extract_landmarks(
    image_path: str,
    model_path: str = None,
    min_confidence: float = 0.3
) -> dict:
    """
    Extract raw face landmarks from an image using MediaPipe FaceLandmarker.

    Uses progressive cropping: if no face is found in the full image, retries
    with the upper 1/2, then upper 1/3 of the image. This handles full-body
    photos where the face is small relative to the image.

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

    # Load image via OpenCV for reliable format handling
    try:
        import cv2
        bgr = cv2.imread(image_path)
        if bgr is None:
            print(f"Error: Could not read image: {image_path}", file=sys.stderr)
            sys.exit(1)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except ImportError:
        # Fall back to MediaPipe's own loader if cv2 not available
        tmp_image = mp.Image.create_from_file(image_path)
        rgb = tmp_image.numpy_view().copy()

    import numpy as np
    rgb = np.ascontiguousarray(rgb)
    height, width = rgb.shape[:2]

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
        # Progressive cropping: try full image, then upper 1/2, then upper 1/3.
        # BlazeFace (short-range) resizes internally to 128-256px, so a face
        # that's <15% of image height may be undetectable without cropping.
        crop_fractions = [1.0, 0.5, 1.0 / 3.0]
        face = None
        used_fraction = 1.0

        for frac in crop_fractions:
            face, used_fraction = _detect_with_crop(detector, rgb, frac, mp)
            if face is not None:
                if frac < 1.0:
                    print(
                        f"Face detected in upper {frac:.0%} crop of image.",
                        file=sys.stderr
                    )
                break

        if face is None:
            print(
                f"Error: No face detected in image ({width}x{height}), "
                f"even after cropping to upper 1/3. "
                f"Try --min-confidence with a lower value (current: {min_confidence}).",
                file=sys.stderr
            )
            sys.exit(1)

        n_landmarks = len(face)

        # Convert landmarks back to full-image normalized coordinates.
        # MediaPipe landmarks are normalized 0-1 within the input crop.
        # x stays the same (width unchanged), y is scaled by crop fraction.
        all_landmarks = [
            [round(lm.x, 6),
             round(lm.y * used_fraction, 6),
             round(lm.z, 6)]
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print diagnostic info (image shape, model path, crop attempts)"
    )

    args = parser.parse_args()

    if args.debug:
        import os
        model_p = args.model_path or str(MODEL_PATH)
        print(f"Model path: {model_p}", file=sys.stderr)
        if os.path.exists(model_p):
            print(f"Model size: {os.path.getsize(model_p)} bytes", file=sys.stderr)
        else:
            print("Model file not yet downloaded.", file=sys.stderr)

        try:
            import cv2
            bgr = cv2.imread(args.image)
            if bgr is not None:
                print(f"Image shape: {bgr.shape}, dtype: {bgr.dtype}", file=sys.stderr)
            else:
                print(f"cv2.imread returned None for: {args.image}", file=sys.stderr)
        except ImportError:
            print("cv2 not available for debug info.", file=sys.stderr)

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
