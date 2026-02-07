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

LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
MODEL_CACHE_DIR = Path.home() / ".cache" / "body2colmap"
LANDMARKER_MODEL_PATH = MODEL_CACHE_DIR / "face_landmarker.task"
DETECTOR_MODEL_PATH = MODEL_CACHE_DIR / "blaze_face_short_range.tflite"


def _ensure_model(url: str, path: Path) -> Path:
    """Download a model file if not cached."""
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {path.name}...", file=sys.stderr)
    urllib.request.urlretrieve(url, str(path))
    print("Done.", file=sys.stderr)
    return path


def _find_face_bbox(rgb, detector, mp_module):
    """
    Find face bounding box using MediaPipe FaceDetector.

    Args:
        rgb: RGB image as numpy array (H, W, 3)
        detector: MediaPipe FaceDetector instance
        mp_module: mediapipe module (for mp.Image)

    Returns:
        (x, y, w, h) bounding box in pixel coordinates, or None
    """
    image = mp_module.Image(
        image_format=mp_module.ImageFormat.SRGB, data=rgb
    )
    result = detector.detect(image)

    if not result.detections:
        return None

    bbox = result.detections[0].bounding_box
    return (bbox.origin_x, bbox.origin_y, bbox.width, bbox.height)


def _crop_to_face(rgb, bbox, padding=0.5):
    """
    Crop image to face bounding box with padding.

    Args:
        rgb: Full RGB image (H, W, 3)
        bbox: (x, y, w, h) face bounding box from detector
        padding: Fraction of face size to add as padding on each side

    Returns:
        (crop, x1, y1) - cropped RGB array and top-left corner in full image
    """
    import numpy as np

    height, width = rgb.shape[:2]
    x, y, w, h = bbox

    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + w + pad_x)
    y2 = min(height, y + h + pad_y)

    crop = np.ascontiguousarray(rgb[y1:y2, x1:x2, :])
    return crop, x1, y1


def extract_landmarks(
    image_path: str,
    landmarker_model_path: str = None,
    detector_model_path: str = None,
    min_confidence: float = 0.3,
    save_crop: str = None,
) -> dict:
    """
    Extract raw face landmarks from an image using MediaPipe.

    Two-stage pipeline for robust detection at any face scale:
    1. Try FaceLandmarker on full image (fast path for selfies/headshots)
    2. If no face found, use FaceDetector to locate the face bounding box,
       crop tightly to it, and re-run FaceLandmarker on the crop
    3. Landmarks are mapped back to full-image normalized coordinates

    Args:
        image_path: Path to input image
        landmarker_model_path: Path to FaceLandmarker .task model
        detector_model_path: Path to FaceDetector .tflite model
        min_confidence: Minimum face detection confidence (0-1)
        save_crop: If set, save the detected face crop to this path

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

    import cv2
    import numpy as np

    # Ensure models are available
    if landmarker_model_path is None:
        landmarker_model_path = str(
            _ensure_model(LANDMARKER_MODEL_URL, LANDMARKER_MODEL_PATH)
        )
    if detector_model_path is None:
        detector_model_path = str(
            _ensure_model(DETECTOR_MODEL_URL, DETECTOR_MODEL_PATH)
        )

    # Load image
    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"Error: Could not read image: {image_path}", file=sys.stderr)
        sys.exit(1)
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    height, width = rgb.shape[:2]

    # Create FaceLandmarker
    lm_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=landmarker_model_path
        ),
        min_face_detection_confidence=min_confidence,
        min_face_presence_confidence=min_confidence,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(lm_options)

    try:
        # Stage 1: Try full image (fast path for selfies/headshots)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(image)

        if result.face_landmarks:
            face = result.face_landmarks[0]
            all_landmarks = [
                [round(lm.x, 6), round(lm.y, 6), round(lm.z, 6)]
                for lm in face
            ]
        else:
            # Stage 2: FaceDetector to locate bbox, then crop for landmarks
            det_options = vision.FaceDetectorOptions(
                base_options=python.BaseOptions(
                    model_asset_path=detector_model_path
                ),
                min_detection_confidence=min_confidence,
            )
            detector = vision.FaceDetector.create_from_options(det_options)

            try:
                bbox = _find_face_bbox(rgb, detector, mp)
            finally:
                detector.close()

            if bbox is None:
                print(
                    f"Error: No face detected in image ({width}x{height}).",
                    file=sys.stderr
                )
                sys.exit(1)

            # Crop to detected face with padding for landmark context
            crop, x1, y1 = _crop_to_face(rgb, bbox, padding=0.5)
            crop_h, crop_w = crop.shape[:2]
            print(
                f"Face detected at ({bbox[0]}, {bbox[1]}, "
                f"{bbox[2]}x{bbox[3]}), cropped to {crop_w}x{crop_h} for "
                f"landmark extraction.",
                file=sys.stderr
            )

            if save_crop:
                cv2.imwrite(save_crop, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                print(f"Crop saved to: {save_crop}", file=sys.stderr)

            crop_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=crop
            )
            result = landmarker.detect(crop_image)

            if not result.face_landmarks:
                print(
                    f"Error: FaceDetector located a face but FaceLandmarker "
                    f"could not extract landmarks from the crop "
                    f"({crop_w}x{crop_h}).",
                    file=sys.stderr
                )
                sys.exit(1)

            face = result.face_landmarks[0]

            # Map crop-normalized landmarks back to full-image normalized coords
            all_landmarks = [
                [round((lm.x * crop_w + x1) / width, 6),
                 round((lm.y * crop_h + y1) / height, 6),
                 round(lm.z, 6)]
                for lm in face
            ]

        n_landmarks = len(face)
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
        landmarker.close()


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
        "--landmarker-model",
        help="Path to FaceLandmarker .task model (auto-downloaded if not specified)"
    )
    parser.add_argument(
        "--detector-model",
        help="Path to FaceDetector .tflite model (auto-downloaded if not specified)"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.3,
        help="Minimum face detection confidence 0-1 (default: 0.3)"
    )
    parser.add_argument(
        "--save-crop",
        help="Save the detected face crop to this path (for verification)"
    )

    args = parser.parse_args()

    result = extract_landmarks(
        args.image,
        landmarker_model_path=args.landmarker_model,
        detector_model_path=args.detector_model,
        min_confidence=args.min_confidence,
        save_crop=args.save_crop,
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
