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


def _find_all_face_bboxes(rgb, detector, mp_module):
    """
    Find all face bounding boxes using MediaPipe FaceDetector.

    Args:
        rgb: RGB image as numpy array (H, W, 3)
        detector: MediaPipe FaceDetector instance
        mp_module: mediapipe module (for mp.Image)

    Returns:
        List of (x, y, w, h) bounding boxes in pixel coordinates
    """
    image = mp_module.Image(
        image_format=mp_module.ImageFormat.SRGB, data=rgb
    )
    result = detector.detect(image)

    bboxes = []
    for det in result.detections:
        bbox = det.bounding_box
        bboxes.append((bbox.origin_x, bbox.origin_y, bbox.width, bbox.height))
    return bboxes


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


# MediaPipe landmark indices for frontality scoring
_MP_RIGHT_EYE_OUTER = 33
_MP_LEFT_EYE_OUTER = 263
_MP_NOSE_BRIDGE = 168
_MP_CHIN = 152


def _frontality_score(face_landmarks):
    """
    Score how frontal a face is from its MediaPipe landmarks.

    Computes the face normal via cross product of the eye vector (horizontal)
    and nose-to-chin vector (vertical). The Z component of the normal
    indicates how much the face points toward the camera.

    Returns a value in [0, 1]: 1 = perfectly frontal, 0 = full profile.
    """
    import numpy as np

    def _xyz(idx):
        lm = face_landmarks[idx]
        return np.array([lm.x, lm.y, lm.z])

    eye_vec = _xyz(_MP_LEFT_EYE_OUTER) - _xyz(_MP_RIGHT_EYE_OUTER)  # left-to-right
    vert_vec = _xyz(_MP_NOSE_BRIDGE) - _xyz(_MP_CHIN)
    normal = np.cross(eye_vec, vert_vec)

    norm = np.linalg.norm(normal)
    if norm < 1e-10:
        return 0.0
    return abs(normal[2]) / norm


def _pick_best_face(face_landmarks_list):
    """
    Pick the most frontal face from a list of detected faces.

    Args:
        face_landmarks_list: List of MediaPipe face landmark lists

    Returns:
        (best_face, best_index) - the most frontal face and its index
    """
    if len(face_landmarks_list) == 1:
        return face_landmarks_list[0], 0

    best_score = -1.0
    best_idx = 0
    for i, face in enumerate(face_landmarks_list):
        score = _frontality_score(face)
        if score > best_score:
            best_score = score
            best_idx = i

    return face_landmarks_list[best_idx], best_idx


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

    # Create FaceLandmarker (allow multiple faces for best-face selection)
    max_faces = 10
    lm_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=landmarker_model_path
        ),
        min_face_detection_confidence=min_confidence,
        min_face_presence_confidence=min_confidence,
        num_faces=max_faces,
    )
    landmarker = vision.FaceLandmarker.create_from_options(lm_options)

    try:
        # Stage 1: Try full image (fast path for selfies/headshots)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(image)

        if result.face_landmarks:
            face, idx = _pick_best_face(result.face_landmarks)
            if len(result.face_landmarks) > 1:
                score = _frontality_score(face)
                print(
                    f"Found {len(result.face_landmarks)} face(s), "
                    f"selected #{idx + 1} (frontality: {score:.2f}).",
                    file=sys.stderr
                )
            all_landmarks = [
                [round(lm.x, 6), round(lm.y, 6), round(lm.z, 6)]
                for lm in face
            ]
        else:
            # Stage 2: FaceDetector to locate bboxes, crop each, pick best
            det_options = vision.FaceDetectorOptions(
                base_options=python.BaseOptions(
                    model_asset_path=detector_model_path
                ),
                min_detection_confidence=min_confidence,
            )
            detector = vision.FaceDetector.create_from_options(det_options)

            try:
                bboxes = _find_all_face_bboxes(rgb, detector, mp)
            finally:
                detector.close()

            if not bboxes:
                print(
                    f"Error: No face detected in image ({width}x{height}).",
                    file=sys.stderr
                )
                sys.exit(1)

            print(
                f"FaceDetector found {len(bboxes)} face(s).",
                file=sys.stderr
            )

            # Extract landmarks from each detected face crop
            candidates = []  # (face_landmarks, crop, x1, y1, bbox)
            for bbox in bboxes:
                crop, x1, y1 = _crop_to_face(rgb, bbox, padding=0.5)
                crop_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=crop
                )
                crop_result = landmarker.detect(crop_image)
                if crop_result.face_landmarks:
                    candidates.append(
                        (crop_result.face_landmarks[0], crop, x1, y1, bbox)
                    )

            if not candidates:
                print(
                    f"Error: FaceDetector located {len(bboxes)} face(s) but "
                    f"FaceLandmarker could not extract landmarks from any.",
                    file=sys.stderr
                )
                sys.exit(1)

            # Pick the most frontal face
            faces_only = [c[0] for c in candidates]
            _, best_idx = _pick_best_face(faces_only)
            face, crop, x1, y1, bbox = candidates[best_idx]
            crop_h, crop_w = crop.shape[:2]

            if len(candidates) > 1:
                score = _frontality_score(face)
                print(
                    f"Selected face #{best_idx + 1} of {len(candidates)} "
                    f"(frontality: {score:.2f}).",
                    file=sys.stderr
                )

            print(
                f"Face at ({bbox[0]}, {bbox[1]}, {bbox[2]}x{bbox[3]}), "
                f"cropped to {crop_w}x{crop_h} for landmark extraction.",
                file=sys.stderr
            )

            if save_crop:
                cv2.imwrite(save_crop, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                print(f"Crop saved to: {save_crop}", file=sys.stderr)

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
