"""
Step 56: Face detection per frame using MediaPipe (primary) / RetinaFace (fallback).

For each extracted frame:
  1. Detect all faces + bounding boxes + landmarks
  2. Align face to a canonical 112×112 crop (eye-aligned)
  3. Return FaceRegion objects for artifact analysis

Why face alignment matters:
  Deepfake artifacts concentrate around face boundaries, eyes, mouth,
  and hair-skin transitions. Aligned crops ensure the CNN sees the
  same anatomy in the same position every time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog

from .frame_extractor import VideoFrame

log = structlog.get_logger(__name__)

FACE_CROP_SIZE = 224   # aligned face crop side length


@dataclass
class FaceRegion:
    """Detected and aligned face region from one video frame."""
    frame_idx:    int
    timestamp_s:  float
    bbox:         tuple[int, int, int, int]   # (x1, y1, x2, y2)
    landmarks:    np.ndarray | None           # shape [5, 2] (x, y) — 5 keypoints
    crop:         np.ndarray                  # shape [FACE_CROP_SIZE, FACE_CROP_SIZE, 3] BGR
    confidence:   float                       # detection confidence
    face_idx:     int                         # index when multiple faces present


@dataclass
class FrameFaces:
    """All detected faces in one frame."""
    frame_idx:   int
    timestamp_s: float
    faces:       list[FaceRegion]
    n_faces:     int
    detection_method: str


# ── MediaPipe face detector ────────────────────────────────────

class MediaPipeFaceDetector:
    """
    Primary face detector. Uses MediaPipe BlazeFace model.
    Very fast (~2ms per frame on CPU), good recall.
    """

    def __init__(self, min_confidence: float = 0.5) -> None:
        self._min_conf = min_confidence
        self._detector = None

    def load(self) -> None:
        try:
            import mediapipe as mp  # type: ignore
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,             # 1 = full-range model (up to 5m)
                min_detection_confidence=self._min_conf,
            )
            log.info("mediapipe_face_detector_loaded")
        except ImportError:
            log.warning("mediapipe_not_installed — will use fallback")
            self._detector = None

    def detect(self, frame: VideoFrame) -> FrameFaces:
        if self._detector is None:
            return _empty_frame_faces(frame)
        try:
            import mediapipe as mp  # type: ignore

            image_rgb = frame.image[:, :, ::-1]  # BGR → RGB
            results   = self._detector.process(image_rgb)

            if not results.detections:
                return _empty_frame_faces(frame)

            faces: list[FaceRegion] = []
            h, w = frame.image.shape[:2]

            for i, det in enumerate(results.detections):
                bbox_mp = det.location_data.relative_bounding_box
                x1 = int(bbox_mp.xmin * w)
                y1 = int(bbox_mp.ymin * h)
                x2 = int((bbox_mp.xmin + bbox_mp.width) * w)
                y2 = int((bbox_mp.ymin + bbox_mp.height) * h)

                # Clamp to frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue   # too small

                crop = _extract_aligned_crop(frame.image, x1, y1, x2, y2)
                conf = det.score[0] if det.score else 0.0

                # Extract 5 facial landmarks if available
                landmarks = None
                kps = det.location_data.relative_keypoints
                if kps:
                    landmarks = np.array(
                        [[kp.x * w, kp.y * h] for kp in kps[:5]],
                        dtype=np.float32,
                    )

                faces.append(FaceRegion(
                    frame_idx=frame.frame_idx,
                    timestamp_s=frame.timestamp_s,
                    bbox=(x1, y1, x2, y2),
                    landmarks=landmarks,
                    crop=crop,
                    confidence=float(conf),
                    face_idx=i,
                ))

            return FrameFaces(
                frame_idx=frame.frame_idx,
                timestamp_s=frame.timestamp_s,
                faces=faces,
                n_faces=len(faces),
                detection_method="mediapipe",
            )

        except Exception as exc:
            log.warning("mediapipe_detection_error",
                        frame=frame.frame_idx, error=str(exc))
            return _empty_frame_faces(frame)


# ── RetinaFace detector (fallback — higher accuracy, slower) ──

class RetinaFaceDetector:
    """
    Fallback face detector. RetinaFace is more accurate than MediaPipe
    for small or partially occluded faces, but ~5× slower.
    Used when MediaPipe returns zero detections.
    """

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        try:
            from retinaface import RetinaFace  # type: ignore
            self._model = RetinaFace
            log.info("retinaface_loaded")
        except ImportError:
            log.warning("retinaface_not_installed")

    def detect(self, frame: VideoFrame) -> FrameFaces:
        if self._model is None:
            return _empty_frame_faces(frame)
        try:
            detections = self._model.detect_faces(frame.image)
            if not detections or not isinstance(detections, dict):
                return _empty_frame_faces(frame)

            faces: list[FaceRegion] = []
            for i, (key, face_data) in enumerate(detections.items()):
                bbox  = face_data["facial_area"]   # [x1, y1, x2, y2]
                score = face_data.get("score", 0.5)
                x1, y1, x2, y2 = bbox
                crop = _extract_aligned_crop(frame.image, x1, y1, x2, y2)

                lm = face_data.get("landmarks")
                landmarks = None
                if lm:
                    landmarks = np.array(list(lm.values()), dtype=np.float32)

                faces.append(FaceRegion(
                    frame_idx=frame.frame_idx,
                    timestamp_s=frame.timestamp_s,
                    bbox=(x1, y1, x2, y2),
                    landmarks=landmarks,
                    crop=crop,
                    confidence=float(score),
                    face_idx=i,
                ))

            return FrameFaces(
                frame_idx=frame.frame_idx,
                timestamp_s=frame.timestamp_s,
                faces=faces,
                n_faces=len(faces),
                detection_method="retinaface",
            )
        except Exception as exc:
            log.warning("retinaface_error", frame=frame.frame_idx, error=str(exc))
            return _empty_frame_faces(frame)


# ── Hybrid detector ────────────────────────────────────────────

class FaceDetector:
    """
    Hybrid detector: MediaPipe first, RetinaFace fallback when no faces found.
    """

    def __init__(self) -> None:
        self._mp  = MediaPipeFaceDetector()
        self._rf  = RetinaFaceDetector()

    def load(self) -> None:
        self._mp.load()
        self._rf.load()

    def detect(self, frame: VideoFrame) -> FrameFaces:
        result = self._mp.detect(frame)
        if result.n_faces == 0:
            result = self._rf.detect(frame)
        return result

    def detect_batch(self, frames: list[VideoFrame]) -> list[FrameFaces]:
        return [self.detect(f) for f in frames]


# ── Helpers ────────────────────────────────────────────────────

def _extract_aligned_crop(
    image: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    margin: float = 0.20,
) -> np.ndarray:
    """
    Crop a face region with a margin and resize to FACE_CROP_SIZE.
    Margin adds context around the face bbox — important for catching
    blending artifacts at the face/background boundary.
    """
    h, w = image.shape[:2]
    bw = x2 - x1
    bh = y2 - y1

    # Add margin
    mx = int(bw * margin)
    my = int(bh * margin)
    cx1 = max(0, x1 - mx)
    cy1 = max(0, y1 - my)
    cx2 = min(w, x2 + mx)
    cy2 = min(h, y2 + my)

    crop = image[cy1:cy2, cx1:cx2]

    if crop.size == 0:
        return np.zeros((FACE_CROP_SIZE, FACE_CROP_SIZE, 3), dtype=np.uint8)

    try:
        import cv2  # type: ignore
        return cv2.resize(crop, (FACE_CROP_SIZE, FACE_CROP_SIZE))
    except Exception:
        # Pure numpy fallback
        from PIL import Image as PILImage  # type: ignore
        pil = PILImage.fromarray(crop[:, :, ::-1])  # BGR → RGB
        pil = pil.resize((FACE_CROP_SIZE, FACE_CROP_SIZE))
        return np.array(pil)[:, :, ::-1]    # RGB → BGR


def _empty_frame_faces(frame: VideoFrame) -> FrameFaces:
    return FrameFaces(
        frame_idx=frame.frame_idx,
        timestamp_s=frame.timestamp_s,
        faces=[],
        n_faces=0,
        detection_method="none",
    )
