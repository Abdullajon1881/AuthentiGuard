"""
Step 57: Artifact analysis per face region.

Three categories of artifacts that deepfakes consistently exhibit:

1. Blinking anomalies
   - Unnatural blink frequency (too regular, too rare)
   - Incomplete blinks (eyelids don't fully close)
   - Missing or asymmetric blink events
   Reason: early deepfakes were trained on still images → no eye movement.
   Modern ones use eye-tracking but still show subtle timing irregularities.

2. Skin texture analysis
   - Loss of fine-grained skin texture (pores, fine lines)
   - Over-smoothing / "plastic" appearance
   - Unnatural colour consistency across face regions
   - High-frequency texture irregularities at blend boundaries

3. Lighting inconsistency
   - Face lighting doesn't match background / scene lighting direction
   - Abnormal specular highlights (wrong position, shape)
   - Inconsistent colour temperature between face and scene
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import structlog

from .face_detector import FaceRegion, FrameFaces

log = structlog.get_logger(__name__)


@dataclass
class ArtifactAnalysis:
    """Per-frame artifact scores and evidence."""
    frame_idx:    int
    timestamp_s:  float

    # Blinking
    eye_openness:         float | None   # 0.0 = fully closed, 1.0 = fully open
    blink_detected:       bool

    # Skin texture
    texture_score:        float   # [0,1] — higher = more artificial texture
    laplacian_variance:   float   # low variance → over-smoothed (deepfake signal)
    lbp_uniformity:       float   # high uniformity → synthetic skin texture

    # Lighting
    lighting_score:       float   # [0,1] — higher = more inconsistent lighting
    illumination_std:     float   # std of illumination gradient direction
    specular_anomaly:     float   # specular highlight position anomaly

    # Combined
    artifact_score:       float   # weighted aggregate of all signals


# ── Blinking analysis ─────────────────────────────────────────

def analyze_blink(face: FaceRegion) -> dict:
    """
    Estimate eye openness from the face crop using pixel intensity
    in the eye region. More accurate with landmarks but degrades
    gracefully when landmarks are missing.
    """
    crop = face.crop   # [224, 224, 3]
    h, w = crop.shape[:2]

    try:
        if face.landmarks is not None and len(face.landmarks) >= 2:
            return _blink_from_landmarks(face, crop)
        return _blink_from_crop(crop, h, w)
    except Exception as exc:
        log.debug("blink_analysis_failed", error=str(exc))
        return {"eye_openness": None, "blink_detected": False}


def _blink_from_landmarks(face: FaceRegion, crop: np.ndarray) -> dict:
    """Use eye landmark positions to compute Eye Aspect Ratio (EAR)."""
    # Landmarks order: right_eye, left_eye, nose, right_mouth, left_mouth
    lm = face.landmarks
    if len(lm) < 2:
        return _blink_from_crop(crop, *crop.shape[:2])

    # Approximate EAR from bounding box — proper EAR needs 6 eye keypoints
    # (as in dlib's 68-point model), but MediaPipe gives us 5 total.
    # We use the y-distance between the eyes as a proxy.
    right_eye = lm[0]
    left_eye  = lm[1]

    # Vertical extent of face vs distance between eyes
    face_height = face.bbox[3] - face.bbox[1]
    eye_height  = abs(right_eye[1] - left_eye[1])

    # Normalize: eyes at roughly same height = open
    openness = 1.0 - min(eye_height / max(face_height * 0.2, 1.0), 1.0)

    return {
        "eye_openness": round(float(openness), 4),
        "blink_detected": openness < 0.25,
    }


def _blink_from_crop(crop: np.ndarray, h: int, w: int) -> dict:
    """
    Fallback: detect eye region by brightness in the upper-middle zone.
    Closed eyes show higher average intensity in the lid region.
    """
    gray = crop.mean(axis=2)   # rough grayscale
    # Eye strip: rows 25–45% of height, cols 15–85% of width
    eye_zone = gray[int(h * 0.25):int(h * 0.45), int(w * 0.15):int(w * 0.85)]
    if eye_zone.size == 0:
        return {"eye_openness": None, "blink_detected": False}

    mean_brightness = float(eye_zone.mean())
    # Dark region = eyes open (iris/pupil); bright = closed (eyelid)
    # Normalized: 0.0 = very dark (open), 1.0 = very bright (closed)
    openness = 1.0 - min(mean_brightness / 200.0, 1.0)
    return {
        "eye_openness": round(openness, 4),
        "blink_detected": openness < 0.30,
    }


# ── Skin texture analysis ─────────────────────────────────────

def analyze_skin_texture(face: FaceRegion) -> dict:
    """
    Analyse skin texture in the cheek and forehead regions.

    Deepfake faces show:
      - Low Laplacian variance (over-smoothed, loss of pores/fine lines)
      - High LBP uniformity (synthetic, repetitive texture patterns)
      - Abnormal frequency content in the mid-frequency range
    """
    crop = face.crop
    gray = crop.mean(axis=2).astype(np.float32)
    h, w = gray.shape

    # Focus on cheek + forehead (avoid eyes/mouth which have natural artifacts)
    regions = {
        "left_cheek":  gray[int(h*0.45):int(h*0.70), int(w*0.10):int(w*0.35)],
        "right_cheek": gray[int(h*0.45):int(h*0.70), int(w*0.65):int(w*0.90)],
        "forehead":    gray[int(h*0.08):int(h*0.28), int(w*0.25):int(w*0.75)],
    }

    lap_vars: list[float] = []
    lbp_unis: list[float] = []

    for name, region in regions.items():
        if region.size < 100:
            continue

        # Laplacian variance: measures texture sharpness
        # Low variance → over-smoothed → deepfake signal
        lap = _laplacian_variance(region)
        lap_vars.append(lap)

        # LBP uniformity: measures texture regularity
        # High uniformity → synthetic pattern → deepfake signal
        lbp = _lbp_uniformity(region)
        lbp_unis.append(lbp)

    if not lap_vars:
        return {"texture_score": 0.5, "laplacian_variance": 0.0, "lbp_uniformity": 0.5}

    mean_lap = float(np.mean(lap_vars))
    mean_lbp = float(np.mean(lbp_unis))

    # Low Laplacian variance is a deepfake signal (over-smoothed)
    # Typical real skin: variance ~300–800; deepfake: ~50–200
    lap_signal = max(0.0, 1.0 - mean_lap / 400.0)   # high when variance is low

    # High LBP uniformity is a deepfake signal
    lbp_signal = min(mean_lbp, 1.0)

    texture_score = 0.60 * lap_signal + 0.40 * lbp_signal

    return {
        "texture_score":      round(float(texture_score), 4),
        "laplacian_variance": round(float(mean_lap), 2),
        "lbp_uniformity":     round(float(mean_lbp), 4),
    }


def _laplacian_variance(region: np.ndarray) -> float:
    """Compute Laplacian variance of a grayscale region."""
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    # Manual 2D convolution (no cv2 dependency)
    from scipy.ndimage import convolve  # type: ignore
    try:
        lap = convolve(region, kernel, mode="nearest")
        return float(np.var(lap))
    except ImportError:
        # Pure numpy fallback — use edge padding to avoid border artifacts
        pad = np.pad(region, 1, mode="edge")
        lap = (pad[:-2, 1:-1] + pad[2:, 1:-1] +
               pad[1:-1, :-2] + pad[1:-1, 2:] - 4 * region)
        return float(np.var(lap))


def _lbp_uniformity(region: np.ndarray) -> float:
    """
    Simplified Local Binary Pattern uniformity score.
    High score = uniform/repetitive texture = synthetic appearance.
    """
    if region.shape[0] < 3 or region.shape[1] < 3:
        return 0.5

    center = region[1:-1, 1:-1]
    # Compare centre pixel to 4 neighbours
    neighbors = np.stack([
        region[0:-2, 1:-1],   # top
        region[2:,   1:-1],   # bottom
        region[1:-1, 0:-2],   # left
        region[1:-1, 2:],     # right
    ])
    # LBP code: binary comparison
    lbp = (neighbors >= center[np.newaxis, :, :]).astype(np.uint8)
    codes = lbp.sum(axis=0)   # range 0–4

    # Uniformity: fraction of pixels with the most common code
    hist = np.bincount(codes.ravel(), minlength=5)
    uniformity = hist.max() / max(hist.sum(), 1)
    return float(uniformity)


# ── Lighting consistency analysis ─────────────────────────────

def analyze_lighting(face: FaceRegion, full_frame: np.ndarray) -> dict:
    """
    Check whether the face illumination is consistent with the scene.
    Deepfakes paste a face from one lighting condition into a scene
    with a different lighting direction.
    """
    try:
        return _lighting_analysis(face, full_frame)
    except Exception as exc:
        log.debug("lighting_analysis_failed", error=str(exc))
        return {"lighting_score": 0.5, "illumination_std": 0.0, "specular_anomaly": 0.0}


def _lighting_analysis(face: FaceRegion, full_frame: np.ndarray) -> dict:
    crop  = face.crop
    gray_face  = crop.mean(axis=2).astype(np.float32)
    gray_scene = full_frame.mean(axis=2).astype(np.float32)

    # Estimate illumination gradient direction in the face region
    grad_x_face = np.gradient(gray_face, axis=1)
    grad_y_face = np.gradient(gray_face, axis=0)
    face_angle  = float(np.arctan2(grad_y_face.mean(), grad_x_face.mean()))

    # Sample scene gradient at a background region (avoid face area)
    x1, y1, x2, y2 = face.bbox
    h, w = full_frame.shape[:2]
    # Use a region to the left or right of the face as background reference
    if x1 > w * 0.2:
        bg = gray_scene[:, :x1]
    elif x2 < w * 0.8:
        bg = gray_scene[:, x2:]
    else:
        bg = gray_scene[y2:, :]   # below face

    if bg.size < 400:
        return {"lighting_score": 0.5, "illumination_std": 0.0, "specular_anomaly": 0.0}

    grad_x_bg  = np.gradient(bg, axis=1)
    grad_y_bg  = np.gradient(bg, axis=0)
    scene_angle = float(np.arctan2(grad_y_bg.mean(), grad_x_bg.mean()))

    # Angular difference between face and scene lighting direction
    angle_diff = abs(face_angle - scene_angle)
    angle_diff = min(angle_diff, np.pi - angle_diff)   # wrap to [0, pi/2]

    # Normalise to [0,1]: 0 = perfectly consistent, 1 = completely inconsistent
    lighting_score = float(angle_diff / (np.pi / 2))

    # Specular highlight detection: bright spots on the face
    bright_mask = (gray_face > 200).astype(float)
    specular_density = float(bright_mask.mean())
    # Typical faces: 0–5% pixels as specular highlights
    specular_anomaly = max(0.0, specular_density - 0.05) * 10.0

    return {
        "lighting_score":   round(lighting_score, 4),
        "illumination_std": round(float(np.std([face_angle, scene_angle])), 4),
        "specular_anomaly": round(min(specular_anomaly, 1.0), 4),
    }


# ── Full per-frame artifact analysis ──────────────────────────

def analyze_frame_artifacts(
    frame_faces: FrameFaces,
    full_frame_image: np.ndarray,
) -> list[ArtifactAnalysis]:
    """
    Run all three artifact analyses on every detected face in a frame.
    Returns one ArtifactAnalysis per face.
    """
    results: list[ArtifactAnalysis] = []

    for face in frame_faces.faces:
        blink   = analyze_blink(face)
        texture = analyze_skin_texture(face)
        light   = analyze_lighting(face, full_frame_image)

        # Weighted composite score
        artifact_score = (
            0.30 * texture["texture_score"]
            + 0.30 * light["lighting_score"]
            + 0.20 * (1.0 - min((texture["laplacian_variance"] or 0) / 400.0, 1.0))
            + 0.20 * light.get("specular_anomaly", 0.0)
        )

        results.append(ArtifactAnalysis(
            frame_idx=frame_faces.frame_idx,
            timestamp_s=frame_faces.timestamp_s,
            eye_openness=blink.get("eye_openness"),
            blink_detected=blink.get("blink_detected", False),
            texture_score=texture["texture_score"],
            laplacian_variance=texture["laplacian_variance"],
            lbp_uniformity=texture["lbp_uniformity"],
            lighting_score=light["lighting_score"],
            illumination_std=light["illumination_std"],
            specular_anomaly=light.get("specular_anomaly", 0.0),
            artifact_score=round(min(artifact_score, 1.0), 4),
        ))

    return results
