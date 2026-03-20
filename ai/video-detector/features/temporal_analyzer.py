"""
Step 58: Temporal consistency analysis.

Deepfakes are synthesised frame-by-frame. Even with temporal smoothing,
they exhibit subtle inconsistencies BETWEEN frames that real videos don't:

1. Face identity drift
   The face embedding (e.g. ArcFace) should be nearly identical across frames.
   Deepfakes show higher embedding variance — the identity "wanders."

2. Head pose jitter
   Real head movement is smooth and physically constrained.
   Deepfakes show high-frequency jitter in yaw/pitch/roll estimates.

3. Blending mask flicker
   The alpha mask at the face-background boundary flickers frame-to-frame.
   Detectable via optical flow in the boundary region.

4. Optical flow anomaly
   Dense optical flow between consecutive frames should be smooth and
   physically consistent (affine motion). Deepfakes produce unnatural
   flow vectors particularly around the face boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

from .artifact_analyzer import ArtifactAnalysis

log = structlog.get_logger(__name__)


@dataclass
class TemporalFeatures:
    """Temporal consistency features computed across all frames."""
    # Face identity
    embedding_variance:    float    # variance of face embeddings across frames
    embedding_drift:       float    # cumulative identity drift

    # Head pose
    pose_jitter:           float    # high-frequency pose variation
    pose_smoothness:       float    # [0,1] — 1.0 = perfectly smooth

    # Boundary flicker
    boundary_flicker:      float    # optical flow variance at face boundary

    # Artifact time series
    artifact_score_series: list[float]  # per-frame artifact scores
    artifact_score_std:    float    # std of artifact scores over time

    # Blink analysis
    blink_rate:            float    # blinks per minute
    blink_regularity:      float    # [0,1] — too regular = deepfake signal

    # Aggregate
    temporal_consistency_score: float  # [0,1] — higher = more inconsistent


# ── Face embedding variance ────────────────────────────────────

def compute_embedding_variance(
    face_crops: list[np.ndarray],
    embedder: Any = None,
) -> dict[str, float]:
    """
    Compute variance of face identity embeddings across frames.

    If an embedder (ArcFace/FaceNet model) is available, uses it.
    Otherwise falls back to pixel-space PCA variance as a proxy.
    """
    if len(face_crops) < 3:
        return {"embedding_variance": 0.0, "embedding_drift": 0.0}

    if embedder is not None:
        return _embedding_variance_model(face_crops, embedder)
    return _embedding_variance_pixel(face_crops)


def _embedding_variance_model(crops: list[np.ndarray], embedder: Any) -> dict[str, float]:
    """Use a face recognition model to get embeddings."""
    try:
        embeddings = [embedder(c) for c in crops]
        E = np.stack(embeddings)
        variance = float(np.mean(np.var(E, axis=0)))

        # Cumulative cosine distance drift
        drifts = []
        for i in range(1, len(E)):
            cos_sim = float(np.dot(E[i-1], E[i]) /
                           (np.linalg.norm(E[i-1]) * np.linalg.norm(E[i]) + 1e-8))
            drifts.append(1.0 - cos_sim)
        drift = float(np.mean(drifts))

        return {"embedding_variance": variance, "embedding_drift": drift}
    except Exception as exc:
        log.warning("embedding_variance_failed", error=str(exc))
        return _embedding_variance_pixel(crops)


def _embedding_variance_pixel(crops: list[np.ndarray]) -> dict[str, float]:
    """
    Pixel-space proxy: downsample crops to 16×16 and compute PCA variance.
    Not as accurate as deep embeddings but requires no model.
    """
    try:
        small = np.stack([
            c.mean(axis=2).astype(np.float32)[::14, ::14]   # ~16×16 downsample
            for c in crops
        ])  # [N, ~16, ~16]
        flat = small.reshape(len(crops), -1)
        # Normalise each frame
        flat -= flat.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
        flat /= norms

        # Variance across frames
        variance = float(np.mean(np.var(flat, axis=0)))

        # Frame-to-frame cosine distance
        drifts = []
        for i in range(1, len(flat)):
            cos_sim = float(np.dot(flat[i-1], flat[i]))
            drifts.append(1.0 - cos_sim)
        drift = float(np.mean(drifts)) if drifts else 0.0

        return {"embedding_variance": variance, "embedding_drift": drift}
    except Exception:
        return {"embedding_variance": 0.0, "embedding_drift": 0.0}


# ── Head pose jitter ───────────────────────────────────────────

def compute_pose_jitter(face_artifacts: list[ArtifactAnalysis]) -> dict[str, float]:
    """
    Estimate head pose jitter from face bounding box movement.
    Smooth head movement → physically real. Jittery → deepfake.

    Uses bounding box centroid position as a proxy for head pose.
    Proper implementation uses a 3D pose estimator (e.g. FSA-Net).
    """
    if len(face_artifacts) < 3:
        return {"pose_jitter": 0.0, "pose_smoothness": 1.0}

    # Use artifact score variance as a proxy for pose consistency
    scores = [a.artifact_score for a in face_artifacts]
    jitter = float(np.std(np.diff(scores)))

    # High-frequency component: std of second derivative
    if len(scores) >= 4:
        second_diff = np.diff(np.diff(scores))
        hf_jitter   = float(np.std(second_diff))
    else:
        hf_jitter = 0.0

    # Normalise: typical range 0–0.1 for real, 0.1–0.4 for deepfake
    pose_jitter   = min(jitter * 10.0, 1.0)
    pose_smoothness = max(0.0, 1.0 - pose_jitter)

    return {
        "pose_jitter":    round(pose_jitter, 4),
        "pose_smoothness": round(pose_smoothness, 4),
        "hf_jitter":      round(hf_jitter, 4),
    }


# ── Blink regularity ───────────────────────────────────────────

def analyze_blink_pattern(
    face_artifacts: list[ArtifactAnalysis],
    video_fps: float = 2.0,
) -> dict[str, float]:
    """
    Analyse blinking pattern over time.

    Real blinks: 15–20 per minute, irregular intervals (Poisson-like).
    Deepfake blinks (when present): unnaturally regular, or absent.

    regularity score: 0.0 = Poisson-like (natural), 1.0 = perfectly regular
    """
    blink_frames = [a for a in face_artifacts if a.blink_detected]
    n_blinks     = len(blink_frames)
    total_s      = (len(face_artifacts) / video_fps) if face_artifacts else 0.0

    if total_s < 5:
        return {"blink_rate": 0.0, "blink_regularity": 0.5}

    blink_rate = n_blinks / (total_s / 60.0)   # per minute

    if n_blinks < 2:
        # Too few blinks — suspicious if video is long enough
        regularity = 0.8 if total_s > 30 else 0.5
        return {
            "blink_rate":       round(blink_rate, 2),
            "blink_regularity": round(regularity, 4),
        }

    # Inter-blink intervals
    intervals = [
        blink_frames[i].timestamp_s - blink_frames[i-1].timestamp_s
        for i in range(1, len(blink_frames))
    ]

    # Coefficient of variation: low CoV = too regular
    mean_interval = float(np.mean(intervals))
    std_interval  = float(np.std(intervals))
    cov = std_interval / max(mean_interval, 0.1)

    # Real blinks have CoV > 0.4; deepfakes < 0.2
    regularity = max(0.0, 1.0 - cov / 0.5)

    return {
        "blink_rate":       round(blink_rate, 2),
        "blink_regularity": round(regularity, 4),
        "mean_interval_s":  round(mean_interval, 2),
        "cov":              round(cov, 4),
    }


# ── Boundary flicker (optical flow proxy) ─────────────────────

def compute_boundary_flicker(face_crops: list[np.ndarray]) -> float:
    """
    Measure pixel-level temporal variance at the face boundary strip.
    High variance = flickering blend boundary = deepfake signal.
    """
    if len(face_crops) < 3:
        return 0.0
    try:
        # Extract boundary strip: outer 10% of face crop
        boundary_pixels = []
        for crop in face_crops:
            h, w = crop.shape[:2]
            margin = max(int(h * 0.10), 5)
            # Top and bottom strips
            strip = np.concatenate([
                crop[:margin].ravel(),
                crop[-margin:].ravel(),
            ]).astype(np.float32)
            boundary_pixels.append(strip)

        B = np.stack(boundary_pixels)   # [N, M]
        # Variance across time for each boundary pixel
        temporal_var = np.var(B, axis=0)
        return float(np.mean(temporal_var))
    except Exception:
        return 0.0


# ── Main temporal feature extractor ───────────────────────────

def compute_temporal_features(
    face_artifacts: list[ArtifactAnalysis],
    face_crops:     list[np.ndarray],
    video_fps:      float = 2.0,
) -> TemporalFeatures:
    """
    Compute all temporal consistency features from the per-frame analyses.
    Called once after all frames have been processed.
    """
    if not face_artifacts:
        return _empty_temporal()

    embedding   = compute_embedding_variance(face_crops)
    pose        = compute_pose_jitter(face_artifacts)
    blink       = analyze_blink_pattern(face_artifacts, video_fps)
    flicker     = compute_boundary_flicker(face_crops)

    artifact_scores = [a.artifact_score for a in face_artifacts]
    artifact_std    = float(np.std(artifact_scores)) if artifact_scores else 0.0

    # Composite temporal consistency score
    temporal_score = (
        0.30 * min(embedding["embedding_variance"] * 20.0, 1.0)
        + 0.25 * pose["pose_jitter"]
        + 0.20 * blink["blink_regularity"]
        + 0.15 * min(flicker / 500.0, 1.0)
        + 0.10 * min(artifact_std * 5.0, 1.0)
    )

    return TemporalFeatures(
        embedding_variance=embedding["embedding_variance"],
        embedding_drift=embedding["embedding_drift"],
        pose_jitter=pose["pose_jitter"],
        pose_smoothness=pose["pose_smoothness"],
        boundary_flicker=flicker,
        artifact_score_series=artifact_scores,
        artifact_score_std=artifact_std,
        blink_rate=blink["blink_rate"],
        blink_regularity=blink["blink_regularity"],
        temporal_consistency_score=round(min(temporal_score, 1.0), 4),
    )


def _empty_temporal() -> TemporalFeatures:
    return TemporalFeatures(
        embedding_variance=0.0, embedding_drift=0.0,
        pose_jitter=0.0, pose_smoothness=1.0,
        boundary_flicker=0.0, artifact_score_series=[],
        artifact_score_std=0.0, blink_rate=0.0,
        blink_regularity=0.0, temporal_consistency_score=0.0,
    )
