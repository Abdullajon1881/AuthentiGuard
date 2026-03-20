"""
Step 64: Full video deepfake detection pipeline.
Wires: frame extraction → face detection → artifact analysis →
temporal consistency → classifier ensemble → calibrate → aggregate.

Target: >85% accuracy on FaceForensics++ benchmarks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from .pipeline.frame_extractor import extract_frames, VideoMetadata, VideoFrame
from .pipeline.face_detector import FaceDetector, FrameFaces
from .features.artifact_analyzer import analyze_frame_artifacts, ArtifactAnalysis
from .features.temporal_analyzer import compute_temporal_features, TemporalFeatures
from .models.classifier import VideoClassifierEnsemble

log = structlog.get_logger(__name__)

# Minimum faces to trust a video-level score
MIN_FACES_FOR_RELIABLE_SCORE = 3


@dataclass
class VideoFrameResult:
    """Detection result for one extracted frame."""
    frame_idx:    int
    timestamp_s:  float
    n_faces:      int
    frame_score:  float     # mean classifier score across all faces in frame
    artifact_score: float
    model_scores: dict[str, float]


@dataclass
class VideoDetectionResult:
    """Full detection result for one video file."""
    score:           float
    label:           str
    confidence:      float
    frame_results:   list[VideoFrameResult]
    flagged_segments: list[dict]
    temporal:        TemporalFeatures
    metadata:        VideoMetadata
    evidence:        dict[str, Any]
    processing_ms:   int


class VideoDetector:
    """
    End-to-end video deepfake detector.
    load_models() once at worker startup; analyze() per request.
    """

    def __init__(
        self,
        checkpoint_dir: Path | None = None,
        device: str | None = None,
    ) -> None:
        self._face_detector = FaceDetector()
        self._classifier    = VideoClassifierEnsemble(checkpoint_dir, device)
        self._loaded = False

    def load_models(self) -> None:
        self._face_detector.load()
        self._classifier.load()
        self._loaded = True
        log.info("video_detector_ready")

    def analyze(self, data: bytes, filename: str) -> VideoDetectionResult:
        if not self._loaded:
            raise RuntimeError("Call load_models() first")

        t_start = int(time.time() * 1000)

        # ── Step 55: Extract frames ────────────────────────────
        frames, meta = extract_frames(data, filename)
        if not frames:
            return self._no_frames_result(meta, t_start)

        log.info("video_frames_ready",
                 n=len(frames), duration_s=round(meta.duration_s, 1))

        # ── Step 56: Detect faces ──────────────────────────────
        all_frame_faces: list[FrameFaces] = self._face_detector.detect_batch(frames)

        frames_with_faces = [ff for ff in all_frame_faces if ff.n_faces > 0]
        if len(frames_with_faces) < MIN_FACES_FOR_RELIABLE_SCORE:
            log.warning("insufficient_faces_detected",
                        n_frames_with_faces=len(frames_with_faces))

        # ── Steps 57–58: Artifact + temporal analysis ──────────
        all_artifact_analyses: list[ArtifactAnalysis] = []
        all_face_crops: list[np.ndarray] = []

        frame_lookup = {f.frame_idx: f for f in frames}

        for frame_faces in all_frame_faces:
            full_frame = frame_lookup.get(frame_faces.frame_idx)
            if full_frame is None:
                continue
            artifacts = analyze_frame_artifacts(frame_faces, full_frame.image)
            all_artifact_analyses.extend(artifacts)
            for face in frame_faces.faces:
                all_face_crops.append(face.crop)

        temporal = compute_temporal_features(
            all_artifact_analyses,
            all_face_crops,
            video_fps=meta.fps or 2.0,
        )

        # ── Steps 59–62: Classifier ensemble per frame ────────
        frame_results: list[VideoFrameResult] = []

        for frame_faces in all_frame_faces:
            if frame_faces.n_faces == 0:
                continue

            face_scores: list[float] = []
            all_model_scores: dict[str, list[float]] = {}

            for face in frame_faces.faces:
                try:
                    model_scores = self._classifier.predict_crop(face.crop)
                    ensemble_s   = self._classifier.predict_crop_ensemble(face.crop)
                    calibrated   = self._classifier.calibrate(ensemble_s)
                    face_scores.append(calibrated)
                    for m, s in model_scores.items():
                        all_model_scores.setdefault(m, []).append(s)
                except Exception as exc:
                    log.warning("frame_classify_failed",
                                frame=frame_faces.frame_idx, error=str(exc))

            if not face_scores:
                continue

            frame_score = float(np.mean(face_scores))
            mean_model  = {m: float(np.mean(v)) for m, v in all_model_scores.items()}

            # Get artifact score for this frame
            frame_art = next(
                (a.artifact_score for a in all_artifact_analyses
                 if a.frame_idx == frame_faces.frame_idx), 0.0
            )

            frame_results.append(VideoFrameResult(
                frame_idx=frame_faces.frame_idx,
                timestamp_s=frame_faces.timestamp_s,
                n_faces=frame_faces.n_faces,
                frame_score=round(frame_score, 4),
                artifact_score=frame_art,
                model_scores=mean_model,
            ))

        # ── Aggregate to video-level score ─────────────────────
        final_score = self._aggregate_score(frame_results, temporal, all_artifact_analyses)

        label = (
            "AI"        if final_score >= 0.75 else
            "HUMAN"     if final_score <= 0.40 else
            "UNCERTAIN"
        )
        confidence = round(abs(final_score - 0.5) * 2, 4)

        # Flagged segments for the timeline view
        flagged = self._build_flagged_segments(frame_results)

        evidence = self._build_evidence(frame_results, temporal, meta, all_artifact_analyses)
        processing_ms = int(time.time() * 1000) - t_start

        log.info("video_detection_complete",
                 score=round(final_score, 4), label=label,
                 n_frames=len(frame_results), ms=processing_ms)

        return VideoDetectionResult(
            score=round(final_score, 4),
            label=label,
            confidence=confidence,
            frame_results=frame_results,
            flagged_segments=flagged,
            temporal=temporal,
            metadata=meta,
            evidence=evidence,
            processing_ms=processing_ms,
        )

    # ── Score aggregation ─────────────────────────────────────

    @staticmethod
    def _aggregate_score(
        frame_results: list[VideoFrameResult],
        temporal: TemporalFeatures,
        artifacts: list[ArtifactAnalysis],
    ) -> float:
        if not frame_results:
            return 0.5

        scores = [r.frame_score for r in frame_results]
        art_scores = [a.artifact_score for a in artifacts]

        max_score  = max(scores)
        mean_score = float(np.mean(scores))
        temporal_s = temporal.temporal_consistency_score
        artifact_s = float(np.mean(art_scores)) if art_scores else 0.0

        # Blend: classifier (50%), temporal (25%), artifacts (15%), max chunk (10%)
        combined = (
            0.50 * mean_score
            + 0.25 * temporal_s
            + 0.15 * artifact_s
            + 0.10 * max_score
        )
        return float(np.clip(combined, 0.01, 0.99))

    @staticmethod
    def _build_flagged_segments(
        frame_results: list[VideoFrameResult],
    ) -> list[dict]:
        """Group consecutive flagged frames into segments."""
        flagged: list[dict] = []
        in_segment = False
        seg_start  = 0.0
        seg_scores: list[float] = []

        THRESHOLD = 0.65

        for r in sorted(frame_results, key=lambda x: x.timestamp_s):
            if r.frame_score >= THRESHOLD:
                if not in_segment:
                    in_segment = True
                    seg_start  = r.timestamp_s
                    seg_scores = []
                seg_scores.append(r.frame_score)
            else:
                if in_segment:
                    mean_s = float(np.mean(seg_scores))
                    flagged.append({
                        "start_s":  seg_start,
                        "end_s":    r.timestamp_s,
                        "score":    round(mean_s, 4),
                        "severity": "high" if mean_s >= 0.85 else "medium",
                    })
                    in_segment = False

        # Close open segment
        if in_segment and frame_results:
            mean_s = float(np.mean(seg_scores))
            flagged.append({
                "start_s":  seg_start,
                "end_s":    frame_results[-1].timestamp_s,
                "score":    round(mean_s, 4),
                "severity": "high" if mean_s >= 0.85 else "medium",
            })

        return flagged

    @staticmethod
    def _build_evidence(
        frame_results: list[VideoFrameResult],
        temporal: TemporalFeatures,
        meta: VideoMetadata,
        artifacts: list[ArtifactAnalysis],
    ) -> dict[str, Any]:
        scores = [r.frame_score for r in frame_results]
        art_s  = [a.artifact_score for a in artifacts]

        signals: list[dict] = []
        if temporal.embedding_variance > 0.05:
            signals.append({"signal": "Face identity drift detected",
                             "value": f"{temporal.embedding_variance:.4f}", "weight": "high"})
        if temporal.blink_regularity > 0.7:
            signals.append({"signal": "Unnaturally regular blinking pattern",
                             "value": f"{temporal.blink_regularity:.2f}", "weight": "medium"})
        if temporal.pose_jitter > 0.3:
            signals.append({"signal": "Unnatural head pose jitter",
                             "value": f"{temporal.pose_jitter:.2f}", "weight": "medium"})
        if art_s and float(np.mean(art_s)) > 0.5:
            signals.append({"signal": "Skin texture anomalies",
                             "value": f"{float(np.mean(art_s)):.2f}", "weight": "medium"})

        return {
            "n_frames_analyzed":   len(frame_results),
            "duration_s":          round(meta.duration_s, 2),
            "fps":                 meta.fps,
            "codec":               meta.codec,
            "resolution":          f"{meta.width}×{meta.height}",
            "frame_score_mean":    round(float(np.mean(scores)), 4) if scores else 0.0,
            "frame_score_max":     round(float(max(scores)), 4) if scores else 0.0,
            "frame_score_std":     round(float(np.std(scores)), 4) if scores else 0.0,
            "temporal_score":      temporal.temporal_consistency_score,
            "embedding_variance":  temporal.embedding_variance,
            "blink_rate":          temporal.blink_rate,
            "blink_regularity":    temporal.blink_regularity,
            "top_signals":         signals,
        }

    @staticmethod
    def _no_frames_result(meta: VideoMetadata, t_start: int) -> VideoDetectionResult:
        from .features.temporal_analyzer import _empty_temporal
        return VideoDetectionResult(
            score=0.5, label="UNCERTAIN", confidence=0.0,
            frame_results=[], flagged_segments=[],
            temporal=_empty_temporal(), metadata=meta,
            evidence={"error": "No frames could be extracted"},
            processing_ms=int(time.time() * 1000) - t_start,
        )
