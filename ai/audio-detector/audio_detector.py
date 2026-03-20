"""
Audio deepfake detector — single entry point for inference.
Called by the Celery audio worker for every job.

Pipeline:
  raw bytes → load_audio → chunk_audio → [extract_features per chunk]
  → [AudioEnsemble.predict_chunk per chunk] → calibrate → aggregate
  → AudioDetectionResult

Step 54: Target >90% accuracy on ASVspoof benchmarks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from ..pipeline.preprocessing import AudioSegment, load_audio, chunk_audio, AudioChunk
from ..features.extractor import extract_features, AudioFeatures
from ..models.classifier import AudioEnsemble

log = structlog.get_logger(__name__)


@dataclass
class ChunkResult:
    """Detection result for one 30-second chunk."""
    chunk_idx:  int
    start_s:    float
    end_s:      float
    score:      float            # calibrated AI probability
    model_scores: dict[str, float]  # per-model breakdown


@dataclass
class AudioDetectionResult:
    """Final result for the full audio file."""
    score:           float        # calibrated [0, 1] AI probability
    label:           str          # "AI" | "HUMAN" | "UNCERTAIN"
    confidence:      float
    chunk_results:   list[ChunkResult]
    flagged_segments: list[dict]  # segments with score > 0.75 (for timeline view)
    evidence:        dict[str, Any]
    processing_ms:   int


class AudioDetector:
    """
    Full audio deepfake detection pipeline.
    load_models() once at worker startup; analyze() per request.
    """

    def __init__(
        self,
        checkpoint_dir: Path | None = None,
        calibration_path: Path | None = None,
        device: str | None = None,
    ) -> None:
        self._ensemble   = AudioEnsemble(checkpoint_dir, device)
        self._platt:   Any = None
        self._isotonic: Any = None
        self._calibration_path = calibration_path
        self._loaded = False

    def load_models(self) -> None:
        self._ensemble.load()

        if self._calibration_path and self._calibration_path.exists():
            import pickle
            with self._calibration_path.open("rb") as f:
                cal = pickle.load(f)
            self._platt    = cal.get("platt")
            self._isotonic = cal.get("isotonic")
            log.info("audio_calibration_loaded")

        self._loaded = True
        log.info("audio_detector_ready")

    def analyze(self, data: bytes, filename: str) -> AudioDetectionResult:
        if not self._loaded:
            raise RuntimeError("Call load_models() first")

        t_start = int(time.time() * 1000)

        # ── Step 47: Load and preprocess ──────────────────────
        segment  = load_audio(data, filename)
        chunks   = chunk_audio(segment)

        log.info("audio_analysis_start",
                 filename=filename,
                 duration_s=round(segment.duration_s, 1),
                 n_chunks=len(chunks))

        # ── Steps 48–49: Extract features per chunk ───────────
        chunk_features: list[tuple[AudioChunk, AudioFeatures]] = []
        for chunk in chunks:
            features = extract_features(chunk)
            chunk_features.append((chunk, features))

        # ── Step 50: Run ensemble per chunk ───────────────────
        chunk_results: list[ChunkResult] = []
        for chunk, features in chunk_features:
            model_scores = self._ensemble.predict_chunk(features)
            raw_score    = float(np.mean(list(model_scores.values())))
            calibrated   = self._calibrate(raw_score)

            chunk_results.append(ChunkResult(
                chunk_idx=chunk.chunk_idx,
                start_s=chunk.start_s,
                end_s=chunk.end_s,
                score=calibrated,
                model_scores=model_scores,
            ))

        # ── Aggregate across chunks ────────────────────────────
        chunk_scores = [r.score for r in chunk_results]

        # Weight by max-score to surface the worst chunk
        # (one genuine deepfake chunk in an otherwise clean recording = deepfake)
        max_score  = max(chunk_scores)
        mean_score = float(np.mean(chunk_scores))
        # Blend: 60% max, 40% mean — catches spliced deepfakes
        final_score = 0.60 * max_score + 0.40 * mean_score
        final_score = max(0.01, min(0.99, final_score))

        label = (
            "AI"        if final_score >= 0.75 else
            "HUMAN"     if final_score <= 0.40 else
            "UNCERTAIN"
        )
        confidence = round(abs(final_score - 0.5) * 2, 4)

        # Flagged segments for the frontend timeline view
        flagged = [
            {
                "start_s":   r.start_s,
                "end_s":     r.end_s,
                "score":     r.score,
                "severity":  "high" if r.score >= 0.85 else "medium",
            }
            for r in chunk_results if r.score >= 0.65
        ]

        # Aggregate feature evidence
        all_features = [f for _, f in chunk_features]
        evidence = self._build_evidence(all_features, chunk_results, segment)

        processing_ms = int(time.time() * 1000) - t_start
        log.info("audio_analysis_complete",
                 score=round(final_score, 4), label=label,
                 n_flagged=len(flagged), ms=processing_ms)

        return AudioDetectionResult(
            score=round(final_score, 4),
            label=label,
            confidence=confidence,
            chunk_results=chunk_results,
            flagged_segments=flagged,
            evidence=evidence,
            processing_ms=processing_ms,
        )

    def _calibrate(self, raw: float) -> float:
        """Apply Platt + isotonic calibration if available."""
        if self._platt and self._isotonic:
            from ..training.train import apply_calibration
            return apply_calibration(raw, self._platt, self._isotonic)
        return float(np.clip(raw, 0.01, 0.99))

    @staticmethod
    def _build_evidence(
        features: list[AudioFeatures],
        chunks: list[ChunkResult],
        segment: AudioSegment,
    ) -> dict[str, Any]:
        if not features:
            return {}

        gdd_means  = [f.gdd_mean  for f in features]
        gdd_stds   = [f.gdd_std   for f in features]
        jitters    = [f.jitter    for f in features]
        shimmers   = [f.shimmer   for f in features]
        f0_means   = [f.f0_mean   for f in features if f.f0_mean > 50]

        scores = [c.score for c in chunks]

        return {
            "duration_s":        round(segment.duration_s, 2),
            "n_chunks":          len(chunks),
            "mean_gdd":          round(float(np.mean(gdd_means)), 4),
            "mean_gdd_std":      round(float(np.mean(gdd_stds)), 4),
            "mean_jitter":       round(float(np.mean(jitters)), 4),
            "mean_shimmer":      round(float(np.mean(shimmers)), 4),
            "mean_f0":           round(float(np.mean(f0_means)), 2) if f0_means else 0.0,
            "chunk_score_max":   round(max(scores), 4),
            "chunk_score_mean":  round(float(np.mean(scores)), 4),
            "chunk_score_std":   round(float(np.std(scores)), 4),
            "original_sr":       segment.original_sr,
            "file_format":       segment.file_format,
            "n_channels_original": segment.n_channels_original,
            "signals": _build_signals(
                float(np.mean(gdd_stds)), float(np.mean(jitters)),
                float(np.mean(shimmers)), max(scores),
            ),
        }


def _build_signals(
    gdd_std: float, jitter: float, shimmer: float, max_chunk_score: float,
) -> list[dict[str, str]]:
    """Human-readable evidence signals for the UI evidence panel."""
    signals = []
    if gdd_std > 1.5:
        signals.append({"signal": "High phase discontinuity (GDD)",
                         "value": f"{gdd_std:.2f}", "weight": "high"})
    if jitter < 0.8 and jitter > 0.0:
        signals.append({"signal": "Unnaturally low F0 jitter",
                         "value": f"{jitter:.2f}%", "weight": "medium"})
    if shimmer < 0.02:
        signals.append({"signal": "Unnaturally low shimmer",
                         "value": f"{shimmer:.4f}", "weight": "medium"})
    if max_chunk_score > 0.85:
        signals.append({"signal": "Flagged segment detected",
                         "value": f"{max_chunk_score:.0%}", "weight": "high"})
    return signals
