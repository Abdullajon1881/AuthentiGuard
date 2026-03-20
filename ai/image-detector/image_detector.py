"""
Step 72: Full image AI detection pipeline.
Target: >88% accuracy on StyleGAN + GAN Fingerprint benchmarks.

Pipeline:
  raw bytes → load_image → extract_all_features →
  ImageClassifierEnsemble.predict_with_features → calibrate → ImageDetectionResult
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from .pipeline.preprocessing import load_image, ImageRecord
from .features.extractor import extract_all_features, ImageFeatures
from .models.classifier import ImageClassifierEnsemble

log = structlog.get_logger(__name__)


@dataclass
class ImageDetectionResult:
    """Final detection result for one image file."""
    score:        float
    label:        str
    confidence:   float
    features:     ImageFeatures
    model_scores: dict[str, float]
    evidence:     dict[str, Any]
    processing_ms: int


class ImageDetector:
    """
    End-to-end image AI detector.
    load_models() once at worker startup; analyze() per request.
    """

    def __init__(
        self,
        checkpoint_dir: Path | None = None,
        device: str | None = None,
    ) -> None:
        self._ensemble = ImageClassifierEnsemble(checkpoint_dir, device)
        self._loaded   = False

    def load_models(self) -> None:
        self._ensemble.load()
        self._loaded = True
        log.info("image_detector_ready")

    def analyze(self, data: bytes, filename: str) -> ImageDetectionResult:
        if not self._loaded:
            raise RuntimeError("Call load_models() first")

        t_start = int(time.time() * 1000)

        # Step 65: Load and preprocess
        record = load_image(data, filename)
        log.info("image_loaded", filename=filename,
                 size=f"{record.width}×{record.height}",
                 format=record.file_format)

        # Steps 66–68: Extract features
        features = extract_all_features(record)

        # Steps 69–70: Classifier ensemble
        model_scores = self._ensemble.predict(record.tensor_224)
        raw_score    = self._ensemble.predict_with_features(
            record.tensor_224, features.feature_vector
        )
        score = self._ensemble.calibrate(raw_score)

        label = (
            "AI"        if score >= 0.75 else
            "HUMAN"     if score <= 0.40 else
            "UNCERTAIN"
        )
        confidence = round(abs(score - 0.5) * 2, 4)
        processing_ms = int(time.time() * 1000) - t_start

        evidence = self._build_evidence(record, features, model_scores, score)

        log.info("image_detection_complete",
                 score=round(score, 4), label=label, ms=processing_ms)

        return ImageDetectionResult(
            score=round(score, 4),
            label=label,
            confidence=confidence,
            features=features,
            model_scores=model_scores,
            evidence=evidence,
            processing_ms=processing_ms,
        )

    @staticmethod
    def _build_evidence(
        record: ImageRecord,
        features: ImageFeatures,
        model_scores: dict[str, float],
        score: float,
    ) -> dict[str, Any]:
        signals: list[dict] = []

        if features.fft_grid_score > 0.35:
            signals.append({"signal": "GAN checkerboard artifact detected",
                             "value": f"{features.fft_grid_score:.2f}",
                             "weight": "high"})
        if features.fft_high_freq_ratio > 0.50:
            signals.append({"signal": "High-frequency content suppressed",
                             "value": f"{features.fft_high_freq_ratio:.2f}",
                             "weight": "high"})
        if features.fingerprint_correlation > 0.40:
            signals.append({"signal": "GAN model fingerprint detected",
                             "value": f"{features.fingerprint_correlation:.2f}",
                             "weight": "high"})
        if features.bilateral_symmetry > 0.90:
            signals.append({"signal": "Unnatural bilateral symmetry",
                             "value": f"{features.bilateral_symmetry:.3f}",
                             "weight": "medium"})
        if features.texture_uniformity > 0.65:
            signals.append({"signal": "Over-uniform texture (synthetic skin)",
                             "value": f"{features.texture_uniformity:.3f}",
                             "weight": "medium"})
        if record.jpeg_quality and record.jpeg_quality > 92:
            signals.append({"signal": "Unusually high JPEG quality (no compression artifacts)",
                             "value": f"Q{record.jpeg_quality}",
                             "weight": "low"})

        return {
            "width":          record.width,
            "height":         record.height,
            "format":         record.file_format,
            "file_size":      record.file_size,
            "jpeg_quality":   record.jpeg_quality,
            "has_alpha":      record.has_alpha,

            # Feature scores
            "fingerprint_correlation": round(features.fingerprint_correlation, 4),
            "fingerprint_energy":      round(features.fingerprint_energy, 4),
            "fft_peak_regularity":     round(features.fft_peak_regularity, 4),
            "fft_high_freq_ratio":     round(features.fft_high_freq_ratio, 4),
            "fft_azimuthal_variance":  round(features.fft_azimuthal_variance, 4),
            "fft_grid_score":          round(features.fft_grid_score, 4),
            "texture_uniformity":      round(features.texture_uniformity, 4),
            "glcm_contrast":           round(features.glcm_contrast, 4),
            "bilateral_symmetry":      round(features.bilateral_symmetry, 4),
            "background_uniformity":   round(features.background_uniformity, 4),
            "color_distribution_score": round(features.color_distribution_score, 4),

            "model_scores": {k: round(v, 4) for k, v in model_scores.items()},
            "top_signals": signals,
        }
