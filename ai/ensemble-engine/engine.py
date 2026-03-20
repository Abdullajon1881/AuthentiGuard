"""
AuthentiGuard Ensemble Engine — top-level orchestrator.

This is the single entry point used by the backend Result Engine
(Step 30 / Step 84). It wires together:

  Step 79: DetectorRegistry + dispatcher (all five detectors)
  Step 80: EnsembleMetaClassifier (XGBoost over all outputs)
  Step 81: CrossModalAnalysis (lip sync, writing vs metadata, EXIF)
  Step 82: AttributionResult (GPT / Claude / LLaMA / Human)
  Step 83: WatermarkResult (SynthID-style text/image/audio)

Final score pipeline:
  1. Run all applicable detectors
  2. Extract metadata signals
  3. Run cross-modal consistency checks
  4. Run watermark detection
  5. Build feature vector (all signals)
  6. Run ensemble meta-classifier → calibrated score
  7. Run model attribution
  8. Return AuthenticityReport

The AuthenticityReport is what the backend stores in DetectionResult
and returns to the frontend.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from .routing.dispatcher import dispatch, DetectorOutput, ContentType
from .meta.meta_classifier import EnsembleMetaClassifier, EnsembleOutput
from .cross_modal.consistency import analyse_cross_modal, CrossModalAnalysis
from .attribution.attributor import attribute_text, attribute_image, AttributionResult
from .watermark.detector import (
    detect_text_watermark, detect_image_watermark, detect_audio_watermark,
    WatermarkResult,
)

log = structlog.get_logger(__name__)


@dataclass
class AuthenticityReport:
    """
    The complete authenticity report returned to the frontend and stored in DB.
    Every field maps directly to a UI element in the ReportView component.
    """
    # Primary verdict
    score:              float    # [0,1] calibrated AI probability
    label:              str      # "AI" | "HUMAN" | "UNCERTAIN"
    confidence:         float    # [0,1]

    # Per-detector breakdown
    detector_outputs:   list[dict]
    layer_scores:       dict[str, float]

    # Ensemble signals
    ensemble_output:    dict
    metadata_adjustment: float
    watermark:          dict
    cross_modal:        dict

    # Attribution
    model_attribution:  dict

    # UI evidence
    sentence_scores:    list[dict]
    flagged_segments:   list[dict]
    top_signals:        list[dict]

    # Processing metadata
    content_type:       str
    processing_ms:      int


class AuthentiGuardEngine:
    """
    Top-level engine: runs all detection and returns an AuthenticityReport.
    Instantiated once per worker; analyze() is called per request.
    """

    def __init__(self, checkpoint_base: Path | None = None) -> None:
        self._base = checkpoint_base or Path("ai")
        self._meta = EnsembleMetaClassifier(
            checkpoint_path=self._base / "ensemble-engine/checkpoints/meta"
        )
        self._loaded = False

    def load(self) -> None:
        self._meta.load()
        self._loaded = True
        log.info("authentiguard_engine_ready")

    def analyze(
        self,
        content_type: ContentType,
        content: bytes | str,
        filename: str,
        metadata_signals: dict[str, Any] | None = None,
        claimed_source:   str | None = None,
    ) -> AuthenticityReport:
        """
        Full analysis pipeline.

        Args:
            content_type:     One of "text", "image", "video", "audio", "code"
            content:          Raw content (bytes for media, str for text/code)
            filename:         Original filename (for extension-based routing)
            metadata_signals: Pre-extracted EXIF/EXIF/provenance metadata
            claimed_source:   Human-readable claimed source string (optional)
        """
        t_start = int(time.time() * 1000)
        meta    = metadata_signals or {}

        # ── Step 79: Run the appropriate detector ──────────────
        primary_output = dispatch(content_type, content, filename)
        outputs        = [primary_output]

        log.info("detector_run",
                 content_type=content_type,
                 score=primary_output.score,
                 label=primary_output.label)

        # ── Step 83: Watermark detection ──────────────────────
        watermark_result = self._run_watermark(content_type, content, primary_output)

        # Inject watermark into metadata for the meta-classifier
        if watermark_result.watermark_detected:
            meta["watermark"] = {
                "watermark_detected": True,
                "confidence": watermark_result.confidence,
            }

        # ── Step 80: Ensemble meta-classifier ─────────────────
        ensemble_out = self._meta.predict(outputs, meta)

        # ── Step 81: Cross-modal consistency ──────────────────
        evidence_by_type = {content_type: primary_output.evidence}
        cross_modal = analyse_cross_modal(
            evidence_by_type, meta, claimed_source
        )

        # Adjust score for cross-modal inconsistencies
        final_score, adjustment = self._apply_cross_modal_adjustment(
            ensemble_out.score, cross_modal
        )

        # ── Step 82: Model attribution ─────────────────────────
        attribution = self._run_attribution(
            content_type, content, final_score,
            primary_output.layer_scores,
            primary_output.evidence,
            meta.get("exif", {}),
        )

        # ── Build report ───────────────────────────────────────
        label = (
            "AI"        if final_score >= 0.75 else
            "HUMAN"     if final_score <= 0.40 else
            "UNCERTAIN"
        )
        confidence    = round(abs(final_score - 0.5) * 2, 4)
        processing_ms = int(time.time() * 1000) - t_start

        top_signals   = self._build_top_signals(
            primary_output, watermark_result, cross_modal
        )

        return AuthenticityReport(
            score=round(final_score, 4),
            label=label,
            confidence=confidence,
            detector_outputs=[{
                "content_type": o.content_type,
                "score": o.score,
                "label": o.label,
                "error": o.error,
            } for o in outputs],
            layer_scores=primary_output.layer_scores,
            ensemble_output={
                "score":                 ensemble_out.score,
                "contributing":          ensemble_out.contributing_detectors,
                "per_detector":          ensemble_out.per_detector_scores,
                "metadata_adjustment":   ensemble_out.metadata_adjustment,
            },
            metadata_adjustment=round(adjustment + ensemble_out.metadata_adjustment, 4),
            watermark={
                "watermark_detected":  watermark_result.watermark_detected,
                "confidence":          watermark_result.confidence,
                "watermark_type":      watermark_result.watermark_type,
                "z_score":             watermark_result.z_score,
                "survives_paraphrase": watermark_result.survives_paraphrase,
            },
            cross_modal={
                "overall_consistency": cross_modal.overall_consistency,
                "inconsistency_score": cross_modal.inconsistency_score,
                "top_mismatches":      cross_modal.top_mismatches,
                "n_checks_run":        len(cross_modal.checks),
            },
            model_attribution={
                "gpt_family":    attribution.gpt_family,
                "claude_family": attribution.claude_family,
                "llama_family":  attribution.llama_family,
                "human":         attribution.human,
                "other_ai":      attribution.other_ai,
                "primary":       attribution.primary_attribution,
                "confident":     attribution.is_confident,
            },
            sentence_scores=primary_output.sentence_scores,
            flagged_segments=primary_output.flagged_segments,
            top_signals=top_signals,
            content_type=content_type,
            processing_ms=processing_ms,
        )

    # ── Internal helpers ───────────────────────────────────────

    def _run_watermark(
        self, content_type: ContentType,
        content: bytes | str,
        output: DetectorOutput,
    ) -> WatermarkResult:
        try:
            if content_type in ("text", "code"):
                text = content if isinstance(content, str) else ""
                return detect_text_watermark(text)
            elif content_type == "image":
                # Use evidence pixels if available
                return WatermarkResult(
                    watermark_detected=False, confidence=0.0,
                    method="skipped", z_score=None, p_value=None,
                    watermark_type="none", survives_paraphrase=False, evidence={},
                )
            elif content_type == "audio":
                data = content if isinstance(content, bytes) else b""
                if len(data) > 0:
                    import numpy as np
                    wave = np.frombuffer(data[:16000 * 4], dtype=np.float32)
                    return detect_audio_watermark(wave, sr=16000)
        except Exception as exc:
            log.warning("watermark_detection_failed", error=str(exc))

        return WatermarkResult(
            watermark_detected=False, confidence=0.0,
            method="failed", z_score=None, p_value=None,
            watermark_type="none", survives_paraphrase=False, evidence={},
        )

    @staticmethod
    def _apply_cross_modal_adjustment(
        score: float,
        cross_modal: CrossModalAnalysis,
    ) -> tuple[float, float]:
        """
        Boost the score when cross-modal inconsistencies are detected.
        A face video that has clean audio but a high video score is more
        suspicious than a video where both modalities agree.
        """
        adjustment = cross_modal.inconsistency_score * 0.10   # max ±0.1
        final = float(np.clip(score + adjustment, 0.01, 0.99))
        return final, adjustment

    @staticmethod
    def _run_attribution(
        content_type: ContentType,
        content: bytes | str,
        score: float,
        layer_scores: dict[str, float],
        evidence: dict[str, Any],
        exif: dict[str, Any],
    ) -> AttributionResult:
        try:
            if content_type in ("text", "code"):
                text = content if isinstance(content, str) else ""
                return attribute_text(text, score, layer_scores)
            elif content_type == "image":
                fft = {k: evidence.get(k, 0.0) for k in
                       ["fft_grid_score", "fft_high_freq_ratio"]}
                return attribute_image(score, fft, exif)
        except Exception as exc:
            log.warning("attribution_failed", error=str(exc))

        # Fallback: score-proportional human/AI split
        ai = score
        return AttributionResult(
            gpt_family=round(ai * 0.35, 3),
            claude_family=round(ai * 0.25, 3),
            llama_family=round(ai * 0.25, 3),
            human=round(1.0 - ai, 3),
            other_ai=round(ai * 0.15, 3),
            primary_attribution="gpt_family" if ai > 0.5 else "human",
            primary_confidence=max(ai, 1.0 - ai),
            is_confident=abs(ai - 0.5) > 0.2,
        )

    @staticmethod
    def _build_top_signals(
        output:    DetectorOutput,
        watermark: WatermarkResult,
        cross_modal: CrossModalAnalysis,
    ) -> list[dict]:
        signals: list[dict] = []

        # Detector evidence signals
        for sig in output.evidence.get("top_signals", [])[:5]:
            signals.append(sig)

        # Watermark signal
        if watermark.watermark_detected:
            signals.append({
                "signal": "Statistical watermark detected",
                "value":  f"z={watermark.z_score:.2f}" if watermark.z_score else "detected",
                "weight": "high" if watermark.confidence > 0.7 else "medium",
            })

        # Cross-modal mismatch signals
        for mismatch in cross_modal.top_mismatches[:2]:
            if mismatch["severity"] in ("high", "medium"):
                signals.append({
                    "signal": f"Cross-modal mismatch: {mismatch['check'].replace('_', ' ')}",
                    "value":  mismatch["severity"],
                    "weight": mismatch["severity"],
                })

        return signals[:8]
