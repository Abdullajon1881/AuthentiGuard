"""
Step 30: Result Engine.
Combines AI detector outputs, metadata signals, and watermark signals
into a unified Authenticity Score. This is the final arbiter before
results are returned to users.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import structlog

log = structlog.get_logger(__name__)

Label = Literal["AI", "HUMAN", "UNCERTAIN"]


@dataclass
class AuthenticityResult:
    """Final unified output from the Result Engine."""
    authenticity_score: float     # [0.0, 1.0] — 1.0 = definitely AI
    confidence:         float     # [0.0, 1.0] — certainty of the score
    label:              Label
    ai_score:           float     # raw AI detection score
    metadata_adjustment: float    # how much metadata shifted the score
    watermark_detected: bool
    provenance_verified: bool
    evidence_summary:   dict[str, Any]
    model_attribution:  dict[str, float]


def compute_authenticity_score(
    ai_score: float,
    metadata_signals: dict[str, Any],
    content_type: str,
) -> AuthenticityResult:
    """
    Step 30 + 84: Combine AI detection + metadata into unified Authenticity Score.

    Weighting logic:
      - AI detection score: primary signal (80–100% weight)
      - EXIF / device fingerprint: adjusts score (±0.10 max)
      - Watermark detection: strong positive signal if detected (±0.15)
      - Provenance (C2PA): if verified human-origin, large negative adjustment

    Args:
        ai_score:         Calibrated [0,1] AI probability from the detector ensemble.
        metadata_signals: Output from run_metadata_extraction().
        content_type:     "text" | "image" | "video" | "audio" | "code"

    Returns:
        AuthenticityResult with the final score and all supporting signals.
    """
    score = float(ai_score)
    adjustment = 0.0

    # ── Watermark signal ──────────────────────────────────────
    watermark = metadata_signals.get("watermark", {})
    watermark_detected = bool(watermark.get("watermark_detected", False))
    watermark_confidence = float(watermark.get("confidence", 0.0))

    if watermark_detected and watermark_confidence > 0.7:
        # Strong watermark → strongly AI
        adjustment += 0.15 * watermark_confidence
        log.info("watermark_signal", confidence=watermark_confidence, adjustment=adjustment)

    # ── EXIF / device fingerprint ─────────────────────────────
    device = metadata_signals.get("device_fingerprint", {})
    suspicious = device.get("suspicious_signals", [])
    likely_ai_device = device.get("likely_ai_generated", False)
    likely_camera = device.get("likely_camera_capture", False)

    if likely_ai_device and content_type == "image":
        # Missing EXIF + AI software tag → push toward AI
        adjustment += 0.08 * min(len(suspicious), 3) / 3
    elif likely_camera:
        # Real camera EXIF → small push toward human
        adjustment -= 0.05

    # ── C2PA provenance ───────────────────────────────────────
    provenance = metadata_signals.get("provenance", {})
    provenance_verified = bool(provenance.get("c2pa_verified", False))
    if provenance_verified:
        # Cryptographically verified human-origin content
        adjustment -= 0.20

    # ── Combine ───────────────────────────────────────────────
    raw_combined = score + adjustment
    final_score  = max(0.01, min(0.99, raw_combined))

    # ── Label ─────────────────────────────────────────────────
    if final_score >= 0.75:
        label: Label = "AI"
    elif final_score <= 0.40:
        label = "HUMAN"
    else:
        label = "UNCERTAIN"

    # ── Confidence ────────────────────────────────────────────
    # High confidence when score is far from 0.5 AND signals agree
    base_confidence = abs(final_score - 0.5) * 2.0
    signal_agreement = 1.0 - abs(adjustment) * 2   # large adjustment = more uncertainty
    confidence = base_confidence * max(signal_agreement, 0.5)
    confidence = max(0.01, min(0.99, confidence))

    # ── Model attribution (placeholder — trained in Phase 10) ─
    model_attribution = _estimate_model_attribution(final_score, metadata_signals)

    evidence_summary = {
        "ai_detection_score":    round(score, 4),
        "metadata_adjustment":   round(adjustment, 4),
        "final_score":           round(final_score, 4),
        "watermark_detected":    watermark_detected,
        "provenance_verified":   provenance_verified,
        "suspicious_signals":    suspicious,
        "device_likely_camera":  likely_camera,
        "device_likely_ai":      likely_ai_device,
    }

    return AuthenticityResult(
        authenticity_score=round(final_score, 4),
        confidence=round(confidence, 4),
        label=label,
        ai_score=round(score, 4),
        metadata_adjustment=round(adjustment, 4),
        watermark_detected=watermark_detected,
        provenance_verified=provenance_verified,
        evidence_summary=evidence_summary,
        model_attribution=model_attribution,
    )


def _estimate_model_attribution(score: float, metadata: dict) -> dict[str, float]:
    """
    Estimate which AI model family produced the content.
    This is a placeholder — Phase 10 builds the real attribution classifier.
    Returns normalized percentages that sum to 1.0.
    """
    if score < 0.40:
        return {"human": 1.0, "gpt_family": 0.0, "claude_family": 0.0,
                "llama_family": 0.0, "other": 0.0}

    # Simple heuristic until the attribution classifier is trained
    human_prob = 1.0 - score
    ai_prob    = score

    return {
        "human":        round(human_prob, 3),
        "gpt_family":   round(ai_prob * 0.40, 3),
        "claude_family": round(ai_prob * 0.25, 3),
        "llama_family": round(ai_prob * 0.25, 3),
        "other":        round(ai_prob * 0.10, 3),
    }
