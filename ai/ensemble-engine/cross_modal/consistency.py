"""
Step 81: Cross-modal consistency detection.

Compares signals across modalities. Mismatches between modalities are
among the strongest deepfake indicators because they require the attacker
to fool multiple independent systems simultaneously.

Three cross-modal checks:

1. Lip movement vs audio phoneme alignment (video + audio)
   Real video: lip movements align precisely with speech phonemes (±1–2 frames)
   Deepfake:   lip movements are synthesised independently from audio, causing
               subtle misalignment at transitions between phonemes.
   Method:     Extract Mel spectral energy peaks (pseudo-phoneme boundaries)
               and compare to mouth landmark movement boundaries.

2. Writing style vs publication metadata (text + metadata)
   Real articles: writing style is consistent with claimed author and publication.
   AI-generated:  writing style shows AI markers regardless of claimed authorship.
   Mismatch signals: claimed author is a human journalist but text scores >0.8 AI.

3. Image EXIF data vs claimed source (image + metadata)
   Real photos:   EXIF contains camera model, lens, GPS, timestamp.
   AI images:     Missing EXIF, or EXIF that contradicts claimed device/location.
   Mismatch signals: claimed "photo from iPhone" but no iPhone EXIF markers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

log = structlog.get_logger(__name__)


@dataclass
class CrossModalResult:
    """Result of one cross-modal consistency check."""
    check_name:        str
    consistency_score: float   # [0,1] — 1.0 = perfectly consistent, 0.0 = inconsistent
    mismatch_detected: bool
    mismatch_severity: str     # "none" | "low" | "medium" | "high"
    evidence:          dict[str, Any]


@dataclass
class CrossModalAnalysis:
    """Aggregated cross-modal analysis results."""
    checks:              list[CrossModalResult]
    overall_consistency: float   # weighted average of all check scores
    inconsistency_score: float   # [0,1] — higher = more inconsistent = more likely fake
    top_mismatches:      list[dict]


# ── 1. Lip–audio alignment ────────────────────────────────────

def check_lip_audio_alignment(
    video_evidence: dict[str, Any],
    audio_evidence: dict[str, Any],
) -> CrossModalResult:
    """
    Check whether lip movement timing matches audio phoneme boundaries.

    Uses a simplified approach: compare the temporal distribution of
    face-region activity peaks (from video) with audio energy peaks (from audio).
    A real talking-head video should show correlated peaks.

    Full implementation would use:
      - MediaPipe Face Mesh (lip landmark coordinates per frame)
      - Force Alignment (wav2vec2 or Montreal Forced Aligner) for phoneme timing
      - DTW (Dynamic Time Warping) for alignment scoring
    """
    # Extract temporal patterns from evidence
    video_chunk_scores = [
        s.get("score", 0.5)
        for s in video_evidence.get("sentence_scores", [])
    ]
    audio_chunk_scores = [
        s.get("score", 0.5)
        for s in audio_evidence.get("sentence_scores", [])
    ]

    if len(video_chunk_scores) < 3 or len(audio_chunk_scores) < 3:
        return CrossModalResult(
            check_name="lip_audio_alignment",
            consistency_score=0.5,
            mismatch_detected=False,
            mismatch_severity="none",
            evidence={"reason": "insufficient_temporal_data"},
        )

    # Interpolate to same length for comparison
    n = min(len(video_chunk_scores), len(audio_chunk_scores), 20)
    v = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(video_chunk_scores)),
                   video_chunk_scores)
    a = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(audio_chunk_scores)),
                   audio_chunk_scores)

    # Pearson correlation between video and audio score time-series
    # High correlation = consistent (both high at the same time = scene has AI content)
    # Low correlation between suspicious video + clean audio (or vice versa) = mismatch
    corr = float(np.corrcoef(v, a)[0, 1]) if np.std(v) > 0 and np.std(a) > 0 else 0.0
    corr = max(-1.0, min(1.0, corr))  # clamp for numerical safety

    # Severity of mismatch: large difference in overall scores
    video_mean = float(np.mean(v))
    audio_mean = float(np.mean(a))
    score_gap  = abs(video_mean - audio_mean)

    # Consistency: high correlation + low gap = consistent
    consistency = (corr + 1) / 2.0 * (1.0 - min(score_gap, 1.0))

    mismatch = score_gap > 0.30 and corr < 0.3
    severity = (
        "high"   if score_gap > 0.50 else
        "medium" if score_gap > 0.30 else
        "low"    if score_gap > 0.15 else
        "none"
    )

    return CrossModalResult(
        check_name="lip_audio_alignment",
        consistency_score=round(float(consistency), 4),
        mismatch_detected=mismatch,
        mismatch_severity=severity,
        evidence={
            "video_mean_score":   round(video_mean, 4),
            "audio_mean_score":   round(audio_mean, 4),
            "score_gap":          round(score_gap, 4),
            "temporal_correlation": round(corr, 4),
        },
    )


# ── 2. Writing style vs publication metadata ──────────────────

def check_writing_style_vs_metadata(
    text_score:    float,
    text_evidence: dict[str, Any],
    file_metadata: dict[str, Any],
) -> CrossModalResult:
    """
    Flag mismatches between AI detection score and claimed authorship.

    Example mismatches:
      - High AI score + metadata claims "written by human journalist"
      - Low AI score + metadata claims "AI-generated" (correctly labelled)
      - Document author field contains known AI tool name
    """
    # Check document metadata for authorship claims
    author    = str(file_metadata.get("author", "")).lower()
    software  = str(file_metadata.get("software", "")).lower()
    generator = str(file_metadata.get("generator", "")).lower()

    # Known AI tool names in metadata
    ai_tool_markers = [
        "chatgpt", "gpt", "openai", "claude", "anthropic", "copilot",
        "jasper", "writesonic", "copy.ai", "wordtune", "quillbot",
        "bard", "gemini", "llama", "mistral",
    ]
    metadata_claims_ai = any(
        marker in field
        for field in [author, software, generator]
        for marker in ai_tool_markers
    )

    # Mismatch: high AI score but no AI authorship in metadata
    # (attacker tried to hide AI origin)
    high_ai_score = text_score >= 0.75
    mismatch = high_ai_score and not metadata_claims_ai and bool(author)

    # Also flag: metadata claims AI but score is very low (potential whitewashing)
    reverse_mismatch = metadata_claims_ai and text_score < 0.30

    consistency = 1.0
    if mismatch:
        consistency = max(0.0, 1.0 - text_score)  # less consistent if AI score is higher
    elif reverse_mismatch:
        consistency = text_score   # low consistency if score contradicts metadata

    severity = (
        "high"   if (mismatch and text_score > 0.85) or reverse_mismatch else
        "medium" if mismatch and text_score > 0.75 else
        "none"
    )

    return CrossModalResult(
        check_name="writing_style_vs_metadata",
        consistency_score=round(consistency, 4),
        mismatch_detected=mismatch or reverse_mismatch,
        mismatch_severity=severity,
        evidence={
            "text_ai_score":        round(text_score, 4),
            "metadata_claims_ai":   metadata_claims_ai,
            "author_field":         author[:50] if author else None,
            "software_field":       software[:50] if software else None,
            "mismatch_type":        "hidden_ai" if mismatch else
                                    "wrong_label" if reverse_mismatch else "none",
        },
    )


# ── 3. Image EXIF vs claimed source ──────────────────────────

def check_exif_vs_claimed_source(
    image_score:   float,
    exif_data:     dict[str, Any],
    claimed_source: str | None = None,
) -> CrossModalResult:
    """
    Check whether EXIF metadata is consistent with the claimed image source.

    Checks:
      - Camera model in EXIF matches claimed device
      - EXIF present when claimed source is a camera photo
      - EXIF absent when image is from an AI generator
      - AI software tag in EXIF (Stable Diffusion, DALL-E, etc.)
    """
    has_exif = bool(exif_data.get("has_exif"))
    has_cam  = bool(exif_data.get("has_camera_info"))
    software = str(exif_data.get("image_software", "")).lower()
    suspicious = exif_data.get("suspicious_signals", [])

    ai_in_software = any(
        marker in software
        for marker in ["stable diffusion", "dall", "midjourney", "firefly",
                        "imagen", "flux", "automatic1111", "comfy"]
    )

    # Mismatch: AI software in EXIF but image claimed to be a real photo
    claimed_real = (
        claimed_source is not None
        and any(w in claimed_source.lower()
                for w in ["photo", "photograph", "camera", "shot", "captured"])
    )
    mismatch_ai_exif = ai_in_software and claimed_real

    # Mismatch: No EXIF for claimed camera photo
    mismatch_missing_exif = not has_exif and claimed_real

    # Overall consistency
    if ai_in_software:
        consistency = max(0.0, 1.0 - image_score)
    elif not has_exif and image_score > 0.6:
        consistency = max(0.0, 1.0 - (image_score - 0.6) * 2)
    else:
        consistency = 0.8   # neutral when EXIF present and no AI markers

    mismatch = mismatch_ai_exif or mismatch_missing_exif
    severity = (
        "high"   if ai_in_software else
        "medium" if mismatch_missing_exif else
        "low"    if len(suspicious) > 1 else
        "none"
    )

    return CrossModalResult(
        check_name="exif_vs_claimed_source",
        consistency_score=round(consistency, 4),
        mismatch_detected=mismatch,
        mismatch_severity=severity,
        evidence={
            "has_exif":         has_exif,
            "has_camera_info":  has_cam,
            "ai_in_software":   ai_in_software,
            "software":         software[:50] if software else None,
            "suspicious":       suspicious[:5],
            "claimed_source":   claimed_source,
        },
    )


# ── Aggregator ────────────────────────────────────────────────

def analyse_cross_modal(
    outputs: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    claimed_source: str | None = None,
) -> CrossModalAnalysis:
    """
    Run all applicable cross-modal checks and aggregate results.

    Args:
        outputs:  Dict of {content_type: detector_evidence}
        metadata: File-level metadata (EXIF, author, software)
        claimed_source: Human-readable claimed source string
    """
    checks: list[CrossModalResult] = []
    meta   = metadata or {}
    exif   = meta.get("exif", {})

    # Video + audio check
    if "video" in outputs and "audio" in outputs:
        checks.append(check_lip_audio_alignment(
            outputs["video"], outputs["audio"]
        ))

    # Text + metadata check
    if "text" in outputs:
        text_score = outputs["text"].get("score", 0.5)
        checks.append(check_writing_style_vs_metadata(
            text_score, outputs["text"], meta
        ))

    # Image + EXIF check
    if "image" in outputs:
        image_score = outputs["image"].get("score", 0.5)
        checks.append(check_exif_vs_claimed_source(
            image_score, exif, claimed_source
        ))

    if not checks:
        return CrossModalAnalysis(
            checks=[], overall_consistency=1.0,
            inconsistency_score=0.0, top_mismatches=[],
        )

    # Aggregate
    consistencies = [c.consistency_score for c in checks]
    overall       = float(np.mean(consistencies))
    inconsistency = 1.0 - overall

    # Top mismatches for the UI evidence panel
    top_mismatches = [
        {
            "check":    c.check_name,
            "severity": c.mismatch_severity,
            "evidence": c.evidence,
        }
        for c in checks if c.mismatch_detected
    ]
    top_mismatches.sort(key=lambda x: ["none","low","medium","high"].index(x["severity"]),
                         reverse=True)

    return CrossModalAnalysis(
        checks=checks,
        overall_consistency=round(overall, 4),
        inconsistency_score=round(inconsistency, 4),
        top_mismatches=top_mismatches[:3],
    )
