"""
Step 79: Integrate all five detectors into the unified processing queue
and result engine.

This module is the single dispatch layer between the Celery queue and
the individual detector pipelines. Every detection job — text, audio,
video, image, or code — flows through here.

Responsibilities:
  1. Receive a DetectionJob from the queue
  2. Route to the correct detector based on content_type
  3. Extract content from S3 or DB
  4. Run the detector
  5. Write DetectionResult to Postgres
  6. Trigger post-processing (metadata, watermark, cross-modal)
  7. Fire completion webhook

Design principles:
  - Each detector is independently deployable (microservice)
  - Detectors share no state — all communication via Postgres + Redis
  - Failures in one detector don't affect others
  - Every detector output uses the same DetectorOutput schema
  - The meta-classifier combines all outputs in a separate step
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import structlog

log = structlog.get_logger(__name__)

ContentType = Literal["text", "image", "video", "audio", "code"]


@dataclass
class DetectorOutput:
    """
    Canonical output schema from any detector.
    The meta-classifier and result engine work exclusively with this type.
    """
    content_type:     ContentType
    score:            float        # calibrated [0.0, 1.0] AI probability
    confidence:       float        # [0.0, 1.0]
    label:            str          # "AI" | "HUMAN" | "UNCERTAIN"
    layer_scores:     dict[str, float]   # per-layer/per-model scores
    sentence_scores:  list[dict]   # per-segment evidence (text: sentences, video: timestamps)
    flagged_segments: list[dict]   # for video/audio timeline view
    evidence:         dict[str, Any]
    processing_ms:    int
    error:            str | None   # set if detection failed


# ── Detector registry ──────────────────────────────────────────

class DetectorRegistry:
    """
    Lazy-loading registry of all detector instances.
    Each detector is loaded once per worker process on first use.
    """

    def __init__(self) -> None:
        self._detectors: dict[str, Any] = {}
        self._checkpoint_base = Path("ai")

    def get(self, content_type: ContentType) -> Any:
        if content_type not in self._detectors:
            self._detectors[content_type] = self._load(content_type)
        return self._detectors[content_type]

    def _load(self, content_type: ContentType) -> Any:
        log.info("loading_detector", content_type=content_type)

        if content_type in ("text", "code"):
            from ..text_detector.ensemble.text_detector import TextDetector  # type: ignore
            det = TextDetector(
                transformer_checkpoint=self._checkpoint_base / "text_detector/checkpoints/transformer/phase3",
                adversarial_checkpoint=self._checkpoint_base / "text_detector/checkpoints/adversarial/phase3",
                meta_checkpoint=self._checkpoint_base / "text_detector/checkpoints/meta",
            )
            det.load_models()
            return det

        elif content_type == "audio":
            from ..audio_detector.audio_detector import AudioDetector  # type: ignore
            det = AudioDetector(
                checkpoint_dir=self._checkpoint_base / "audio_detector/checkpoints/phase3",
                calibration_path=self._checkpoint_base / "audio_detector/checkpoints/calibration.pkl",
            )
            det.load_models()
            return det

        elif content_type == "video":
            from ..video_detector.video_detector import VideoDetector  # type: ignore
            det = VideoDetector(
                checkpoint_dir=self._checkpoint_base / "video_detector/checkpoints/phase3",
            )
            det.load_models()
            return det

        elif content_type == "image":
            from ..image_detector.image_detector import ImageDetector  # type: ignore
            det = ImageDetector(
                checkpoint_dir=self._checkpoint_base / "image_detector/checkpoints/phase3",
            )
            det.load_models()
            return det

        raise ValueError(f"Unknown content type: {content_type}")


# ── Dispatch ───────────────────────────────────────────────────

_registry = DetectorRegistry()


def dispatch(
    content_type: ContentType,
    content: bytes | str,
    filename: str = "input",
) -> DetectorOutput:
    """
    Route content to the correct detector and return a unified DetectorOutput.
    This is the single call site for all detection — called by every Celery worker.
    """
    t_start = int(time.time() * 1000)

    try:
        detector = _registry.get(content_type)
        result   = _run_detector(detector, content_type, content, filename)
        return _normalise(result, content_type, t_start)

    except Exception as exc:
        log.error("dispatch_failed", content_type=content_type, error=str(exc))
        return DetectorOutput(
            content_type=content_type,
            score=0.5, confidence=0.0, label="UNCERTAIN",
            layer_scores={}, sentence_scores=[], flagged_segments=[],
            evidence={"error": str(exc)},
            processing_ms=int(time.time() * 1000) - t_start,
            error=str(exc),
        )


def _run_detector(
    detector: Any,
    content_type: ContentType,
    content: bytes | str,
    filename: str,
) -> Any:
    """Call the detector's analyze() method with the right arguments."""
    if content_type in ("text", "code"):
        text = content if isinstance(content, str) else content.decode("utf-8", errors="replace")
        return detector.analyze(text)

    elif content_type in ("audio", "video", "image"):
        data = content if isinstance(content, bytes) else content.encode()
        return detector.analyze(data, filename)

    raise ValueError(f"No runner for content_type={content_type}")


def _normalise(result: Any, content_type: ContentType, t_start: int) -> DetectorOutput:
    """
    Convert any detector's native result type into the canonical DetectorOutput.
    Each detector has its own result dataclass; this adapter handles them all.
    """
    # All detector result types share these fields
    score      = float(getattr(result, "score",      0.5))
    label      = str(getattr(result,   "label",      "UNCERTAIN"))
    confidence = float(getattr(result, "confidence", 0.0))
    evidence   = dict(getattr(result,  "evidence",   {}))
    proc_ms    = int(getattr(result,   "processing_ms", int(time.time()*1000) - t_start))

    # Layer scores — different shapes per detector
    layer_scores: dict[str, float] = {}
    if hasattr(result, "layer_results"):       # text detector
        layer_scores = {r.layer_name: r.score for r in result.layer_results}
    elif hasattr(result, "model_scores"):       # image/code
        layer_scores = dict(result.model_scores)
    elif hasattr(result, "chunk_results"):      # audio
        layer_scores = {"audio_ensemble": score}
    elif hasattr(result, "frame_results"):      # video
        layer_scores = {"video_ensemble": score}

    # Sentence-level scores
    sentence_scores: list[dict] = []
    if hasattr(result, "evidence_summary"):
        sentence_scores = result.evidence_summary.get("sentence_scores", [])
    elif hasattr(result, "chunk_results"):
        sentence_scores = [
            {"text": f"{c.start_s:.1f}s–{c.end_s:.1f}s",
             "score": c.score, "evidence": {}}
            for c in result.chunk_results
        ]
    elif hasattr(result, "frame_results"):
        sentence_scores = [
            {"text": f"{f.timestamp_s:.1f}s",
             "score": f.frame_score, "evidence": {}}
            for f in result.frame_results
        ]

    # Flagged segments (video/audio timeline)
    flagged: list[dict] = []
    if hasattr(result, "flagged_segments"):
        flagged = list(result.flagged_segments)

    return DetectorOutput(
        content_type=content_type,
        score=round(score, 4),
        confidence=round(confidence, 4),
        label=label,
        layer_scores=layer_scores,
        sentence_scores=sentence_scores,
        flagged_segments=flagged,
        evidence=evidence,
        processing_ms=proc_ms,
        error=None,
    )
