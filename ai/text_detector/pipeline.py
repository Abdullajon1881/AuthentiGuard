"""
AuthentiGuard text detection — single entry point.

All inference goes through this module. Callers should use:

    from ai.text_detector.pipeline import analyze, MODEL_VERSION
    result = analyze("Some text to classify...")

`analyze()` lazily constructs and warms a process-local TextDetector
singleton on the first call. Subsequent calls reuse the loaded models
(~200–500ms per document on CPU).

Checkpoints are resolved automatically from `ai/text_detector/checkpoints/`:
  - L3 DeBERTa fine-tuned weights: checkpoints/transformer_v3_hard/phase1/
  - Stage 2 LR meta:               checkpoints/meta_classifier.joblib
  - Isotonic calibrator bundle:    checkpoints/meta_calibrator.joblib

Override checkpoint paths for tests / evaluation via `build_detector()`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .detector import MODEL_VERSION, TextDetector
from .meta import EnsembleResult

# ── Checkpoint resolution ──────────────────────────────────────────────────

_CHECKPOINT_ROOT = Path(__file__).resolve().parent / "checkpoints"
_DEFAULT_TRANSFORMER_CKPT = _CHECKPOINT_ROOT / "transformer_v3_hard" / "phase1"


_DETECTOR: TextDetector | None = None


def build_detector(
    transformer_checkpoint: Path | None = None,
    device: str | None = None,
) -> TextDetector:
    """
    Construct and warm a TextDetector. Prefer `analyze()` — this is here for
    callers that need an explicit instance (tests, batch evaluation, workers
    that want to control lifecycle).
    """
    ckpt = transformer_checkpoint or (
        _DEFAULT_TRANSFORMER_CKPT if _DEFAULT_TRANSFORMER_CKPT.exists() else None
    )
    det = TextDetector(transformer_checkpoint=ckpt, device=device)
    det.load_models()
    return det


def _get_detector() -> TextDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = build_detector()
    return _DETECTOR


def analyze(text: str) -> EnsembleResult:
    """
    Run full ensemble detection on `text`.

    Returns an EnsembleResult with:
      score               calibrated [0, 1] AI probability
      label               "AI" | "HUMAN" | "UNCERTAIN"
      confidence          [0, 1]
      layer_results       per-layer raw outputs
      feature_vector      26-dim meta-classifier features
      evidence_summary    UI-facing evidence + product block
    """
    return _get_detector().analyze(text)


def analyze_to_dict(text: str) -> dict[str, Any]:
    """Thin dict adapter for callers that don't want to import EnsembleResult."""
    r = analyze(text)
    return {
        "score": r.score,
        "label": r.label,
        "confidence": r.confidence,
        "evidence_summary": r.evidence_summary,
        "model_version": MODEL_VERSION,
    }


__all__ = [
    "analyze",
    "analyze_to_dict",
    "build_detector",
    "MODEL_VERSION",
    "EnsembleResult",
    "TextDetector",
]
