"""
Full text detection ensemble — combines all four layers + meta-classifier.

This is the single entry point for text detection at inference time.
The backend AI Detection Service calls TextDetector.analyze(text).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from ..layers.base import LayerResult
from ..layers.layer1_perplexity import PerplexityLayer
from ..layers.layer2_stylometry import StylometryLayer
from ..layers.layer3_transformer import TransformerLayer
from ..layers.layer4_adversarial import AdversarialLayer
from .meta_classifier import MetaClassifier, EnsembleResult, build_feature_vector

log = structlog.get_logger(__name__)

# Score thresholds per roadmap
LABEL_THRESHOLDS = {
    "AI":        (0.75, 1.00),
    "UNCERTAIN": (0.40, 0.75),
    "HUMAN":     (0.00, 0.40),
}


def _score_to_label(score: float) -> str:
    for label, (low, high) in LABEL_THRESHOLDS.items():
        if low <= score < high:
            return label
    return "UNCERTAIN"


def _score_to_confidence(score: float) -> float:
    """Convert probability to confidence (distance from 0.5)."""
    return round(abs(score - 0.5) * 2, 4)


class TextDetector:
    """
    Main text detection ensemble.

    load_models() is called once at worker startup (heavy).
    analyze() is called per-request (fast, ~200–500ms).
    """

    def __init__(
        self,
        transformer_checkpoint: Path | None = None,
        adversarial_checkpoint: Path | None = None,
        meta_checkpoint: Path | None = None,
        device: str | None = None,
    ) -> None:
        self._layer1 = PerplexityLayer(device=device)
        self._layer2 = StylometryLayer(use_spacy=True)
        self._transformer_checkpoint = transformer_checkpoint
        self._adversarial_checkpoint = adversarial_checkpoint
        self._layer3: TransformerLayer | None = None
        self._layer4: AdversarialLayer | None = None
        self._meta   = MetaClassifier()
        self._meta_checkpoint = meta_checkpoint
        self._loaded = False
        self._active_layers: list[int] = []  # indices of active layers
        self._device = device

    def load_models(self) -> None:
        """Load all model weights. Call once at startup."""
        log.info("loading_text_detector_models")

        # L1 (perplexity) and L2 (stylometry) always load — no training needed
        self._layer1.load_model()
        self._layer2.load_model()
        self._active_layers = [0, 1]

        # L3 (transformer) — only load if fine-tuned checkpoint exists
        if self._transformer_checkpoint and self._transformer_checkpoint.exists():
            self._layer3 = TransformerLayer(
                checkpoint_path=self._transformer_checkpoint, device=self._device
            )
            self._layer3.load_model()
            self._active_layers.append(2)
            log.info("layer3_transformer_loaded")
        else:
            log.warning("layer3_skipped — no fine-tuned checkpoint, would produce random output")

        # L4 (adversarial) — only load if fine-tuned checkpoint exists
        if self._adversarial_checkpoint and self._adversarial_checkpoint.exists():
            self._layer4 = AdversarialLayer(
                checkpoint_path=self._adversarial_checkpoint, device=self._device
            )
            self._layer4.load_model()
            self._active_layers.append(3)
            log.info("layer4_adversarial_loaded")
        else:
            log.warning("layer4_skipped — no fine-tuned checkpoint, would produce random output")

        if self._meta_checkpoint and self._meta_checkpoint.exists():
            self._meta = MetaClassifier.load(self._meta_checkpoint)
        else:
            log.warning("meta_classifier_checkpoint_not_found — using heuristic fallback")

        active_count = len(self._active_layers)
        log.info("text_detector_ready", active_layers=active_count,
                 layers=self._active_layers)
        if active_count < 4:
            log.info("running_in_mvp_mode — L1+L2 only. Fine-tune L3/L4 for full accuracy.")

        self._loaded = True

    def analyze(self, text: str) -> EnsembleResult:
        """
        Run full 4-layer ensemble analysis on text.

        Returns EnsembleResult with:
          - score: calibrated [0, 1] AI probability
          - label: "AI" | "UNCERTAIN" | "HUMAN"
          - confidence: [0, 1]
          - layer_results: individual layer outputs
          - feature_vector: 26-dim vector
          - evidence_summary: key signals for the UI
        """
        if not self._loaded:
            raise RuntimeError("Call load_models() first.")

        # ── Run active layers (safe — errors return neutral 0.5) ──
        layer_results: list[LayerResult] = [
            self._layer1.analyze_safe(text),
            self._layer2.analyze_safe(text),
        ]
        # Only run L3/L4 if they have fine-tuned checkpoints loaded
        if self._layer3 is not None:
            layer_results.append(self._layer3.analyze_safe(text))
        if self._layer4 is not None:
            layer_results.append(self._layer4.analyze_safe(text))

        # ── Build feature vector ────────────────────────────────
        feature_vector = build_feature_vector(layer_results, text)

        # ── Meta-classifier or fallback ─────────────────────────
        if self._meta._is_fitted:
            score = self._meta.predict(feature_vector)
        else:
            # Fallback: weighted average based on active layers
            #   2 layers (L1+L2 MVP):    50/50 perplexity + stylometry
            #   3 layers (L1+L2+L3):     25/25/50
            #   4 layers (full ensemble): 20/20/35/25
            active_count = len(layer_results)
            if active_count == 2:
                weights = [0.50, 0.50]
            elif active_count == 3:
                weights = [0.25, 0.25, 0.50]
            else:
                weights = [0.20, 0.20, 0.35, 0.25]
            scores = [r.score for r in layer_results]
            score = sum(s * w for s, w in zip(scores, weights))
            score = max(0.01, min(0.99, score))

        label      = _score_to_label(score)
        confidence = _score_to_confidence(score)

        # ── Evidence summary for the UI ─────────────────────────
        by_name = {r.layer_name: r for r in layer_results}
        evidence_summary: dict[str, Any] = {
            "layer_scores": {
                "perplexity":   by_name.get("perplexity", LayerResult("perplexity", 0.5)).score,
                "stylometry":   by_name.get("stylometry", LayerResult("stylometry", 0.5)).score,
                "transformer":  by_name.get("transformer", LayerResult("transformer", 0.5)).score if "transformer" in by_name else None,
                "adversarial":  by_name.get("adversarial", LayerResult("adversarial", 0.5)).score if "adversarial" in by_name else None,
            },
            "layer_errors": {
                r.layer_name: r.error
                for r in layer_results if r.error
            },
            "top_signals": _build_top_signals(layer_results),
            "sentence_scores": _merge_sentence_scores(layer_results),
        }

        return EnsembleResult(
            score=round(score, 4),
            label=label,
            confidence=confidence,
            layer_results=layer_results,
            feature_vector=feature_vector,
            evidence_summary=evidence_summary,
        )


def _build_top_signals(layer_results: list[LayerResult]) -> list[dict]:
    """Extract the most important evidence signals for UI display."""
    signals = []
    by_name = {r.layer_name: r for r in layer_results}

    l1 = by_name.get("perplexity")
    if l1 and not l1.error:
        ppl = l1.evidence.get("mean_perplexity", 0)
        if ppl < 50:
            signals.append({"signal": "Very low perplexity", "value": f"{ppl:.0f}", "weight": "high"})
        burst = l1.evidence.get("burstiness", 100)
        if burst < 20:
            signals.append({"signal": "Uniform sentence flow (low burstiness)", "value": f"{burst:.1f}", "weight": "medium"})

    l2 = by_name.get("stylometry")
    if l2 and not l2.error:
        hedge = l2.evidence.get("ai_hedge_rate", 0)
        if hedge > 0.01:
            signals.append({"signal": "High AI hedge word usage", "value": f"{hedge:.3f}", "weight": "medium"})
        em_dash = l2.evidence.get("em_dash_rate", 0)
        if em_dash > 0.005:
            signals.append({"signal": "Excessive em-dash usage", "value": f"{em_dash:.3f}", "weight": "low"})
        ttr = l2.evidence.get("type_token_ratio", 1.0)
        if ttr < 0.45:
            signals.append({"signal": "Narrow vocabulary (low TTR)", "value": f"{ttr:.3f}", "weight": "medium"})

    return signals[:8]   # top 8 signals for the UI


def _merge_sentence_scores(layer_results: list[LayerResult]) -> list[dict]:
    """
    Merge per-sentence scores from all layers into a unified list.
    For each sentence, average available layer scores.
    """
    # Use Layer 3 (transformer) as the primary sentence scorer
    l3 = next((r for r in layer_results if r.layer_name == "transformer"), None)
    if not l3 or not l3.sentence_scores:
        # Fall back to Layer 1
        l1 = next((r for r in layer_results if r.layer_name == "perplexity"), None)
        if l1 and l1.sentence_scores:
            return [
                {"text": s.text, "score": s.score, "evidence": s.evidence}
                for s in l1.sentence_scores
            ]
        return []

    return [
        {"text": s.text, "score": s.score, "evidence": s.evidence}
        for s in l3.sentence_scores
    ]
