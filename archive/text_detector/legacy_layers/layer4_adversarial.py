"""
Step 19: Layer 4 — Adversarial Detector.

A separate transformer fine-tuned EXCLUSIVELY on adversarially attacked samples:
paraphrased AI text, back-translated AI text, grammar-corrected AI text,
and mixed human-AI text.

Layers 1–3 degrade significantly when text is paraphrased or translated.
Layer 4 is designed to catch exactly those cases.

Architecture: same as Layer 3 (DeBERTa-v3) but trained on the
datasets/adversarial/ split rather than the clean train split.
This specialisation is what gives it robustness that Layers 1–3 lack.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import structlog

from .base import BaseDetectionLayer, LayerResult, SentenceScore
from .layer3_transformer import TransformerLayer

log = structlog.get_logger(__name__)

DEFAULT_MODEL_NAME = "microsoft/deberta-v3-base"


def _split_sentences(text: str) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sents if len(s.split()) >= 4]


class AdversarialLayer(BaseDetectionLayer):
    """
    Layer 4: Adversarial detector.

    Identical inference interface to TransformerLayer (Layer 3),
    but fine-tuned on adversarial samples only. This means it learns
    residual signals that survive paraphrasing and translation:
      - Deep syntactic patterns rather than surface tokens
      - Structural rhythm (how clauses are chained)
      - Semantic coherence patterns unique to each model family
      - Length and clause distribution patterns

    The model is trained in two phases:
      Phase A — warm up on clean AI vs human data (same as Layer 3)
      Phase B — fine-tune on adversarial samples only, smaller LR

    This two-phase approach avoids catastrophic forgetting.
    """

    name = "adversarial"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        checkpoint_path: Path | None = None,
        device: str | None = None,
    ) -> None:
        # Delegate to a TransformerLayer instance — same architecture, different weights
        self._inner = TransformerLayer(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        # Override the layer name so the meta-classifier can distinguish them
        self._inner.name = self.name

    def load_model(self) -> None:
        self._inner.load_model()

    def analyze(self, text: str) -> LayerResult:
        result = self._inner.analyze(text)
        # Re-stamp the layer name (inner uses its own name)
        return LayerResult(
            layer_name=self.name,
            score=result.score,
            sentence_scores=result.sentence_scores,
            evidence={**result.evidence, "layer": "adversarial"},
        )

    # ── Attack-specific feature augmentation ──────────────────

    def _backtranslation_indicators(self, text: str) -> dict[str, float]:
        """
        Heuristic features that survive back-translation.
        Back-translated AI text has unnatural clause ordering and
        occasionally awkward idiomatic phrases.
        """
        words = text.lower().split()
        n = max(len(words), 1)

        # Passive voice approximation
        passive_count = sum(
            1 for i in range(1, len(words))
            if words[i - 1] in {"was", "were", "is", "are", "been", "being"}
            and words[i].endswith("ed")
        )

        # Coordinating conjunctions at sentence start (AI pattern)
        sentences = _split_sentences(text)
        coord_start = sum(
            1 for s in sentences
            if s.lower().startswith(("furthermore", "moreover", "additionally",
                                      "consequently", "therefore", "however"))
        )

        return {
            "passive_rate":       round(passive_count / n, 4),
            "coord_start_rate":   round(coord_start / max(len(sentences), 1), 4),
        }

    def analyze_with_augmentation(self, text: str) -> LayerResult:
        """
        Extended analysis that adds backtranslation heuristics on top
        of the transformer score. Used when the upstream pipeline flags
        a text as a likely backtranslation attack.
        """
        result = self.analyze(text)
        aug_features = self._backtranslation_indicators(text)

        # Small adjustment based on augmented features
        adjustment = (
            aug_features["passive_rate"] * 0.5
            + aug_features["coord_start_rate"] * 0.3
        )
        adjusted_score = min(0.99, result.score + adjustment * (1.0 - result.score))

        return LayerResult(
            layer_name=self.name,
            score=round(adjusted_score, 4),
            sentence_scores=result.sentence_scores,
            evidence={**result.evidence, **aug_features, "augmented": True},
        )
