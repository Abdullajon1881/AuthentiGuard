"""
Shared types and base class for all text detection layers.
Every layer returns a LayerResult — a calibrated probability + supporting evidence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SentenceScore:
    """Per-sentence score produced by a detection layer."""
    text: str
    score: float              # [0.0, 1.0] — probability of AI authorship
    evidence: dict[str, Any]  # layer-specific supporting signals


@dataclass
class LayerResult:
    """
    Output from a single detection layer.

    score       — calibrated [0.0, 1.0] AI probability for the full document
    sentence_scores — per-sentence breakdown (not all layers produce these)
    evidence    — dict of raw signals that explain the score
    layer_name  — identifier for logging and the meta-classifier feature vector
    """
    layer_name: str
    score: float
    sentence_scores: list[SentenceScore] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    error: str | None = None  # set if layer failed gracefully


class BaseDetectionLayer(ABC):
    """
    Abstract base for all four detection layers.
    Subclasses implement analyze() and optionally load_model().
    """

    name: str = "base"

    def load_model(self) -> None:
        """
        Load model weights. Called once at worker startup, not per-request.
        Override in layers that need heavy model loading.
        """

    @abstractmethod
    def analyze(self, text: str) -> LayerResult:
        """
        Run detection on the provided text.

        Args:
            text: Raw input text (pre-cleaned).

        Returns:
            LayerResult with score, evidence, and optional per-sentence scores.
        """

    def analyze_safe(self, text: str) -> LayerResult:
        """
        Wrapper that catches exceptions and returns a neutral result on failure
        rather than crashing the whole ensemble.
        """
        try:
            return self.analyze(text)
        except Exception as exc:  # noqa: BLE001
            return LayerResult(
                layer_name=self.name,
                score=0.5,   # neutral — don't bias the meta-classifier
                error=str(exc),
            )
