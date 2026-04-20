"""
Layer 3 — Semantic Classifier (DeBERTa-v3-small).

Compact transformer for AI text detection. Uses sequence classification
on top of a fine-tuned DeBERTa encoder. Sliding window for long texts,
per-sentence scoring for evidence.

Model: microsoft/deberta-v3-small (~44M params, ~180MB on disk)
Inference: ~150-300ms on CPU per document

This module is inference-only. Training is handled by
training/train_transformer.py.

IMPORTANT: This layer requires a fine-tuned checkpoint to produce
meaningful scores. A pretrained model with random classification head
will output near-random probabilities. The pretrained model at
/models/deberta_v3_small/ is the starting point for fine-tuning.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import structlog

from .base import BaseDetectionLayer, LayerResult, SentenceScore

log = structlog.get_logger(__name__)

MODEL_NAME = "microsoft/deberta-v3-small"
MAX_LENGTH = 512
STRIDE = 128
BATCH_SIZE = 8


def _split_sentences(text: str) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sents if len(s.split()) >= 4]


class SemanticLayer(BaseDetectionLayer):
    """
    Layer 3: Fine-tuned DeBERTa-v3-small classifier.

    Loads a fine-tuned checkpoint for AI text detection.
    Produces document-level and per-sentence scores.

    Uses layer_name="transformer" for schema compatibility with
    LayerScoresSchema and evidence_summary in TextDetector.
    """

    name = "transformer"

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        device: str | None = None,
    ) -> None:
        self._checkpoint = checkpoint_path
        self._device_str = device
        self._model: Any = None
        self._tokenizer: Any = None

    def load_model(self) -> None:
        import torch
        from transformers import (  # type: ignore
            AutoTokenizer,
            AutoModelForSequenceClassification,
        )

        device = self._device_str or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(device)

        load_path = str(self._checkpoint) if self._checkpoint else MODEL_NAME
        log.info("loading_semantic_layer", model=MODEL_NAME, path=load_path, device=device)

        self._tokenizer = AutoTokenizer.from_pretrained(load_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            load_path,
            num_labels=2,
            ignore_mismatched_sizes=True,
        )
        self._model.eval()
        self._model.to(self._device)
        log.info("semantic_layer_loaded", params="~44M")

    def _score_chunk(self, text: str) -> float:
        """Score a single text chunk (<= MAX_LENGTH tokens). Returns AI probability."""
        import torch

        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self._model(**enc).logits  # [1, 2]

        # label 1 = AI, label 0 = human
        probs = torch.softmax(logits, dim=-1)
        return float(probs[0, 1].item())

    def _sliding_window_score(self, text: str) -> float:
        """Score long texts via overlapping windows, averaged."""
        tokens = self._tokenizer(text, return_tensors="pt", add_special_tokens=False)
        ids = tokens["input_ids"][0]
        total = len(ids)

        if total <= MAX_LENGTH - 2:
            return self._score_chunk(text)

        scores: list[float] = []
        step = MAX_LENGTH - STRIDE - 2

        for start in range(0, total, step):
            end = min(start + MAX_LENGTH - 2, total)
            chunk_ids = ids[start:end]
            chunk_text = self._tokenizer.decode(chunk_ids, skip_special_tokens=True)
            if len(chunk_text.split()) < 10:
                continue
            scores.append(self._score_chunk(chunk_text))
            if end >= total:
                break

        return sum(scores) / len(scores) if scores else 0.5

    def _score_sentences(self, sentences: list[str]) -> list[SentenceScore]:
        """Score each sentence independently for per-sentence evidence."""
        results: list[SentenceScore] = []
        for i in range(0, len(sentences), BATCH_SIZE):
            batch = sentences[i : i + BATCH_SIZE]
            for sent in batch:
                if len(sent.split()) < 4:
                    continue
                try:
                    score = self._score_chunk(sent)
                    results.append(SentenceScore(
                        text=sent,
                        score=round(score, 4),
                        evidence={"semantic_ai_prob": round(score, 4)},
                    ))
                except Exception:  # noqa: BLE001
                    log.warning("semantic_sentence_score_failed", sentence=sent[:50])
        return results

    def analyze(self, text: str) -> LayerResult:
        if self._model is None:
            raise RuntimeError("Call load_model() before analyze()")

        doc_score = self._sliding_window_score(text)

        sentences = _split_sentences(text)
        s_scores: list[SentenceScore] = []
        if len(sentences) > 1:
            s_scores = self._score_sentences(sentences)

        return LayerResult(
            layer_name=self.name,
            score=round(doc_score, 4),
            sentence_scores=s_scores,
            evidence={
                "model": MODEL_NAME,
                "doc_score": round(doc_score, 4),
                "n_sentences_scored": len(s_scores),
            },
        )
