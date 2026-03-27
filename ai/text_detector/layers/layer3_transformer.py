"""
Step 18: Layer 3 — Transformer Classifier.

Fine-tunes DeBERTa-v3-base (or RoBERTa-large) on the human vs. AI text
classification task. Uses sequence classification head on top of the
pre-trained encoder.

Inference: sliding window over long texts, scores are averaged across windows.
Produces per-sentence scores by running each sentence independently.

Training is handled by training/train_transformer.py — this module is
inference-only.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import structlog

from .base import BaseDetectionLayer, LayerResult, SentenceScore

log = structlog.get_logger(__name__)

DEFAULT_MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 512
STRIDE     = 128    # overlap between windows for long texts
BATCH_SIZE = 8


def _split_sentences(text: str) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sents if len(s.split()) >= 4]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class TransformerLayer(BaseDetectionLayer):
    """
    Layer 3: Fine-tuned transformer classifier (DeBERTa-v3 or RoBERTa-large).

    Supports:
      - Pre-trained HuggingFace checkpoint (before fine-tuning)
      - Fine-tuned local checkpoint (after training/train_transformer.py)
      - Sliding window inference for texts exceeding MAX_LENGTH tokens
    """

    name = "transformer"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        checkpoint_path: Path | None = None,
        device: str | None = None,
    ) -> None:
        self._model_name    = model_name
        self._checkpoint    = checkpoint_path
        self._device_str    = device
        self._model: Any    = None
        self._tokenizer: Any = None

    def load_model(self) -> None:
        import torch
        from transformers import (  # type: ignore
            AutoTokenizer,
            AutoModelForSequenceClassification,
        )

        device = self._device_str or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(device)

        load_path = str(self._checkpoint) if self._checkpoint else self._model_name
        log.info("loading_transformer", path=load_path, device=device)

        self._tokenizer = AutoTokenizer.from_pretrained(load_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            load_path,
            num_labels=2,
            ignore_mismatched_sizes=True,   # safe for zero-shot pre-trained loading
        )
        self._model.eval()
        self._model.to(self._device)
        log.info("transformer_loaded")

    def _score_chunk(self, text: str) -> float:
        """Score a single text chunk (≤ MAX_LENGTH tokens). Returns AI probability."""
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
            logits = self._model(**enc).logits  # shape: [1, 2]

        # label 1 = AI, label 0 = human (as per our training convention)
        probs = torch.softmax(logits, dim=-1)
        return float(probs[0, 1].item())

    def _sliding_window_score(self, text: str) -> float:
        """
        For long texts, tokenize fully then score overlapping windows.
        Average the window scores (weighted by token count).
        """
        tokens = self._tokenizer(text, return_tensors="pt", add_special_tokens=False)
        ids    = tokens["input_ids"][0]
        total  = len(ids)

        if total <= MAX_LENGTH - 2:
            return self._score_chunk(text)

        # Slide a window across the full token sequence
        scores: list[float] = []
        step = MAX_LENGTH - STRIDE - 2   # -2 for [CLS]/[SEP]

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
        # Batch sentences for efficiency
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
                        evidence={"transformer_ai_prob": round(score, 4)},
                    ))
                except Exception as exc:  # noqa: BLE001
                    log.warning("sentence_score_failed", error=str(exc))
        return results

    def analyze(self, text: str) -> LayerResult:
        if self._model is None:
            raise RuntimeError("Call load_model() before analyze()")

        # Document-level score via sliding window
        doc_score = self._sliding_window_score(text)

        # Per-sentence scores (only for texts with multiple sentences)
        sentences = _split_sentences(text)
        s_scores: list[SentenceScore] = []
        if len(sentences) > 1:
            s_scores = self._score_sentences(sentences)

        return LayerResult(
            layer_name=self.name,
            score=round(doc_score, 4),
            sentence_scores=s_scores,
            evidence={
                "model":      self._model_name,
                "n_windows":  max(1, len(text.split()) // (MAX_LENGTH // 2)),
                "doc_score":  round(doc_score, 4),
            },
        )
