"""
AuthentiGuard text detection layers — consolidated module.

Three active layers:
    L1  PerplexityLayer   GPT-2 log-probability / burstiness        (~50ms CPU)
    L2  StylometryLayer   Lexical + syntactic style features         (~5ms CPU)
    L3  SemanticLayer     Fine-tuned DeBERTa-v3-small classifier     (~200ms CPU)

All layers implement the same BaseDetectionLayer contract. Each returns a
LayerResult with a calibrated [0, 1] AI probability, optional per-sentence
scores, and a dict of raw signals used by the meta-classifier.

Previously split across layers/{base,layer1_perplexity,layer2_stylometry,
layer3_semantic}.py. The legacy L4 adversarial layer has never been fitted
in production and has been moved to archive/text_detector/legacy_layers/.
"""

from __future__ import annotations

import math
import re
import string  # noqa: F401  (kept for downstream callers that imported it)
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property  # noqa: F401
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# BASE TYPES
# ═══════════════════════════════════════════════════════════════════════════


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
    error       — set if the layer failed gracefully; score will be 0.5
    """
    layer_name: str
    score: float
    sentence_scores: list[SentenceScore] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class BaseDetectionLayer(ABC):
    """Abstract base for all detection layers."""

    name: str = "base"

    def load_model(self) -> None:
        """Load model weights. Called once at worker startup."""

    @abstractmethod
    def analyze(self, text: str) -> LayerResult:
        """Run detection on `text` and return a LayerResult."""

    def analyze_safe(self, text: str) -> LayerResult:
        """analyze() wrapper that degrades to a neutral result on exceptions."""
        try:
            return self.analyze(text)
        except Exception as exc:  # noqa: BLE001
            return LayerResult(
                layer_name=self.name,
                score=0.5,
                error=str(exc),
            )


# ═══════════════════════════════════════════════════════════════════════════
# L1  PERPLEXITY   (GPT-2 reference language model)
# ═══════════════════════════════════════════════════════════════════════════
#
# Calibration constants measured on 40-sample calibration set
# (scripts/calibrate_perplexity.py, 2026-04-12):
#   Human text: mean ~85, std ~70, median ~69
#   AI text:    mean ~36, std ~16, median ~35

HUMAN_PPL_MEAN = 85.0
HUMAN_PPL_STD  = 70.0
AI_PPL_MEAN    = 36.0
AI_PPL_STD     = 16.0

LOW_PPL_THRESHOLD        = 42.0   # midpoint of AI mean (36) and AI p75 (47)
LOW_BURSTINESS_THRESHOLD = 37.0   # midpoint of AI mean (9.6) and human mean (63.6)

GPT2_MODEL_NAME = "gpt2"


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if len(s.split()) >= 4]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class PerplexityLayer(BaseDetectionLayer):
    """L1: GPT-2 perplexity + burstiness analysis."""

    name = "perplexity"

    def __init__(self, model_name: str = GPT2_MODEL_NAME, device: str | None = None) -> None:
        self._model_name = model_name
        self._device_str = device
        self._model: Any = None
        self._tokenizer: Any = None

    def load_model(self) -> None:
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # type: ignore

        device = self._device_str or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("loading_perplexity_model", model=self._model_name, device=device)

        self._tokenizer = GPT2TokenizerFast.from_pretrained(self._model_name)
        self._model = GPT2LMHeadModel.from_pretrained(self._model_name)
        self._model.eval()
        self._model.to(device)
        self._device = torch.device(device)
        log.info("perplexity_model_loaded")

    def _compute_sentence_perplexity(self, sentence: str) -> float | None:
        import torch

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Call load_model() before analyze()")

        tokens = self._tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        input_ids = tokens["input_ids"].to(self._device)

        if input_ids.shape[1] < 3:
            return None

        with torch.no_grad():
            outputs = self._model(input_ids, labels=input_ids)
            loss = outputs.loss.item()

        return math.exp(loss)

    def analyze(self, text: str) -> LayerResult:
        sentences = _split_sentences(text)
        if not sentences:
            sentences = [text]

        sentence_ppls: list[tuple[str, float]] = []
        for sent in sentences:
            ppl = self._compute_sentence_perplexity(sent)
            if ppl is not None:
                sentence_ppls.append((sent, ppl))

        if not sentence_ppls:
            return LayerResult(
                layer_name=self.name,
                score=0.5,
                evidence={"error": "no scorable sentences"},
            )

        ppls = [p for _, p in sentence_ppls]
        mean_ppl     = sum(ppls) / len(ppls)
        variance     = sum((p - mean_ppl) ** 2 for p in ppls) / len(ppls)
        std_ppl      = math.sqrt(variance)
        low_ppl_frac = sum(1 for p in ppls if p < LOW_PPL_THRESHOLD) / len(ppls)
        burstiness   = std_ppl

        # Normalise mean_ppl: how far from AI mean vs human mean?
        ppl_range  = HUMAN_PPL_MEAN - AI_PPL_MEAN
        ppl_signal = max(0.0, min(1.0, (HUMAN_PPL_MEAN - mean_ppl) / ppl_range))
        burst_signal = 1.0 - min(burstiness / (HUMAN_PPL_STD * 2), 1.0)

        # Weights from Fisher discriminant ratios (calibration):
        #   perplexity 0.685, burstiness 0.543, low_ppl_frac 1.042
        raw_score = 0.45 * ppl_signal + 0.25 * burst_signal + 0.30 * low_ppl_frac
        score = max(0.01, min(0.99, _sigmoid((raw_score - 0.5) * 8)))

        s_scores = [
            SentenceScore(
                text=sent,
                score=max(0.01, min(0.99, 1.0 - min(ppl / HUMAN_PPL_MEAN, 1.0))),
                evidence={"perplexity": round(ppl, 2)},
            )
            for sent, ppl in sentence_ppls
        ]

        return LayerResult(
            layer_name=self.name,
            score=round(score, 4),
            sentence_scores=s_scores,
            evidence={
                "mean_perplexity":    round(mean_ppl, 2),
                "std_perplexity":     round(std_ppl, 2),
                "burstiness":         round(burstiness, 2),
                "low_ppl_fraction":   round(low_ppl_frac, 3),
                "n_sentences_scored": len(ppls),
                "ppl_signal":         round(ppl_signal, 4),
                "burst_signal":       round(burst_signal, 4),
            },
        )


# ═══════════════════════════════════════════════════════════════════════════
# L2  STYLOMETRY   (lexical + syntactic features; optional spaCy)
# ═══════════════════════════════════════════════════════════════════════════

FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "not",
    "no", "nor", "so", "yet", "both", "either", "neither", "each",
    "that", "which", "who", "whom", "this", "these", "those", "it",
    "its", "they", "them", "their", "we", "our", "you", "your", "he",
    "she", "his", "her", "i", "my", "me",
}

AI_HEDGE_WORDS = {
    "furthermore", "moreover", "additionally", "consequently", "therefore",
    "nevertheless", "nonetheless", "however", "indeed", "certainly",
    "undoubtedly", "essentially", "fundamentally", "ultimately", "notably",
    "importantly", "significantly", "particularly", "specifically",
    "delve", "dive", "tapestry", "nuanced", "multifaceted", "comprehensive",
    "robust", "leverage", "utilize", "facilitate", "paradigm",
}

HUMAN_CASUAL_WORDS = {
    "actually", "basically", "kind of", "sort of", "you know", "i mean",
    "well", "anyway", "stuff", "thing", "things", "pretty", "really",
    "just", "like", "though", "although", "but then", "so",
}


def _get_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if len(s.split()) >= 3]


def _word_tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _descriptive_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "skew": 0.0}
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)
    skew = 0.0
    if std > 0 and n >= 3:
        skew = (sum((v - mean) ** 3 for v in values) / n) / (std ** 3)
    return {
        "mean":  round(mean, 3),
        "std":   round(std, 3),
        "min":   round(min(values), 1),
        "max":   round(max(values), 1),
        "skew":  round(skew, 3),
    }


class StylometryLayer(BaseDetectionLayer):
    """L2: Stylometric fingerprinting (lexical + optional spaCy POS/syntax)."""

    name = "stylometry"

    def __init__(self, use_spacy: bool = True) -> None:
        self._use_spacy = use_spacy
        self._nlp: Any = None

    def load_model(self) -> None:
        if not self._use_spacy:
            return
        try:
            import spacy  # type: ignore
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                log.warning(
                    "spacy_model_not_found",
                    hint="python -m spacy download en_core_web_sm",
                )
                self._nlp = None
        except ImportError:
            log.warning("spacy_not_installed")
            self._nlp = None

    def _sentence_length_features(self, sentences: list[str]) -> dict[str, Any]:
        lengths = [len(s.split()) for s in sentences]
        stats = _descriptive_stats([float(l) for l in lengths])
        return {"sent_len_stats": stats, "high_variance": stats["std"] > 8.0}

    def _punctuation_features(self, text: str) -> dict[str, Any]:
        word_count = max(len(text.split()), 1)
        return {
            "comma_rate":     text.count(",")   / word_count,
            "semicolon_rate": text.count(";")   / word_count,
            "colon_rate":     text.count(":")   / word_count,
            "em_dash_rate":   text.count("—")   / word_count,
            "hyphen_rate":    text.count("-")   / word_count,
            "exclaim_rate":   text.count("!")   / word_count,
            "question_rate":  text.count("?")   / word_count,
            "ellipsis_rate":  text.count("...") / word_count,
            "paren_rate":     text.count("(")   / word_count,
        }

    def _lexical_features(self, words: list[str]) -> dict[str, Any]:
        n = max(len(words), 1)
        unique = set(words)

        func_count   = sum(1 for w in words if w in FUNCTION_WORDS)
        hedge_count  = sum(1 for w in words if w in AI_HEDGE_WORDS)
        casual_count = sum(1 for w in words if w in HUMAN_CASUAL_WORDS)

        freq = Counter(words)
        top20 = sorted(freq.values(), reverse=True)[:20]
        freq_drop = 0.0
        if len(top20) >= 2 and top20[0] > 0:
            freq_drop = top20[-1] / top20[0]

        return {
            "type_token_ratio":   round(len(unique) / n, 4),
            "func_word_rate":     round(func_count / n, 4),
            "ai_hedge_rate":      round(hedge_count / n, 4),
            "human_casual_rate":  round(casual_count / n, 4),
            "freq_drop":          round(freq_drop, 4),
            "avg_word_length":    round(sum(len(w) for w in words) / n, 2),
        }

    def _pos_features(self, text: str) -> dict[str, Any]:
        if self._nlp is None:
            return {}
        doc = self._nlp(text[:5000])
        total = max(len(doc), 1)
        pos_counts: Counter = Counter(tok.pos_ for tok in doc)
        return {
            "noun_rate":  round(pos_counts.get("NOUN", 0) / total, 4),
            "verb_rate":  round(pos_counts.get("VERB", 0) / total, 4),
            "adj_rate":   round(pos_counts.get("ADJ",  0) / total, 4),
            "adv_rate":   round(pos_counts.get("ADV",  0) / total, 4),
            "conj_rate":  round(
                (pos_counts.get("CCONJ", 0) + pos_counts.get("SCONJ", 0)) / total, 4
            ),
        }

    def _syntax_features(self, text: str) -> dict[str, Any]:
        if self._nlp is None:
            return {}

        def _subtree_depth(token: Any) -> int:
            children = list(token.children)
            if not children:
                return 0
            return 1 + max(_subtree_depth(c) for c in children)

        doc = self._nlp(text[:3000])
        depths = [float(_subtree_depth(sent.root)) for sent in doc.sents]
        return {"tree_depth_stats": _descriptive_stats(depths)}

    def _sentence_diversity_features(self, sentences: list[str]) -> dict[str, Any]:
        if len(sentences) < 3:
            return {}
        first_words = [s.split()[0].lower() for s in sentences if s.split()]
        diversity = len(set(first_words)) / len(first_words)
        return {
            "sent_initial_diversity": round(diversity, 4),
            "low_diversity": diversity < 0.5,
        }

    def _compute_score(self, features: dict[str, Any]) -> float:
        signals: list[float] = []

        sent_stats = features.get("sent_len_stats", {})
        signals.append(max(0.0, 1.0 - sent_stats.get("std", 10.0) / 15.0))

        signals.append(min(features.get("ai_hedge_rate", 0.0) * 80.0, 1.0))
        signals.append(max(0.0, 1.0 - features.get("human_casual_rate", 0.0) * 50.0))
        signals.append(max(0.0, 1.0 - features.get("type_token_ratio", 0.7) / 0.7))
        signals.append(max(0.0, 1.0 - features.get("sent_initial_diversity", 0.7)))
        signals.append(min(features.get("em_dash_rate", 0.0) * 200.0, 1.0))
        signals.append(min(max(0.0, (features.get("comma_rate", 0.05) - 0.05) / 0.10), 1.0))

        if not signals:
            return 0.5
        return max(0.01, min(0.99, sum(signals) / len(signals)))

    def analyze(self, text: str) -> LayerResult:
        sentences = _get_sentences(text)
        words     = _word_tokenize(text)

        features: dict[str, Any] = {}
        features.update(self._sentence_length_features(sentences))
        features.update(self._punctuation_features(text))
        features.update(self._lexical_features(words))
        features.update(self._pos_features(text))
        features.update(self._syntax_features(text))
        features.update(self._sentence_diversity_features(sentences))

        score = self._compute_score(features)

        s_scores: list[SentenceScore] = []
        for sent in sentences:
            sent_words = _word_tokenize(sent)
            n = max(len(sent_words), 1)
            hedge  = sum(1 for w in sent_words if w in AI_HEDGE_WORDS)       / n
            casual = sum(1 for w in sent_words if w in HUMAN_CASUAL_WORDS)   / n
            s_score = min(0.99, max(0.01, 0.5 + hedge * 20.0 - casual * 10.0))
            s_scores.append(SentenceScore(
                text=sent,
                score=round(s_score, 4),
                evidence={"hedge_rate": round(hedge, 4), "casual_rate": round(casual, 4)},
            ))

        return LayerResult(
            layer_name=self.name,
            score=round(score, 4),
            sentence_scores=s_scores,
            evidence=features,
        )


# ═══════════════════════════════════════════════════════════════════════════
# L3  SEMANTIC   (fine-tuned DeBERTa-v3-small; layer_name="transformer")
# ═══════════════════════════════════════════════════════════════════════════
#
# Model:  microsoft/deberta-v3-small (~44M params, ~180MB)
# Infer:  ~150-300ms CPU per document (sliding-window for long inputs)
# Weights loaded from: checkpoints/transformer_v3_hard/phase1/ (see pipeline.py)

DEBERTA_MODEL_NAME = "microsoft/deberta-v3-small"
DEBERTA_MAX_LENGTH = 512
DEBERTA_STRIDE     = 128
DEBERTA_BATCH_SIZE = 8


class SemanticLayer(BaseDetectionLayer):
    """
    L3: Fine-tuned DeBERTa-v3-small classifier.

    layer_name is "transformer" (not "semantic") for schema compatibility with
    the meta-classifier feature vector and the UI evidence summary.
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

        load_path = str(self._checkpoint) if self._checkpoint else DEBERTA_MODEL_NAME
        log.info(
            "loading_semantic_layer",
            model=DEBERTA_MODEL_NAME,
            path=load_path,
            device=device,
        )

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
        import torch

        enc = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=DEBERTA_MAX_LENGTH,
            padding=True,
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self._model(**enc).logits

        probs = torch.softmax(logits, dim=-1)
        return float(probs[0, 1].item())

    def _sliding_window_score(self, text: str) -> float:
        tokens = self._tokenizer(text, return_tensors="pt", add_special_tokens=False)
        ids = tokens["input_ids"][0]
        total = len(ids)

        if total <= DEBERTA_MAX_LENGTH - 2:
            return self._score_chunk(text)

        scores: list[float] = []
        step = DEBERTA_MAX_LENGTH - DEBERTA_STRIDE - 2

        for start in range(0, total, step):
            end = min(start + DEBERTA_MAX_LENGTH - 2, total)
            chunk_ids = ids[start:end]
            chunk_text = self._tokenizer.decode(chunk_ids, skip_special_tokens=True)
            if len(chunk_text.split()) < 10:
                continue
            scores.append(self._score_chunk(chunk_text))
            if end >= total:
                break

        return sum(scores) / len(scores) if scores else 0.5

    def _score_sentences(self, sentences: list[str]) -> list[SentenceScore]:
        results: list[SentenceScore] = []
        for i in range(0, len(sentences), DEBERTA_BATCH_SIZE):
            batch = sentences[i : i + DEBERTA_BATCH_SIZE]
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
                "model": DEBERTA_MODEL_NAME,
                "doc_score": round(doc_score, 4),
                "n_sentences_scored": len(s_scores),
            },
        )


__all__ = [
    "BaseDetectionLayer",
    "LayerResult",
    "SentenceScore",
    "PerplexityLayer",
    "StylometryLayer",
    "SemanticLayer",
    "FUNCTION_WORDS",
    "AI_HEDGE_WORDS",
    "HUMAN_CASUAL_WORDS",
]
