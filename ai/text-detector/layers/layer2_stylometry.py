"""
Step 17: Layer 2 — Stylometric Fingerprinting.

Analyzes the statistical "style signature" of the text.
Humans write with irregular, idiosyncratic patterns.
AI models produce statistically smooth, consistent output.

Features extracted:
  - Sentence length distribution (mean, std, skew, kurtosis)
  - Punctuation distribution (comma rate, semicolon rate, em-dash rate, etc.)
  - POS tag distribution (noun/verb/adj/adv ratios)
  - Syntax tree depth statistics
  - Function word frequency (top 50 function words)
  - Type-token ratio (vocabulary richness)
  - Sentence-initial word diversity
  - Hedge word frequency ("perhaps", "might", "seems")
  - Author embedding cosine similarity to human/AI centroids

No model loading required — pure linguistic feature extraction via spaCy.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Any

import structlog

from .base import BaseDetectionLayer, LayerResult, SentenceScore

log = structlog.get_logger(__name__)

# ── Linguistic constants ───────────────────────────────────────

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

# AI text tends to overuse these "polished" hedges
AI_HEDGE_WORDS = {
    "furthermore", "moreover", "additionally", "consequently", "therefore",
    "nevertheless", "nonetheless", "however", "indeed", "certainly",
    "undoubtedly", "essentially", "fundamentally", "ultimately", "notably",
    "importantly", "significantly", "particularly", "specifically",
    "delve", "dive", "tapestry", "nuanced", "multifaceted", "comprehensive",
    "robust", "leverage", "utilize", "facilitate", "paradigm",
}

# Human casual markers
HUMAN_CASUAL_WORDS = {
    "actually", "basically", "kind of", "sort of", "you know", "i mean",
    "well", "anyway", "stuff", "thing", "things", "pretty", "really",
    "just", "like", "though", "although", "but then", "so",
}


def _get_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if len(s.split()) >= 3]


def _word_tokenize(text: str) -> list[str]:
    """Fast whitespace+punctuation tokenizer (no spaCy required)."""
    return re.findall(r"\b\w+\b", text.lower())


def _descriptive_stats(values: list[float]) -> dict[str, float]:
    """Mean, std, min, max, skewness of a list of floats."""
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
    """
    Layer 2: Stylometric fingerprinting.
    Pure Python + optional spaCy (for POS/syntax). Degrades gracefully
    to lexical features if spaCy is unavailable.
    """

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
                log.warning("spacy_model_not_found", hint="python -m spacy download en_core_web_sm")
                self._nlp = None
        except ImportError:
            log.warning("spacy_not_installed")
            self._nlp = None

    # ── Feature extractors ────────────────────────────────────

    def _sentence_length_features(self, sentences: list[str]) -> dict[str, Any]:
        lengths = [len(s.split()) for s in sentences]
        stats = _descriptive_stats([float(l) for l in lengths])
        # High std = human-like variation; low std = AI-like smoothness
        return {
            "sent_len_stats": stats,
            "high_variance": stats["std"] > 8.0,
        }

    def _punctuation_features(self, text: str) -> dict[str, Any]:
        word_count = max(len(text.split()), 1)
        return {
            "comma_rate":     text.count(",")  / word_count,
            "semicolon_rate": text.count(";")  / word_count,
            "colon_rate":     text.count(":")  / word_count,
            "em_dash_rate":   text.count("—")  / word_count,
            "hyphen_rate":    text.count("-")  / word_count,
            "exclaim_rate":   text.count("!")  / word_count,
            "question_rate":  text.count("?")  / word_count,
            "ellipsis_rate":  text.count("...") / word_count,
            "paren_rate":     text.count("(")  / word_count,
        }

    def _lexical_features(self, words: list[str]) -> dict[str, Any]:
        n = max(len(words), 1)
        unique = set(words)

        func_count  = sum(1 for w in words if w in FUNCTION_WORDS)
        hedge_count = sum(1 for w in words if w in AI_HEDGE_WORDS)
        casual_count= sum(1 for w in words if w in HUMAN_CASUAL_WORDS)

        # Zipf's law deviation: AI text often has unnaturally smooth rank-frequency
        freq = Counter(words)
        top20_freq = sorted(freq.values(), reverse=True)[:20]
        freq_drop = 0.0
        if len(top20_freq) >= 2 and top20_freq[0] > 0:
            freq_drop = top20_freq[-1] / top20_freq[0]  # low = sharp drop = natural

        return {
            "type_token_ratio":   round(len(unique) / n, 4),
            "func_word_rate":     round(func_count / n, 4),
            "ai_hedge_rate":      round(hedge_count / n, 4),
            "human_casual_rate":  round(casual_count / n, 4),
            "freq_drop":          round(freq_drop, 4),
            "avg_word_length":    round(sum(len(w) for w in words) / n, 2),
        }

    def _pos_features(self, text: str) -> dict[str, Any]:
        """POS distribution via spaCy (degrades gracefully)."""
        if self._nlp is None:
            return {}
        doc = self._nlp(text[:5000])  # limit to first 5K chars for speed
        total = max(len(doc), 1)
        pos_counts: Counter = Counter(tok.pos_ for tok in doc)
        return {
            "noun_rate":  round(pos_counts.get("NOUN", 0)  / total, 4),
            "verb_rate":  round(pos_counts.get("VERB", 0)  / total, 4),
            "adj_rate":   round(pos_counts.get("ADJ", 0)   / total, 4),
            "adv_rate":   round(pos_counts.get("ADV", 0)   / total, 4),
            "conj_rate":  round((pos_counts.get("CCONJ", 0) + pos_counts.get("SCONJ", 0)) / total, 4),
        }

    def _syntax_features(self, text: str) -> dict[str, Any]:
        """Parse tree depth statistics via spaCy."""
        if self._nlp is None:
            return {}

        def _subtree_depth(token: Any) -> int:
            children = list(token.children)
            if not children:
                return 0
            return 1 + max(_subtree_depth(c) for c in children)

        doc = self._nlp(text[:3000])
        depths = []
        for sent in doc.sents:
            root = sent.root
            depths.append(_subtree_depth(root))

        stats = _descriptive_stats([float(d) for d in depths])
        return {"tree_depth_stats": stats}

    def _sentence_diversity_features(self, sentences: list[str]) -> dict[str, Any]:
        """How diverse are sentence-initial words? AI tends to repeat patterns."""
        if len(sentences) < 3:
            return {}
        first_words = [s.split()[0].lower() for s in sentences if s.split()]
        diversity = len(set(first_words)) / len(first_words)
        return {
            "sent_initial_diversity": round(diversity, 4),
            "low_diversity": diversity < 0.5,
        }

    # ── Scoring ───────────────────────────────────────────────

    def _compute_score(self, features: dict[str, Any]) -> float:
        """
        Heuristic scoring function combining stylometric signals.
        Each signal contributes a partial AI-probability score.
        """
        signals: list[float] = []

        # 1. Sentence length variance — low std → AI
        sent_stats = features.get("sent_len_stats", {})
        std = sent_stats.get("std", 10.0)
        signals.append(max(0.0, 1.0 - std / 15.0))   # 15 = typical human std

        # 2. AI hedge words — high rate → AI
        hedge_rate = features.get("ai_hedge_rate", 0.0)
        signals.append(min(hedge_rate * 80.0, 1.0))   # 0.0125 → signal=1.0

        # 3. Human casual words — high rate → human
        casual_rate = features.get("human_casual_rate", 0.0)
        signals.append(max(0.0, 1.0 - casual_rate * 50.0))

        # 4. Type-token ratio — AI has narrower vocabulary
        ttr = features.get("type_token_ratio", 0.7)
        signals.append(max(0.0, 1.0 - ttr / 0.7))    # <0.5 → suspicious

        # 5. Sentence-initial diversity — low → AI
        sid = features.get("sent_initial_diversity", 0.7)
        signals.append(max(0.0, 1.0 - sid))

        # 6. Em-dash overuse — AI loves em-dashes
        em_rate = features.get("em_dash_rate", 0.0)
        signals.append(min(em_rate * 200.0, 1.0))

        # 7. Comma rate — AI tends to use more commas
        comma_rate = features.get("comma_rate", 0.05)
        signals.append(min(max(0.0, (comma_rate - 0.05) / 0.10), 1.0))

        if not signals:
            return 0.5

        raw = sum(signals) / len(signals)
        return max(0.01, min(0.99, raw))

    # ── Main ──────────────────────────────────────────────────

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

        # Per-sentence scores (simplified: score each by its own hedge/casual balance)
        s_scores: list[SentenceScore] = []
        for sent in sentences:
            sent_words = _word_tokenize(sent)
            n = max(len(sent_words), 1)
            hedge  = sum(1 for w in sent_words if w in AI_HEDGE_WORDS) / n
            casual = sum(1 for w in sent_words if w in HUMAN_CASUAL_WORDS) / n
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
