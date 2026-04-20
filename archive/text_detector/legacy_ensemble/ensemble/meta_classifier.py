"""
Steps 20–22: Meta-Classifier + Platt Scaling + Isotonic Regression Calibration.

A gradient-boosted classifier (XGBoost) that combines the outputs of all four
detection layers into a single calibrated AI probability score.

Why not just average?
  - Each layer has different accuracy on different text types
  - Layers can have systematic biases (e.g. Layer 1 fails on short text)
  - The meta-classifier learns these relationships from data
  - XGBoost handles missing layer outputs (when a layer errors) gracefully

Feature vector (per document):
  Layer scores:   l1_score, l2_score, l3_score, l4_score              [4]
  Confidence:     l1_error (bool), ..., l4_error (bool)               [4]
  Evidence:       mean_perplexity, burstiness, ai_hedge_rate, ttr,    [~15]
                  sent_len_std, em_dash_rate, passive_rate, ...
  Text meta:      n_words, n_sentences, avg_sent_len                   [3]
  ─────────────────────────────────────────────────────────────────
  Total:          ~26 features

Calibration (Step 21):
  Raw XGBoost probabilities are well-calibrated in theory but in practice
  show miscalibration at the tails. We apply:
    1. Platt scaling   — logistic regression on val-set predictions
    2. Isotonic regression — non-parametric, handles non-monotone miscalibration
  Final score is the average of both calibrated outputs.

Step 22 (ECE validation) is in evaluation/calibration.py.
"""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from ..layers.base import LayerResult

log = structlog.get_logger(__name__)

FEATURE_NAMES = [
    # Layer scores
    "l1_score", "l2_score", "l3_score", "l4_score",
    # Layer availability (0 = layer errored, 1 = ok)
    "l1_ok", "l2_ok", "l3_ok", "l4_ok",
    # Layer 1 evidence
    "mean_perplexity", "std_perplexity", "burstiness", "low_ppl_fraction",
    # Layer 2 evidence
    "sent_len_mean", "sent_len_std", "ai_hedge_rate", "human_casual_rate",
    "type_token_ratio", "em_dash_rate", "comma_rate", "sent_initial_diversity",
    # Layer 3+4 evidence
    "doc_score_l3", "doc_score_l4",
    # Text metadata
    "n_words", "n_sentences", "avg_sent_len",
    # Ensemble disagreement
    "layer_score_std",
]

N_FEATURES = len(FEATURE_NAMES)


@dataclass
class EnsembleResult:
    """Final output from the full 4-layer ensemble + meta-classifier."""
    score: float                      # calibrated [0.0, 1.0] AI probability
    label: str                        # "AI" | "HUMAN" | "UNCERTAIN"
    confidence: float                 # [0.0, 1.0] — how confident the ensemble is
    layer_results: list[LayerResult]  # individual layer outputs
    feature_vector: list[float]       # the 26-dim feature vector
    evidence_summary: dict[str, Any]  # key signals for the UI evidence panel


def build_feature_vector(
    layer_results: list[LayerResult],
    text: str,
) -> list[float]:
    """
    Convert a list of LayerResults into the 26-dimensional feature vector
    that the meta-classifier consumes.
    """
    # Index results by name for reliable access
    by_name = {r.layer_name: r for r in layer_results}

    def get(name: str, key: str, default: float = 0.5) -> float:
        r = by_name.get(name)
        if r is None or r.error:
            return default
        return float(r.evidence.get(key, default))

    def score(name: str) -> float:
        r = by_name.get(name)
        return float(r.score) if r and not r.error else 0.5

    def ok(name: str) -> float:
        r = by_name.get(name)
        return 0.0 if (r is None or r.error) else 1.0

    sentences  = [s for s in text.split(".") if len(s.split()) >= 4]
    words      = text.split()
    n_words    = len(words)
    n_sents    = max(len(sentences), 1)
    avg_sent   = n_words / n_sents

    # Retrieve sent_len stats from Layer 2 evidence (nested dict)
    l2 = by_name.get("stylometry")
    sent_stats = (l2.evidence.get("sent_len_stats", {}) if l2 else {}) or {}

    layer_scores = [score("perplexity"), score("stylometry"),
                    score("transformer"), score("adversarial")]
    scores_ok    = [s for s, o in zip(layer_scores, [ok("perplexity"), ok("stylometry"),
                                                      ok("transformer"), ok("adversarial")])
                    if o > 0.5]
    std_scores   = 0.0
    if len(scores_ok) >= 2:
        mean_s = sum(scores_ok) / len(scores_ok)
        std_scores = math.sqrt(sum((s - mean_s) ** 2 for s in scores_ok) / len(scores_ok))

    vec = [
        # Layer scores
        layer_scores[0],
        layer_scores[1],
        layer_scores[2],
        layer_scores[3],
        # Layer availability
        ok("perplexity"),
        ok("stylometry"),
        ok("transformer"),
        ok("adversarial"),
        # Layer 1 evidence
        get("perplexity",  "mean_perplexity", 80.0),
        get("perplexity",  "std_perplexity",  30.0),
        get("perplexity",  "burstiness",      30.0),
        get("perplexity",  "low_ppl_fraction", 0.3),
        # Layer 2 evidence
        float(sent_stats.get("mean", avg_sent)),
        float(sent_stats.get("std",  5.0)),
        get("stylometry",  "ai_hedge_rate",      0.01),
        get("stylometry",  "human_casual_rate",  0.01),
        get("stylometry",  "type_token_ratio",   0.60),
        get("stylometry",  "em_dash_rate",       0.0),
        get("stylometry",  "comma_rate",         0.05),
        get("stylometry",  "sent_initial_diversity", 0.7),
        # Layer 3+4 evidence
        get("transformer", "doc_score", 0.5),
        get("adversarial", "doc_score", 0.5),
        # Text metadata
        float(n_words),
        float(n_sents),
        float(avg_sent),
        # Ensemble disagreement
        std_scores,
    ]

    assert len(vec) == N_FEATURES, f"Expected {N_FEATURES} features, got {len(vec)}"
    return vec


class MetaClassifier:
    """
    XGBoost meta-classifier with Platt + Isotonic calibration.

    Usage:
        # Training (run via training/train_meta.py):
        clf = MetaClassifier()
        clf.fit(X_train, y_train, X_val, y_val)
        clf.save(Path("ai/text_detector/checkpoints/meta"))

        # Inference:
        clf = MetaClassifier.load(Path("ai/text_detector/checkpoints/meta"))
        score = clf.predict(feature_vector)
    """

    def __init__(self) -> None:
        self._xgb: Any        = None
        self._platt: Any      = None
        self._isotonic: Any   = None
        self._is_fitted: bool = False

    # ── Training ──────────────────────────────────────────────

    def fit(
        self,
        X_train: Any,          # np.ndarray [n, N_FEATURES]
        y_train: Any,          # np.ndarray [n] binary labels
        X_val: Any,
        y_val: Any,
    ) -> None:
        """
        Step 20: Fit XGBoost meta-classifier.
        Step 21: Fit Platt scaling + isotonic regression on val set.
        """
        import numpy as np
        from xgboost import XGBClassifier              # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.isotonic import IsotonicRegression      # type: ignore

        log.info("fitting_meta_classifier", n_train=len(X_train), n_val=len(X_val))

        self._xgb = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            early_stopping_rounds=20,
            n_jobs=-1,
            random_state=42,
        )
        self._xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        # ── Step 21: Calibration ──────────────────────────────
        # Get raw probabilities on the held-out val set
        raw_probs = self._xgb.predict_proba(X_val)[:, 1]

        # Platt scaling: fit a logistic regression on log-odds of raw probs
        raw_probs_clipped = np.clip(raw_probs, 1e-6, 1 - 1e-6)
        log_odds = np.log(raw_probs_clipped / (1 - raw_probs_clipped)).reshape(-1, 1)
        self._platt = LogisticRegression(C=1e4)
        self._platt.fit(log_odds, y_val)
        log.info("platt_scaling_fitted")

        # Isotonic regression
        self._isotonic = IsotonicRegression(out_of_bounds="clip")
        self._isotonic.fit(raw_probs, y_val)
        log.info("isotonic_regression_fitted")

        self._is_fitted = True
        log.info("meta_classifier_fitted")

    # ── Inference ─────────────────────────────────────────────

    def predict(self, feature_vector: list[float]) -> float:
        """
        Return calibrated AI probability for a single feature vector.
        Averages Platt and isotonic calibrated outputs per the roadmap.
        """
        import numpy as np

        if not self._is_fitted:
            raise RuntimeError("MetaClassifier is not fitted. Load a checkpoint first.")

        X = np.array([feature_vector], dtype=np.float32)

        raw_prob = float(self._xgb.predict_proba(X)[0, 1])

        if self._platt and self._isotonic:
            raw_clipped = max(1e-6, min(1 - 1e-6, raw_prob))
            log_odds    = math.log(raw_clipped / (1 - raw_clipped))
            platt_prob  = float(self._platt.predict_proba([[log_odds]])[0, 1])
            iso_prob    = float(self._isotonic.predict([raw_prob])[0])
            calibrated  = (platt_prob + iso_prob) / 2.0
        else:
            calibrated = raw_prob

        return float(np.clip(calibrated, 0.01, 0.99))

    # ── Persistence ───────────────────────────────────────────

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "xgb.pkl").open("wb") as f:
            pickle.dump(self._xgb, f)
        with (directory / "platt.pkl").open("wb") as f:
            pickle.dump(self._platt, f)
        with (directory / "isotonic.pkl").open("wb") as f:
            pickle.dump(self._isotonic, f)
        meta = {"n_features": N_FEATURES, "feature_names": FEATURE_NAMES}
        with (directory / "meta.json").open("w") as f:
            json.dump(meta, f, indent=2)
        log.info("meta_classifier_saved", directory=str(directory))

    @classmethod
    def load(cls, directory: Path) -> "MetaClassifier":
        obj = cls()
        try:
            with (directory / "xgb.pkl").open("rb") as f:
                obj._xgb = pickle.load(f)
            with (directory / "platt.pkl").open("rb") as f:
                obj._platt = pickle.load(f)
            with (directory / "isotonic.pkl").open("rb") as f:
                obj._isotonic = pickle.load(f)
            obj._is_fitted = True
            log.info("meta_classifier_loaded", directory=str(directory))
        except Exception as exc:
            log.error(
                "meta_classifier_load_failed",
                directory=str(directory),
                error=str(exc),
            )
            # Return unfitted instance — TextDetector.analyze() will use
            # heuristic weighted average fallback instead of crashing.
            obj._is_fitted = False
        return obj
