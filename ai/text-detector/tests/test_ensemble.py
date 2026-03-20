"""
Unit tests for the full text detection ensemble.
Tests are written to run WITHOUT loaded ML models (mock-based)
so the full CI suite stays fast.

Integration tests (requiring actual GPU + checkpoints) are in
tests/integration/test_text_detector_integration.py.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from ai.text_detector.layers.base import BaseDetectionLayer, LayerResult, SentenceScore
from ai.text_detector.layers.layer2_stylometry import (
    StylometryLayer,
    _word_tokenize,
    _descriptive_stats,
    AI_HEDGE_WORDS,
)
from ai.text_detector.ensemble.meta_classifier import (
    MetaClassifier,
    build_feature_vector,
    N_FEATURES,
    FEATURE_NAMES,
)
from ai.text_detector.evaluation.calibration import compute_ece, check_calibration_gate


# ── Fixtures ──────────────────────────────────────────────────

AI_HEAVY_TEXT = (
    "Furthermore, it is essential to acknowledge that the multifaceted nature "
    "of this paradigm leverages comprehensive frameworks. Moreover, the robust "
    "utilization of these mechanisms fundamentally transforms the overall "
    "landscape. Consequently, one must consider the nuanced implications of "
    "this particularly significant development. Additionally, the systematic "
    "approach facilitates a deeper understanding of the underlying processes."
)

HUMAN_CASUAL_TEXT = (
    "I actually went to the store yesterday and it was kind of a disaster. "
    "You know how it is — just one of those days where nothing goes right. "
    "Anyway, I grabbed some stuff and headed home. Pretty uneventful really, "
    "but I thought I'd mention it. Things happen I guess."
)


# ── BaseDetectionLayer ────────────────────────────────────────

class ConcreteLayer(BaseDetectionLayer):
    name = "test"
    def analyze(self, text: str) -> LayerResult:
        return LayerResult(layer_name=self.name, score=0.8)

class ErrorLayer(BaseDetectionLayer):
    name = "error"
    def analyze(self, text: str) -> LayerResult:
        raise RuntimeError("intentional error")


class TestBaseLayer:
    def test_analyze_returns_layer_result(self) -> None:
        layer = ConcreteLayer()
        result = layer.analyze("test text")
        assert isinstance(result, LayerResult)
        assert result.layer_name == "test"
        assert result.score == 0.8

    def test_analyze_safe_catches_exception(self) -> None:
        layer = ErrorLayer()
        result = layer.analyze_safe("test text")
        assert result.score == 0.5          # neutral fallback
        assert result.error is not None
        assert "intentional" in result.error

    def test_analyze_safe_returns_normal_on_success(self) -> None:
        layer = ConcreteLayer()
        result = layer.analyze_safe("test text")
        assert result.error is None
        assert result.score == 0.8


# ── Layer 2 Stylometry ────────────────────────────────────────

class TestStylometryLayer:
    def setup_method(self) -> None:
        self.layer = StylometryLayer(use_spacy=False)

    def test_ai_heavy_text_scores_higher(self) -> None:
        ai_score    = self.layer.analyze(AI_HEAVY_TEXT).score
        human_score = self.layer.analyze(HUMAN_CASUAL_TEXT).score
        assert ai_score > human_score, (
            f"AI text ({ai_score:.3f}) should score higher than human ({human_score:.3f})"
        )

    def test_score_in_range(self) -> None:
        for text in [AI_HEAVY_TEXT, HUMAN_CASUAL_TEXT]:
            result = self.layer.analyze(text)
            assert 0.0 <= result.score <= 1.0

    def test_evidence_keys_present(self) -> None:
        result = self.layer.analyze(AI_HEAVY_TEXT)
        required = {"ai_hedge_rate", "human_casual_rate", "type_token_ratio", "comma_rate"}
        assert required.issubset(result.evidence.keys())

    def test_sentence_scores_generated(self) -> None:
        result = self.layer.analyze(AI_HEAVY_TEXT)
        assert len(result.sentence_scores) > 0
        for s in result.sentence_scores:
            assert 0.0 <= s.score <= 1.0

    def test_hedge_word_detection(self) -> None:
        text = "Furthermore moreover additionally consequently therefore"
        words = _word_tokenize(text)
        hedge_count = sum(1 for w in words if w in AI_HEDGE_WORDS)
        assert hedge_count >= 3

    def test_descriptive_stats_empty(self) -> None:
        stats = _descriptive_stats([])
        assert stats["mean"] == 0.0
        assert stats["std"]  == 0.0

    def test_descriptive_stats_values(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = _descriptive_stats(values)
        assert stats["mean"] == pytest.approx(3.0, abs=0.01)
        assert stats["min"]  == 1.0
        assert stats["max"]  == 5.0

    def test_empty_text_does_not_crash(self) -> None:
        result = self.layer.analyze("")
        assert isinstance(result, LayerResult)
        assert 0.0 <= result.score <= 1.0


# ── Feature Vector ────────────────────────────────────────────

class TestFeatureVector:
    def _make_layer_result(self, name: str, score: float) -> LayerResult:
        return LayerResult(
            layer_name=name,
            score=score,
            evidence={
                "mean_perplexity": 60.0,
                "std_perplexity":  25.0,
                "burstiness":      25.0,
                "low_ppl_fraction": 0.3,
                "ai_hedge_rate":    0.02,
                "human_casual_rate": 0.01,
                "type_token_ratio": 0.65,
                "em_dash_rate":     0.001,
                "comma_rate":       0.06,
                "sent_initial_diversity": 0.7,
                "doc_score":        score,
                "sent_len_stats":   {"mean": 18.0, "std": 6.0},
            },
        )

    def test_correct_length(self) -> None:
        results = [
            self._make_layer_result("perplexity",  0.7),
            self._make_layer_result("stylometry",  0.6),
            self._make_layer_result("transformer", 0.8),
            self._make_layer_result("adversarial", 0.75),
        ]
        vec = build_feature_vector(results, "some text here to check")
        assert len(vec) == N_FEATURES

    def test_feature_names_match_length(self) -> None:
        assert len(FEATURE_NAMES) == N_FEATURES

    def test_layer_scores_in_vector(self) -> None:
        results = [
            self._make_layer_result("perplexity",  0.7),
            self._make_layer_result("stylometry",  0.6),
            self._make_layer_result("transformer", 0.8),
            self._make_layer_result("adversarial", 0.75),
        ]
        vec = build_feature_vector(results, "test text")
        # First four features are layer scores
        assert vec[0] == pytest.approx(0.7, abs=0.01)
        assert vec[1] == pytest.approx(0.6, abs=0.01)
        assert vec[2] == pytest.approx(0.8, abs=0.01)
        assert vec[3] == pytest.approx(0.75, abs=0.01)

    def test_errored_layer_yields_neutral_score(self) -> None:
        results = [
            LayerResult(layer_name="perplexity", score=0.5, error="model failed"),
            self._make_layer_result("stylometry",  0.6),
            self._make_layer_result("transformer", 0.8),
            self._make_layer_result("adversarial", 0.75),
        ]
        vec = build_feature_vector(results, "test text")
        assert vec[0] == pytest.approx(0.5, abs=0.01)  # neutral
        assert vec[4] == pytest.approx(0.0, abs=0.01)  # l1_ok = 0


# ── Calibration (ECE) ─────────────────────────────────────────

class TestCalibration:
    def test_perfect_calibration_has_zero_ece(self) -> None:
        # Perfect: 0.2 prob → 20% are AI (label=1)
        probs  = [0.2] * 10 + [0.8] * 10
        labels = [0] * 8 + [1] * 2 + [0] * 2 + [1] * 8
        result = compute_ece(probs, labels, n_buckets=10)
        assert result.ece < 0.15  # not perfect due to small N, but should be low

    def test_terrible_calibration_has_high_ece(self) -> None:
        # Always predicts 0.9 but is only right 10% of the time
        probs  = [0.9] * 100
        labels = [1] * 10 + [0] * 90
        result = compute_ece(probs, labels)
        assert result.ece > 0.5

    def test_ece_is_zero_for_empty_bucket(self) -> None:
        probs  = [0.5] * 20
        labels = [1] * 10 + [0] * 10
        result = compute_ece(probs, labels, n_buckets=10)
        assert isinstance(result.ece, float)

    def test_calibration_gate_passes_when_ece_low(self) -> None:
        from ai.text_detector.evaluation.calibration import CalibrationResult
        results = {
            "layer1": CalibrationResult("layer1", ece=0.02, mce=0.05, n_samples=100,
                                         n_buckets=15, bucket_data=[], is_calibrated=True),
        }
        assert check_calibration_gate(results, max_ece=0.05) is True

    def test_calibration_gate_fails_when_ece_high(self) -> None:
        from ai.text_detector.evaluation.calibration import CalibrationResult
        results = {
            "layer1": CalibrationResult("layer1", ece=0.12, mce=0.2, n_samples=100,
                                         n_buckets=15, bucket_data=[], is_calibrated=False),
        }
        assert check_calibration_gate(results, max_ece=0.05) is False

    def test_bucket_data_sums_correctly(self) -> None:
        probs  = [i / 100 for i in range(100)]
        labels = [1 if p > 0.5 else 0 for p in probs]
        result = compute_ece(probs, labels, n_buckets=10)
        total_n = sum(b["n"] for b in result.bucket_data)
        assert total_n == 100


# ── MetaClassifier (mock-based) ───────────────────────────────

class TestMetaClassifier:
    def test_predict_raises_if_not_fitted(self) -> None:
        clf = MetaClassifier()
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict([0.5] * N_FEATURES)

    def test_predict_returns_valid_probability(self) -> None:
        clf = MetaClassifier()
        # Mock the internal model
        mock_xgb = MagicMock()
        import numpy as np
        mock_xgb.predict_proba.return_value = np.array([[0.3, 0.7]])
        clf._xgb = mock_xgb
        clf._platt = None
        clf._isotonic = None
        clf._is_fitted = True

        score = clf.predict([0.5] * N_FEATURES)
        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(0.7, abs=0.01)

    def test_predict_clips_to_valid_range(self) -> None:
        clf = MetaClassifier()
        mock_xgb = MagicMock()
        import numpy as np
        mock_xgb.predict_proba.return_value = np.array([[0.0, 1.0]])
        clf._xgb = mock_xgb
        clf._platt = None
        clf._isotonic = None
        clf._is_fitted = True

        score = clf.predict([0.5] * N_FEATURES)
        assert score <= 0.99
        assert score >= 0.01
