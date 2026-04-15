"""
Stage 3 — unit tests for monitoring + drift infrastructure.

Covers:
  - prediction_log.log_prediction():
      * writes to correct daily file
      * sample rate honored
      * disabled-mode no-op
      * text truncation
      * exception safety (broken result object must not raise)
      * layer extraction by name (errored layer -> None)
  - compute_daily_metrics.compute_metrics():
      * shape + math against known input
      * handles empty log
  - compute_drift.compute_psi():
      * identical histograms -> 0
      * disjoint histograms -> large positive number
      * classification thresholds
      * empty-bucket epsilon handling

All tests are hermetic — no network, no model loads, no filesystem
outside a pytest tmp_path. Runs in milliseconds.
"""

from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# scripts/ is not a package; add it explicitly so we can import the module files
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


# ── Helpers ─────────────────────────────────────────────────────────


def _fake_layer(name: str, score: float, error: str | None = None):
    return SimpleNamespace(layer_name=name, score=score, error=error)


def _fake_result(
    score: float = 0.75,
    label: str = "AI",
    confidence: float = 0.5,
    layers: list | None = None,
):
    if layers is None:
        layers = [
            _fake_layer("perplexity", 0.6),
            _fake_layer("stylometry", 0.55),
            _fake_layer("transformer", 0.8),
        ]
    return SimpleNamespace(
        score=score,
        label=label,
        confidence=confidence,
        layer_results=layers,
    )


# ── prediction_log ──────────────────────────────────────────────────


class TestPredictionLog:
    def test_writes_to_daily_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PREDICTION_LOG_DIR", str(tmp_path))
        monkeypatch.setenv("PREDICTION_SAMPLE_RATE", "0.0")
        monkeypatch.setenv("PREDICTION_LOG_ENABLED", "1")

        from app.observability import prediction_log as pl
        now = datetime(2026, 4, 16, 12, 0, 0, tzinfo=timezone.utc)
        pl.log_prediction(
            text="hello world " * 20,
            result=_fake_result(score=0.91, label="AI"),
            model_version="test-1.0",
            now=now,
        )

        expected = tmp_path / "2026-04-16.jsonl"
        assert expected.exists()
        lines = expected.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["timestamp"] == "2026-04-16T12:00:00+00:00"
        assert rec["model_version"] == "test-1.0"
        assert rec["input_length"] == len("hello world " * 20)
        assert rec["l1_score"] == pytest.approx(0.6)
        assert rec["l2_score"] == pytest.approx(0.55)
        assert rec["l3_score"] == pytest.approx(0.8)
        assert rec["meta_probability"] == pytest.approx(0.91)
        assert rec["final_label"] == "AI"
        # Main log never contains text
        assert "text" not in rec

    def test_sample_rate_zero_never_samples(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PREDICTION_LOG_DIR", str(tmp_path))
        monkeypatch.setenv("PREDICTION_SAMPLE_RATE", "0.0")
        from app.observability import prediction_log as pl
        for _ in range(10):
            pl.log_prediction(
                text="x", result=_fake_result(), model_version="v",
                now=datetime(2026, 4, 16, tzinfo=timezone.utc),
            )
        samples = tmp_path / "2026-04-16.samples.jsonl"
        assert not samples.exists() or samples.read_text() == ""

    def test_sample_rate_one_always_samples(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PREDICTION_LOG_DIR", str(tmp_path))
        monkeypatch.setenv("PREDICTION_SAMPLE_RATE", "1.0")
        from app.observability import prediction_log as pl
        for _ in range(5):
            pl.log_prediction(
                text="hello world",
                result=_fake_result(),
                model_version="v",
                now=datetime(2026, 4, 16, tzinfo=timezone.utc),
            )
        samples = tmp_path / "2026-04-16.samples.jsonl"
        assert samples.exists()
        lines = samples.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 5
        rec = json.loads(lines[0])
        assert rec["text"] == "hello world"
        assert rec["sample_rate"] == 1.0

    def test_disabled_is_noop(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PREDICTION_LOG_DIR", str(tmp_path))
        monkeypatch.setenv("PREDICTION_LOG_ENABLED", "0")
        monkeypatch.setenv("PREDICTION_SAMPLE_RATE", "1.0")
        from app.observability import prediction_log as pl
        pl.log_prediction(
            text="x",
            result=_fake_result(),
            model_version="v",
            now=datetime(2026, 4, 16, tzinfo=timezone.utc),
        )
        # No files created
        files = list(tmp_path.iterdir())
        assert files == []

    def test_text_truncation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PREDICTION_LOG_DIR", str(tmp_path))
        monkeypatch.setenv("PREDICTION_SAMPLE_RATE", "1.0")
        from app.observability import prediction_log as pl
        long_text = "A" * 5000
        pl.log_prediction(
            text=long_text,
            result=_fake_result(),
            model_version="v",
            now=datetime(2026, 4, 16, tzinfo=timezone.utc),
        )
        samples = tmp_path / "2026-04-16.samples.jsonl"
        rec = json.loads(samples.read_text().strip())
        assert "text_truncated" in rec
        assert len(rec["text"]) < 5000
        # Main record still records the full ORIGINAL length
        main = tmp_path / "2026-04-16.jsonl"
        main_rec = json.loads(main.read_text().strip())
        assert main_rec["input_length"] == 5000

    def test_errored_layer_records_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PREDICTION_LOG_DIR", str(tmp_path))
        monkeypatch.setenv("PREDICTION_SAMPLE_RATE", "0.0")
        from app.observability import prediction_log as pl
        result = _fake_result(layers=[
            _fake_layer("perplexity", 0.5, error="timeout"),
            _fake_layer("stylometry", 0.55),
            _fake_layer("transformer", 0.8),
        ])
        pl.log_prediction(
            text="x",
            result=result,
            model_version="v",
            now=datetime(2026, 4, 16, tzinfo=timezone.utc),
        )
        main = tmp_path / "2026-04-16.jsonl"
        rec = json.loads(main.read_text().strip())
        assert rec["l1_score"] is None  # errored layer → None
        assert rec["l2_score"] == pytest.approx(0.55)
        assert rec["l3_score"] == pytest.approx(0.8)

    def test_exception_safe_on_bad_result(self, tmp_path, monkeypatch):
        """A result object with no layer_results must not raise."""
        monkeypatch.setenv("PREDICTION_LOG_DIR", str(tmp_path))
        monkeypatch.setenv("PREDICTION_SAMPLE_RATE", "0.0")
        from app.observability import prediction_log as pl

        bad_result = SimpleNamespace(score=0.5, label="UNCERTAIN")
        # Missing layer_results attribute entirely — this must not raise
        pl.log_prediction(
            text="x",
            result=bad_result,
            model_version="v",
            now=datetime(2026, 4, 16, tzinfo=timezone.utc),
        )
        main = tmp_path / "2026-04-16.jsonl"
        assert main.exists()
        rec = json.loads(main.read_text().strip())
        # All three layer scores None because the layer_results list is empty
        assert rec["l1_score"] is None
        assert rec["l2_score"] is None
        assert rec["l3_score"] is None
        assert rec["meta_probability"] == pytest.approx(0.5)

    def test_exception_safe_on_unwritable_dir(self, tmp_path, monkeypatch, caplog):
        """Even a completely broken log path must not raise."""
        # Point at a file (not a dir) so mkdir fails cleanly
        bad = tmp_path / "not_a_dir_just_a_file"
        bad.write_text("x")
        monkeypatch.setenv("PREDICTION_LOG_DIR", str(bad / "nested" / "deeper"))
        monkeypatch.setenv("PREDICTION_SAMPLE_RATE", "0.0")
        from app.observability import prediction_log as pl
        # Must not raise
        pl.log_prediction(
            text="x",
            result=_fake_result(),
            model_version="v",
            now=datetime(2026, 4, 16, tzinfo=timezone.utc),
        )


# ── compute_daily_metrics ───────────────────────────────────────────


class TestDailyMetrics:
    def test_empty_log(self):
        import compute_daily_metrics as cdm
        m = cdm.compute_metrics([])
        assert m == {"n_predictions": 0, "empty": True}

    def test_known_input(self):
        import compute_daily_metrics as cdm
        rows = [
            {
                "timestamp": "2026-04-16T00:00:00+00:00",
                "model_version": "v",
                "input_length": 200,
                "l1_score": 0.4,
                "l2_score": 0.5,
                "l3_score": 0.9,
                "meta_probability": 0.85,
                "final_label": "AI",
            },
            {
                "timestamp": "2026-04-16T00:01:00+00:00",
                "model_version": "v",
                "input_length": 1500,
                "l1_score": 0.2,
                "l2_score": 0.3,
                "l3_score": 0.1,
                "meta_probability": 0.05,
                "final_label": "HUMAN",
            },
            {
                "timestamp": "2026-04-16T00:02:00+00:00",
                "model_version": "v",
                "input_length": 50,
                "l1_score": None,  # fallback path — L3 unavailable
                "l2_score": 0.5,
                "l3_score": None,
                "meta_probability": 0.5,
                "final_label": "UNCERTAIN",
            },
        ]
        m = cdm.compute_metrics(rows)
        assert m["n_predictions"] == 3
        assert m["mean_probability"] == pytest.approx((0.85 + 0.05 + 0.5) / 3)
        assert m["class_balance"]["AI"] == 1
        assert m["class_balance"]["HUMAN"] == 1
        assert m["class_balance"]["UNCERTAIN"] == 1
        # uniform over 3 labels: entropy = log2(3) ≈ 1.585
        assert m["entropy_binary_bits"] == pytest.approx(math.log2(3), abs=1e-6)
        assert m["fallback_rate"] == pytest.approx(1 / 3)
        # length buckets
        buckets = m["input_length_distribution"]["buckets"]
        assert buckets["0-100"] == 1      # 50
        assert buckets["100-500"] == 1    # 200
        assert buckets["500-2000"] == 1   # 1500

    def test_shannon_entropy_uniform_and_degenerate(self):
        import compute_daily_metrics as cdm
        assert cdm._shannon_entropy_bits({"A": 10, "B": 10}) == pytest.approx(1.0)
        assert cdm._shannon_entropy_bits({"A": 100, "B": 0}) == 0.0
        assert cdm._shannon_entropy_bits({}) == 0.0


# ── compute_drift (PSI) ─────────────────────────────────────────────


class TestPSI:
    def test_identical_distributions_zero(self):
        import compute_drift as cd
        ref = [100, 200, 300, 400]
        prod = [100, 200, 300, 400]
        psi, details = cd.compute_psi(ref, prod)
        assert psi == pytest.approx(0.0, abs=1e-9)
        assert len(details) == 4
        for d in details:
            assert d["contribution"] == pytest.approx(0.0, abs=1e-9)

    def test_proportional_scaling_zero(self):
        """PSI measures distribution shape, not absolute counts."""
        import compute_drift as cd
        ref = [100, 200, 300, 400]
        prod = [10, 20, 30, 40]  # same proportions, 10x smaller
        psi, _ = cd.compute_psi(ref, prod)
        assert psi == pytest.approx(0.0, abs=1e-9)

    def test_shifted_distribution_positive(self):
        import compute_drift as cd
        ref = [250, 250, 250, 250]  # uniform
        prod = [900, 50, 25, 25]    # heavily skewed to bucket 0
        psi, _ = cd.compute_psi(ref, prod)
        assert psi > 0.25  # significant drift

    def test_classification_thresholds(self):
        import compute_drift as cd
        assert cd.classify_psi(0.05) == "STABLE"
        assert cd.classify_psi(0.099) == "STABLE"
        assert cd.classify_psi(0.10) == "MODERATE"
        assert cd.classify_psi(0.20) == "MODERATE"
        assert cd.classify_psi(0.249999) == "MODERATE"
        assert cd.classify_psi(0.25) == "SIGNIFICANT"
        assert cd.classify_psi(0.50) == "SIGNIFICANT"

    def test_empty_bucket_epsilon_no_nan(self):
        """An empty bucket in either side must not produce NaN/inf."""
        import compute_drift as cd
        ref = [100, 100, 100, 100]
        prod = [400, 0, 0, 0]  # production collapsed into one bucket
        psi, details = cd.compute_psi(ref, prod)
        assert math.isfinite(psi)
        assert psi > 0.25  # significant
        # Every bucket has a finite contribution
        for d in details:
            assert math.isfinite(d["contribution"])

    def test_bucketize_probability_range(self):
        import compute_drift as cd
        edges = [0.0, 0.25, 0.5, 0.75, 1.0]
        # Edge cases: 0.0 in first bucket, 1.0 in last, 0.5 in middle
        counts = cd._bucketize([0.0, 0.1, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0], edges)
        assert sum(counts) == 8
        # First bucket (0.0, 0.25] gets 0.0, 0.1, 0.25 = 3 items (0.0 inclusive)
        assert counts[0] == 3
        # Last bucket (0.75, 1.0] gets 0.9 and 1.0 = 2 items
        assert counts[3] == 2

    def test_bucket_length_mismatch_raises(self):
        import compute_drift as cd
        with pytest.raises(ValueError, match="bucket count mismatch"):
            cd.compute_psi([1, 2, 3], [1, 2])
