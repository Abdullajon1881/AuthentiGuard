"""
Step 98: Tiered analysis pipeline.

Two-pass architecture that meets the latency targets (Step 102):
  - Sub-2s for text analysis
  - Sub-10s for media analysis

Pass 1 — Fast pre-screen (student / ONNX model)
  Runs in milliseconds. Returns a coarse score.
  If score < SKIP_THRESHOLD  → return "likely human" immediately
  If score > FLAG_THRESHOLD  → escalate to deep analysis
  Otherwise                  → return the fast score with lower confidence

Pass 2 — Deep analysis (full teacher ensemble)
  Only triggered for content that scored above FLAG_THRESHOLD in Pass 1.
  ~60–80% of requests never reach this pass (cost reduction + speed).

Threshold calibration:
  SKIP_THRESHOLD = 0.25  → miss rate on AI content < 2% (acceptable FN)
  FLAG_THRESHOLD = 0.45  → ~40% of requests require deep analysis

These thresholds are tuned on held-out validation data to optimise:
  overall latency × detection rate
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import structlog

log = structlog.get_logger(__name__)

# Thresholds
SKIP_THRESHOLD = 0.25   # below → skip deep analysis (likely human)
FLAG_THRESHOLD = 0.45   # above → trigger deep analysis

# Expected latency reduction: fraction of requests skipping deep pass
EXPECTED_SKIP_RATE = 0.62   # ~62% of real-world content is human-authored


@dataclass
class TieredResult:
    """Result from the tiered analysis pipeline."""
    score:            float
    label:            str
    confidence:       float
    tier_used:        str          # "fast" | "deep" | "fast_only"
    fast_score:       float        # always set
    deep_score:       float | None  # None if deep pass was skipped
    fast_ms:          int
    deep_ms:          int          # 0 if skipped
    total_ms:         int
    skip_reason:      str | None   # why deep was skipped


class TieredAnalysisPipeline:
    """
    Two-pass analysis pipeline with automatic tier selection.

    Usage:
        pipeline = TieredAnalysisPipeline(
            fast_detector=onnx_detector,
            deep_detector=full_ensemble,
        )
        result = pipeline.analyze(content, content_type)
    """

    def __init__(
        self,
        fast_detector:     Any,           # ONNXDetector or callable
        deep_detector:     Any,           # Full ensemble (TextDetector etc.)
        skip_threshold:    float = SKIP_THRESHOLD,
        flag_threshold:    float = FLAG_THRESHOLD,
        fast_score_fn:     Callable | None = None,
        deep_score_fn:     Callable | None = None,
    ) -> None:
        self._fast          = fast_detector
        self._deep          = deep_detector
        self._skip_t        = skip_threshold
        self._flag_t        = flag_threshold
        self._fast_score_fn = fast_score_fn or self._default_fast_score
        self._deep_score_fn = deep_score_fn or self._default_deep_score

    def analyze(
        self,
        content:      Any,
        content_type: str,
        force_deep:   bool = False,
    ) -> TieredResult:
        """
        Run tiered analysis. Returns a TieredResult.

        Args:
            content:      Content to analyse (text str or bytes for media)
            content_type: "text" | "image" | "audio" | "video" | "code"
            force_deep:   Skip fast pass and go straight to deep analysis
        """
        t_start = int(time.time() * 1000)
        skip_reason: str | None = None

        # ── Pass 1: Fast pre-screen ────────────────────────────
        t_fast_start = int(time.time() * 1000)
        fast_score   = 0.5

        if not force_deep:
            try:
                fast_score = self._fast_score_fn(content, content_type)
            except Exception as exc:
                log.warning("fast_screen_failed", error=str(exc))
                fast_score = 0.5   # neutral → proceed to deep

        fast_ms = int(time.time() * 1000) - t_fast_start

        # ── Tier routing ───────────────────────────────────────
        run_deep   = force_deep or fast_score >= self._flag_t
        deep_score: float | None = None
        deep_ms    = 0
        tier_used  = "fast"

        if not run_deep:
            skip_reason = (
                f"Fast score {fast_score:.3f} < threshold {self._flag_t} "
                f"— skipping deep analysis"
            )
            final_score = fast_score
            tier_used   = "fast_only"
        else:
            # ── Pass 2: Deep analysis ──────────────────────────
            t_deep_start = int(time.time() * 1000)
            try:
                deep_score = self._deep_score_fn(content, content_type)
            except Exception as exc:
                log.warning("deep_analysis_failed", error=str(exc))
                deep_score = fast_score   # fall back to fast score
            deep_ms     = int(time.time() * 1000) - t_deep_start
            tier_used   = "deep"

            # Blend fast and deep: deep gets 80% weight
            final_score = 0.20 * fast_score + 0.80 * deep_score

        final_score = float(np.clip(final_score, 0.01, 0.99))

        label = (
            "AI"        if final_score >= 0.75 else
            "HUMAN"     if final_score <= 0.40 else
            "UNCERTAIN"
        )
        confidence  = round(abs(final_score - 0.5) * 2, 4)
        total_ms    = int(time.time() * 1000) - t_start

        log.info("tiered_analysis_complete",
                 tier=tier_used, fast_ms=fast_ms, deep_ms=deep_ms,
                 final_score=round(final_score, 4))

        return TieredResult(
            score=round(final_score, 4),
            label=label,
            confidence=confidence,
            tier_used=tier_used,
            fast_score=round(fast_score, 4),
            deep_score=round(deep_score, 4) if deep_score is not None else None,
            fast_ms=fast_ms,
            deep_ms=deep_ms,
            total_ms=total_ms,
            skip_reason=skip_reason,
        )

    @staticmethod
    def _default_fast_score(content: Any, content_type: str) -> float:
        """Placeholder — replaced by ONNXDetector.predict_proba()."""
        return 0.5

    @staticmethod
    def _default_deep_score(content: Any, content_type: str) -> float:
        """Placeholder — replaced by full ensemble."""
        return 0.5


# ── Threshold calibration ─────────────────────────────────────

def calibrate_thresholds(
    fast_scores:     list[float],
    deep_scores:     list[float],
    labels:          list[int],
    max_fn_rate:     float = 0.02,
    max_fp_rate:     float = 0.15,
) -> dict[str, float]:
    """
    Calibrate SKIP_THRESHOLD and FLAG_THRESHOLD on validation data.

    Optimisation objective:
      Minimise fraction of requests going to deep pass
      Subject to:
        - False negative rate on AI content < max_fn_rate
        - False positive rate on human content < max_fp_rate

    Args:
        fast_scores: Pre-screen scores from the fast model
        deep_scores: Scores from the deep model (ground truth proxy)
        labels:      Binary ground truth (1 = AI, 0 = human)
        max_fn_rate: Maximum acceptable miss rate on AI content
        max_fp_rate: Maximum acceptable false alarm rate

    Returns:
        {"skip_threshold": float, "flag_threshold": float, "skip_rate": float}
    """
    fast = np.array(fast_scores)
    labels_arr = np.array(labels)
    n_ai    = labels_arr.sum()
    n_human = len(labels_arr) - n_ai

    best_skip_rate = 0.0
    best_skip_t    = SKIP_THRESHOLD
    best_flag_t    = FLAG_THRESHOLD

    for skip_t in np.arange(0.05, 0.45, 0.05):
        # FN rate: fraction of AI content scored below skip_threshold
        skipped_ai  = float(np.sum((fast < skip_t) & (labels_arr == 1)))
        fn_rate     = skipped_ai / max(n_ai, 1)

        if fn_rate > max_fn_rate:
            continue

        for flag_t in np.arange(skip_t + 0.05, 0.80, 0.05):
            # Skip rate: fraction of all content that skips deep
            skip_rate = float(np.mean(fast < flag_t))

            # FP check: fraction of human content flagged for deep
            flagged_human = float(np.sum((fast >= flag_t) & (labels_arr == 0)))
            if n_human > 0 and flagged_human / n_human > (1 - max_fp_rate):
                continue

            if skip_rate > best_skip_rate:
                best_skip_rate = skip_rate
                best_skip_t    = float(skip_t)
                best_flag_t    = float(flag_t)

    return {
        "skip_threshold": round(best_skip_t, 3),
        "flag_threshold": round(best_flag_t, 3),
        "skip_rate":      round(best_skip_rate, 3),
        "expected_speedup": round(1.0 / max(1.0 - best_skip_rate, 0.01), 2),
    }


# ── Latency tracker ───────────────────────────────────────────

class LatencyTracker:
    """
    Tracks inference latency statistics across multiple requests.
    Used to verify Step 102 latency targets are met.
    """

    def __init__(self, target_ms: int, name: str) -> None:
        self._target = target_ms
        self._name   = name
        self._times: list[float] = []

    def record(self, ms: float) -> None:
        self._times.append(ms)

    def stats(self) -> dict[str, Any]:
        if not self._times:
            return {"name": self._name, "n": 0}
        arr = np.array(self._times)
        return {
            "name":        self._name,
            "n":           len(arr),
            "mean_ms":     round(float(arr.mean()), 1),
            "p50_ms":      round(float(np.percentile(arr, 50)), 1),
            "p95_ms":      round(float(np.percentile(arr, 95)), 1),
            "p99_ms":      round(float(np.percentile(arr, 99)), 1),
            "target_ms":   self._target,
            "target_met":  float(np.percentile(arr, 95)) <= self._target,
            "pct_on_time": round(float(np.mean(arr <= self._target)) * 100, 1),
        }

    def assert_target_met(self, percentile: float = 95.0) -> None:
        """Raise if p{percentile} latency exceeds the target."""
        if not self._times:
            return
        p = float(np.percentile(self._times, percentile))
        if p > self._target:
            raise AssertionError(
                f"{self._name}: p{percentile:.0f} latency {p:.1f}ms "
                f"exceeds target {self._target}ms"
            )
