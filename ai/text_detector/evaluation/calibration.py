"""
Step 22: Calibration validation.

Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
for each layer and the meta-classifier. Generates reliability diagram data
that the frontend can visualize.

A well-calibrated model means: when it says 80% AI probability,
it should be correct 80% of the time.

Reference: Guo et al. (2017) "On Calibration of Modern Neural Networks"
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import structlog

log = structlog.get_logger(__name__)


@dataclass
class CalibrationResult:
    """Calibration statistics for a single model/layer."""
    layer_name: str
    ece: float          # Expected Calibration Error — primary metric
    mce: float          # Maximum Calibration Error (worst bucket)
    n_samples: int
    n_buckets: int
    bucket_data: list[dict]   # for reliability diagram
    is_calibrated: bool       # ECE < 0.05 threshold per roadmap


def compute_ece(
    probabilities: Sequence[float],
    labels: Sequence[int],
    n_buckets: int = 15,
) -> CalibrationResult:
    """
    Compute Expected Calibration Error.

    Args:
        probabilities: Predicted AI probabilities, one per sample.
        labels:        Ground truth labels (1 = AI, 0 = human).
        n_buckets:     Number of equal-width probability buckets.

    Returns:
        CalibrationResult with ECE, MCE, and per-bucket data.
    """
    n = len(probabilities)
    assert n == len(labels), "probabilities and labels must be the same length"

    buckets: list[dict] = []
    total_ece = 0.0
    max_ce    = 0.0

    bucket_width = 1.0 / n_buckets

    for b in range(n_buckets):
        low  = b * bucket_width
        high = (b + 1) * bucket_width

        # Collect samples in this bucket
        in_bucket = [
            (p, l) for p, l in zip(probabilities, labels)
            if low <= p < high
        ]

        if not in_bucket:
            buckets.append({
                "bucket_low": round(low, 3),
                "bucket_high": round(high, 3),
                "n": 0,
                "confidence": round((low + high) / 2, 3),
                "accuracy": None,
                "gap": 0.0,
            })
            continue

        probs_b  = [p for p, _ in in_bucket]
        labels_b = [l for _, l in in_bucket]

        confidence = sum(probs_b) / len(probs_b)
        accuracy   = sum(labels_b) / len(labels_b)
        gap        = abs(confidence - accuracy)
        weight     = len(in_bucket) / n

        total_ece += weight * gap
        max_ce     = max(max_ce, gap)

        buckets.append({
            "bucket_low":  round(low, 3),
            "bucket_high": round(high, 3),
            "n":           len(in_bucket),
            "confidence":  round(confidence, 4),
            "accuracy":    round(accuracy, 4),
            "gap":         round(gap, 4),
        })

    # ECE < 0.05 is considered "well-calibrated" (industry standard)
    return CalibrationResult(
        layer_name="unknown",
        ece=round(total_ece, 4),
        mce=round(max_ce, 4),
        n_samples=n,
        n_buckets=n_buckets,
        bucket_data=buckets,
        is_calibrated=total_ece < 0.05,
    )


def validate_calibration(
    predictions: dict[str, tuple[list[float], list[int]]],
    output_dir: Path | None = None,
) -> dict[str, CalibrationResult]:
    """
    Validate calibration for multiple models simultaneously.

    Args:
        predictions: {layer_name: (probabilities_list, labels_list)}
        output_dir:  If set, writes calibration_report.json here.

    Returns:
        {layer_name: CalibrationResult}
    """
    results: dict[str, CalibrationResult] = {}

    for layer_name, (probs, labels) in predictions.items():
        result = compute_ece(probs, labels)
        result.layer_name = layer_name
        results[layer_name] = result

        status = "✓ CALIBRATED" if result.is_calibrated else "✗ NEEDS CALIBRATION"
        log.info(
            "calibration_result",
            layer=layer_name,
            ece=result.ece,
            mce=result.mce,
            n=result.n_samples,
            status=status,
        )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {k: asdict(v) for k, v in results.items()}
        out_path = output_dir / "calibration_report.json"
        with out_path.open("w") as f:
            json.dump(report, f, indent=2)
        log.info("calibration_report_written", path=str(out_path))

    return results


def check_calibration_gate(
    results: dict[str, CalibrationResult],
    max_ece: float = 0.05,
) -> bool:
    """
    CI gate: returns True only if ALL layers pass ECE < max_ece.
    Called in the training pipeline before a model is allowed to deploy.
    """
    all_pass = True
    for name, result in results.items():
        if result.ece >= max_ece:
            log.error(
                "calibration_gate_failed",
                layer=name,
                ece=result.ece,
                threshold=max_ece,
            )
            all_pass = False

    if all_pass:
        log.info("calibration_gate_passed", n_layers=len(results))

    return all_pass
