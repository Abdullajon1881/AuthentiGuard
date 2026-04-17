"""
Generate 7 days of realistic demo prediction logs + daily metrics + drift
data so the dashboard has content on first launch.

Uses the REAL measured distributions from the evaluation artifacts:
  - calibrated meta scores are bimodal (cluster near 0.0 and 1.0)
  - L2 scores center ~0.32 with narrow spread
  - L3 scores center ~0.49 with wide spread
  - ~14% of v1 test rows have < 50 words

Outputs:
  logs/predictions/YYYY-MM-DD.jsonl        (7 days, ~300 rows each)
  metrics/daily_metrics.json               (7 days of aggregates)
  metrics/drift.json                       (7 days of PSI)
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Deterministic
random.seed(42)

# Realistic score generators based on measured distributions
def _gen_prediction(day_offset: int, model_version: str) -> dict:
    """Generate one realistic prediction record."""
    is_ai = random.random() < 0.48  # ~48% AI in the test sets

    if is_ai:
        l3 = random.betavariate(8, 2) * 0.4 + 0.55  # 0.55–0.95
        l1 = random.betavariate(2, 5) * 0.3 + 0.05   # 0.05–0.35
    else:
        l3 = random.betavariate(2, 8) * 0.35 + 0.08   # 0.08–0.43
        l1 = random.betavariate(5, 2) * 0.15 + 0.01   # 0.01–0.16
    l2 = random.gauss(0.32, 0.08)
    l2 = max(0.03, min(0.57, l2))

    # Calibrated probability (bimodal: near 0 or near 1)
    if is_ai:
        meta = random.betavariate(12, 1.5) * 0.3 + 0.70  # 0.70–1.00
        meta = min(0.99, meta)
    else:
        meta = random.betavariate(1.5, 12) * 0.15         # 0.00–0.15
        meta = max(0.01, meta)
    # Add some noise to create a few mid-range scores
    if random.random() < 0.08:
        meta = random.uniform(0.30, 0.70)

    # Decision logic (matches text_detector.py)
    word_count = random.choices(
        [random.randint(15, 49), random.randint(50, 350)],
        weights=[14, 86],
    )[0]

    if meta >= 0.70:
        label = "AI"
    elif meta <= 0.30:
        label = "HUMAN"
    else:
        label = "UNCERTAIN"

    if word_count < 50 and label != "UNCERTAIN":
        label = "UNCERTAIN"
    if abs(l2 - l3) > 0.40 and label != "UNCERTAIN":
        label = "UNCERTAIN"

    confidence_margin = round(meta - 0.41, 4)
    zone = "AI" if meta >= 0.70 else ("HUMAN" if meta <= 0.30 else "UNCERTAIN")

    base_time = datetime(2026, 4, 10, tzinfo=timezone.utc) + timedelta(days=day_offset)
    ts = base_time + timedelta(
        hours=random.randint(8, 22),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )

    return {
        "timestamp": ts.isoformat(),
        "model_version": model_version,
        "input_length": word_count * random.randint(4, 7),
        "l1_score": round(l1, 4),
        "l2_score": round(l2, 4),
        "l3_score": round(l3, 4),
        "meta_probability": round(meta, 4),
        "final_label": label,
        "confidence_margin": confidence_margin,
        "zone": zone,
    }


def _shannon_entropy(counts: dict) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for v in counts.values():
        if v <= 0:
            continue
        p = v / total
        h -= p * math.log2(p)
    return h


def _compute_psi(ref_counts: list[int], prod_counts: list[int]) -> float:
    eps = 1e-6
    n_ref = max(sum(ref_counts), 1)
    n_prod = max(sum(prod_counts), 1)
    psi = 0.0
    for rc, pc in zip(ref_counts, prod_counts):
        pr = max(rc / n_ref, eps)
        pp = max(pc / n_prod, eps)
        psi += (pp - pr) * math.log(pp / pr)
    return psi


def main():
    log_dir = _REPO_ROOT / "logs" / "predictions"
    metrics_dir = _REPO_ROOT / "metrics"
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load training reference distribution for PSI
    ref_path = _REPO_ROOT / "ai" / "text_detector" / "accuracy" / "training_distribution.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
        ref_counts = ref["psi"]["counts"]
        ref_edges = ref["psi"]["bucket_edges"]
    else:
        ref_counts = [500, 0, 0, 0, 0, 0, 0, 0, 0, 500]
        ref_edges = [round(i * 0.1, 1) for i in range(11)]

    versions = [
        (0, 1, "3.0-stage1-fixed-weights"),
        (2, 3, "3.0-stage2-lr_meta-isotonic"),
        (4, 4, "3.1-reliability-gated"),  # deprecated: G2 disagreement gate
        (5, 6, "3.2-g2-removed-product-output"),  # current production
    ]

    daily_metrics_all: dict[str, dict] = {}
    drift_all: dict[str, dict] = {}

    for day in range(7):
        date = (datetime(2026, 4, 10, tzinfo=timezone.utc) + timedelta(days=day)).date()
        date_str = date.isoformat()

        version = "3.2-g2-removed-product-output"
        for lo, hi, ver in versions:
            if lo <= day <= hi:
                version = ver
                break

        n_preds = random.randint(250, 400)
        records = [_gen_prediction(day, version) for _ in range(n_preds)]
        records.sort(key=lambda r: r["timestamp"])

        # Write JSONL
        log_path = log_dir / f"{date_str}.jsonl"
        with log_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        # Daily metrics
        probs = [r["meta_probability"] for r in records]
        labels = [r["final_label"] for r in records]
        label_counts = {"AI": 0, "HUMAN": 0, "UNCERTAIN": 0}
        for lab in labels:
            label_counts[lab] = label_counts.get(lab, 0) + 1

        n = len(records)
        daily_metrics_all[date_str] = {
            "date": date_str,
            "n_predictions": n,
            "mean_probability": round(sum(probs) / n, 4) if n else 0,
            "entropy_binary_bits": round(_shannon_entropy(label_counts), 4),
            "class_balance": {
                **label_counts,
                "ai_fraction": round(label_counts["AI"] / n, 4),
                "human_fraction": round(label_counts["HUMAN"] / n, 4),
                "uncertain_fraction": round(label_counts["UNCERTAIN"] / n, 4),
            },
            "model_version": version,
        }

        # PSI
        n_buckets = len(ref_edges) - 1
        prod_counts = [0] * n_buckets
        for p in probs:
            for i in range(n_buckets):
                lo, hi = ref_edges[i], ref_edges[i + 1]
                if i == 0 and lo <= p <= hi:
                    prod_counts[i] += 1
                    break
                elif i == n_buckets - 1 and lo < p <= hi:
                    prod_counts[i] += 1
                    break
                elif lo < p <= hi:
                    prod_counts[i] += 1
                    break

        psi = _compute_psi(ref_counts, prod_counts)
        cls = "STABLE" if psi < 0.10 else ("MODERATE" if psi < 0.25 else "SIGNIFICANT")
        drift_all[date_str] = {
            "date": date_str,
            "psi": round(psi, 6),
            "classification": cls,
            "n_production": n,
        }

        print(f"  {date_str}: {n} predictions, {label_counts}, PSI={psi:.4f} {cls}")

    # Write aggregated metrics
    dm_path = metrics_dir / "daily_metrics.json"
    dm = {"schema_version": 1, "days": daily_metrics_all,
          "last_updated_utc": datetime.now(timezone.utc).isoformat()}
    dm_path.write_text(json.dumps(dm, indent=2))

    dr_path = metrics_dir / "drift.json"
    dr = {"schema_version": 1, "days": drift_all,
          "last_updated_utc": datetime.now(timezone.utc).isoformat()}
    dr_path.write_text(json.dumps(dr, indent=2))

    print(f"\nWrote 7 days of logs to {log_dir}")
    print(f"Wrote daily_metrics to {dm_path}")
    print(f"Wrote drift to {dr_path}")


if __name__ == "__main__":
    main()
