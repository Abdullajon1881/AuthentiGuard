"""
Stage 3 — Part 2: daily distribution metrics.

Reads a day's prediction log (`logs/predictions/YYYY-MM-DD.jsonl`)
and computes aggregate distribution statistics:

  - n_predictions
  - mean_probability          (mean of meta_probability)
  - probability_stddev
  - entropy_binary            (Shannon entropy of label distribution
                               over {AI, HUMAN, UNCERTAIN})
  - class_balance             ({AI: n, HUMAN: n, UNCERTAIN: n, ai_fraction})
  - input_length_distribution (n, mean, median, p95, max, bucket counts)
  - l1_mean / l2_mean / l3_mean
  - fallback_rate             (fraction of predictions that lack l3_score,
                               i.e. were produced by the heuristic fallback)

Writes / updates `metrics/daily_metrics.json` with one entry per date
(keyed by ISO date). Re-running for an already-computed date
overwrites that date's entry only.

Usage:
    # Compute for today
    python scripts/compute_daily_metrics.py

    # Compute for a specific date
    python scripts/compute_daily_metrics.py --date 2026-04-16

    # Use custom log dir / output path
    python scripts/compute_daily_metrics.py \
        --log-dir logs/predictions \
        --output metrics/daily_metrics.json

Deterministic; pure read of the log, pure write of the metrics JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent

LENGTH_BUCKETS = [
    (0, 100, "0-100"),
    (100, 500, "100-500"),
    (500, 2000, "500-2000"),
    (2000, 10000, "2000-10000"),
    (10000, float("inf"), "10000+"),
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"WARNING: {path}:{i+1} malformed JSON line, skipping: {exc}",
                    file=sys.stderr,
                )
    return rows


def _shannon_entropy_bits(counts: dict[str, int]) -> float:
    """Shannon entropy in bits of a discrete distribution.

    Defined so that a uniform k-way distribution has entropy log2(k).
    Used here over {AI, HUMAN, UNCERTAIN}.
    """
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for v in counts.values():
        if v <= 0:
            continue
        p = v / total
        h -= p * math.log2(p)
    return float(h)


def _bucketize_lengths(lengths: list[int]) -> dict[str, int]:
    counts = {label: 0 for _, _, label in LENGTH_BUCKETS}
    for n in lengths:
        for lo, hi, label in LENGTH_BUCKETS:
            if lo <= n < hi:
                counts[label] += 1
                break
    return counts


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(round(q * (len(s) - 1)))
    idx = max(0, min(len(s) - 1, idx))
    return float(s[idx])


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n_predictions": 0, "empty": True}

    probs = [float(r["meta_probability"]) for r in rows if "meta_probability" in r]
    lengths = [int(r.get("input_length", 0)) for r in rows]
    labels = [str(r.get("final_label", "UNCERTAIN")) for r in rows]

    label_counts: dict[str, int] = {"AI": 0, "HUMAN": 0, "UNCERTAIN": 0}
    for lab in labels:
        label_counts[lab] = label_counts.get(lab, 0) + 1

    l1_values = [r["l1_score"] for r in rows if r.get("l1_score") is not None]
    l2_values = [r["l2_score"] for r in rows if r.get("l2_score") is not None]
    l3_values = [r["l3_score"] for r in rows if r.get("l3_score") is not None]
    fallback_rows = [r for r in rows if r.get("l3_score") is None]

    def _mean(xs: list[float]) -> float | None:
        return float(statistics.fmean(xs)) if xs else None

    def _std(xs: list[float]) -> float | None:
        return float(statistics.pstdev(xs)) if len(xs) > 1 else None

    metrics: dict[str, Any] = {
        "n_predictions": n,
        "mean_probability": _mean(probs),
        "probability_stddev": _std(probs),
        "entropy_binary_bits": _shannon_entropy_bits(label_counts),
        "class_balance": {
            **label_counts,
            "ai_fraction": (
                label_counts["AI"] / n if n > 0 else 0.0
            ),
            "human_fraction": (
                label_counts["HUMAN"] / n if n > 0 else 0.0
            ),
            "uncertain_fraction": (
                label_counts["UNCERTAIN"] / n if n > 0 else 0.0
            ),
        },
        "input_length_distribution": {
            "n": len(lengths),
            "mean": _mean([float(x) for x in lengths]),
            "median": (
                float(statistics.median(lengths)) if lengths else None
            ),
            "p95": _percentile([float(x) for x in lengths], 0.95),
            "max": int(max(lengths)) if lengths else 0,
            "buckets": _bucketize_lengths(lengths),
        },
        "l1_mean": _mean(l1_values),
        "l2_mean": _mean(l2_values),
        "l3_mean": _mean(l3_values),
        "fallback_rate": (
            len(fallback_rows) / n if n > 0 else 0.0
        ),
    }
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--date",
        default=None,
        help="ISO date (YYYY-MM-DD) to compute metrics for. Default: today (UTC).",
    )
    ap.add_argument(
        "--log-dir",
        type=Path,
        default=_REPO_ROOT / "logs" / "predictions",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "metrics" / "daily_metrics.json",
    )
    args = ap.parse_args()

    date_str = args.date or datetime.now(timezone.utc).date().isoformat()
    log_path = args.log_dir / f"{date_str}.jsonl"

    print(f"[daily_metrics] date={date_str}")
    print(f"[daily_metrics] log: {log_path}")
    rows = _load_jsonl(log_path)
    print(f"[daily_metrics] rows: {len(rows)}")

    metrics = compute_metrics(rows)
    metrics["date"] = date_str
    metrics["computed_at_utc"] = datetime.now(timezone.utc).isoformat()
    metrics["source_log"] = str(log_path).replace("\\", "/")

    # Upsert into the metrics file: keep prior dates, overwrite this date.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if args.output.exists():
        try:
            with args.output.open("r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception as exc:
            print(
                f"WARNING: existing {args.output} is not valid JSON, overwriting: {exc}",
                file=sys.stderr,
            )
            existing = {}

    # Supports two shapes: a flat dict keyed by date, or a legacy single-date dict
    if "days" not in existing:
        existing = {"days": {}, "schema_version": 1}
    existing["days"][date_str] = metrics
    existing["last_updated_utc"] = datetime.now(timezone.utc).isoformat()

    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, sort_keys=False)
    os.replace(tmp, args.output)
    print(f"[daily_metrics] wrote {args.output}")

    # Console summary
    if metrics.get("empty"):
        print(f"[daily_metrics] no predictions found for {date_str}")
        return 0
    print()
    print(f"=== Daily metrics for {date_str} ===")
    print(f"n_predictions:     {metrics['n_predictions']}")
    print(f"mean_probability:  {metrics['mean_probability']:.4f}")
    print(f"entropy (bits):    {metrics['entropy_binary_bits']:.4f}")
    print(f"class_balance:     {metrics['class_balance']}")
    print(f"length_mean:       {metrics['input_length_distribution']['mean']:.1f}")
    print(f"fallback_rate:     {metrics['fallback_rate']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
