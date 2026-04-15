"""
Stage 3 — Part 3: Population Stability Index (PSI) drift detection.

Compares the production prediction distribution for a given day
against the frozen training reference distribution at
`ai/text_detector/accuracy/training_distribution.json`.

PSI formula:
    PSI = Σ_i (p_prod_i - p_ref_i) * ln(p_prod_i / p_ref_i)

where p_prod_i and p_ref_i are the proportions of samples in bucket i
of the production and reference distributions respectively. Buckets
are equal-width 10 bins over [0, 1] — frozen at the bootstrap step.

Empty-bucket handling: any bucket with zero count in either
distribution is replaced with a small epsilon (1e-6) before the log,
the industry-standard fix. Otherwise a single empty bucket makes the
PSI undefined (ln(0) or 0 * ln(0/x)).

Alert thresholds (industry convention):
    PSI < 0.10               -> STABLE
    0.10 <= PSI < 0.25       -> MODERATE drift
    PSI >= 0.25              -> SIGNIFICANT drift

Usage:
    # Today
    python scripts/compute_drift.py

    # Specific date
    python scripts/compute_drift.py --date 2026-04-16

    # Custom paths
    python scripts/compute_drift.py \
        --log-dir logs/predictions \
        --reference ai/text_detector/accuracy/training_distribution.json \
        --output metrics/drift.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent


# Alert thresholds — kept in sync with the constants in
# bootstrap_training_distribution.py and the spec.
PSI_STABLE_MAX = 0.10
PSI_MODERATE_MAX = 0.25

# Epsilon to avoid log(0) in empty buckets. Industry-standard value.
_PSI_EPSILON = 1e-6


def compute_psi(
    reference_counts: list[int],
    production_counts: list[int],
) -> tuple[float, list[dict[str, Any]]]:
    """Compute PSI between two histograms over matching bucket edges.

    Returns (psi_total, per_bucket_details). Per-bucket details are
    helpful for diagnosing WHICH buckets are shifting; they're included
    in the JSON report so operators can see where the drift is.
    """
    if len(reference_counts) != len(production_counts):
        raise ValueError(
            f"bucket count mismatch: ref={len(reference_counts)} "
            f"prod={len(production_counts)}"
        )

    n_ref = max(sum(reference_counts), 1)
    n_prod = max(sum(production_counts), 1)

    psi_total = 0.0
    details: list[dict[str, Any]] = []
    for i, (ref_c, prod_c) in enumerate(zip(reference_counts, production_counts)):
        p_ref = ref_c / n_ref
        p_prod = prod_c / n_prod
        p_ref_eps = max(p_ref, _PSI_EPSILON)
        p_prod_eps = max(p_prod, _PSI_EPSILON)
        contrib = (p_prod_eps - p_ref_eps) * math.log(p_prod_eps / p_ref_eps)
        psi_total += contrib
        details.append(
            {
                "bucket": i,
                "ref_count": int(ref_c),
                "prod_count": int(prod_c),
                "p_ref": round(p_ref, 6),
                "p_prod": round(p_prod, 6),
                "contribution": round(contrib, 6),
            }
        )
    return float(psi_total), details


def classify_psi(psi: float) -> str:
    if psi < PSI_STABLE_MAX:
        return "STABLE"
    if psi < PSI_MODERATE_MAX:
        return "MODERATE"
    return "SIGNIFICANT"


def _bucketize(probs: list[float], edges: list[float]) -> list[int]:
    """Same bucketization rule as bootstrap_training_distribution.py.

    First bucket inclusive on both sides so 0.0 is counted; last
    bucket inclusive on the right so 1.0 is counted; all middle
    buckets (lo, hi].
    """
    counts = [0] * (len(edges) - 1)
    n_buckets = len(edges) - 1
    for p in probs:
        p = float(p)
        for i in range(n_buckets):
            lo, hi = edges[i], edges[i + 1]
            if i == 0:
                if lo <= p <= hi:
                    counts[i] += 1
                    break
            elif i == n_buckets - 1:
                if lo < p <= hi:
                    counts[i] += 1
                    break
            else:
                if lo < p <= hi:
                    counts[i] += 1
                    break
    return counts


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--date",
        default=None,
        help="ISO date YYYY-MM-DD (default: today UTC)",
    )
    ap.add_argument(
        "--log-dir",
        type=Path,
        default=_REPO_ROOT / "logs" / "predictions",
    )
    ap.add_argument(
        "--reference",
        type=Path,
        default=_REPO_ROOT / "ai/text_detector/accuracy/training_distribution.json",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "metrics" / "drift.json",
    )
    args = ap.parse_args()

    if not args.reference.exists():
        print(
            f"ERROR: reference distribution not found: {args.reference}. "
            f"Run scripts/bootstrap_training_distribution.py first.",
            file=sys.stderr,
        )
        return 2

    date_str = args.date or datetime.now(timezone.utc).date().isoformat()
    log_path = args.log_dir / f"{date_str}.jsonl"

    print(f"[drift] date={date_str}")
    print(f"[drift] log: {log_path}")
    print(f"[drift] reference: {args.reference}")

    # Load reference
    with args.reference.open("r", encoding="utf-8") as f:
        ref_record = json.load(f)
    ref_edges = ref_record["psi"]["bucket_edges"]
    ref_counts = ref_record["psi"]["counts"]

    # Load production
    rows = _load_jsonl(log_path)
    prod_probs = [
        float(r["meta_probability"])
        for r in rows
        if "meta_probability" in r and r["meta_probability"] is not None
    ]
    print(f"[drift] production rows: {len(prod_probs)}")

    if not prod_probs:
        result_record = {
            "date": date_str,
            "computed_at_utc": datetime.now(timezone.utc).isoformat(),
            "n_production": 0,
            "psi": None,
            "classification": "INSUFFICIENT_DATA",
            "note": f"No predictions logged for {date_str}",
            "reference_git_sha": ref_record.get("git_sha"),
            "reference_timestamp": ref_record.get("timestamp_utc"),
        }
    else:
        prod_counts = _bucketize(prod_probs, ref_edges)
        psi, details = compute_psi(ref_counts, prod_counts)
        classification = classify_psi(psi)
        result_record = {
            "date": date_str,
            "computed_at_utc": datetime.now(timezone.utc).isoformat(),
            "n_production": len(prod_probs),
            "n_reference": int(sum(ref_counts)),
            "psi": round(psi, 6),
            "classification": classification,
            "bucket_edges": ref_edges,
            "per_bucket": details,
            "thresholds": {
                "stable_max": PSI_STABLE_MAX,
                "moderate_max": PSI_MODERATE_MAX,
            },
            "reference_git_sha": ref_record.get("git_sha"),
            "reference_timestamp": ref_record.get("timestamp_utc"),
            "source_log": str(log_path).replace("\\", "/"),
        }

    # Upsert into output file
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if args.output.exists():
        try:
            with args.output.open("r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    if "days" not in existing:
        existing = {"days": {}, "schema_version": 1}
    existing["days"][date_str] = result_record
    existing["last_updated_utc"] = datetime.now(timezone.utc).isoformat()

    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    os.replace(tmp, args.output)
    print(f"[drift] wrote {args.output}")

    # Console summary
    if result_record.get("psi") is None:
        print(f"[drift] {date_str}: INSUFFICIENT_DATA (no predictions)")
        # Exit 0 — no drift data is not a failure; alerting layer decides
        # what to do with an empty day.
        return 0

    psi_val = result_record["psi"]
    cls = result_record["classification"]
    print()
    print(f"=== Drift report for {date_str} ===")
    print(f"n_production:    {result_record['n_production']}")
    print(f"n_reference:     {result_record['n_reference']}")
    print(f"PSI:             {psi_val:.4f}")
    print(f"classification:  {cls}")
    print(f"thresholds:      stable<{PSI_STABLE_MAX}  moderate<{PSI_MODERATE_MAX}")

    # Exit codes mapping: 0 stable, 1 moderate, 2 significant.
    # Lets cron / CI trivially wire an alert:  `compute_drift.py || notify`
    if cls == "STABLE":
        return 0
    if cls == "MODERATE":
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
