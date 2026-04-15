"""
Stage 3 — Part 3 (setup): bootstrap the training reference distribution.

Loads `datasets/meta/meta_train.parquet`, runs the production
TextDetector with Stage 2 meta enabled on each row, and records the
bucketed distribution of calibrated meta probabilities. This bucketed
histogram becomes the REFERENCE distribution for all future PSI drift
checks.

Output: `ai/text_detector/accuracy/training_distribution.json`

Why run the detector instead of just reading `final_score_current`
from the parquet?  Because `final_score_current` is the Stage 1
fixed-weight score; the Stage 2 meta produces a different (calibrated)
score distribution, and drift monitoring compares production scores to
the distribution the CURRENT model produces, not the distribution some
previous model produced.

This is a one-time bootstrap, re-run whenever Stage 2 retrains the
meta classifier. Deterministic; no external state.

Usage:
    python scripts/bootstrap_training_distribution.py \
        --checkpoint ai/text_detector/checkpoints/transformer_v3_hard/phase1 \
        --meta-dataset datasets/meta/meta_train.parquet \
        --output ai/text_detector/accuracy/training_distribution.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# The PSI bucket edges must be FROZEN here and reused by compute_drift.py.
# Equal-width 10 bins over [0, 1] is the standard choice for a bounded
# probability. Both scripts read this file so the edges stay consistent.
PSI_BUCKETS = 10
PSI_EDGES = np.round(np.linspace(0.0, 1.0, PSI_BUCKETS + 1), 4).tolist()


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("ascii").strip()
    except Exception:
        return "unknown"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _bucketize(probs: np.ndarray, edges: list[float]) -> list[int]:
    """Return the count of values in each (edge[i], edge[i+1]] bucket.

    First bucket is inclusive on the left so 0.0 is counted;
    last bucket is inclusive on the right so 1.0 is counted.
    """
    counts = [0] * (len(edges) - 1)
    for p in probs:
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if i == 0:
                if lo <= p <= hi:
                    counts[i] += 1
                    break
            elif i == len(edges) - 2:
                if lo < p <= hi:
                    counts[i] += 1
                    break
            else:
                if lo < p <= hi:
                    counts[i] += 1
                    break
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=_REPO_ROOT / "ai/text_detector/checkpoints/transformer_v3_hard/phase1",
    )
    ap.add_argument(
        "--meta-dataset",
        type=Path,
        default=_REPO_ROOT / "datasets/meta/meta_train.parquet",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "ai/text_detector/accuracy/training_distribution.json",
    )
    args = ap.parse_args()

    if not args.meta_dataset.exists():
        print(f"ERROR: meta dataset not found: {args.meta_dataset}", file=sys.stderr)
        return 2
    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    # Load the meta parquet to get (l1, l2, l3) per row — fastest
    # path, no need to re-run the L1/L2/L3 layers from text.
    import pyarrow.parquet as pq
    table = pq.read_table(args.meta_dataset)
    cols = table.column_names
    for required in ("l1_score", "l2_score", "l3_score", "label"):
        if required not in cols:
            print(f"ERROR: {args.meta_dataset} missing column {required!r}", file=sys.stderr)
            return 2
    data = table.to_pandas() if hasattr(table, "to_pandas") else None
    if data is not None:
        X = data[["l1_score", "l2_score", "l3_score"]].to_numpy(dtype=np.float64)
        y = data["label"].to_numpy(dtype=np.int32)
    else:
        X = np.stack(
            [np.asarray(table.column(c).to_pylist()) for c in ("l1_score", "l2_score", "l3_score")],
            axis=1,
        ).astype(np.float64)
        y = np.asarray(table.column("label").to_pylist(), dtype=np.int32)
    print(f"[bootstrap] loaded {len(X)} rows from {args.meta_dataset}")

    # Load the Stage 2 meta calibrator bundle directly — no need to
    # spin up the full TextDetector for this bootstrap.
    import joblib
    cal_path = args.checkpoint.parent.parent / "meta_calibrator.joblib"
    lr_path = args.checkpoint.parent.parent / "meta_classifier.joblib"
    if not cal_path.exists() or not lr_path.exists():
        print(
            f"ERROR: meta artifacts not found at {cal_path} / {lr_path}. "
            f"Run scripts/train_meta_classifier.py first.",
            file=sys.stderr,
        )
        return 2

    lr = joblib.load(lr_path)
    bundle = joblib.load(cal_path)
    calibrator = bundle["calibrator"]
    feature_order = bundle.get("feature_order", ["l1_score", "l2_score", "l3_score"])
    if feature_order != ["l1_score", "l2_score", "l3_score"]:
        print(
            f"WARNING: non-standard feature_order in calibrator bundle: {feature_order}. "
            "Re-ordering X to match.",
            file=sys.stderr,
        )
        col_idx = {"l1_score": 0, "l2_score": 1, "l3_score": 2}
        X = X[:, [col_idx[f] for f in feature_order]]

    t_start = time.time()
    cal_probs = calibrator.predict_proba(X)[:, 1].astype(np.float64)
    t_end = time.time()
    print(f"[bootstrap] calibrated probs computed in {t_end - t_start:.2f}s")
    print(
        f"[bootstrap] prob stats: "
        f"min={cal_probs.min():.4f} mean={cal_probs.mean():.4f} "
        f"max={cal_probs.max():.4f}"
    )

    counts = _bucketize(cal_probs, PSI_EDGES)
    total = int(sum(counts))
    proportions = [c / total if total > 0 else 0.0 for c in counts]

    # Reference label distribution (for secondary drift checks)
    label_counts = {
        "human_0": int((y == 0).sum()),
        "ai_1": int((y == 1).sum()),
    }

    output_record = {
        "schema_version": 1,
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(args.meta_dataset).replace("\\", "/"),
        "source_dataset_sha256": _sha256_file(args.meta_dataset),
        "checkpoint": str(args.checkpoint).replace("\\", "/"),
        "meta_classifier_path": str(lr_path).replace("\\", "/"),
        "meta_calibrator_path": str(cal_path).replace("\\", "/"),
        "feature_order": feature_order,
        "n_samples": int(len(X)),
        "psi": {
            "bucket_edges": PSI_EDGES,
            "counts": counts,
            "proportions": [round(p, 6) for p in proportions],
            "n_buckets": PSI_BUCKETS,
        },
        "prob_stats": {
            "min": float(cal_probs.min()),
            "mean": float(cal_probs.mean()),
            "max": float(cal_probs.max()),
            "std": float(cal_probs.std()),
            "median": float(np.median(cal_probs)),
        },
        "label_distribution": label_counts,
        "notes": (
            "Reference distribution for Population Stability Index drift "
            "monitoring. Bucket edges are FROZEN at 10 equal-width bins "
            "over [0, 1]. Regenerate this file whenever the meta "
            "classifier or calibrator is retrained."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(output_record, f, indent=2)
    os.replace(tmp, args.output)
    print(f"[bootstrap] wrote {args.output}")
    print(f"[bootstrap] bucket counts: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
