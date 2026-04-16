"""
Reliability evaluation — measures accuracy ONLY on non-UNCERTAIN predictions.

Stage 6 metric. The detector's 3-zone decision policy routes low-confidence
predictions to UNCERTAIN (abstain). This script measures how good the
system is on the predictions it IS willing to make:

  reliability  = accuracy on {AI, HUMAN}-labeled predictions only
  coverage     = fraction of inputs that received a definitive label
                 (i.e. NOT marked UNCERTAIN)
  F1           = standard F1 on the definitive predictions
  abstain_rate = 1 - coverage

The goal of the reliability policy is: **increase reliability even if
coverage decreases.** A system that returns AI/HUMAN on only 70% of
inputs but is right 95% of the time is more useful than one that
returns AI/HUMAN on 100% of inputs and is right 85%.

Runs the EXISTING production TextDetector end-to-end (same loader as
evaluate_end_to_end.py). Does NOT modify model logic, weights, or
architecture.

Usage:
    python scripts/evaluate_reliability.py \
        --checkpoint ai/text_detector/checkpoints/transformer_v3_hard/phase1 \
        --data-dir datasets/processed \
        --split test \
        --output ai/text_detector/accuracy/reliability_eval.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, stderr=subprocess.DEVNULL
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    split_path = args.data_dir / f"{args.split}.parquet"
    if not split_path.exists():
        print(f"ERROR: {split_path} not found", file=sys.stderr)
        return 2

    import pyarrow.parquet as pq
    table = pq.read_table(split_path)
    rows = table.to_pylist()
    if args.limit:
        rows = rows[: args.limit]
    print(f"[reliability] {len(rows)} rows from {split_path}", file=sys.stderr)

    from ai.text_detector.ensemble.text_detector import TextDetector, MODEL_VERSION

    det = TextDetector(
        transformer_checkpoint=args.checkpoint,
        adversarial_checkpoint=None,
        meta_checkpoint=None,
        device=args.device,
    )
    det.load_models()
    print(
        f"[reliability] detector loaded, active_layers={det._active_layers}, "
        f"lr_meta={'yes' if det._lr_meta else 'no'}, version={MODEL_VERSION}",
        file=sys.stderr,
    )

    # Run inference
    all_true: list[int] = []
    all_pred_label: list[str] = []
    all_score: list[float] = []
    per_source_data: dict[str, dict[str, list]] = {}

    try:
        if not args.no_progress:
            from tqdm import tqdm
            iterator = tqdm(rows, desc="reliability", unit="row")
        else:
            iterator = rows
    except ImportError:
        iterator = rows

    t0 = time.time()
    for row in iterator:
        text = row["text"]
        true_label = int(row["label"])
        result = det.analyze(text)

        all_true.append(true_label)
        all_pred_label.append(result.label)
        all_score.append(float(result.score))

        source = row.get("source")
        if source:
            bucket = per_source_data.setdefault(source, {"true": [], "pred": [], "score": []})
            bucket["true"].append(true_label)
            bucket["pred"].append(result.label)
            bucket["score"].append(float(result.score))
    t1 = time.time()
    print(f"[reliability] inference complete in {t1 - t0:.1f}s", file=sys.stderr)

    # Compute metrics
    n_total = len(all_true)
    label_counts = Counter(all_pred_label)

    # Split into definitive (AI/HUMAN) and abstained (UNCERTAIN)
    definitive_idx = [i for i, l in enumerate(all_pred_label) if l in ("AI", "HUMAN")]
    abstained_idx = [i for i, l in enumerate(all_pred_label) if l == "UNCERTAIN"]

    n_definitive = len(definitive_idx)
    n_abstained = len(abstained_idx)
    coverage = n_definitive / n_total if n_total > 0 else 0.0
    abstain_rate = n_abstained / n_total if n_total > 0 else 0.0

    # Accuracy + F1 on definitive predictions only
    if n_definitive > 0:
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            confusion_matrix, roc_auc_score,
        )

        def_true = [all_true[i] for i in definitive_idx]
        def_pred_binary = [1 if all_pred_label[i] == "AI" else 0 for i in definitive_idx]
        def_scores = [all_score[i] for i in definitive_idx]

        reliability = float(accuracy_score(def_true, def_pred_binary))
        f1 = float(f1_score(def_true, def_pred_binary, zero_division=0))
        precision = float(precision_score(def_true, def_pred_binary, zero_division=0))
        recall = float(recall_score(def_true, def_pred_binary, zero_division=0))
        cm = confusion_matrix(def_true, def_pred_binary).tolist()

        try:
            auroc = float(roc_auc_score(def_true, def_scores))
        except Exception:
            auroc = None

        metrics_definitive = {
            "n": n_definitive,
            "reliability_accuracy": round(reliability, 4),
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "auroc": auroc,
            "confusion_matrix": cm,
        }
    else:
        metrics_definitive = {
            "n": 0,
            "reliability_accuracy": None,
            "f1": None,
            "note": "all predictions were UNCERTAIN",
        }

    # Full-set F1 (for comparison — treats UNCERTAIN as not-AI)
    all_pred_binary = [1 if l == "AI" else 0 for l in all_pred_label]
    from sklearn.metrics import f1_score as _f1
    full_f1 = float(_f1(all_true, all_pred_binary, zero_division=0))

    # Per-source reliability
    per_source_reliability: dict[str, Any] = {}
    if per_source_data:
        for src, bucket in sorted(per_source_data.items()):
            src_def = [
                (t, p)
                for t, p in zip(bucket["true"], bucket["pred"])
                if p in ("AI", "HUMAN")
            ]
            n_src_def = len(src_def)
            n_src_total = len(bucket["true"])
            if n_src_def > 0 and len(set(t for t, _ in src_def)) > 1:
                src_true = [t for t, _ in src_def]
                src_pred = [1 if p == "AI" else 0 for _, p in src_def]
                src_acc = float(accuracy_score(src_true, src_pred))
                src_f1 = float(f1_score(src_true, src_pred, zero_division=0))
                per_source_reliability[src] = {
                    "n_total": n_src_total,
                    "n_definitive": n_src_def,
                    "coverage": round(n_src_def / n_src_total, 4),
                    "reliability_accuracy": round(src_acc, 4),
                    "f1": round(src_f1, 4),
                }
            else:
                per_source_reliability[src] = {
                    "n_total": n_src_total,
                    "n_definitive": n_src_def,
                    "coverage": round(n_src_def / n_src_total, 4) if n_src_total else 0.0,
                    "reliability_accuracy": None,
                    "f1": None,
                    "note": "single-class or all-UNCERTAIN subset",
                }

    report = {
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION,
        "checkpoint": str(args.checkpoint).replace("\\", "/"),
        "data_split": str(split_path).replace("\\", "/"),
        "data_split_sha256": _sha256_file(split_path),
        "n_total": n_total,
        "n_definitive": n_definitive,
        "n_abstained": n_abstained,
        "coverage": round(coverage, 4),
        "abstain_rate": round(abstain_rate, 4),
        "label_distribution": dict(label_counts),
        "metrics_on_definitive_only": metrics_definitive,
        "full_set_f1_treating_uncertain_as_not_ai": round(full_f1, 4),
        "per_source_reliability": per_source_reliability,
        "wallclock_seconds": round(t1 - t0, 2),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, args.output)
    print(f"[reliability] wrote {args.output}", file=sys.stderr)

    # Console summary
    print(file=sys.stderr)
    print(f"=== Reliability evaluation on {args.split} ===", file=sys.stderr)
    print(f"  total:                  {n_total}", file=sys.stderr)
    print(f"  definitive (AI/HUMAN):  {n_definitive}", file=sys.stderr)
    print(f"  abstained (UNCERTAIN):  {n_abstained}", file=sys.stderr)
    print(f"  coverage:               {coverage:.2%}", file=sys.stderr)
    print(f"  abstain rate:           {abstain_rate:.2%}", file=sys.stderr)
    if metrics_definitive.get("reliability_accuracy") is not None:
        print(f"  reliability (accuracy): {metrics_definitive['reliability_accuracy']:.4f}", file=sys.stderr)
        print(f"  F1 (definitive only):   {metrics_definitive['f1']:.4f}", file=sys.stderr)
        print(f"  precision:              {metrics_definitive['precision']:.4f}", file=sys.stderr)
        print(f"  recall:                 {metrics_definitive['recall']:.4f}", file=sys.stderr)
    print(f"  full-set F1 (UNCERTAIN=not-AI): {full_f1:.4f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
