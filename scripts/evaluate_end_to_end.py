"""
End-to-end evaluation of the production text-detection ensemble.

Stage 1 artifact (see ai/text_detector/ACCURACY.md). This script is the
single source of truth for "what F1 does the production pipeline actually
achieve on a held-out split." It loads the TextDetector with the EXACT
same arguments `backend/app/workers/text_worker.py::_get_detector` uses
in production (adversarial_checkpoint=None, meta_checkpoint=None,
device="cpu"), runs it over every row of the given parquet split, and
writes a JSON report with metrics, git SHA, timestamp, and dataset hash.

The prediction label comes from `detector.analyze(text).label` — the
PRODUCTION threshold logic — not from an externally-defined threshold.
Binarization for F1:  label == "AI"  → 1,  else → 0. "UNCERTAIN" is
treated as not-AI, matching the conservative moderation stance.
AUROC is computed on the raw ensemble `score` (not on binary labels).

Usage:
    python scripts/evaluate_end_to_end.py \
        --checkpoint ai/text_detector/checkpoints/transformer_v3_hard/phase1 \
        --data-dir datasets/processed \
        --split test \
        --output ai/text_detector/checkpoints/transformer_v3_hard/phase1/ensemble_test_eval.json

Deterministic by design: torch/numpy seeds are pinned inside TextDetector.
No retraining, no tuning, no architecture changes. Read-only against
weights; write-only against the --output JSON.
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

# Ensure the repo root is on sys.path so `ai.text_detector...` imports work
# regardless of where the script is invoked from. This mirrors the path
# discovery in backend/app/workers/text_worker.py::_get_detector.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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


def _build_production_detector(checkpoint: Path, device: str) -> Any:
    """Construct TextDetector with the EXACT production arguments.

    See backend/app/workers/text_worker.py::_get_detector lines ~320–340.
    adversarial_checkpoint and meta_checkpoint are intentionally None —
    the production system does not load L4 or the meta-classifier.
    """
    from ai.text_detector.ensemble.text_detector import TextDetector  # type: ignore

    detector = TextDetector(
        transformer_checkpoint=checkpoint,
        adversarial_checkpoint=None,
        meta_checkpoint=None,
        device=device,
    )
    detector.load_models()
    return detector


def _load_parquet(path: Path, limit: int | None) -> "list[dict[str, Any]]":
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    cols = table.column_names
    if "text" not in cols or "label" not in cols:
        raise ValueError(
            f"{path} missing required columns: expected text, label — got {cols}"
        )
    # Convert to a list of dicts so we can iterate deterministically and
    # carry through optional `source` / `sample_type` metadata.
    df = table.to_pylist()
    if limit is not None:
        df = df[:limit]
    return df


def _compute_metrics(
    y_true: "list[int]",
    y_pred: "list[int]",
    y_score: "list[float]",
) -> dict[str, Any]:
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics: dict[str, Any] = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    try:
        # roc_auc requires both classes present in y_true
        if len(set(y_true)) == 2:
            metrics["auroc"] = float(roc_auc_score(y_true, y_score))
        else:
            metrics["auroc"] = None
    except Exception as exc:
        metrics["auroc"] = None
        metrics["auroc_error"] = str(exc)
    return metrics


def _label_to_binary(label: str) -> int:
    """Production label → binary prediction.

    AI → 1, HUMAN → 0, UNCERTAIN → 0 (conservative moderation stance:
    only flag as AI when the detector is confident enough to say so).
    Changing this mapping changes the semantics of the reported F1.
    """
    return 1 if label == "AI" else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint", required=True, type=Path,
        help="Path to the L3 transformer checkpoint directory",
    )
    ap.add_argument(
        "--data-dir", required=True, type=Path,
        help="Directory containing {split}.parquet",
    )
    ap.add_argument(
        "--split", default="test",
        help="Parquet split file basename (default: test)",
    )
    ap.add_argument(
        "--output", required=True, type=Path,
        help="JSON path to write metrics to",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Optional row cap for smoke runs",
    )
    ap.add_argument(
        "--device", default="cpu",
        help="torch device (default: cpu — matches production)",
    )
    ap.add_argument(
        "--no-progress", action="store_true",
        help="Disable tqdm progress bar (CI-friendly)",
    )
    args = ap.parse_args()

    split_path = args.data_dir / f"{args.split}.parquet"
    if not split_path.exists():
        print(f"ERROR: split file not found: {split_path}", file=sys.stderr)
        return 2
    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[eval] loading dataset: {split_path}")
    rows = _load_parquet(split_path, args.limit)
    print(f"[eval] rows: {len(rows)}")

    print(f"[eval] loading detector from: {args.checkpoint}")
    t_load_start = time.time()
    detector = _build_production_detector(args.checkpoint, args.device)
    t_load_end = time.time()
    print(f"[eval] detector loaded in {t_load_end - t_load_start:.1f}s")
    print(f"[eval] active layers: {detector._active_layers}")

    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    per_source_truth: dict[str, list[int]] = {}
    per_source_pred: dict[str, list[int]] = {}
    label_counts: Counter = Counter()

    use_tqdm = not args.no_progress
    try:
        if use_tqdm:
            from tqdm import tqdm  # type: ignore
            iterator = tqdm(rows, desc="analyze", unit="row")
        else:
            iterator = rows
    except ImportError:
        iterator = rows

    t_analyze_start = time.time()
    for row in iterator:
        text = row["text"]
        true_label = int(row["label"])
        result = detector.analyze(text)
        score = float(result.score)
        binary_pred = _label_to_binary(result.label)

        y_true.append(true_label)
        y_pred.append(binary_pred)
        y_score.append(score)
        label_counts[result.label] += 1

        source = row.get("source")
        if source:
            per_source_truth.setdefault(source, []).append(true_label)
            per_source_pred.setdefault(source, []).append(binary_pred)

    t_analyze_end = time.time()

    print(f"[eval] analyze complete in {t_analyze_end - t_analyze_start:.1f}s")
    print(f"[eval] label distribution: {dict(label_counts)}")

    metrics = _compute_metrics(y_true, y_pred, y_score)

    per_source_f1: dict[str, Any] = {}
    if per_source_truth:
        from sklearn.metrics import f1_score
        for src, truths in per_source_truth.items():
            preds = per_source_pred[src]
            if len(set(truths)) < 2:
                per_source_f1[src] = {
                    "n": len(truths),
                    "f1": None,
                    "note": "single-class — F1 undefined",
                }
            else:
                per_source_f1[src] = {
                    "n": len(truths),
                    "f1": float(f1_score(truths, preds, zero_division=0)),
                }

    report = {
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint).replace("\\", "/"),
        "data_split": str(split_path).replace("\\", "/"),
        "data_split_sha256": _sha256_file(split_path),
        "n_samples": len(rows),
        "active_layers": list(detector._active_layers),
        "meta_classifier_loaded": detector._meta._is_fitted,
        "adversarial_layer_loaded": detector._layer4 is not None,
        "device": args.device,
        "metrics": metrics,
        "label_distribution": dict(label_counts),
        "per_source_f1": per_source_f1,
        "wallclock_seconds": {
            "load": round(t_load_end - t_load_start, 2),
            "analyze": round(t_analyze_end - t_analyze_start, 2),
            "total": round(time.time() - t_load_start, 2),
        },
    }

    # Atomic write: temp file then rename
    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=False)
    os.replace(tmp_path, args.output)

    print(f"[eval] wrote {args.output}")
    print(f"[eval] f1={metrics['f1']:.4f}  "
          f"precision={metrics['precision']:.4f}  "
          f"recall={metrics['recall']:.4f}  "
          f"auroc={metrics.get('auroc')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
