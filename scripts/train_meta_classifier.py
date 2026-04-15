"""
Stage 2 — Parts 2 & 3: train the stacking meta-classifier and calibrator.

Reads the parquet produced by `scripts/build_meta_dataset.py`:
    datasets/meta/meta_train.parquet

Splits 80/20 stratified (random_state=42), trains a 3-feature
LogisticRegression with class_weight="balanced" on the 80% split,
evaluates on the 20% split, then fits an isotonic
CalibratedClassifierCV(cv="prefit") on the same 20% split.

Artifacts written:
    ai/text_detector/checkpoints/meta_classifier.joblib
        the raw LogisticRegression estimator
    ai/text_detector/checkpoints/meta_calibrator.joblib
        a dict: {"calibrator": CalibratedClassifierCV,
                 "threshold": float,
                 "feature_order": ["l1_score", "l2_score", "l3_score"]}
    ai/text_detector/accuracy/meta_classifier_metrics.json
        F1, precision, recall, AUROC, confusion matrix on the 20% val
    ai/text_detector/accuracy/calibration_metrics.json
        ECE, Brier score, baseline Brier, threshold sweep summary

The calibrator bundle carries the threshold so TextDetector only has to
load one file. Feature order is frozen so downstream consumers cannot
silently swap columns.

Deterministic: random_state pinned everywhere. Read-only against the
meta training parquet; writes only the four artifacts above.

Usage:
    python scripts/train_meta_classifier.py \
        --meta-dataset datasets/meta/meta_train.parquet \
        --model-dir ai/text_detector/checkpoints \
        --metrics-dir ai/text_detector/accuracy
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent

RANDOM_STATE = 42
N_ECE_BINS = 10
FEATURE_ORDER = ["l1_score", "l2_score", "l3_score"]


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


def _expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = N_ECE_BINS,
) -> float:
    """Equal-width binned ECE with `n_bins` bins.

    ECE = sum_b (n_b / N) * |mean(y_prob[b]) - mean(y_true[b])|

    The last bin is inclusive on both ends so probabilities of exactly
    1.0 are not dropped. All other bins are [lo, hi).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_prob)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        mean_pred = float(y_prob[mask].mean())
        mean_true = float(y_true[mask].mean())
        ece += (mask.sum() / n) * abs(mean_pred - mean_true)
    return float(ece)


def _sweep_threshold_for_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    lo: float = 0.30,
    hi: float = 0.70,
    step: float = 0.01,
) -> tuple[float, float]:
    """Find the threshold that maximizes F1 on (y_true, y_prob).

    Returns (best_threshold, best_f1). Tie-break: closer to 0.5
    (natural argmax for a calibrated probability).
    """
    from sklearn.metrics import f1_score

    thresholds = np.round(np.arange(lo, hi + 1e-9, step), 4)
    best_f1 = -1.0
    best_t = 0.5
    for t in thresholds:
        pred = (y_prob >= t).astype(np.int32)
        f1 = float(f1_score(y_true, pred, zero_division=0))
        if f1 > best_f1 + 1e-9:
            best_f1 = f1
            best_t = float(t)
        elif abs(f1 - best_f1) < 1e-9 and abs(t - 0.5) < abs(best_t - 0.5):
            best_t = float(t)
    return best_t, best_f1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--meta-dataset",
        type=Path,
        default=_REPO_ROOT / "datasets/meta/meta_train.parquet",
        help="Path to meta_train.parquet produced by build_meta_dataset.py",
    )
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=_REPO_ROOT / "ai/text_detector/checkpoints",
        help="Where to save meta_classifier.joblib and meta_calibrator.joblib",
    )
    ap.add_argument(
        "--metrics-dir",
        type=Path,
        default=_REPO_ROOT / "ai/text_detector/accuracy",
        help="Where to save meta_classifier_metrics.json and calibration_metrics.json",
    )
    ap.add_argument("--test-size", type=float, default=0.20)
    args = ap.parse_args()

    if not args.meta_dataset.exists():
        print(
            f"ERROR: meta dataset not found: {args.meta_dataset}. "
            f"Run scripts/build_meta_dataset.py first.",
            file=sys.stderr,
        )
        return 2

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────
    import pyarrow.parquet as pq
    table = pq.read_table(args.meta_dataset)
    cols = table.column_names
    for required in FEATURE_ORDER + ["label", "final_score_current"]:
        if required not in cols:
            print(
                f"ERROR: {args.meta_dataset} missing required column {required!r}. "
                f"Got: {cols}",
                file=sys.stderr,
            )
            return 2
    df = table.to_pandas() if hasattr(table, "to_pandas") else None
    if df is None:
        # Fallback if pandas not present
        data = {c: np.asarray(table.column(c).to_pylist()) for c in cols}
        X_all = np.stack([data[c] for c in FEATURE_ORDER], axis=1).astype(np.float64)
        y_all = data["label"].astype(np.int32)
        fsc_all = data["final_score_current"].astype(np.float64)
    else:
        X_all = df[FEATURE_ORDER].to_numpy(dtype=np.float64)
        y_all = df["label"].to_numpy(dtype=np.int32)
        fsc_all = df["final_score_current"].to_numpy(dtype=np.float64)

    print(f"[train_meta] loaded {len(y_all)} rows from {args.meta_dataset}")
    print(f"[train_meta] label distribution: {np.bincount(y_all).tolist()}")

    # ── Stratified 80/20 split ───────────────────────────────────
    from sklearn.model_selection import train_test_split

    (
        X_train, X_val,
        y_train, y_val,
        fsc_train, fsc_val,
    ) = train_test_split(
        X_all, y_all, fsc_all,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )
    print(
        f"[train_meta] split: train={len(y_train)} "
        f"({np.bincount(y_train).tolist()}), "
        f"val={len(y_val)} ({np.bincount(y_val).tolist()})"
    )

    # ── Train LogisticRegression ─────────────────────────────────
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=1000,
        solver="lbfgs",
    )
    lr.fit(X_train, y_train)
    print(f"[train_meta] LR coef={lr.coef_.ravel().tolist()} intercept={float(lr.intercept_[0])}")

    # ── Evaluate LR on 20% val ───────────────────────────────────
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        roc_auc_score, confusion_matrix, brier_score_loss,
    )

    val_probs_raw = lr.predict_proba(X_val)[:, 1]
    val_pred_raw = (val_probs_raw >= 0.5).astype(np.int32)
    raw_metrics = {
        "f1_at_0.5": float(f1_score(y_val, val_pred_raw, zero_division=0)),
        "precision_at_0.5": float(precision_score(y_val, val_pred_raw, zero_division=0)),
        "recall_at_0.5": float(recall_score(y_val, val_pred_raw, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_val, val_probs_raw)),
        "confusion_matrix_at_0.5": confusion_matrix(y_val, val_pred_raw).tolist(),
    }

    # ── Fit isotonic calibrator (prefit LR) ──────────────────────
    # sklearn 1.6+ deprecated `cv="prefit"` in favor of FrozenEstimator:
    # wrap the trained LR, pass to CalibratedClassifierCV without cv arg.
    # See https://scikit-learn.org/stable/modules/generated/sklearn.frozen.FrozenEstimator.html
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    frozen_lr = FrozenEstimator(lr)
    calibrator = CalibratedClassifierCV(frozen_lr, method="isotonic")
    calibrator.fit(X_val, y_val)
    cal_probs_val = calibrator.predict_proba(X_val)[:, 1]

    # ── Sweep threshold on calibrated probs ──────────────────────
    best_t, best_t_f1 = _sweep_threshold_for_f1(
        y_val, cal_probs_val, lo=0.30, hi=0.70, step=0.01
    )
    print(f"[train_meta] calibrated threshold sweep: best_t={best_t} best_f1={best_t_f1:.4f}")

    # Metrics at the fit threshold
    cal_pred = (cal_probs_val >= best_t).astype(np.int32)
    cal_metrics_at_t = {
        "threshold": best_t,
        "f1": float(f1_score(y_val, cal_pred, zero_division=0)),
        "precision": float(precision_score(y_val, cal_pred, zero_division=0)),
        "recall": float(recall_score(y_val, cal_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_val, cal_probs_val)),
        "confusion_matrix": confusion_matrix(y_val, cal_pred).tolist(),
    }

    # Metrics at the natural 0.5 threshold
    cal_pred_05 = (cal_probs_val >= 0.5).astype(np.int32)
    cal_metrics_at_05 = {
        "threshold": 0.5,
        "f1": float(f1_score(y_val, cal_pred_05, zero_division=0)),
        "precision": float(precision_score(y_val, cal_pred_05, zero_division=0)),
        "recall": float(recall_score(y_val, cal_pred_05, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_val, cal_pred_05).tolist(),
    }

    # ── Calibration quality ──────────────────────────────────────
    ece_raw = _expected_calibration_error(y_val, val_probs_raw)
    ece_cal = _expected_calibration_error(y_val, cal_probs_val)
    brier_raw = float(brier_score_loss(y_val, val_probs_raw))
    brier_cal = float(brier_score_loss(y_val, cal_probs_val))

    # Baseline Brier: using the CURRENT fixed-weight ensemble score
    # (final_score_current column) as the "probability" — clipped to
    # [0, 1] because it is already a combined score in that range.
    fsc_val_clipped = np.clip(fsc_val, 0.0, 1.0)
    brier_baseline = float(brier_score_loss(y_val, fsc_val_clipped))
    ece_baseline = _expected_calibration_error(y_val, fsc_val_clipped)

    print(
        f"[train_meta] calibration quality: "
        f"ECE raw={ece_raw:.4f} cal={ece_cal:.4f} baseline={ece_baseline:.4f}"
    )
    print(
        f"[train_meta] Brier score: "
        f"raw={brier_raw:.4f} cal={brier_cal:.4f} baseline={brier_baseline:.4f}"
    )

    # ── Save artifacts ───────────────────────────────────────────
    import joblib

    lr_path = args.model_dir / "meta_classifier.joblib"
    cal_path = args.model_dir / "meta_calibrator.joblib"

    joblib.dump(lr, lr_path, compress=3)
    print(f"[train_meta] wrote {lr_path} ({lr_path.stat().st_size} bytes)")

    # Save calibrator + threshold + feature order as a single bundle
    cal_bundle = {
        "calibrator": calibrator,
        "threshold": float(best_t),
        "feature_order": FEATURE_ORDER,
    }
    joblib.dump(cal_bundle, cal_path, compress=3)
    print(f"[train_meta] wrote {cal_path} ({cal_path.stat().st_size} bytes)")

    git_sha = _git_sha()
    timestamp = datetime.now(timezone.utc).isoformat()

    # meta_classifier_metrics.json — LR performance on the 20% val split
    meta_metrics_path = args.metrics_dir / "meta_classifier_metrics.json"
    meta_metrics = {
        "git_sha": git_sha,
        "timestamp_utc": timestamp,
        "meta_dataset": str(args.meta_dataset).replace("\\", "/"),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "train_label_distribution": [int(x) for x in np.bincount(y_train).tolist()],
        "val_label_distribution": [int(x) for x in np.bincount(y_val).tolist()],
        "feature_order": FEATURE_ORDER,
        "lr_coef": [float(x) for x in lr.coef_.ravel().tolist()],
        "lr_intercept": float(lr.intercept_[0]),
        "lr_class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "metrics_raw_lr_on_val": raw_metrics,
        "metrics_calibrated_on_val_at_fit_threshold": cal_metrics_at_t,
        "metrics_calibrated_on_val_at_0.5": cal_metrics_at_05,
    }
    with meta_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(meta_metrics, f, indent=2)
    print(f"[train_meta] wrote {meta_metrics_path}")

    # calibration_metrics.json — ECE + Brier
    cal_metrics_path = args.metrics_dir / "calibration_metrics.json"
    cal_metrics_report = {
        "git_sha": git_sha,
        "timestamp_utc": timestamp,
        "method": "isotonic",
        "cv": "prefit",
        "n_val": int(len(y_val)),
        "ece_bins": N_ECE_BINS,
        "ece_raw_lr": ece_raw,
        "ece_calibrated": ece_cal,
        "ece_baseline_fixed_weight": ece_baseline,
        "brier_raw_lr": brier_raw,
        "brier_calibrated": brier_cal,
        "brier_baseline_fixed_weight": brier_baseline,
        "brier_improvement_vs_baseline": round(brier_baseline - brier_cal, 6),
        "best_threshold": float(best_t),
        "threshold_search_range": [0.30, 0.70, 0.01],
        "notes": (
            "Baseline Brier is computed against `final_score_current` "
            "(the fixed-weight ensemble score already in meta_train.parquet), "
            "clipped to [0, 1]. Stage 2 acceptance requires brier_calibrated < "
            "brier_baseline_fixed_weight and ece_calibrated < 0.05."
        ),
    }
    with cal_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(cal_metrics_report, f, indent=2)
    print(f"[train_meta] wrote {cal_metrics_path}")

    # ── Console summary ──────────────────────────────────────────
    print()
    print("=== Stage 2 Meta Classifier Training Summary ===")
    print(f"LR val F1 @ 0.5:          {raw_metrics['f1_at_0.5']:.4f}")
    print(f"LR val ROC-AUC:           {raw_metrics['roc_auc']:.4f}")
    print(f"Calibrated val F1 @ {best_t:.2f}: {cal_metrics_at_t['f1']:.4f}")
    print(f"Calibrated val F1 @ 0.50: {cal_metrics_at_05['f1']:.4f}")
    print(f"ECE calibrated:           {ece_cal:.4f}  (target < 0.05)")
    print(f"ECE baseline:             {ece_baseline:.4f}")
    print(f"Brier calibrated:         {brier_cal:.4f}")
    print(f"Brier baseline:           {brier_baseline:.4f}")
    print(f"Brier improvement:        {brier_baseline - brier_cal:+.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
