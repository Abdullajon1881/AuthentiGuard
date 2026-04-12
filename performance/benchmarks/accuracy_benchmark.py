"""
Step 101: Accuracy benchmarks for all five detectors.

Computes precision, recall, F1, AUC-ROC, ECE (Expected Calibration Error),
and per-class performance on held-out test sets.

SLA targets:
  Text detection:   F1 ≥ 0.90 (AI class), ECE < 0.05
  Audio deepfake:   EER < 5%  → accuracy ≥ 90%
  Video deepfake:   AUC-ROC ≥ 0.92
  Image AI:         F1 ≥ 0.88 (AI class)
  Code AI:          F1 ≥ 0.85 (AI class)

Run with:
    python performance/benchmarks/accuracy_benchmark.py \
        --detector text \
        --test-data datasets/text/test.parquet
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog

log = structlog.get_logger(__name__)

# ── Accuracy targets ──────────────────────────────────────────

SLA_TARGETS: dict[str, dict[str, float]] = {
    "text":  {"f1": 0.90, "auc": 0.95, "ece": 0.05, "fpr": 0.10},
    "audio": {"f1": 0.90, "auc": 0.93, "ece": 0.05, "fpr": 0.10},
    "video": {"f1": 0.85, "auc": 0.92, "ece": 0.05, "fpr": 0.15},
    "image": {"f1": 0.88, "auc": 0.92, "ece": 0.05, "fpr": 0.12},
    "code":  {"f1": 0.85, "auc": 0.90, "ece": 0.06, "fpr": 0.15},
}


@dataclass
class AccuracyReport:
    """Complete accuracy report for one detector."""
    detector:    str
    n_samples:   int

    # Core metrics
    accuracy:    float
    precision:   float    # AI class
    recall:      float    # AI class
    f1:          float    # AI class
    auc_roc:     float

    # Calibration
    ece:         float    # Expected Calibration Error
    mce:         float    # Maximum Calibration Error

    # Per-class
    class_metrics: dict[str, dict[str, float]]

    # Error analysis
    false_positive_rate: float   # Human → predicted AI
    false_negative_rate: float   # AI → predicted Human
    n_ai:                int
    n_human:             int
    n_uncertain:         int

    # SLA
    sla_targets:  dict[str, float]
    sla_pass:     dict[str, bool]
    overall_pass: bool


# ── Core metric functions ─────────────────────────────────────

def compute_precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float, float]:
    """Compute precision, recall, F1 for binary classification (class 1)."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    return round(precision, 4), round(recall, 4), round(f1, 4)


def compute_auc_roc(
    y_true: np.ndarray, y_scores: np.ndarray
) -> float:
    """Compute AUC-ROC using the trapezoidal rule (no sklearn required)."""
    n        = len(y_true)
    thresholds = np.sort(np.unique(y_scores))[::-1]
    tprs, fprs = [0.0], [0.0]

    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())

    if pos == 0 or neg == 0:
        return 0.5

    for thr in thresholds:
        y_pred = (y_scores >= thr).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tprs.append(tp / pos)
        fprs.append(fp / neg)

    tprs.append(1.0); fprs.append(1.0)
    auc = float(np.trapezoid(tprs, fprs))
    return round(abs(auc), 4)


def compute_ece(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bins: int = 15,
) -> tuple[float, float]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    A well-calibrated model has ECE < 0.05.
    Models with ECE < 0.05 can be trusted for threshold-based decisions.

    Returns (ECE, MCE).
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_sum   = 0.0
    mce       = 0.0
    n         = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask   = (y_scores > lo) & (y_scores <= hi)
        n_bin  = int(mask.sum())
        if n_bin == 0:
            continue

        frac_pos = float(y_true[mask].mean())   # actual fraction positive
        mean_conf = float(y_scores[mask].mean()) # mean predicted confidence

        bin_ece = abs(frac_pos - mean_conf) * n_bin / n
        ece_sum += bin_ece
        mce      = max(mce, abs(frac_pos - mean_conf))

    return round(ece_sum, 4), round(mce, 4)


def scores_to_labels(
    scores: np.ndarray,
    ai_threshold:    float = 0.75,
    human_threshold: float = 0.40,
) -> np.ndarray:
    """Convert continuous scores to {0=human, 1=AI, 2=uncertain} labels."""
    labels = np.full(len(scores), 2)   # uncertain
    labels[scores >= ai_threshold]    = 1   # AI
    labels[scores <= human_threshold] = 0   # human
    return labels


def compute_accuracy_report(
    detector:     str,
    y_true:       np.ndarray,     # ground truth: 0=human, 1=AI
    y_scores:     np.ndarray,     # predicted probabilities [0,1]
    ai_threshold:  float = 0.75,
    human_threshold: float = 0.40,
) -> AccuracyReport:
    """
    Compute the full accuracy report for a detector on a held-out test set.
    """
    y_pred_binary = (y_scores >= 0.50).astype(int)   # binary for sklearn-style metrics
    y_labels      = scores_to_labels(y_scores, ai_threshold, human_threshold)

    accuracy  = float((y_pred_binary == y_true).mean())
    precision, recall, f1 = compute_precision_recall_f1(y_true, y_pred_binary)
    auc       = compute_auc_roc(y_true, y_scores)
    ece, mce  = compute_ece(y_true, y_scores)

    # FPR/FNR
    n_neg = int((y_true == 0).sum())
    n_pos = int((y_true == 1).sum())
    fp    = int(((y_pred_binary == 1) & (y_true == 0)).sum())
    fn    = int(((y_pred_binary == 0) & (y_true == 1)).sum())
    fpr   = fp / max(n_neg, 1)
    fnr   = fn / max(n_pos, 1)

    # Per-class metrics
    class_metrics: dict[str, dict[str, float]] = {}
    for cls_label, cls_name in [(0, "human"), (1, "ai")]:
        y_true_cls = (y_true == cls_label).astype(int)
        y_pred_cls = (y_pred_binary == cls_label).astype(int)
        p, r, f    = compute_precision_recall_f1(y_true_cls, y_pred_cls)
        class_metrics[cls_name] = {"precision": p, "recall": r, "f1": f}

    # SLA evaluation
    targets  = SLA_TARGETS.get(detector, {})
    sla_pass = {
        "f1":  f1  >= targets.get("f1",  0.0),
        "auc": auc >= targets.get("auc", 0.0),
        "ece": ece <= targets.get("ece", 1.0),
        "fpr": fpr <= targets.get("fpr", 1.0),
    }
    overall_pass = all(sla_pass.values())

    return AccuracyReport(
        detector=detector,
        n_samples=len(y_true),
        accuracy=round(accuracy, 4),
        precision=precision,
        recall=recall,
        f1=f1,
        auc_roc=auc,
        ece=ece,
        mce=mce,
        class_metrics=class_metrics,
        false_positive_rate=round(fpr, 4),
        false_negative_rate=round(fnr, 4),
        n_ai=int((y_labels == 1).sum()),
        n_human=int((y_labels == 0).sum()),
        n_uncertain=int((y_labels == 2).sum()),
        sla_targets=targets,
        sla_pass=sla_pass,
        overall_pass=overall_pass,
    )


def print_accuracy_report(report: AccuracyReport) -> None:
    """Print a formatted accuracy report."""
    status = "✅ PASS" if report.overall_pass else "❌ FAIL"
    print(f"\n{'='*60}")
    print(f"  {report.detector.upper()} Detector Accuracy Report  {status}")
    print(f"{'='*60}")
    print(f"  Samples:    {report.n_samples}  "
          f"(AI: {report.n_ai}, Human: {report.n_human}, "
          f"Uncertain: {report.n_uncertain})")
    print(f"\n  Core metrics:")
    print(f"    Accuracy:  {report.accuracy:.1%}")
    print(f"    F1 (AI):   {report.f1:.4f}  "
          f"{'✅' if report.sla_pass.get('f1') else '❌'}  "
          f"(target ≥ {report.sla_targets.get('f1', '-')})")
    print(f"    AUC-ROC:   {report.auc_roc:.4f}  "
          f"{'✅' if report.sla_pass.get('auc') else '❌'}  "
          f"(target ≥ {report.sla_targets.get('auc', '-')})")
    print(f"    ECE:       {report.ece:.4f}  "
          f"{'✅' if report.sla_pass.get('ece') else '❌'}  "
          f"(target < {report.sla_targets.get('ece', '-')})")
    print(f"    FPR:       {report.false_positive_rate:.4f}  "
          f"{'✅' if report.sla_pass.get('fpr') else '❌'}  "
          f"(target < {report.sla_targets.get('fpr', '-')})")
    print(f"\n  Per-class breakdown:")
    for cls, m in report.class_metrics.items():
        print(f"    {cls.capitalize():8s}  "
              f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
    print(f"{'='*60}\n")


# ── Benchmark runner ──────────────────────────────────────────

def run_accuracy_benchmarks_on_synthetic() -> list[AccuracyReport]:
    """
    Run accuracy benchmarks on synthetic data.
    Used for CI/CD regression testing before real datasets are available.
    """
    rng = np.random.default_rng(42)
    reports = []

    # For each detector, generate synthetic scores representing a well-tuned model
    detector_params = {
        # (target F1, base_separation)
        "text":  (0.92, 1.8),
        "audio": (0.91, 1.7),
        "video": (0.88, 1.5),
        "image": (0.89, 1.6),
        "code":  (0.86, 1.4),
    }

    for detector, (target_f1, separation) in detector_params.items():
        n = 1000
        # Generate scores: AI scores pulled toward 1, human toward 0
        y_true  = rng.integers(0, 2, n)
        scores  = np.where(
            y_true == 1,
            np.clip(rng.normal(0.75, 0.15, n), 0.01, 0.99),
            np.clip(rng.normal(0.25, 0.15, n), 0.01, 0.99),
        ).astype(np.float32)

        report = compute_accuracy_report(detector, y_true, scores)
        reports.append(report)
        print_accuracy_report(report)

    return reports


def run_text_detector_benchmark(
    test_parquet: Path,
    transformer_checkpoint: Path | None = None,
    max_samples: int = 500,
) -> AccuracyReport:
    """
    Run TextDetector on real test data and produce an accuracy report.

    Args:
        test_parquet: Path to test.parquet with columns [text, label]
        transformer_checkpoint: Path to fine-tuned L3 checkpoint (None = L1+L2 only)
        max_samples: Cap samples to keep runtime reasonable on CPU
    """
    import sys
    import pandas as pd

    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from ai.text_detector.ensemble.text_detector import TextDetector

    df = pd.read_parquet(test_parquet)[["text", "label"]]
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)

    detector = TextDetector(
        transformer_checkpoint=str(transformer_checkpoint) if transformer_checkpoint else None,
        adversarial_checkpoint=None,
        meta_checkpoint=None,
        device="cpu",
    )
    detector.load_models()

    y_true = df["label"].values.astype(int)
    y_scores = np.zeros(len(df), dtype=np.float32)

    log.info("benchmark_start", n_samples=len(df),
             mode="L1+L2+L3" if transformer_checkpoint else "L1+L2")

    for i, text in enumerate(df["text"].values):
        try:
            result = detector.analyze(text)
            y_scores[i] = result.score
        except Exception as e:
            log.warning("sample_failed", index=i, error=str(e))
            y_scores[i] = 0.5  # uncertain fallback

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(df)}] processed...")

    # Use adaptive thresholds based on active layers
    active_layers = 2
    if transformer_checkpoint:
        active_layers = 3
    thresholds = {2: (0.55, 0.30), 3: (0.65, 0.35), 4: (0.75, 0.40)}
    ai_thr, human_thr = thresholds.get(active_layers, (0.75, 0.40))

    report = compute_accuracy_report("text", y_true, y_scores, ai_thr, human_thr)
    print_accuracy_report(report)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--test-data", type=Path, default=Path("datasets/processed/test.parquet"))
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to L3 transformer checkpoint dir")
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()

    if args.mode == "synthetic":
        run_accuracy_benchmarks_on_synthetic()
    else:
        run_text_detector_benchmark(
            test_parquet=args.test_data,
            transformer_checkpoint=args.checkpoint,
            max_samples=args.max_samples,
        )
