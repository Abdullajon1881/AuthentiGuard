"""
Grid-search fit of (w1, w2, w3, threshold) for the 3-layer text ensemble.

Stage 1 artifact. Fits the combiner weights and the AI-class threshold
ON VALIDATION DATA ONLY. The test split is evaluated afterwards for
verification — the test F1 is REPORTED but NEVER used to re-pick
hyperparameters. This is the one rule that keeps test honest.

Approach:
  1. Run L1, L2, L3 once per row of val.parquet. Cache the per-layer
     scores to `fit_cache_val.npz` under the checkpoint dir. Same for
     test (used only for the verification read).
  2. Grid-search:
       - weights (w1, w2, w3) on a 0.05-step simplex, ~231 combos
       - threshold t ∈ {0.40, 0.41, …, 0.80}, 41 values
  3. Objective: maximize F1 on val for `(w1*l1 + w2*l2 + w3*l3) > t`.
     Tie-break: prefer higher w3 (L3 is the only trained component).
  4. Report val F1 (picked) and test F1 (verification only).
  5. Write results to --output JSON.

Usage:
    python scripts/fit_ensemble_weights.py \
        --checkpoint ai/text_detector/checkpoints/transformer_v3_hard/phase1 \
        --data-dir datasets/processed \
        --output ai/text_detector/checkpoints/transformer_v3_hard/phase1/fit_weights.json

Deterministic. Read-only against weights. Write-only to the cache and
the --output JSON. Does NOT edit text_detector.py — that's a manual
step after reviewing fit_weights.json.
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

# Ensure repo root on sys.path
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
    from ai.text_detector.ensemble.text_detector import TextDetector  # type: ignore
    detector = TextDetector(
        transformer_checkpoint=checkpoint,
        adversarial_checkpoint=None,
        meta_checkpoint=None,
        device=device,
    )
    detector.load_models()
    return detector


def _compute_layer_scores(
    detector: Any,
    parquet_path: Path,
    cache_path: Path,
    no_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Run L1, L2, L3 on every row. Cache to disk. Return (X, y).

    X has shape (n_rows, 3): columns are [l1_score, l2_score, l3_score].
    y has shape (n_rows,) with 0/1 labels.

    Cache invalidation: cache is rebuilt if it is older than the parquet
    file OR older than the checkpoint's model.safetensors.
    """
    import pyarrow.parquet as pq

    checkpoint_file = Path(_REPO_ROOT) / "ai" / "text_detector" / "checkpoints" / "transformer_v3_hard" / "phase1" / "model.safetensors"
    cache_mtime = cache_path.stat().st_mtime if cache_path.exists() else 0
    pq_mtime = parquet_path.stat().st_mtime
    ckpt_mtime = checkpoint_file.stat().st_mtime if checkpoint_file.exists() else 0
    if cache_path.exists() and cache_mtime > pq_mtime and cache_mtime > ckpt_mtime:
        print(f"[fit] using cached scores: {cache_path}")
        data = np.load(cache_path)
        return data["X"], data["y"]

    print(f"[fit] computing layer scores for {parquet_path}")
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()

    X = np.zeros((len(rows), 3), dtype=np.float64)
    y = np.zeros(len(rows), dtype=np.int32)

    try:
        if not no_progress:
            from tqdm import tqdm  # type: ignore
            iterator = tqdm(enumerate(rows), total=len(rows), desc="layer_scores", unit="row")
        else:
            iterator = enumerate(rows)
    except ImportError:
        iterator = enumerate(rows)

    for i, row in iterator:
        text = row["text"]
        y[i] = int(row["label"])
        # Run each layer via the same `analyze_safe` wrapper TextDetector
        # uses at inference. This guarantees the cached scores match what
        # production would produce.
        r1 = detector._layer1.analyze_safe(text)
        r2 = detector._layer2.analyze_safe(text)
        if detector._layer3 is not None:
            r3 = detector._layer3.analyze_safe(text)
        else:
            # Should not happen if checkpoint is present; fail loudly.
            raise RuntimeError("L3 not loaded — fit script requires the trained L3 checkpoint")
        X[i, 0] = float(r1.score) if not r1.error else 0.5
        X[i, 1] = float(r2.score) if not r2.error else 0.5
        X[i, 2] = float(r3.score) if not r3.error else 0.5

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, X=X, y=y)
    print(f"[fit] cached to {cache_path}")
    return X, y


def _simplex_weights(step: float) -> "list[tuple[float, float, float]]":
    """All (w1, w2, w3) on the 3-simplex with the given step size.

    step=0.05 produces ~231 combinations. Each combination sums to 1.0
    exactly (after the float-tolerance comparison below).
    """
    n_steps = int(round(1.0 / step))
    out: list[tuple[float, float, float]] = []
    for i in range(n_steps + 1):
        for j in range(n_steps + 1 - i):
            k = n_steps - i - j
            w1 = i * step
            w2 = j * step
            w3 = k * step
            out.append((round(w1, 6), round(w2, 6), round(w3, 6)))
    return out


def _grid_search(
    X_val: np.ndarray, y_val: np.ndarray,
    weight_step: float = 0.05,
    threshold_lo: float = 0.40,
    threshold_hi: float = 0.80,
    threshold_step: float = 0.01,
) -> dict[str, Any]:
    """Exhaustive search over (w, t). Returns best params + grid summary.

    F1 is computed as the AI-class F1 from binary predictions
    `(w1*l1 + w2*l2 + w3*l3) > t`. Ties broken by higher w3.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    thresholds = np.round(
        np.arange(threshold_lo, threshold_hi + 1e-9, threshold_step), 4
    )
    weights = _simplex_weights(weight_step)
    print(f"[fit] grid: {len(weights)} weight combos x {len(thresholds)} thresholds = "
          f"{len(weights) * len(thresholds)} evaluations")

    best_f1 = -1.0
    best_w: tuple[float, float, float] = (0.0, 0.0, 1.0)
    best_t: float = 0.5
    best_prec = 0.0
    best_rec = 0.0

    for (w1, w2, w3) in weights:
        # Vectorized: compute combined score once per weight, sweep threshold
        combined = w1 * X_val[:, 0] + w2 * X_val[:, 1] + w3 * X_val[:, 2]
        for t in thresholds:
            pred = (combined > t).astype(np.int32)
            f1 = f1_score(y_val, pred, zero_division=0)
            # Tie-break: strictly higher F1, or same F1 with higher w3.
            if (f1 > best_f1 + 1e-9) or (abs(f1 - best_f1) < 1e-9 and w3 > best_w[2]):
                best_f1 = float(f1)
                best_w = (float(w1), float(w2), float(w3))
                best_t = float(t)
                best_prec = float(precision_score(y_val, pred, zero_division=0))
                best_rec = float(recall_score(y_val, pred, zero_division=0))

    return {
        "best_weights_l1_l2_l3": list(best_w),
        "best_threshold": round(best_t, 4),
        "val_f1": round(best_f1, 6),
        "val_precision": round(best_prec, 6),
        "val_recall": round(best_rec, 6),
        "grid_size": len(weights) * len(thresholds),
    }


def _verify_on_test(
    X_test: np.ndarray, y_test: np.ndarray,
    weights: "list[float]", threshold: float,
) -> dict[str, float]:
    from sklearn.metrics import f1_score, precision_score, recall_score
    combined = weights[0] * X_test[:, 0] + weights[1] * X_test[:, 1] + weights[2] * X_test[:, 2]
    pred = (combined > threshold).astype(np.int32)
    return {
        "test_f1_verify_only": round(float(f1_score(y_test, pred, zero_division=0)), 6),
        "test_precision_verify_only": round(float(precision_score(y_test, pred, zero_division=0)), 6),
        "test_recall_verify_only": round(float(recall_score(y_test, pred, zero_division=0)), 6),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--weight-step", type=float, default=0.05)
    ap.add_argument("--threshold-lo", type=float, default=0.40)
    ap.add_argument("--threshold-hi", type=float, default=0.80)
    ap.add_argument("--threshold-step", type=float, default=0.01)
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    val_parquet = args.data_dir / "val.parquet"
    test_parquet = args.data_dir / "test.parquet"
    for p in (val_parquet, test_parquet):
        if not p.exists():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 2

    val_cache = args.checkpoint / "fit_cache_val.npz"
    test_cache = args.checkpoint / "fit_cache_test.npz"

    print(f"[fit] loading detector from {args.checkpoint}")
    t_load_start = time.time()
    detector = _build_production_detector(args.checkpoint, args.device)
    t_load_end = time.time()
    print(f"[fit] detector loaded in {t_load_end - t_load_start:.1f}s")

    X_val, y_val = _compute_layer_scores(detector, val_parquet, val_cache, args.no_progress)
    X_test, y_test = _compute_layer_scores(detector, test_parquet, test_cache, args.no_progress)

    print(f"[fit] val X shape={X_val.shape}  label dist={np.bincount(y_val).tolist()}")
    print(f"[fit] test X shape={X_test.shape}  label dist={np.bincount(y_test).tolist()}")

    t_search_start = time.time()
    fit = _grid_search(
        X_val, y_val,
        weight_step=args.weight_step,
        threshold_lo=args.threshold_lo,
        threshold_hi=args.threshold_hi,
        threshold_step=args.threshold_step,
    )
    t_search_end = time.time()
    print(f"[fit] grid search complete in {t_search_end - t_search_start:.2f}s")
    print(f"[fit] best val F1={fit['val_f1']:.4f}  "
          f"weights={fit['best_weights_l1_l2_l3']}  "
          f"threshold={fit['best_threshold']}")

    # Verification on test — reported only, NOT used for selection
    verify = _verify_on_test(
        X_test, y_test,
        fit["best_weights_l1_l2_l3"],
        fit["best_threshold"],
    )
    print(f"[fit] test verify F1 (not used for selection): {verify['test_f1_verify_only']:.4f}")

    report = {
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint).replace("\\", "/"),
        "val_split": str(val_parquet).replace("\\", "/"),
        "val_split_sha256": _sha256_file(val_parquet),
        "test_split": str(test_parquet).replace("\\", "/"),
        "test_split_sha256": _sha256_file(test_parquet),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "grid_size": fit["grid_size"],
        "search_seconds": round(t_search_end - t_search_start, 2),
        "total_seconds": round(time.time() - t_load_start, 2),
        "best_weights_l1_l2_l3": fit["best_weights_l1_l2_l3"],
        "best_threshold": fit["best_threshold"],
        "val_f1": fit["val_f1"],
        "val_precision": fit["val_precision"],
        "val_recall": fit["val_recall"],
        **verify,
        "val_test_f1_gap": round(fit["val_f1"] - verify["test_f1_verify_only"], 6),
        "notes": (
            "Weights and threshold selected on val only. Test F1 reported "
            "for verification — MUST NOT be used to re-pick hyperparameters. "
            "Tie-break rule: higher w3 preferred."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp_path, args.output)

    print(f"[fit] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
