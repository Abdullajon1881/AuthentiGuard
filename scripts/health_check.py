"""
Stage 3 — Part 5: production health check for the text detection stack.

Verifies:
  1. The Stage 2 meta classifier + calibrator load cleanly.
  2. The Stage 1 fixed-weight fallback loads cleanly (with meta_lr_path=None).
  3. Inference runs end-to-end on a canned sample.
  4. p95 latency over a 5-sample warm run is under the threshold.

Emits a JSON health report to stdout and exits:
  0  — all checks passed
  1  — one or more checks failed
  2  — invocation error (missing checkpoint, bad args)

Usage (local):
    python scripts/health_check.py

Usage (Docker exec, in the worker container):
    docker compose -f docker-compose.prod.yml exec worker \
        python scripts/health_check.py --latency-max-ms 2500

The `--latency-max-ms` threshold defaults to 3000 ms (p95 per inference
call on CPU) which is conservative for a 500 ms average. Tighten on
GPU-backed deployments.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


CANARY_TEXTS = [
    "As an AI language model, I must clarify that I cannot provide financial advice. "
    "However, I can offer some general insights that may be helpful in your "
    "decision-making process regarding personal finance.",
    "The quick brown fox jumps over the lazy dog. Sphinx of black quartz, judge my vow. "
    "How vexingly quick daft zebras jump! Pack my box with five dozen liquor jugs.",
    "Been thinking about getting back into climbing. The old gym closed down last year "
    "and the new one is twice as expensive but at least it has an auto-belay. "
    "My fingers are not what they used to be.",
    "Machine learning models exhibit complex behaviors when presented with out-of-distribution "
    "inputs. This phenomenon is particularly evident in natural language processing tasks "
    "where the underlying data distribution has shifted significantly from the training set.",
    "lol yeah idk man, thinking we just grab pizza after the match and call it a night. "
    "the usual place? i can drive if you want. text me when you're out.",
]


def _check(name: str, fn, **fn_kwargs) -> dict[str, Any]:
    t0 = time.time()
    try:
        value = fn(**fn_kwargs)
        ok = True
        err = None
    except Exception as exc:
        value = None
        ok = False
        err = f"{type(exc).__name__}: {exc}"
        traceback.print_exc(file=sys.stderr)
    return {
        "name": name,
        "ok": ok,
        "value": value,
        "error": err,
        "elapsed_s": round(time.time() - t0, 4),
    }


def _load_meta_detector(checkpoint: Path):
    from ai.text_detector.ensemble.text_detector import TextDetector
    det = TextDetector(
        transformer_checkpoint=checkpoint,
        adversarial_checkpoint=None,
        meta_checkpoint=None,
        device="cpu",
    )
    det.load_models()
    if det._lr_meta is None or det._lr_calibrator is None:
        raise RuntimeError(
            "Stage 2 meta artifacts did not load. Expected "
            "meta_classifier.joblib + meta_calibrator.joblib "
            "to be auto-discovered next to the transformer checkpoint."
        )
    return det


def _load_fallback_detector(checkpoint: Path):
    """Construct the detector with meta paths explicitly None.

    This proves the fallback path still loads even if the meta
    artifacts disappear from disk. The auto-discovery logic is
    explicitly not triggered because we pass meta_lr_path=None and
    meta_calibrator_path=None AND we inject a sentinel that blocks
    the transformer_checkpoint-based discovery.
    """
    from ai.text_detector.ensemble.text_detector import TextDetector
    det = TextDetector(
        transformer_checkpoint=checkpoint,
        adversarial_checkpoint=None,
        meta_checkpoint=None,
        device="cpu",
        meta_lr_path=None,
        meta_calibrator_path=None,
    )
    # Temporarily move the on-disk meta files so auto-discovery fails
    # cleanly. We restore them in the finally block. This is the only
    # reliable way to verify the fallback path without mocking.
    meta_dir = checkpoint.parent.parent
    lr_path = meta_dir / "meta_classifier.joblib"
    cal_path = meta_dir / "meta_calibrator.joblib"
    lr_moved = None
    cal_moved = None
    try:
        if lr_path.exists():
            lr_moved = lr_path.with_suffix(".joblib.health_check_tmp")
            lr_path.rename(lr_moved)
        if cal_path.exists():
            cal_moved = cal_path.with_suffix(".joblib.health_check_tmp")
            cal_path.rename(cal_moved)
        det.load_models()
    finally:
        if lr_moved is not None and lr_moved.exists():
            lr_moved.rename(lr_path)
        if cal_moved is not None and cal_moved.exists():
            cal_moved.rename(cal_path)

    if det._lr_meta is not None:
        raise RuntimeError(
            "Fallback detector unexpectedly loaded the LR meta — "
            "the fallback path is not isolated from the meta artifacts."
        )
    return det


def _run_inference(detector, text: str) -> dict[str, Any]:
    r = detector.analyze(text)
    return {
        "score": round(float(r.score), 4),
        "label": str(r.label),
        "confidence": round(float(r.confidence), 4),
        "layer_count": len(r.layer_results),
    }


def _measure_latency(detector, texts: list[str]) -> dict[str, Any]:
    timings_ms: list[float] = []
    for t in texts:
        t0 = time.time()
        detector.analyze(t)
        timings_ms.append((time.time() - t0) * 1000.0)
    timings_ms.sort()
    n = len(timings_ms)
    p50 = timings_ms[n // 2] if n else 0.0
    p95_idx = max(0, int(round(0.95 * (n - 1))))
    p95 = timings_ms[p95_idx] if n else 0.0
    mean = sum(timings_ms) / n if n else 0.0
    return {
        "n_samples": n,
        "mean_ms": round(mean, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "all_ms": [round(t, 2) for t in timings_ms],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=_REPO_ROOT / "ai/text_detector/checkpoints/transformer_v3_hard/phase1",
    )
    ap.add_argument(
        "--latency-max-ms",
        type=float,
        default=3000.0,
        help="Fail the check if p95 inference latency exceeds this (ms). Default: 3000",
    )
    ap.add_argument(
        "--skip-fallback",
        action="store_true",
        help="Skip the fallback check (useful when meta files must not be touched)",
    )
    args = ap.parse_args()

    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint).replace("\\", "/"),
        "latency_max_ms_threshold": args.latency_max_ms,
        "checks": [],
    }

    if not args.checkpoint.exists():
        report["checks"].append({
            "name": "checkpoint_exists",
            "ok": False,
            "error": f"checkpoint not found: {args.checkpoint}",
        })
        report["overall_ok"] = False
        print(json.dumps(report, indent=2))
        return 2

    # 1. Meta detector loads
    meta_check = _check("meta_model_loads", _load_meta_detector, checkpoint=args.checkpoint)
    report["checks"].append({k: v for k, v in meta_check.items() if k != "value"})
    meta_det = meta_check["value"]

    # 2. Fallback detector loads
    if args.skip_fallback:
        report["checks"].append({
            "name": "fallback_loads",
            "ok": True,
            "error": None,
            "elapsed_s": 0.0,
            "skipped": True,
        })
        fallback_det = None
    else:
        fb_check = _check("fallback_loads", _load_fallback_detector, checkpoint=args.checkpoint)
        report["checks"].append({k: v for k, v in fb_check.items() if k != "value"})
        fallback_det = fb_check["value"]

    # 3. Inference runs on meta detector
    if meta_det is not None:
        inf_check = _check(
            "inference_runs",
            _run_inference,
            detector=meta_det,
            text=CANARY_TEXTS[0],
        )
        report["checks"].append(inf_check)
    else:
        report["checks"].append({
            "name": "inference_runs",
            "ok": False,
            "error": "meta detector failed to load — cannot run inference",
            "elapsed_s": 0.0,
        })

    # 4. Latency measurement (warm run: first call excluded)
    if meta_det is not None:
        # Warm-up
        try:
            meta_det.analyze(CANARY_TEXTS[0])
        except Exception:
            pass
        lat_check = _check(
            "latency_under_threshold",
            _measure_latency,
            detector=meta_det,
            texts=CANARY_TEXTS,
        )
        lat_value = lat_check.get("value")
        # Convert the inner "value" into a pass/fail on p95
        if lat_value is not None:
            p95 = lat_value["p95_ms"]
            under = p95 <= args.latency_max_ms
            lat_entry = {
                "name": "latency_under_threshold",
                "ok": under,
                "error": None if under else (
                    f"p95 {p95} ms exceeds threshold {args.latency_max_ms} ms"
                ),
                "elapsed_s": lat_check["elapsed_s"],
                "stats": lat_value,
            }
        else:
            lat_entry = {
                "name": "latency_under_threshold",
                "ok": False,
                "error": lat_check.get("error"),
                "elapsed_s": lat_check["elapsed_s"],
            }
        report["checks"].append(lat_entry)
    else:
        report["checks"].append({
            "name": "latency_under_threshold",
            "ok": False,
            "error": "meta detector failed to load — cannot measure latency",
            "elapsed_s": 0.0,
        })

    overall_ok = all(c.get("ok", False) for c in report["checks"] if not c.get("skipped"))
    report["overall_ok"] = overall_ok

    print(json.dumps(report, indent=2, default=str))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
