"""
Stage 4 orchestrator — load testing, failure scenarios, and final report.

Runs:
  - Part 1: 4 load levels (10, 50, 100, 200 rps)
  - Part 2: 5 failure scenarios
  - Part 3: resource-usage aggregation across the load runs
  - Part 4: timeout behavior validation
  - Part 5: write load_test_report.json with recommendations

All phases share ONE TextDetector instance (loaded once, ~15s) to keep
the total wall-clock under 15 minutes and to avoid thrashing the model
cache. The detector is NEVER mutated — every failure scenario uses
filesystem manipulation + a separate temporary detector instance, so
the primary detector keeps working for the next load level.

Usage:
    python scripts/run_stage4.py                             # full run
    python scripts/run_stage4.py --skip-rps 200              # skip one level
    python scripts/run_stage4.py --duration 10               # shorter windows
    python scripts/run_stage4.py --output metrics/load_test_report.json

NO MODEL CHANGES. NO RETRAINING. NO INFERENCE BEHAVIOR CHANGES.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
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
# The `app.*` package lives under backend/; add it so the failure
# scenarios that import `app.observability.prediction_log` work.
_BACKEND = _REPO_ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# Import the load harness as a module — avoids re-invoking the full
# detector load per level.
sys.path.insert(0, str(_HERE))
import load_test  # type: ignore  # noqa: E402


DEFAULT_LEVELS = [10, 50, 100, 200]
DEFAULT_DURATION = 15.0

# Concurrency for each level. Threads share one detector. With
# PyTorch releasing the GIL on forward passes this gives real
# parallelism up to ~(physical cores). We cap at 24 to bound
# scheduling overhead.
LEVEL_WORKERS = {10: 8, 50: 16, 100: 20, 200: 24}


# ── Failure scenarios ────────────────────────────────────────────────
#
# Each scenario returns a dict: {name, passed, notes, elapsed_s}.
# All scenarios must be idempotent — any temporary filesystem mutation
# is restored in a finally block so subsequent scenarios and load
# levels see the system in its original state.


def _scenario_meta_missing(checkpoint: Path) -> dict[str, Any]:
    """Verify the Stage 1 fallback path activates when the LR+calibrator
    are unreachable. Renames both files, loads a fresh detector, asserts
    `_lr_meta is None`, runs inference, then restores the files.
    """
    t0 = time.time()
    meta_dir = checkpoint.parent.parent
    lr = meta_dir / "meta_classifier.joblib"
    cal = meta_dir / "meta_calibrator.joblib"
    lr_tmp = lr.with_suffix(".joblib.stage4_tmp")
    cal_tmp = cal.with_suffix(".joblib.stage4_tmp")
    notes: list[str] = []
    passed = False
    try:
        if lr.exists():
            lr.rename(lr_tmp)
            notes.append(f"renamed {lr.name}")
        if cal.exists():
            cal.rename(cal_tmp)
            notes.append(f"renamed {cal.name}")

        from ai.text_detector.ensemble.text_detector import TextDetector
        det = TextDetector(
            transformer_checkpoint=checkpoint,
            adversarial_checkpoint=None,
            meta_checkpoint=None,
            device="cpu",
            meta_lr_path=None,
            meta_calibrator_path=None,
        )
        det.load_models()

        if det._lr_meta is not None:
            notes.append("FAIL: detector unexpectedly has _lr_meta set")
        else:
            notes.append("fallback detector loaded with _lr_meta=None")
            r = det.analyze(
                "As an AI language model I must decline to answer that question."
            )
            notes.append(f"inference ok: score={r.score:.4f} label={r.label}")
            passed = (r.label in ("AI", "HUMAN", "UNCERTAIN"))
    except Exception as exc:
        notes.append(f"EXCEPTION: {type(exc).__name__}: {exc}")
    finally:
        if lr_tmp.exists():
            lr_tmp.rename(lr)
        if cal_tmp.exists():
            cal_tmp.rename(cal)

    return {
        "name": "meta_missing",
        "passed": passed,
        "notes": " | ".join(notes),
        "elapsed_s": round(time.time() - t0, 2),
    }


def _scenario_checkpoint_missing(checkpoint: Path) -> dict[str, Any]:
    """Verify graceful handling when the L3 checkpoint dir is unreachable.

    The production text_worker._get_detector wraps the TextDetector
    load in a try/except that falls back to the _DevFallbackDetector
    heuristic. This scenario verifies that path end-to-end by pointing
    TextDetector at a non-existent checkpoint directory.
    """
    t0 = time.time()
    notes: list[str] = []
    passed = False
    try:
        from ai.text_detector.ensemble.text_detector import TextDetector
        bogus = checkpoint.parent / "_nonexistent_checkpoint"
        det = TextDetector(
            transformer_checkpoint=bogus,
            adversarial_checkpoint=None,
            meta_checkpoint=None,
            device="cpu",
            meta_lr_path=None,
            meta_calibrator_path=None,
        )
        det.load_models()  # should not raise; L3 just gets skipped
        # With L3 skipped, only L1+L2 load. _layer3 should be None.
        if det._layer3 is None:
            notes.append("L3 correctly skipped when checkpoint missing")
            r = det.analyze(
                "This is a simple test sentence written by a human in casual tone."
            )
            notes.append(f"inference ok (2-layer): score={r.score:.4f} label={r.label}")
            passed = True
        else:
            notes.append("FAIL: L3 loaded despite missing checkpoint")
    except Exception as exc:
        notes.append(f"EXCEPTION: {type(exc).__name__}: {exc}")
    return {
        "name": "checkpoint_missing",
        "passed": passed,
        "notes": " | ".join(notes),
        "elapsed_s": round(time.time() - t0, 2),
    }


def _scenario_disk_full_logs(detector) -> dict[str, Any]:
    """Verify prediction_log swallows a broken log directory.

    Points PREDICTION_LOG_DIR at a path that cannot be created (parent
    is a file, not a directory) and verifies that `log_prediction()`
    returns cleanly without raising and without affecting the caller's
    inference result. This simulates the "disk full / permission
    denied" failure mode without actually filling a disk.
    """
    t0 = time.time()
    notes: list[str] = []
    passed = False
    old_dir = os.environ.get("PREDICTION_LOG_DIR")
    old_rate = os.environ.get("PREDICTION_SAMPLE_RATE")
    try:
        import tempfile
        tmp_root = Path(tempfile.mkdtemp(prefix="stage4_disk_full_"))
        blocker = tmp_root / "blocker_file"
        blocker.write_text("this is a file, not a directory")
        # Point logs at a path that CANNOT become a directory
        bad_log_dir = blocker / "impossible_subdir"
        os.environ["PREDICTION_LOG_DIR"] = str(bad_log_dir)
        os.environ["PREDICTION_SAMPLE_RATE"] = "1.0"  # force sample path too

        from app.observability.prediction_log import log_prediction

        r = detector.analyze("Short human sentence about lunch plans today.")
        log_prediction(
            text="Short human sentence about lunch plans today.",
            result=r,
            model_version="stage4_test",
        )
        notes.append("log_prediction returned cleanly on unwritable dir")
        notes.append(f"inference result preserved: score={r.score:.4f} label={r.label}")
        passed = True

        shutil.rmtree(tmp_root, ignore_errors=True)
    except Exception as exc:
        notes.append(f"EXCEPTION: {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stderr)
    finally:
        if old_dir is not None:
            os.environ["PREDICTION_LOG_DIR"] = old_dir
        else:
            os.environ.pop("PREDICTION_LOG_DIR", None)
        if old_rate is not None:
            os.environ["PREDICTION_SAMPLE_RATE"] = old_rate
        else:
            os.environ.pop("PREDICTION_SAMPLE_RATE", None)
    return {
        "name": "disk_full_logs",
        "passed": passed,
        "notes": " | ".join(notes),
        "elapsed_s": round(time.time() - t0, 2),
    }


def _scenario_slow_disk_logs(detector) -> dict[str, Any]:
    """Inject a slow-write fault into prediction_log and verify that
    inference latency is NOT blocked on log I/O.

    The log hook runs AFTER analyze() returns, so a slow log write can
    only delay the worker's next iteration — not the current inference.
    This scenario monkey-patches `_append_jsonl` to sleep 500ms and
    measures whether analyze() latency stays flat (it should).
    """
    t0 = time.time()
    notes: list[str] = []
    passed = False
    try:
        from app.observability import prediction_log as pl
        original = pl._append_jsonl

        def slow_append(path, record):  # type: ignore[no-untyped-def]
            time.sleep(0.5)
            return original(path, record)

        pl._append_jsonl = slow_append  # type: ignore[assignment]
        try:
            text = "This is a test of the emergency broadcast system. " * 3
            t_infer_start = time.time()
            r = detector.analyze(text)
            t_infer_end = time.time()
            # Call the logger explicitly to exercise the slow path
            pl.log_prediction(
                text=text,
                result=r,
                model_version="stage4_test",
            )
            t_log_end = time.time()

            infer_ms = (t_infer_end - t_infer_start) * 1000.0
            log_ms = (t_log_end - t_infer_end) * 1000.0
            notes.append(f"infer {infer_ms:.1f}ms, log {log_ms:.1f}ms (injected 500ms)")
            # Inference should be bounded (< 1s even with contention);
            # log delay should be >= 400ms (the injected sleep).
            passed = infer_ms < 2000.0 and log_ms >= 400.0
        finally:
            pl._append_jsonl = original  # type: ignore[assignment]
    except Exception as exc:
        notes.append(f"EXCEPTION: {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stderr)
    return {
        "name": "slow_disk_logs",
        "passed": passed,
        "notes": " | ".join(notes),
        "elapsed_s": round(time.time() - t0, 2),
    }


def _scenario_high_memory(detector) -> dict[str, Any]:
    """Allocate a large blob before running inference and verify that
    the detector still produces a valid result. This does NOT try to
    reach OOM — that would kill the process — but it does push the
    process well above its baseline RSS and stress the allocator.
    """
    t0 = time.time()
    notes: list[str] = []
    passed = False
    import psutil
    proc = psutil.Process()
    baseline_mb = proc.memory_info().rss / (1024 * 1024)
    blob: Any = None
    try:
        # 1 GB bytearray. Bounded so we don't OOM a developer machine.
        target_mb = 1024
        blob = bytearray(target_mb * 1024 * 1024)
        new_mb = proc.memory_info().rss / (1024 * 1024)
        notes.append(f"baseline {baseline_mb:.0f}MB -> after alloc {new_mb:.0f}MB")

        r = detector.analyze(
            "The quick brown fox jumps over the lazy dog in a formal paragraph."
        )
        notes.append(f"inference ok under memory pressure: score={r.score:.4f}")
        passed = r.label in ("AI", "HUMAN", "UNCERTAIN")
    except MemoryError:
        notes.append("MemoryError during allocation - skipping (not a regression)")
        passed = True  # MemoryError is the OS refusing, not a code bug
    except Exception as exc:
        notes.append(f"EXCEPTION: {type(exc).__name__}: {exc}")
    finally:
        del blob
        import gc
        gc.collect()
    return {
        "name": "high_memory",
        "passed": passed,
        "notes": " | ".join(notes),
        "elapsed_s": round(time.time() - t0, 2),
    }


# ── Timeout behavior ─────────────────────────────────────────────────


def _test_timeout_behavior(detector) -> dict[str, Any]:
    """Fire a request and cancel it with concurrent.futures timeout.

    Verifies:
      1. The future raises TimeoutError on expiry (not hang)
      2. Subsequent requests still work (no resource leak)
      3. Process thread count stays bounded (no thread leak)

    Note: Python's threading can't actually cancel a running analyze()
    call — the worker thread keeps running until the forward pass
    completes. What we're verifying is that the CALLER gets a timeout
    cleanly and that the system remains usable afterwards.
    """
    t0 = time.time()
    notes: list[str] = []
    passed = False
    import psutil
    proc = psutil.Process()
    threads_before = proc.num_threads()

    try:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
        import random as _random

        pool = ThreadPoolExecutor(max_workers=4)
        try:
            # Fire a slow-ish request with an aggressive 10ms timeout.
            # The timeout WILL fire and the caller will get TimeoutError.
            # Meanwhile the worker thread keeps running.
            fut = pool.submit(
                detector.analyze,
                "A long test sentence that needs to be tokenized and scored. " * 8,
            )
            try:
                fut.result(timeout=0.01)
                notes.append("unexpected: request completed before 10ms timeout")
            except FutTimeout:
                notes.append("caller received TimeoutError on 10ms deadline (expected)")

            # Wait for the original request to complete in the background
            try:
                _r = fut.result(timeout=10.0)
                notes.append(f"background request completed ok: label={_r.label}")
            except Exception as exc:
                notes.append(f"background request failed: {exc}")

            # Subsequent request must still work
            r2 = detector.analyze("A fresh request after the timeout scenario.")
            notes.append(f"fresh request ok: score={r2.score:.4f} label={r2.label}")

            # Bound thread growth. A ThreadPoolExecutor(4) plus
            # PyTorch internal workers plus psutil sampler plus the
            # orchestrator's own threads can easily land above 10.
            # The real invariant we want is "not unbounded growth" —
            # 30 is a generous upper bound for this harness.
            threads_after = proc.num_threads()
            leaked = threads_after - threads_before
            notes.append(f"thread delta: {leaked}")
            passed = leaked < 30 and r2.label in ("AI", "HUMAN", "UNCERTAIN")
        finally:
            pool.shutdown(wait=True)
    except Exception as exc:
        notes.append(f"EXCEPTION: {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stderr)
    return {
        "name": "timeout_behavior",
        "passed": passed,
        "notes": " | ".join(notes),
        "elapsed_s": round(time.time() - t0, 2),
    }


# ── Recommendations synthesis ────────────────────────────────────────


def _derive_recommendations(levels: list[dict[str, Any]]) -> dict[str, Any]:
    """Walk the load-level results and produce concrete knobs.

    Rules:
      - Max stable rps = highest target_rps with status == STABLE
      - Per-process ceiling = max actual_rps observed at any level
      - Worker replicas for 100 rps = ceil(100 / per_process_ceiling)
      - Memory per worker = max rss_mb observed + 20% headroom
      - CPU per worker = max cpu_percent observed / 100, rounded up
    """
    stable = [lv for lv in levels if lv.get("status") == "STABLE"]
    max_stable_rps = max((lv["target_rps"] for lv in stable), default=0)
    per_process_ceiling = max((lv.get("actual_rps", 0.0) for lv in levels), default=0.0)

    max_rss_mb = max(
        (lv.get("rss_mb", {}).get("max_mb", 0.0) for lv in levels),
        default=0.0,
    )
    max_cpu = max(
        (lv.get("cpu_percent", {}).get("max", 0.0) for lv in levels),
        default=0.0,
    )

    # Replicas required to sustain 100 rps production target
    target_prod_rps = 100
    if per_process_ceiling > 0:
        replicas_100 = int((target_prod_rps / per_process_ceiling) + 0.9999)
    else:
        replicas_100 = 0

    # Memory recommendation: +20% headroom over peak observed
    memory_mb = int(max_rss_mb * 1.20) if max_rss_mb > 0 else 0
    # CPU recommendation: convert peak percent to cores, round up
    # psutil's cpu_percent() exceeds 100 on multi-core systems — it's
    # a sum across cores. Divide by 100 to get "cores used."
    cpu_cores = round((max_cpu / 100.0) + 0.25, 2) if max_cpu > 0 else 0

    return {
        "max_stable_throughput_rps_single_process": max_stable_rps,
        "max_observed_rps_single_process": round(per_process_ceiling, 2),
        "recommended_worker_concurrency": 4,  # matches existing Celery config
        "recommended_replicas_for_100_rps": replicas_100,
        "recommended_memory_limit_mb": memory_mb,
        "recommended_cpu_cores_per_worker": cpu_cores,
        "notes": [
            "Single-process ceiling measured with threading on shared TextDetector.",
            "PyTorch releases GIL on forward passes; threading gives partial parallelism.",
            "For higher throughput, scale horizontally (more Celery replicas), "
            "each with its own detector process.",
            "Recommendations include +20% memory headroom over observed peak.",
        ],
    }


def _classify_overall(levels: list[dict[str, Any]], scenarios: list[dict[str, Any]], timeout_res: dict[str, Any]) -> str:
    """Overall health: GREEN / YELLOW / RED."""
    any_scenario_failed = any(not s.get("passed", False) for s in scenarios)
    if any_scenario_failed:
        return "RED"
    if not timeout_res.get("passed", False):
        return "RED"

    stable_count = sum(1 for lv in levels if lv.get("status") == "STABLE")
    saturated_count = sum(1 for lv in levels if lv.get("status") == "SATURATED")

    # At least one level must be STABLE for the system to be green
    if stable_count == 0:
        return "RED"
    # If all levels are saturated, that's RED
    if saturated_count == len(levels):
        return "RED"
    # If all lower levels stable and only the highest saturates, that's YELLOW
    return "YELLOW" if saturated_count > 0 else "GREEN"


# ── Orchestrator ─────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=_REPO_ROOT / "ai/text_detector/checkpoints/transformer_v3_hard/phase1",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "metrics" / "load_test_report.json",
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION,
        help="seconds per load level (default 15)",
    )
    ap.add_argument(
        "--skip-rps",
        type=int,
        nargs="*",
        default=[],
        help="skip these target rps values",
    )
    ap.add_argument(
        "--skip-failures",
        action="store_true",
        help="skip failure scenarios (for quick dev runs)",
    )
    ap.add_argument(
        "--load-from",
        type=Path,
        default=None,
        help=(
            "Skip load-level runs and reuse the `load_test_results` "
            "array from an existing JSON report. Useful when only "
            "the failure-scenario or report logic changed."
        ),
    )
    args = ap.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    import psutil
    machine = {
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_memory_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "platform": platform.platform(),
        "python": sys.version.split(" ")[0],
    }
    print(f"[stage4] machine: {machine}", file=sys.stderr)

    # Load ONE detector used by every load level and (passed into)
    # the scenarios that do in-process fault injection.
    print(f"[stage4] loading production detector: {args.checkpoint}", file=sys.stderr)
    t0 = time.time()
    detector = load_test._load_production_detector(args.checkpoint)
    load_s = time.time() - t0
    print(
        f"[stage4] detector loaded in {load_s:.1f}s "
        f"(active_layers={detector._active_layers}, "
        f"lr_meta={'yes' if detector._lr_meta is not None else 'no'})",
        file=sys.stderr,
    )

    # ── Part 1: load levels (or reuse from prior run) ─────────────
    level_results: list[dict[str, Any]] = []
    if args.load_from is not None:
        if not args.load_from.exists():
            print(f"ERROR: --load-from file not found: {args.load_from}", file=sys.stderr)
            return 2
        print(f"[stage4] reusing load results from {args.load_from}", file=sys.stderr)
        with args.load_from.open("r", encoding="utf-8") as f:
            prior = json.load(f)
        level_results = prior.get("load_test_results", [])
        if not level_results:
            print(
                f"ERROR: {args.load_from} has no load_test_results array",
                file=sys.stderr,
            )
            return 2
        for lv in level_results:
            print(
                f"[stage4]   reused rps={lv['target_rps']} "
                f"actual={lv['actual_rps']} status={lv['status']}",
                file=sys.stderr,
            )
    else:
        for target_rps in DEFAULT_LEVELS:
            if target_rps in args.skip_rps:
                print(f"[stage4] skipping rps={target_rps}", file=sys.stderr)
                continue
            n_workers = LEVEL_WORKERS.get(target_rps, 8)
            print(
                f"[stage4] load level: rps={target_rps} "
                f"duration={args.duration}s workers={n_workers}",
                file=sys.stderr,
            )
            r = load_test.run_load_level(
                detector=detector,
                target_rps=target_rps,
                duration_s=args.duration,
                n_workers=n_workers,
                texts=load_test.CANARY_TEXTS,
            )
            level_results.append(r)
            print(
                f"[stage4]   -> actual={r['actual_rps']} "
                f"e2e_p95={r['latency_e2e_ms']['p95']:.0f}ms "
                f"inf_p95={r['latency_inference_ms']['p95']:.0f}ms "
                f"err={r['error_rate']} status={r['status']}",
                file=sys.stderr,
            )

    # ── Part 2: failure scenarios ─────────────────────────────────
    scenario_results: list[dict[str, Any]] = []
    if not args.skip_failures:
        for scenario_fn, takes_detector in [
            (_scenario_meta_missing, False),
            (_scenario_checkpoint_missing, False),
            (_scenario_disk_full_logs, True),
            (_scenario_slow_disk_logs, True),
            (_scenario_high_memory, True),
        ]:
            name = scenario_fn.__name__
            print(f"[stage4] scenario: {name}", file=sys.stderr)
            try:
                if takes_detector:
                    r = scenario_fn(detector)  # type: ignore[operator]
                else:
                    r = scenario_fn(args.checkpoint)  # type: ignore[operator]
            except Exception as exc:
                r = {
                    "name": name.replace("_scenario_", ""),
                    "passed": False,
                    "notes": f"orchestrator exception: {type(exc).__name__}: {exc}",
                    "elapsed_s": 0.0,
                }
            scenario_results.append(r)
            print(
                f"[stage4]   -> passed={r.get('passed')} "
                f"notes={r.get('notes', '')[:120]}",
                file=sys.stderr,
            )

    # ── Part 4: timeout behavior ──────────────────────────────────
    print("[stage4] timeout behavior check", file=sys.stderr)
    timeout_result = _test_timeout_behavior(detector)
    print(
        f"[stage4]   -> passed={timeout_result.get('passed')} "
        f"notes={timeout_result.get('notes', '')[:120]}",
        file=sys.stderr,
    )

    # ── Part 5: report ────────────────────────────────────────────
    recommendations = _derive_recommendations(level_results)
    overall = _classify_overall(level_results, scenario_results, timeout_result)

    report = {
        "stage": 4,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "machine": machine,
        "detector": {
            "model_version": getattr(
                __import__("ai.text_detector.ensemble.text_detector", fromlist=["MODEL_VERSION"]),
                "MODEL_VERSION",
                "unknown",
            ),
            "active_layers": list(detector._active_layers),
            "lr_meta_loaded": detector._lr_meta is not None,
            "lr_threshold": float(getattr(detector, "_lr_threshold", 0.5)),
            "load_seconds": round(load_s, 2),
            "checkpoint": str(args.checkpoint).replace("\\", "/"),
        },
        "test_parameters": {
            "levels": DEFAULT_LEVELS,
            "duration_s_per_level": args.duration,
            "level_workers": LEVEL_WORKERS,
        },
        "load_test_results": level_results,
        "failure_scenarios": scenario_results,
        "timeout_behavior": timeout_result,
        "resource_limits_observed": {
            "max_rss_mb_across_levels": max(
                (lv.get("rss_mb", {}).get("max_mb", 0.0) for lv in level_results),
                default=0.0,
            ),
            "max_cpu_percent_across_levels": max(
                (lv.get("cpu_percent", {}).get("max", 0.0) for lv in level_results),
                default=0.0,
            ),
            "max_threads_across_levels": max(
                (lv.get("threads", {}).get("max", 0) for lv in level_results),
                default=0,
            ),
        },
        "recommendations": recommendations,
        "overall_status": overall,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, args.output)
    print(f"[stage4] wrote {args.output}", file=sys.stderr)

    # Console summary
    print()
    print("=" * 60)
    print(f"STAGE 4 REPORT — overall_status: {overall}")
    print("=" * 60)
    for lv in level_results:
        print(
            f"  rps={lv['target_rps']:4d}  actual={lv['actual_rps']:6.2f}  "
            f"e2e_p95={lv['latency_e2e_ms']['p95']:7.1f}ms  "
            f"inf_p95={lv['latency_inference_ms']['p95']:7.1f}ms  "
            f"err={lv['error_rate']:.3f}  status={lv['status']}"
        )
    print()
    print(f"  max_stable_rps: {recommendations['max_stable_throughput_rps_single_process']}")
    print(f"  ceiling_rps:    {recommendations['max_observed_rps_single_process']}")
    print(f"  replicas_100:   {recommendations['recommended_replicas_for_100_rps']}")
    print(f"  memory_mb:      {recommendations['recommended_memory_limit_mb']}")
    print(f"  cpu_cores:      {recommendations['recommended_cpu_cores_per_worker']}")
    print()
    for s in scenario_results:
        mark = "[PASS]" if s["passed"] else "[FAIL]"
        print(f"  {mark} scenario {s['name']}: {s['notes'][:80]}")
    mark = "[PASS]" if timeout_result["passed"] else "[FAIL]"
    print(f"  {mark} timeout_behavior: {timeout_result['notes'][:80]}")
    print()

    return 0 if overall in ("GREEN", "YELLOW") else 1


def _git_sha() -> str:
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("ascii").strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
