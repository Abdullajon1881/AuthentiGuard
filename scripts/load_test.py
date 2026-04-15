"""
Stage 4 — Part 1: reusable load test harness for the text-detection hot path.

Drives a target arrival rate (rps) into `TextDetector.analyze()` for a
given duration, using a thread pool for concurrent execution. Measures:

  - end-to-end latency (arrival -> completion)
  - inference latency (worker pickup -> completion)
  - queue wait time (arrival -> worker pickup)
  - actual throughput vs target
  - error rate + error kinds
  - CPU% and RSS over time (via psutil, sampled at 1 Hz)

Writes a single JSON report to `--output` or stdout. Deterministic
inter-arrival spacing (not Poisson) — makes p95/p99 stable at modest
sample sizes.

THIS SCRIPT DOES NOT MODIFY MODEL LOGIC. It loads a TextDetector with
the production constructor, runs it, and measures. Nothing about the
inference path is changed.

Usage:
    python scripts/load_test.py --rps 10 --duration 20 --workers 8
    python scripts/load_test.py --rps 100 --duration 10 --workers 20 --output out.json

Notes:
  * `--workers` is the size of the ThreadPoolExecutor that shares a
    SINGLE detector instance. PyTorch releases the GIL during matmul,
    so threads give real parallelism on CPU. The ceiling is a blend of
    (physical cores) and (GIL contention).
  * At saturation, the producer's arrival rate exceeds the consumer's
    completion rate and the queue grows. End-to-end latency balloons
    while inference latency stays flat — the canonical signature of a
    saturated queueing system. The report reflects this honestly.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import statistics
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


CANARY_TEXTS = [
    "As an AI language model, I must clarify that I cannot provide personalized financial advice. "
    "However, I can offer general insights that may be helpful for your decision-making process.",
    "Been thinking about getting back into climbing. The old gym closed last year and the new one "
    "is twice as expensive but at least it has an auto-belay. My fingers are not what they used to be.",
    "The transformer architecture fundamentally changed how we approach sequence modeling by "
    "replacing recurrent mechanisms with attention-based computation on parallelizable operations.",
    "lol yeah idk man, i think we should just grab pizza after the match and call it a night. "
    "the usual place? i can drive if you want. text me when you're out.",
    "Climate change presents numerous challenges for modern society, including rising sea levels, "
    "more frequent extreme weather events, and accelerating biodiversity loss across many biomes.",
    "Machine learning models exhibit complex behaviors when presented with out-of-distribution inputs. "
    "This phenomenon is particularly evident in NLP tasks where the data distribution has shifted.",
]


def _load_production_detector(checkpoint: Path):
    """Load the detector with the same shape used by the Celery worker.

    Stage 2 meta auto-discovery fires if the meta joblibs sit next to
    the checkpoint, so this function measures the REAL production path.
    """
    from ai.text_detector.ensemble.text_detector import TextDetector
    det = TextDetector(
        transformer_checkpoint=checkpoint,
        adversarial_checkpoint=None,
        meta_checkpoint=None,
        device="cpu",
    )
    det.load_models()
    return det


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(round(q * (len(s) - 1)))
    return float(s[max(0, min(len(s) - 1, idx))])


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "n": 0, "mean": 0.0, "min": 0.0, "max": 0.0,
            "p50": 0.0, "p95": 0.0, "p99": 0.0,
        }
    return {
        "n": len(values),
        "mean": round(float(statistics.fmean(values)), 2),
        "min": round(float(min(values)), 2),
        "max": round(float(max(values)), 2),
        "p50": round(_percentile(values, 0.50), 2),
        "p95": round(_percentile(values, 0.95), 2),
        "p99": round(_percentile(values, 0.99), 2),
    }


class _ResourceSampler(threading.Thread):
    """Samples CPU% and RSS of the current process at 1 Hz."""

    # NOTE: do NOT name the instance attribute `_stop` — threading.Thread
    # has an internal `_stop` method and shadowing it breaks join().

    def __init__(self, interval_s: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self._stop_event = threading.Event()
        self.cpu_samples: list[float] = []
        self.rss_mb_samples: list[float] = []
        self.thread_samples: list[int] = []

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        import psutil
        proc = psutil.Process()
        proc.cpu_percent(interval=None)  # priming call
        while not self._stop_event.is_set():
            try:
                cpu = proc.cpu_percent(interval=None)
                mem = proc.memory_info().rss / (1024 * 1024)
                nthreads = proc.num_threads()
                self.cpu_samples.append(cpu)
                self.rss_mb_samples.append(mem)
                self.thread_samples.append(nthreads)
            except Exception:
                pass
            self._stop_event.wait(self.interval_s)


def run_load_level(
    detector,
    target_rps: int,
    duration_s: float,
    n_workers: int,
    texts: list[str],
) -> dict[str, Any]:
    """Run one load level and return a dict of measurements.

    `detector` must be pre-loaded. The function is re-entrant so the
    orchestrator can call it multiple times without reloading weights.
    """
    import psutil  # noqa: F401 — imported for the sampler module

    request_q: "queue.Queue[tuple[int, float] | None]" = queue.Queue()
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    lock = threading.Lock()

    sampler = _ResourceSampler(interval_s=1.0)
    sampler.start()

    start_wall = time.time()
    stop_producer = threading.Event()

    def producer() -> None:
        """Fire request arrivals at (1 / target_rps) spacing."""
        interval = 1.0 / target_rps
        next_fire = start_wall
        seq = 0
        while not stop_producer.is_set():
            now = time.time()
            if now - start_wall >= duration_s:
                break
            if next_fire > now:
                time.sleep(max(0.0, next_fire - now))
            request_q.put((seq, time.time()))
            seq += 1
            next_fire += interval

    def worker() -> None:
        text_cycle = 0
        while True:
            item = request_q.get()
            if item is None:
                request_q.task_done()
                return
            seq, arrival = item
            text = texts[text_cycle % len(texts)]
            text_cycle += 1
            t_start = time.time()
            try:
                r = detector.analyze(text)
                t_end = time.time()
                rec = {
                    "seq": seq,
                    "arrival": arrival,
                    "start": t_start,
                    "end": t_end,
                    "e2e_ms": (t_end - arrival) * 1000.0,
                    "infer_ms": (t_end - t_start) * 1000.0,
                    "queue_ms": (t_start - arrival) * 1000.0,
                    "label": str(getattr(r, "label", "?")),
                    "score": float(getattr(r, "score", 0.0)),
                }
                with lock:
                    results.append(rec)
            except Exception as exc:
                with lock:
                    errors.append({
                        "seq": seq,
                        "arrival": arrival,
                        "kind": type(exc).__name__,
                        "message": str(exc)[:200],
                    })
            finally:
                request_q.task_done()

    workers = [threading.Thread(target=worker, daemon=True) for _ in range(n_workers)]
    for w in workers:
        w.start()

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()

    producer_thread.join()

    # All arrivals produced — send stop sentinels and wait for workers
    # to drain the queue. Give them a bounded grace period so a
    # saturated-queue run doesn't hang forever.
    drain_deadline = time.time() + max(30.0, duration_s * 3.0)
    for _ in workers:
        request_q.put(None)
    for w in workers:
        remaining = drain_deadline - time.time()
        w.join(timeout=max(0.1, remaining))

    sampler.stop()
    sampler.join(timeout=2.0)

    end_wall = time.time()
    wall_s = end_wall - start_wall

    # Statistics
    e2e = [r["e2e_ms"] for r in results]
    infer = [r["infer_ms"] for r in results]
    qwait = [r["queue_ms"] for r in results]

    n_done = len(results)
    n_err = len(errors)
    n_attempted = n_done + n_err
    actual_rps = n_done / wall_s if wall_s > 0 else 0.0
    throughput_ratio = actual_rps / target_rps if target_rps > 0 else 0.0

    # Classify the run
    if n_err == 0 and throughput_ratio >= 0.95 and (not e2e or _percentile(e2e, 0.95) <= 1000.0):
        status = "STABLE"
    elif throughput_ratio >= 0.75:
        status = "DEGRADED"
    else:
        status = "SATURATED"

    # Count fallbacks: a layer with score == 0.5 and label == UNCERTAIN
    # on a non-trivial input isn't a fallback per se, but we can't
    # distinguish here — we record only what the detector returned.
    label_counts: dict[str, int] = {}
    for r in results:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

    cpu_stats = {
        "mean": round(statistics.fmean(sampler.cpu_samples), 2) if sampler.cpu_samples else 0.0,
        "max": round(max(sampler.cpu_samples), 2) if sampler.cpu_samples else 0.0,
        "samples": len(sampler.cpu_samples),
    }
    rss_stats = {
        "mean_mb": round(statistics.fmean(sampler.rss_mb_samples), 1) if sampler.rss_mb_samples else 0.0,
        "max_mb": round(max(sampler.rss_mb_samples), 1) if sampler.rss_mb_samples else 0.0,
    }
    thread_stats = {
        "mean": round(statistics.fmean(sampler.thread_samples), 1) if sampler.thread_samples else 0.0,
        "max": int(max(sampler.thread_samples)) if sampler.thread_samples else 0,
    }

    return {
        "target_rps": target_rps,
        "duration_s": duration_s,
        "n_workers": n_workers,
        "wall_s": round(wall_s, 3),
        "n_attempted": n_attempted,
        "n_completed": n_done,
        "n_errors": n_err,
        "error_rate": round(n_err / n_attempted, 4) if n_attempted else 0.0,
        "actual_rps": round(actual_rps, 3),
        "throughput_ratio": round(throughput_ratio, 3),
        "latency_e2e_ms": _stats(e2e),
        "latency_inference_ms": _stats(infer),
        "latency_queue_ms": _stats(qwait),
        "cpu_percent": cpu_stats,
        "rss_mb": rss_stats,
        "threads": thread_stats,
        "label_counts": label_counts,
        "error_kinds": _count_error_kinds(errors),
        "status": status,
    }


def _count_error_kinds(errors: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in errors:
        k = e.get("kind", "Unknown")
        counts[k] = counts.get(k, 0) + 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=_REPO_ROOT / "ai/text_detector/checkpoints/transformer_v3_hard/phase1",
    )
    ap.add_argument("--rps", type=int, required=True, help="target requests/sec")
    ap.add_argument("--duration", type=float, default=15.0, help="test window seconds")
    ap.add_argument("--workers", type=int, default=8, help="concurrent worker threads")
    ap.add_argument("--output", type=Path, default=None, help="JSON output path (stdout if omitted)")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    print(
        f"[load_test] rps={args.rps} duration={args.duration}s workers={args.workers}",
        file=sys.stderr,
    )
    t0 = time.time()
    detector = _load_production_detector(args.checkpoint)
    t1 = time.time()
    print(f"[load_test] detector loaded in {t1 - t0:.1f}s", file=sys.stderr)
    print(
        f"[load_test] active_layers={detector._active_layers} "
        f"lr_meta={'yes' if detector._lr_meta is not None else 'no'}",
        file=sys.stderr,
    )

    result = run_load_level(
        detector=detector,
        target_rps=args.rps,
        duration_s=args.duration,
        n_workers=args.workers,
        texts=CANARY_TEXTS,
    )

    result["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    result["load_s_model"] = round(t1 - t0, 2)
    result["lr_meta_loaded"] = detector._lr_meta is not None
    result["active_layers"] = list(detector._active_layers)
    try:
        from ai.text_detector.ensemble.text_detector import MODEL_VERSION
        result["model_version"] = MODEL_VERSION
    except Exception:
        result["model_version"] = "unknown"

    text_result = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text_result, encoding="utf-8")
        print(f"[load_test] wrote {args.output}", file=sys.stderr)
    else:
        print(text_result)

    # Console summary
    print(
        f"[load_test] target={args.rps} actual={result['actual_rps']} "
        f"e2e_p95={result['latency_e2e_ms']['p95']:.1f}ms "
        f"inf_p95={result['latency_inference_ms']['p95']:.1f}ms "
        f"err={result['error_rate']} status={result['status']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
