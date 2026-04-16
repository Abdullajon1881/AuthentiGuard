"""
Stage 5 — production deployment sizing and capacity validation.

Measures the REAL worker topology the production Celery prefork pool
runs: N independent processes, each loading its own TextDetector
instance, each processing jobs serially (prefetch=1, concurrency=N).

This script uses `multiprocessing.Pool` with an initializer that loads
the detector once per worker process. That is topologically identical
to Celery prefork for memory / CPU / per-worker throughput — the only
things missing are broker + DB roundtrips (~10–50 ms per task), which
do not affect per-worker memory or CPU shape. The full-stack Celery
benchmark is not feasible here: Celery prefork does not run on Windows
(no fork()), and running Redis + Postgres + the full backend in this
environment is beyond the scope of "measure real worker memory usage."

Phases:
  Part 1 — startup RSS per worker (idle, after first inference,
           after sustained load)
  Part 2 — throughput at 5 / 10 / 20 / 40 rps (30 s each)
  Part 3 — 10-minute burst at ~1.3x the max stable rate (OOM check)
  Part 4 — scaling recommendations derived from measured throughput
  Part 5 — rewrite docker-compose.prod.yml worker limits to measured
           values (NOT guessed)
  Part 6 — 15-minute sustained load at the max stable rate (stability)

Outputs:
  metrics/worker_memory_profile.json
  metrics/celery_throughput_profile.json
  metrics/scaling_recommendations.json
  metrics/deployment_capacity_report.json  (the Stage 5 headline file)

Usage:
    python scripts/stage5_capacity.py --duration-level 30 --duration-burst 600 --duration-sustain 900

  Or override for dev runs:
    python scripts/stage5_capacity.py --duration-level 10 --duration-burst 60 --duration-sustain 60

NO MODEL LOGIC CHANGES. NO RETRAINING. NO INFERENCE BEHAVIOR CHANGES.
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

import multiprocessing as mp

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent


# ── Worker process side ─────────────────────────────────────────────
#
# These functions run inside worker processes. They touch only
# multiprocessing-safe globals and return small dicts via the Pool's
# normal result channel (pickle-safe).

_DETECTOR = None  # populated by the initializer in each worker process
_MODEL_VERSION = "unknown"


def _worker_init(checkpoint_str: str, repo_root_str: str, startup_q) -> None:  # type: ignore[no-untyped-def]
    """Worker process initializer.

    Runs once per worker. Loads the full production TextDetector and
    reports startup RSS + load time back to the parent via the shared
    Manager queue.

    `repo_root_str` is the absolute path of the repo root, passed
    explicitly by the parent. On Windows (spawn start method) the
    worker boots from a fresh Python process with no inherited
    sys.path, so we must re-prepend the repo root here.
    """
    global _DETECTOR, _MODEL_VERSION

    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    t_load_start = time.time()
    from ai.text_detector.ensemble.text_detector import (  # noqa: E402
        TextDetector,
        MODEL_VERSION,
    )
    _MODEL_VERSION = MODEL_VERSION

    detector = TextDetector(
        transformer_checkpoint=Path(checkpoint_str),
        adversarial_checkpoint=None,
        meta_checkpoint=None,
        device="cpu",
    )
    detector.load_models()
    _DETECTOR = detector
    t_load_end = time.time()

    # Report startup state back to the parent
    import psutil
    rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    startup_q.put({
        "pid": os.getpid(),
        "load_seconds": round(t_load_end - t_load_start, 2),
        "startup_rss_mb": round(rss_mb, 1),
        "active_layers": list(getattr(detector, "_active_layers", [])),
        "lr_meta_loaded": detector._lr_meta is not None,
        "model_version": MODEL_VERSION,
    })


def _worker_analyze(text: str) -> dict[str, Any]:
    """Run one inference and return latency + RSS + result."""
    import psutil

    if _DETECTOR is None:
        return {
            "pid": os.getpid(),
            "error": "detector_not_loaded",
        }

    t_start = time.time()
    try:
        r = _DETECTOR.analyze(text)
        t_end = time.time()
        return {
            "pid": os.getpid(),
            "t_start": t_start,
            "t_end": t_end,
            "infer_ms": round((t_end - t_start) * 1000.0, 2),
            "score": round(float(getattr(r, "score", 0.5)), 4),
            "label": str(getattr(r, "label", "UNCERTAIN")),
            "rss_mb": round(psutil.Process().memory_info().rss / (1024 * 1024), 1),
        }
    except Exception as exc:
        return {
            "pid": os.getpid(),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _worker_probe_rss() -> dict[str, Any]:
    """Return current RSS of this worker without running inference."""
    import psutil
    return {
        "pid": os.getpid(),
        "rss_mb": round(psutil.Process().memory_info().rss / (1024 * 1024), 1),
    }


# ── Parent-side measurement ─────────────────────────────────────────


CANARY_TEXTS = [
    "As an AI language model, I must clarify that I cannot provide personalized financial advice. "
    "However, I can offer general insights about investment strategies and risk management.",
    "Been thinking about getting back into climbing. The old gym closed last year and the new one "
    "is twice as expensive but at least it has an auto-belay. My fingers are not what they used to be.",
    "The transformer architecture fundamentally changed how we approach sequence modeling tasks "
    "by replacing recurrent mechanisms with attention-based computation on parallelizable operations.",
    "lol yeah idk man, i think we should just grab pizza after the match and call it a night. "
    "the usual place? i can drive if you want, text me when you're out.",
    "Climate change presents numerous challenges for modern society, including rising sea levels, "
    "more frequent extreme weather events, and accelerating biodiversity loss across many biomes.",
    "Machine learning models exhibit complex behaviors when presented with out-of-distribution inputs. "
    "This phenomenon is particularly evident in NLP tasks where the data distribution has shifted.",
]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(round(q * (len(s) - 1)))
    return float(s[max(0, min(len(s) - 1, idx))])


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "mean": round(float(statistics.fmean(values)), 2),
        "p50": round(_percentile(values, 0.50), 2),
        "p95": round(_percentile(values, 0.95), 2),
        "p99": round(_percentile(values, 0.99), 2),
        "max": round(float(max(values)), 2),
    }


class _ParentResourceSampler(threading.Thread):
    """Samples parent + worker process resources at 1 Hz.

    Uses psutil to walk the process tree starting at the pool master
    and aggregates RSS across workers. Reports max observed RSS total
    — this is the number that governs the OOM safety margin.
    """

    def __init__(self, worker_pids: list[int], interval_s: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.worker_pids = worker_pids
        self.interval_s = interval_s
        self._stop_event = threading.Event()
        self.samples: list[dict[str, Any]] = []

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        import psutil

        procs: dict[int, psutil.Process] = {}
        for pid in self.worker_pids:
            try:
                procs[pid] = psutil.Process(pid)
                procs[pid].cpu_percent(interval=None)  # prime
            except psutil.NoSuchProcess:
                pass

        while not self._stop_event.is_set():
            total_rss = 0.0
            max_rss = 0.0
            total_cpu = 0.0
            alive_count = 0
            per_worker: list[dict[str, Any]] = []
            for pid, proc in list(procs.items()):
                try:
                    rss = proc.memory_info().rss / (1024 * 1024)
                    cpu = proc.cpu_percent(interval=None)
                    total_rss += rss
                    total_cpu += cpu
                    max_rss = max(max_rss, rss)
                    alive_count += 1
                    per_worker.append({
                        "pid": pid,
                        "rss_mb": round(rss, 1),
                        "cpu_percent": round(cpu, 1),
                    })
                except psutil.NoSuchProcess:
                    procs.pop(pid, None)
            self.samples.append({
                "t": round(time.time(), 2),
                "alive_workers": alive_count,
                "total_rss_mb": round(total_rss, 1),
                "max_worker_rss_mb": round(max_rss, 1),
                "total_cpu_percent": round(total_cpu, 1),
                "per_worker": per_worker,
            })
            self._stop_event.wait(self.interval_s)


def drive_load(
    pool,  # type: ignore[no-untyped-def]
    worker_pids: list[int],
    target_rps: float,
    duration_s: float,
    label: str,
) -> dict[str, Any]:
    """Drive target_rps arrivals into the pool for duration_s seconds.

    Returns a dict with throughput, latency stats, queue-wait stats,
    per-worker resource stats, and the raw sampler trace.
    """
    lock = threading.Lock()
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    in_flight_counter = [0]

    sampler = _ParentResourceSampler(worker_pids, interval_s=1.0)
    sampler.start()

    start_wall = time.time()
    stop = threading.Event()

    def on_done(res: dict[str, Any]) -> None:
        with lock:
            in_flight_counter[0] -= 1
            if "error" in res:
                errors.append(res.get("error", ""))
                return
            results.append(res)

    def on_error(exc):  # type: ignore[no-untyped-def]
        with lock:
            in_flight_counter[0] -= 1
            errors.append(f"{type(exc).__name__}: {exc}")

    def producer() -> None:
        interval = 1.0 / target_rps if target_rps > 0 else 0.1
        next_fire = start_wall
        seq = 0
        while not stop.is_set():
            now = time.time()
            if now - start_wall >= duration_s:
                return
            if next_fire > now:
                time.sleep(max(0.0, next_fire - now))
            text = CANARY_TEXTS[seq % len(CANARY_TEXTS)]
            arrival = time.time()
            with lock:
                in_flight_counter[0] += 1
            fut = pool.apply_async(
                _worker_analyze,
                (text,),
                callback=lambda r, arr=arrival: _record(r, arr),
                error_callback=on_error,
            )
            seq += 1
            next_fire += interval

    def _record(res: dict[str, Any], arrival: float) -> None:
        """Attach the arrival timestamp before tracking."""
        res["arrival"] = arrival
        res["e2e_ms"] = round((res.get("t_end", arrival) - arrival) * 1000.0, 2)
        res["queue_ms"] = round((res.get("t_start", arrival) - arrival) * 1000.0, 2)
        on_done(res)

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()
    producer_thread.join()

    # Drain in-flight requests — give them 3x the test window as a grace period
    deadline = time.time() + max(30.0, duration_s * 3.0)
    while True:
        with lock:
            if in_flight_counter[0] <= 0:
                break
        if time.time() > deadline:
            break
        time.sleep(0.1)

    stop.set()
    sampler.stop()
    sampler.join(timeout=2.0)

    end_wall = time.time()
    wall_s = end_wall - start_wall

    infer = [r["infer_ms"] for r in results if "infer_ms" in r]
    e2e = [r["e2e_ms"] for r in results if "e2e_ms" in r]
    qwait = [r["queue_ms"] for r in results if "queue_ms" in r]

    # Per-worker throughput split
    pid_counts: dict[int, int] = {}
    for r in results:
        pid_counts[r["pid"]] = pid_counts.get(r["pid"], 0) + 1

    total_rss_samples = [s["total_rss_mb"] for s in sampler.samples]
    max_worker_rss_samples = [s["max_worker_rss_mb"] for s in sampler.samples]
    cpu_samples = [s["total_cpu_percent"] for s in sampler.samples]

    n_done = len(results)
    n_err = len(errors)
    n_total = n_done + n_err
    actual_rps = n_done / wall_s if wall_s > 0 else 0.0
    throughput_ratio = actual_rps / target_rps if target_rps > 0 else 0.0

    if n_err == 0 and throughput_ratio >= 0.95 and (not e2e or _percentile(e2e, 0.95) <= 1500.0):
        status = "STABLE"
    elif throughput_ratio >= 0.75 and n_err == 0:
        status = "DEGRADED"
    else:
        status = "SATURATED"

    return {
        "label": label,
        "target_rps": target_rps,
        "duration_s": duration_s,
        "wall_s": round(wall_s, 2),
        "n_workers": len(worker_pids),
        "n_completed": n_done,
        "n_errors": n_err,
        "error_rate": round(n_err / n_total, 4) if n_total else 0.0,
        "actual_rps": round(actual_rps, 3),
        "throughput_ratio": round(throughput_ratio, 3),
        "latency_inference_ms": _stats(infer),
        "latency_e2e_ms": _stats(e2e),
        "latency_queue_ms": _stats(qwait),
        "per_worker_completion_counts": pid_counts,
        "total_rss_mb": {
            "mean": round(statistics.fmean(total_rss_samples), 1) if total_rss_samples else 0.0,
            "max": round(max(total_rss_samples), 1) if total_rss_samples else 0.0,
        },
        "max_worker_rss_mb": {
            "mean": round(statistics.fmean(max_worker_rss_samples), 1) if max_worker_rss_samples else 0.0,
            "max": round(max(max_worker_rss_samples), 1) if max_worker_rss_samples else 0.0,
        },
        "total_cpu_percent": {
            "mean": round(statistics.fmean(cpu_samples), 1) if cpu_samples else 0.0,
            "max": round(max(cpu_samples), 1) if cpu_samples else 0.0,
        },
        "error_kinds": _count_error_kinds(errors),
        "status": status,
    }


def _count_error_kinds(errs: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in errs:
        k = e.split(":", 1)[0].strip() if e else "Unknown"
        out[k] = out.get(k, 0) + 1
    return out


# ── Report / file patching ─────────────────────────────────────────


def _patch_docker_compose(
    compose_path: Path,
    *,
    memory_limit_mb: int,
    memory_request_mb: int,
    concurrency: int,
) -> dict[str, Any]:
    """Rewrite the worker service's resource block in docker-compose.prod.yml.

    Targets the existing block:
        deploy:
          resources:
            limits:
              memory: 4G
            reservations:
              memory: 2G

    and replaces it with the measured values. Also comments the block
    with a reference to the Stage 5 report for traceability.

    Returns a dict describing what was changed (or would have been).
    """
    if not compose_path.exists():
        return {
            "applied": False,
            "reason": f"compose file not found: {compose_path}",
        }

    src = compose_path.read_text(encoding="utf-8")

    def _format_mb(mb: int) -> str:
        if mb >= 1024:
            return f"{mb / 1024:.1f}G".replace(".0G", "G")
        return f"{mb}M"

    limit_str = _format_mb(memory_limit_mb)
    reserve_str = _format_mb(memory_request_mb)

    new_block = (
        "    deploy:\n"
        "      resources:\n"
        "        limits:\n"
        f"          memory: {limit_str}\n"
        "        reservations:\n"
        f"          memory: {reserve_str}\n"
        "      # Stage 5 (2026-04-16): sized from measured per-worker RSS.\n"
        "      # Formula: peak_worker_rss * concurrency * 1.2 (20% headroom).\n"
        "      # See metrics/deployment_capacity_report.json for the evidence.\n"
    )

    # Find the worker service's deploy block. The file structure is
    # roughly:
    #   worker:
    #     ...
    #     deploy:
    #       resources:
    #         limits:
    #           memory: 4G
    #         reservations:
    #           memory: 2G
    # We do a surgical replace on the FIRST `deploy:` block AFTER the
    # line containing `  worker:`.
    worker_idx = src.find("\n  worker:\n")
    if worker_idx == -1:
        return {
            "applied": False,
            "reason": "could not locate 'worker:' service in compose file",
        }

    # Locate the deploy block inside the worker service
    deploy_start = src.find("    deploy:", worker_idx)
    if deploy_start == -1:
        return {
            "applied": False,
            "reason": "'deploy:' block not found under worker service",
        }

    # The deploy block ends at the next line that starts with "    "
    # at the SAME indent level but a different key — or at a lower
    # indent level. Simpler: look for the next "    healthcheck:" or
    # the next top-level service block.
    end_markers = [
        "    healthcheck:",
        "  beat:",
        "  frontend:",
        "  flower:",
        "  caddy:",
        "volumes:",
        "secrets:",
    ]
    end_idx = len(src)
    for m in end_markers:
        i = src.find("\n" + m, deploy_start)
        if i != -1 and i < end_idx:
            end_idx = i + 1  # keep the leading newline out of the replacement

    old_block = src[deploy_start:end_idx]
    new_src = src[:deploy_start] + new_block + src[end_idx:]

    if new_src == src:
        return {"applied": False, "reason": "no change (already matches)"}

    compose_path.write_text(new_src, encoding="utf-8")
    return {
        "applied": True,
        "compose_path": str(compose_path).replace("\\", "/"),
        "old_block_chars": len(old_block),
        "new_block_chars": len(new_block),
        "memory_limit_mb": memory_limit_mb,
        "memory_request_mb": memory_request_mb,
        "concurrency": concurrency,
    }


def _derive_recommendations(
    startups: list[dict[str, Any]],
    throughput_runs: dict[int, dict[str, Any]],
    burst: dict[str, Any] | None,
    *,
    concurrency: int,
) -> dict[str, Any]:
    """Produce the Stage 5 recommendations from the measurements."""

    # Peak RSS per worker = max of (startup RSS, any in-run max)
    startup_rss = [s["startup_rss_mb"] for s in startups]
    max_startup_rss = max(startup_rss) if startup_rss else 0.0

    run_max_worker_rss = 0.0
    for r in throughput_runs.values():
        run_max_worker_rss = max(run_max_worker_rss, r["max_worker_rss_mb"]["max"])
    if burst is not None:
        run_max_worker_rss = max(run_max_worker_rss, burst["max_worker_rss_mb"]["max"])

    peak_worker_rss_mb = max(max_startup_rss, run_max_worker_rss)

    # OOM formula: peak_worker_rss * concurrency * 1.2
    memory_limit_mb = int(peak_worker_rss_mb * concurrency * 1.2)
    memory_request_mb = int(peak_worker_rss_mb * concurrency * 1.0)

    # Max stable throughput per pod = highest target_rps marked STABLE
    stable_levels = [r for r in throughput_runs.values() if r["status"] == "STABLE"]
    max_stable_rps_per_pod = max(
        (r["target_rps"] for r in stable_levels),
        default=0,
    )
    max_observed_rps_per_pod = max(
        (r["actual_rps"] for r in throughput_runs.values()),
        default=0.0,
    )

    def _pods_for(target: int) -> int:
        if max_stable_rps_per_pod <= 0:
            return 0
        return int((target / max_stable_rps_per_pod) + 0.9999)

    pods_50 = _pods_for(50)
    pods_100 = _pods_for(100)
    pods_200 = _pods_for(200)

    # CPU sizing: peak per-pod cpu_percent across the throughput runs.
    # cpu_percent from psutil on the summed workers > 100 on multi-core.
    # Convert % -> cores.
    max_cpu_percent_total = max(
        (r["total_cpu_percent"]["max"] for r in throughput_runs.values()),
        default=0.0,
    )
    if burst is not None:
        max_cpu_percent_total = max(max_cpu_percent_total, burst["total_cpu_percent"]["max"])
    recommended_cpu_cores = round(max_cpu_percent_total / 100.0 + 0.5, 2)

    return {
        "concurrency": concurrency,
        "peak_worker_rss_mb": round(peak_worker_rss_mb, 1),
        "startup_rss_mb_max": round(max_startup_rss, 1),
        "run_max_worker_rss_mb": round(run_max_worker_rss, 1),
        "memory_request_mb": memory_request_mb,
        "memory_limit_mb": memory_limit_mb,
        "memory_headroom_factor": 1.2,
        "recommended_cpu_cores": recommended_cpu_cores,
        "max_stable_rps_per_pod": max_stable_rps_per_pod,
        "max_observed_rps_per_pod": round(max_observed_rps_per_pod, 2),
        "pods_required_for_50_rps": pods_50,
        "pods_required_for_100_rps": pods_100,
        "pods_required_for_200_rps": pods_200,
        "burst_headroom_pods": max(1, int(0.3 * pods_100)),
        "notes": [
            "Memory limit formula: peak_worker_rss * concurrency * 1.2",
            "Peak worker RSS is the max observed across startup / per-level / burst.",
            "Pod counts assume max stable rps per pod is reproducible in production; "
            "real-world throughput may differ by ±10% due to network/broker/DB latency.",
            "burst_headroom_pods is 30% over the 100 rps fleet, rounded up to at least 1.",
        ],
    }


# ── Orchestrator ────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=_REPO_ROOT / "ai/text_detector/checkpoints/transformer_v3_hard/phase1",
    )
    ap.add_argument("--concurrency", type=int, default=4, help="worker pool size")
    ap.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[5, 10, 20, 40],
        help="throughput levels in rps",
    )
    ap.add_argument("--duration-level", type=float, default=30.0)
    ap.add_argument("--duration-burst", type=float, default=600.0)  # 10 min
    ap.add_argument("--duration-sustain", type=float, default=900.0)  # 15 min
    ap.add_argument("--burst-multiplier", type=float, default=1.3)
    ap.add_argument(
        "--metrics-dir",
        type=Path,
        default=_REPO_ROOT / "metrics",
    )
    ap.add_argument(
        "--compose-path",
        type=Path,
        default=_REPO_ROOT / "docker-compose.prod.yml",
    )
    ap.add_argument(
        "--skip-patch",
        action="store_true",
        help="do not modify docker-compose.prod.yml",
    )
    ap.add_argument(
        "--skip-burst",
        action="store_true",
        help="skip the 10-minute burst phase",
    )
    ap.add_argument(
        "--skip-sustain",
        action="store_true",
        help="skip the 15-minute sustained phase",
    )
    args = ap.parse_args()

    if not args.checkpoint.exists():
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    git_sha = _git_sha()
    timestamp = datetime.now(timezone.utc).isoformat()

    import psutil
    machine = {
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_memory_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "platform": __import__("platform").platform(),
        "python": sys.version.split(" ")[0],
    }
    print(f"[stage5] machine: {machine}", file=sys.stderr)
    print(
        f"[stage5] pool=prefork-simulation concurrency={args.concurrency} "
        f"levels={args.levels}",
        file=sys.stderr,
    )

    # ── Part 1: spin up the pool and collect startup measurements ──
    manager = mp.Manager()
    startup_q = manager.Queue()

    print(
        f"[stage5] spinning up {args.concurrency} worker processes "
        f"(each loads its own detector)...",
        file=sys.stderr,
    )
    t0 = time.time()
    pool = mp.Pool(
        processes=args.concurrency,
        initializer=_worker_init,
        initargs=(str(args.checkpoint.resolve()), str(_REPO_ROOT), startup_q),
    )
    # Wait for every worker to report its startup state. Bounded wait
    # so a hung worker can't block the test forever.
    startups: list[dict[str, Any]] = []
    deadline = time.time() + max(120.0, args.concurrency * 30.0)
    while len(startups) < args.concurrency and time.time() < deadline:
        try:
            s = startup_q.get(timeout=5.0)
            startups.append(s)
            print(
                f"[stage5]   worker startup: pid={s['pid']} "
                f"load={s['load_seconds']}s rss={s['startup_rss_mb']}MB",
                file=sys.stderr,
            )
        except Exception:
            continue
    pool_startup_wall = time.time() - t0
    print(f"[stage5] pool ready in {pool_startup_wall:.1f}s", file=sys.stderr)

    if len(startups) != args.concurrency:
        print(
            f"ERROR: only {len(startups)} of {args.concurrency} workers "
            f"reported startup. Aborting.",
            file=sys.stderr,
        )
        pool.terminate()
        pool.join()
        return 2

    worker_pids = [s["pid"] for s in startups]

    # Sanity one-shot: ensure each worker is responsive + measure
    # first-inference RSS
    probes: list[dict[str, Any]] = []
    for text in CANARY_TEXTS[: args.concurrency]:
        probes.append(pool.apply(_worker_analyze, (text,)))  # type: ignore[arg-type]
    first_infer_rss: dict[int, float] = {}
    for p in probes:
        if "pid" in p and "rss_mb" in p:
            first_infer_rss[p["pid"]] = max(
                first_infer_rss.get(p["pid"], 0.0), p["rss_mb"]
            )
    for s in startups:
        s["first_inference_rss_mb"] = first_infer_rss.get(s["pid"], 0.0)

    memory_profile = {
        "schema_version": 1,
        "stage": 5,
        "phase": "part1_worker_memory",
        "git_sha": git_sha,
        "timestamp_utc": timestamp,
        "machine": machine,
        "concurrency": args.concurrency,
        "pool_startup_wall_s": round(pool_startup_wall, 2),
        "workers": startups,
        "first_inference_probes": probes,
    }
    mem_path = args.metrics_dir / "worker_memory_profile.json"
    _write_json(mem_path, memory_profile)

    # ── Part 2: throughput levels ─────────────────────────────────
    throughput_runs: dict[int, dict[str, Any]] = {}
    for rps in args.levels:
        print(
            f"[stage5] throughput level: rps={rps} duration={args.duration_level}s",
            file=sys.stderr,
        )
        r = drive_load(
            pool=pool,
            worker_pids=worker_pids,
            target_rps=float(rps),
            duration_s=args.duration_level,
            label=f"level_{rps}rps",
        )
        throughput_runs[rps] = r
        print(
            f"[stage5]   -> actual={r['actual_rps']} "
            f"inf_p95={r['latency_inference_ms']['p95']}ms "
            f"e2e_p95={r['latency_e2e_ms']['p95']}ms "
            f"total_rss_max={r['total_rss_mb']['max']}MB "
            f"status={r['status']}",
            file=sys.stderr,
        )

    throughput_profile = {
        "schema_version": 1,
        "stage": 5,
        "phase": "part2_throughput",
        "git_sha": git_sha,
        "timestamp_utc": timestamp,
        "machine": machine,
        "concurrency": args.concurrency,
        "pool": "prefork-simulation (multiprocessing.Pool)",
        "levels": {str(k): v for k, v in throughput_runs.items()},
    }
    tp_path = args.metrics_dir / "celery_throughput_profile.json"
    _write_json(tp_path, throughput_profile)

    # ── Part 3: 10-min burst ──────────────────────────────────────
    # Burst rate = burst_multiplier * max_stable_rps, with a floor.
    stable_levels = [r for r in throughput_runs.values() if r["status"] == "STABLE"]
    max_stable_rps = max((r["target_rps"] for r in stable_levels), default=0)
    if max_stable_rps == 0:
        # Fall back to max observed actual rps (even if degraded)
        max_stable_rps = int(max(
            (r["actual_rps"] for r in throughput_runs.values()),
            default=0.0,
        ))
    burst_rps = max(1.0, round(max_stable_rps * args.burst_multiplier, 2))

    burst_result: dict[str, Any] | None = None
    if args.skip_burst:
        print("[stage5] skipping burst phase (--skip-burst)", file=sys.stderr)
    else:
        print(
            f"[stage5] burst phase: rps={burst_rps} duration={args.duration_burst}s "
            f"(1.3x max_stable={max_stable_rps})",
            file=sys.stderr,
        )
        burst_result = drive_load(
            pool=pool,
            worker_pids=worker_pids,
            target_rps=burst_rps,
            duration_s=args.duration_burst,
            label=f"burst_{burst_rps}rps",
        )
        print(
            f"[stage5]   -> actual={burst_result['actual_rps']} "
            f"total_rss_max={burst_result['total_rss_mb']['max']}MB "
            f"cpu_max={burst_result['total_cpu_percent']['max']}% "
            f"errors={burst_result['n_errors']} "
            f"status={burst_result['status']}",
            file=sys.stderr,
        )

    # ── Part 4: scaling recommendations ───────────────────────────
    recommendations = _derive_recommendations(
        startups=startups,
        throughput_runs=throughput_runs,
        burst=burst_result,
        concurrency=args.concurrency,
    )
    print(
        f"[stage5] recommendations: "
        f"mem_limit={recommendations['memory_limit_mb']}MB "
        f"cpu_cores={recommendations['recommended_cpu_cores']} "
        f"max_stable_rps_per_pod={recommendations['max_stable_rps_per_pod']} "
        f"pods_100={recommendations['pods_required_for_100_rps']}",
        file=sys.stderr,
    )

    scaling_path = args.metrics_dir / "scaling_recommendations.json"
    scaling_record = {
        "schema_version": 1,
        "stage": 5,
        "phase": "part4_scaling",
        "git_sha": git_sha,
        "timestamp_utc": timestamp,
        "recommendations": recommendations,
        "burst_phase": burst_result,
    }
    _write_json(scaling_path, scaling_record)

    # ── Part 5: patch docker-compose.prod.yml ─────────────────────
    if args.skip_patch:
        patch_result = {"applied": False, "reason": "--skip-patch set"}
    else:
        patch_result = _patch_docker_compose(
            args.compose_path,
            memory_limit_mb=recommendations["memory_limit_mb"],
            memory_request_mb=recommendations["memory_request_mb"],
            concurrency=args.concurrency,
        )
    print(f"[stage5] compose patch: {patch_result}", file=sys.stderr)

    # ── Part 6: 15-min sustained load ─────────────────────────────
    sustain_rps = max(1.0, float(max_stable_rps)) if max_stable_rps > 0 else 1.0
    sustain_result: dict[str, Any] | None = None
    if args.skip_sustain:
        print("[stage5] skipping sustained phase (--skip-sustain)", file=sys.stderr)
    else:
        print(
            f"[stage5] sustained phase: rps={sustain_rps} "
            f"duration={args.duration_sustain}s",
            file=sys.stderr,
        )
        sustain_result = drive_load(
            pool=pool,
            worker_pids=worker_pids,
            target_rps=sustain_rps,
            duration_s=args.duration_sustain,
            label=f"sustain_{sustain_rps}rps",
        )
        print(
            f"[stage5]   -> actual={sustain_result['actual_rps']} "
            f"total_rss_max={sustain_result['total_rss_mb']['max']}MB "
            f"errors={sustain_result['n_errors']} "
            f"status={sustain_result['status']}",
            file=sys.stderr,
        )

    # ── Cleanup + final report ────────────────────────────────────
    pool.close()
    pool.join()
    manager.shutdown()

    # Check for OOM / worker restart signals in the sustained run
    sustain_ok = True
    sustain_reasons: list[str] = []
    if sustain_result is not None:
        if sustain_result["n_errors"] > 0:
            sustain_ok = False
            sustain_reasons.append(f"errors={sustain_result['n_errors']}")
        max_total_rss = sustain_result["total_rss_mb"]["max"]
        if max_total_rss > recommendations["memory_limit_mb"]:
            sustain_ok = False
            sustain_reasons.append(
                f"max_rss {max_total_rss}MB > limit {recommendations['memory_limit_mb']}MB"
            )
        if sustain_result["status"] == "SATURATED":
            sustain_ok = False
            sustain_reasons.append("status=SATURATED")

    # Overall status
    all_scenarios_ok = True
    if burst_result is not None and burst_result["n_errors"] > 0:
        all_scenarios_ok = False
    if not sustain_ok and sustain_result is not None:
        all_scenarios_ok = False

    overall = "GREEN" if all_scenarios_ok and sustain_ok else ("YELLOW" if sustain_ok else "RED")

    capacity_report = {
        "schema_version": 1,
        "stage": 5,
        "git_sha": git_sha,
        "timestamp_utc": timestamp,
        "machine": machine,
        "pool": "prefork-simulation (multiprocessing.Pool on Windows; "
                "topologically identical to Celery prefork for memory/CPU)",
        "concurrency": args.concurrency,
        "part_1_worker_memory_profile": memory_profile,
        "part_2_throughput_profile": {
            "levels": {str(k): v for k, v in throughput_runs.items()},
        },
        "part_3_burst": burst_result,
        "part_4_scaling_recommendations": recommendations,
        "part_5_compose_patch": patch_result,
        "part_6_sustained_load": {
            "result": sustain_result,
            "passed": sustain_ok,
            "reasons": sustain_reasons,
        },
        "safe_memory_per_pod_mb": recommendations["memory_limit_mb"],
        "safe_cpu_per_pod_cores": recommendations["recommended_cpu_cores"],
        "recommended_concurrency": args.concurrency,
        "max_stable_throughput_rps_per_pod": recommendations["max_stable_rps_per_pod"],
        "recommended_replica_count_for_100_rps": recommendations["pods_required_for_100_rps"],
        "overall_status": overall,
        "test_parameters": {
            "duration_level_s": args.duration_level,
            "duration_burst_s": args.duration_burst,
            "duration_sustain_s": args.duration_sustain,
            "burst_multiplier": args.burst_multiplier,
            "levels_rps": args.levels,
        },
    }

    cap_path = args.metrics_dir / "deployment_capacity_report.json"
    _write_json(cap_path, capacity_report)

    print()
    print("=" * 60)
    print(f"STAGE 5 CAPACITY REPORT — {overall}")
    print("=" * 60)
    print(f"  safe_memory_per_pod_mb:                    {capacity_report['safe_memory_per_pod_mb']}")
    print(f"  safe_cpu_per_pod_cores:                    {capacity_report['safe_cpu_per_pod_cores']}")
    print(f"  recommended_concurrency:                   {capacity_report['recommended_concurrency']}")
    print(f"  max_stable_throughput_rps_per_pod:         {capacity_report['max_stable_throughput_rps_per_pod']}")
    print(f"  recommended_replica_count_for_100_rps:     {capacity_report['recommended_replica_count_for_100_rps']}")
    print(f"  pods_50 / pods_100 / pods_200:             "
          f"{recommendations['pods_required_for_50_rps']} / "
          f"{recommendations['pods_required_for_100_rps']} / "
          f"{recommendations['pods_required_for_200_rps']}")
    if sustain_result is not None:
        print(
            f"  sustained_load (15m @ {sustain_rps} rps): "
            f"n={sustain_result['n_completed']} errors={sustain_result['n_errors']} "
            f"e2e_p95={sustain_result['latency_e2e_ms']['p95']}ms "
            f"rss_max={sustain_result['total_rss_mb']['max']}MB"
        )
    print(f"  compose patched: {patch_result.get('applied', False)}")
    print(f"  wrote: {cap_path}")

    return 0 if overall == "GREEN" else (1 if overall == "YELLOW" else 2)


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)
    print(f"[stage5] wrote {path}", file=sys.stderr)


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
    # On Windows, multiprocessing requires the __main__ guard AND the
    # freeze_support() call (no-op on POSIX) so child processes can
    # re-import the module without re-running main().
    mp.freeze_support()
    sys.exit(main())
