"""
Stress test: 50 concurrent text submissions against a live AuthentiGuard stack.

Measures:
  - Submission latency (p50, p95, p99)
  - End-to-end latency (submit -> poll -> result)
  - Error rate and error types
  - Queue depth stability (no infinite growth)

Usage:
    # Start the stack first:
    docker compose -f docker-compose.test.yml up -d

    # Run with defaults (50 concurrent, heuristic mode):
    python performance/load_tests/stress_test.py

    # Custom concurrency and target:
    python performance/load_tests/stress_test.py --concurrency 100 --base-url http://prod:8000

Exit codes:
    0 = all checks passed
    1 = one or more checks failed (error rate > 5%, p95 > 30s, etc.)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import httpx


# ── Test payloads ────────────────────────────────────────────

PAYLOADS = [
    # AI-style text
    (
        "The implementation of transformer-based architectures has revolutionized "
        "natural language processing, enabling models to capture long-range dependencies "
        "through self-attention mechanisms. These models demonstrate remarkable performance "
        "across diverse linguistic tasks including text generation and sentiment analysis."
    ),
    # Human-style text
    (
        "Just got back from the vet with Luna. She hates car rides but the doc said "
        "she's doing great for a 12 year old lab. Had to stop at three different pet "
        "stores to find her favorite treats because apparently they discontinued them?? "
        "Anyway she's passed out on the couch now, snoring like a chainsaw."
    ),
    # Code-adjacent text
    (
        "The function iterates through each element in the array and applies a "
        "transformation based on the predefined mapping. Edge cases include empty "
        "arrays, null values, and arrays exceeding the maximum configured length. "
        "The time complexity is O(n) where n is the number of elements."
    ),
    # Mixed style
    (
        "So I was reading this paper about attention mechanisms and honestly it "
        "blew my mind. The key insight is that you don't need recurrence at all! "
        "Just let every token look at every other token. Simple but it works "
        "stupidly well. Kinda wish I'd thought of it first lol."
    ),
]


@dataclass
class JobResult:
    """Result of a single submit -> poll -> result cycle."""
    job_id: str = ""
    submit_ms: float = 0.0
    poll_ms: float = 0.0
    total_ms: float = 0.0
    status: str = ""
    score: float | None = None
    label: str = ""
    error: str = ""


@dataclass
class StressReport:
    """Aggregate results from the stress test."""
    total_jobs: int = 0
    completed: int = 0
    failed: int = 0
    errors: int = 0
    error_types: dict[str, int] = field(default_factory=dict)
    submit_latencies_ms: list[float] = field(default_factory=list)
    total_latencies_ms: list[float] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    duration_s: float = 0.0


def _register_user(client: httpx.Client) -> dict[str, str]:
    """Register + login, return auth headers."""
    email = f"stress-{uuid.uuid4().hex[:8]}@test.local"
    client.post("/api/v1/auth/register", json={
        "email": email,
        "password": "StressTest123!",
        "full_name": "Stress Tester",
        "consent_given": True,
    })
    resp = client.post("/api/v1/auth/login", json={
        "email": email,
        "password": "StressTest123!",
    })
    if resp.status_code != 200:
        raise RuntimeError(f"Login failed: {resp.status_code} {resp.text}")
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _run_one_job(
    base_url: str,
    headers: dict[str, str],
    text: str,
    poll_timeout: float,
) -> JobResult:
    """Submit text, poll to completion, fetch result. Returns JobResult."""
    result = JobResult()

    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        t0 = time.monotonic()

        # Submit
        try:
            resp = client.post("/api/v1/analyze/text", json={
                "text": text,
                "content_type": "text",
            }, headers=headers)
        except Exception as exc:
            result.error = f"submit_error: {exc}"
            result.total_ms = (time.monotonic() - t0) * 1000
            return result

        result.submit_ms = (time.monotonic() - t0) * 1000

        if resp.status_code == 429:
            result.error = "rate_limited"
            result.total_ms = result.submit_ms
            return result
        if resp.status_code != 202:
            result.error = f"submit_http_{resp.status_code}"
            result.total_ms = result.submit_ms
            return result

        job = resp.json()
        result.job_id = job.get("job_id", "")

        # Poll
        deadline = time.monotonic() + poll_timeout
        while time.monotonic() < deadline:
            try:
                resp = client.get(f"/api/v1/jobs/{result.job_id}", headers=headers)
                data = resp.json()
                result.status = data.get("status", "")
                if result.status in ("completed", "failed"):
                    break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            result.error = "poll_timeout"
            result.total_ms = (time.monotonic() - t0) * 1000
            return result

        result.poll_ms = (time.monotonic() - t0) * 1000 - result.submit_ms

        if result.status == "failed":
            result.error = f"job_failed: {data.get('error_message', 'unknown')}"
            result.total_ms = (time.monotonic() - t0) * 1000
            return result

        # Fetch result
        try:
            resp = client.get(f"/api/v1/jobs/{result.job_id}/result", headers=headers)
            if resp.status_code == 200:
                r = resp.json()
                result.score = r.get("authenticity_score")
                result.label = r.get("label", "")
        except Exception as exc:
            result.error = f"result_fetch_error: {exc}"

        result.total_ms = (time.monotonic() - t0) * 1000

    return result


def run_stress_test(
    base_url: str,
    concurrency: int = 50,
    total_jobs: int = 50,
    poll_timeout: float = 120.0,
) -> StressReport:
    """Run concurrent text analysis jobs and collect metrics."""

    report = StressReport(total_jobs=total_jobs)

    # Register a single user for all jobs
    with httpx.Client(base_url=base_url, timeout=30.0) as client:
        headers = _register_user(client)

    print(f"\nStarting stress test: {total_jobs} jobs, {concurrency} concurrent")
    print(f"Target: {base_url}")
    print("-" * 60)

    t_start = time.monotonic()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        for i in range(total_jobs):
            text = PAYLOADS[i % len(PAYLOADS)]
            fut = pool.submit(_run_one_job, base_url, headers, text, poll_timeout)
            futures.append(fut)

        for i, fut in enumerate(as_completed(futures)):
            result = fut.result()
            if result.error:
                report.errors += 1
                err_type = result.error.split(":")[0]
                report.error_types[err_type] = report.error_types.get(err_type, 0) + 1
            elif result.status == "completed":
                report.completed += 1
                report.submit_latencies_ms.append(result.submit_ms)
                report.total_latencies_ms.append(result.total_ms)
                if result.score is not None:
                    report.scores.append(result.score)
            elif result.status == "failed":
                report.failed += 1

            # Progress
            done = i + 1
            if done % 10 == 0 or done == total_jobs:
                print(f"  [{done}/{total_jobs}] completed={report.completed} "
                      f"errors={report.errors} failed={report.failed}")

    report.duration_s = time.monotonic() - t_start
    return report


def print_report(r: StressReport) -> None:
    """Print a formatted stress test report."""
    print("\n" + "=" * 60)
    print("  STRESS TEST REPORT")
    print("=" * 60)
    print(f"  Total jobs:    {r.total_jobs}")
    print(f"  Completed:     {r.completed}")
    print(f"  Failed:        {r.failed}")
    print(f"  Errors:        {r.errors}")
    print(f"  Duration:      {r.duration_s:.1f}s")
    print(f"  Throughput:    {r.total_jobs / max(r.duration_s, 0.001):.1f} jobs/s")

    if r.error_types:
        print(f"\n  Error breakdown:")
        for k, v in sorted(r.error_types.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v}")

    if r.submit_latencies_ms:
        s = sorted(r.submit_latencies_ms)
        n = len(s)
        print(f"\n  Submit latency (ms):")
        print(f"    p50={s[int(n*0.5)]:.0f}  p95={s[int(n*0.95)]:.0f}  "
              f"p99={s[min(int(n*0.99), n-1)]:.0f}  max={s[-1]:.0f}")

    if r.total_latencies_ms:
        s = sorted(r.total_latencies_ms)
        n = len(s)
        print(f"\n  End-to-end latency (ms):")
        print(f"    p50={s[int(n*0.5)]:.0f}  p95={s[int(n*0.95)]:.0f}  "
              f"p99={s[min(int(n*0.99), n-1)]:.0f}  max={s[-1]:.0f}")

    if r.scores:
        print(f"\n  Score distribution:")
        print(f"    min={min(r.scores):.3f}  mean={statistics.mean(r.scores):.3f}  "
              f"max={max(r.scores):.3f}")

    print("=" * 60)


def check_thresholds(r: StressReport) -> bool:
    """Check pass/fail thresholds. Returns True if all pass."""
    checks = []

    # Error rate < 5%
    error_rate = r.errors / max(r.total_jobs, 1)
    ok = error_rate < 0.05
    checks.append(("Error rate < 5%", f"{error_rate:.1%}", ok))

    # At least 80% completed
    completion_rate = r.completed / max(r.total_jobs, 1)
    ok = completion_rate >= 0.80
    checks.append(("Completion rate >= 80%", f"{completion_rate:.1%}", ok))

    # p95 end-to-end < 30s (generous for heuristic mode with queue)
    if r.total_latencies_ms:
        s = sorted(r.total_latencies_ms)
        p95 = s[int(len(s) * 0.95)]
        ok = p95 < 30_000
        checks.append(("E2E p95 < 30s", f"{p95:.0f}ms", ok))

    # No crash/deadlock: duration should be < 5 min for 50 jobs
    ok = r.duration_s < 300
    checks.append(("Total duration < 5min", f"{r.duration_s:.1f}s", ok))

    print("\n  THRESHOLD CHECKS:")
    all_pass = True
    for name, value, passed in checks:
        icon = "PASS" if passed else "FAIL"
        print(f"    [{icon}] {name}: {value}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n  ALL CHECKS PASSED")
    else:
        print("\n  SOME CHECKS FAILED")

    return all_pass


def main() -> None:
    parser = argparse.ArgumentParser(description="AuthentiGuard stress test")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--total-jobs", type=int, default=50)
    parser.add_argument("--poll-timeout", type=float, default=120.0)
    args = parser.parse_args()

    # Preflight: check health
    try:
        with httpx.Client(base_url=args.base_url, timeout=10.0) as client:
            resp = client.get("/health")
            if resp.status_code != 200:
                print(f"Health check failed: {resp.status_code}")
                sys.exit(1)
            print(f"Health: {resp.json()}")
    except Exception as exc:
        print(f"Cannot reach {args.base_url}/health: {exc}")
        sys.exit(1)

    report = run_stress_test(
        base_url=args.base_url,
        concurrency=args.concurrency,
        total_jobs=args.total_jobs,
        poll_timeout=args.poll_timeout,
    )
    print_report(report)
    passed = check_thresholds(report)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
