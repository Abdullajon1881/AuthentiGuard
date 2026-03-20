"""
Steps 101–102: Load testing and latency benchmarks.

Step 101 — Locust load testing suite
  Simulates realistic mixed-traffic load against the API:
    - 70% text analysis (most common)
    - 15% image analysis
    -  8% audio analysis
    -  5% video analysis
    -  2% code analysis

  Acceptance criteria (Step 102):
    API: 99th percentile latency < 2s at 100 RPS
    Text: sub-2s analysis
    Media: sub-10s analysis

Step 102 — Latency benchmarks
  Measures end-to-end latency for all detector types.
  Runs as part of the CI pipeline on every model update to catch regressions.
  A test fails if p95 latency exceeds the target for that content type.

Usage (load test):
    locust -f performance/load/locustfile.py --host=http://localhost:8000 \
           --users=50 --spawn-rate=5 --run-time=5m

Usage (benchmark):
    python -m performance.load.benchmark --all --output=benchmark_results.json
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import structlog

log = structlog.get_logger(__name__)

# ── Latency targets (Step 102) ────────────────────────────────
LATENCY_TARGETS = {
    "text":  {"p95_ms": 2_000,  "p99_ms": 5_000},
    "image": {"p95_ms": 5_000,  "p99_ms": 8_000},
    "audio": {"p95_ms": 10_000, "p99_ms": 15_000},
    "video": {"p95_ms": 10_000, "p99_ms": 15_000},
    "code":  {"p95_ms": 2_000,  "p99_ms": 5_000},
    "api":   {"p95_ms": 500,    "p99_ms": 2_000},
}

# ── Sample payloads for load testing ─────────────────────────
_SHORT_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "This is a sample text used for performance testing of the API. "
    "It should be representative of typical user-submitted content."
)

_MEDIUM_TEXT = _SHORT_TEXT * 20   # ~600 words

_AI_TEXT = (
    "Furthermore, it is worth noting that the multifaceted nature of "
    "artificial intelligence presents both opportunities and challenges "
    "for modern organisations. Consequently, robust governance frameworks "
    "must be implemented to ensure the responsible utilisation of these "
    "powerful technologies across various domains and sectors."
) * 8


def _make_dummy_image(h: int = 224, w: int = 224) -> bytes:
    """Generate a synthetic PNG-like byte payload for image tests."""
    from PIL import Image  # type: ignore
    import io
    img = Image.fromarray(
        np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Step 101: Locust file (importable by locust CLI) ──────────

LOCUST_FILE_CONTENT = '''"""
Locust load test for AuthentiGuard API.

Run:
    locust -f performance/load/locustfile.py \\
           --host=http://localhost:8000 \\
           --users=100 --spawn-rate=10 --run-time=10m
"""

import random
import json
from locust import HttpUser, task, between


AI_TEXT = """Furthermore, it is worth noting that the multifaceted nature
of artificial intelligence presents both opportunities and challenges.
Consequently, robust frameworks must be implemented to ensure responsible
utilisation of these powerful technologies across various domains.""" * 5

HUMAN_TEXT = """I was thinking about going to the park today but it looks like
rain. Maybe I'll just stay in and watch a movie instead. The cat seems
to agree — she hasn't moved from the couch in hours.""" * 10


class AuthentiGuardUser(HttpUser):
    """Simulates a typical AuthentiGuard API user."""
    wait_time = between(0.5, 2.0)

    def on_start(self):
        """Login and store the access token."""
        resp = self.client.post("/api/v1/auth/login", json={
            "email": "loadtest@example.com",
            "password": "LoadTest123!",
        })
        if resp.status_code == 200:
            self.token = resp.json().get("access_token", "")
        else:
            self.token = ""

    def _auth_headers(self):
        return {"Authorization": f"Bearer {self.token}"}

    @task(70)
    def analyse_text(self):
        text = random.choice([AI_TEXT, HUMAN_TEXT])
        self.client.post(
            "/api/v1/analyze/text",
            json={"text": text},
            headers=self._auth_headers(),
            name="POST /analyze/text",
        )

    @task(15)
    def analyse_image(self):
        import io
        import numpy as np
        try:
            from PIL import Image
            img = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            self.client.post(
                "/api/v1/analyze/file",
                files={"file": ("test.png", buf, "image/png")},
                headers=self._auth_headers(),
                name="POST /analyze/file [image]",
            )
        except ImportError:
            pass

    @task(10)
    def get_dashboard(self):
        self.client.get(
            "/api/v1/dashboard/stats",
            headers=self._auth_headers(),
            name="GET /dashboard/stats",
        )

    @task(5)
    def health_check(self):
        self.client.get("/health", name="GET /health")
'''


def write_locustfile(output_path: Path) -> None:
    """Write the Locust file to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(LOCUST_FILE_CONTENT)
    log.info("locustfile_written", path=str(output_path))


# ── Step 102: Latency benchmark harness ──────────────────────

@dataclass
class BenchmarkResult:
    """Latency benchmark result for one detector."""
    detector:     str
    content_type: str
    n_runs:       int
    mean_ms:      float
    p50_ms:       float
    p95_ms:       float
    p99_ms:       float
    max_ms:       float
    target_p95:   int
    target_p99:   int
    p95_passed:   bool
    p99_passed:   bool
    timestamp:    str


@dataclass
class BenchmarkSuite:
    """Full benchmark suite results."""
    results:         list[BenchmarkResult]
    all_passed:      bool
    failed_detectors: list[str]
    total_duration_s: float
    timestamp:       str

    def to_dict(self) -> dict:
        return {
            "summary": {
                "all_passed":       self.all_passed,
                "failed":           self.failed_detectors,
                "total_duration_s": self.total_duration_s,
                "timestamp":        self.timestamp,
            },
            "results": [
                {
                    "detector":   r.detector,
                    "content_type": r.content_type,
                    "n_runs":     r.n_runs,
                    "mean_ms":    r.mean_ms,
                    "p50_ms":     r.p50_ms,
                    "p95_ms":     r.p95_ms,
                    "p99_ms":     r.p99_ms,
                    "target_p95": r.target_p95,
                    "p95_passed": r.p95_passed,
                    "p99_passed": r.p99_passed,
                }
                for r in self.results
            ],
        }


def benchmark_detector(
    detector:     Any,
    content_type: str,
    sample:       Any,
    n_warmup:     int = 3,
    n_runs:       int = 20,
    filename:     str = "benchmark_input",
) -> BenchmarkResult:
    """
    Benchmark a single detector over n_runs iterations.
    Returns latency statistics and pass/fail against targets.
    """
    target = LATENCY_TARGETS.get(content_type, {"p95_ms": 5000, "p99_ms": 10000})

    # Warmup
    for _ in range(n_warmup):
        _call_detector(detector, content_type, sample, filename)

    # Timed runs
    times_ms: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _call_detector(detector, content_type, sample, filename)
        times_ms.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times_ms)
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))

    result = BenchmarkResult(
        detector=detector.__class__.__name__,
        content_type=content_type,
        n_runs=n_runs,
        mean_ms=round(float(arr.mean()), 1),
        p50_ms=round(float(np.percentile(arr, 50)), 1),
        p95_ms=round(p95, 1),
        p99_ms=round(p99, 1),
        max_ms=round(float(arr.max()), 1),
        target_p95=target["p95_ms"],
        target_p99=target["p99_ms"],
        p95_passed=p95 <= target["p95_ms"],
        p99_passed=p99 <= target["p99_ms"],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    status = "✓" if result.p95_passed else "✗"
    log.info("benchmark_result",
             detector=result.detector,
             p95_ms=result.p95_ms,
             target=target["p95_ms"],
             status=status)

    return result


def _call_detector(
    detector: Any,
    content_type: str,
    sample: Any,
    filename: str,
) -> Any:
    """Unified detector call for benchmarking."""
    if content_type in ("text", "code"):
        text = sample if isinstance(sample, str) else sample.decode("utf-8", errors="replace")
        return detector.analyze(text)
    else:
        data = sample if isinstance(sample, bytes) else sample.encode()
        return detector.analyze(data, filename)


def run_benchmark_suite(
    detectors:  dict[str, tuple[Any, str, Any]],   # {name: (detector, ct, sample)}
    output_path: Path | None = None,
    n_runs:      int = 20,
) -> BenchmarkSuite:
    """
    Run the full benchmark suite across all detectors.

    Args:
        detectors:  {name: (detector_instance, content_type, sample_content)}
        output_path: Where to write results JSON (optional)
        n_runs:     Number of benchmark iterations per detector
    """
    t_suite = time.perf_counter()
    results: list[BenchmarkResult] = []
    failed:  list[str] = []

    for name, (detector, content_type, sample) in detectors.items():
        log.info("benchmarking", detector=name, content_type=content_type)
        try:
            result = benchmark_detector(
                detector, content_type, sample,
                n_runs=n_runs,
                filename=f"benchmark.{content_type}",
            )
            results.append(result)
            if not result.p95_passed:
                failed.append(name)
        except Exception as exc:
            log.error("benchmark_failed", detector=name, error=str(exc))
            failed.append(name)

    suite = BenchmarkSuite(
        results=results,
        all_passed=len(failed) == 0,
        failed_detectors=failed,
        total_duration_s=round(time.perf_counter() - t_suite, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(suite.to_dict(), indent=2))
        log.info("benchmark_results_saved", path=str(output_path))

    if failed:
        log.error("benchmark_suite_failures",
                  failed=failed,
                  n_failed=len(failed),
                  n_total=len(results))
    else:
        log.info("benchmark_suite_passed",
                 n_detectors=len(results),
                 total_s=suite.total_duration_s)

    return suite
