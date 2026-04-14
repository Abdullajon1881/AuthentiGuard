"""
Shared Prometheus metrics — importable by API, workers, and middleware.

All metrics are registered in the default prometheus_client registry,
which is scraped at GET /metrics.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ── HTTP-level (also defined in main.py middleware — kept in sync) ────
# These are created in main.py's prometheus_middleware. We only define
# business metrics here to avoid duplicate registration.

# ── Detection pipeline ────────────────────────────────────────────────

DETECTION_DURATION = Histogram(
    "detection_duration_seconds",
    "End-to-end detection latency (worker processing time)",
    ["content_type", "detector_mode"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)

DETECTION_SCORE = Histogram(
    "detection_score",
    "Distribution of AI detection scores",
    ["content_type"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

DETECTION_JOBS_TOTAL = Counter(
    "detection_jobs_total",
    "Total detection jobs by outcome",
    ["status"],  # completed, failed, timeout
)

# ── Rate limiting ─────────────────────────────────────────────────────

RATE_LIMIT_HITS = Counter(
    "rate_limit_hits_total",
    "Requests rejected by rate limiter",
    ["tier"],
)

# ── Model loading ─────────────────────────────────────────────────────

MODEL_LOAD_DURATION = Gauge(
    "model_load_duration_seconds",
    "Time taken to load the ML model on worker startup",
)

DETECTOR_FALLBACK = Gauge(
    "detector_fallback_active",
    "1 if fallback detector is active, 0 if ML detector loaded",
)

# ── Worker health ─────────────────────────────────────────────────────

STUCK_JOBS_CLEANED = Counter(
    "stuck_jobs_cleaned_total",
    "Jobs cleaned up by the periodic cleanup task",
    ["reason"],  # processing_timeout, pending_timeout
)

# ── Queue depth ──────────────────────────────────────────────────────

QUEUE_DEPTH = Gauge(
    "celery_queue_depth",
    "Current number of messages in a Celery queue",
    ["queue"],
)
