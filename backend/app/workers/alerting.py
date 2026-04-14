"""
Minimal internal alerting — logs warnings when production thresholds are breached.

Checks (every 60 seconds via Celery Beat):
  1. Detector fallback active → CRITICAL
  2. Job failure rate > 10% in last 5 min → WARNING
  3. Queue depth > 100 → WARNING

No external services — structlog + Prometheus metrics only.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy import func, select

from .celery_app import celery_app
from ..models.models import DetectionJob, JobStatus

log = structlog.get_logger("alerting")

# Thresholds
FAILURE_RATE_THRESHOLD = 0.10  # 10%
QUEUE_DEPTH_THRESHOLD = 100
LOOKBACK_MINUTES = 5


@celery_app.task(name="app.workers.alerting.check_health", queue="text")
def check_health() -> dict:
    """Periodic health check — fires alerts via structured logging."""
    from .base_worker import run_async

    alerts: list[dict] = []

    # ── 1. Detector fallback ─────────────────────────────────
    try:
        from .text_worker import get_detector_mode
        mode = get_detector_mode()
        if mode == "fallback":
            log.critical(
                "ALERT_DETECTOR_FALLBACK",
                detector_mode=mode,
                action="ML models failed to load — heuristic fallback is active",
            )
            alerts.append({"type": "detector_fallback", "severity": "critical"})
        try:
            from ..core.metrics import DETECTOR_FALLBACK
            DETECTOR_FALLBACK.set(1 if mode == "fallback" else 0)
        except Exception:
            pass
    except Exception as exc:
        log.error("alert_check_detector_failed", error=str(exc))

    # ── 2. Job failure rate ──────────────────────────────────
    try:
        counts = run_async(_check_failure_rate())
        total = counts["completed"] + counts["failed"]
        if total > 0:
            rate = counts["failed"] / total
            if rate > FAILURE_RATE_THRESHOLD:
                log.warning(
                    "ALERT_HIGH_FAILURE_RATE",
                    failure_rate=round(rate, 3),
                    failed=counts["failed"],
                    completed=counts["completed"],
                    window_minutes=LOOKBACK_MINUTES,
                    threshold=FAILURE_RATE_THRESHOLD,
                )
                alerts.append({
                    "type": "high_failure_rate",
                    "severity": "warning",
                    "rate": round(rate, 3),
                })
    except Exception as exc:
        log.error("alert_check_failure_rate_failed", error=str(exc))

    # ── 3. Queue depth ───────────────────────────────────────
    try:
        depths = _check_queue_depths()
        for queue_name, depth in depths.items():
            try:
                from ..core.metrics import QUEUE_DEPTH
                QUEUE_DEPTH.labels(queue=queue_name).set(depth)
            except Exception:
                pass
            if depth > QUEUE_DEPTH_THRESHOLD:
                log.warning(
                    "ALERT_QUEUE_DEPTH_HIGH",
                    queue=queue_name,
                    depth=depth,
                    threshold=QUEUE_DEPTH_THRESHOLD,
                )
                alerts.append({
                    "type": "queue_depth_high",
                    "severity": "warning",
                    "queue": queue_name,
                    "depth": depth,
                })
    except Exception as exc:
        log.error("alert_check_queue_depth_failed", error=str(exc))

    if not alerts:
        log.debug("health_check_ok")

    return {"alerts": alerts, "alert_count": len(alerts)}


async def _check_failure_rate() -> dict:
    """Query recent job outcomes from the database."""
    from ..core.database import AsyncSessionLocal

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=LOOKBACK_MINUTES)

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(
                DetectionJob.status,
                func.count().label("cnt"),
            )
            .where(DetectionJob.completed_at >= cutoff)
            .where(DetectionJob.status.in_([JobStatus.COMPLETED, JobStatus.FAILED]))
            .group_by(DetectionJob.status)
        )
        rows = {row.status: row.cnt for row in result}
        return {
            "completed": rows.get(JobStatus.COMPLETED, 0),
            "failed": rows.get(JobStatus.FAILED, 0),
        }


def _check_queue_depths() -> dict[str, int]:
    """Check Redis queue lengths for all Celery queues."""
    try:
        from ..core.config import get_settings
        import redis

        settings = get_settings()
        r = redis.from_url(settings.REDIS_URL, socket_timeout=5)

        queues = ["text", "image", "audio", "video", "webhook"]
        depths = {}
        for q in queues:
            depth = r.llen(q)
            depths[q] = depth
        return depths
    except Exception as exc:
        log.error("queue_depth_check_failed", error=str(exc))
        return {}
