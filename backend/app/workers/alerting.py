"""
Minimal internal alerting — logs warnings when production thresholds are breached
and, if `ALERT_WEBHOOK_URL` is configured, posts a short JSON notification to
an external webhook (Slack incoming-webhook or any generic endpoint).

Checks (every 60 seconds via Celery Beat):
  1. Detector fallback active → CRITICAL
  2. Job failure rate > 10% in last 5 min → WARNING
  3. Queue depth > 100 → WARNING

Design notes:
  - structlog output is ALWAYS emitted (pod logs remain the audit trail).
  - External webhook is best-effort; a failing webhook never breaks the task.
  - Cooldown prevents a firing alert from spamming the channel every 60s.
  - No hard dependency on Prometheus/Alertmanager — this is the launch-day
    fallback path so at least one human gets notified when something breaks.
"""

from __future__ import annotations

import os
import time
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

# Per-alert-key cooldown so a firing condition does not spam the webhook on
# every 60-second tick. Keys are short strings like "detector_fallback" or
# "queue_depth_high:video". Values are monotonic timestamps of last send.
_ALERT_COOLDOWN_SECONDS = 15 * 60  # 15 minutes between repeat notifications
_last_sent: dict[str, float] = {}


def _should_send(key: str) -> bool:
    """True if we haven't sent this alert in the cooldown window."""
    now = time.monotonic()
    prev = _last_sent.get(key)
    if prev is not None and (now - prev) < _ALERT_COOLDOWN_SECONDS:
        return False
    _last_sent[key] = now
    return True


def _post_alert_webhook(key: str, severity: str, message: str, extra: dict | None = None) -> None:
    """Best-effort POST to ALERT_WEBHOOK_URL. Never raises.

    Payload shape is Slack-compatible (`{"text": ...}`) with extra keys for
    non-Slack consumers. Respects the per-key cooldown to prevent spam.
    """
    try:
        from ..core.config import get_settings
        settings = get_settings()
        url = settings.ALERT_WEBHOOK_URL
        if not url:
            return
        if not _should_send(key):
            return

        payload: dict = {
            "text": f"[{severity.upper()}] AuthentiGuard: {message}",
            "severity": severity,
            "alert": key,
            "env": settings.APP_ENV,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            payload["details"] = extra

        # httpx is already a backend dep (used by webhook_worker). Import
        # lazily so this module stays importable in environments that don't
        # need alerting (tests, tooling).
        import httpx
        with httpx.Client(timeout=settings.ALERT_WEBHOOK_TIMEOUT_SECONDS) as client:
            resp = client.post(url, json=payload)
            if resp.status_code >= 400:
                log.error(
                    "alert_webhook_non_2xx",
                    status=resp.status_code,
                    alert=key,
                )
            else:
                log.info("alert_webhook_sent", alert=key, severity=severity)
    except Exception as exc:
        # Alerting must not break the health check itself. The structlog
        # entries at the call sites above are still the canonical record.
        log.error("alert_webhook_failed", alert=key, error=str(exc))


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
            _post_alert_webhook(
                key="detector_fallback",
                severity="critical",
                message="ML text detector is in heuristic fallback mode — accuracy is degraded.",
                extra={"detector_mode": mode},
            )
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
                _post_alert_webhook(
                    key="high_failure_rate",
                    severity="warning",
                    message=(
                        f"Job failure rate is {rate:.1%} over the last "
                        f"{LOOKBACK_MINUTES} min (threshold {FAILURE_RATE_THRESHOLD:.0%})."
                    ),
                    extra={
                        "failed": counts["failed"],
                        "completed": counts["completed"],
                        "window_minutes": LOOKBACK_MINUTES,
                    },
                )
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
                _post_alert_webhook(
                    key=f"queue_depth_high:{queue_name}",
                    severity="warning",
                    message=(
                        f"Celery queue `{queue_name}` depth is {depth} "
                        f"(threshold {QUEUE_DEPTH_THRESHOLD})."
                    ),
                    extra={"queue": queue_name, "depth": depth},
                )
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
