"""
Periodic cleanup task — marks stuck jobs as FAILED.

Handles two failure modes:
  1. PROCESSING jobs stuck >10 min (worker crash / SoftTimeLimitExceeded)
  2. PENDING jobs stuck >30 min (never picked up — no workers, queue full)

Runs every 5 minutes via Celery Beat.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy import update

from .celery_app import celery_app
from ..models.models import DetectionJob, JobStatus

log = structlog.get_logger(__name__)

STUCK_PROCESSING_MINUTES = 10
STUCK_PENDING_MINUTES = 30


@celery_app.task(name="app.workers.cleanup.cleanup_stuck_jobs", queue="text")
def cleanup_stuck_jobs() -> dict:
    """Find and fail jobs stuck in PROCESSING or PENDING state."""
    from .base_worker import run_async

    async def _cleanup() -> dict:
        from ..core.database import AsyncSessionLocal

        now = datetime.now(timezone.utc)
        processing_cutoff = now - timedelta(minutes=STUCK_PROCESSING_MINUTES)
        pending_cutoff = now - timedelta(minutes=STUCK_PENDING_MINUTES)

        async with AsyncSessionLocal() as db:
            # 1. Stuck PROCESSING jobs (worker crash, timeout kill)
            r1 = await db.execute(
                update(DetectionJob)
                .where(
                    DetectionJob.status == JobStatus.PROCESSING,
                    DetectionJob.started_at < processing_cutoff,
                )
                .values(
                    status=JobStatus.FAILED,
                    error_message=(
                        f"Job timed out after {STUCK_PROCESSING_MINUTES} minutes "
                        "(worker may have crashed or been killed by OOM/timeout)"
                    ),
                    completed_at=now,
                )
            )

            # 2. Stuck PENDING jobs (never picked up)
            r2 = await db.execute(
                update(DetectionJob)
                .where(
                    DetectionJob.status == JobStatus.PENDING,
                    DetectionJob.created_at < pending_cutoff,
                )
                .values(
                    status=JobStatus.FAILED,
                    error_message=(
                        f"Job was not picked up within {STUCK_PENDING_MINUTES} minutes "
                        "(no available workers or queue backlog)"
                    ),
                    completed_at=now,
                )
            )

            await db.commit()
            return {
                "processing": r1.rowcount,
                "pending": r2.rowcount,
            }

    counts = run_async(_cleanup())
    total = counts["processing"] + counts["pending"]
    if total > 0:
        log.warning(
            "stuck_jobs_cleaned",
            processing=counts["processing"],
            pending=counts["pending"],
        )
        try:
            from ..core.metrics import STUCK_JOBS_CLEANED
            if counts["processing"] > 0:
                STUCK_JOBS_CLEANED.labels(reason="processing_timeout").inc(counts["processing"])
            if counts["pending"] > 0:
                STUCK_JOBS_CLEANED.labels(reason="pending_timeout").inc(counts["pending"])
        except Exception:
            pass
    return {"cleaned_processing": counts["processing"], "cleaned_pending": counts["pending"]}
