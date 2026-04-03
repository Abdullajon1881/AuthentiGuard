"""
Periodic cleanup task — marks stuck PROCESSING jobs as FAILED.

Jobs can get stuck if a worker crashes mid-detection. This task runs
every 5 minutes and fails any job that has been PROCESSING for more
than 10 minutes without completing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import update

from .celery_app import celery_app
from ..models.models import DetectionJob, JobStatus

STUCK_THRESHOLD_MINUTES = 10


@celery_app.task(name="app.workers.cleanup.cleanup_stuck_jobs", queue="webhook")
def cleanup_stuck_jobs() -> dict:
    """Find and fail jobs stuck in PROCESSING state."""
    import asyncio
    from ..core.database import AsyncSessionLocal

    async def _cleanup() -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=STUCK_THRESHOLD_MINUTES)
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                update(DetectionJob)
                .where(
                    DetectionJob.status == JobStatus.PROCESSING,
                    DetectionJob.started_at < cutoff,
                )
                .values(
                    status=JobStatus.FAILED,
                    error_message=f"Job timed out after {STUCK_THRESHOLD_MINUTES} minutes (worker may have crashed)",
                    completed_at=datetime.now(timezone.utc),
                )
            )
            await db.commit()
            return result.rowcount

    from .base_worker import run_async
    count = run_async(_cleanup())
    if count > 0:
        import structlog
        structlog.get_logger().warning("stuck_jobs_cleaned", count=count)
    return {"cleaned": count}
