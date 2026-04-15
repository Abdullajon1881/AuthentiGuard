"""
Base detection worker — shared logic for all content-type workers.

Eliminates ~312 lines of duplicated code across text/image/audio/video workers.
Each concrete worker only needs to implement:
  - get_detector() → lazy-loaded detector singleton
  - get_input(job) → extract input data from job (text, bytes, etc.)
  - run_detection(detector, input_data, job) → run the detector
  - build_result(job, output, elapsed_ms) → create DetectionResult
"""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog
from celery.exceptions import SoftTimeLimitExceeded  # type: ignore
from sqlalchemy import select, update

from ..models.models import DetectionJob, DetectionResult, JobStatus

log = structlog.get_logger(__name__)

# Shared event loop per worker process — avoids creating/destroying loops per task.
# Safe with Celery prefork pool (one process = one loop).
_worker_loop: asyncio.AbstractEventLoop | None = None


def run_async(coro):
    """Run an async coroutine in the worker's dedicated event loop."""
    global _worker_loop
    if _worker_loop is None or _worker_loop.is_closed():
        _worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_worker_loop)
    return _worker_loop.run_until_complete(coro)


# Sentinel returned by _claim_job when the job is owned by a live in-flight
# attempt elsewhere, or already reached a terminal state — caller must exit
# cleanly (no raise).
_ALREADY_HANDLED = object()

# A PROCESSING row whose `started_at` is older than this is considered stale
# (the owning worker almost certainly died — OOM kill, hard timeout, SIGKILL).
# Kept in sync with cleanup.STUCK_PROCESSING_MINUTES so reclaim happens within
# the same window that the periodic cleanup would have flipped the row to FAILED.
STALE_PROCESSING_MINUTES = 10


async def _claim_job(
    db,
    job,
    job_id: str,
    content_type: str,
    max_attempts: int = 3,
) -> int | object:
    """
    Transition PENDING → PROCESSING (or reclaim stale PROCESSING) with
    optimistic locking and re-fetch-on-conflict.

    On conflict we re-fetch authoritative state and decide per-status:
      - COMPLETED / FAILED                   → _ALREADY_HANDLED (terminal)
      - PROCESSING with fresh `started_at`   → _ALREADY_HANDLED (live owner)
      - PROCESSING with stale `started_at`   → reclaim: bump version + started_at
      - PENDING with newer version           → retry claim with fresh version

    Returns:
      int — new version number on successful claim/reclaim
      _ALREADY_HANDLED — caller should exit cleanly without raising
    Raises:
      RuntimeError — persistent conflict after max_attempts, or job missing
    """
    stale_cutoff_delta = timedelta(minutes=STALE_PROCESSING_MINUTES)
    current_version = job.version

    for attempt in range(max_attempts):
        now = datetime.now(timezone.utc)

        rows = await db.execute(
            update(DetectionJob)
            .where(
                DetectionJob.id == job.id,
                DetectionJob.version == current_version,
                DetectionJob.status == JobStatus.PENDING,
            )
            .values(
                status=JobStatus.PROCESSING,
                started_at=now,
                version=current_version + 1,
            )
        )
        await db.commit()

        if rows.rowcount == 1:
            return current_version + 1

        # Conflict — re-fetch the authoritative state
        result = await db.execute(
            select(DetectionJob).where(DetectionJob.id == job.id)
        )
        fresh = result.scalar_one_or_none()
        if fresh is None:
            raise RuntimeError(f"job disappeared: {job_id}")

        if fresh.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            log.info(
                "job_already_terminal_skipping",
                job_id=job_id,
                content_type=content_type,
                status=fresh.status.value,
                attempt=attempt + 1,
            )
            return _ALREADY_HANDLED

        if fresh.status == JobStatus.PROCESSING:
            # Distinguish a live in-flight attempt from an abandoned one.
            started_at = fresh.started_at
            is_stale = (
                started_at is None
                or (now - started_at) >= stale_cutoff_delta
            )

            if not is_stale:
                log.info(
                    "job_owned_by_live_worker_skipping",
                    job_id=job_id,
                    content_type=content_type,
                    started_at=started_at.isoformat() if started_at else None,
                    attempt=attempt + 1,
                )
                return _ALREADY_HANDLED

            # Stale PROCESSING — previous worker almost certainly died. Reclaim
            # atomically: same status, fresh started_at, version bump. If
            # another reclaimer wins the race, loop and re-inspect.
            reclaim = await db.execute(
                update(DetectionJob)
                .where(
                    DetectionJob.id == job.id,
                    DetectionJob.version == fresh.version,
                    DetectionJob.status == JobStatus.PROCESSING,
                )
                .values(
                    started_at=now,
                    version=fresh.version + 1,
                )
            )
            await db.commit()
            if reclaim.rowcount == 1:
                log.warning(
                    "stale_processing_job_reclaimed",
                    job_id=job_id,
                    content_type=content_type,
                    previous_started_at=started_at.isoformat() if started_at else None,
                    attempt=attempt + 1,
                )
                return fresh.version + 1
            # Lost the reclaim race — fall through to next loop iteration.
            current_version = fresh.version
            continue

        # Still PENDING but row version was bumped (rare) — refresh and retry
        current_version = fresh.version

    raise RuntimeError(
        f"optimistic lock conflict on claim after {max_attempts} attempts: {job_id}"
    )


class BaseDetectionWorker(ABC):
    """Abstract base for all detection workers."""

    content_type: str  # "text", "image", "audio", "video"

    @abstractmethod
    def get_detector(self) -> Any:
        """Return the lazy-loaded detector instance."""

    @abstractmethod
    async def get_input(self, job: DetectionJob) -> Any:
        """Extract input data from the job (text string, image bytes, etc.)."""

    @abstractmethod
    def run_detection(self, detector: Any, input_data: Any, job: DetectionJob) -> Any:
        """Run the detector on the input data. Returns detector-specific output."""

    @abstractmethod
    def build_result(
        self,
        job: DetectionJob,
        detection_output: Any,
        elapsed_ms: int,
    ) -> DetectionResult:
        """Convert detector output to a DetectionResult ORM instance."""

    async def execute(self, job_id: str) -> dict:
        """
        Common detection flow: fetch → validate → process → persist → webhook.

        Returns dict with job_id, score, label, processing_ms on success,
        or dict with error key on failure.
        Raises self.retry()-compatible exceptions for transient errors.
        """
        from ..core.database import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            # ── Fetch job ────────────────────────────────────
            result = await db.execute(
                select(DetectionJob).where(DetectionJob.id == uuid.UUID(job_id))
            )
            job = result.scalar_one_or_none()

            if not job:
                log.error("job_not_found", job_id=job_id, content_type=self.content_type)
                return {"error": "Job not found"}

            # ── Claim the job (PENDING → PROCESSING) ─────────
            # Retries on optimistic lock conflict; returns _ALREADY_HANDLED if the
            # job is already terminal (e.g. cleanup marked it FAILED before a retry).
            claim = await _claim_job(db, job, job_id, self.content_type)
            if claim is _ALREADY_HANDLED:
                return {"job_id": job_id, "skipped": True, "reason": "already_handled"}
            current_version = claim  # type: ignore[assignment]

            try:
                start_ms = int(time.time() * 1000)

                # ── Get input data ───────────────────────────
                input_data = await self.get_input(job)

                # ── Run detection ────────────────────────────
                detector = self.get_detector()
                detection_output = self.run_detection(detector, input_data, job)

                elapsed_ms = int(time.time() * 1000) - start_ms

                # ── Build result (but don't persist yet) ─────
                detection_result = self.build_result(job, detection_output, elapsed_ms)

                # ── Mark completed (optimistic lock) ─────────
                rows = await db.execute(
                    update(DetectionJob)
                    .where(
                        DetectionJob.id == job.id,
                        DetectionJob.version == current_version,
                    )
                    .values(
                        status=JobStatus.COMPLETED,
                        completed_at=datetime.now(timezone.utc),
                        version=current_version + 1,
                    )
                )

                if rows.rowcount == 0:
                    await db.rollback()
                    # Someone (usually cleanup) flipped the row while we were working.
                    # Re-fetch: if already terminal, accept it silently; otherwise raise
                    # so Celery's retry mechanism can take another run.
                    refetch = await db.execute(
                        select(DetectionJob).where(DetectionJob.id == job.id)
                    )
                    fresh = refetch.scalar_one_or_none()
                    if fresh is not None and fresh.status in (
                        JobStatus.COMPLETED, JobStatus.FAILED,
                    ):
                        log.warning(
                            "completion_raced_with_terminal_state",
                            job_id=job_id,
                            observed_status=fresh.status.value,
                        )
                        return {"job_id": job_id, "skipped": True, "reason": "raced"}
                    raise RuntimeError(
                        f"optimistic lock conflict on completion: {job_id}"
                    )

                # ── Persist result only after version check passes ──
                db.add(detection_result)
                await db.commit()

                log.info(
                    f"{self.content_type}_detection_complete",
                    job_id=job_id,
                    score=detection_result.authenticity_score,
                    label=detection_result.label,
                    ms=elapsed_ms,
                )

                # ── Record Prometheus metrics ────────────────
                try:
                    from ..core.metrics import (
                        DETECTION_DURATION, DETECTION_SCORE, DETECTION_JOBS_TOTAL,
                    )
                    detector_mode = detection_result.evidence_summary.get("detector_mode", "unknown")
                    DETECTION_DURATION.labels(
                        content_type=self.content_type,
                        detector_mode=detector_mode,
                    ).observe(elapsed_ms / 1000.0)
                    DETECTION_SCORE.labels(
                        content_type=self.content_type,
                    ).observe(detection_result.authenticity_score)
                    DETECTION_JOBS_TOTAL.labels(status="completed").inc()
                except Exception:
                    pass  # Metrics must never break detection

                # ── Trigger webhook ──────────────────────────
                from .webhook_worker import dispatch_webhook
                dispatch_webhook.delay(job_id, "job.completed")

                return {
                    "job_id": job_id,
                    "score": detection_result.authenticity_score,
                    "label": detection_result.label,
                    "processing_ms": elapsed_ms,
                }

            except SoftTimeLimitExceeded:
                # Worker is about to be killed — mark FAILED immediately, don't retry.
                # Keep this handler minimal: we have ~60s before the hard kill (180s).
                log.error(
                    f"{self.content_type}_soft_timeout",
                    job_id=job_id,
                )
                await db.execute(
                    update(DetectionJob)
                    .where(
                        DetectionJob.id == job.id,
                        DetectionJob.version == current_version,
                    )
                    .values(
                        status=JobStatus.FAILED,
                        error_message="Detection timed out (exceeded 120s soft limit)",
                        completed_at=datetime.now(timezone.utc),
                        version=current_version + 1,
                    )
                )
                await db.commit()
                try:
                    from ..core.metrics import DETECTION_JOBS_TOTAL
                    DETECTION_JOBS_TOTAL.labels(status="timeout").inc()
                except Exception:
                    pass
                return {"error": "timeout", "job_id": job_id}

            except Exception as exc:
                log.error(
                    f"{self.content_type}_detection_failed",
                    job_id=job_id,
                    error=str(exc),
                )
                await db.execute(
                    update(DetectionJob)
                    .where(
                        DetectionJob.id == job.id,
                        DetectionJob.version == current_version,
                    )
                    .values(
                        status=JobStatus.FAILED,
                        error_message=str(exc)[:500],
                        completed_at=datetime.now(timezone.utc),
                        version=current_version + 1,
                    )
                )
                await db.commit()

                try:
                    from ..core.metrics import DETECTION_JOBS_TOTAL
                    DETECTION_JOBS_TOTAL.labels(status="failed").inc()
                except Exception:
                    pass

                # Trigger failure webhook
                from .webhook_worker import dispatch_webhook
                dispatch_webhook.delay(job_id, "job.failed")

                # Standardized retry logic:
                # ValueError = user's fault (bad input) → don't retry
                # Everything else = transient → retry
                if isinstance(exc, ValueError):
                    return {"error": str(exc)}

                raise  # Let Celery's retry mechanism handle it
