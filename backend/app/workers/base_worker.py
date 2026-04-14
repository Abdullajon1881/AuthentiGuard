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
from datetime import datetime, timezone
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

            # ── Mark as processing (optimistic lock) ─────────
            current_version = job.version
            rows = await db.execute(
                update(DetectionJob)
                .where(
                    DetectionJob.id == job.id,
                    DetectionJob.version == current_version,
                )
                .values(
                    status=JobStatus.PROCESSING,
                    started_at=datetime.now(timezone.utc),
                    version=current_version + 1,
                )
            )
            await db.commit()

            if rows.rowcount == 0:
                log.warning("optimistic_lock_conflict", job_id=job_id, phase="processing")
                return {"error": "Job was modified by another worker"}

            # Refresh version for next transition
            current_version += 1

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
                    log.warning("optimistic_lock_conflict", job_id=job_id, phase="completed")
                    return {"error": "Job was modified by another worker"}

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
