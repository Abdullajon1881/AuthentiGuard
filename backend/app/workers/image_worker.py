"""
Image detection worker — Celery task for AI-generated image detection.

Celery task that:
  1. Downloads image bytes from S3
  2. Runs the ImageDetector ensemble (EfficientNet-B4 + ViT-B/16)
  3. Writes DetectionResult to Postgres
  4. Triggers the webhook worker on completion
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

import structlog
from celery import Task  # type: ignore
from sqlalchemy import select

from .celery_app import celery_app
from ..models.models import DetectionJob, DetectionResult, JobStatus

log = structlog.get_logger(__name__)

# Module-level detector — loaded once per worker process, not per task
_detector = None

MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50 MB


def _get_detector():
    global _detector
    if _detector is None:
        from pathlib import Path
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../"))
        from ai.image_detector.image_detector import ImageDetector  # type: ignore

        _detector = ImageDetector(
            checkpoint_dir=Path("ai/image_detector/checkpoints/phase3"),
        )
        _detector.load_models()
        log.info("image_detector_loaded_in_worker")
    return _detector


@celery_app.task(
    bind=True,
    name="workers.image_worker.run_image_detection",
    queue="image",
    max_retries=3,
    default_retry_delay=10,
)
def run_image_detection(self: Task, job_id: str) -> dict:
    """
    Celery task: run full image detection ensemble for a job.

    Args:
        job_id: UUID string of the DetectionJob to process.

    Returns:
        dict with job_id, score, label, processing_ms.
    """
    from ..core.database import AsyncSessionLocal
    import asyncio

    async def _run() -> dict:
        async with AsyncSessionLocal() as db:
            # -- Fetch job --
            result = await db.execute(
                select(DetectionJob).where(DetectionJob.id == uuid.UUID(job_id))
            )
            job = result.scalar_one_or_none()

            if not job:
                log.error("job_not_found", job_id=job_id)
                return {"error": "Job not found"}

            # -- Mark as processing --
            job.status     = JobStatus.PROCESSING
            job.started_at = datetime.now(timezone.utc)
            await db.commit()

            try:
                # -- Validate --
                if not job.s3_key:
                    raise ValueError("Image job has no S3 key — images must be uploaded as files")

                if job.file_size and job.file_size > MAX_IMAGE_SIZE:
                    raise ValueError(
                        f"Image file too large ({job.file_size / 1024 / 1024:.1f} MB, max {MAX_IMAGE_SIZE / 1024 / 1024:.0f} MB)"
                    )

                # -- Download from S3 --
                start_ms = int(time.time() * 1000)
                from ..services.s3_service import fetch_from_s3
                data = await fetch_from_s3(job.s3_key)

                # -- Run detection --
                detector = _get_detector()
                image_result = detector.analyze(data, job.file_name or "image.jpg")

                elapsed_ms = int(time.time() * 1000) - start_ms

                # -- Persist result --
                detection_result = DetectionResult(
                    job_id=job.id,
                    authenticity_score=image_result.score,
                    confidence=image_result.confidence,
                    label=image_result.label,
                    layer_scores=image_result.model_scores,
                    evidence_summary=image_result.evidence,
                    sentence_scores=[],
                    processing_ms=elapsed_ms,
                )
                db.add(detection_result)

                job.status       = JobStatus.COMPLETED
                job.completed_at = datetime.now(timezone.utc)
                await db.commit()

                log.info(
                    "image_detection_complete",
                    job_id=job_id,
                    score=image_result.score,
                    label=image_result.label,
                    ms=elapsed_ms,
                )

                # -- Trigger webhook --
                from .webhook_worker import dispatch_webhook
                dispatch_webhook.delay(job_id, "job.completed")

                return {
                    "job_id":        job_id,
                    "score":         image_result.score,
                    "label":         image_result.label,
                    "processing_ms": elapsed_ms,
                }

            except Exception as exc:
                log.error("image_detection_failed", job_id=job_id, error=str(exc))
                job.status        = JobStatus.FAILED
                job.error_message = str(exc)
                job.completed_at  = datetime.now(timezone.utc)
                await db.commit()

                # Trigger failure webhook
                from .webhook_worker import dispatch_webhook
                dispatch_webhook.delay(job_id, "job.failed")

                # Retry transient errors (not validation errors)
                if not isinstance(exc, ValueError):
                    raise self.retry(exc=exc, countdown=10)

                return {"error": str(exc)}

    return asyncio.run(_run())
