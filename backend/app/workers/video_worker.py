"""
Video detection worker — Celery task for video deepfake detection.

Celery task that:
  1. Downloads video bytes from S3
  2. Runs the VideoDetector (frame extraction → face detection →
     artifact analysis → temporal consistency → XceptionNet/EfficientNet/ViT ensemble)
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

_detector = None

MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500 MB


def _get_detector():
    global _detector
    if _detector is None:
        from pathlib import Path
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../"))
        from ai.video_detector.video_detector import VideoDetector  # type: ignore

        _detector = VideoDetector(
            checkpoint_dir=Path("ai/video_detector/checkpoints/phase3"),
        )
        _detector.load_models()
        log.info("video_detector_loaded_in_worker")
    return _detector


@celery_app.task(
    bind=True,
    name="workers.video_worker.run_video_detection",
    queue="video",
    max_retries=3,
    default_retry_delay=30,
    soft_time_limit=300,
    time_limit=360,
)
def run_video_detection(self: Task, job_id: str) -> dict:
    """
    Celery task: run full video deepfake detection for a job.

    Args:
        job_id: UUID string of the DetectionJob to process.

    Returns:
        dict with job_id, score, label, processing_ms.
    """
    from ..core.database import AsyncSessionLocal
    import asyncio

    async def _run() -> dict:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(DetectionJob).where(DetectionJob.id == uuid.UUID(job_id))
            )
            job = result.scalar_one_or_none()

            if not job:
                log.error("job_not_found", job_id=job_id)
                return {"error": "Job not found"}

            job.status     = JobStatus.PROCESSING
            job.started_at = datetime.now(timezone.utc)
            await db.commit()

            try:
                if not job.s3_key:
                    raise ValueError("Video job has no S3 key — videos must be uploaded as files")

                if job.file_size and job.file_size > MAX_VIDEO_SIZE:
                    raise ValueError(
                        f"Video file too large ({job.file_size / 1024 / 1024:.1f} MB, max {MAX_VIDEO_SIZE / 1024 / 1024:.0f} MB)"
                    )

                start_ms = int(time.time() * 1000)
                from ..services.s3_service import fetch_from_s3
                data = await fetch_from_s3(job.s3_key)

                detector = _get_detector()
                video_result = detector.analyze(data, job.file_name or "video.mp4")

                elapsed_ms = int(time.time() * 1000) - start_ms

                # Map frame results to sentence_scores for timeline view
                frame_scores = [
                    {
                        "text": f"{fr.timestamp_s:.1f}s",
                        "score": fr.frame_score,
                        "evidence": fr.model_scores,
                    }
                    for fr in video_result.frame_results
                ]

                detection_result = DetectionResult(
                    job_id=job.id,
                    authenticity_score=video_result.score,
                    confidence=video_result.confidence,
                    label=video_result.label,
                    layer_scores={"video_ensemble": video_result.score},
                    evidence_summary=video_result.evidence,
                    sentence_scores=frame_scores,
                    processing_ms=elapsed_ms,
                )
                db.add(detection_result)

                job.status       = JobStatus.COMPLETED
                job.completed_at = datetime.now(timezone.utc)
                await db.commit()

                log.info(
                    "video_detection_complete",
                    job_id=job_id,
                    score=video_result.score,
                    label=video_result.label,
                    frames=len(video_result.frame_results),
                    ms=elapsed_ms,
                )

                from .webhook_worker import dispatch_webhook
                dispatch_webhook.delay(job_id, "job.completed")

                return {
                    "job_id":        job_id,
                    "score":         video_result.score,
                    "label":         video_result.label,
                    "processing_ms": elapsed_ms,
                }

            except Exception as exc:
                log.error("video_detection_failed", job_id=job_id, error=str(exc))
                job.status        = JobStatus.FAILED
                job.error_message = str(exc)
                job.completed_at  = datetime.now(timezone.utc)
                await db.commit()

                from .webhook_worker import dispatch_webhook
                dispatch_webhook.delay(job_id, "job.failed")

                if not isinstance(exc, ValueError):
                    raise self.retry(exc=exc, countdown=30)

                return {"error": str(exc)}

    return asyncio.run(_run())
