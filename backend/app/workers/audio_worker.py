"""
Audio detection worker — Celery task for audio deepfake detection.

Celery task that:
  1. Downloads audio bytes from S3
  2. Runs the AudioDetector (chunking → feature extraction →
     CNN/ResNet/Wav2Vec2 ensemble → calibration → aggregation)
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

MAX_AUDIO_SIZE = 200 * 1024 * 1024  # 200 MB


def _get_detector():
    global _detector
    if _detector is None:
        from pathlib import Path
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../"))
        from ai.audio_detector.audio_detector import AudioDetector  # type: ignore

        _detector = AudioDetector(
            checkpoint_dir=Path("ai/audio_detector/checkpoints/phase3"),
            calibration_path=Path("ai/audio_detector/checkpoints/calibration.pkl"),
        )
        _detector.load_models()
        log.info("audio_detector_loaded_in_worker")
    return _detector


@celery_app.task(
    bind=True,
    name="workers.audio_worker.run_audio_detection",
    queue="audio",
    max_retries=3,
    default_retry_delay=15,
)
def run_audio_detection(self: Task, job_id: str) -> dict:
    """
    Celery task: run full audio deepfake detection for a job.

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
                    raise ValueError("Audio job has no S3 key — audio must be uploaded as files")

                if job.file_size and job.file_size > MAX_AUDIO_SIZE:
                    raise ValueError(
                        f"Audio file too large ({job.file_size / 1024 / 1024:.1f} MB, max {MAX_AUDIO_SIZE / 1024 / 1024:.0f} MB)"
                    )

                start_ms = int(time.time() * 1000)
                from ..services.s3_service import fetch_from_s3
                data = await fetch_from_s3(job.s3_key)

                detector = _get_detector()
                audio_result = detector.analyze(data, job.file_name or "audio.wav")

                elapsed_ms = int(time.time() * 1000) - start_ms

                # Map chunk results to sentence_scores for timeline view
                chunk_scores = [
                    {
                        "text": f"{cr.start_s:.1f}s \u2013 {cr.end_s:.1f}s",
                        "score": cr.score,
                        "evidence": cr.model_scores,
                    }
                    for cr in audio_result.chunk_results
                ]

                detection_result = DetectionResult(
                    job_id=job.id,
                    authenticity_score=audio_result.score,
                    confidence=audio_result.confidence,
                    label=audio_result.label,
                    layer_scores={"audio_ensemble": audio_result.score},
                    evidence_summary={
                        **audio_result.evidence,
                        "flagged_segments": audio_result.flagged_segments,
                    },
                    sentence_scores=chunk_scores,
                    processing_ms=elapsed_ms,
                )
                db.add(detection_result)

                job.status       = JobStatus.COMPLETED
                job.completed_at = datetime.now(timezone.utc)
                await db.commit()

                log.info(
                    "audio_detection_complete",
                    job_id=job_id,
                    score=audio_result.score,
                    label=audio_result.label,
                    chunks=len(audio_result.chunk_results),
                    ms=elapsed_ms,
                )

                from .webhook_worker import dispatch_webhook
                dispatch_webhook.delay(job_id, "job.completed")

                return {
                    "job_id":        job_id,
                    "score":         audio_result.score,
                    "label":         audio_result.label,
                    "processing_ms": elapsed_ms,
                }

            except Exception as exc:
                log.error("audio_detection_failed", job_id=job_id, error=str(exc))
                job.status        = JobStatus.FAILED
                job.error_message = str(exc)
                job.completed_at  = datetime.now(timezone.utc)
                await db.commit()

                from .webhook_worker import dispatch_webhook
                dispatch_webhook.delay(job_id, "job.failed")

                if not isinstance(exc, ValueError):
                    raise self.retry(exc=exc, countdown=15)

                return {"error": str(exc)}

    return asyncio.run(_run())
