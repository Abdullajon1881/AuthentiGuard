"""
Celery worker task for audio deepfake detection.
Mirrors the text_worker pattern — singleton model loading per process.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import structlog
from celery import Task  # type: ignore

log = structlog.get_logger(__name__)

_audio_detector = None


def _get_audio_detector():
    global _audio_detector
    if _audio_detector is None:
        from ..audio_detector import AudioDetector
        _audio_detector = AudioDetector(
            checkpoint_dir=Path("ai/audio_detector/checkpoints/phase3"),
            calibration_path=Path("ai/audio_detector/checkpoints/calibration.pkl"),
        )
        _audio_detector.load_models()
    return _audio_detector


def run_audio_detection_task(job_id: str) -> dict:
    """
    Main logic called by the Celery task.
    Separated from the task decorator to allow unit testing.
    """
    import asyncio

    async def _run() -> dict:
        from ...backend.app.core.database import AsyncSessionLocal
        from ...backend.app.models.models import (
            DetectionJob, DetectionResult, JobStatus
        )
        from sqlalchemy import select
        import boto3
        from ...backend.app.core.config import get_settings

        settings = get_settings()

        async with AsyncSessionLocal() as db:
            res = await db.execute(
                select(DetectionJob).where(DetectionJob.id == uuid.UUID(job_id))
            )
            job = res.scalar_one_or_none()
            if not job:
                return {"error": "Job not found"}

            job.status     = JobStatus.PROCESSING
            job.started_at = datetime.now(timezone.utc)
            await db.commit()

            try:
                # Download file from S3
                s3_kwargs = {
                    "region_name":           settings.AWS_REGION,
                    "aws_access_key_id":     settings.AWS_ACCESS_KEY_ID,
                    "aws_secret_access_key": settings.AWS_SECRET_ACCESS_KEY,
                }
                if settings.S3_ENDPOINT_URL:
                    s3_kwargs["endpoint_url"] = settings.S3_ENDPOINT_URL

                s3  = boto3.client("s3", **s3_kwargs)
                obj = s3.get_object(Bucket=settings.S3_BUCKET_UPLOADS, Key=job.s3_key)
                data = obj["Body"].read()

                detector = _get_audio_detector()
                result   = detector.analyze(data, job.file_name or "audio.wav")

                detection = DetectionResult(
                    job_id=job.id,
                    authenticity_score=result.score,
                    confidence=result.confidence,
                    label=result.label,
                    layer_scores={"audio_ensemble": result.score},
                    evidence_summary={
                        **result.evidence,
                        "flagged_segments": result.flagged_segments,
                        "top_signals": result.evidence.get("signals", []),
                    },
                    sentence_scores=[
                        {
                            "text":    f"{r.start_s:.1f}s – {r.end_s:.1f}s",
                            "score":   r.score,
                            "evidence": r.model_scores,
                        }
                        for r in result.chunk_results
                    ],
                    processing_ms=result.processing_ms,
                )
                db.add(detection)
                job.status       = JobStatus.COMPLETED
                job.completed_at = datetime.now(timezone.utc)
                await db.commit()

                return {"job_id": job_id, "score": result.score, "label": result.label}

            except Exception as exc:
                log.error("audio_detection_failed", job_id=job_id, error=str(exc))
                job.status        = JobStatus.FAILED
                job.error_message = str(exc)
                job.completed_at  = datetime.now(timezone.utc)
                await db.commit()
                return {"error": str(exc)}

    return asyncio.run(_run())
