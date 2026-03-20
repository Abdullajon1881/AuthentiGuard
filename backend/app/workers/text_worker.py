"""
Step 28: AI Detection Service — text/code worker task.

Celery task that:
  1. Loads text from DB (paste) or S3 (file upload)
  2. Runs the TextDetector ensemble
  3. Writes DetectionResult to Postgres
  4. Triggers the webhook worker on completion
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

import structlog
from celery import Task  # type: ignore
from sqlalchemy import select, update

from .celery_app import celery_app
from ..models.models import ContentType, DetectionJob, DetectionResult, JobStatus

log = structlog.get_logger(__name__)

# Module-level detector — loaded once per worker process, not per task
_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        from pathlib import Path
        # Import here to avoid circular imports and heavy model loading at import time
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../"))
        from ai.text_detector.ensemble.text_detector import TextDetector  # type: ignore

        _detector = TextDetector(
            transformer_checkpoint=Path("ai/text-detector/checkpoints/transformer/phase3"),
            adversarial_checkpoint=Path("ai/text-detector/checkpoints/adversarial/phase3"),
            meta_checkpoint=Path("ai/text-detector/checkpoints/meta"),
        )
        _detector.load_models()
        log.info("text_detector_loaded_in_worker")
    return _detector


@celery_app.task(
    bind=True,
    name="workers.text_worker.run_text_detection",
    queue="text",
    max_retries=3,
    default_retry_delay=10,
)
def run_text_detection(self: Task, job_id: str) -> dict:
    """
    Celery task: run full text detection ensemble for a job.

    Args:
        job_id: UUID string of the DetectionJob to process.

    Returns:
        dict with job_id, score, label, processing_ms.
    """
    from ..core.database import AsyncSessionLocal
    import asyncio

    async def _run() -> dict:
        async with AsyncSessionLocal() as db:
            # ── Fetch job ─────────────────────────────────────
            result = await db.execute(
                select(DetectionJob).where(DetectionJob.id == uuid.UUID(job_id))
            )
            job = result.scalar_one_or_none()

            if not job:
                log.error("job_not_found", job_id=job_id)
                return {"error": "Job not found"}

            # ── Mark as processing ────────────────────────────
            job.status     = JobStatus.PROCESSING
            job.started_at = datetime.now(timezone.utc)
            await db.commit()

            try:
                start_ms = int(time.time() * 1000)

                # ── Get text content ──────────────────────────
                text = await _resolve_text(job)

                if not text or len(text.strip()) < 20:
                    raise ValueError("Text content is too short to analyze (minimum 20 characters)")

                # ── Run detection ─────────────────────────────
                detector = _get_detector()
                ensemble_result = detector.analyze(text)

                elapsed_ms = int(time.time() * 1000) - start_ms

                # ── Persist result ────────────────────────────
                detection_result = DetectionResult(
                    job_id=job.id,
                    authenticity_score=ensemble_result.score,
                    confidence=ensemble_result.confidence,
                    label=ensemble_result.label,
                    layer_scores={
                        r.layer_name: r.score
                        for r in ensemble_result.layer_results
                    },
                    evidence_summary=ensemble_result.evidence_summary,
                    sentence_scores=ensemble_result.evidence_summary.get("sentence_scores", []),
                    processing_ms=elapsed_ms,
                )
                db.add(detection_result)

                job.status       = JobStatus.COMPLETED
                job.completed_at = datetime.now(timezone.utc)
                await db.commit()

                log.info(
                    "text_detection_complete",
                    job_id=job_id,
                    score=ensemble_result.score,
                    label=ensemble_result.label,
                    ms=elapsed_ms,
                )

                # ── Trigger webhook ───────────────────────────
                from .webhook_worker import dispatch_webhook
                dispatch_webhook.delay(job_id, "job.completed")

                return {
                    "job_id":       job_id,
                    "score":        ensemble_result.score,
                    "label":        ensemble_result.label,
                    "processing_ms": elapsed_ms,
                }

            except Exception as exc:
                log.error("text_detection_failed", job_id=job_id, error=str(exc))
                job.status        = JobStatus.FAILED
                job.error_message = str(exc)
                job.completed_at  = datetime.now(timezone.utc)
                await db.commit()

                # Trigger failure webhook
                from .webhook_worker import dispatch_webhook
                dispatch_webhook.delay(job_id, "job.failed")

                # Retry transient errors (not validation errors)
                if "too short" not in str(exc):
                    raise self.retry(exc=exc, countdown=10)

                return {"error": str(exc)}

    return asyncio.run(_run())


async def _resolve_text(job: DetectionJob) -> str:
    """Get text content from job — either direct paste or S3 file."""
    if job.input_text:
        return job.input_text

    if job.s3_key:
        import boto3
        from ..core.config import get_settings
        settings = get_settings()

        kwargs = {
            "region_name":          settings.AWS_REGION,
            "aws_access_key_id":    settings.AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": settings.AWS_SECRET_ACCESS_KEY,
        }
        if settings.S3_ENDPOINT_URL:
            kwargs["endpoint_url"] = settings.S3_ENDPOINT_URL

        s3 = boto3.client("s3", **kwargs)
        obj = s3.get_object(Bucket=settings.S3_BUCKET_UPLOADS, Key=job.s3_key)
        data = obj["Body"].read()

        # Handle PDF and DOCX extraction
        if job.s3_key.endswith(".pdf"):
            return _extract_pdf_text(data)
        elif job.s3_key.endswith(".docx"):
            return _extract_docx_text(data)
        else:
            return data.decode("utf-8", errors="replace")

    raise ValueError("Job has no text content or S3 key")


def _extract_pdf_text(data: bytes) -> str:
    try:
        import pypdf  # type: ignore
        import io
        reader = pypdf.PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        return data.decode("utf-8", errors="replace")


def _extract_docx_text(data: bytes) -> str:
    try:
        import docx  # type: ignore
        import io
        doc = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    except ImportError:
        return data.decode("utf-8", errors="replace")
