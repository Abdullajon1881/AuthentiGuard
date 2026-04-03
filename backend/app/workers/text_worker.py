"""
Text/code detection worker — Celery task using BaseDetectionWorker.

Loads text from DB (paste) or S3 (file upload), runs the TextDetector ensemble,
writes DetectionResult, and triggers webhook on completion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from celery import Task  # type: ignore

from .base_worker import BaseDetectionWorker, run_async
from .celery_app import celery_app
from ..models.models import DetectionJob, DetectionResult

log = structlog.get_logger(__name__)

# Module-level detector — loaded once per worker process, not per task
_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        try:
            from ai.text_detector.ensemble.text_detector import TextDetector  # type: ignore
        except ImportError:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../"))
            from ai.text_detector.ensemble.text_detector import TextDetector  # type: ignore

        _detector = TextDetector(
            transformer_checkpoint=Path("ai/text_detector/checkpoints/transformer/phase3"),
            adversarial_checkpoint=Path("ai/text_detector/checkpoints/adversarial/phase3"),
            meta_checkpoint=Path("ai/text_detector/checkpoints/meta"),
        )
        _detector.load_models()
        log.info("text_detector_loaded_in_worker")
    return _detector


class TextDetectionWorker(BaseDetectionWorker):
    content_type = "text"

    def get_detector(self) -> Any:
        return _get_detector()

    async def get_input(self, job: DetectionJob) -> str:
        text = await _resolve_text(job)
        if not text or len(text.strip()) < 20:
            raise ValueError("Text content is too short to analyze (minimum 20 characters)")
        return text

    def run_detection(self, detector: Any, input_data: str, job: DetectionJob) -> Any:
        return detector.analyze(input_data)

    def build_result(
        self,
        job: DetectionJob,
        detection_output: Any,
        elapsed_ms: int,
    ) -> DetectionResult:
        return DetectionResult(
            job_id=job.id,
            authenticity_score=detection_output.score,
            confidence=detection_output.confidence,
            label=detection_output.label,
            layer_scores={
                r.layer_name: r.score
                for r in detection_output.layer_results
            },
            evidence_summary=detection_output.evidence_summary,
            sentence_scores=detection_output.evidence_summary.get("sentence_scores", []),
            processing_ms=elapsed_ms,
        )


_worker = TextDetectionWorker()


@celery_app.task(
    bind=True,
    name="workers.text_worker.run_text_detection",
    queue="text",
    max_retries=3,
    default_retry_delay=10,
)
def run_text_detection(self: Task, job_id: str) -> dict:
    """Celery task: run full text detection ensemble for a job."""
    try:
        return run_async(_worker.execute(job_id))
    except Exception as exc:
        if isinstance(exc, ValueError):
            return {"error": str(exc)}
        raise self.retry(exc=exc, countdown=10)


# ── Text resolution helpers ──────────────────────────────────

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
