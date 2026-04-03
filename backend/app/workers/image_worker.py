"""
Image detection worker — Celery task using BaseDetectionWorker.

Downloads image bytes from S3, runs the ImageDetector ensemble,
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

_detector = None

MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50 MB


def _get_detector():
    global _detector
    if _detector is None:
        try:
            from ai.image_detector.image_detector import ImageDetector  # type: ignore
        except ImportError:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../"))
            from ai.image_detector.image_detector import ImageDetector  # type: ignore

        _detector = ImageDetector(
            checkpoint_dir=Path("ai/image_detector/checkpoints/phase3"),
        )
        _detector.load_models()
        log.info("image_detector_loaded_in_worker")
    return _detector


class ImageDetectionWorker(BaseDetectionWorker):
    content_type = "image"

    def get_detector(self) -> Any:
        return _get_detector()

    async def get_input(self, job: DetectionJob) -> bytes:
        if not job.s3_key:
            raise ValueError("Image job has no S3 key — images must be uploaded as files")
        if job.file_size and job.file_size > MAX_IMAGE_SIZE:
            raise ValueError(
                f"Image file too large ({job.file_size / 1024 / 1024:.1f} MB, "
                f"max {MAX_IMAGE_SIZE / 1024 / 1024:.0f} MB)"
            )
        from ..services.s3_service import fetch_from_s3
        return await fetch_from_s3(job.s3_key)

    def run_detection(self, detector: Any, input_data: bytes, job: DetectionJob) -> Any:
        return detector.analyze(input_data, job.file_name or "image.jpg")

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
            layer_scores=detection_output.model_scores,
            evidence_summary=detection_output.evidence,
            sentence_scores=[],
            processing_ms=elapsed_ms,
        )


_worker = ImageDetectionWorker()


@celery_app.task(
    bind=True,
    name="workers.image_worker.run_image_detection",
    queue="image",
    max_retries=3,
    default_retry_delay=10,
)
def run_image_detection(self: Task, job_id: str) -> dict:
    """Celery task: run full image detection ensemble for a job."""
    try:
        return run_async(_worker.execute(job_id))
    except Exception as exc:
        if isinstance(exc, ValueError):
            return {"error": str(exc)}
        raise self.retry(exc=exc, countdown=10)
