"""
Video detection worker — Celery task using BaseDetectionWorker.

Downloads video bytes from S3, runs the VideoDetector ensemble,
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

MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500 MB


def _get_detector():
    global _detector
    if _detector is None:
        try:
            from ai.video_detector.video_detector import VideoDetector  # type: ignore
        except ImportError:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../"))
            from ai.video_detector.video_detector import VideoDetector  # type: ignore

        _detector = VideoDetector(
            checkpoint_dir=Path("ai/video_detector/checkpoints/phase3"),
        )
        _detector.load_models()
        log.info("video_detector_loaded_in_worker")
    return _detector


class VideoDetectionWorker(BaseDetectionWorker):
    content_type = "video"

    def get_detector(self) -> Any:
        return _get_detector()

    async def get_input(self, job: DetectionJob) -> bytes:
        if not job.s3_key:
            raise ValueError("Video job has no S3 key — videos must be uploaded as files")
        if job.file_size and job.file_size > MAX_VIDEO_SIZE:
            raise ValueError(
                f"Video file too large ({job.file_size / 1024 / 1024:.1f} MB, "
                f"max {MAX_VIDEO_SIZE / 1024 / 1024:.0f} MB)"
            )
        from ..services.s3_service import fetch_from_s3
        return await fetch_from_s3(job.s3_key)

    def run_detection(self, detector: Any, input_data: bytes, job: DetectionJob) -> Any:
        return detector.analyze(input_data, job.file_name or "video.mp4")

    def build_result(
        self,
        job: DetectionJob,
        detection_output: Any,
        elapsed_ms: int,
    ) -> DetectionResult:
        frame_scores = [
            {
                "text": f"{fr.timestamp_s:.1f}s",
                "score": fr.frame_score,
                "evidence": fr.model_scores,
            }
            for fr in detection_output.frame_results
        ]

        return DetectionResult(
            job_id=job.id,
            authenticity_score=detection_output.score,
            confidence=detection_output.confidence,
            label=detection_output.label,
            layer_scores={"video_ensemble": detection_output.score},
            evidence_summary=detection_output.evidence,
            sentence_scores=frame_scores,
            processing_ms=elapsed_ms,
        )


_worker = VideoDetectionWorker()


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
    """Celery task: run full video deepfake detection for a job."""
    try:
        return run_async(_worker.execute(job_id))
    except Exception as exc:
        if isinstance(exc, ValueError):
            return {"error": str(exc)}
        raise self.retry(exc=exc, countdown=30)
