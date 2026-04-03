"""
Audio detection worker — Celery task using BaseDetectionWorker.

Downloads audio bytes from S3, runs the AudioDetector ensemble,
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

MAX_AUDIO_SIZE = 200 * 1024 * 1024  # 200 MB


def _get_detector():
    global _detector
    if _detector is None:
        try:
            from ai.audio_detector.audio_detector import AudioDetector  # type: ignore
        except ImportError:
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


class AudioDetectionWorker(BaseDetectionWorker):
    content_type = "audio"

    def get_detector(self) -> Any:
        return _get_detector()

    async def get_input(self, job: DetectionJob) -> bytes:
        if not job.s3_key:
            raise ValueError("Audio job has no S3 key — audio must be uploaded as files")
        if job.file_size and job.file_size > MAX_AUDIO_SIZE:
            raise ValueError(
                f"Audio file too large ({job.file_size / 1024 / 1024:.1f} MB, "
                f"max {MAX_AUDIO_SIZE / 1024 / 1024:.0f} MB)"
            )
        from ..services.s3_service import fetch_from_s3
        return await fetch_from_s3(job.s3_key)

    def run_detection(self, detector: Any, input_data: bytes, job: DetectionJob) -> Any:
        return detector.analyze(input_data, job.file_name or "audio.wav")

    def build_result(
        self,
        job: DetectionJob,
        detection_output: Any,
        elapsed_ms: int,
    ) -> DetectionResult:
        chunk_scores = [
            {
                "text": f"{cr.start_s:.1f}s \u2013 {cr.end_s:.1f}s",
                "score": cr.score,
                "evidence": cr.model_scores,
            }
            for cr in detection_output.chunk_results
        ]

        return DetectionResult(
            job_id=job.id,
            authenticity_score=detection_output.score,
            confidence=detection_output.confidence,
            label=detection_output.label,
            layer_scores={"audio_ensemble": detection_output.score},
            evidence_summary={
                **detection_output.evidence,
                "flagged_segments": detection_output.flagged_segments,
            },
            sentence_scores=chunk_scores,
            processing_ms=elapsed_ms,
        )


_worker = AudioDetectionWorker()


@celery_app.task(
    bind=True,
    name="workers.audio_worker.run_audio_detection",
    queue="audio",
    max_retries=3,
    default_retry_delay=15,
)
def run_audio_detection(self: Task, job_id: str) -> dict:
    """Celery task: run full audio deepfake detection for a job."""
    try:
        return run_async(_worker.execute(job_id))
    except Exception as exc:
        if isinstance(exc, ValueError):
            return {"error": str(exc)}
        raise self.retry(exc=exc, countdown=15)
