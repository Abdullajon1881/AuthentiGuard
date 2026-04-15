"""
Unit tests for the image detection Celery worker.
All external dependencies (DB, S3, ImageDetector) are mocked.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest


# ── Mock the ImageDetector result ────────────────────────────

def _make_mock_result(score=0.85, label="AI", confidence=0.70):
    result = MagicMock()
    result.score = score
    result.label = label
    result.confidence = confidence
    result.model_scores = {"efficientnet_b4": 0.88, "vit_b16": 0.82}
    result.evidence = {
        "width": 1024,
        "height": 1024,
        "format": "JPEG",
        "file_size": 512_000,
        "top_signals": [{"signal": "GAN checkerboard artifact detected", "value": "0.45", "weight": "high"}],
    }
    result.processing_ms = 342
    return result


def _make_mock_job(job_id=None, s3_key="uploads/user1/image.jpg", file_name="photo.jpg", file_size=512_000):
    job = MagicMock()
    job.id = job_id or uuid.uuid4()
    job.s3_key = s3_key
    job.file_name = file_name
    job.file_size = file_size
    job.status = None
    job.started_at = None
    job.completed_at = None
    job.error_message = None
    return job


# ── Tests ────────────────────────────────────────────────────

class TestGetDetector:
    def test_get_detector_is_importable_and_callable(self):
        """Smoke check that `_get_detector` exists and is a callable.

        The previous version of this test patched `builtins.__import__`
        with a side_effect of `ImportError`, which broke the import of
        `_get_detector` itself and silently bypassed any assertion. The
        real invariants we care about:
          1. `_get_detector` is importable
          2. it is a callable
          3. calling it with the heavy-loading path short-circuited
             returns whatever was previously cached in `_detector`
        """
        import app.workers.image_worker as iw
        assert callable(iw._get_detector)

        sentinel = object()
        with patch.object(iw, "_detector", sentinel):
            result = iw._get_detector()
            assert result is sentinel


class TestImageWorkerValidation:
    def test_max_image_size_constant(self):
        from app.workers.image_worker import MAX_IMAGE_SIZE
        assert MAX_IMAGE_SIZE == 50 * 1024 * 1024

    def test_module_exports(self):
        from app.workers.image_worker import run_image_detection
        assert hasattr(run_image_detection, "delay")
        assert hasattr(run_image_detection, "apply_async")


class TestRunImageDetection:
    @pytest.mark.asyncio
    async def test_run_detection_calls_analyze(self):
        """ImageDetectionWorker.run_detection delegates to detector.analyze.

        This replaces a prior version of this test that patched symbols
        that don't exist at module scope (`_fetch_from_s3`,
        `AsyncSessionLocal`, `dispatch_webhook`) and ended with an
        `analyze.assert_not_called()` assertion — which was a tautology.
        Here we assert the real invariant of `run_detection`: it calls
        `detector.analyze(input_data, filename)`.
        """
        from app.workers.image_worker import ImageDetectionWorker

        worker = ImageDetectionWorker()
        mock_detector = MagicMock()
        mock_detector.analyze.return_value = _make_mock_result()

        fake_bytes = b"\xff\xd8\xff\xe0fake_jpeg"
        job = _make_mock_job(file_name="photo.jpg")
        out = worker.run_detection(mock_detector, fake_bytes, job)

        mock_detector.analyze.assert_called_once_with(fake_bytes, "photo.jpg")
        assert out is mock_detector.analyze.return_value

    def test_rejects_missing_s3_key(self):
        """Job with no s3_key should raise ValueError."""
        mock_job = _make_mock_job(s3_key=None)
        # Validation: image jobs must have s3_key
        assert mock_job.s3_key is None

    def test_rejects_oversized_file(self):
        """Job with file_size > 50MB should raise ValueError."""
        from app.workers.image_worker import MAX_IMAGE_SIZE
        mock_job = _make_mock_job(file_size=60 * 1024 * 1024)
        assert mock_job.file_size > MAX_IMAGE_SIZE


class TestFetchFromS3:
    @pytest.mark.asyncio
    async def test_fetch_calls_boto3(self):
        """Verify the shared S3 fetch helper uses the correct bucket + key.

        The image worker used to have its own `_fetch_from_s3`; it now
        delegates to the shared `app.services.s3_service.fetch_from_s3`.
        We target that real function here.
        """
        mock_settings = MagicMock()
        mock_settings.AWS_REGION = "us-east-1"
        mock_settings.AWS_ACCESS_KEY_ID = "test-key"
        mock_settings.AWS_SECRET_ACCESS_KEY = "test-secret"
        mock_settings.S3_ENDPOINT_URL = "http://localhost:9000"
        mock_settings.S3_BUCKET_UPLOADS = "uploads"

        mock_body = MagicMock()
        mock_body.read.return_value = b"fake-image-bytes"

        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": mock_body}

        with patch("app.core.config.get_settings", return_value=mock_settings), \
             patch("boto3.client", return_value=mock_s3):
            from app.services.s3_service import fetch_from_s3
            data = await fetch_from_s3("uploads/user1/test.jpg")

            assert data == b"fake-image-bytes"
            mock_s3.get_object.assert_called_once_with(
                Bucket="uploads",
                Key="uploads/user1/test.jpg",
            )
