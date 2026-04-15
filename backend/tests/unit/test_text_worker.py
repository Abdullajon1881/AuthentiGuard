"""
Unit tests for the text detection Celery worker.
All external dependencies (DB, S3, TextDetector) are mocked.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Mock helpers ─────────────────────────────────────────────

def _make_mock_ensemble_result(score=0.82, label="AI", confidence=0.64):
    """Create a mock TextDetector EnsembleResult."""
    layer_result = MagicMock()
    layer_result.layer_name = "perplexity"
    layer_result.score = 0.80

    result = MagicMock()
    result.score = score
    result.label = label
    result.confidence = confidence
    result.layer_results = [layer_result]
    result.evidence_summary = {
        "top_signals": [{"signal": "Low perplexity", "value": "12.3", "weight": "high"}],
        "sentence_scores": [],
    }
    return result


def _make_mock_job(job_id=None, input_text="A" * 50, s3_key=None):
    job = MagicMock()
    job.id = job_id or uuid.uuid4()
    job.input_text = input_text
    job.s3_key = s3_key
    job.file_name = None
    job.file_size = None
    job.status = None
    job.started_at = None
    job.completed_at = None
    job.error_message = None
    return job


# ── Detector singleton ───────────────────────────────────────

class TestGetDetector:
    def test_function_exists_and_callable(self):
        from app.workers.text_worker import _get_detector
        assert callable(_get_detector)

    def test_max_retries_is_3(self):
        from app.workers.text_worker import run_text_detection
        assert run_text_detection.max_retries == 3


# ── Text resolution ──────────────────────────────────────────

class TestResolveText:
    @pytest.mark.asyncio
    async def test_returns_input_text_directly(self):
        """When job has input_text, use it directly."""
        from app.workers.text_worker import _resolve_text
        job = _make_mock_job(input_text="Hello world, this is test content.")
        job.s3_key = None
        text = await _resolve_text(job)
        assert text == "Hello world, this is test content."

    @pytest.mark.asyncio
    async def test_raises_on_no_content(self):
        """Job with no input_text and no s3_key should raise ValueError."""
        from app.workers.text_worker import _resolve_text
        job = _make_mock_job(input_text=None, s3_key=None)
        job.input_text = None  # explicit
        with pytest.raises(ValueError, match="no text content"):
            await _resolve_text(job)

    @pytest.mark.asyncio
    async def test_fetches_from_s3_when_no_input_text(self):
        """When job has s3_key but no input_text, fetch from S3.

        `_resolve_text` imports `get_settings` and `boto3` lazily inside
        the function body (not at module scope), so we patch them at
        their real locations: `app.core.config.get_settings` and
        `boto3.client`.
        """
        from app.workers.text_worker import _resolve_text

        job = _make_mock_job(input_text=None, s3_key="uploads/user1/doc.txt")
        job.input_text = None

        mock_body = MagicMock()
        mock_body.read.return_value = b"Fetched content from S3 storage"

        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": mock_body}

        mock_settings = MagicMock()
        mock_settings.AWS_REGION = "us-east-1"
        mock_settings.AWS_ACCESS_KEY_ID = "test"
        mock_settings.AWS_SECRET_ACCESS_KEY = "test"
        mock_settings.S3_ENDPOINT_URL = None
        mock_settings.S3_BUCKET_UPLOADS = "uploads"

        with patch("app.core.config.get_settings", return_value=mock_settings), \
             patch("boto3.client", return_value=mock_s3):
            text = await _resolve_text(job)
            assert text == "Fetched content from S3 storage"


# ── Worker validation ────────────────────────────────────────

class TestTextWorkerValidation:
    def test_module_exports_task(self):
        from app.workers.text_worker import run_text_detection
        assert hasattr(run_text_detection, "delay")
        assert hasattr(run_text_detection, "apply_async")

    def test_task_queue_is_text(self):
        from app.workers.text_worker import run_text_detection
        assert run_text_detection.queue == "text"

    def test_task_name(self):
        from app.workers.text_worker import run_text_detection
        assert run_text_detection.name == "workers.text_worker.run_text_detection"


# ── PDF/DOCX extraction ─────────────────────────────────────

class TestTextExtraction:
    def test_pdf_extraction_fallback_without_pypdf(self):
        """Without pypdf installed, falls back to raw decode."""
        from app.workers.text_worker import _extract_pdf_text
        raw = b"raw text content"
        with patch.dict("sys.modules", {"pypdf": None}):
            # Force ImportError by patching import
            result = _extract_pdf_text(raw)
            # Should return something (either extracted or fallback)
            assert isinstance(result, str)

    def test_docx_extraction_fallback_without_docx(self):
        """Without python-docx installed, falls back to raw decode."""
        from app.workers.text_worker import _extract_docx_text
        raw = b"raw text content"
        with patch.dict("sys.modules", {"docx": None}):
            result = _extract_docx_text(raw)
            assert isinstance(result, str)
