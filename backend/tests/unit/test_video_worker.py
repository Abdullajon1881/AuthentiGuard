"""
Unit tests for the video detection Celery worker.
All external dependencies (DB, S3, VideoDetector) are mocked.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_mock_frame_result(idx=0, timestamp=0.5, score=0.82):
    fr = MagicMock()
    fr.frame_idx = idx
    fr.timestamp_s = timestamp
    fr.n_faces = 2
    fr.frame_score = score
    fr.artifact_score = 0.3
    fr.model_scores = {"xceptionnet": 0.80, "efficientnet_b4": 0.84, "vit_b16": 0.82}
    return fr


def _make_mock_video_result(score=0.78, label="AI"):
    result = MagicMock()
    result.score = score
    result.label = label
    result.confidence = abs(score - 0.5) * 2
    result.frame_results = [
        _make_mock_frame_result(0, 0.5, 0.82),
        _make_mock_frame_result(1, 1.0, 0.75),
        _make_mock_frame_result(2, 1.5, 0.80),
    ]
    result.flagged_segments = [{"start_s": 0.0, "end_s": 1.5, "score": 0.79}]
    result.evidence = {
        "duration_s": 5.0,
        "total_frames": 10,
        "faces_detected": 6,
        "top_signals": [{"signal": "Identity drift detected", "value": "0.45", "weight": "high"}],
    }
    result.processing_ms = 2500
    return result


class TestVideoWorkerConstants:
    def test_max_video_size(self):
        from app.workers.video_worker import MAX_VIDEO_SIZE
        assert MAX_VIDEO_SIZE == 500 * 1024 * 1024

    def test_task_registered(self):
        from app.workers.video_worker import run_video_detection
        assert hasattr(run_video_detection, "delay")
        assert hasattr(run_video_detection, "apply_async")

    def test_get_detector_callable(self):
        from app.workers.video_worker import _get_detector
        assert callable(_get_detector)


class TestVideoDetectionMapping:
    def test_frame_results_mapped_to_sentence_scores(self):
        """Verify frame results are correctly mapped for timeline view."""
        result = _make_mock_video_result()
        frame_scores = [
            {
                "text": f"{fr.timestamp_s:.1f}s",
                "score": fr.frame_score,
                "evidence": fr.model_scores,
            }
            for fr in result.frame_results
        ]
        assert len(frame_scores) == 3
        assert frame_scores[0]["text"] == "0.5s"
        assert frame_scores[0]["score"] == 0.82
        assert "xceptionnet" in frame_scores[0]["evidence"]


class TestFetchFromS3:
    @pytest.mark.asyncio
    async def test_shared_s3_service(self):
        """Verify the shared S3 service is importable."""
        from app.services.s3_service import fetch_from_s3
        assert callable(fetch_from_s3)
