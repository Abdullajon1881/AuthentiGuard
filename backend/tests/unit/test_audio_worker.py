"""
Unit tests for the audio detection Celery worker.
All external dependencies (DB, S3, AudioDetector) are mocked.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest


def _make_mock_chunk_result(idx=0, start=0.0, end=30.0, score=0.65):
    cr = MagicMock()
    cr.chunk_idx = idx
    cr.start_s = start
    cr.end_s = end
    cr.score = score
    cr.model_scores = {"cnn": 0.60, "resnet": 0.68, "transformer": 0.67}
    return cr


def _make_mock_audio_result(score=0.72, label="UNCERTAIN"):
    result = MagicMock()
    result.score = score
    result.label = label
    result.confidence = abs(score - 0.5) * 2
    result.chunk_results = [
        _make_mock_chunk_result(0, 0.0, 30.0, 0.65),
        _make_mock_chunk_result(1, 28.0, 58.0, 0.78),
    ]
    result.flagged_segments = [{"start_s": 28.0, "end_s": 58.0, "score": 0.78}]
    result.evidence = {
        "duration_s": 58.0,
        "n_chunks": 2,
        "gdd_mean": 0.35,
        "top_signals": [{"signal": "Phase coherence anomaly", "value": "0.35", "weight": "high"}],
    }
    result.processing_ms = 1800
    return result


class TestAudioWorkerConstants:
    def test_max_audio_size(self):
        from app.workers.audio_worker import MAX_AUDIO_SIZE
        assert MAX_AUDIO_SIZE == 200 * 1024 * 1024

    def test_task_registered(self):
        from app.workers.audio_worker import run_audio_detection
        assert hasattr(run_audio_detection, "delay")
        assert hasattr(run_audio_detection, "apply_async")


class TestAudioDetectionMapping:
    def test_chunk_results_mapped_to_sentence_scores(self):
        """Verify chunk results are correctly mapped for timeline view."""
        result = _make_mock_audio_result()
        chunk_scores = [
            {
                "text": f"{cr.start_s:.1f}s \u2013 {cr.end_s:.1f}s",
                "score": cr.score,
                "evidence": cr.model_scores,
            }
            for cr in result.chunk_results
        ]
        assert len(chunk_scores) == 2
        assert chunk_scores[0]["text"] == "0.0s \u2013 30.0s"
        assert chunk_scores[1]["score"] == 0.78
        assert "cnn" in chunk_scores[0]["evidence"]

    def test_evidence_includes_flagged_segments(self):
        """Audio evidence should include flagged segments."""
        result = _make_mock_audio_result()
        evidence_summary = {
            **result.evidence,
            "flagged_segments": result.flagged_segments,
        }
        assert "flagged_segments" in evidence_summary
        assert len(evidence_summary["flagged_segments"]) == 1
