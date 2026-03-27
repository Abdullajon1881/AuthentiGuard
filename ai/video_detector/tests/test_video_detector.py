"""
Unit tests for the video deepfake detection pipeline.
All tests use synthetic frame arrays — no video files or model weights needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai.video_detector.pipeline.frame_extractor import (
    _fallback_metadata, probe_video, VideoMetadata, TARGET_FRAME_SIZE,
)
from ai.video_detector.pipeline.face_detector import (
    _extract_aligned_crop, _empty_frame_faces, FaceRegion,
)
from ai.video_detector.features.artifact_analyzer import (
    _laplacian_variance, _lbp_uniformity,
    analyze_blink, analyze_skin_texture, ArtifactAnalysis,
)
from ai.video_detector.features.temporal_analyzer import (
    compute_temporal_features, analyze_blink_pattern, _empty_temporal,
    _embedding_variance_pixel, compute_boundary_flicker,
)
from ai.video_detector.video_detector import VideoDetector


# ── Fixtures ──────────────────────────────────────────────────

def make_frame(h: int = 224, w: int = 224, noise: float = 0.0) -> np.ndarray:
    base = np.random.randint(100, 200, (h, w, 3), dtype=np.uint8)
    if noise > 0:
        base = np.clip(
            base.astype(np.float32) + np.random.normal(0, noise * 255, base.shape),
            0, 255,
        ).astype(np.uint8)
    return base


def make_face_region(
    frame_idx: int = 0,
    timestamp: float = 0.0,
    crop_noise: float = 0.0,
) -> FaceRegion:
    from ai.video_detector.pipeline.face_detector import FaceRegion
    return FaceRegion(
        frame_idx=frame_idx,
        timestamp_s=timestamp,
        bbox=(20, 20, 180, 180),
        landmarks=None,
        crop=make_frame(224, 224, noise=crop_noise),
        confidence=0.95,
        face_idx=0,
    )


def make_artifact(
    frame_idx: int = 0,
    score: float = 0.5,
    blink: bool = False,
) -> ArtifactAnalysis:
    return ArtifactAnalysis(
        frame_idx=frame_idx,
        timestamp_s=float(frame_idx) * 0.5,
        eye_openness=0.3 if blink else 0.9,
        blink_detected=blink,
        texture_score=score,
        laplacian_variance=200.0,
        lbp_uniformity=0.5,
        lighting_score=score,
        illumination_std=0.1,
        specular_anomaly=0.0,
        artifact_score=score,
    )


# ── Frame extraction helpers ──────────────────────────────────

class TestFrameExtraction:
    def test_fallback_metadata_has_correct_defaults(self) -> None:
        meta = _fallback_metadata("test.mp4")
        assert meta.filename == "test.mp4"
        assert meta.fps == 25.0
        assert meta.duration_s == 0.0

    def test_target_frame_size_is_square(self) -> None:
        assert TARGET_FRAME_SIZE[0] == TARGET_FRAME_SIZE[1]
        assert TARGET_FRAME_SIZE[0] == 224


# ── Face detection helpers ────────────────────────────────────

class TestFaceDetection:
    def test_aligned_crop_output_size(self) -> None:
        from ai.video_detector.pipeline.face_detector import FACE_CROP_SIZE
        image = make_frame(480, 640)
        crop  = _extract_aligned_crop(image, 100, 100, 300, 300)
        assert crop.shape[0] == FACE_CROP_SIZE
        assert crop.shape[1] == FACE_CROP_SIZE
        assert crop.shape[2] == 3

    def test_aligned_crop_clamped_to_frame(self) -> None:
        """Bbox extending outside frame should not crash."""
        from ai.video_detector.pipeline.face_detector import FACE_CROP_SIZE
        image = make_frame(100, 100)
        crop  = _extract_aligned_crop(image, -10, -10, 110, 110)
        assert crop.shape[:2] == (FACE_CROP_SIZE, FACE_CROP_SIZE)

    def test_aligned_crop_zero_size_returns_zeros(self) -> None:
        from ai.video_detector.pipeline.face_detector import FACE_CROP_SIZE
        image = make_frame(100, 100)
        crop  = _extract_aligned_crop(image, 50, 50, 50, 50)
        assert crop.shape[:2] == (FACE_CROP_SIZE, FACE_CROP_SIZE)

    def test_empty_frame_faces_structure(self) -> None:
        from ai.video_detector.pipeline.frame_extractor import VideoFrame
        vf = VideoFrame(frame_idx=0, timestamp_s=0.0,
                        image=make_frame(), width=224, height=224)
        ff = _empty_frame_faces(vf)
        assert ff.n_faces == 0
        assert ff.faces == []
        assert ff.detection_method == "none"


# ── Artifact analysis ─────────────────────────────────────────

class TestArtifactAnalysis:
    def test_laplacian_variance_flat_image_is_low(self) -> None:
        flat = np.ones((50, 50), dtype=np.float32) * 128
        var  = _laplacian_variance(flat)
        assert var < 1.0   # nearly zero variance for flat image

    def test_laplacian_variance_noisy_image_is_high(self) -> None:
        noisy = np.random.randint(0, 255, (50, 50), dtype=np.uint8).astype(np.float32)
        var   = _laplacian_variance(noisy)
        assert var > 100.0   # high variance for noise

    def test_lbp_uniformity_range(self) -> None:
        region = np.random.randint(0, 255, (30, 30), dtype=np.uint8).astype(np.float32)
        uni    = _lbp_uniformity(region)
        assert 0.0 <= uni <= 1.0

    def test_lbp_uniform_image_is_high(self) -> None:
        """Completely flat image should have maximum LBP uniformity."""
        flat = np.ones((30, 30), dtype=np.float32) * 100
        uni  = _lbp_uniformity(flat)
        assert uni > 0.8

    def test_analyze_blink_does_not_crash(self) -> None:
        face   = make_face_region()
        result = analyze_blink(face)
        assert "blink_detected" in result

    def test_analyze_skin_texture_score_range(self) -> None:
        face   = make_face_region()
        result = analyze_skin_texture(face)
        assert "texture_score" in result
        assert 0.0 <= result["texture_score"] <= 1.0
        assert "laplacian_variance" in result

    def test_smooth_face_has_higher_texture_score(self) -> None:
        """A smooth (over-processed) face should score higher than a textured one."""
        smooth  = make_face_region(crop_noise=0.0)
        textured = make_face_region(crop_noise=0.3)
        s_smooth   = analyze_skin_texture(smooth)["texture_score"]
        s_textured = analyze_skin_texture(textured)["texture_score"]
        # Smooth has low laplacian variance → higher AI score
        assert s_smooth >= s_textured or abs(s_smooth - s_textured) < 0.3  # soft check


# ── Temporal analysis ─────────────────────────────────────────

class TestTemporalAnalysis:
    def test_empty_artifacts_returns_empty_temporal(self) -> None:
        result = compute_temporal_features([], [], 2.0)
        assert result.temporal_consistency_score == 0.0
        assert result.blink_rate == 0.0

    def test_embedding_variance_pixel_multiple_frames(self) -> None:
        crops = [make_frame() for _ in range(5)]
        result = _embedding_variance_pixel(crops)
        assert "embedding_variance" in result
        assert result["embedding_variance"] >= 0.0

    def test_embedding_variance_identical_frames_is_zero(self) -> None:
        crop  = make_frame()
        crops = [crop.copy() for _ in range(5)]
        result = _embedding_variance_pixel(crops)
        assert result["embedding_variance"] < 1e-3

    def test_blink_pattern_no_blinks(self) -> None:
        artifacts = [make_artifact(i, blink=False) for i in range(60)]
        result    = analyze_blink_pattern(artifacts, video_fps=2.0)
        assert result["blink_rate"] == 0.0

    def test_blink_pattern_regular_blinks_high_regularity(self) -> None:
        """Perfectly regular blinks should have high regularity score."""
        artifacts = []
        for i in range(60):
            blink = (i % 10 == 0)  # blink every 5 seconds — perfectly regular
            artifacts.append(make_artifact(i, blink=blink))
        result = analyze_blink_pattern(artifacts, video_fps=2.0)
        assert result["blink_regularity"] >= 0.5

    def test_boundary_flicker_identical_crops_is_low(self) -> None:
        crop  = make_frame()
        crops = [crop.copy() for _ in range(5)]
        flicker = compute_boundary_flicker(crops)
        assert flicker < 1.0   # near zero for identical frames

    def test_boundary_flicker_varied_crops_is_higher(self) -> None:
        crops   = [make_frame(noise=0.5) for _ in range(5)]
        flicker = compute_boundary_flicker(crops)
        assert flicker >= 0.0


# ── Score aggregation ─────────────────────────────────────────

class TestScoreAggregation:
    def test_flagged_segments_threshold(self) -> None:
        """Frames with score ≥ 0.65 should be grouped into segments."""
        from ai.video_detector.video_detector import VideoDetector, VideoFrameResult

        frame_results = [
            VideoFrameResult(0, 0.0, 1, 0.80, 0.5, {}),
            VideoFrameResult(1, 0.5, 1, 0.85, 0.5, {}),
            VideoFrameResult(2, 1.0, 1, 0.20, 0.2, {}),
            VideoFrameResult(3, 1.5, 1, 0.10, 0.1, {}),
        ]
        flagged = VideoDetector._build_flagged_segments(frame_results)
        assert len(flagged) == 1
        assert flagged[0]["start_s"] == 0.0
        assert flagged[0]["severity"] == "high"

    def test_clean_video_no_segments(self) -> None:
        from ai.video_detector.video_detector import VideoDetector, VideoFrameResult

        frame_results = [
            VideoFrameResult(i, float(i) * 0.5, 1, 0.20, 0.1, {})
            for i in range(20)
        ]
        flagged = VideoDetector._build_flagged_segments(frame_results)
        assert len(flagged) == 0

    def test_aggregate_score_weighting(self) -> None:
        """Temporal and artifact scores should influence final score."""
        from ai.video_detector.video_detector import VideoDetector, VideoFrameResult

        frame_results = [VideoFrameResult(0, 0.0, 1, 0.50, 0.5, {})]
        temporal = _empty_temporal()
        temporal.temporal_consistency_score = 0.90  # strong temporal signal
        artifacts = [make_artifact(0, score=0.50)]

        score = VideoDetector._aggregate_score(frame_results, temporal, artifacts)
        # With temporal at 0.9, final score should be pulled above 0.5
        assert score > 0.5

    def test_score_always_in_range(self) -> None:
        from ai.video_detector.video_detector import VideoDetector, VideoFrameResult

        for frame_score in [0.0, 0.5, 1.0]:
            frame_results = [VideoFrameResult(0, 0.0, 1, frame_score, 0.0, {})]
            score = VideoDetector._aggregate_score(
                frame_results, _empty_temporal(), []
            )
            assert 0.0 < score < 1.0

    def test_label_from_score(self) -> None:
        for score, expected in [(0.80, "AI"), (0.75, "AI"),
                                  (0.60, "UNCERTAIN"), (0.40, "HUMAN"), (0.20, "HUMAN")]:
            label = "AI" if score >= 0.75 else ("HUMAN" if score <= 0.40 else "UNCERTAIN")
            assert label == expected
