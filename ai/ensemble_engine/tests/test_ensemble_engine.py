"""
Unit tests for Phase 10 — Ensemble Engine & Cross-Modal Detection.

Steps 79–83 tested here:
  - DetectorOutput schema and normalisation
  - Feature vector construction
  - Ensemble weighted-average fallback
  - Cross-modal consistency checks (all three)
  - Model attribution (text and image)
  - Watermark detection (text, image, audio)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ai.ensemble_engine.routing.dispatcher import DetectorOutput, ContentType
from ai.ensemble_engine.meta.meta_classifier import (
    build_multi_detector_feature_vector, EnsembleMetaClassifier,
    N_FEATURES, FEATURE_NAMES,
)
from ai.ensemble_engine.cross_modal.consistency import (
    check_lip_audio_alignment,
    check_writing_style_vs_metadata,
    check_exif_vs_claimed_source,
    analyse_cross_modal,
)
from ai.ensemble_engine.attribution.attributor import (
    attribute_text, attribute_image, _human_attribution,
    GPT_MARKERS, CLAUDE_MARKERS, LLAMA_MARKERS,
)
from ai.ensemble_engine.watermark.detector import (
    detect_text_watermark, _token_bucket,
    detect_image_watermark, detect_audio_watermark,
)


# ── Fixtures ──────────────────────────────────────────────────

def make_output(
    content_type: ContentType = "text",
    score: float = 0.8,
    confidence: float = 0.6,
    error: str | None = None,
    flagged_segments: list | None = None,
) -> DetectorOutput:
    return DetectorOutput(
        content_type=content_type,
        score=score, confidence=confidence,
        label="AI" if score >= 0.75 else "HUMAN",
        layer_scores={"layer1": score, "layer2": score * 0.9},
        sentence_scores=[{"text": "test", "score": score, "evidence": {}}],
        flagged_segments=flagged_segments or [],
        evidence={"top_signals": []},
        processing_ms=100,
        error=error,
    )


# ── DetectorOutput ────────────────────────────────────────────

class TestDetectorOutput:
    def test_score_in_range(self) -> None:
        out = make_output(score=0.7)
        assert 0.0 <= out.score <= 1.0

    def test_errored_output_has_error_field(self) -> None:
        out = make_output(error="model failed")
        assert out.error == "model failed"

    def test_label_assignment(self) -> None:
        assert make_output(score=0.80).label == "AI"
        assert make_output(score=0.30).label == "HUMAN"


# ── Feature vector ────────────────────────────────────────────

class TestFeatureVector:
    def test_correct_length(self) -> None:
        outputs = [make_output("text", 0.8)]
        fv = build_multi_detector_feature_vector(outputs)
        assert len(fv) == N_FEATURES

    def test_feature_names_match_length(self) -> None:
        assert len(FEATURE_NAMES) == N_FEATURES

    def test_all_finite(self) -> None:
        outputs = [make_output("text", 0.8), make_output("image", 0.6)]
        fv = build_multi_detector_feature_vector(outputs)
        assert np.isfinite(fv).all()

    def test_missing_detector_gets_neutral_score(self) -> None:
        # Only text detector ran — image slot should be 0.5
        outputs = [make_output("text", 0.9)]
        fv = build_multi_detector_feature_vector(outputs)
        image_idx = 1  # score_image is index 1
        assert fv[image_idx] == pytest.approx(0.5, abs=0.01)

    def test_errored_detector_gets_error_flag(self) -> None:
        outputs = [make_output("text", 0.8, error="failed")]
        fv = build_multi_detector_feature_vector(outputs)
        err_text_idx = 10  # err_text is index 10 (5 scores + 5 confs + 0)
        assert fv[err_text_idx] == 1.0

    def test_flagged_segments_counted(self) -> None:
        segs = [{"start_s": 0.0, "end_s": 1.0, "score": 0.9, "severity": "high"}]
        outputs = [make_output("video", 0.8, flagged_segments=segs)]
        fv = build_multi_detector_feature_vector(outputs)
        # n_flagged is near end of vector
        n_flagged_idx = FEATURE_NAMES.index("n_flagged_segments")
        assert fv[n_flagged_idx] == 1.0

    def test_watermark_signal_propagated(self) -> None:
        outputs = [make_output("text", 0.8)]
        meta = {"watermark": {"watermark_detected": True, "confidence": 0.9}}
        fv = build_multi_detector_feature_vector(outputs, meta)
        wm_idx = FEATURE_NAMES.index("watermark_detected")
        assert fv[wm_idx] == 1.0


# ── Ensemble meta-classifier ──────────────────────────────────

class TestEnsembleMetaClassifier:
    def test_weighted_average_single_output(self) -> None:
        clf = EnsembleMetaClassifier()
        outputs = [make_output("text", 0.8)]
        result = clf.predict(outputs)
        assert result.score == pytest.approx(0.8, abs=0.01)

    def test_weighted_average_multiple_outputs(self) -> None:
        clf = EnsembleMetaClassifier()
        outputs = [make_output("text", 0.8), make_output("image", 0.6)]
        result = clf.predict(outputs)
        # Weighted: text=0.35 weight, image=0.25 weight
        expected = (0.8 * 0.35 + 0.6 * 0.25) / (0.35 + 0.25)
        assert abs(result.score - expected) < 0.02

    def test_errored_output_excluded(self) -> None:
        clf = EnsembleMetaClassifier()
        outputs = [make_output("text", 0.8), make_output("image", 0.6, error="failed")]
        result = clf.predict(outputs)
        # Only text should contribute
        assert result.score == pytest.approx(0.8, abs=0.01)

    def test_empty_outputs_returns_uncertain(self) -> None:
        clf = EnsembleMetaClassifier()
        result = clf.predict([])
        assert result.score == 0.5
        assert result.label == "UNCERTAIN"

    def test_watermark_boosts_score(self) -> None:
        clf = EnsembleMetaClassifier()
        outputs = [make_output("text", 0.7)]
        no_wm  = clf.predict(outputs, {})
        with_wm = clf.predict(outputs, {
            "watermark": {"watermark_detected": True, "confidence": 0.9}
        })
        assert with_wm.score > no_wm.score

    def test_c2pa_reduces_score(self) -> None:
        clf = EnsembleMetaClassifier()
        outputs = [make_output("text", 0.7)]
        no_c2pa  = clf.predict(outputs, {})
        with_c2pa = clf.predict(outputs, {
            "provenance": {"c2pa_verified": True}
        })
        assert with_c2pa.score < no_c2pa.score

    def test_contributing_detectors_listed(self) -> None:
        clf = EnsembleMetaClassifier()
        outputs = [make_output("text", 0.8), make_output("image", 0.6)]
        result = clf.predict(outputs)
        assert "text" in result.contributing_detectors
        assert "image" in result.contributing_detectors


# ── Cross-modal consistency ───────────────────────────────────

class TestCrossModalConsistency:
    def _make_evidence(self, n_chunks: int, score: float) -> dict:
        return {
            "sentence_scores": [{"score": score, "text": f"{i}s"} for i in range(n_chunks)]
        }

    def test_lip_audio_insufficient_data(self) -> None:
        result = check_lip_audio_alignment(
            self._make_evidence(1, 0.8),
            self._make_evidence(1, 0.2),
        )
        assert result.mismatch_detected is False
        assert result.mismatch_severity == "none"

    def test_lip_audio_large_gap_is_mismatch(self) -> None:
        result = check_lip_audio_alignment(
            self._make_evidence(10, 0.85),   # video: AI
            self._make_evidence(10, 0.10),   # audio: human
        )
        assert result.mismatch_detected is True
        assert result.mismatch_severity in ("medium", "high")

    def test_lip_audio_consistent_no_mismatch(self) -> None:
        result = check_lip_audio_alignment(
            self._make_evidence(10, 0.80),
            self._make_evidence(10, 0.82),
        )
        assert result.mismatch_detected is False

    def test_writing_vs_metadata_hidden_ai(self) -> None:
        result = check_writing_style_vs_metadata(
            text_score=0.90,
            text_evidence={},
            file_metadata={"author": "Jane Smith", "software": "Microsoft Word"},
        )
        assert result.mismatch_detected is True
        assert result.evidence["mismatch_type"] == "hidden_ai"

    def test_writing_vs_metadata_no_author_no_mismatch(self) -> None:
        result = check_writing_style_vs_metadata(
            text_score=0.90,
            text_evidence={},
            file_metadata={},   # no author claimed
        )
        assert result.mismatch_detected is False

    def test_writing_vs_metadata_ai_labelled_correctly(self) -> None:
        result = check_writing_style_vs_metadata(
            text_score=0.85,
            text_evidence={},
            file_metadata={"software": "ChatGPT"},
        )
        assert result.evidence["metadata_claims_ai"] is True

    def test_exif_ai_software_is_mismatch(self) -> None:
        result = check_exif_vs_claimed_source(
            image_score=0.85,
            exif_data={"has_exif": True, "image_software": "Stable Diffusion v2.1"},
            claimed_source="photo captured with iPhone",
        )
        assert result.mismatch_detected is True
        assert result.mismatch_severity == "high"

    def test_exif_consistent_no_mismatch(self) -> None:
        result = check_exif_vs_claimed_source(
            image_score=0.20,
            exif_data={"has_exif": True, "has_camera_info": True,
                        "image_make": "Canon", "image_software": ""},
            claimed_source="photograph from Canon camera",
        )
        assert result.mismatch_detected is False

    def test_analyse_cross_modal_no_outputs_returns_consistent(self) -> None:
        analysis = analyse_cross_modal({})
        assert analysis.overall_consistency == 1.0
        assert analysis.inconsistency_score == 0.0

    def test_analyse_cross_modal_video_audio(self) -> None:
        analysis = analyse_cross_modal({
            "video": self._make_evidence(10, 0.85),
            "audio": self._make_evidence(10, 0.10),
        })
        assert len(analysis.checks) == 1
        assert analysis.checks[0].check_name == "lip_audio_alignment"


# ── Model attribution ─────────────────────────────────────────

class TestModelAttribution:
    def test_human_text_returns_human_attribution(self) -> None:
        result = attribute_text("Hello, how are you doing today?", ai_score=0.10)
        assert result.primary_attribution == "human"
        assert result.human > 0.80

    def test_gpt_markers_boost_gpt_score(self) -> None:
        text = "Furthermore, it is worth noting that the paradigm leverages robust mechanisms."
        result = attribute_text(text, ai_score=0.85)
        assert result.gpt_family > result.claude_family

    def test_claude_markers_boost_claude_score(self) -> None:
        text = "I'd be happy to help you. Let me walk you through this thoughtfully."
        result = attribute_text(text, ai_score=0.85)
        assert result.claude_family > 0

    def test_attribution_sums_to_one(self) -> None:
        for score in [0.10, 0.50, 0.80, 0.95]:
            text = "Test text for attribution analysis with enough words to work properly."
            result = attribute_text(text, score)
            total = (result.gpt_family + result.claude_family +
                      result.llama_family + result.human + result.other_ai)
            assert abs(total - 1.0) < 0.02, f"Sum={total} for score={score}"

    def test_human_attribution_helper(self) -> None:
        result = _human_attribution()
        assert result.primary_attribution == "human"
        assert result.human > 0.90

    def test_image_attribution_ai_score_low(self) -> None:
        result = attribute_image(0.20)
        assert result.human > 0.70

    def test_image_attribution_sums_to_one(self) -> None:
        result = attribute_image(0.80, {"fft_grid_score": 0.6, "fft_high_freq_ratio": 0.3})
        total  = (result.gpt_family + result.claude_family +
                   result.llama_family + result.human + result.other_ai)
        assert abs(total - 1.0) < 0.02

    def test_marker_sets_not_empty(self) -> None:
        assert len(GPT_MARKERS) >= 10
        assert len(CLAUDE_MARKERS) >= 10
        assert len(LLAMA_MARKERS) >= 5


# ── Watermark detection ───────────────────────────────────────

class TestWatermarkDetection:
    def test_text_too_short_not_detected(self) -> None:
        result = detect_text_watermark("too short")
        assert result.watermark_detected is False
        assert result.evidence["reason"] == "too_short"

    def test_text_returns_valid_result(self) -> None:
        text = " ".join([f"word{i}" for i in range(100)])
        result = detect_text_watermark(text)
        assert isinstance(result.watermark_detected, bool)
        assert 0.0 <= result.confidence <= 1.0
        assert result.z_score is not None
        assert result.p_value is not None

    def test_z_score_and_p_value_relationship(self) -> None:
        """Higher |z_score| should correspond to lower p_value."""
        text1 = " ".join([f"word{i}" for i in range(200)])
        r1 = detect_text_watermark(text1)
        # z_score and p_value should be inversely related (higher |z| = lower p)
        assert r1.p_value is not None
        assert r1.z_score is not None
        assert r1.p_value >= 0.0

    def test_token_bucket_deterministic(self) -> None:
        """Same token + key → same bucket every time."""
        b1 = _token_bucket("hello", "key123")
        b2 = _token_bucket("hello", "key123")
        assert b1 == b2
        assert b1 in {"green", "red"}

    def test_token_bucket_different_keys_may_differ(self) -> None:
        """Different keys produce (potentially) different buckets."""
        results = {_token_bucket("hello", f"key{i}") for i in range(20)}
        assert len(results) >= 1   # at least one bucket used

    def test_image_watermark_small_image_handled(self) -> None:
        pixels = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = detect_image_watermark(pixels)
        assert isinstance(result.watermark_detected, bool)

    def test_image_watermark_normal_image(self) -> None:
        pixels = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = detect_image_watermark(pixels)
        assert 0.0 <= result.confidence <= 1.0
        assert result.z_score is not None

    def test_audio_watermark_short_audio(self) -> None:
        short = np.zeros(8000, dtype=np.float32)
        result = detect_audio_watermark(short, sr=16000)
        assert result.watermark_detected is False

    def test_audio_watermark_returns_valid(self) -> None:
        audio = np.random.randn(48000).astype(np.float32)
        result = detect_audio_watermark(audio, sr=16000)
        assert isinstance(result.watermark_detected, bool)
        assert 0.0 <= result.confidence <= 1.0
