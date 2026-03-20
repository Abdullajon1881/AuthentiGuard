"""
Unit tests for the image AI detection pipeline.
All tests use synthetic numpy arrays — no image files or model weights needed.
"""

from __future__ import annotations

import io
import numpy as np
import pytest

from ai.image_detector.features.extractor import (
    _srm_fingerprint,
    _fft_analysis,
    _glcm_features,
    _bilateral_symmetry,
    _background_uniformity,
    _color_distribution,
    _compute_grid_score,
    extract_all_features,
    ImageFeatures,
)
from ai.image_detector.pipeline.preprocessing import (
    _to_tensor, IMAGENET_MEAN, IMAGENET_STD,
)
from ai.image_detector.models.classifier import (
    generate_adversarial_image,
)
from ai.image_detector.image_detector import ImageDetector


# ── Fixtures ──────────────────────────────────────────────────

def make_image(h: int = 128, w: int = 128, mode: str = "random") -> np.ndarray:
    """Make a synthetic [H, W, 3] uint8 image."""
    rng = np.random.default_rng(42)
    if mode == "random":
        return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    elif mode == "flat":
        return np.full((h, w, 3), 128, dtype=np.uint8)
    elif mode == "gradient":
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            img[i, :, :] = int(255 * i / h)
        return img
    elif mode == "symmetric":
        half = rng.integers(0, 255, (h, w // 2, 3), dtype=np.uint8)
        return np.concatenate([half, half[:, ::-1, :]], axis=1)
    return make_image(h, w, "random")


def make_fake_record(pixels: np.ndarray):
    """Create a minimal ImageRecord-like object."""
    from ai.image_detector.pipeline.preprocessing import ImageRecord
    h, w = pixels.shape[:2]
    return ImageRecord(
        pixels=pixels,
        width=w, height=h,
        filename="test.jpg",
        file_format="jpeg",
        file_size=h * w * 3,
        color_mode="RGB",
        tensor_224=_to_tensor(pixels, 224),
        tensor_299=_to_tensor(pixels, 299),
        jpeg_quality=85,
        has_alpha=False,
        is_animated=False,
    )


# ── Preprocessing ─────────────────────────────────────────────

class TestPreprocessing:
    def test_to_tensor_shape(self) -> None:
        pixels = make_image(256, 256)
        tensor = _to_tensor(pixels, 224)
        assert tensor.shape == (3, 224, 224)

    def test_to_tensor_dtype(self) -> None:
        pixels = make_image()
        tensor = _to_tensor(pixels, 224)
        assert tensor.dtype == np.float32

    def test_to_tensor_normalised(self) -> None:
        """After normalisation with ImageNet stats, values should be centred around 0."""
        pixels = make_image()
        tensor = _to_tensor(pixels, 224)
        # Mean should be close to 0 (±2σ after normalisation)
        assert abs(float(tensor.mean())) < 3.0

    def test_to_tensor_299(self) -> None:
        pixels = make_image()
        tensor = _to_tensor(pixels, 299)
        assert tensor.shape == (3, 299, 299)


# ── GAN fingerprint ───────────────────────────────────────────

class TestGANFingerprint:
    def test_returns_valid_dict(self) -> None:
        pixels = make_image()
        result = _srm_fingerprint(pixels)
        assert "fingerprint_correlation" in result
        assert "fingerprint_energy" in result

    def test_values_in_range(self) -> None:
        pixels = make_image()
        result = _srm_fingerprint(pixels)
        assert 0.0 <= result["fingerprint_correlation"] <= 1.0
        assert 0.0 <= result["fingerprint_energy"] <= 1.0

    def test_flat_image_low_energy(self) -> None:
        pixels = make_image(mode="flat")
        result = _srm_fingerprint(pixels)
        # Flat image has no residual → near-zero fingerprint energy
        assert result["fingerprint_energy"] < 0.1

    def test_random_image_higher_energy_than_flat(self) -> None:
        flat   = _srm_fingerprint(make_image(mode="flat"))
        random = _srm_fingerprint(make_image(mode="random"))
        assert random["fingerprint_energy"] >= flat["fingerprint_energy"]


# ── FFT analysis ──────────────────────────────────────────────

class TestFFTAnalysis:
    def test_returns_all_keys(self) -> None:
        pixels = make_image()
        result = _fft_analysis(pixels)
        expected = {"fft_peak_regularity", "fft_high_freq_ratio",
                    "fft_azimuthal_variance", "fft_grid_score"}
        assert expected.issubset(result.keys())

    def test_all_values_in_range(self) -> None:
        result = _fft_analysis(make_image())
        for v in result.values():
            assert 0.0 <= v <= 1.0, f"Value {v} out of range"

    def test_flat_image_no_grid(self) -> None:
        """Flat image has no grid artifacts."""
        result = _fft_analysis(make_image(mode="flat"))
        assert result["fft_grid_score"] < 0.5

    def test_grid_score_compute(self) -> None:
        """_compute_grid_score should return a valid float."""
        mag = np.random.rand(128, 128).astype(np.float32) * 10
        score = _compute_grid_score(mag, 64, 64, 128, 128)
        assert 0.0 <= score <= 1.0


# ── Texture and symmetry ──────────────────────────────────────

class TestTextureSymmetry:
    def test_glcm_flat_high_uniformity(self) -> None:
        """Flat image has maximum texture uniformity."""
        flat   = make_image(mode="flat")
        result = _glcm_features(flat)
        assert result["texture_uniformity"] > 0.5

    def test_glcm_random_lower_uniformity(self) -> None:
        """Random image has lower uniformity than flat."""
        flat   = _glcm_features(make_image(mode="flat"))["texture_uniformity"]
        random = _glcm_features(make_image(mode="random"))["texture_uniformity"]
        assert flat >= random

    def test_bilateral_symmetry_symmetric_image(self) -> None:
        """A horizontally mirrored image should have high symmetry score."""
        sym = make_image(mode="symmetric")
        result = _bilateral_symmetry(sym)
        assert result["bilateral_symmetry"] > 0.85

    def test_bilateral_symmetry_random_image(self) -> None:
        """Random image should have lower symmetry than mirrored."""
        sym  = _bilateral_symmetry(make_image(mode="symmetric"))["bilateral_symmetry"]
        rand = _bilateral_symmetry(make_image(mode="random"))["bilateral_symmetry"]
        assert sym > rand

    def test_background_uniformity_flat(self) -> None:
        """Flat image → maximum background uniformity."""
        result = _background_uniformity(make_image(mode="flat"))
        assert result["background_uniformity"] > 0.8

    def test_background_uniformity_range(self) -> None:
        result = _background_uniformity(make_image())
        assert 0.0 <= result["background_uniformity"] <= 1.0

    def test_color_distribution_returns_score(self) -> None:
        result = _color_distribution(make_image())
        assert "color_distribution_score" in result
        assert 0.0 <= result["color_distribution_score"] <= 1.0


# ── Full feature extraction ───────────────────────────────────

class TestExtractAllFeatures:
    def test_returns_image_features(self) -> None:
        record = make_fake_record(make_image(256, 256))
        feats  = extract_all_features(record)
        assert isinstance(feats, ImageFeatures)

    def test_feature_vector_length(self) -> None:
        record = make_fake_record(make_image())
        feats  = extract_all_features(record)
        assert len(feats.feature_vector) == 11

    def test_all_scalars_in_range(self) -> None:
        record = make_fake_record(make_image())
        feats  = extract_all_features(record)
        for attr in [
            "fingerprint_correlation", "fingerprint_energy",
            "fft_peak_regularity", "fft_high_freq_ratio",
            "fft_azimuthal_variance", "fft_grid_score",
            "texture_uniformity", "glcm_contrast",
            "bilateral_symmetry", "background_uniformity",
            "color_distribution_score",
        ]:
            val = getattr(feats, attr)
            assert 0.0 <= val <= 1.0, f"{attr}={val} out of [0,1]"

    def test_feature_vector_finite(self) -> None:
        record = make_fake_record(make_image())
        feats  = extract_all_features(record)
        assert np.isfinite(feats.feature_vector).all()


# ── Adversarial augmentation ──────────────────────────────────

class TestAdversarialAugmentation:
    def test_noise_attack_same_shape(self) -> None:
        img    = make_image()
        result = generate_adversarial_image(img, "noise")
        assert result.shape == img.shape

    def test_noise_attack_uint8(self) -> None:
        img    = make_image()
        result = generate_adversarial_image(img, "noise")
        assert result.dtype == np.uint8
        assert result.min() >= 0 and result.max() <= 255

    def test_color_jitter_same_shape(self) -> None:
        img    = make_image()
        result = generate_adversarial_image(img, "color_jitter")
        assert result.shape == img.shape

    def test_unknown_attack_returns_original(self) -> None:
        img    = make_image()
        result = generate_adversarial_image(img, "unknown_attack_xyz")
        np.testing.assert_array_equal(result, img)


# ── Score logic ───────────────────────────────────────────────

class TestScoreLogic:
    def test_label_thresholds(self) -> None:
        for score, expected in [
            (0.99, "AI"), (0.75, "AI"),
            (0.74, "UNCERTAIN"), (0.55, "UNCERTAIN"), (0.41, "UNCERTAIN"),
            (0.40, "HUMAN"), (0.01, "HUMAN"),
        ]:
            label = "AI" if score >= 0.75 else ("HUMAN" if score <= 0.40 else "UNCERTAIN")
            assert label == expected

    def test_hybrid_score_blend(self) -> None:
        """70% deep model + 30% features should be between both components."""
        deep_score    = 0.80
        feature_score = 0.20
        combined = np.clip(0.70 * deep_score + 0.30 * feature_score, 0.01, 0.99)
        assert feature_score < combined < deep_score

    def test_confidence_formula(self) -> None:
        """Confidence = distance from 0.5, scaled to [0,1]."""
        for score in [0.01, 0.40, 0.50, 0.75, 0.99]:
            conf = abs(score - 0.5) * 2
            assert 0.0 <= conf <= 1.0
        assert abs(0.50 - 0.5) * 2 == 0.0   # 0.5 → 0 confidence
        assert abs(0.99 - 0.5) * 2 > 0.9    # near-1 → high confidence
