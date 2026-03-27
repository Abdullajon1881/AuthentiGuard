"""
Steps 66–68: Image AI detection feature extraction.

Step 66 — GAN Fingerprint Analysis
  Generative models leave unique, reproducible fingerprints in their output
  images. These are invisible to the human eye but detectable via correlation
  analysis. Proposed in: Marra et al. (2019) "Do GANs leave artificial
  fingerprints?"

Step 67 — FFT Frequency Domain Artifact Detection
  AI-generated images (GANs, diffusion models) exhibit characteristic
  frequency-domain artifacts:
  - Grid-like spectral peaks at regular intervals (GAN checkerboard)
  - Missing high-frequency content (over-smoothed by training)
  - Abnormal frequency distribution compared to real camera images
  Proposed in: Dzanic et al. (2020) "Fourier Spectrum Discrepancies in
  Deep Network Generated Images"

Step 68 — Texture and Symmetry Analysis
  - Perceptual texture: AI images have abnormally uniform texture at
    medium scales (GLCM features)
  - Facial symmetry: AI face generators produce unnaturally symmetric faces
  - Background coherence: AI images often have repetitive or incoherent
    backgrounds (copy-paste artifacts in diffusion models)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

from .preprocessing import ImageRecord

log = structlog.get_logger(__name__)


@dataclass
class ImageFeatures:
    """All extracted features for one image."""
    # GAN fingerprint
    fingerprint_correlation: float    # correlation with known GAN fingerprints
    fingerprint_energy:      float    # energy in the fingerprint frequency bands

    # FFT frequency domain
    fft_peak_regularity:     float    # regularity of spectral peaks (GAN checkerboard)
    fft_high_freq_ratio:     float    # high-freq energy ratio (low = over-smoothed)
    fft_azimuthal_variance:  float    # variance of azimuthal spectrum profile
    fft_grid_score:          float    # strength of grid-like frequency artifacts

    # Texture and symmetry
    texture_uniformity:      float    # GLCM-based texture uniformity
    glcm_contrast:           float    # local contrast measure
    bilateral_symmetry:      float    # horizontal symmetry score
    background_uniformity:   float    # peripheral region texture variance
    color_distribution_score: float   # unnatural colour palette indicator

    # Combined feature vector
    feature_vector:          np.ndarray   # 1D concatenation of all scalar features


# ── Step 66: GAN Fingerprint Analysis ─────────────────────────

def extract_gan_fingerprint(record: ImageRecord) -> dict[str, float]:
    """
    Extract GAN residual noise fingerprint using the SRM (Spatial Rich Model)
    high-pass filter approach.

    Real camera images contain a noise fingerprint from the sensor's
    photo-response non-uniformity (PRNU). AI-generated images either
    lack this fingerprint or contain a different, model-specific one.

    Returns:
      fingerprint_correlation: how strongly the image matches a camera fingerprint
      fingerprint_energy:      total energy of the high-frequency residual
    """
    try:
        return _srm_fingerprint(record.pixels)
    except Exception as exc:
        log.warning("gan_fingerprint_failed", error=str(exc))
        return {"fingerprint_correlation": 0.0, "fingerprint_energy": 0.0}


def _srm_fingerprint(pixels: np.ndarray) -> dict[str, float]:
    """
    Apply SRM high-pass filters to extract noise residual,
    then compute energy statistics.
    """
    gray = pixels.mean(axis=2).astype(np.float32)

    # SRM filter 1: 3×3 high-pass (removes low-frequency scene content)
    # Equivalent to: pixel - average_of_neighbours
    srm_kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1],
    ], dtype=np.float32) / 8.0

    # Manual 2D convolution (edge-padded)
    pad   = np.pad(gray, 1, mode="edge")
    h, w  = gray.shape
    residual = np.zeros_like(gray)
    for dy in range(3):
        for dx in range(3):
            residual += srm_kernel[dy, dx] * pad[dy:dy+h, dx:dx+w]

    energy      = float(np.mean(residual ** 2))
    # Normalised cross-correlation of residual with itself (autocorrelation peak)
    # A strong periodic peak → model fingerprint
    fft_res     = np.fft.fft2(residual)
    power       = np.abs(fft_res) ** 2
    autocorr    = np.fft.ifft2(power).real
    autocorr   /= max(autocorr[0, 0], 1e-8)

    # Look for off-centre peaks (model fingerprint signature)
    centre = autocorr.copy()
    cy, cx  = h // 2, w // 2
    mask_r  = 5
    centre[cy-mask_r:cy+mask_r, cx-mask_r:cx+mask_r] = 0
    peak_ratio = float(np.max(np.abs(centre)))

    return {
        "fingerprint_correlation": round(min(peak_ratio, 1.0), 4),
        "fingerprint_energy":      round(min(energy / 100.0, 1.0), 4),
    }


# ── Step 67: FFT Frequency Domain Analysis ────────────────────

def extract_fft_features(record: ImageRecord) -> dict[str, float]:
    """
    Analyse the 2D Fourier spectrum for AI-generation artifacts.

    Key signals:
      Grid-like peaks  — checkerboard artifact from transposed convolutions in GANs
      Low high-freq    — diffusion models over-smooth, removing camera noise
      Azimuthal var    — real cameras have isotropic noise; GANs are anisotropic
    """
    try:
        return _fft_analysis(record.pixels)
    except Exception as exc:
        log.warning("fft_analysis_failed", error=str(exc))
        return {
            "fft_peak_regularity": 0.0, "fft_high_freq_ratio": 0.5,
            "fft_azimuthal_variance": 0.0, "fft_grid_score": 0.0,
        }


def _fft_analysis(pixels: np.ndarray) -> dict[str, float]:
    gray   = pixels.mean(axis=2).astype(np.float32)
    h, w   = gray.shape

    # 2D FFT with Hanning window (reduces spectral leakage)
    window = np.outer(np.hanning(h), np.hanning(w))
    fft    = np.fft.fft2(gray * window)
    fft_s  = np.fft.fftshift(fft)
    mag    = np.log1p(np.abs(fft_s))   # log magnitude for visual range compression

    cy, cx = h // 2, w // 2

    # ── Grid artifact score ───────────────────────────────────
    # GAN checkerboard: peaks at multiples of (h/stride, w/stride)
    # Stride is typically 2 in transposed convolutions
    # Look for periodicity at h/2, w/2 positions
    grid_score = _compute_grid_score(mag, cy, cx, h, w)

    # ── High-frequency energy ratio ───────────────────────────
    # High-freq region: outside 75% of Nyquist radius
    Y, X      = np.ogrid[-cy:h-cy, -cx:w-cx]
    dist      = np.sqrt(X**2 + Y**2)
    max_dist  = min(cy, cx)
    hf_mask   = dist > max_dist * 0.75
    lf_mask   = dist < max_dist * 0.25

    total_energy = float(np.sum(mag))
    hf_energy    = float(np.sum(mag[hf_mask])) / max(total_energy, 1e-8)
    lf_energy    = float(np.sum(mag[lf_mask])) / max(total_energy, 1e-8)

    # Real camera images: hf_ratio ~0.08–0.15
    # AI images: hf_ratio ~0.02–0.06 (over-smoothed)
    hf_signal = max(0.0, 1.0 - hf_energy / 0.10)   # high when hf is low

    # ── Azimuthal variance ────────────────────────────────────
    # Profile the magnitude along angular slices
    n_angles  = 36
    profiles: list[float] = []
    for i in range(n_angles):
        angle = i * np.pi / n_angles
        dy    = np.sin(angle)
        dx    = np.cos(angle)
        # Sample along this direction
        rs    = np.arange(1, min(cy, cx))
        ys    = np.clip((cy + dy * rs).astype(int), 0, h - 1)
        xs    = np.clip((cx + dx * rs).astype(int), 0, w - 1)
        profiles.append(float(mag[ys, xs].mean()))

    azimuthal_var = float(np.std(profiles) / max(np.mean(profiles), 1e-8))

    # ── Peak regularity ───────────────────────────────────────
    # Find local maxima in spectrum and measure spacing regularity
    # (simplified: std of peak positions)
    flat_mag = mag.ravel()
    threshold = np.percentile(flat_mag, 99.5)
    peak_mask = mag > threshold
    # Exclude DC component (centre)
    peak_mask[cy-5:cy+5, cx-5:cx+5] = False
    peak_positions = np.argwhere(peak_mask)

    if len(peak_positions) >= 4:
        # Check if peaks form a regular grid
        dists = np.linalg.norm(peak_positions - np.array([cy, cx]), axis=1)
        regularity = float(1.0 - np.std(dists) / max(np.mean(dists), 1e-8))
        regularity = max(0.0, min(regularity, 1.0))
    else:
        regularity = 0.0

    return {
        "fft_peak_regularity":    round(regularity, 4),
        "fft_high_freq_ratio":    round(hf_signal, 4),
        "fft_azimuthal_variance": round(azimuthal_var, 4),
        "fft_grid_score":         round(grid_score, 4),
    }


def _compute_grid_score(mag: np.ndarray, cy: int, cx: int, h: int, w: int) -> float:
    """Score for grid-like frequency artifacts (GAN checkerboard)."""
    # Check energy at h/2 and w/2 offsets from centre
    grid_positions = [
        (cy + h // 4, cx), (cy - h // 4, cx),
        (cy, cx + w // 4), (cy, cx - w // 4),
        (cy + h // 4, cx + w // 4), (cy - h // 4, cx - w // 4),
    ]
    centre_energy = float(mag[cy, cx])
    grid_energies = []
    for gy, gx in grid_positions:
        gy = max(0, min(h-1, gy))
        gx = max(0, min(w-1, gx))
        grid_energies.append(float(mag[gy, gx]))

    if centre_energy < 1e-8:
        return 0.0
    grid_score = float(np.mean(grid_energies)) / centre_energy
    return round(min(grid_score, 1.0), 4)


# ── Step 68: Texture and Symmetry Analysis ────────────────────

def extract_texture_symmetry(record: ImageRecord) -> dict[str, float]:
    """
    Analyse texture using GLCM statistics and measure bilateral symmetry.
    """
    try:
        texture = _glcm_features(record.pixels)
        symmetry = _bilateral_symmetry(record.pixels)
        background = _background_uniformity(record.pixels)
        color = _color_distribution(record.pixels)
        return {**texture, **symmetry, **background, **color}
    except Exception as exc:
        log.warning("texture_symmetry_failed", error=str(exc))
        return {
            "texture_uniformity": 0.5, "glcm_contrast": 0.0,
            "bilateral_symmetry": 0.5, "background_uniformity": 0.5,
            "color_distribution_score": 0.5,
        }


def _glcm_features(pixels: np.ndarray) -> dict[str, float]:
    """
    Simplified GLCM (Gray-Level Co-occurrence Matrix) features.
    Measures local texture regularity — AI images are abnormally uniform.
    """
    gray = pixels.mean(axis=2).astype(np.uint8)
    h, w = gray.shape

    # Co-occurrence: count pairs of adjacent pixels at offset (1,0)
    # Quantise to 32 levels for speed
    quantised = (gray // 8).astype(np.int32)   # 0–31
    levels    = 32

    # Build GLCM for horizontal adjacency
    glcm = np.zeros((levels, levels), dtype=np.float32)
    for row in range(h):
        for col in range(w - 1):
            i = quantised[row, col]
            j = quantised[row, col + 1]
            glcm[i, j] += 1.0

    # Normalise
    total = glcm.sum()
    if total > 0:
        glcm /= total

    # GLCM features
    # Uniformity (angular second moment): high = uniform texture
    uniformity = float(np.sum(glcm ** 2))

    # Contrast: weighted by distance from diagonal
    i_idx, j_idx = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
    contrast = float(np.sum(glcm * (i_idx - j_idx) ** 2))

    return {
        "texture_uniformity": round(min(uniformity * 100.0, 1.0), 4),
        "glcm_contrast":      round(min(contrast / 10.0, 1.0), 4),
    }


def _bilateral_symmetry(pixels: np.ndarray) -> dict[str, float]:
    """
    Measure horizontal bilateral symmetry.
    AI face generators (StyleGAN, DALL-E) produce unnaturally symmetric faces.
    Real photos have natural asymmetry.
    """
    gray = pixels.mean(axis=2).astype(np.float32)
    flipped = gray[:, ::-1]
    diff = np.abs(gray - flipped)
    symmetry = float(1.0 - diff.mean() / max(gray.mean(), 1.0))
    return {"bilateral_symmetry": round(max(0.0, min(symmetry, 1.0)), 4)}


def _background_uniformity(pixels: np.ndarray) -> dict[str, float]:
    """
    Measure uniformity of the peripheral (background) regions.
    Diffusion models often have repetitive or overly smooth backgrounds.
    """
    h, w = pixels.shape[:2]
    margin = max(int(min(h, w) * 0.15), 10)

    # Sample border strips
    top    = pixels[:margin, :].astype(np.float32)
    bottom = pixels[-margin:, :].astype(np.float32)
    left   = pixels[:, :margin].astype(np.float32)
    right  = pixels[:, -margin:].astype(np.float32)

    all_bg = np.concatenate([top.ravel(), bottom.ravel(),
                              left.ravel(), right.ravel()])
    # Low variance → uniform background → possibly AI-generated
    variance = float(np.var(all_bg))
    uniformity = max(0.0, 1.0 - min(variance / 2000.0, 1.0))
    return {"background_uniformity": round(uniformity, 4)}


def _color_distribution(pixels: np.ndarray) -> dict[str, float]:
    """
    Analyse colour distribution for AI-generation patterns.
    AI images often have artificially smooth colour transitions
    and lack the long-tail distribution of natural photos.
    """
    # Compute per-channel histogram smoothness
    scores: list[float] = []
    for ch in range(3):
        hist, _ = np.histogram(pixels[:, :, ch].ravel(), bins=64, range=(0, 256))
        hist = hist.astype(np.float32)
        if hist.sum() == 0:
            continue
        hist /= hist.sum()
        # Smoothness = low variance in histogram = unnaturally uniform distribution
        smoothness = 1.0 - float(np.std(hist)) * 20.0
        scores.append(max(0.0, min(smoothness, 1.0)))

    return {"color_distribution_score": round(float(np.mean(scores)) if scores else 0.5, 4)}


# ── Combined extractor ────────────────────────────────────────

def extract_all_features(record: ImageRecord) -> ImageFeatures:
    """Full feature extraction pipeline for one image."""
    fp  = extract_gan_fingerprint(record)
    fft = extract_fft_features(record)
    tex = extract_texture_symmetry(record)

    fv = np.array([
        fp["fingerprint_correlation"], fp["fingerprint_energy"],
        fft["fft_peak_regularity"],   fft["fft_high_freq_ratio"],
        fft["fft_azimuthal_variance"], fft["fft_grid_score"],
        tex["texture_uniformity"],    tex["glcm_contrast"],
        tex["bilateral_symmetry"],    tex["background_uniformity"],
        tex["color_distribution_score"],
    ], dtype=np.float32)

    return ImageFeatures(
        fingerprint_correlation=fp["fingerprint_correlation"],
        fingerprint_energy=fp["fingerprint_energy"],
        fft_peak_regularity=fft["fft_peak_regularity"],
        fft_high_freq_ratio=fft["fft_high_freq_ratio"],
        fft_azimuthal_variance=fft["fft_azimuthal_variance"],
        fft_grid_score=fft["fft_grid_score"],
        texture_uniformity=tex["texture_uniformity"],
        glcm_contrast=tex["glcm_contrast"],
        bilateral_symmetry=tex["bilateral_symmetry"],
        background_uniformity=tex["background_uniformity"],
        color_distribution_score=tex["color_distribution_score"],
        feature_vector=fv,
    )
