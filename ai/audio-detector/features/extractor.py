"""
Steps 48–49: Spectrogram generation and feature extraction.

Features extracted per chunk:
  Step 48 — Mel spectrogram (primary input to CNN/Transformer)
  Step 49 — MFCC (13 + delta + delta-delta = 39 coefficients)
           — Pitch features (F0, jitter, shimmer)
           — Phase coherence (Group Delay Deviation — key deepfake signal)
           — Spectral features (centroid, rolloff, flux, bandwidth)
           — Prosodic features (energy envelope, zero-crossing rate)

Key insight on deepfakes:
  TTS and voice conversion systems generate each frame independently,
  leading to unnatural phase transitions between frames (phase discontinuity).
  The Group Delay Deviation (GDD) metric captures this directly.
  Real voices maintain smooth phase coherence; deepfakes show jagged GDD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

from .preprocessing import AudioChunk, AudioSegment

log = structlog.get_logger(__name__)

# ── Feature config ─────────────────────────────────────────────
N_MELS      = 128        # mel filter banks
N_MFCC      = 13         # base MFCC coefficients (+ deltas = 39 total)
HOP_LENGTH  = 512        # samples between frames (~32ms at 16kHz)
WIN_LENGTH  = 1024       # FFT window size (~64ms)
N_FFT       = 1024
FMIN        = 60.0       # min frequency for mel (excludes subsonics)
FMAX        = 8000.0     # max frequency (Nyquist for 16kHz)


@dataclass
class AudioFeatures:
    """All features extracted from one AudioChunk."""
    # Spectrograms (used as image inputs to CNN)
    mel_spectrogram:  np.ndarray   # shape [N_MELS, T] in dB
    # MFCC + deltas
    mfcc:             np.ndarray   # shape [39, T]
    # Pitch
    f0:               np.ndarray   # shape [T_pitch] fundamental frequency
    f0_mean:          float
    f0_std:           float
    jitter:           float        # cycle-to-cycle F0 variation (%)
    shimmer:          float        # amplitude variation (dB)
    # Phase coherence
    gdd_mean:         float        # Group Delay Deviation mean
    gdd_std:          float        # higher = less coherent = more likely deepfake
    # Spectral
    spectral_centroid_mean: float
    spectral_rolloff_mean:  float
    spectral_flux_mean:     float
    zcr_mean:         float        # Zero-crossing rate
    # Energy
    rms_mean:         float
    rms_std:          float
    # Combined feature vector for classical ML
    feature_vector:   np.ndarray   # 1D concatenation of scalar features


# ── Step 48: Mel spectrogram ──────────────────────────────────

def compute_mel_spectrogram(
    waveform: np.ndarray,
    sr: int,
    n_mels: int = N_MELS,
) -> np.ndarray:
    """
    Compute log-Mel spectrogram. Returns [n_mels, T] in decibels.
    Used as the 2D image input to CNN/ResNet classifiers.
    """
    try:
        import librosa  # type: ignore
        S = librosa.feature.melspectrogram(
            y=waveform, sr=sr,
            n_mels=n_mels, n_fft=N_FFT,
            hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
            fmin=FMIN, fmax=FMAX,
        )
        return librosa.power_to_db(S, ref=np.max).astype(np.float32)
    except Exception as exc:
        log.warning("mel_spectrogram_failed", error=str(exc))
        # Return zeros rather than crash — the model handles silence
        T = 1 + len(waveform) // HOP_LENGTH
        return np.zeros((n_mels, T), dtype=np.float32)


# ── Step 49a: MFCC + deltas ───────────────────────────────────

def compute_mfcc(waveform: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute MFCC + delta + delta-delta.
    Returns [39, T] — standard speech feature representation.
    """
    try:
        import librosa  # type: ignore
        mfcc     = librosa.feature.mfcc(
            y=waveform, sr=sr, n_mfcc=N_MFCC,
            n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        )
        delta    = librosa.feature.delta(mfcc)
        delta2   = librosa.feature.delta(mfcc, order=2)
        return np.vstack([mfcc, delta, delta2]).astype(np.float32)
    except Exception as exc:
        log.warning("mfcc_failed", error=str(exc))
        T = 1 + len(waveform) // HOP_LENGTH
        return np.zeros((3 * N_MFCC, T), dtype=np.float32)


# ── Step 49b: Pitch features ──────────────────────────────────

def compute_pitch_features(waveform: np.ndarray, sr: int) -> dict[str, Any]:
    """
    Estimate fundamental frequency (F0) and voice quality measures.

    Jitter — cycle-to-cycle variation in F0 (ms). Elevated in deepfakes.
    Shimmer — amplitude variation per cycle (dB). Elevated in deepfakes.
    """
    try:
        import librosa  # type: ignore

        # pyin: probabilistic YIN — more accurate than basic YIN
        f0, voiced_flag, voiced_prob = librosa.pyin(
            waveform, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr, hop_length=HOP_LENGTH,
        )
        f0_voiced = f0[voiced_flag & ~np.isnan(f0)]

        if len(f0_voiced) < 3:
            return _empty_pitch()

        f0_mean = float(np.nanmean(f0_voiced))
        f0_std  = float(np.nanstd(f0_voiced))

        # Jitter: mean absolute cycle-to-cycle difference / mean F0
        diffs  = np.abs(np.diff(f0_voiced))
        jitter = float(np.mean(diffs) / max(f0_mean, 1.0) * 100)

        # Shimmer: variation in RMS across voiced frames
        frame_rms = np.array([
            np.sqrt(np.mean(waveform[i*HOP_LENGTH:(i+1)*HOP_LENGTH]**2))
            for i in range(len(f0_voiced))
        ])
        shimmer = float(np.std(frame_rms) / max(np.mean(frame_rms), 1e-8))

        return {
            "f0": f0,
            "f0_mean": f0_mean,
            "f0_std":  f0_std,
            "jitter":  min(jitter, 100.0),   # cap at 100%
            "shimmer": min(shimmer, 1.0),
        }
    except Exception as exc:
        log.warning("pitch_features_failed", error=str(exc))
        return _empty_pitch()


def _empty_pitch() -> dict[str, Any]:
    T = 100
    return {
        "f0": np.zeros(T), "f0_mean": 0.0, "f0_std": 0.0,
        "jitter": 0.0, "shimmer": 0.0,
    }


# ── Step 49c: Phase coherence — Group Delay Deviation ─────────

def compute_phase_coherence(waveform: np.ndarray) -> dict[str, float]:
    """
    Compute Group Delay Deviation (GDD) — a key anti-spoofing feature.

    Genuine speech produced by vocal tract has smooth, predictable phase
    relationships across frequency. Neural vocoders (WaveNet, HiFi-GAN etc.)
    generate waveforms frame-by-frame, producing unnatural phase jumps.

    GDD measures the deviation of the group delay from its expected value.
    High GDD std → likely deepfake.

    Reference: Tak et al. (2022) "RawBoost: A Raw Data Boosting and Denoising
    Strategy for Spoofed Speech Detection"
    """
    try:
        n    = len(waveform)
        if n < N_FFT * 2:
            return {"gdd_mean": 0.0, "gdd_std": 0.0}

        # Compute STFT
        frames = _stft_frames(waveform)

        gdd_values: list[float] = []
        for frame in frames:
            # Complex spectrum
            H      = np.fft.rfft(frame * np.hanning(len(frame)))
            # Group delay = -d(phase)/d(omega)
            # Approximate via central difference on unwrapped phase
            phase  = np.unwrap(np.angle(H))
            gd     = -np.diff(phase)
            # Expected group delay for linear phase: constant
            # Deviation = std of group delay across frequency
            gdd_values.append(float(np.std(gd)))

        if not gdd_values:
            return {"gdd_mean": 0.0, "gdd_std": 0.0}

        return {
            "gdd_mean": float(np.mean(gdd_values)),
            "gdd_std":  float(np.std(gdd_values)),
        }
    except Exception as exc:
        log.warning("phase_coherence_failed", error=str(exc))
        return {"gdd_mean": 0.0, "gdd_std": 0.0}


def _stft_frames(waveform: np.ndarray, max_frames: int = 200) -> list[np.ndarray]:
    """Extract overlapping STFT frames."""
    frames = []
    step   = HOP_LENGTH
    for i in range(0, len(waveform) - WIN_LENGTH, step):
        frames.append(waveform[i:i + WIN_LENGTH])
        if len(frames) >= max_frames:
            break
    return frames


# ── Step 49d: Spectral and energy features ────────────────────

def compute_spectral_features(waveform: np.ndarray, sr: int) -> dict[str, float]:
    """Spectral centroid, rolloff, flux, ZCR, RMS energy."""
    try:
        import librosa  # type: ignore

        centroid  = librosa.feature.spectral_centroid(y=waveform, sr=sr, hop_length=HOP_LENGTH)
        rolloff   = librosa.feature.spectral_rolloff(y=waveform, sr=sr, hop_length=HOP_LENGTH)
        zcr       = librosa.feature.zero_crossing_rate(waveform, hop_length=HOP_LENGTH)
        rms       = librosa.feature.rms(y=waveform, hop_length=HOP_LENGTH)

        # Spectral flux: frame-to-frame spectral change
        S     = np.abs(librosa.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH))
        flux  = np.mean(np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0)))

        return {
            "spectral_centroid_mean": float(np.mean(centroid)),
            "spectral_rolloff_mean":  float(np.mean(rolloff)),
            "spectral_flux_mean":     float(flux),
            "zcr_mean":               float(np.mean(zcr)),
            "rms_mean":               float(np.mean(rms)),
            "rms_std":                float(np.std(rms)),
        }
    except Exception as exc:
        log.warning("spectral_features_failed", error=str(exc))
        return {
            "spectral_centroid_mean": 0.0, "spectral_rolloff_mean": 0.0,
            "spectral_flux_mean": 0.0, "zcr_mean": 0.0,
            "rms_mean": 0.0, "rms_std": 0.0,
        }


# ── Combined extractor ────────────────────────────────────────

def extract_features(chunk: AudioChunk) -> AudioFeatures:
    """
    Full feature extraction pipeline for one AudioChunk.
    Called by the Celery worker once per chunk.
    """
    wav = chunk.waveform
    sr  = chunk.sample_rate

    mel    = compute_mel_spectrogram(wav, sr)
    mfcc   = compute_mfcc(wav, sr)
    pitch  = compute_pitch_features(wav, sr)
    phase  = compute_phase_coherence(wav)
    spec   = compute_spectral_features(wav, sr)

    # Scalar feature vector for classical ML / meta-classifier
    scalars = np.array([
        pitch["f0_mean"], pitch["f0_std"],
        pitch["jitter"],  pitch["shimmer"],
        phase["gdd_mean"], phase["gdd_std"],
        spec["spectral_centroid_mean"], spec["spectral_rolloff_mean"],
        spec["spectral_flux_mean"],     spec["zcr_mean"],
        spec["rms_mean"],               spec["rms_std"],
    ], dtype=np.float32)

    # MFCC statistics (mean + std per coefficient) → 78 values
    mfcc_stats = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1)
    ])

    feature_vector = np.concatenate([scalars, mfcc_stats])

    return AudioFeatures(
        mel_spectrogram=mel,
        mfcc=mfcc,
        f0=pitch["f0"],
        f0_mean=pitch["f0_mean"],
        f0_std=pitch["f0_std"],
        jitter=pitch["jitter"],
        shimmer=pitch["shimmer"],
        gdd_mean=phase["gdd_mean"],
        gdd_std=phase["gdd_std"],
        spectral_centroid_mean=spec["spectral_centroid_mean"],
        spectral_rolloff_mean=spec["spectral_rolloff_mean"],
        spectral_flux_mean=spec["spectral_flux_mean"],
        zcr_mean=spec["zcr_mean"],
        rms_mean=spec["rms_mean"],
        rms_std=spec["rms_std"],
        feature_vector=feature_vector,
    )
