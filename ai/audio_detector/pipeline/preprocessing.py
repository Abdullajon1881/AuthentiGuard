"""
Step 47: Audio upload and preprocessing pipeline.

Handles all supported audio formats (MP3, WAV, FLAC, M4A, OGG, AAC).
Normalises to a standard representation: 16kHz mono float32 waveform.
This canonical form is what all feature extractors consume.

Pipeline:
  raw bytes → format detection → decode → resample → mono → normalise → waveform
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog

log = structlog.get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────

TARGET_SR    = 16_000      # target sample rate (Hz) — standard for speech models
MAX_DURATION = 600.0       # 10 minutes max; longer clips are chunked
MIN_DURATION = 0.5         # reject clips shorter than 0.5s
CHUNK_DURATION = 30.0      # analyse in 30-second windows

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".opus"}


@dataclass
class AudioSegment:
    """Canonical audio representation consumed by all feature extractors."""
    waveform:   np.ndarray     # float32 shape [n_samples] normalised to [-1, 1]
    sample_rate: int           # always TARGET_SR after preprocessing
    duration_s:  float
    n_channels_original: int   # before downmix
    original_sr: int           # before resampling
    filename:    str
    file_format: str           # "wav", "mp3", etc.


@dataclass
class AudioChunk:
    """A 30-second window from a longer recording."""
    waveform:     np.ndarray
    sample_rate:  int
    start_s:      float
    end_s:        float
    chunk_idx:    int


# ── Core preprocessing ────────────────────────────────────────

def load_audio(data: bytes, filename: str) -> AudioSegment:
    """
    Decode and normalise raw audio bytes to a canonical AudioSegment.

    Uses torchaudio as primary decoder (fastest, handles most formats).
    Falls back to soundfile, then librosa for edge cases.
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported audio format: {ext}")

    waveform, sr, n_channels = _decode(data, filename)
    waveform = _to_mono(waveform)
    waveform = _resample(waveform, sr, TARGET_SR)
    waveform = _normalise(waveform)

    duration = len(waveform) / TARGET_SR

    if duration < MIN_DURATION:
        raise ValueError(f"Audio too short: {duration:.2f}s (minimum {MIN_DURATION}s)")

    if duration > MAX_DURATION:
        log.warning("audio_truncated",
                    original_s=round(duration, 1),
                    truncated_to=MAX_DURATION)
        waveform = waveform[:int(MAX_DURATION * TARGET_SR)]
        duration = MAX_DURATION

    log.info("audio_loaded",
             filename=filename,
             duration_s=round(duration, 2),
             original_sr=sr,
             n_channels=n_channels)

    return AudioSegment(
        waveform=waveform,
        sample_rate=TARGET_SR,
        duration_s=float(len(waveform) / TARGET_SR),
        n_channels_original=n_channels,
        original_sr=sr,
        filename=filename,
        file_format=ext.lstrip("."),
    )


def _decode(data: bytes, filename: str) -> tuple[np.ndarray, int, int]:
    """Try torchaudio → soundfile → librosa in order."""
    buf = io.BytesIO(data)

    # ── torchaudio (fastest, GPU-ready) ──────────────────────
    try:
        import torch
        import torchaudio  # type: ignore
        buf.seek(0)
        waveform_t, sr = torchaudio.load(buf)
        n_ch = waveform_t.shape[0]
        return waveform_t.numpy(), sr, n_ch
    except Exception as e:
        log.debug("torchaudio_failed", error=str(e))

    # ── soundfile (excellent WAV/FLAC support) ────────────────
    try:
        import soundfile as sf  # type: ignore
        buf.seek(0)
        data_sf, sr = sf.read(buf, dtype="float32", always_2d=True)
        n_ch = data_sf.shape[1]
        return data_sf.T, sr, n_ch   # transpose to [channels, samples]
    except Exception as e:
        log.debug("soundfile_failed", error=str(e))

    # ── librosa (most permissive — handles MP3 via ffmpeg) ────
    try:
        import librosa  # type: ignore
        buf.seek(0)
        y, sr = librosa.load(buf, sr=None, mono=False)
        if y.ndim == 1:
            y = y[np.newaxis, :]
        return y, sr, y.shape[0]
    except Exception as e:
        raise RuntimeError(f"All audio decoders failed for {filename}: {e}") from e


def _to_mono(waveform: np.ndarray) -> np.ndarray:
    """Downmix multi-channel audio to mono by averaging channels."""
    if waveform.ndim == 1:
        return waveform.astype(np.float32)
    return waveform.mean(axis=0).astype(np.float32)


def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return waveform
    try:
        import torchaudio.functional as F  # type: ignore
        import torch
        t = torch.from_numpy(waveform).unsqueeze(0)
        resampled = F.resample(t, orig_sr, target_sr)
        return resampled.squeeze(0).numpy()
    except Exception:
        try:
            import librosa  # type: ignore
            return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
        except Exception:
            # Simple decimation fallback (low quality but never fails)
            ratio = orig_sr / target_sr
            indices = np.round(np.arange(0, len(waveform), ratio)).astype(int)
            indices = indices[indices < len(waveform)]
            return waveform[indices]


def _normalise(waveform: np.ndarray) -> np.ndarray:
    """Peak normalise to [-1, 1]. Handles silence gracefully."""
    peak = np.abs(waveform).max()
    if peak < 1e-8:
        return waveform.astype(np.float32)
    return (waveform / peak).astype(np.float32)


# ── Chunking ──────────────────────────────────────────────────

def chunk_audio(segment: AudioSegment, chunk_s: float = CHUNK_DURATION) -> list[AudioChunk]:
    """
    Split a long AudioSegment into fixed-length chunks with 2-second overlap.
    Shorter clips return a single chunk.
    """
    sr = segment.sample_rate
    chunk_samples = int(chunk_s * sr)
    overlap_samples = int(2.0 * sr)   # 2-second overlap
    step = chunk_samples - overlap_samples

    chunks: list[AudioChunk] = []
    waveform = segment.waveform
    total = len(waveform)

    if total <= chunk_samples:
        return [AudioChunk(
            waveform=waveform, sample_rate=sr,
            start_s=0.0, end_s=segment.duration_s, chunk_idx=0,
        )]

    start = 0
    idx   = 0
    while start < total:
        end  = min(start + chunk_samples, total)
        chunk_wave = waveform[start:end]
        # Pad last chunk to full length with zeros
        if len(chunk_wave) < chunk_samples:
            chunk_wave = np.pad(chunk_wave, (0, chunk_samples - len(chunk_wave)))

        chunks.append(AudioChunk(
            waveform=chunk_wave,
            sample_rate=sr,
            start_s=start / sr,
            end_s=end / sr,
            chunk_idx=idx,
        ))
        start += step
        idx   += 1
        if end >= total:
            break

    log.debug("audio_chunked", n_chunks=len(chunks), chunk_s=chunk_s)
    return chunks
