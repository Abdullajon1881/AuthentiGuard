"""
Step 29: Metadata Service.
Extracts EXIF data, device fingerprints, and watermark signals from files.
Runs in parallel with the AI detection pipeline via Celery.
"""

from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)


# ── EXIF Extraction ───────────────────────────────────────────

def extract_exif(data: bytes, filename: str) -> dict[str, Any]:
    """
    Extract EXIF metadata from image/video files.
    Returns empty dict for unsupported types — never raises.
    """
    ext = Path(filename).suffix.lower()

    if ext in {".jpg", ".jpeg", ".tiff"}:
        return _extract_image_exif(data)
    elif ext in {".mp4", ".mov", ".avi", ".mkv"}:
        return _extract_video_metadata(data)
    return {}


def _extract_image_exif(data: bytes) -> dict[str, Any]:
    try:
        import exifread  # type: ignore
        tags = exifread.process_file(io.BytesIO(data), details=False)
        result: dict[str, Any] = {}

        fields_of_interest = [
            "Image Make", "Image Model", "Image Software",
            "EXIF DateTimeOriginal", "EXIF DateTimeDigitized",
            "GPS GPSLatitude", "GPS GPSLongitude",
            "EXIF ExifImageWidth", "EXIF ExifImageLength",
            "Image XResolution", "Image YResolution",
            "EXIF ColorSpace", "EXIF Flash",
            "EXIF FocalLength", "EXIF ISOSpeedRatings",
        ]
        for field in fields_of_interest:
            if field in tags:
                result[field.replace(" ", "_").lower()] = str(tags[field])

        # Flag suspicious absence of EXIF (AI images often have no EXIF)
        result["has_exif"] = len(result) > 0
        result["has_gps"]  = "gps_gpslatitude" in result
        result["has_camera_info"] = any(k in result for k in ["image_make", "image_model"])

        return result
    except Exception as exc:
        log.warning("exif_extraction_failed", error=str(exc))
        return {"has_exif": False, "error": str(exc)}


def _extract_video_metadata(data: bytes) -> dict[str, Any]:
    """Basic MP4/MOV metadata — creation time, encoder, resolution."""
    try:
        import struct
        meta: dict[str, Any] = {}

        # Check for ftyp box (MP4 signature)
        if len(data) >= 8:
            box_size = struct.unpack(">I", data[:4])[0]
            box_type = data[4:8].decode("ascii", errors="ignore")
            meta["container_type"] = box_type
            meta["has_ftyp"] = box_type in {"ftyp", "moov", "mdat"}

        return meta
    except Exception:
        return {}


# ── Device Fingerprinting ─────────────────────────────────────

def extract_device_fingerprint(exif: dict[str, Any]) -> dict[str, Any]:
    """
    Infer device fingerprint signals from EXIF metadata.
    AI-generated images typically lack camera model, sensor noise patterns, etc.
    """
    signals: dict[str, Any] = {
        "likely_camera_capture": False,
        "likely_ai_generated":   False,
        "suspicious_signals":    [],
    }

    if not exif.get("has_exif"):
        signals["suspicious_signals"].append("no_exif_data")
        signals["likely_ai_generated"] = True
        return signals

    if not exif.get("has_camera_info"):
        signals["suspicious_signals"].append("no_camera_model")

    if not exif.get("has_gps") and exif.get("has_camera_info"):
        # Camera captures often have GPS; absence alone isn't suspicious
        pass

    # AI images from diffusion models sometimes have software tags like
    # "Stable Diffusion", "DALL-E", "Midjourney"
    software = exif.get("image_software", "").lower()
    ai_software_markers = ["stable diffusion", "dall", "midjourney", "firefly",
                            "imagen", "flux", "automatic1111", "comfy"]
    for marker in ai_software_markers:
        if marker in software:
            signals["suspicious_signals"].append(f"ai_software_tag:{marker}")
            signals["likely_ai_generated"] = True

    if exif.get("has_camera_info") and not signals["suspicious_signals"]:
        signals["likely_camera_capture"] = True

    return signals


# ── SynthID-style Watermark Detection ────────────────────────

def detect_watermark(text: str | None = None, data: bytes | None = None) -> dict[str, Any]:
    """
    Step 29 + 83: Detect SynthID-style statistical watermarks.

    For text: test token bucket patterns against known watermark keys.
    For images: test DCT coefficient distributions in frequency domain.

    Returns:
        {
          "watermark_detected": bool,
          "confidence":         float [0,1],
          "method":             str,
        }
    """
    if text:
        return _detect_text_watermark(text)
    if data:
        return _detect_image_watermark(data)
    return {"watermark_detected": False, "confidence": 0.0, "method": "none"}


def _detect_text_watermark(text: str) -> dict[str, Any]:
    """
    Simplified SynthID text watermark detector.

    Real implementation: split tokens into two buckets (green/red) using
    a pseudorandom function keyed by context. AI text overuses green tokens.
    Here we approximate with a statistical test on word-hash distributions.

    Reference: Kirchenbauer et al. (2023) "A Watermark for Large Language Models"
    """
    import hashlib

    words = text.lower().split()
    if len(words) < 50:
        return {"watermark_detected": False, "confidence": 0.0, "method": "synthid_text", "reason": "too_short"}

    # Simulated green-list test: words whose hash mod 2 == 0 are "green"
    # A watermarked text will have significantly more green tokens than ~50%
    green_count = sum(1 for w in words if int(hashlib.md5(w.encode()).hexdigest(), 16) % 2 == 0)
    green_ratio = green_count / len(words)

    # Under null (no watermark): green_ratio ~ 0.5, std ~ sqrt(0.25/n)
    import math
    null_mean = 0.5
    null_std  = math.sqrt(0.25 / len(words))
    z_score   = (green_ratio - null_mean) / max(null_std, 1e-10)

    # z > 3.0 → p < 0.001 → likely watermarked
    # This is a simulation — real SynthID uses a cryptographic key
    detected = abs(z_score) > 3.0

    return {
        "watermark_detected": detected,
        "confidence":         min(abs(z_score) / 6.0, 1.0),
        "method":             "synthid_text_simulation",
        "z_score":            round(z_score, 4),
        "green_ratio":        round(green_ratio, 4),
        "note": "Simulation only. Real deployment requires the SynthID API key.",
    }


def _detect_image_watermark(data: bytes) -> dict[str, Any]:
    """
    Detect invisible watermarks in images via DCT frequency analysis.
    Real SynthID embeds signals in mid-frequency DCT coefficients.
    """
    try:
        import numpy as np
        from PIL import Image  # type: ignore

        img = Image.open(io.BytesIO(data)).convert("L")  # grayscale
        arr = np.array(img, dtype=np.float32)

        # Compute 2D DCT via FFT approximation
        fft = np.fft.fft2(arr)
        magnitude = np.abs(fft)

        # Check mid-frequency energy ratio
        h, w = magnitude.shape
        mid_h = slice(h // 8, h // 4)
        mid_w = slice(w // 8, w // 4)

        mid_energy   = float(np.sum(magnitude[mid_h, mid_w]))
        total_energy = float(np.sum(magnitude))
        ratio        = mid_energy / max(total_energy, 1.0)

        # Placeholder threshold — real detection needs the embedding key
        detected = ratio > 0.15

        return {
            "watermark_detected": detected,
            "confidence":         min(ratio / 0.20, 1.0),
            "method":             "dct_mid_frequency",
            "mid_energy_ratio":   round(ratio, 4),
            "note": "Approximation. Real SynthID image detection requires Google API.",
        }
    except Exception as exc:
        log.warning("image_watermark_detection_failed", error=str(exc))
        return {"watermark_detected": False, "confidence": 0.0, "method": "failed", "error": str(exc)}


# ── Main metadata runner ──────────────────────────────────────

async def run_metadata_extraction(
    job_id: str,
    s3_key: str | None,
    text: str | None,
    filename: str | None,
    content_type: str,
) -> dict[str, Any]:
    """
    Step 29: Full metadata extraction pipeline.
    Returns a metadata_signals dict that feeds into the Result Engine.
    """
    signals: dict[str, Any] = {
        "exif": {},
        "device_fingerprint": {},
        "watermark": {},
        "content_type": content_type,
    }

    try:
        if s3_key and filename:
            import boto3
            from ..core.config import get_settings
            settings = get_settings()

            kwargs = {
                "region_name":           settings.AWS_REGION,
                "aws_access_key_id":     settings.AWS_ACCESS_KEY_ID,
                "aws_secret_access_key": settings.AWS_SECRET_ACCESS_KEY,
            }
            if settings.S3_ENDPOINT_URL:
                kwargs["endpoint_url"] = settings.S3_ENDPOINT_URL

            s3 = boto3.client("s3", **kwargs)
            obj = s3.get_object(Bucket=settings.S3_BUCKET_UPLOADS, Key=s3_key)
            data = obj["Body"].read()

            signals["exif"]              = extract_exif(data, filename)
            signals["device_fingerprint"] = extract_device_fingerprint(signals["exif"])
            signals["watermark"]          = detect_watermark(data=data)

        if text:
            signals["watermark"] = detect_watermark(text=text)

    except Exception as exc:
        log.warning("metadata_extraction_error", job_id=job_id, error=str(exc))
        signals["error"] = str(exc)

    return signals
