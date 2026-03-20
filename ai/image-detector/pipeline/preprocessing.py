"""
Step 65: Image preprocessing pipeline.

Handles all common image formats (JPEG, PNG, WebP, GIF, BMP, TIFF).
Produces a canonical ImageRecord consumed by all feature extractors
and classifiers downstream.

Key design decisions:
  - Preserve original resolution for frequency-domain analysis (FFT needs it)
  - Also produce a 224×224 normalised tensor for CNN/ViT classifiers
  - Extract multiple crops at different scales for robustness
  - Keep the raw uint8 for EXIF-sensitive operations
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

log = structlog.get_logger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif",
                         ".bmp", ".tiff", ".tif", ".heic", ".avif"}

# CNN input size
CNN_INPUT_SIZE = 224

# ImageNet normalisation constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class ImageRecord:
    """Canonical image representation consumed by all downstream modules."""
    # Raw pixel data
    pixels:        np.ndarray    # [H, W, 3] uint8 RGB — original size
    width:         int
    height:        int
    filename:      str
    file_format:   str           # "jpeg", "png", "webp", etc.
    file_size:     int           # original file size in bytes
    color_mode:    str           # "RGB", "RGBA", "L", "P" (before conversion)

    # Preprocessed for classifiers
    tensor_224:    np.ndarray    # [3, 224, 224] float32 normalised (ImageNet stats)
    tensor_299:    np.ndarray    # [3, 299, 299] for XceptionNet

    # Quality metadata
    jpeg_quality:  int | None    # estimated JPEG quality factor (None if not JPEG)
    has_alpha:     bool          # original image had transparency
    is_animated:   bool          # GIF or APNG with multiple frames


def load_image(data: bytes, filename: str) -> ImageRecord:
    """
    Decode raw image bytes into a canonical ImageRecord.

    Decoding chain: Pillow → (fallback) imageio
    Always converts to RGB regardless of source color mode.
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {ext}")

    try:
        return _load_with_pillow(data, filename, ext)
    except Exception as exc:
        log.warning("pillow_failed", error=str(exc))
        return _load_with_imageio(data, filename, ext)


def _load_with_pillow(data: bytes, filename: str, ext: str) -> ImageRecord:
    from PIL import Image, ExifTags  # type: ignore

    img = Image.open(io.BytesIO(data))

    color_mode  = img.mode
    is_animated = getattr(img, "is_animated", False)
    has_alpha   = color_mode in {"RGBA", "LA", "PA"}

    # Auto-rotate based on EXIF orientation
    try:
        img = _apply_exif_rotation(img)
    except Exception:
        pass

    # Convert to RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    pixels = np.array(img, dtype=np.uint8)   # [H, W, 3]
    h, w   = pixels.shape[:2]

    # Estimate JPEG quality
    jpeg_quality = _estimate_jpeg_quality(data) if ext in {".jpg", ".jpeg"} else None

    return ImageRecord(
        pixels=pixels,
        width=w,
        height=h,
        filename=filename,
        file_format=ext.lstrip("."),
        file_size=len(data),
        color_mode=color_mode,
        tensor_224=_to_tensor(pixels, CNN_INPUT_SIZE),
        tensor_299=_to_tensor(pixels, 299),
        jpeg_quality=jpeg_quality,
        has_alpha=has_alpha,
        is_animated=is_animated,
    )


def _load_with_imageio(data: bytes, filename: str, ext: str) -> ImageRecord:
    import imageio  # type: ignore
    img = imageio.imread(io.BytesIO(data))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]   # drop alpha
    pixels = img.astype(np.uint8)
    h, w   = pixels.shape[:2]
    return ImageRecord(
        pixels=pixels, width=w, height=h,
        filename=filename, file_format=ext.lstrip("."),
        file_size=len(data), color_mode="RGB",
        tensor_224=_to_tensor(pixels, CNN_INPUT_SIZE),
        tensor_299=_to_tensor(pixels, 299),
        jpeg_quality=None, has_alpha=False, is_animated=False,
    )


def _apply_exif_rotation(img):
    """Correct image orientation from EXIF data."""
    from PIL import Image, ExifTags  # type: ignore
    try:
        exif = img._getexif()
        if exif:
            orientation_key = next(
                k for k, v in ExifTags.TAGS.items() if v == "Orientation"
            )
            orientation = exif.get(orientation_key)
            rotations = {3: 180, 6: 270, 8: 90}
            if orientation in rotations:
                img = img.rotate(rotations[orientation], expand=True)
    except Exception:
        pass
    return img


def _to_tensor(pixels: np.ndarray, size: int) -> np.ndarray:
    """
    Resize to (size × size), normalise with ImageNet stats,
    return [3, size, size] float32.
    """
    from PIL import Image  # type: ignore
    pil   = Image.fromarray(pixels).resize((size, size), Image.BILINEAR)
    arr   = np.array(pil, dtype=np.float32) / 255.0   # [H, W, 3] in [0,1]
    arr   = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1).astype(np.float32)   # [3, H, W]


def _estimate_jpeg_quality(data: bytes) -> int | None:
    """
    Estimate JPEG quality factor from quantisation tables.
    Returns None if tables cannot be parsed.
    """
    try:
        from PIL import Image  # type: ignore
        img = Image.open(io.BytesIO(data))
        qt  = img.quantization
        if not qt:
            return None
        # Average the luma quantisation table values
        luma_table = qt.get(0, qt.get(1, None))
        if luma_table is None:
            return None
        avg_q = float(np.mean(list(luma_table.values())
                              if isinstance(luma_table, dict)
                              else luma_table))
        # Rough quality estimate: lower avg_q = higher quality
        quality = int(np.clip(100 - avg_q * 0.5, 1, 100))
        return quality
    except Exception:
        return None
