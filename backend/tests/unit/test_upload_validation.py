"""
Unit tests for the upload MIME + size validation added to upload_service.py.

Covers:
  - valid upload (correct extension + MIME)
  - extension valid, MIME wrong  → 415
  - extension valid, MIME missing → 415
  - oversized file               → 413
  - boundary sizes (exactly at limit, +1 byte over)
  - MIME normalization (strips `;charset=...`)
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException, UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.models.models import ContentType  # noqa: E402
from app.services.upload_service import (  # noqa: E402
    MAX_SIZES,
    MIME_ALLOWLIST,
    MIME_SIGNATURES,
    SIGNATURE_PREFIX_BYTES,
    _normalize_mime,
    validate_file_signature,
    validate_file_size,
    validate_mime_type,
)


def _make_upload(filename: str, mime: str | None, body: bytes = b"") -> UploadFile:
    """Construct a FastAPI UploadFile mirroring what Starlette builds from a
    multipart request. `content_type` here plays the role of the HTTP
    `Content-Type` part header."""
    headers = {}
    if mime is not None:
        headers["content-type"] = mime
    # Starlette's UploadFile derives content_type from headers.
    return UploadFile(
        filename=filename,
        file=io.BytesIO(body),
        headers=headers,  # type: ignore[arg-type]
    )


# ── _normalize_mime ────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("text/plain", "text/plain"),
        ("Text/Plain", "text/plain"),
        ("text/plain; charset=utf-8", "text/plain"),
        ("  application/pdf  ", "application/pdf"),
        ("", None),
        (None, None),
        (";charset=utf-8", None),
    ],
)
def test_normalize_mime(raw, expected):
    assert _normalize_mime(raw) == expected


# ── validate_mime_type ─────────────────────────────────────────

def test_valid_mime_passes():
    f = _make_upload("doc.pdf", "application/pdf")
    validate_mime_type(f, ContentType.TEXT)  # no raise


def test_valid_mime_with_charset_parameter_passes():
    """text/plain; charset=utf-8 must be accepted as text/plain."""
    f = _make_upload("sample.txt", "text/plain; charset=utf-8")
    validate_mime_type(f, ContentType.TEXT)


def test_missing_mime_raises_415():
    """When the Content-Type header is absent, Starlette sets content_type to
    None — our validator must treat that as invalid."""
    f = _make_upload("doc.pdf", None)
    with pytest.raises(HTTPException) as exc:
        validate_mime_type(f, ContentType.TEXT)
    assert exc.value.status_code == 415


def test_empty_mime_raises_415():
    f = _make_upload("doc.pdf", "")
    with pytest.raises(HTTPException) as exc:
        validate_mime_type(f, ContentType.TEXT)
    assert exc.value.status_code == 415


def test_wrong_mime_for_correct_extension_raises_415():
    """Extension says image, but MIME says text/plain — spoof/mislabel
    case. Must reject with 415."""
    f = _make_upload("photo.png", "text/plain")
    with pytest.raises(HTTPException) as exc:
        validate_mime_type(f, ContentType.IMAGE)
    assert exc.value.status_code == 415
    assert "image" in exc.value.detail.lower()


def test_mime_for_wrong_content_type_raises_415():
    """A valid audio MIME rejected if we're validating for video."""
    f = _make_upload("file.mov", "audio/mpeg")
    with pytest.raises(HTTPException) as exc:
        validate_mime_type(f, ContentType.VIDEO)
    assert exc.value.status_code == 415


@pytest.mark.parametrize(
    "ct,mime",
    [
        (ContentType.IMAGE, "image/jpeg"),
        (ContentType.IMAGE, "image/png"),
        (ContentType.IMAGE, "image/webp"),
        (ContentType.AUDIO, "audio/mpeg"),
        (ContentType.AUDIO, "audio/wav"),
        (ContentType.VIDEO, "video/mp4"),
        (ContentType.VIDEO, "video/webm"),
        (ContentType.TEXT, "application/pdf"),
        (ContentType.TEXT, "text/plain"),
        (ContentType.CODE, "text/plain"),
        (ContentType.CODE, "text/x-python"),
    ],
)
def test_allowlist_membership_positive(ct, mime):
    f = _make_upload("f", mime)
    validate_mime_type(f, ct)  # no raise


# ── validate_file_size ─────────────────────────────────────────

def test_size_exactly_at_limit_passes():
    limit = MAX_SIZES[ContentType.IMAGE]
    validate_file_size(limit, ContentType.IMAGE)  # no raise


def test_size_one_byte_over_limit_raises_413():
    limit = MAX_SIZES[ContentType.IMAGE]
    with pytest.raises(HTTPException) as exc:
        validate_file_size(limit + 1, ContentType.IMAGE)
    assert exc.value.status_code == 413
    assert "10 MB" in exc.value.detail


@pytest.mark.parametrize(
    "ct,expected_mb",
    [
        (ContentType.TEXT, 1),
        (ContentType.CODE, 1),
        (ContentType.IMAGE, 10),
        (ContentType.AUDIO, 50),
        (ContentType.VIDEO, 200),
    ],
)
def test_max_sizes_match_spec(ct, expected_mb):
    assert MAX_SIZES[ct] == expected_mb * 1024 * 1024


def test_size_zero_passes():
    validate_file_size(0, ContentType.TEXT)  # no raise


# ── Coverage sanity: every ContentType has an allowlist ───────

def test_every_content_type_has_allowlist_and_size():
    for ct in ContentType:
        assert ct in MIME_ALLOWLIST, f"missing MIME allowlist for {ct}"
        assert MIME_ALLOWLIST[ct], f"empty allowlist for {ct}"
        assert ct in MAX_SIZES, f"missing size limit for {ct}"


# ── validate_file_signature ───────────────────────────────────
#
# These tests exercise the third validation layer: magic-byte checks.
# For each case we craft a minimal byte prefix that either matches or
# deliberately mismatches a real file signature. We never need real
# image/audio payloads — the validator only inspects the first 32 bytes.

# Known-good headers, padded with zeros to fill SIGNATURE_PREFIX_BYTES so
# offset-based rules (ftyp at 4, WEBP/WAVE/AVI at 8) all resolve.
_PNG_HEADER  = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
_JPEG_HEADER = b"\xff\xd8\xff\xe0" + b"\x00" * 28
_GIF89_HEADER = b"GIF89a" + b"\x00" * 26
_WEBP_HEADER = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 16
_WAV_HEADER  = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 16
_AVI_HEADER  = b"RIFF" + b"\x00\x00\x00\x00" + b"AVI " + b"\x00" * 16
_BMP_HEADER  = b"BM" + b"\x00" * 30
_TIFF_HEADER = b"II*\x00" + b"\x00" * 28
_MP3_ID3_HEADER    = b"ID3\x03\x00" + b"\x00" * 27
_MP3_FRAMESYNC_FB  = b"\xff\xfb\x90\x00" + b"\x00" * 28
_FLAC_HEADER = b"fLaC" + b"\x00" * 28
_OGG_HEADER  = b"OggS" + b"\x00" * 28
_MP4_HEADER  = b"\x00\x00\x00\x18" + b"ftypmp42" + b"\x00" * 20
_MKV_HEADER  = b"\x1a\x45\xdf\xa3" + b"\x00" * 28
_FLV_HEADER  = b"FLV\x01" + b"\x00" * 28
_PDF_HEADER  = b"%PDF-1.7" + b"\x00" * 24
_DOCX_HEADER = b"PK\x03\x04" + b"\x00" * 28   # ZIP local file header
_RTF_HEADER  = b"{\\rtf1" + b"\x00" * 26


def _upload_with_data(filename: str, mime: str, data: bytes) -> UploadFile:
    """Like _make_upload, but carries a `data` body for signature checks."""
    return UploadFile(
        filename=filename,
        file=io.BytesIO(data),
        headers={"content-type": mime},  # type: ignore[arg-type]
    )


# ── Positive cases: valid MIME + valid signature ──────────────

@pytest.mark.parametrize(
    "ct,mime,data",
    [
        (ContentType.IMAGE, "image/png",  _PNG_HEADER),
        (ContentType.IMAGE, "image/jpeg", _JPEG_HEADER),
        (ContentType.IMAGE, "image/gif",  _GIF89_HEADER),
        (ContentType.IMAGE, "image/webp", _WEBP_HEADER),
        (ContentType.IMAGE, "image/bmp",  _BMP_HEADER),
        (ContentType.IMAGE, "image/tiff", _TIFF_HEADER),
        (ContentType.AUDIO, "audio/mpeg", _MP3_ID3_HEADER),
        (ContentType.AUDIO, "audio/mpeg", _MP3_FRAMESYNC_FB),
        (ContentType.AUDIO, "audio/wav",  _WAV_HEADER),
        (ContentType.AUDIO, "audio/flac", _FLAC_HEADER),
        (ContentType.AUDIO, "audio/ogg",  _OGG_HEADER),
        (ContentType.AUDIO, "audio/mp4",  _MP4_HEADER),
        (ContentType.VIDEO, "video/mp4",  _MP4_HEADER),
        (ContentType.VIDEO, "video/quicktime", _MP4_HEADER),
        (ContentType.VIDEO, "video/x-msvideo",  _AVI_HEADER),
        (ContentType.VIDEO, "video/x-matroska", _MKV_HEADER),
        (ContentType.VIDEO, "video/webm",  _MKV_HEADER),
        (ContentType.VIDEO, "video/x-flv", _FLV_HEADER),
        (ContentType.TEXT,  "application/pdf", _PDF_HEADER),
        (ContentType.TEXT,  "application/vnd.openxmlformats-officedocument.wordprocessingml.document", _DOCX_HEADER),
        (ContentType.TEXT,  "application/rtf", _RTF_HEADER),
    ],
)
def test_signature_valid_passes(ct, mime, data):
    f = _upload_with_data("f", mime, data)
    validate_file_signature(data, f, ct)  # no raise


# ── Negative cases: correct MIME but wrong signature ──────────

def test_png_mime_with_pdf_bytes_raises_415():
    f = _upload_with_data("fake.png", "image/png", _PDF_HEADER)
    with pytest.raises(HTTPException) as exc:
        validate_file_signature(_PDF_HEADER, f, ContentType.IMAGE)
    assert exc.value.status_code == 415
    assert "image/png" in exc.value.detail


def test_jpeg_mime_with_png_bytes_raises_415():
    f = _upload_with_data("fake.jpg", "image/jpeg", _PNG_HEADER)
    with pytest.raises(HTTPException) as exc:
        validate_file_signature(_PNG_HEADER, f, ContentType.IMAGE)
    assert exc.value.status_code == 415


def test_pdf_mime_with_zip_bytes_raises_415():
    f = _upload_with_data("fake.pdf", "application/pdf", _DOCX_HEADER)
    with pytest.raises(HTTPException) as exc:
        validate_file_signature(_DOCX_HEADER, f, ContentType.TEXT)
    assert exc.value.status_code == 415


def test_mp4_mime_with_wrong_box_offset_raises_415():
    """ftyp must appear at offset 4, not at offset 0."""
    bogus = b"ftyp" + b"\x00" * 28  # ftyp at offset 0, not 4
    f = _upload_with_data("fake.mp4", "video/mp4", bogus)
    with pytest.raises(HTTPException) as exc:
        validate_file_signature(bogus, f, ContentType.VIDEO)
    assert exc.value.status_code == 415


def test_wav_with_only_riff_no_wave_raises_415():
    """RIFF header alone isn't enough — WAVE magic at offset 8 is required."""
    partial = b"RIFF" + b"\x00\x00\x00\x00" + b"AVI " + b"\x00" * 16
    f = _upload_with_data("fake.wav", "audio/wav", partial)
    with pytest.raises(HTTPException) as exc:
        validate_file_signature(partial, f, ContentType.AUDIO)
    assert exc.value.status_code == 415


def test_docx_mime_with_plain_text_raises_415():
    body = b"Hello, world!" + b"\x00" * 19
    f = _upload_with_data("fake.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", body)
    with pytest.raises(HTTPException) as exc:
        validate_file_signature(body, f, ContentType.TEXT)
    assert exc.value.status_code == 415


# ── Edge cases ────────────────────────────────────────────────

def test_empty_file_fails_signature_check():
    f = _upload_with_data("empty.png", "image/png", b"")
    with pytest.raises(HTTPException) as exc:
        validate_file_signature(b"", f, ContentType.IMAGE)
    assert exc.value.status_code == 415


def test_file_shorter_than_signature_fails():
    """2 bytes can't satisfy an 8-byte PNG signature."""
    partial = b"\x89P"
    f = _upload_with_data("short.png", "image/png", partial)
    with pytest.raises(HTTPException) as exc:
        validate_file_signature(partial, f, ContentType.IMAGE)
    assert exc.value.status_code == 415


def test_file_shorter_than_mp4_offset_fails():
    """ftyp at offset 4 needs 8 bytes minimum; 3 bytes fail."""
    partial = b"\x00\x00\x00"
    f = _upload_with_data("short.mp4", "video/mp4", partial)
    with pytest.raises(HTTPException) as exc:
        validate_file_signature(partial, f, ContentType.VIDEO)
    assert exc.value.status_code == 415


def test_exactly_signature_length_passes():
    """File exactly as long as the PNG signature (8 bytes) must pass."""
    exact = b"\x89PNG\r\n\x1a\n"
    f = _upload_with_data("tiny.png", "image/png", exact)
    validate_file_signature(exact, f, ContentType.IMAGE)  # no raise


# ── MIMEs without signature rules must bypass the check ───────

@pytest.mark.parametrize(
    "mime,ct,data",
    [
        ("text/plain",        ContentType.TEXT, b""),
        ("text/plain",        ContentType.CODE, b"def hello(): pass"),
        ("text/x-python",     ContentType.CODE, b"\x00\x00\x00\x00"),
        ("application/javascript", ContentType.CODE, b"anything"),
        ("text/markdown",     ContentType.TEXT, b"# hello"),
    ],
)
def test_mimes_without_signature_skip_check(mime, ct, data):
    assert mime not in MIME_SIGNATURES  # invariant for the test
    f = _upload_with_data("f", mime, data)
    validate_file_signature(data, f, ct)  # no raise, no error


def test_missing_mime_bypasses_signature_check():
    """Signature stage is only reached after validate_mime_type passes.
    If it's called with a missing MIME (defensive), it should no-op rather
    than double-raising — 415 is already owned by validate_mime_type."""
    f = UploadFile(filename="f", file=io.BytesIO(b"\x00" * 32), headers={})
    validate_file_signature(b"\x00" * 32, f, ContentType.IMAGE)  # no raise


# ── Constant-time invariant ───────────────────────────────────

def test_signature_check_only_reads_prefix():
    """validate_file_signature must not depend on bytes past SIGNATURE_PREFIX_BYTES."""
    good_head = _PNG_HEADER[:SIGNATURE_PREFIX_BYTES]
    huge = good_head + b"garbage" * 1_000_000
    f = _upload_with_data("big.png", "image/png", huge)
    validate_file_signature(huge, f, ContentType.IMAGE)  # no raise
    assert len(huge) > SIGNATURE_PREFIX_BYTES * 1000
