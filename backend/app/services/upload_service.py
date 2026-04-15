"""
Step 26: Upload Service.
Validates uploaded files, computes SHA-256 content hash, stores to S3/R2.
Returns S3 key for the processing queue.
"""

from __future__ import annotations

import hashlib
import mimetypes
import uuid
from pathlib import Path

import boto3
import structlog
from fastapi import HTTPException, UploadFile, status

from ..core.config import get_settings
from ..models.models import ContentType

log = structlog.get_logger(__name__)

# Max size per content type (bytes).
#
# These caps are the SINGLE SOURCE OF TRUTH for what the file-upload
# endpoint actually accepts. They are capped at 10 MB across the board
# to match the global request-body limit enforced by main.py's
# BodySizeLimit middleware — anything larger is rejected before it even
# reaches this validator, so larger numbers here would be a lie.
#
# If large-media uploads (50+ MB audio, 200+ MB video) become a real
# product requirement, the fix is NOT to raise these numbers alone — it
# requires: (1) raising the body-size middleware cap (or adding a
# per-route override), (2) switching `store_upload` from
# `await file.read()` to a streaming SHA-256 + streaming boto3 upload,
# and (3) validating magic bytes on the streamed prefix. Until that
# work is scoped, image/audio/video share the same 10 MB ceiling.
#
# Text and code are tightened further to 1 MB because the text
# detector's 100k-character processing guardrail makes anything larger
# pointless and a smaller cap shrinks the memory-exhaustion attack
# surface.
MAX_SIZES = {
    ContentType.TEXT:  1  * 1024 * 1024,   # 1 MB
    ContentType.CODE:  1  * 1024 * 1024,   # 1 MB
    ContentType.IMAGE: 10 * 1024 * 1024,   # 10 MB (= body cap)
    ContentType.AUDIO: 10 * 1024 * 1024,   # 10 MB (= body cap; was 50 MB but body cap rejected first)
    ContentType.VIDEO: 10 * 1024 * 1024,   # 10 MB (= body cap; was 200 MB but body cap rejected first)
}

EXT_TO_CONTENT_TYPE: dict[str, ContentType] = {
    **{ext: ContentType.TEXT  for ext in [".txt", ".md", ".pdf", ".docx", ".rtf"]},
    **{ext: ContentType.IMAGE for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"]},
    **{ext: ContentType.AUDIO for ext in [".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"]},
    **{ext: ContentType.VIDEO for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"]},
    **{ext: ContentType.CODE  for ext in [".py", ".js", ".ts", ".jsx", ".tsx", ".java",
                                           ".cpp", ".c", ".go", ".rs", ".rb", ".php", ".cs"]},
}

# Allowlist of acceptable MIME types per inferred content type. Extension
# still drives routing (`detect_content_type`); MIME is a second layer that
# catches mislabelled or spoofed extensions. Clients — especially browsers —
# send `text/plain` for many code languages, so that entry is shared.
MIME_ALLOWLIST: dict[ContentType, set[str]] = {
    ContentType.TEXT: {
        "text/plain",
        "text/markdown",
        "text/x-markdown",
        "text/rtf",
        "application/rtf",
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    },
    ContentType.CODE: {
        # Browsers rarely send language-specific MIMEs for source code;
        # text/plain + a handful of well-known ones cover the real cases.
        "text/plain",
        "text/x-python",
        "application/x-python-code",
        "text/javascript",
        "application/javascript",
        "application/x-javascript",
        "text/typescript",
        "application/typescript",
        "text/x-java-source",
        "text/x-c",
        "text/x-c++",
        "text/x-csrc",
        "text/x-chdr",
        "text/x-go",
        "text/x-rustsrc",
        "text/x-ruby",
        "application/x-ruby",
        "application/x-httpd-php",
        "text/x-php",
        "text/x-csharp",
    },
    ContentType.IMAGE: {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif",
        "image/bmp",
        "image/tiff",
    },
    ContentType.AUDIO: {
        "audio/mpeg",      # mp3
        "audio/mp3",
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/flac",
        "audio/x-flac",
        "audio/mp4",       # m4a
        "audio/x-m4a",
        "audio/ogg",
        "audio/aac",
    },
    ContentType.VIDEO: {
        "video/mp4",
        "video/quicktime",       # mov
        "video/x-msvideo",       # avi
        "video/x-matroska",      # mkv
        "video/webm",
        "video/x-flv",
    },
}


def _normalize_mime(raw: str | None) -> str | None:
    """Strip parameters and whitespace from a Content-Type header value.

    `text/plain; charset=utf-8` → `text/plain`. Returns None for empty input
    so callers can treat missing and unparseable MIME identically.
    """
    if not raw:
        return None
    main = raw.split(";", 1)[0].strip().lower()
    return main or None


# ── File-signature (magic-byte) allowlist ─────────────────────
# A "rule" is a list of (offset, expected_bytes) pairs — all must match.
# A MIME passes signature validation if ANY of its rules matches.
# MIMEs not present here (plain text, source code, etc.) skip signature
# validation because they have no reliable magic bytes.
#
# Only the first SIGNATURE_PREFIX_BYTES of the upload are inspected — checks
# are constant-time and operate on the already-buffered `data`.
SIGNATURE_PREFIX_BYTES = 32

_SigRule = list[tuple[int, bytes]]

MIME_SIGNATURES: dict[str, list[_SigRule]] = {
    # ── Images ──
    "image/png":  [[(0, b"\x89PNG\r\n\x1a\n")]],
    "image/jpeg": [[(0, b"\xff\xd8\xff")]],
    "image/gif":  [[(0, b"GIF87a")], [(0, b"GIF89a")]],
    "image/webp": [[(0, b"RIFF"), (8, b"WEBP")]],
    "image/bmp":  [[(0, b"BM")]],
    "image/tiff": [[(0, b"II*\x00")], [(0, b"MM\x00*")]],

    # ── Audio ──
    # MP3: either an ID3 header or an MPEG audio frame sync (first 11 bits =
    # 1, i.e. FF Ex / FF Fx for the common MPEG-1/2 Layer III profiles).
    "audio/mpeg": [[(0, b"ID3")], [(0, b"\xff\xfb")], [(0, b"\xff\xf3")], [(0, b"\xff\xf2")]],
    "audio/mp3":  [[(0, b"ID3")], [(0, b"\xff\xfb")], [(0, b"\xff\xf3")], [(0, b"\xff\xf2")]],
    "audio/wav":   [[(0, b"RIFF"), (8, b"WAVE")]],
    "audio/x-wav": [[(0, b"RIFF"), (8, b"WAVE")]],
    "audio/wave":  [[(0, b"RIFF"), (8, b"WAVE")]],
    "audio/flac":   [[(0, b"fLaC")]],
    "audio/x-flac": [[(0, b"fLaC")]],
    "audio/ogg":   [[(0, b"OggS")]],
    "audio/mp4":   [[(4, b"ftyp")]],
    "audio/x-m4a": [[(4, b"ftyp")]],

    # ── Video ──
    "video/mp4":        [[(4, b"ftyp")]],
    "video/quicktime":  [[(4, b"ftyp")]],   # .mov also uses ftyp box
    "video/x-msvideo":  [[(0, b"RIFF"), (8, b"AVI ")]],
    "video/x-matroska": [[(0, b"\x1a\x45\xdf\xa3")]],
    "video/webm":       [[(0, b"\x1a\x45\xdf\xa3")]],
    "video/x-flv":      [[(0, b"FLV")]],

    # ── Documents / archives ──
    "application/pdf": [[(0, b"%PDF-")]],
    "application/msword": [[(0, b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")]],  # OLE CFB
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        [[(0, b"PK\x03\x04")]],
    "application/rtf": [[(0, b"{\\rtf")]],
    "text/rtf":        [[(0, b"{\\rtf")]],
}


def _rule_matches(head: bytes, rule: _SigRule) -> bool:
    for offset, sig in rule:
        end = offset + len(sig)
        if end > len(head) or head[offset:end] != sig:
            return False
    return True


def validate_file_signature(
    data: bytes,
    file: UploadFile,
    content_type: ContentType,
) -> None:
    """Raise 415 if the file's leading bytes don't match the declared MIME.

    Operates on the already-buffered `data` — no additional I/O. Only the
    first SIGNATURE_PREFIX_BYTES are inspected, so cost is constant regardless
    of file size. MIMEs without a well-defined magic sequence (plain text,
    source code) bypass this layer entirely — extension + MIME allowlist are
    sufficient for those.
    """
    mime = _normalize_mime(file.content_type)
    rules = MIME_SIGNATURES.get(mime) if mime else None
    if not rules:
        return  # Not applicable for this MIME — skip by design.

    head = data[:SIGNATURE_PREFIX_BYTES]
    if any(_rule_matches(head, r) for r in rules):
        return

    # Log only the filename + first 16 bytes hex — never the file body.
    log.warning(
        "upload_signature_mismatch",
        filename=file.filename,
        content_type=content_type.value,
        provided_mime=mime,
        head_hex=head[:16].hex(),
    )
    raise HTTPException(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        detail=(
            f"File signature does not match declared MIME type {mime!r}. "
            f"The file's contents look different from a valid "
            f"{content_type.value} upload."
        ),
    )


def _get_s3_client():
    settings = get_settings()
    kwargs = {
        "region_name": settings.AWS_REGION,
        "aws_access_key_id":     settings.AWS_ACCESS_KEY_ID,
        "aws_secret_access_key": settings.AWS_SECRET_ACCESS_KEY,
    }
    if settings.S3_ENDPOINT_URL:
        kwargs["endpoint_url"] = settings.S3_ENDPOINT_URL
    return boto3.client("s3", **kwargs)


def detect_content_type(filename: str) -> ContentType:
    """Infer content type from file extension."""
    ext = Path(filename).suffix.lower()
    ct  = EXT_TO_CONTENT_TYPE.get(ext)
    if ct is None:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file extension: {ext}",
        )
    return ct


def validate_file_size(size_bytes: int, content_type: ContentType) -> None:
    """Raise 413 if file exceeds the limit for its content type."""
    limit = MAX_SIZES[content_type]
    if size_bytes > limit:
        limit_mb = limit // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Limit for {content_type.value} is {limit_mb} MB.",
        )


def validate_mime_type(file: UploadFile, content_type: ContentType) -> None:
    """Raise 415 if the file's declared MIME type is missing or not on the
    allowlist for its inferred content type.

    Extension-based routing still runs first (`detect_content_type`). This is
    an additional layer that rejects mislabelled or spoofed extensions whose
    MIME contradicts the detector that would process them.
    """
    raw_mime = file.content_type
    mime = _normalize_mime(raw_mime)
    allowed = MIME_ALLOWLIST.get(content_type, set())

    if mime is None:
        log.warning(
            "upload_mime_missing",
            filename=file.filename,
            content_type=content_type.value,
        )
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Missing Content-Type for upload. "
                f"Expected one of: {sorted(allowed)}"
            ),
        )

    if mime not in allowed:
        log.warning(
            "upload_mime_mismatch",
            filename=file.filename,
            content_type=content_type.value,
            provided_mime=mime,
        )
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"MIME type {mime!r} is not allowed for {content_type.value}. "
                f"Expected one of: {sorted(allowed)}"
            ),
        )


async def store_upload(
    file: UploadFile,
    user_id: str,
) -> tuple[str, str, ContentType, int]:
    """
    Validate, hash, and store an uploaded file to S3.

    Returns:
        (s3_key, sha256_hash, content_type, file_size_bytes)
    """
    settings = get_settings()

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )

    content_type = detect_content_type(file.filename)

    # MIME allowlist check — guards against spoofed/mislabelled extensions.
    # Runs before the read so a 415 short-circuits large uploads.
    validate_mime_type(file, content_type)

    # Read file into memory for hashing + size check
    data = await file.read()
    size = len(data)
    validate_file_size(size, content_type)

    # Magic-byte signature check — third layer after extension + MIME.
    # Uses the already-buffered `data`; inspects only the first 32 bytes.
    validate_file_signature(data, file, content_type)

    # SHA-256 content hash (Step 87 — used for integrity + deduplication)
    sha256 = hashlib.sha256(data).hexdigest()

    # Construct S3 key: uploads/{user_id}/{uuid}/{filename}
    job_id  = str(uuid.uuid4())
    safe_name = Path(file.filename).name.replace(" ", "_")
    s3_key  = f"uploads/{user_id}/{job_id}/{safe_name}"

    try:
        s3 = _get_s3_client()
        s3.put_object(
            Bucket=settings.S3_BUCKET_UPLOADS,
            Key=s3_key,
            Body=data,
            ContentType=file.content_type or "application/octet-stream",
            Metadata={
                "user-id":      user_id,
                "job-id":       job_id,
                "sha256":       sha256,
                "content-type": content_type.value,
            },
            ServerSideEncryption="AES256",   # Step 88: encryption at rest
        )
    except Exception as exc:
        log.error("s3_upload_failed", error=str(exc), s3_key=s3_key)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="File storage temporarily unavailable",
        ) from exc

    log.info(
        "file_stored",
        s3_key=s3_key,
        size=size,
        sha256=sha256[:16] + "...",
        content_type=content_type,
    )
    return s3_key, sha256, content_type, size


async def get_presigned_url(s3_key: str, expiry_seconds: int = 3600) -> str:
    """Generate a presigned S3 URL for secure file download."""
    settings = get_settings()
    s3 = _get_s3_client()
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.S3_BUCKET_UPLOADS, "Key": s3_key},
        ExpiresIn=expiry_seconds,
    )


async def delete_file(s3_key: str) -> None:
    """Delete a file from S3 (for GDPR deletion and retention policy)."""
    settings = get_settings()
    s3 = _get_s3_client()
    s3.delete_object(Bucket=settings.S3_BUCKET_UPLOADS, Key=s3_key)
    log.info("file_deleted", s3_key=s3_key)
