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

# Max size per content type (bytes)
MAX_SIZES = {
    ContentType.TEXT:  10   * 1024 * 1024,   # 10 MB
    ContentType.IMAGE: 50   * 1024 * 1024,   # 50 MB
    ContentType.AUDIO: 200  * 1024 * 1024,   # 200 MB
    ContentType.VIDEO: 500  * 1024 * 1024,   # 500 MB
    ContentType.CODE:  5    * 1024 * 1024,   # 5 MB
}

EXT_TO_CONTENT_TYPE: dict[str, ContentType] = {
    **{ext: ContentType.TEXT  for ext in [".txt", ".md", ".pdf", ".docx", ".rtf"]},
    **{ext: ContentType.IMAGE for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"]},
    **{ext: ContentType.AUDIO for ext in [".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"]},
    **{ext: ContentType.VIDEO for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"]},
    **{ext: ContentType.CODE  for ext in [".py", ".js", ".ts", ".jsx", ".tsx", ".java",
                                           ".cpp", ".c", ".go", ".rs", ".rb", ".php", ".cs"]},
}


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
            detail=f"File too large. Limit for {content_type} is {limit_mb} MB.",
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

    # Read file into memory for hashing + size check
    data = await file.read()
    size = len(data)
    validate_file_size(size, content_type)

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
