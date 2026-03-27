"""
Shared S3/MinIO helper — used by all workers to fetch and store files.
Avoids duplicating boto3 logic across text/image/audio/video workers.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)


async def fetch_from_s3(s3_key: str) -> bytes:
    """Download file bytes from S3/MinIO."""
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
    log.info("s3_fetch_complete", key=s3_key, size=len(data))
    return data


async def upload_to_s3(key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    """Upload bytes to S3/MinIO. Returns the key."""
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
    s3.put_object(
        Bucket=settings.S3_BUCKET_UPLOADS,
        Key=key,
        Body=data,
        ContentType=content_type,
        ServerSideEncryption="AES256",
    )
    log.info("s3_upload_complete", key=key, size=len(data))
    return key
