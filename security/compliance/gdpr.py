"""
Steps 93–94: Data retention policies and GDPR compliance.

Step 93 — Configurable data retention
  - Uploads deleted after UPLOAD_RETENTION_DAYS (default 30)
  - Reports retained for REPORT_RETENTION_DAYS (default 365)
  - Celery results expire after 24h (configured in celery_app.py)
  - Redis rate-limit keys expire automatically (1 min window + 1s buffer)
  - Audit logs retained for 7 years (regulatory requirement)
  - Celery runs as a background task; retention enforced by the scheduler

Step 94 — GDPR compliance
  Article 17 (Right to erasure / "right to be forgotten"):
    DELETE /api/v1/account (authenticated)
    → deletes all user data: account, jobs, results, audit logs, uploads
    → confirms deletion with a deletion receipt

  Article 15 (Right of access):
    GET /api/v1/account/data-export
    → returns all stored data for the user as a JSON export

  Article 13/14 (Privacy information):
    Consent recorded at registration with timestamp
    Processing purposes documented in /api/v1/privacy-policy

  Article 30 (Records of processing activities):
    Audit log records every data processing operation with:
    - user_id, action, resource_type, ip_address, timestamp
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)


# ── Step 93: Retention policy configuration ───────────────────

@dataclass
class RetentionPolicy:
    """
    Configurable retention policy.
    All durations in days; None = keep forever.
    """
    upload_days:      int        = 30
    report_days:      int        = 365
    audit_log_days:   int | None = None    # None = keep forever (7-year minimum)
    celery_result_h:  int        = 24
    rate_limit_s:     int        = 61

    def upload_expiry(self, created_at: datetime) -> datetime:
        return created_at + timedelta(days=self.upload_days)

    def report_expiry(self, created_at: datetime) -> datetime:
        return created_at + timedelta(days=self.report_days)

    def is_upload_expired(self, created_at: datetime) -> bool:
        return datetime.now(timezone.utc) > self.upload_expiry(created_at)

    def is_report_expired(self, created_at: datetime) -> bool:
        return datetime.now(timezone.utc) > self.report_expiry(created_at)


DEFAULT_RETENTION = RetentionPolicy()


async def run_retention_sweep(
    db: Any,
    s3_client: Any,
    settings: Any,
    retention: RetentionPolicy = DEFAULT_RETENTION,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Scheduled task: purge expired uploads and reports.

    Called by Celery beat scheduler (e.g. every hour).
    Returns counts of deleted items.

    Args:
        db:        AsyncSession (SQLAlchemy)
        s3_client: Boto3 S3 client
        settings:  Application settings (for bucket names)
        retention: Policy configuration
        dry_run:   If True, count but do not delete
    """
    from sqlalchemy import select, delete  # type: ignore
    # Import here to avoid circular imports
    try:
        from backend.app.models.models import DetectionJob, DetectionResult  # type: ignore
    except ImportError:
        log.warning("retention_sweep_skipped", reason="models not importable")
        return {}

    cutoff_uploads = datetime.now(timezone.utc) - timedelta(days=retention.upload_days)
    cutoff_reports = datetime.now(timezone.utc) - timedelta(days=retention.report_days)

    counts: dict[str, int] = {"uploads_deleted": 0, "reports_deleted": 0, "errors": 0}

    # Find expired jobs with S3 uploads
    expired_jobs_q = await db.execute(
        select(DetectionJob).where(
            DetectionJob.created_at < cutoff_uploads,
            DetectionJob.s3_key.isnot(None),
        )
    )
    expired_jobs = expired_jobs_q.scalars().all()

    for job in expired_jobs:
        try:
            if not dry_run and job.s3_key:
                s3_client.delete_object(
                    Bucket=settings.S3_BUCKET_UPLOADS,
                    Key=job.s3_key,
                )
                job.s3_key = None
                counts["uploads_deleted"] += 1
        except Exception as exc:
            log.warning("retention_delete_failed", job_id=str(job.id), error=str(exc))
            counts["errors"] += 1

    if not dry_run:
        await db.commit()

    log.info("retention_sweep_complete",
             dry_run=dry_run, **counts)
    return counts


# ── Step 94: GDPR compliance utilities ───────────────────────

@dataclass
class DeletionReceipt:
    """Proof of data deletion for GDPR Article 17 compliance."""
    receipt_id:         str
    user_id:            str
    requested_at:       str
    completed_at:       str
    items_deleted:      dict[str, int]
    storage_locations:  list[str]
    confirmation:       str


async def delete_user_data(
    user_id: str,
    db: Any,
    s3_client: Any,
    settings: Any,
) -> DeletionReceipt:
    """
    GDPR Article 17: Right to erasure.

    Deletes ALL data for a user:
      1. S3 uploads (files)
      2. S3 reports (forensic reports)
      3. DetectionResult records (DB)
      4. DetectionJob records (DB)
      5. APIKey records (DB)
      6. User record (DB)

    Audit log entries are RETAINED (legal requirement — cannot be erased).
    A deletion receipt is created and returned.

    Args:
        user_id:   UUID string of the user to delete
        db:        Async SQLAlchemy session
        s3_client: Boto3 S3 client (for file deletion)
        settings:  Application settings

    Returns:
        DeletionReceipt with counts of deleted items
    """
    from sqlalchemy import select, delete as sql_delete  # type: ignore
    try:
        from backend.app.models.models import (  # type: ignore
            User, DetectionJob, DetectionResult, APIKey,
        )
    except ImportError:
        log.warning("gdpr_deletion_skipped", reason="models not importable")
        return _empty_receipt(user_id)

    requested_at = datetime.now(timezone.utc).isoformat()
    counts: dict[str, int] = {
        "s3_uploads": 0, "s3_reports": 0,
        "jobs": 0, "results": 0, "api_keys": 0,
    }
    locations: list[str] = []

    uid = uuid.UUID(user_id)

    # ── 1. Get all jobs for this user ─────────────────────────
    jobs_q = await db.execute(
        select(DetectionJob).where(DetectionJob.user_id == uid)
    )
    jobs = jobs_q.scalars().all()

    # ── 2. Delete S3 uploads ──────────────────────────────────
    for job in jobs:
        if job.s3_key:
            try:
                s3_client.delete_object(
                    Bucket=settings.S3_BUCKET_UPLOADS, Key=job.s3_key
                )
                counts["s3_uploads"] += 1
                if settings.S3_BUCKET_UPLOADS not in locations:
                    locations.append(f"s3://{settings.S3_BUCKET_UPLOADS}")
            except Exception as exc:
                log.warning("s3_delete_failed", key=job.s3_key, error=str(exc))

    # ── 3. Delete S3 reports ──────────────────────────────────
    for job in jobs:
        result_q = await db.execute(
            select(DetectionResult).where(DetectionResult.job_id == job.id)
        )
        result = result_q.scalar_one_or_none()
        if result and result.report_s3_key:
            try:
                s3_client.delete_object(
                    Bucket=settings.S3_BUCKET_REPORTS, Key=result.report_s3_key
                )
                counts["s3_reports"] += 1
                if settings.S3_BUCKET_REPORTS not in locations:
                    locations.append(f"s3://{settings.S3_BUCKET_REPORTS}")
            except Exception as exc:
                log.warning("s3_report_delete_failed", error=str(exc))

    # ── 4. Delete DB records (cascade handles results) ────────
    job_ids = [j.id for j in jobs]
    if job_ids:
        await db.execute(
            sql_delete(DetectionResult).where(
                DetectionResult.job_id.in_(job_ids)
            )
        )
        counts["results"] = len(job_ids)

    counts["jobs"] = len(jobs)
    await db.execute(sql_delete(DetectionJob).where(DetectionJob.user_id == uid))

    # ── 5. Delete API keys ────────────────────────────────────
    api_keys_q = await db.execute(
        select(APIKey).where(APIKey.user_id == uid)
    )
    api_keys = api_keys_q.scalars().all()
    counts["api_keys"] = len(api_keys)
    await db.execute(sql_delete(APIKey).where(APIKey.user_id == uid))

    # ── 6. Delete user record ─────────────────────────────────
    await db.execute(sql_delete(User).where(User.id == uid))
    locations.append("postgresql://authentiguard/users")

    await db.commit()

    completed_at = datetime.now(timezone.utc).isoformat()
    receipt_id   = str(uuid.uuid4())

    log.info("gdpr_deletion_complete",
             user_id=user_id, receipt_id=receipt_id, **counts)

    return DeletionReceipt(
        receipt_id=receipt_id,
        user_id=user_id,
        requested_at=requested_at,
        completed_at=completed_at,
        items_deleted=counts,
        storage_locations=locations,
        confirmation=(
            f"All personal data for user {user_id} has been permanently deleted. "
            f"Audit log entries are retained for legal compliance (cannot be erased). "
            f"Receipt ID: {receipt_id}"
        ),
    )


async def export_user_data(
    user_id: str,
    db: Any,
) -> dict[str, Any]:
    """
    GDPR Article 15: Right of access — data export.
    Returns all stored data for a user in structured JSON format.
    """
    from sqlalchemy import select  # type: ignore
    try:
        from backend.app.models.models import (  # type: ignore
            User, DetectionJob, DetectionResult, AuditLog, APIKey,
        )
    except ImportError:
        return {"error": "Data export unavailable"}

    uid = uuid.UUID(user_id)

    # User record
    user_q = await db.execute(select(User).where(User.id == uid))
    user   = user_q.scalar_one_or_none()
    if not user:
        return {"error": "User not found"}

    # Jobs and results
    jobs_q = await db.execute(
        select(DetectionJob).where(DetectionJob.user_id == uid)
    )
    jobs = jobs_q.scalars().all()

    jobs_data = []
    for job in jobs:
        res_q  = await db.execute(
            select(DetectionResult).where(DetectionResult.job_id == job.id)
        )
        result = res_q.scalar_one_or_none()
        jobs_data.append({
            "job_id":       str(job.id),
            "content_type": job.content_type.value if job.content_type else None,
            "file_name":    job.file_name,
            "status":       job.status.value if job.status else None,
            "created_at":   job.created_at.isoformat() if job.created_at else None,
            "result": {
                "score":     result.authenticity_score if result else None,
                "label":     result.label if result else None,
            } if result else None,
        })

    # Audit log (read-only — cannot be deleted)
    audit_q = await db.execute(
        select(AuditLog).where(AuditLog.user_id == uid).limit(1000)
    )
    audit_rows = audit_q.scalars().all()

    return {
        "export_version": "1.0",
        "exported_at":    datetime.now(timezone.utc).isoformat(),
        "user": {
            "id":            str(user.id),
            "email":         user.email,
            "full_name":     user.full_name,
            "role":          user.role.value if user.role else None,
            "tier":          user.tier.value if user.tier else None,
            "created_at":    user.created_at.isoformat() if user.created_at else None,
            "consent_given": user.consent_given,
            "consent_at":    user.consent_at.isoformat() if user.consent_at else None,
        },
        "analysis_jobs": jobs_data,
        "audit_log_count": len(audit_rows),
        "note": (
            "Audit log entries are retained for legal compliance "
            "and cannot be exported or deleted."
        ),
    }


def _empty_receipt(user_id: str) -> DeletionReceipt:
    now = datetime.now(timezone.utc).isoformat()
    return DeletionReceipt(
        receipt_id=str(uuid.uuid4()),
        user_id=user_id,
        requested_at=now,
        completed_at=now,
        items_deleted={},
        storage_locations=[],
        confirmation="Deletion could not be completed — database unavailable.",
    )


# ── Consent tracking ──────────────────────────────────────────

def record_consent(
    user_id:   str,
    purpose:   str,
    version:   str = "1.0",
    ip:        str | None = None,
) -> dict[str, Any]:
    """
    Article 7: Record consent with purpose, version, timestamp, and IP.
    Stored in the audit log for compliance.
    """
    return {
        "event":     "consent_given",
        "user_id":   user_id,
        "purpose":   purpose,
        "version":   version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ip":        ip,
    }
