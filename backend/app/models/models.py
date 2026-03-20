"""
Step 34+35: PostgreSQL ORM models.
All business entities: users, API keys, detection jobs, results, audit logs.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean, DateTime, Enum, Float, ForeignKey,
    Integer, String, Text, UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..core.database import Base


# ── Enums ─────────────────────────────────────────────────────

class UserRole(str, enum.Enum):
    ADMIN    = "admin"
    ANALYST  = "analyst"
    API_CONSUMER = "api_consumer"


class UserTier(str, enum.Enum):
    FREE       = "free"
    PRO        = "pro"
    ENTERPRISE = "enterprise"


class JobStatus(str, enum.Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    COMPLETED  = "completed"
    FAILED     = "failed"


class ContentType(str, enum.Enum):
    TEXT  = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    CODE  = "code"


# ── User ──────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255))
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole), nullable=False, default=UserRole.API_CONSUMER
    )
    tier: Mapped[UserTier] = mapped_column(
        Enum(UserTier), nullable=False, default=UserTier.FREE
    )
    is_active: Mapped[bool]  = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # GDPR
    consent_given: Mapped[bool]      = mapped_column(Boolean, default=False)
    consent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deletion_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationships
    jobs:     Mapped[list["DetectionJob"]] = relationship(back_populates="user", lazy="select")
    api_keys: Mapped[list["APIKey"]]       = relationship(back_populates="user", lazy="select")

    def __repr__(self) -> str:
        return f"<User {self.email} role={self.role}>"


# ── API Key ────────────────────────────────────────────────────

class APIKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID]    = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str]        = mapped_column(String(100), nullable=False)
    key_hash: Mapped[str]    = mapped_column(String(255), nullable=False, unique=True)
    key_prefix: Mapped[str]  = mapped_column(String(10), nullable=False)  # e.g. "ag_sk_ab12"
    is_active: Mapped[bool]  = mapped_column(Boolean, default=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at:   Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped["User"] = relationship(back_populates="api_keys")


# ── Detection Job ─────────────────────────────────────────────

class DetectionJob(Base):
    __tablename__ = "detection_jobs"

    id: Mapped[uuid.UUID]      = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    celery_task_id: Mapped[str | None] = mapped_column(String(255), index=True)

    # Input
    content_type: Mapped[ContentType] = mapped_column(Enum(ContentType), nullable=False)
    input_text:   Mapped[str | None]  = mapped_column(Text)           # for text/code paste
    s3_key:       Mapped[str | None]  = mapped_column(String(1024))   # for file uploads
    file_name:    Mapped[str | None]  = mapped_column(String(512))
    file_size:    Mapped[int | None]  = mapped_column(Integer)
    content_hash: Mapped[str | None]  = mapped_column(String(64))     # SHA-256

    # Status
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus), nullable=False, default=JobStatus.PENDING, index=True
    )
    error_message: Mapped[str | None]   = mapped_column(Text)
    started_at:    Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at:  Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationships
    user:   Mapped["User | None"]          = relationship(back_populates="jobs")
    result: Mapped["DetectionResult | None"] = relationship(back_populates="job", uselist=False)

    def __repr__(self) -> str:
        return f"<DetectionJob {self.id} type={self.content_type} status={self.status}>"


# ── Detection Result ──────────────────────────────────────────

class DetectionResult(Base):
    __tablename__ = "detection_results"

    id: Mapped[uuid.UUID]  = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("detection_jobs.id", ondelete="CASCADE"), nullable=False, unique=True
    )

    # Scores
    authenticity_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence:         Mapped[float] = mapped_column(Float, nullable=False)
    label: Mapped[str]  = mapped_column(String(20), nullable=False)   # AI / HUMAN / UNCERTAIN

    # Layer scores
    layer_scores:      Mapped[dict] = mapped_column(JSONB, default=dict)
    evidence_summary:  Mapped[dict] = mapped_column(JSONB, default=dict)
    sentence_scores:   Mapped[list] = mapped_column(JSONB, default=list)
    model_attribution: Mapped[dict] = mapped_column(JSONB, default=dict)

    # Metadata
    metadata_signals: Mapped[dict] = mapped_column(JSONB, default=dict)

    # Report
    report_s3_key:  Mapped[str | None] = mapped_column(String(1024))
    report_hash:    Mapped[str | None] = mapped_column(String(64))
    report_signature: Mapped[str | None] = mapped_column(Text)   # digital signature

    # Processing time
    processing_ms: Mapped[int | None] = mapped_column(Integer)

    job: Mapped["DetectionJob"] = relationship(back_populates="result")


# ── Audit Log ─────────────────────────────────────────────────

class AuditLog(Base):
    """
    Step 36: Immutable audit log for every detection operation,
    API call, and admin action.
    """
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID]       = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), index=True)
    action: Mapped[str]         = mapped_column(String(100), nullable=False, index=True)
    resource_type: Mapped[str | None] = mapped_column(String(50))
    resource_id: Mapped[str | None]   = mapped_column(String(255))
    ip_address:  Mapped[str | None]   = mapped_column(String(45))
    user_agent:  Mapped[str | None]   = mapped_column(Text)
    details:     Mapped[dict]         = mapped_column(JSONB, default=dict)
    success:     Mapped[bool]         = mapped_column(Boolean, default=True)
    error_msg:   Mapped[str | None]   = mapped_column(Text)

    # No updated_at on audit logs — they are immutable
    __table_args__ = (
        {"postgresql_partition_by": "RANGE (created_at)"},  # ready for time-partitioning
    )
