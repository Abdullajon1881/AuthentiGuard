"""
Pydantic v2 schemas — request validation and response serialization.
Strict types, no 'any' fields. All user-facing API contracts live here.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator


# ── Auth schemas ──────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: Annotated[str, Field(min_length=10, max_length=128)]
    full_name: Annotated[str, Field(min_length=1, max_length=255)] | None = None
    consent_given: bool

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v

    @model_validator(mode="after")
    def consent_required(self) -> "RegisterRequest":
        if not self.consent_given:
            raise ValueError("You must provide consent to create an account (GDPR)")
        return self


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    Literal["bearer"] = "bearer"
    expires_in:    int   # seconds until access token expires


class RefreshRequest(BaseModel):
    refresh_token: str


# ── User schemas ──────────────────────────────────────────────

class UserResponse(BaseModel):
    id:         uuid.UUID
    email:      str
    full_name:  str | None
    role:       str
    tier:       str
    is_active:  bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Upload schemas ────────────────────────────────────────────

class TextSubmitRequest(BaseModel):
    """Direct text paste — no file upload needed."""
    text: Annotated[str, Field(min_length=20, max_length=100_000)]
    content_type: Literal["text", "code"] = "text"


class AnalysisJobResponse(BaseModel):
    """Returned immediately after submission — client polls for results."""
    job_id:      uuid.UUID
    status:      str
    content_type: str
    created_at:  datetime
    poll_url:    str


# ── Result schemas ────────────────────────────────────────────

class SentenceScoreSchema(BaseModel):
    text:     str
    score:    float
    evidence: dict[str, Any] = {}


class LayerScoresSchema(BaseModel):
    perplexity:  float | None = None
    stylometry:  float | None = None
    transformer: float | None = None
    adversarial: float | None = None


class EvidenceSignalSchema(BaseModel):
    signal:  str
    value:   str
    weight:  Literal["high", "medium", "low"]


class ModelAttributionSchema(BaseModel):
    gpt_family:    float = 0.0
    claude_family: float = 0.0
    llama_family:  float = 0.0
    human:         float = 0.0
    other:         float = 0.0


class DetectionResultResponse(BaseModel):
    job_id:            uuid.UUID
    status:            str
    content_type:      str
    authenticity_score: float
    confidence:        float
    label:             Literal["AI", "HUMAN", "UNCERTAIN"]
    layer_scores:      LayerScoresSchema
    sentence_scores:   list[SentenceScoreSchema] = []
    top_signals:       list[EvidenceSignalSchema] = []
    model_attribution: ModelAttributionSchema
    processing_ms:     int | None
    report_url:        str | None
    created_at:        datetime
    completed_at:      datetime | None

    model_config = {"from_attributes": True}


class JobStatusResponse(BaseModel):
    job_id:    uuid.UUID
    status:    str
    progress:  str | None  = None   # "extracting frames", "running transformer", etc.
    created_at: datetime
    completed_at: datetime | None

    model_config = {"from_attributes": True}


# ── Report schemas ────────────────────────────────────────────

class ReportResponse(BaseModel):
    report_url:       str
    report_hash:      str
    report_signature: str
    generated_at:     datetime


# ── Webhook schemas ───────────────────────────────────────────

class WebhookCreateRequest(BaseModel):
    url:    Annotated[str, Field(max_length=2048)]
    events: list[Literal["job.completed", "job.failed"]]
    secret: Annotated[str, Field(min_length=16, max_length=128)] | None = None


class WebhookResponse(BaseModel):
    id:       uuid.UUID
    url:      str
    events:   list[str]
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Dashboard schemas ─────────────────────────────────────────

class UsageStatsResponse(BaseModel):
    total_scans:     int
    scans_this_month: int
    ai_detected:     int
    human_detected:  int
    uncertain:       int
    avg_score:       float | None
    tier_limit:      int
    tier_used:       int


# ── Error schema ──────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error:   str
    message: str
    detail:  Any = None
