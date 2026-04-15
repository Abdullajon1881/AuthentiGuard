"""
Step 9 (enforcement): All configuration from environment variables.
Pydantic Settings validates types and raises on startup if required vars are missing.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import AnyHttpUrl, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _read_secret_file(env_var: str) -> str | None:
    """Read a Docker-style `*_FILE` secret if the env var is set.

    Returns the file contents (stripped of trailing whitespace/newline) or None.
    Errors intentionally surface — a misconfigured secret path must fail loudly.
    """
    path = os.environ.get(env_var)
    if not path:
        return None
    return Path(path).read_text(encoding="utf-8").rstrip("\r\n")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── App ────────────────────────────────────────────────────
    APP_ENV:   Literal["development", "staging", "production"] = "development"
    APP_DEBUG: bool  = False
    APP_NAME:  str   = "AuthentiGuard API"
    APP_VERSION: str = "0.1.0"
    APP_SECRET_KEY: str

    # ── Database ───────────────────────────────────────────────
    DATABASE_URL: str   # postgresql+asyncpg://...
    DB_POOL_SIZE: int   = 20
    DB_MAX_OVERFLOW: int = 40

    # ── Redis ──────────────────────────────────────────────────
    REDIS_URL: str       # redis://:pass@host:port/db

    # ── Object storage ─────────────────────────────────────────
    # Credentials are resolved in this order (first non-empty wins):
    #   1. AWS_ACCESS_KEY_ID_FILE / AWS_SECRET_ACCESS_KEY_FILE  (Docker secrets)
    #   2. AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY            (plain env — dev/CI only)
    # Production must use the file-backed path so credentials never appear in
    # `docker inspect`, process env listings, or container image layers.
    S3_BUCKET_UPLOADS: str  = "ag-uploads"
    S3_BUCKET_REPORTS: str  = "ag-reports"
    S3_ENDPOINT_URL:   str | None = None   # None = real AWS
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "us-east-1"

    # ── Auth ───────────────────────────────────────────────────
    JWT_SECRET_KEY: str
    JWT_ALGORITHM:  str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES:  int = 15
    JWT_REFRESH_TOKEN_EXPIRE_DAYS:    int = 30

    # ── Rate limiting ──────────────────────────────────────────
    RATE_LIMIT_FREE_TIER:       int = 10    # requests per minute
    RATE_LIMIT_PRO_TIER:        int = 100
    RATE_LIMIT_ENTERPRISE_TIER: int = 1000

    # ── Celery ─────────────────────────────────────────────────
    CELERY_BROKER_URL:   str
    CELERY_RESULT_BACKEND: str

    # ── Detector ──────────────────────────────────────────────
    # "ml" = load real ML models (L1+L2+L3 ensemble)
    # "heuristic" = use lightweight heuristic fallback (no GPU/model deps)
    DETECTOR_MODE: Literal["ml", "heuristic"] = "ml"

    # ── Schema bootstrap ───────────────────────────────────────
    # Dev-only escape hatch. When True, startup runs SQLAlchemy
    # `Base.metadata.create_all` so a fresh dev machine can boot without
    # running Alembic. Production MUST leave this False — Alembic is the
    # single source of truth, and a prod boot with this flag set will fail
    # fast (see validator + main.py lifespan).
    ALLOW_DB_CREATE_ALL: bool = False

    # ── File upload ────────────────────────────────────────────
    MAX_UPLOAD_SIZE_MB:     int = 50
    ALLOWED_TEXT_EXTENSIONS: list[str] = [".txt", ".md", ".pdf", ".docx"]
    ALLOWED_AUDIO_EXTENSIONS: list[str] = [".mp3", ".wav", ".flac", ".m4a", ".ogg"]
    ALLOWED_VIDEO_EXTENSIONS: list[str] = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
    ALLOWED_IMAGE_EXTENSIONS: list[str] = [".jpg", ".jpeg", ".png", ".webp", ".gif"]
    ALLOWED_CODE_EXTENSIONS: list[str]  = [".py", ".js", ".ts", ".java", ".cpp",
                                            ".c", ".go", ".rs", ".rb", ".php"]

    # ── Retention ──────────────────────────────────────────────
    UPLOAD_RETENTION_DAYS:  int = 30
    REPORT_RETENTION_DAYS:  int = 365

    # ── Encryption ────────────────────────────────────────────
    ENCRYPTION_KEY: str   # Fernet key for at-rest encryption

    # ── Alerting ──────────────────────────────────────────────
    # Generic outgoing-webhook URL. The alerting task posts a small JSON
    # body ({"text": "...", "severity": "...", "alert": "..."}) when a
    # detector-fallback, high-failure-rate, or high-queue-depth condition
    # is detected. Leave blank to disable webhook notifications (alerts
    # still land in structlog). A Slack-compatible "Incoming Webhook" URL
    # works out of the box. Production deployments without Prometheus/
    # Alertmanager must set this so somebody actually gets paged.
    ALERT_WEBHOOK_URL: str = ""
    ALERT_WEBHOOK_TIMEOUT_SECONDS: float = 5.0

    # ── CORS ──────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        """Parse comma-separated CORS origins from env var string."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_db_url(cls, v: str) -> str:
        if not v.startswith("postgresql"):
            raise ValueError("DATABASE_URL must be a PostgreSQL connection string")
        return v

    @field_validator("JWT_SECRET_KEY")
    @classmethod
    def validate_jwt_strength(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters for HS256 security")
        return v

    @model_validator(mode="after")
    def _resolve_s3_credentials_from_files(self) -> "Settings":
        """Load S3 creds from Docker-secret files when `*_FILE` env vars are set.

        Only fills in empty fields — an explicit plain env value always wins so
        dev and CI (which set plain values) are unaffected.
        """
        file_user = _read_secret_file("AWS_ACCESS_KEY_ID_FILE")
        if file_user and not self.AWS_ACCESS_KEY_ID:
            object.__setattr__(self, "AWS_ACCESS_KEY_ID", file_user)
        file_secret = _read_secret_file("AWS_SECRET_ACCESS_KEY_FILE")
        if file_secret and not self.AWS_SECRET_ACCESS_KEY:
            object.__setattr__(self, "AWS_SECRET_ACCESS_KEY", file_secret)
        if not self.AWS_ACCESS_KEY_ID or not self.AWS_SECRET_ACCESS_KEY:
            raise ValueError(
                "S3 credentials missing: set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY "
                "or AWS_ACCESS_KEY_ID_FILE/AWS_SECRET_ACCESS_KEY_FILE"
            )
        return self

    @model_validator(mode="after")
    def _forbid_create_all_in_production(self) -> "Settings":
        """Fail loud if a production build leaves the dev-only create_all gate on.

        Catches the case where someone ships with `ALLOW_DB_CREATE_ALL=true`
        in their prod env file. Alembic must be the only schema authority in
        production; create_all running alongside migrations causes silent
        schema drift on the first post-launch column change.
        """
        if self.APP_ENV == "production" and self.ALLOW_DB_CREATE_ALL:
            raise ValueError(
                "ALLOW_DB_CREATE_ALL must be False in production. "
                "Alembic is the source of truth; run `alembic upgrade head` instead."
            )
        return self

    @model_validator(mode="after")
    def validate_rate_limit_ordering(self) -> "Settings":
        if not (self.RATE_LIMIT_FREE_TIER <= self.RATE_LIMIT_PRO_TIER <= self.RATE_LIMIT_ENTERPRISE_TIER):
            raise ValueError(
                "Rate limits must be ordered: free <= pro <= enterprise "
                f"(got {self.RATE_LIMIT_FREE_TIER}, {self.RATE_LIMIT_PRO_TIER}, {self.RATE_LIMIT_ENTERPRISE_TIER})"
            )
        return self

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
