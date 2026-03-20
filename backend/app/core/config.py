"""
Step 9 (enforcement): All configuration from environment variables.
Pydantic Settings validates types and raises on startup if required vars are missing.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    DB_POOL_SIZE: int   = 10
    DB_MAX_OVERFLOW: int = 20

    # ── Redis ──────────────────────────────────────────────────
    REDIS_URL: str       # redis://:pass@host:port/db

    # ── Object storage ─────────────────────────────────────────
    S3_BUCKET_UPLOADS: str  = "ag-uploads"
    S3_BUCKET_REPORTS: str  = "ag-reports"
    S3_ENDPOINT_URL:   str | None = None   # None = real AWS
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
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

    # ── File upload ────────────────────────────────────────────
    MAX_UPLOAD_SIZE_MB:     int = 500
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

    # ── CORS ──────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_db_url(cls, v: str) -> str:
        if not v.startswith("postgresql"):
            raise ValueError("DATABASE_URL must be a PostgreSQL connection string")
        return v

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
