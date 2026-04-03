"""
Unit tests for config validation — CORS parsing, DB URL validation.
"""

from __future__ import annotations

import pytest


class TestDatabaseUrlValidator:
    def test_accepts_postgresql(self):
        from app.core.config import Settings
        # The validator checks startswith("postgresql")
        result = Settings.validate_db_url("postgresql+asyncpg://user:pass@localhost/db")
        assert result.startswith("postgresql")

    def test_rejects_non_postgresql(self):
        from app.core.config import Settings
        with pytest.raises(ValueError, match="PostgreSQL"):
            Settings.validate_db_url("mysql://user:pass@localhost/db")

    def test_rejects_sqlite(self):
        from app.core.config import Settings
        with pytest.raises(ValueError, match="PostgreSQL"):
            Settings.validate_db_url("sqlite:///test.db")


class TestSettingsDefaults:
    def test_default_app_env(self):
        from app.core.config import Settings
        assert Settings.model_fields["APP_ENV"].default == "development"

    def test_default_rate_limits(self):
        from app.core.config import Settings
        assert Settings.model_fields["RATE_LIMIT_FREE_TIER"].default == 10
        assert Settings.model_fields["RATE_LIMIT_PRO_TIER"].default == 100
        assert Settings.model_fields["RATE_LIMIT_ENTERPRISE_TIER"].default == 1000

    def test_default_jwt_algorithm(self):
        from app.core.config import Settings
        assert Settings.model_fields["JWT_ALGORITHM"].default == "HS256"

    def test_default_upload_retention(self):
        from app.core.config import Settings
        assert Settings.model_fields["UPLOAD_RETENTION_DAYS"].default == 30
        assert Settings.model_fields["REPORT_RETENTION_DAYS"].default == 365

    def test_allowed_extensions_defined(self):
        from app.core.config import Settings
        assert ".py" in Settings.model_fields["ALLOWED_CODE_EXTENSIONS"].default
        assert ".jpg" in Settings.model_fields["ALLOWED_IMAGE_EXTENSIONS"].default
        assert ".mp3" in Settings.model_fields["ALLOWED_AUDIO_EXTENSIONS"].default
