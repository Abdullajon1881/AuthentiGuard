"""
Unit tests for CORS configuration.
Verifies that CORS origins are parsed correctly from env vars.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestCorsOriginsParsing:
    def test_parse_comma_separated_string(self):
        """Env var string should be split into a list."""
        from app.core.config import Settings

        # Simulate the validator
        result = Settings.parse_cors_origins("https://example.com,https://www.example.com")
        assert result == ["https://example.com", "https://www.example.com"]

    def test_parse_single_origin(self):
        """Single origin string should become a one-element list."""
        from app.core.config import Settings
        result = Settings.parse_cors_origins("https://example.com")
        assert result == ["https://example.com"]

    def test_parse_trims_whitespace(self):
        """Whitespace around origins should be stripped."""
        from app.core.config import Settings
        result = Settings.parse_cors_origins("  https://a.com , https://b.com  ")
        assert result == ["https://a.com", "https://b.com"]

    def test_parse_list_passthrough(self):
        """If already a list, pass through unchanged."""
        from app.core.config import Settings
        origins = ["https://a.com", "https://b.com"]
        result = Settings.parse_cors_origins(origins)
        assert result == origins

    def test_parse_empty_entries_filtered(self):
        """Empty entries from trailing commas should be filtered out."""
        from app.core.config import Settings
        result = Settings.parse_cors_origins("https://a.com,,https://b.com,")
        assert result == ["https://a.com", "https://b.com"]

    def test_default_is_localhost(self):
        """Default CORS_ORIGINS should be localhost:3000."""
        from app.core.config import Settings
        # Check the field default
        default = Settings.model_fields["CORS_ORIGINS"].default
        assert default == ["http://localhost:3000"]
