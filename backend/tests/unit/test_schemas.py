"""
Unit tests for Pydantic schemas — validation rules, defaults, edge cases.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.schemas import (
    DetectionResultResponse,
    LayerScoresSchema,
    LoginRequest,
    ModelAttributionSchema,
    RegisterRequest,
    TextSubmitRequest,
    UrlSubmitRequest,
    WebhookCreateRequest,
)


class TestRegisterRequest:
    def test_valid_registration(self):
        req = RegisterRequest(
            email="user@example.com",
            password="StrongPass1!",
            consent_given=True,
        )
        assert req.email == "user@example.com"

    def test_rejects_no_consent(self):
        with pytest.raises(ValidationError, match="consent"):
            RegisterRequest(
                email="user@example.com",
                password="StrongPass1!",
                consent_given=False,
            )

    def test_rejects_no_uppercase(self):
        with pytest.raises(ValidationError, match="uppercase"):
            RegisterRequest(
                email="user@example.com",
                password="weakpassword1!",
                consent_given=True,
            )

    def test_rejects_no_digit(self):
        with pytest.raises(ValidationError, match="digit"):
            RegisterRequest(
                email="user@example.com",
                password="WeakPassword!!",
                consent_given=True,
            )

    def test_rejects_short_password(self):
        with pytest.raises(ValidationError):
            RegisterRequest(
                email="user@example.com",
                password="Short1!",  # < 10 chars
                consent_given=True,
            )

    def test_rejects_invalid_email(self):
        with pytest.raises(ValidationError):
            RegisterRequest(
                email="not-an-email",
                password="StrongPass1!",
                consent_given=True,
            )


class TestTextSubmitRequest:
    def test_valid_text(self):
        req = TextSubmitRequest(text="A" * 20)
        assert req.content_type == "text"

    def test_rejects_short_text(self):
        with pytest.raises(ValidationError):
            TextSubmitRequest(text="too short")

    def test_accepts_code_type(self):
        req = TextSubmitRequest(text="A" * 20, content_type="code")
        assert req.content_type == "code"

    def test_rejects_invalid_content_type(self):
        with pytest.raises(ValidationError):
            TextSubmitRequest(text="A" * 20, content_type="image")


class TestModelAttributionSchema:
    def test_defaults_all_zero(self):
        schema = ModelAttributionSchema()
        assert schema.gpt_family == 0.0
        assert schema.claude_family == 0.0
        assert schema.llama_family == 0.0
        assert schema.human == 0.0
        assert schema.other == 0.0

    def test_from_empty_dict(self):
        """Empty dict should produce default ModelAttributionSchema."""
        schema = ModelAttributionSchema(**{})
        assert schema.gpt_family == 0.0


class TestLayerScoresSchema:
    def test_all_optional(self):
        schema = LayerScoresSchema()
        assert schema.perplexity is None
        assert schema.stylometry is None
        assert schema.transformer is None
        assert schema.adversarial is None

    def test_partial_scores(self):
        schema = LayerScoresSchema(perplexity=0.8, stylometry=0.6)
        assert schema.perplexity == 0.8
        assert schema.transformer is None


class TestWebhookCreateRequest:
    def test_valid_webhook(self):
        req = WebhookCreateRequest(
            url="https://example.com/webhook",
            events=["job.completed"],
        )
        assert req.url == "https://example.com/webhook"

    def test_rejects_invalid_event(self):
        with pytest.raises(ValidationError):
            WebhookCreateRequest(
                url="https://example.com/webhook",
                events=["invalid.event"],
            )

    def test_secret_min_length(self):
        with pytest.raises(ValidationError):
            WebhookCreateRequest(
                url="https://example.com/webhook",
                events=["job.completed"],
                secret="short",  # < 16 chars
            )
