"""
Unit tests for the backend services.
Uses pytest-asyncio for async tests. DB/Redis are mocked.
"""

from __future__ import annotations

import hashlib
import math
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.schemas.schemas import (
    LoginRequest,
    RegisterRequest,
    TextSubmitRequest,
)
from app.services.result_engine import compute_authenticity_score
from app.services.metadata_service import (
    detect_watermark,
    extract_device_fingerprint,
)


# ── Security / Auth ───────────────────────────────────────────

class TestSecurity:
    def test_password_hash_and_verify(self) -> None:
        plain    = "SecurePass123!"
        hashed   = hash_password(plain)
        assert hashed != plain
        assert verify_password(plain, hashed)
        assert not verify_password("WrongPass123!", hashed)

    def test_create_and_decode_access_token(self) -> None:
        with patch("app.core.security.get_settings") as mock_cfg:
            mock_cfg.return_value.JWT_SECRET_KEY = "test-secret-32-chars-exactly-here"
            mock_cfg.return_value.JWT_ALGORITHM  = "HS256"
            mock_cfg.return_value.JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 15

            token = create_access_token("user-123", "api_consumer", "test@example.com")
            payload = decode_access_token(token)

            assert payload["sub"]   == "user-123"
            assert payload["role"]  == "api_consumer"
            assert payload["email"] == "test@example.com"
            assert payload["type"]  == "access"

    def test_decode_invalid_token_raises(self) -> None:
        with patch("app.core.security.get_settings") as mock_cfg:
            mock_cfg.return_value.JWT_SECRET_KEY = "test-secret-32-chars-exactly-here"
            mock_cfg.return_value.JWT_ALGORITHM  = "HS256"
            with pytest.raises(ValueError, match="Invalid"):
                decode_access_token("not.a.valid.token")

    def test_decode_wrong_type_raises(self) -> None:
        """Refresh tokens must not be accepted as access tokens."""
        with patch("app.core.security.get_settings") as mock_cfg:
            mock_cfg.return_value.JWT_SECRET_KEY = "test-secret-32-chars-exactly-here"
            mock_cfg.return_value.JWT_ALGORITHM  = "HS256"
            mock_cfg.return_value.JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 15
            from jose import jwt
            import datetime, uuid
            now = datetime.datetime.now(datetime.timezone.utc)
            payload = {
                "sub": "user-123", "role": "free", "email": "t@t.com",
                "exp": now + datetime.timedelta(days=30),
                "iat": now, "jti": str(uuid.uuid4()),
                "type": "refresh",   # wrong type
            }
            token = jwt.encode(payload, "test-secret-32-chars-exactly-here", algorithm="HS256")
            with pytest.raises(ValueError, match="not an access token"):
                decode_access_token(token)


# ── Schema Validation ─────────────────────────────────────────

class TestSchemas:
    def test_register_requires_consent(self) -> None:
        with pytest.raises(Exception):
            RegisterRequest(
                email="test@example.com",
                password="StrongPass1!",
                consent_given=False,
            )

    def test_register_weak_password(self) -> None:
        with pytest.raises(Exception):
            RegisterRequest(
                email="test@example.com",
                password="weakpassword",  # no uppercase, no digit
                consent_given=True,
            )

    def test_register_valid(self) -> None:
        req = RegisterRequest(
            email="test@example.com",
            password="StrongPass1!",
            consent_given=True,
        )
        assert req.email == "test@example.com"

    def test_text_submit_min_length(self) -> None:
        with pytest.raises(Exception):
            TextSubmitRequest(text="short")   # < 20 chars

    def test_text_submit_valid(self) -> None:
        req = TextSubmitRequest(text="A" * 20)
        assert req.content_type == "text"


# ── Result Engine ─────────────────────────────────────────────

class TestResultEngine:
    def _metadata(self, **overrides) -> dict:
        base = {
            "watermark": {"watermark_detected": False, "confidence": 0.0},
            "device_fingerprint": {"likely_ai_generated": False, "likely_camera_capture": False, "suspicious_signals": []},
            "provenance": {},
        }
        base.update(overrides)
        return base

    def test_high_ai_score_labels_as_ai(self) -> None:
        result = compute_authenticity_score(0.90, self._metadata(), "text")
        assert result.label == "AI"
        assert result.authenticity_score >= 0.75

    def test_low_ai_score_labels_as_human(self) -> None:
        result = compute_authenticity_score(0.10, self._metadata(), "text")
        assert result.label == "HUMAN"
        assert result.authenticity_score <= 0.40

    def test_mid_range_is_uncertain(self) -> None:
        result = compute_authenticity_score(0.55, self._metadata(), "text")
        assert result.label == "UNCERTAIN"

    def test_watermark_pushes_score_up(self) -> None:
        no_wm  = compute_authenticity_score(0.70, self._metadata(), "text")
        with_wm = compute_authenticity_score(
            0.70,
            self._metadata(watermark={"watermark_detected": True, "confidence": 0.9}),
            "text",
        )
        assert with_wm.authenticity_score > no_wm.authenticity_score

    def test_camera_exif_pushes_score_down(self) -> None:
        no_cam = compute_authenticity_score(0.60, self._metadata(), "image")
        with_cam = compute_authenticity_score(
            0.60,
            self._metadata(device_fingerprint={
                "likely_camera_capture": True,
                "likely_ai_generated": False,
                "suspicious_signals": [],
            }),
            "image",
        )
        assert with_cam.authenticity_score < no_cam.authenticity_score

    def test_provenance_verified_reduces_score(self) -> None:
        no_prov  = compute_authenticity_score(0.65, self._metadata(), "text")
        with_prov = compute_authenticity_score(
            0.65,
            self._metadata(provenance={"c2pa_verified": True}),
            "text",
        )
        assert with_prov.authenticity_score < no_prov.authenticity_score

    def test_score_always_in_range(self) -> None:
        for score in [0.0, 0.01, 0.5, 0.99, 1.0]:
            result = compute_authenticity_score(score, self._metadata(), "text")
            assert 0.0 < result.authenticity_score < 1.0

    def test_model_attribution_sums_to_one(self) -> None:
        result = compute_authenticity_score(0.80, self._metadata(), "text")
        total = sum(result.model_attribution.values())
        assert abs(total - 1.0) < 0.01


# ── Metadata Service ──────────────────────────────────────────

class TestMetadataService:
    def test_watermark_detection_long_text(self) -> None:
        # 50+ words needed for the test
        text = " ".join(["word"] * 100)
        result = detect_watermark(text=text)
        assert "watermark_detected" in result
        assert "confidence" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_watermark_short_text_skipped(self) -> None:
        result = detect_watermark(text="too short")
        assert result["watermark_detected"] is False

    def test_device_fingerprint_no_exif(self) -> None:
        fp = extract_device_fingerprint({"has_exif": False})
        assert "no_exif_data" in fp["suspicious_signals"]
        assert fp["likely_ai_generated"] is True

    def test_device_fingerprint_with_camera(self) -> None:
        fp = extract_device_fingerprint({
            "has_exif": True,
            "has_camera_info": True,
            "image_make": "Canon",
            "image_model": "EOS R5",
        })
        assert fp["likely_camera_capture"] is True
        assert fp["likely_ai_generated"] is False

    def test_ai_software_tag_detected(self) -> None:
        fp = extract_device_fingerprint({
            "has_exif": True,
            "has_camera_info": False,
            "image_software": "Stable Diffusion v2.1",
        })
        assert any("stable diffusion" in s for s in fp["suspicious_signals"])
        assert fp["likely_ai_generated"] is True
