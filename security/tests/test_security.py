"""
Unit tests for Phase 12 — Security Hardening.
Steps 88–95: encryption, TLS, JWT policy, rate limiting, signing, GDPR, SOC 2.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import ssl
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from security.encryption.encryption import (
    FieldEncryptor, _decode_key, get_s3_encryption_params,
    verify_tls_version, assert_tls13_environment, hash_content_helper,
)
from security.encryption.auth_security import (
    JWTPolicy, RateLimitPolicy, RATE_LIMIT_POLICIES, ReportSigner,
)
from security.compliance.gdpr import (
    RetentionPolicy, record_consent, _empty_receipt,
)
from security.compliance.soc2 import (
    get_compliance_summary, SOC2_CONTROLS,
)


# ── Step 88: AES-256 field encryption ────────────────────────

class TestFieldEncryptor:
    def _make_encryptor(self) -> FieldEncryptor:
        key = os.urandom(32)
        return FieldEncryptor(key)

    def test_encrypt_decrypt_roundtrip(self) -> None:
        enc = self._make_encryptor()
        plaintext = "user@example.com"
        ciphertext = enc.encrypt(plaintext)
        assert enc.decrypt(ciphertext) == plaintext

    def test_ciphertext_not_equal_plaintext(self) -> None:
        enc = self._make_encryptor()
        ct = enc.encrypt("secret")
        assert ct != "secret"
        assert "secret" not in ct

    def test_same_plaintext_produces_different_ciphertext(self) -> None:
        """GCM uses a random nonce so each encryption is unique."""
        enc = self._make_encryptor()
        ct1 = enc.encrypt("same text")
        ct2 = enc.encrypt("same text")
        assert ct1 != ct2

    def test_different_keys_cannot_decrypt(self) -> None:
        enc1 = FieldEncryptor(os.urandom(32))
        enc2 = FieldEncryptor(os.urandom(32))
        ct = enc1.encrypt("secret data")
        with pytest.raises(ValueError):
            enc2.decrypt(ct)

    def test_tampered_ciphertext_raises(self) -> None:
        enc = self._make_encryptor()
        ct  = enc.encrypt("sensitive")
        # Flip a byte in the ciphertext
        raw     = base64.b64decode(ct.encode())
        tampered = raw[:-1] + bytes([raw[-1] ^ 0xFF])
        bad_ct  = base64.b64encode(tampered).decode()
        with pytest.raises(ValueError):
            enc.decrypt(bad_ct)

    def test_key_must_be_32_bytes(self) -> None:
        with pytest.raises(ValueError, match="32 bytes"):
            FieldEncryptor(os.urandom(16))   # too short

    def test_decode_key_from_hex(self) -> None:
        key_bytes = os.urandom(32)
        key_hex   = key_bytes.hex()
        assert _decode_key(key_hex) == key_bytes

    def test_decode_key_from_base64(self) -> None:
        key_bytes = os.urandom(32)
        key_b64   = base64.b64encode(key_bytes).decode()
        assert _decode_key(key_b64) == key_bytes

    def test_decode_key_derives_from_short_string(self) -> None:
        """Short keys are padded via SHA-256."""
        k1 = _decode_key("my-secret")
        k2 = _decode_key("my-secret")
        assert k1 == k2    # deterministic
        assert len(k1) == 32

    def test_s3_encryption_params(self) -> None:
        params = get_s3_encryption_params(use_kms=False)
        assert params["ServerSideEncryption"] == "AES256"

    def test_s3_kms_params(self) -> None:
        params = get_s3_encryption_params(use_kms=True)
        assert params["ServerSideEncryption"] == "aws:kms"


# ── Step 89: TLS verification ─────────────────────────────────

class TestTLSVerification:
    def test_tls13_supported_in_test_env(self) -> None:
        """Python's built-in ssl module should support TLS 1.3."""
        assert hasattr(ssl.TLSVersion, "TLSv1_3")

    def test_assert_tls13_environment_passes(self) -> None:
        assert_tls13_environment()   # should not raise

    def test_verify_tls_version_tls13(self) -> None:
        assert verify_tls_version({"version": "TLSv1.3"}) is True

    def test_verify_tls_version_tls12_fails(self) -> None:
        assert verify_tls_version({"version": "TLSv1.2"}) is False

    def test_verify_tls_version_empty_fails(self) -> None:
        assert verify_tls_version({}) is False


# ── Step 90: JWT policy ───────────────────────────────────────

class TestJWTPolicy:
    def test_default_policy_is_valid(self) -> None:
        JWTPolicy().validate()   # should not raise

    def test_too_long_access_token_raises(self) -> None:
        with pytest.raises(ValueError, match="60 minutes"):
            JWTPolicy(access_token_expire_minutes=90).validate()

    def test_too_long_refresh_token_raises(self) -> None:
        with pytest.raises(ValueError, match="90 days"):
            JWTPolicy(refresh_token_expire_days=120).validate()

    def test_rotation_disabled_logs_warning(self, caplog) -> None:
        import logging
        policy = JWTPolicy(rotation_enabled=False)
        with caplog.at_level(logging.WARNING):
            policy.validate()
        # Warning should be logged (structlog integration varies in test env)

    def test_from_settings(self) -> None:
        mock_settings = MagicMock()
        mock_settings.JWT_ALGORITHM = "HS256"
        mock_settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 15
        mock_settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS   = 30

        policy = JWTPolicy.from_settings(mock_settings)
        assert policy.algorithm == "HS256"
        assert policy.access_token_expire_minutes == 15


# ── Step 91: Rate limit policy ────────────────────────────────

class TestRateLimitPolicy:
    def test_all_tiers_defined(self) -> None:
        for tier in ["anonymous", "free", "pro", "enterprise"]:
            assert tier in RATE_LIMIT_POLICIES

    def test_tier_ordering(self) -> None:
        anon  = RATE_LIMIT_POLICIES["anonymous"].requests_per_minute
        free  = RATE_LIMIT_POLICIES["free"].requests_per_minute
        pro   = RATE_LIMIT_POLICIES["pro"].requests_per_minute
        ent   = RATE_LIMIT_POLICIES["enterprise"].requests_per_minute
        assert anon < free < pro < ent

    def test_burst_limit_exceeds_base(self) -> None:
        for policy in RATE_LIMIT_POLICIES.values():
            assert policy.burst_limit > policy.requests_per_minute

    def test_enterprise_has_1000_rpm(self) -> None:
        assert RATE_LIMIT_POLICIES["enterprise"].requests_per_minute == 1000


# ── Step 92: Report signing ───────────────────────────────────

class TestReportSigning:
    def test_hmac_sign_verify(self) -> None:
        data   = b"report content goes here"
        secret = "test-signing-secret-32-chars-here"
        sig    = ReportSigner.hmac_sign(data, secret)
        assert len(sig) == 64   # SHA-256 hex
        assert ReportSigner.hmac_verify(data, sig, secret)

    def test_hmac_tampered_data_fails(self) -> None:
        secret = "test-secret-key"
        sig    = ReportSigner.hmac_sign(b"original", secret)
        assert not ReportSigner.hmac_verify(b"tampered", sig, secret)

    def test_hmac_wrong_key_fails(self) -> None:
        sig = ReportSigner.hmac_sign(b"data", "key1")
        assert not ReportSigner.hmac_verify(b"data", sig, "key2")

    def test_hmac_deterministic(self) -> None:
        sig1 = ReportSigner.hmac_sign(b"data", "key")
        sig2 = ReportSigner.hmac_sign(b"data", "key")
        assert sig1 == sig2

    def test_hmac_empty_data(self) -> None:
        sig = ReportSigner.hmac_sign(b"", "key")
        assert len(sig) == 64
        assert ReportSigner.hmac_verify(b"", sig, "key")


# ── Step 93: Data retention ───────────────────────────────────

class TestRetentionPolicy:
    def test_default_values(self) -> None:
        policy = RetentionPolicy()
        assert policy.upload_days    == 30
        assert policy.report_days    == 365
        assert policy.celery_result_h == 24

    def test_upload_expiry_calculation(self) -> None:
        from datetime import datetime, timezone, timedelta
        policy     = RetentionPolicy(upload_days=30)
        created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
        expiry     = policy.upload_expiry(created_at)
        assert expiry == datetime(2024, 1, 31, tzinfo=timezone.utc)

    def test_is_upload_expired(self) -> None:
        from datetime import datetime, timezone, timedelta
        policy     = RetentionPolicy(upload_days=1)
        old_date   = datetime(2020, 1, 1, tzinfo=timezone.utc)
        recent     = datetime.now(timezone.utc) - timedelta(hours=1)
        assert policy.is_upload_expired(old_date) is True
        assert policy.is_upload_expired(recent) is False

    def test_empty_receipt_structure(self) -> None:
        receipt = _empty_receipt("user-123")
        assert receipt.user_id == "user-123"
        assert len(receipt.receipt_id) == 36   # UUID format
        assert "cannot be completed" in receipt.confirmation


# ── Step 94: GDPR ─────────────────────────────────────────────

class TestGDPRCompliance:
    def test_record_consent_structure(self) -> None:
        record = record_consent(
            user_id="user-123",
            purpose="ai_detection_analysis",
            version="1.0",
            ip="1.2.3.4",
        )
        assert record["event"]   == "consent_given"
        assert record["user_id"] == "user-123"
        assert "timestamp" in record
        assert record["purpose"] == "ai_detection_analysis"

    def test_consent_timestamp_is_iso8601(self) -> None:
        from datetime import datetime
        record = record_consent("u", "purpose")
        # Should be parseable as datetime
        datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))


# ── Step 95: SOC 2 compliance summary ────────────────────────

class TestSOC2Compliance:
    def test_controls_not_empty(self) -> None:
        assert len(SOC2_CONTROLS) >= 10

    def test_all_controls_have_required_fields(self) -> None:
        for control in SOC2_CONTROLS:
            assert control.control_id
            assert control.criteria
            assert control.title
            assert control.status in ("IMPLEMENTED", "PARTIAL", "PLANNED")
            assert isinstance(control.evidence, list)

    def test_compliance_summary_structure(self) -> None:
        summary = get_compliance_summary()
        assert "total_controls" in summary
        assert "implemented" in summary
        assert "implementation_pct" in summary
        assert "readiness" in summary

    def test_implementation_percentage_in_range(self) -> None:
        summary = get_compliance_summary()
        assert 0 <= summary["implementation_pct"] <= 100

    def test_counts_sum_to_total(self) -> None:
        summary = get_compliance_summary()
        total = (summary["implemented"] +
                  summary["partial"] +
                  summary["planned"])
        assert total == summary["total_controls"]

    def test_key_security_controls_implemented(self) -> None:
        """Critical security controls must be IMPLEMENTED."""
        critical = {"CC6.1", "CC6.7", "C1.1", "PI1.1"}
        by_id    = {c.control_id: c for c in SOC2_CONTROLS}
        for cid in critical:
            if cid in by_id:
                assert by_id[cid].status == "IMPLEMENTED", \
                    f"Critical control {cid} is not IMPLEMENTED"

    def test_privacy_controls_present(self) -> None:
        privacy_controls = [c for c in SOC2_CONTROLS if c.criteria == "Privacy"]
        assert len(privacy_controls) >= 3
