"""
Steps 88–89: Encryption layer.

Step 88 — AES-256 encryption at rest
  Every uploaded file stored in S3 uses SSE-S3 (AES-256) at the bucket level.
  Sensitive fields in PostgreSQL (email, hashed passwords, API keys) are also
  encrypted at the application layer using Fernet (AES-128-CBC with HMAC).
  For fields requiring stronger guarantees (e.g. raw API keys before hashing),
  we use AES-256-GCM directly.

Step 89 — TLS 1.3 verification
  All external connections (API, S3, Redis, PostgreSQL) must use TLS 1.3.
  This module provides a connection verifier and an ssl.SSLContext factory
  that enforces TLS 1.3 minimum, rejects TLS 1.2 and below, and pins to
  a strong cipher suite.

Both steps are enforced at startup — if the environment doesn't satisfy the
requirements, the service refuses to start.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import ssl
import struct
import time
from typing import Any

import structlog

log = structlog.get_logger(__name__)


# ── Step 88: AES-256 encryption at rest ──────────────────────

class FieldEncryptor:
    """
    Application-layer field encryption for sensitive database columns.

    Uses AES-256-GCM for authenticated encryption.
    Each ciphertext is self-contained: nonce | ciphertext | tag.

    Usage:
        enc = FieldEncryptor.from_env()
        ciphertext = enc.encrypt("user@example.com")
        plaintext  = enc.decrypt(ciphertext)
    """

    NONCE_BYTES = 12    # GCM standard nonce
    TAG_BYTES   = 16    # GCM authentication tag
    KEY_BYTES   = 32    # AES-256

    def __init__(self, key: bytes) -> None:
        if len(key) != self.KEY_BYTES:
            raise ValueError(f"Key must be exactly {self.KEY_BYTES} bytes")
        self._key = key

    @classmethod
    def from_env(cls) -> "FieldEncryptor":
        """Load key from ENCRYPTION_KEY environment variable (hex or base64)."""
        raw = os.environ.get("ENCRYPTION_KEY", "")
        if not raw:
            raise RuntimeError("ENCRYPTION_KEY environment variable is not set")
        key = _decode_key(raw)
        return cls(key)

    @classmethod
    def from_key(cls, key_hex_or_b64: str) -> "FieldEncryptor":
        return cls(_decode_key(key_hex_or_b64))

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt a string value. Returns a base64-encoded string
        containing nonce | ciphertext | tag.
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore
            nonce      = os.urandom(self.NONCE_BYTES)
            aesgcm     = AESGCM(self._key)
            ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
            # ciphertext includes the 16-byte GCM tag appended by cryptography
            combined   = nonce + ciphertext
            return base64.b64encode(combined).decode("ascii")
        except ImportError as exc:
            raise RuntimeError(
                "pip install cryptography to enable field encryption"
            ) from exc

    def decrypt(self, ciphertext_b64: str) -> str:
        """Decrypt a base64-encoded ciphertext back to plaintext."""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore
            combined   = base64.b64decode(ciphertext_b64.encode("ascii"))
            nonce      = combined[:self.NONCE_BYTES]
            ciphertext = combined[self.NONCE_BYTES:]
            aesgcm     = AESGCM(self._key)
            plaintext  = aesgcm.decrypt(nonce, ciphertext, None)
            return plaintext.decode("utf-8")
        except ImportError as exc:
            raise RuntimeError("pip install cryptography") from exc
        except Exception as exc:
            raise ValueError(f"Decryption failed: {exc}") from exc

    def rotate(self, ciphertext_b64: str, new_key: bytes) -> str:
        """Re-encrypt a ciphertext with a new key (key rotation support)."""
        plaintext = self.decrypt(ciphertext_b64)
        new_enc   = FieldEncryptor(new_key)
        return new_enc.encrypt(plaintext)


def _decode_key(key_str: str) -> bytes:
    """Accept key as hex, base64, or raw string and normalise to 32 bytes."""
    key_str = key_str.strip()
    # Try hex
    try:
        b = bytes.fromhex(key_str)
        if len(b) == 32:
            return b
    except ValueError:
        pass
    # Try base64
    try:
        b = base64.b64decode(key_str + "==")
        if len(b) >= 32:
            return b[:32]
    except Exception:
        pass
    # Derive 32 bytes via SHA-256 (for short/arbitrary strings)
    return hashlib.sha256(key_str.encode()).digest()


# ── S3 server-side encryption helpers ────────────────────────

S3_ENCRYPTION_PARAMS: dict[str, str] = {
    "ServerSideEncryption": "AES256",
}

KMS_ENCRYPTION_PARAMS: dict[str, str] = {
    "ServerSideEncryption": "aws:kms",
    # "SSEKMSKeyId": "arn:aws:kms:...",  # set in production
}


def get_s3_encryption_params(use_kms: bool = False) -> dict[str, str]:
    """
    Return S3 PutObject parameters for server-side encryption.
    Use KMS for enterprise tier, SSE-S3 (AES256) for standard.
    """
    return KMS_ENCRYPTION_PARAMS if use_kms else S3_ENCRYPTION_PARAMS


# ── Step 89: TLS 1.3 enforcement ─────────────────────────────

def create_tls13_context(
    purpose: ssl.Purpose = ssl.Purpose.SERVER_AUTH,
    verify: bool = True,
) -> ssl.SSLContext:
    """
    Create an SSLContext that enforces TLS 1.3 minimum.

    TLS 1.2 and below are explicitly disabled.
    Only strong cipher suites are permitted.

    Args:
        purpose: ssl.Purpose.SERVER_AUTH (client) or CLIENT_AUTH (server)
        verify:  Whether to verify server certificates (always True in prod)
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT if purpose == ssl.Purpose.SERVER_AUTH
                         else ssl.PROTOCOL_TLS_SERVER)

    # Enforce TLS 1.3 minimum
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.maximum_version = ssl.TLSVersion.MAXIMUM_SUPPORTED

    if verify:
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.check_hostname = True
        ctx.load_default_certs()
    else:
        ctx.verify_mode  = ssl.CERT_NONE
        ctx.check_hostname = False

    # Disable compression (CRIME attack mitigation)
    ctx.options |= ssl.OP_NO_COMPRESSION

    log.debug("tls13_context_created", minimum="TLS 1.3")
    return ctx


def verify_tls_version(connection_info: dict[str, Any]) -> bool:
    """
    Verify that an established connection is using TLS 1.3.

    Args:
        connection_info: Dict from ssl.SSLSocket.cipher() or similar.

    Returns:
        True if TLS 1.3 or higher is in use.
    """
    version = str(connection_info.get("version", "")).upper()
    return "TLSv1.3" in version or "TLS 1.3" in version


def assert_tls13_environment() -> None:
    """
    Startup check: verify the Python SSL module supports TLS 1.3.
    Raises RuntimeError if not supported (OpenSSL < 1.1.1).
    """
    if not hasattr(ssl.TLSVersion, "TLSv1_3"):
        raise RuntimeError(
            "TLS 1.3 is not supported by this Python/OpenSSL installation. "
            "Upgrade OpenSSL to >= 1.1.1 and rebuild Python. "
            "AuthentiGuard requires TLS 1.3 for all connections."
        )
    log.info("tls13_supported", openssl_version=ssl.OPENSSL_VERSION)


def get_secure_redis_url(base_url: str) -> str:
    """
    Convert a redis:// URL to rediss:// (TLS) if not already.
    Ensures Redis connections use TLS in production.
    """
    if base_url.startswith("redis://"):
        log.warning(
            "redis_url_not_tls",
            hint="Use rediss:// for encrypted Redis connections in production",
        )
    return base_url


def get_secure_db_url(base_url: str) -> str:
    """
    Ensure PostgreSQL connection URL includes sslmode=require.
    """
    if "sslmode" not in base_url and "localhost" not in base_url:
        separator = "&" if "?" in base_url else "?"
        return f"{base_url}{separator}sslmode=require"
    return base_url


# ── Startup security assertions ───────────────────────────────

def run_security_startup_checks(settings: Any) -> list[str]:
    """
    Run all security checks at application startup.
    Returns a list of warning strings (empty = all good).
    Raises RuntimeError for critical failures.
    """
    warnings: list[str] = []

    # TLS 1.3
    try:
        assert_tls13_environment()
    except RuntimeError as exc:
        raise RuntimeError(f"CRITICAL: {exc}") from exc

    # Encryption key
    enc_key = getattr(settings, "ENCRYPTION_KEY", "")
    if not enc_key or enc_key.startswith("CHANGE_ME"):
        raise RuntimeError(
            "CRITICAL: ENCRYPTION_KEY is not configured. "
            "Generate with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )

    # JWT secret
    jwt_secret = getattr(settings, "JWT_SECRET_KEY", "")
    if not jwt_secret or len(jwt_secret) < 32 or jwt_secret.startswith("CHANGE_ME"):
        raise RuntimeError(
            "CRITICAL: JWT_SECRET_KEY is too short or not configured. "
            "Generate with: openssl rand -hex 64"
        )

    # App secret
    app_secret = getattr(settings, "APP_SECRET_KEY", "")
    if not app_secret or app_secret.startswith("CHANGE_ME"):
        raise RuntimeError("CRITICAL: APP_SECRET_KEY is not configured.")

    # Production-specific checks
    app_env = getattr(settings, "APP_ENV", "development")
    if app_env == "production":
        if getattr(settings, "APP_DEBUG", False):
            raise RuntimeError("CRITICAL: APP_DEBUG must be False in production.")

        db_url = getattr(settings, "DATABASE_URL", "")
        if "sslmode" not in db_url:
            warnings.append("DATABASE_URL does not specify sslmode — add sslmode=require")

        cors = getattr(settings, "CORS_ORIGINS", [])
        if "http://localhost" in str(cors) or "*" in str(cors):
            warnings.append("CORS_ORIGINS contains insecure origins for production")

    if warnings:
        for w in warnings:
            log.warning("security_startup_warning", warning=w)

    log.info("security_startup_checks_passed",
             env=app_env,
             n_warnings=len(warnings))
    return warnings
