"""
Steps 90–92: Auth security — JWT rotation, rate limiting, digital signatures.

Step 90 — JWT refresh token rotation
  Already implemented in backend/app/core/security.py.
  This module provides the security policy configuration and
  validation utilities used across services.

Step 91 — API rate limiting per tier
  Already implemented in backend/app/middleware/middleware.py.
  This module provides the rate limit policy definitions and
  a Redis-based per-key rate limit checker usable outside FastAPI.

Step 92 — Cryptographic digital signatures on every authenticity report
  Every report is signed with HMAC-SHA256.
  Already implemented in ai/authenticity-engine/reports/integrity.py.
  This module provides the asymmetric signing upgrade path (ECDSA P-256)
  for enterprise-grade non-repudiation.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

import structlog

log = structlog.get_logger(__name__)


# ── Step 90: JWT token policy ─────────────────────────────────

@dataclass
class JWTPolicy:
    """
    Centralised JWT policy. Applied consistently across all token issuance.
    Values are loaded from environment variables at startup.
    """
    algorithm:                  str   = "HS256"
    access_token_expire_minutes: int  = 15
    refresh_token_expire_days:   int  = 30
    # Rotation: a refresh token is valid for one use only.
    # After use, it is deleted and a new one is issued.
    rotation_enabled:            bool = True
    # Maximum concurrent refresh tokens per user (DoS prevention)
    max_refresh_tokens_per_user: int  = 10

    @classmethod
    def from_settings(cls, settings: Any) -> "JWTPolicy":
        return cls(
            algorithm=getattr(settings, "JWT_ALGORITHM", "HS256"),
            access_token_expire_minutes=getattr(
                settings, "JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 15
            ),
            refresh_token_expire_days=getattr(
                settings, "JWT_REFRESH_TOKEN_EXPIRE_DAYS", 30
            ),
        )

    def validate(self) -> None:
        """Assert the policy meets minimum security requirements."""
        if self.access_token_expire_minutes > 60:
            raise ValueError(
                f"Access token expiry ({self.access_token_expire_minutes} min) "
                f"exceeds maximum 60 minutes. Short-lived tokens reduce breach impact."
            )
        if self.refresh_token_expire_days > 90:
            raise ValueError(
                f"Refresh token expiry ({self.refresh_token_expire_days} days) "
                f"exceeds maximum 90 days."
            )
        if not self.rotation_enabled:
            log.warning("jwt_rotation_disabled",
                        warning="Refresh token rotation is disabled — security risk")
        log.info("jwt_policy_validated",
                 access_exp_min=self.access_token_expire_minutes,
                 refresh_exp_days=self.refresh_token_expire_days)


# ── Step 91: Rate limit policies ──────────────────────────────

@dataclass
class RateLimitPolicy:
    """Per-tier rate limit configuration."""
    tier:         str
    requests_per_minute: int
    burst_multiplier:    float = 1.5    # allow short bursts up to this multiple
    window_seconds:      int  = 60

    @property
    def burst_limit(self) -> int:
        return int(self.requests_per_minute * self.burst_multiplier)


RATE_LIMIT_POLICIES: dict[str, RateLimitPolicy] = {
    "anonymous":  RateLimitPolicy("anonymous",  requests_per_minute=5),
    "free":       RateLimitPolicy("free",        requests_per_minute=10),
    "pro":        RateLimitPolicy("pro",         requests_per_minute=100),
    "enterprise": RateLimitPolicy("enterprise",  requests_per_minute=1000),
}


class RateLimiter:
    """
    Redis-based sliding window rate limiter.
    Standalone implementation for use outside the FastAPI middleware.
    """

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    async def check(
        self,
        identifier: str,
        tier: str = "free",
    ) -> tuple[bool, int, int]:
        """
        Check rate limit for an identifier.

        Returns:
            (is_allowed, current_count, reset_timestamp)
        """
        policy = RATE_LIMIT_POLICIES.get(tier, RATE_LIMIT_POLICIES["free"])
        key    = f"ratelimit:{identifier}"
        now    = int(time.time() * 1000)
        window = policy.window_seconds * 1000

        pipe = self._redis.pipeline()
        pipe.zremrangebyscore(key, 0, now - window)
        pipe.zadd(key, {str(uuid.uuid4()): now})
        pipe.zcard(key)
        pipe.expire(key, policy.window_seconds + 1)
        results  = await pipe.execute()
        current  = int(results[2])
        reset_at = (now + window) // 1000

        return current <= policy.requests_per_minute, current, reset_at


# ── Step 92: ECDSA asymmetric signing ────────────────────────

class ReportSigner:
    """
    Asymmetric report signing using ECDSA P-256.

    Provides non-repudiation: the platform can prove it issued a report
    even if the secret key is later compromised (unlike HMAC).

    Key generation:
        from cryptography.hazmat.primitives.asymmetric import ec
        private_key = ec.generate_private_key(ec.SECP256R1())

    Verification (public side):
        signer = ReportSigner.from_public_pem(public_pem_bytes)
        signer.verify(report_bytes, signature_b64)
    """

    def __init__(self, private_key: Any = None, public_key: Any = None) -> None:
        self._private_key = private_key
        self._public_key  = public_key

    @classmethod
    def from_pem(cls, private_pem: bytes, password: bytes | None = None) -> "ReportSigner":
        """Load a signer from a PEM-encoded ECDSA private key."""
        try:
            from cryptography.hazmat.primitives.serialization import load_pem_private_key  # type: ignore
            key = load_pem_private_key(private_pem, password=password)
            return cls(private_key=key, public_key=key.public_key())
        except ImportError as exc:
            raise RuntimeError("pip install cryptography") from exc

    @classmethod
    def from_public_pem(cls, public_pem: bytes) -> "ReportSigner":
        """Load a verifier from a PEM-encoded ECDSA public key."""
        try:
            from cryptography.hazmat.primitives.serialization import load_pem_public_key  # type: ignore
            key = load_pem_public_key(public_pem)
            return cls(public_key=key)
        except ImportError as exc:
            raise RuntimeError("pip install cryptography") from exc

    def sign(self, data: bytes) -> str:
        """
        Sign data with ECDSA P-256 + SHA-256. Returns base64-encoded DER signature.
        """
        if self._private_key is None:
            raise RuntimeError("No private key loaded — cannot sign")
        try:
            from cryptography.hazmat.primitives import hashes                    # type: ignore
            from cryptography.hazmat.primitives.asymmetric import ec             # type: ignore
            signature = self._private_key.sign(data, ec.ECDSA(hashes.SHA256()))
            return base64.b64encode(signature).decode("ascii")
        except ImportError as exc:
            raise RuntimeError("pip install cryptography") from exc

    def verify(self, data: bytes, signature_b64: str) -> bool:
        """
        Verify an ECDSA signature. Returns True if valid.
        """
        if self._public_key is None:
            raise RuntimeError("No public key loaded — cannot verify")
        try:
            from cryptography.hazmat.primitives import hashes                    # type: ignore
            from cryptography.hazmat.primitives.asymmetric import ec             # type: ignore
            from cryptography.exceptions import InvalidSignature                 # type: ignore
            signature = base64.b64decode(signature_b64.encode("ascii"))
            self._public_key.verify(signature, data, ec.ECDSA(hashes.SHA256()))
            return True
        except InvalidSignature:
            return False
        except ImportError as exc:
            raise RuntimeError("pip install cryptography") from exc

    # HMAC fallback for environments without cryptography
    @staticmethod
    def hmac_sign(data: bytes, secret: str) -> str:
        return hmac.new(secret.encode(), data, hashlib.sha256).hexdigest()

    @staticmethod
    def hmac_verify(data: bytes, signature: str, secret: str) -> bool:
        expected = hmac.new(secret.encode(), data, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)
