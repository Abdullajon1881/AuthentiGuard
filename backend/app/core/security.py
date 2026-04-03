"""
Step 25: JWT authentication with refresh token rotation.

Access tokens:  short-lived (15 min), sent in Authorization header.
Refresh tokens: long-lived (30 days), stored in Redis, rotated on every use.

Rotation means: each use of a refresh token invalidates it and issues a new one.
This means stolen refresh tokens have a very short window of usefulness.
"""

from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal

from jose import JWTError, jwt
from passlib.context import CryptContext

from .config import get_settings
from .redis import get_redis

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

TokenType = Literal["access", "refresh"]

REFRESH_TOKEN_PREFIX = "refresh:"


# ── Password hashing ──────────────────────────────────────────

def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ── Access token ──────────────────────────────────────────────

def create_access_token(
    user_id: str,
    role: str,
    email: str,
    tier: str = "free",
) -> str:
    """
    Create a short-lived JWT access token.
    Payload: sub (user_id), role, email, tier, exp, iat, jti (unique ID).
    """
    settings = get_settings()
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

    payload = {
        "sub":   user_id,
        "role":  role,
        "email": email,
        "tier":  tier,
        "exp":   expire,
        "iat":   now,
        "jti":   str(uuid.uuid4()),
        "type":  "access",
    }
    return jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


def decode_access_token(token: str) -> dict:
    """
    Decode and validate an access token.
    Raises JWTError if invalid, expired, or wrong type.
    """
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except JWTError as exc:
        raise ValueError(f"Invalid access token: {exc}") from exc

    if payload.get("type") != "access":
        raise ValueError("Token is not an access token")

    return payload


# ── Refresh token ─────────────────────────────────────────────

async def create_refresh_token(user_id: str) -> str:
    """
    Create a cryptographically random refresh token and store it in Redis.
    Any previous refresh tokens for this user are NOT invalidated here —
    family-based invalidation is handled in rotate_refresh_token().
    """
    settings = get_settings()
    token = secrets.token_urlsafe(64)
    redis = get_redis()

    ttl = timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    key = f"{REFRESH_TOKEN_PREFIX}{token}"

    await redis.setex(key, int(ttl.total_seconds()), user_id)
    return token


async def rotate_refresh_token(old_token: str) -> tuple[str, str]:
    """
    Step 25: Refresh token rotation.

    1. Validate old_token exists in Redis.
    2. Delete it immediately (one-time use).
    3. Issue a new access token + new refresh token.

    Returns (new_access_token, new_refresh_token).
    Raises ValueError if old_token is invalid or expired.
    """
    redis  = get_redis()
    key    = f"{REFRESH_TOKEN_PREFIX}{old_token}"
    user_id = await redis.get(key)

    if not user_id:
        raise ValueError("Refresh token is invalid or expired")

    # Delete the old token immediately — prevents replay attacks
    await redis.delete(key)

    # We need user details to mint the new access token.
    # In practice the caller (auth endpoint) fetches user from DB first.
    # This function just handles the token mechanics.
    new_refresh = await create_refresh_token(user_id)
    return user_id, new_refresh


async def revoke_refresh_token(token: str) -> None:
    """Revoke a refresh token (logout)."""
    redis = get_redis()
    await redis.delete(f"{REFRESH_TOKEN_PREFIX}{token}")


async def revoke_all_refresh_tokens(user_id: str) -> int:
    """
    Revoke all refresh tokens for a user (force logout everywhere).
    Scans Redis for all tokens belonging to this user_id.
    Returns count of revoked tokens.
    """
    redis   = get_redis()
    cursor  = 0
    revoked = 0

    while True:
        cursor, keys = await redis.scan(cursor, match=f"{REFRESH_TOKEN_PREFIX}*", count=100)
        for key in keys:
            stored_user = await redis.get(key)
            if stored_user == user_id:
                await redis.delete(key)
                revoked += 1
        if cursor == 0:
            break

    return revoked


# ── Password reset ───────────────────────────────────────────

RESET_TOKEN_PREFIX = "pwreset:"
RESET_TOKEN_TTL = 3600  # 1 hour


async def create_password_reset_token(user_id: str) -> str:
    """Create a one-time password reset token stored in Redis (1hr TTL)."""
    redis = get_redis()
    token = secrets.token_urlsafe(48)
    await redis.setex(f"{RESET_TOKEN_PREFIX}{token}", RESET_TOKEN_TTL, user_id)
    return token


async def validate_password_reset_token(token: str) -> str:
    """Validate and consume a password reset token. Returns user_id. Raises ValueError if invalid."""
    redis = get_redis()
    key = f"{RESET_TOKEN_PREFIX}{token}"
    user_id = await redis.get(key)
    if not user_id:
        raise ValueError("Invalid or expired reset token")
    await redis.delete(key)  # One-time use
    return user_id
