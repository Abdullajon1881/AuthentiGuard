"""
Unit tests for security module — JWT creation, password hashing, token validation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, AsyncMock

import pytest
from jose import jwt


class TestPasswordHashing:
    def test_hash_and_verify(self):
        from app.core.security import hash_password, verify_password
        hashed = hash_password("SecurePass123!")
        assert hashed != "SecurePass123!"
        assert verify_password("SecurePass123!", hashed)

    def test_wrong_password_fails(self):
        from app.core.security import hash_password, verify_password
        hashed = hash_password("CorrectPass1!")
        assert not verify_password("WrongPass1!", hashed)

    def test_different_hashes_for_same_password(self):
        """Bcrypt should generate different salts each time."""
        from app.core.security import hash_password
        h1 = hash_password("SamePass1!")
        h2 = hash_password("SamePass1!")
        assert h1 != h2  # different salts


class TestAccessToken:
    def _patch_settings(self):
        return patch("app.core.security.get_settings", return_value=type("S", (), {
            "JWT_SECRET_KEY": "test-secret-32-chars-exactly-here",
            "JWT_ALGORITHM": "HS256",
            "JWT_ACCESS_TOKEN_EXPIRE_MINUTES": 15,
        })())

    def test_create_and_decode(self):
        from app.core.security import create_access_token, decode_access_token
        with self._patch_settings():
            token = create_access_token("user-123", "api_consumer", "test@test.com")
            payload = decode_access_token(token)
            assert payload["sub"] == "user-123"
            assert payload["role"] == "api_consumer"
            assert payload["email"] == "test@test.com"
            assert payload["type"] == "access"

    def test_token_has_jti(self):
        from app.core.security import create_access_token, decode_access_token
        with self._patch_settings():
            token = create_access_token("user-123", "admin", "a@b.com")
            payload = decode_access_token(token)
            assert "jti" in payload
            # JTI should be a valid UUID
            uuid.UUID(payload["jti"])

    def test_invalid_token_raises(self):
        from app.core.security import decode_access_token
        with self._patch_settings():
            with pytest.raises(ValueError, match="Invalid"):
                decode_access_token("not.a.real.token")

    def test_refresh_token_rejected_as_access(self):
        """Tokens with type=refresh must be rejected by decode_access_token."""
        from app.core.security import decode_access_token
        with self._patch_settings():
            now = datetime.now(timezone.utc)
            payload = {
                "sub": "user-123", "role": "free", "email": "t@t.com",
                "exp": now + timedelta(days=30),
                "iat": now, "jti": str(uuid.uuid4()),
                "type": "refresh",
            }
            token = jwt.encode(payload, "test-secret-32-chars-exactly-here", algorithm="HS256")
            with pytest.raises(ValueError, match="not an access token"):
                decode_access_token(token)

    def test_expired_token_raises(self):
        """Expired tokens should raise ValueError."""
        from app.core.security import decode_access_token
        with self._patch_settings():
            now = datetime.now(timezone.utc)
            payload = {
                "sub": "user-123", "role": "free", "email": "t@t.com",
                "exp": now - timedelta(hours=1),  # expired 1 hour ago
                "iat": now - timedelta(hours=2),
                "jti": str(uuid.uuid4()),
                "type": "access",
            }
            token = jwt.encode(payload, "test-secret-32-chars-exactly-here", algorithm="HS256")
            with pytest.raises(ValueError, match="Invalid"):
                decode_access_token(token)


class TestRefreshToken:
    @pytest.mark.asyncio
    async def test_create_stores_in_redis(self):
        from app.core.security import create_refresh_token

        mock_redis = AsyncMock()
        mock_redis.setex = AsyncMock()

        with patch("app.core.security.get_settings", return_value=type("S", (), {
            "JWT_REFRESH_TOKEN_EXPIRE_DAYS": 30,
        })()), \
             patch("app.core.security.get_redis", return_value=mock_redis):
            token = await create_refresh_token("user-123")
            assert isinstance(token, str)
            assert len(token) > 32  # should be a long random string
            mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_rotate_deletes_old_token(self):
        from app.core.security import rotate_refresh_token

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value="user-123")
        mock_redis.delete = AsyncMock()
        mock_redis.setex = AsyncMock()

        with patch("app.core.security.get_settings", return_value=type("S", (), {
            "JWT_REFRESH_TOKEN_EXPIRE_DAYS": 30,
        })()), \
             patch("app.core.security.get_redis", return_value=mock_redis):
            user_id, new_token = await rotate_refresh_token("old-token-abc")
            assert user_id == "user-123"
            assert isinstance(new_token, str)
            # Old token should be deleted
            mock_redis.delete.assert_called_once_with("refresh:old-token-abc")

    @pytest.mark.asyncio
    async def test_rotate_invalid_token_raises(self):
        from app.core.security import rotate_refresh_token

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)  # token not found

        with patch("app.core.security.get_redis", return_value=mock_redis):
            with pytest.raises(ValueError, match="invalid or expired"):
                await rotate_refresh_token("nonexistent-token")

    @pytest.mark.asyncio
    async def test_revoke_deletes_from_redis(self):
        from app.core.security import revoke_refresh_token

        mock_redis = AsyncMock()
        mock_redis.delete = AsyncMock()

        with patch("app.core.security.get_redis", return_value=mock_redis):
            await revoke_refresh_token("token-to-revoke")
            mock_redis.delete.assert_called_once_with("refresh:token-to-revoke")
