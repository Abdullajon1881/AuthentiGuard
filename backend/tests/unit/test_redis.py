"""
Unit tests for Redis client — caching, health check.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


class TestRedisClient:
    def test_singleton_creation(self):
        """get_redis should return the same client instance."""
        import app.core.redis as redis_mod
        # Reset singleton
        redis_mod._redis_client = None

        mock_from_url = AsyncMock()
        with patch("redis.asyncio.from_url", return_value=mock_from_url):
            client1 = redis_mod.get_redis()
            client2 = redis_mod.get_redis()
            assert client1 is client2

        # Cleanup
        redis_mod._redis_client = None


class TestRedisCache:
    @pytest.mark.asyncio
    async def test_cache_set_and_get(self):
        from app.core.redis import cache_set, cache_get

        mock_redis = AsyncMock()
        mock_redis.setex = AsyncMock()
        mock_redis.get = AsyncMock(return_value='{"key": "value"}')

        with patch("app.core.redis.get_redis", return_value=mock_redis):
            await cache_set("test-key", {"key": "value"}, ttl_seconds=60)
            mock_redis.setex.assert_called_once()

            result = await cache_get("test-key")
            assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_cache_get_miss(self):
        from app.core.redis import cache_get

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)

        with patch("app.core.redis.get_redis", return_value=mock_redis):
            result = await cache_get("nonexistent-key")
            assert result is None

    @pytest.mark.asyncio
    async def test_cache_delete(self):
        from app.core.redis import cache_delete

        mock_redis = AsyncMock()
        mock_redis.delete = AsyncMock()

        with patch("app.core.redis.get_redis", return_value=mock_redis):
            await cache_delete("test-key")
            mock_redis.delete.assert_called_once_with("test-key")


class TestRedisHealthCheck:
    @pytest.mark.asyncio
    async def test_ping_success(self):
        from app.core.redis import redis_ping

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)

        with patch("app.core.redis.get_redis", return_value=mock_redis):
            assert await redis_ping() is True

    @pytest.mark.asyncio
    async def test_ping_failure(self):
        from app.core.redis import redis_ping

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=ConnectionError("Redis down"))

        with patch("app.core.redis.get_redis", return_value=mock_redis):
            assert await redis_ping() is False
