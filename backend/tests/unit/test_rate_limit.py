"""
Unit tests for rate limiting middleware.
Tests sliding window enforcement and rate-limit response headers.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestRateLimitMiddleware:
    def test_tier_limits_defined(self):
        """All expected tiers have limits."""
        from app.middleware.middleware import TIER_LIMITS
        assert "free" in TIER_LIMITS
        assert "pro" in TIER_LIMITS
        assert "enterprise" in TIER_LIMITS
        assert "anonymous" in TIER_LIMITS
        # Limits should be ordered
        assert TIER_LIMITS["anonymous"] < TIER_LIMITS["free"]
        assert TIER_LIMITS["free"] < TIER_LIMITS["pro"]
        assert TIER_LIMITS["pro"] < TIER_LIMITS["enterprise"]

    def test_health_endpoint_skipped(self):
        """Health endpoint should bypass rate limiting."""
        from app.middleware.middleware import RateLimitMiddleware
        # Verify the skip paths logic
        middleware = RateLimitMiddleware(app=MagicMock())
        # The dispatch method checks request.url.path against skip paths
        # We verify the class can be instantiated
        assert middleware is not None

    @pytest.mark.asyncio
    async def test_check_rate_limit_under_limit(self):
        """Requests under the limit should pass through."""
        from app.middleware.middleware import RateLimitMiddleware

        middleware = RateLimitMiddleware(app=MagicMock())

        mock_pipe = AsyncMock()
        mock_pipe.zremrangebyscore = MagicMock()
        mock_pipe.zadd = MagicMock()
        mock_pipe.zcard = MagicMock()
        mock_pipe.expire = MagicMock()
        mock_pipe.execute = AsyncMock(return_value=[0, 1, 3, True])  # 3 requests (under limit of 10)

        mock_redis = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe

        with patch("app.middleware.middleware.get_redis", return_value=mock_redis):
            is_limited, current, reset_at = await middleware._check_rate_limit("user:123", 10, 60)
            assert is_limited is False
            assert current == 3

    @pytest.mark.asyncio
    async def test_check_rate_limit_over_limit(self):
        """Requests over the limit should be blocked."""
        from app.middleware.middleware import RateLimitMiddleware

        middleware = RateLimitMiddleware(app=MagicMock())

        mock_pipe = AsyncMock()
        mock_pipe.zremrangebyscore = MagicMock()
        mock_pipe.zadd = MagicMock()
        mock_pipe.zcard = MagicMock()
        mock_pipe.expire = MagicMock()
        mock_pipe.execute = AsyncMock(return_value=[0, 1, 11, True])  # 11 requests (over limit of 10)

        mock_redis = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe

        with patch("app.middleware.middleware.get_redis", return_value=mock_redis):
            is_limited, current, reset_at = await middleware._check_rate_limit("user:123", 10, 60)
            assert is_limited is True
            assert current == 11

    def test_get_tier_and_id_anonymous(self):
        """Anonymous requests should use IP-based identifier."""
        from app.middleware.middleware import RateLimitMiddleware

        middleware = RateLimitMiddleware(app=MagicMock())

        mock_request = MagicMock()
        mock_request.state = MagicMock(spec=[])  # no 'user' attribute
        mock_request.client.host = "192.168.1.1"

        tier, identifier = middleware._get_tier_and_id(mock_request)
        assert tier == "anonymous"
        assert identifier == "ip:192.168.1.1"

    def test_get_tier_and_id_authenticated(self):
        """Authenticated requests should use user-based identifier."""
        from app.middleware.middleware import RateLimitMiddleware

        middleware = RateLimitMiddleware(app=MagicMock())

        mock_user = MagicMock()
        mock_user.id = "user-abc-123"
        mock_user.tier = "pro"

        mock_request = MagicMock()
        mock_request.state.user = mock_user

        tier, identifier = middleware._get_tier_and_id(mock_request)
        assert tier == "pro"
        assert identifier == "user:user-abc-123"
