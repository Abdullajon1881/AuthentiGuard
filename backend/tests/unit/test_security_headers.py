"""
Unit tests for security headers middleware.
Verifies X-Frame-Options, X-Content-Type-Options, etc. are set on all responses.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock, AsyncMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


@pytest_asyncio.fixture
async def client():
    from app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestSecurityHeaders:
    @pytest.mark.asyncio
    async def test_x_frame_options_deny(self, client):
        """All responses should have X-Frame-Options: DENY."""
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.rollback = AsyncMock()

        with patch("app.api.v1.endpoints.routes.get_db") as mock_get_db, \
             patch("app.core.redis.redis_ping", new_callable=AsyncMock, return_value=True), \
             patch("app.workers.celery_app.celery_app") as mock_celery:

            async def _fake_get_db():
                yield mock_db

            mock_get_db.return_value = _fake_get_db()
            mock_insp = MagicMock()
            mock_insp.ping.return_value = {"w1": {"ok": "pong"}}
            mock_celery.control.inspect.return_value = mock_insp

            resp = await client.get("/api/v1/health")
            assert resp.headers.get("x-frame-options") == "DENY"

    @pytest.mark.asyncio
    async def test_nosniff_header(self, client):
        """All responses should have X-Content-Type-Options: nosniff."""
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.rollback = AsyncMock()

        with patch("app.api.v1.endpoints.routes.get_db") as mock_get_db, \
             patch("app.core.redis.redis_ping", new_callable=AsyncMock, return_value=True), \
             patch("app.workers.celery_app.celery_app") as mock_celery:

            async def _fake_get_db():
                yield mock_db

            mock_get_db.return_value = _fake_get_db()
            mock_insp = MagicMock()
            mock_insp.ping.return_value = {"w1": {"ok": "pong"}}
            mock_celery.control.inspect.return_value = mock_insp

            resp = await client.get("/api/v1/health")
            assert resp.headers.get("x-content-type-options") == "nosniff"

    @pytest.mark.asyncio
    async def test_referrer_policy(self, client):
        """All responses should have Referrer-Policy header."""
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.rollback = AsyncMock()

        with patch("app.api.v1.endpoints.routes.get_db") as mock_get_db, \
             patch("app.core.redis.redis_ping", new_callable=AsyncMock, return_value=True), \
             patch("app.workers.celery_app.celery_app") as mock_celery:

            async def _fake_get_db():
                yield mock_db

            mock_get_db.return_value = _fake_get_db()
            mock_insp = MagicMock()
            mock_insp.ping.return_value = {"w1": {"ok": "pong"}}
            mock_celery.control.inspect.return_value = mock_insp

            resp = await client.get("/api/v1/health")
            assert resp.headers.get("referrer-policy") == "strict-origin-when-cross-origin"

    @pytest.mark.asyncio
    async def test_hsts_only_in_production(self, client):
        """HSTS should NOT be set in development mode."""
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.rollback = AsyncMock()

        with patch("app.api.v1.endpoints.routes.get_db") as mock_get_db, \
             patch("app.core.redis.redis_ping", new_callable=AsyncMock, return_value=True), \
             patch("app.workers.celery_app.celery_app") as mock_celery:

            async def _fake_get_db():
                yield mock_db

            mock_get_db.return_value = _fake_get_db()
            mock_insp = MagicMock()
            mock_insp.ping.return_value = {"w1": {"ok": "pong"}}
            mock_celery.control.inspect.return_value = mock_insp

            resp = await client.get("/api/v1/health")
            # In development mode (default), HSTS should not be present
            assert resp.headers.get("strict-transport-security") is None

    @pytest.mark.asyncio
    async def test_permissions_policy(self, client):
        """All responses should restrict camera, microphone, geolocation."""
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.rollback = AsyncMock()

        with patch("app.api.v1.endpoints.routes.get_db") as mock_get_db, \
             patch("app.core.redis.redis_ping", new_callable=AsyncMock, return_value=True), \
             patch("app.workers.celery_app.celery_app") as mock_celery:

            async def _fake_get_db():
                yield mock_db

            mock_get_db.return_value = _fake_get_db()
            mock_insp = MagicMock()
            mock_insp.ping.return_value = {"w1": {"ok": "pong"}}
            mock_celery.control.inspect.return_value = mock_insp

            resp = await client.get("/api/v1/health")
            pp = resp.headers.get("permissions-policy")
            assert pp is not None
            assert "camera=()" in pp
            assert "microphone=()" in pp
