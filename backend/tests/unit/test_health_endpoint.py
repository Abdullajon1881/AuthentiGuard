"""
Unit tests for the health endpoint.
Verifies that /health returns 200 when all services healthy, 503 when degraded.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


@pytest_asyncio.fixture
async def client():
    from app.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_all_healthy_returns_200(self, client):
        """When DB, Redis, and Celery are all ok, return 200."""
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.rollback = AsyncMock()

        mock_insp = MagicMock()
        mock_insp.ping.return_value = {"worker1": {"ok": "pong"}}

        with patch("app.api.v1.endpoints.routes.get_db") as mock_get_db, \
             patch("app.core.redis.redis_ping", new_callable=AsyncMock, return_value=True), \
             patch("app.workers.celery_app.celery_app") as mock_celery:

            async def _fake_get_db():
                yield mock_db

            mock_get_db.return_value = _fake_get_db()
            mock_celery.control.inspect.return_value = mock_insp

            resp = await client.get("/api/v1/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["checks"]["database"] == "ok"
            assert data["checks"]["redis"] == "ok"

    @pytest.mark.asyncio
    async def test_redis_down_returns_503(self, client):
        """When Redis is down, return 503."""
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.rollback = AsyncMock()

        mock_insp = MagicMock()
        mock_insp.ping.return_value = {"worker1": {"ok": "pong"}}

        with patch("app.api.v1.endpoints.routes.get_db") as mock_get_db, \
             patch("app.core.redis.redis_ping", new_callable=AsyncMock, return_value=False), \
             patch("app.workers.celery_app.celery_app") as mock_celery:

            async def _fake_get_db():
                yield mock_db

            mock_get_db.return_value = _fake_get_db()
            mock_celery.control.inspect.return_value = mock_insp

            resp = await client.get("/api/v1/health")
            assert resp.status_code == 503
            data = resp.json()
            assert data["status"] == "degraded"
            assert data["checks"]["redis"] == "degraded"

    @pytest.mark.asyncio
    async def test_celery_down_returns_503(self, client):
        """When no Celery workers are running, return 503."""
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
            mock_celery.control.inspect.return_value.ping.return_value = None

            resp = await client.get("/api/v1/health")
            assert resp.status_code == 503
            data = resp.json()
            assert data["checks"]["celery"] == "degraded"

    @pytest.mark.asyncio
    async def test_response_includes_timestamp(self, client):
        """Health response must include a timestamp."""
        mock_db = AsyncMock()
        mock_db.execute = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.rollback = AsyncMock()

        mock_insp = MagicMock()
        mock_insp.ping.return_value = {"worker1": {"ok": "pong"}}

        with patch("app.api.v1.endpoints.routes.get_db") as mock_get_db, \
             patch("app.core.redis.redis_ping", new_callable=AsyncMock, return_value=True), \
             patch("app.workers.celery_app.celery_app") as mock_celery:

            async def _fake_get_db():
                yield mock_db

            mock_get_db.return_value = _fake_get_db()
            mock_celery.control.inspect.return_value = mock_insp

            resp = await client.get("/api/v1/health")
            data = resp.json()
            assert "timestamp" in data
