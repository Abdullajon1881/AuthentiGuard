"""
Unit tests for the health endpoint.

Verifies that `/api/v1/health` returns 200 when all services healthy,
503 when any service is degraded.

Implementation notes:
  - FastAPI captures dependency callables at decorator-evaluation time,
    so `patch("app.api.v1.endpoints.routes.get_db")` does NOT rebind the
    dependency the endpoint actually uses. We use `app.dependency_overrides`
    — FastAPI's supported test hook — instead.
  - `redis_ping` and `celery_app` are imported lazily INSIDE the endpoint
    body, so `unittest.mock.patch` at their canonical locations works.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


@pytest_asyncio.fixture
async def client():
    from app.main import app
    from app.api.v1.endpoints.routes import get_db

    # Override the DB dependency with an async generator that yields a
    # fully-mocked AsyncSession. Cleared in the finally block so parallel
    # tests in the same process do not leak state.
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock()
    mock_db.commit = AsyncMock()
    mock_db.rollback = AsyncMock()

    async def _fake_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = _fake_get_db
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
    finally:
        app.dependency_overrides.pop(get_db, None)


def _ok_celery():
    """celery_app mock where `.control.inspect().ping()` returns one worker."""
    mock_celery = MagicMock()
    mock_insp = MagicMock()
    mock_insp.ping.return_value = {"worker1": {"ok": "pong"}}
    mock_celery.control.inspect.return_value = mock_insp
    return mock_celery


def _down_celery():
    """celery_app mock where `.control.inspect().ping()` returns None."""
    mock_celery = MagicMock()
    mock_insp = MagicMock()
    mock_insp.ping.return_value = None
    mock_celery.control.inspect.return_value = mock_insp
    return mock_celery


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_all_healthy_returns_200(self, client):
        """When DB, Redis, and Celery are all ok, return 200."""
        with patch("app.core.redis.redis_ping", new_callable=AsyncMock, return_value=True), \
             patch("app.workers.celery_app.celery_app", _ok_celery()):
            resp = await client.get("/api/v1/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["checks"]["database"] == "ok"
            assert data["checks"]["redis"] == "ok"
            assert data["checks"]["celery"] == "ok"

    @pytest.mark.asyncio
    async def test_redis_down_returns_503(self, client):
        """When Redis is down, return 503."""
        with patch("app.core.redis.redis_ping", new_callable=AsyncMock, return_value=False), \
             patch("app.workers.celery_app.celery_app", _ok_celery()):
            resp = await client.get("/api/v1/health")
            assert resp.status_code == 503
            data = resp.json()
            assert data["status"] == "degraded"
            assert data["checks"]["redis"] == "degraded"

    @pytest.mark.asyncio
    async def test_celery_down_returns_503(self, client):
        """When no Celery workers are running, return 503."""
        with patch("app.core.redis.redis_ping", new_callable=AsyncMock, return_value=True), \
             patch("app.workers.celery_app.celery_app", _down_celery()):
            resp = await client.get("/api/v1/health")
            assert resp.status_code == 503
            data = resp.json()
            assert data["checks"]["celery"] == "degraded"

    @pytest.mark.asyncio
    async def test_response_includes_timestamp(self, client):
        """Health response must include a timestamp."""
        with patch("app.core.redis.redis_ping", new_callable=AsyncMock, return_value=True), \
             patch("app.workers.celery_app.celery_app", _ok_celery()):
            resp = await client.get("/api/v1/health")
            data = resp.json()
            assert "timestamp" in data
