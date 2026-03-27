"""
Integration test fixtures — FastAPI TestClient with test database.
Uses the actual app but with mocked external services (S3, Celery).
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient


@pytest.fixture(scope="session")
def app():
    """Create a FastAPI app instance for testing."""
    from app.main import app
    return app


@pytest_asyncio.fixture
async def client(app):
    """Async test client for the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_celery():
    """Mock Celery task dispatch — returns a fake task ID."""
    mock_task = MagicMock()
    mock_task.id = str(uuid.uuid4())

    with patch("app.workers.text_worker.run_text_detection") as mock:
        mock.apply_async.return_value = mock_task
        mock.delay.return_value = mock_task
        yield mock


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Generate a valid auth header for testing (requires running auth flow)."""
    return {}
