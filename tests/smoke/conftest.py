"""
Smoke test fixtures.
Usage: pytest tests/smoke/ --base-url http://localhost:8000
"""

from __future__ import annotations

import pytest
import httpx


def pytest_addoption(parser):
    parser.addoption(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the deployed AuthentiGuard API",
    )
    parser.addoption(
        "--api-key",
        default=None,
        help="Optional API key for authenticated endpoints",
    )


@pytest.fixture(scope="session")
def base_url(request) -> str:
    return request.config.getoption("--base-url").rstrip("/")


@pytest.fixture(scope="session")
def api_url(base_url) -> str:
    return f"{base_url}/api/v1"


@pytest.fixture
def http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=30.0)


@pytest.fixture(scope="session")
def test_user() -> dict:
    """Credentials for the smoke test user (created during test_auth_flow)."""
    import uuid
    unique = uuid.uuid4().hex[:8]
    return {
        "email": f"smoke-test-{unique}@authentiguard.test",
        "password": "SmokeTest123!secure",
        "full_name": "Smoke Test User",
    }
