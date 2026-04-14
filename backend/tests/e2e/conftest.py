"""
E2E test fixtures — real HTTP client against a running backend.
Requires: docker compose -f docker-compose.test.yml up -d
"""

from __future__ import annotations

import os

import pytest
import httpx


BASE_URL = os.environ.get("E2E_BASE_URL", "http://localhost:8000")
E2E_TIMEOUT = float(os.environ.get("E2E_TIMEOUT", "120"))  # max seconds to poll


@pytest.fixture(scope="session")
def base_url() -> str:
    return BASE_URL


@pytest.fixture(scope="session")
def http_client() -> httpx.Client:
    """Synchronous HTTP client for E2E tests (real network calls)."""
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        yield client


@pytest.fixture(scope="session")
def poll_timeout() -> float:
    return E2E_TIMEOUT
