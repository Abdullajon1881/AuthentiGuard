"""Smoke test: health check endpoint."""

import pytest
import httpx


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_health_returns_ok(api_url, http_client):
    """GET /health should return 200 with status ok."""
    async with http_client:
        resp = await http_client.get(f"{api_url}/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "timestamp" in data
