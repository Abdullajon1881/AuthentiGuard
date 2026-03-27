"""Smoke test: full auth lifecycle — register, login, refresh, logout."""

import pytest
import httpx


@pytest.mark.timeout(30)
@pytest.mark.asyncio
async def test_full_auth_flow(api_url, http_client, test_user):
    """Register → login → refresh → logout cycle."""
    async with http_client:
        # Register
        resp = await http_client.post(f"{api_url}/auth/register", json={
            "email": test_user["email"],
            "password": test_user["password"],
            "full_name": test_user["full_name"],
            "consent_given": True,
        })
        assert resp.status_code in (201, 409), f"Register failed: {resp.text}"

        # Login
        resp = await http_client.post(f"{api_url}/auth/login", json={
            "email": test_user["email"],
            "password": test_user["password"],
        })
        assert resp.status_code == 200
        tokens = resp.json()
        assert "access_token" in tokens
        assert "refresh_token" in tokens
        assert tokens["token_type"] == "bearer"

        access_token = tokens["access_token"]
        refresh_token = tokens["refresh_token"]

        # Refresh
        resp = await http_client.post(f"{api_url}/auth/refresh", json={
            "refresh_token": refresh_token,
        })
        assert resp.status_code == 200
        new_tokens = resp.json()
        assert "access_token" in new_tokens
        new_refresh = new_tokens["refresh_token"]

        # Logout
        resp = await http_client.post(f"{api_url}/auth/logout", json={
            "refresh_token": new_refresh,
        })
        assert resp.status_code == 204
