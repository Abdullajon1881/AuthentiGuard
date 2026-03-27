"""
Integration test: full analysis flow.
Register → login → submit text → poll → get result.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_register_and_login(client):
    """Register a new user and login successfully."""
    email = f"test-{uuid.uuid4().hex[:8]}@test.com"

    # Register
    resp = await client.post("/api/v1/auth/register", json={
        "email": email,
        "password": "TestPassword123!",
        "full_name": "Test User",
        "consent_given": True,
    })
    assert resp.status_code == 201
    user = resp.json()
    assert user["email"] == email

    # Login
    resp = await client.post("/api/v1/auth/login", json={
        "email": email,
        "password": "TestPassword123!",
    })
    assert resp.status_code == 200
    tokens = resp.json()
    assert "access_token" in tokens
    assert "refresh_token" in tokens


@pytest.mark.asyncio
async def test_submit_text_creates_job(client, mock_celery):
    """Submitting text should create a detection job."""
    email = f"test-{uuid.uuid4().hex[:8]}@test.com"

    # Setup user
    await client.post("/api/v1/auth/register", json={
        "email": email,
        "password": "TestPassword123!",
        "consent_given": True,
    })
    resp = await client.post("/api/v1/auth/login", json={
        "email": email,
        "password": "TestPassword123!",
    })
    token = resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Submit text
    resp = await client.post("/api/v1/analyze/text", json={
        "text": "This is a test text that needs to be at least twenty characters for validation.",
        "content_type": "text",
    }, headers=headers)
    assert resp.status_code == 202
    job = resp.json()
    assert "job_id" in job
    assert job["status"] == "pending"
    assert job["content_type"] == "text"

    # Verify Celery was called
    mock_celery.apply_async.assert_called_once()


@pytest.mark.asyncio
async def test_duplicate_registration_returns_409(client):
    """Registering with an existing email should return 409."""
    email = f"test-{uuid.uuid4().hex[:8]}@test.com"

    await client.post("/api/v1/auth/register", json={
        "email": email,
        "password": "TestPassword123!",
        "consent_given": True,
    })

    resp = await client.post("/api/v1/auth/register", json={
        "email": email,
        "password": "TestPassword123!",
        "consent_given": True,
    })
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_wrong_password_returns_401(client):
    """Login with wrong password should return 401."""
    email = f"test-{uuid.uuid4().hex[:8]}@test.com"

    await client.post("/api/v1/auth/register", json={
        "email": email,
        "password": "TestPassword123!",
        "consent_given": True,
    })

    resp = await client.post("/api/v1/auth/login", json={
        "email": email,
        "password": "WrongPassword456!",
    })
    assert resp.status_code == 401
