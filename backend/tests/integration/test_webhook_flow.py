"""
Integration test: webhook management CRUD.
Create → list → update → delete webhooks.
"""

from __future__ import annotations

import uuid

import pytest


async def _get_auth_headers(client) -> dict:
    """Helper: register + login, return auth headers."""
    email = f"test-{uuid.uuid4().hex[:8]}@test.com"
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
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_webhook_crud(client):
    """Full webhook lifecycle: create → list → delete."""
    headers = await _get_auth_headers(client)

    # Create
    resp = await client.post("/api/v1/webhooks", json={
        "url": "https://example.com/webhook",
        "events": ["job.completed"],
        "secret": "mysecretkey1234567",
    }, headers=headers)
    assert resp.status_code == 201
    webhook = resp.json()
    assert webhook["url"] == "https://example.com/webhook"
    webhook_id = webhook["id"]

    # List
    resp = await client.get("/api/v1/webhooks", headers=headers)
    assert resp.status_code == 200
    webhooks = resp.json()
    assert len(webhooks) >= 1
    assert any(w["id"] == webhook_id for w in webhooks)

    # Delete
    resp = await client.delete(f"/api/v1/webhooks/{webhook_id}", headers=headers)
    assert resp.status_code == 204

    # Verify deleted
    resp = await client.get("/api/v1/webhooks", headers=headers)
    assert not any(w["id"] == webhook_id for w in resp.json())


@pytest.mark.asyncio
async def test_webhook_limit_enforced(client):
    """Cannot create more than 10 webhooks."""
    headers = await _get_auth_headers(client)

    for i in range(10):
        resp = await client.post("/api/v1/webhooks", json={
            "url": f"https://example.com/hook{i}",
            "events": ["job.completed"],
            "secret": f"secret{i}key1234567",
        }, headers=headers)
        assert resp.status_code == 201

    # 11th should fail
    resp = await client.post("/api/v1/webhooks", json={
        "url": "https://example.com/hook-overflow",
        "events": ["job.completed"],
        "secret": "overflow_secret1234",
    }, headers=headers)
    assert resp.status_code == 400
