"""Smoke test: submit text for analysis, poll, and verify result."""

import pytest
import httpx
import asyncio


@pytest.mark.timeout(120)
@pytest.mark.asyncio
async def test_text_analysis_flow(api_url, http_client, test_user):
    """Submit text → poll until done → verify result schema."""
    async with http_client:
        # Login first
        resp = await http_client.post(f"{api_url}/auth/login", json={
            "email": test_user["email"],
            "password": test_user["password"],
        })
        if resp.status_code != 200:
            # Register if login fails (user not created yet)
            await http_client.post(f"{api_url}/auth/register", json={
                "email": test_user["email"],
                "password": test_user["password"],
                "full_name": test_user["full_name"],
                "consent_given": True,
            })
            resp = await http_client.post(f"{api_url}/auth/login", json={
                "email": test_user["email"],
                "password": test_user["password"],
            })
        assert resp.status_code == 200
        token = resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Submit text
        test_text = (
            "This is a sample text for smoke testing the AuthentiGuard detection pipeline. "
            "It needs to be long enough to pass the minimum length validation of twenty characters. "
            "The text detection ensemble will analyze this for signs of AI generation."
        )
        resp = await http_client.post(
            f"{api_url}/analyze/text",
            json={"text": test_text, "content_type": "text"},
            headers=headers,
        )
        assert resp.status_code == 202
        job = resp.json()
        assert "job_id" in job
        assert job["status"] == "pending"
        assert "poll_url" in job
        job_id = job["job_id"]

        # Poll until completed (max 90s)
        for _ in range(60):
            resp = await http_client.get(f"{api_url}/jobs/{job_id}", headers=headers)
            assert resp.status_code == 200
            status = resp.json()["status"]
            if status in ("completed", "failed"):
                break
            await asyncio.sleep(1.5)

        assert status == "completed", f"Job did not complete: {status}"

        # Get result
        resp = await http_client.get(f"{api_url}/jobs/{job_id}/result", headers=headers)
        assert resp.status_code == 200
        result = resp.json()

        # Verify result schema
        assert "authenticity_score" in result
        assert 0.0 <= result["authenticity_score"] <= 1.0
        assert result["label"] in ("AI", "HUMAN", "UNCERTAIN")
        assert "confidence" in result
        assert "layer_scores" in result
