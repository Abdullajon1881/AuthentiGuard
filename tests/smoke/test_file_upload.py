"""Smoke test: upload a small test file and verify processing."""

import pytest
import httpx
import asyncio


def _create_minimal_text_file() -> bytes:
    """Create a minimal text file for upload testing."""
    content = (
        "This is a test file uploaded during smoke testing. "
        "It contains enough text to pass the minimum length validation check. "
        "The AuthentiGuard pipeline should detect and process this as text content."
    )
    return content.encode("utf-8")


@pytest.mark.timeout(120)
@pytest.mark.asyncio
async def test_file_upload_flow(api_url, http_client, test_user):
    """Upload file → poll → verify result."""
    async with http_client:
        # Login
        resp = await http_client.post(f"{api_url}/auth/login", json={
            "email": test_user["email"],
            "password": test_user["password"],
        })
        if resp.status_code != 200:
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

        token = resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Upload file
        file_bytes = _create_minimal_text_file()
        files = {"file": ("test_smoke.txt", file_bytes, "text/plain")}
        resp = await http_client.post(
            f"{api_url}/analyze/file",
            files=files,
            headers=headers,
        )
        assert resp.status_code == 202
        job = resp.json()
        job_id = job["job_id"]

        # Poll
        for _ in range(60):
            resp = await http_client.get(f"{api_url}/jobs/{job_id}", headers=headers)
            assert resp.status_code == 200
            status = resp.json()["status"]
            if status in ("completed", "failed"):
                break
            await asyncio.sleep(1.5)

        assert status == "completed", f"Job did not complete: {status}"

        # Verify result
        resp = await http_client.get(f"{api_url}/jobs/{job_id}/result", headers=headers)
        assert resp.status_code == 200
        result = resp.json()
        assert result["label"] in ("AI", "HUMAN", "UNCERTAIN")
