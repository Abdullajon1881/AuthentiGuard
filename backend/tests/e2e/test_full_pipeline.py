"""
E2E integration test: full submit -> queue -> worker -> result pipeline.

Requires a running stack (postgres + redis + backend + worker):
    docker compose -f docker-compose.test.yml up -d
    pytest tests/e2e/ -v

No mocks. Real database, real Redis, real Celery worker.
"""

from __future__ import annotations

import time
import uuid

import pytest
import httpx


# ── Sample texts ─────────────────────────────────────────────

AI_TEXT = (
    "The implementation of advanced neural network architectures has "
    "fundamentally transformed the landscape of natural language processing. "
    "These sophisticated models leverage attention mechanisms and transformer "
    "architectures to achieve unprecedented performance across a diverse "
    "array of linguistic tasks, including but not limited to text generation, "
    "sentiment analysis, and machine translation. The paradigm shift brought "
    "about by large language models represents a significant advancement in "
    "artificial intelligence research and development."
)

HUMAN_TEXT = (
    "I went to the grocery store yesterday and forgot my list again. "
    "Ended up buying way too much cheese, as usual. My dog was so "
    "excited when I got home - she always thinks the bags are for her. "
    "Spent the evening trying a new pasta recipe from that Italian "
    "cookbook my mom gave me last Christmas. It turned out okay, "
    "nothing special, but the leftovers will be good for lunch."
)


# ── Helpers ──────────────────────────────────────────────────

def _register_and_login(client: httpx.Client) -> dict[str, str]:
    """Register a fresh user and return auth headers."""
    email = f"e2e-{uuid.uuid4().hex[:8]}@test.local"
    password = "E2eTestPass123!"

    resp = client.post("/api/v1/auth/register", json={
        "email": email,
        "password": password,
        "full_name": "E2E Test User",
        "consent_given": True,
    })
    assert resp.status_code == 201, f"Register failed: {resp.status_code} {resp.text}"

    resp = client.post("/api/v1/auth/login", json={
        "email": email,
        "password": password,
    })
    assert resp.status_code == 200, f"Login failed: {resp.status_code} {resp.text}"
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _submit_text(client: httpx.Client, text: str, headers: dict) -> dict:
    """Submit text for analysis, return job response."""
    resp = client.post("/api/v1/analyze/text", json={
        "text": text,
        "content_type": "text",
    }, headers=headers)
    assert resp.status_code == 202, f"Submit failed: {resp.status_code} {resp.text}"
    return resp.json()


def _poll_until_done(
    client: httpx.Client,
    job_id: str,
    headers: dict,
    timeout: float,
) -> dict:
    """Poll job status until completed or failed. Returns status response."""
    deadline = time.monotonic() + timeout
    last_status = None

    while time.monotonic() < deadline:
        resp = client.get(f"/api/v1/jobs/{job_id}", headers=headers)
        assert resp.status_code == 200, f"Poll failed: {resp.status_code} {resp.text}"
        data = resp.json()
        last_status = data["status"]

        if last_status in ("completed", "failed"):
            return data

        time.sleep(1.0)

    pytest.fail(f"Job {job_id} did not finish within {timeout}s (last status: {last_status})")


def _get_result(client: httpx.Client, job_id: str, headers: dict) -> dict:
    """Fetch the detection result for a completed job."""
    resp = client.get(f"/api/v1/jobs/{job_id}/result", headers=headers)
    assert resp.status_code == 200, f"Result fetch failed: {resp.status_code} {resp.text}"
    return resp.json()


# ── Tests ────────────────────────────────────────────────────

class TestHealthCheck:
    """Verify the stack is running before other tests."""

    def test_health_endpoint(self, http_client: httpx.Client):
        resp = http_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["checks"]["database"] == "ok"
        assert data["checks"]["redis"] == "ok"


class TestFullTextPipeline:
    """Full submit -> queue -> worker -> result flow."""

    def test_submit_poll_result(self, http_client: httpx.Client, poll_timeout: float):
        """Core pipeline: submit text, poll to completion, validate result schema."""
        headers = _register_and_login(http_client)

        # Submit
        job = _submit_text(http_client, AI_TEXT, headers)
        job_id = job["job_id"]
        assert job["status"] == "pending"
        assert job["content_type"] == "text"
        assert "poll_url" in job

        # Poll until done
        start = time.monotonic()
        status_resp = _poll_until_done(http_client, job_id, headers, poll_timeout)
        elapsed = time.monotonic() - start
        assert status_resp["status"] == "completed", (
            f"Job failed: {status_resp}"
        )

        # Fetch result
        result = _get_result(http_client, job_id, headers)

        # ── Schema assertions ────────────────────────────
        assert result["job_id"] == job_id
        assert result["status"] == "completed"
        assert result["content_type"] == "text"

        # Score is a float in [0, 1]
        score = result["authenticity_score"]
        assert isinstance(score, (int, float))
        assert 0.0 <= score <= 1.0

        confidence = result["confidence"]
        assert isinstance(confidence, (int, float))
        assert 0.0 <= confidence <= 1.0

        # Label is one of the valid values
        assert result["label"] in ("AI", "HUMAN", "UNCERTAIN")

        # Layer scores present (at least perplexity and stylometry)
        layers = result["layer_scores"]
        assert layers is not None
        assert "perplexity" in layers
        assert "stylometry" in layers

        # Detector mode reflects actual configuration
        assert result["detector_mode"] in ("ml", "fallback")

        # Processing time is reasonable (< timeout)
        if result.get("processing_ms") is not None:
            assert result["processing_ms"] < poll_timeout * 1000

        # Timing: job completed within the polling window
        assert elapsed < poll_timeout, f"Job took {elapsed:.1f}s (timeout: {poll_timeout}s)"

    def test_ai_text_scores_high(self, http_client: httpx.Client, poll_timeout: float):
        """AI-generated text should score above 0.5 (likely AI)."""
        headers = _register_and_login(http_client)
        job = _submit_text(http_client, AI_TEXT, headers)
        _poll_until_done(http_client, job["job_id"], headers, poll_timeout)
        result = _get_result(http_client, job["job_id"], headers)

        assert result["status"] == "completed"
        # AI text should score above 0.5 (even heuristic mode should catch this)
        assert result["authenticity_score"] >= 0.5, (
            f"AI text scored only {result['authenticity_score']:.3f} — expected >= 0.5"
        )

    def test_human_text_scores_low(self, http_client: httpx.Client, poll_timeout: float):
        """Human-written text should score below 0.5 (likely human)."""
        headers = _register_and_login(http_client)
        job = _submit_text(http_client, HUMAN_TEXT, headers)
        _poll_until_done(http_client, job["job_id"], headers, poll_timeout)
        result = _get_result(http_client, job["job_id"], headers)

        assert result["status"] == "completed"
        # Human text should score below 0.5
        assert result["authenticity_score"] <= 0.5, (
            f"Human text scored {result['authenticity_score']:.3f} — expected <= 0.5"
        )

    def test_anonymous_submission(self, http_client: httpx.Client, poll_timeout: float):
        """Submit without auth — should use demo user and still work."""
        resp = http_client.post("/api/v1/analyze/text", json={
            "text": AI_TEXT,
            "content_type": "text",
        })
        # Should succeed (202) or get rate-limited (429) — not 401/403
        if resp.status_code == 429:
            pytest.skip("Rate limited — anonymous submission blocked by rate limiter")
        assert resp.status_code == 202, f"Anonymous submit: {resp.status_code} {resp.text}"

        job_id = resp.json()["job_id"]
        status_resp = _poll_until_done(http_client, job_id, {}, poll_timeout)
        assert status_resp["status"] == "completed"

    def test_short_text_rejected(self, http_client: httpx.Client):
        """Text shorter than 20 chars should be rejected at the API level."""
        headers = _register_and_login(http_client)
        resp = http_client.post("/api/v1/analyze/text", json={
            "text": "Too short",
            "content_type": "text",
        }, headers=headers)
        assert resp.status_code == 422  # Validation error


class TestJobStateTransitions:
    """Verify correct job lifecycle states."""

    def test_pending_then_completed(self, http_client: httpx.Client, poll_timeout: float):
        """Job should transition: pending -> processing -> completed."""
        headers = _register_and_login(http_client)
        job = _submit_text(http_client, AI_TEXT, headers)
        job_id = job["job_id"]

        # Initial state is pending
        assert job["status"] == "pending"

        # Poll — capture intermediate states
        observed_states = {"pending"}
        deadline = time.monotonic() + poll_timeout

        while time.monotonic() < deadline:
            resp = http_client.get(f"/api/v1/jobs/{job_id}", headers=headers)
            data = resp.json()
            observed_states.add(data["status"])

            if data["status"] in ("completed", "failed"):
                break
            time.sleep(0.5)

        assert "completed" in observed_states, (
            f"Job never reached 'completed'. States seen: {observed_states}"
        )
        # completed_at should be set
        final = http_client.get(f"/api/v1/jobs/{job_id}", headers=headers).json()
        assert final["completed_at"] is not None

    def test_nonexistent_job_returns_404(self, http_client: httpx.Client):
        """Requesting a non-existent job should return 404."""
        headers = _register_and_login(http_client)
        fake_id = str(uuid.uuid4())
        resp = http_client.get(f"/api/v1/jobs/{fake_id}", headers=headers)
        assert resp.status_code == 404
