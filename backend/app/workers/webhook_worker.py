"""
Step 32: Webhook Service.
Delivers job completion/failure notifications to client-registered URLs.
Uses async HTTP with exponential backoff retry (3 attempts).
Payloads are HMAC-SHA256 signed so clients can verify authenticity.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog

from .celery_app import celery_app

log = structlog.get_logger(__name__)

MAX_RETRIES   = 3
RETRY_DELAYS  = [10, 30, 120]   # exponential backoff: 10s, 30s, 2min
REQUEST_TIMEOUT = 10             # seconds


def _sign_payload(payload: bytes, secret: str) -> str:
    """HMAC-SHA256 signature for webhook payload verification."""
    return "sha256=" + hmac.new(
        secret.encode(), payload, hashlib.sha256
    ).hexdigest()


@celery_app.task(
    bind=True,
    name="workers.webhook_worker.dispatch_webhook",
    queue="webhook",
    max_retries=MAX_RETRIES,
)
def dispatch_webhook(self, job_id: str, event: str) -> None:
    """
    Celery task: fetch registered webhooks for this user/event and deliver them.
    """
    import asyncio
    asyncio.run(_dispatch_async(job_id, event))


async def _dispatch_async(job_id: str, event: str) -> None:
    from ..core.database import AsyncSessionLocal
    from ..models.models import DetectionJob
    from sqlalchemy import select

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(DetectionJob).where(DetectionJob.id == uuid.UUID(job_id))
        )
        job = result.scalar_one_or_none()
        if not job or not job.user_id:
            return

        # Fetch registered webhooks for this user
        from ..models.webhook import Webhook  # imported here to keep models clean
        hooks_result = await db.execute(
            select(Webhook).where(
                Webhook.user_id == job.user_id,
                Webhook.is_active == True,
                Webhook.events.contains([event]),  # type: ignore
            )
        )
        webhooks = hooks_result.scalars().all()

    for webhook in webhooks:
        await _deliver(
            url=webhook.url,
            secret=webhook.secret or "",
            payload={
                "id":         str(uuid.uuid4()),
                "event":      event,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "data":       {"job_id": job_id, "event": event},
            },
        )


async def _deliver(url: str, secret: str, payload: dict[str, Any]) -> None:
    """
    Deliver a single webhook with retry logic.
    Each attempt uses exponential backoff.
    """
    import httpx

    body = json.dumps(payload).encode()
    signature = _sign_payload(body, secret) if secret else ""

    headers = {
        "Content-Type":            "application/json",
        "User-Agent":              "AuthentiGuard-Webhook/1.0",
        "X-AuthentiGuard-Event":   payload["event"],
        "X-AuthentiGuard-Signature": signature,
        "X-Delivery-ID":           payload["id"],
    }

    for attempt, delay in enumerate(RETRY_DELAYS, start=1):
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                resp = await client.post(url, content=body, headers=headers)
                resp.raise_for_status()
                log.info("webhook_delivered", url=url, status=resp.status_code, attempt=attempt)
                return

        except Exception as exc:
            log.warning(
                "webhook_delivery_failed",
                url=url,
                attempt=attempt,
                error=str(exc),
            )
            if attempt < len(RETRY_DELAYS):
                import asyncio
                await asyncio.sleep(delay)

    log.error("webhook_all_retries_exhausted", url=url, event=payload["event"])
