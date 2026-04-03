"""
Step 27: Processing Queue — Redis + Celery with priority queues.

Queues:
  text   — text/code detection jobs
  image  — image detection jobs
  audio  — audio detection jobs
  video  — video detection jobs (heaviest — separate workers)

Priority: each queue supports 10 priority levels (0=low, 9=high).
Enterprise jobs are submitted at priority 9, free tier at priority 1.
"""

from __future__ import annotations

from celery import Celery  # type: ignore

from ..core.config import get_settings

_settings = get_settings()

celery_app = Celery(
    "authentiguard",
    broker=_settings.CELERY_BROKER_URL,
    backend=_settings.CELERY_RESULT_BACKEND,
    include=[
        "app.workers.text_worker",
        "app.workers.image_worker",
        "app.workers.audio_worker",
        "app.workers.video_worker",
        "app.workers.webhook_worker",
        "app.workers.cleanup",
    ],
)

celery_app.conf.update(
    # ── Serialization ──────────────────────────────────────────
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # ── Queues ─────────────────────────────────────────────────
    task_queues={
        "text":    {"exchange": "text",    "routing_key": "text"},
        "image":   {"exchange": "image",   "routing_key": "image"},
        "audio":   {"exchange": "audio",   "routing_key": "audio"},
        "video":   {"exchange": "video",   "routing_key": "video"},
        "webhook": {"exchange": "webhook", "routing_key": "webhook"},
    },
    task_default_queue="text",
    task_default_exchange="text",
    task_default_routing_key="text",

    # ── Priority support ───────────────────────────────────────
    task_queue_max_priority=10,
    task_default_priority=5,
    worker_prefetch_multiplier=1,  # process one task at a time per worker (GPU)

    # ── Reliability ────────────────────────────────────────────
    task_acks_late=True,         # ack only after task completes
    task_reject_on_worker_lost=True,
    task_track_started=True,

    # ── Timeouts ───────────────────────────────────────────────
    task_soft_time_limit=120,    # seconds — sends SoftTimeLimitExceeded
    task_time_limit=180,         # hard kill after 180s

    # ── Result retention ───────────────────────────────────────
    result_expires=86400,        # Celery result TTL: 24h (full result in Postgres)

    # ── Timezone ───────────────────────────────────────────────
    timezone="UTC",
    enable_utc=True,
)


CONTENT_TYPE_TO_QUEUE = {
    "text":  "text",
    "code":  "text",
    "image": "image",
    "audio": "audio",
    "video": "video",
}

TIER_TO_PRIORITY = {
    "free":       1,
    "pro":        5,
    "enterprise": 9,
}


# ── Periodic tasks ────────────────────────────────────────────

celery_app.conf.beat_schedule = {
    "cleanup-stuck-jobs": {
        "task": "app.workers.cleanup.cleanup_stuck_jobs",
        "schedule": 300.0,  # every 5 minutes
    },
}
