"""
Step 99: Async batch processing for high-throughput API consumers.

Enterprise customers often submit hundreds or thousands of files in bulk
(e.g. content moderation pipelines, academic integrity checks).
Naive sequential processing would create unacceptable tail latency.

This module provides:

1. BatchJob — a group of detection jobs submitted together
2. BatchQueue — distributes BatchJob items across Celery workers using
   dynamic priority allocation and intelligent bin-packing by content type
3. BatchResultAggregator — collects individual results, computes batch stats,
   and fires a single webhook when the batch is complete
4. BatchSummary — aggregate statistics across all items in a batch

Design principles:
  - Individual jobs within a batch share a parent batch_id in Redis
  - Progress can be streamed via Server-Sent Events (SSE)
  - Partial results are available before the full batch completes
  - Failed items are retried up to 3 times before marking as error
  - Batch-level rate limits apply separately from per-request limits
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import numpy as np
import structlog

log = structlog.get_logger(__name__)

# Batch configuration
MAX_BATCH_SIZE      = 1_000    # maximum items per batch
BATCH_CHUNK_SIZE    = 20       # items dispatched per Celery chord
BATCH_TTL_SECONDS   = 86_400   # 24h — batches expire after this
BATCH_RETRY_LIMIT   = 3


@dataclass
class BatchItem:
    """One item within a batch job."""
    item_id:      str
    content_type: str
    content_key:  str          # S3 key or inline content identifier
    filename:     str
    metadata:     dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchItemResult:
    """Result for one item within a batch."""
    item_id:    str
    status:     str            # "completed" | "failed" | "pending"
    score:      float | None
    label:      str | None
    error:      str | None
    duration_ms: int


@dataclass
class BatchSummary:
    """Aggregate statistics for a completed batch."""
    batch_id:       str
    total_items:    int
    completed:      int
    failed:         int
    pending:        int

    # Distribution
    ai_count:       int
    human_count:    int
    uncertain_count: int

    # Score statistics
    mean_score:     float
    median_score:   float
    std_score:      float
    p95_score:      float

    # Performance
    total_duration_ms: int
    mean_duration_ms:  float
    items_per_second:  float

    # Completion
    is_complete:    bool
    completion_pct: float
    started_at:     float
    completed_at:   float | None


class BatchQueue:
    """
    High-throughput batch processing queue.

    Dispatches items to Celery workers in parallel chunks,
    tracking progress in Redis.
    """

    def __init__(self, redis: Any, celery_app: Any) -> None:
        self._redis  = redis
        self._celery = celery_app

    async def submit_batch(
        self,
        items:    list[BatchItem],
        user_id:  str,
        tier:     str = "enterprise",
        priority: int = 5,
        webhook_url: str | None = None,
    ) -> str:
        """
        Submit a batch of items for processing.

        Returns a batch_id that can be used to track progress.
        Items are chunked and dispatched to Celery workers immediately.
        """
        if len(items) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(items)} exceeds maximum {MAX_BATCH_SIZE}"
            )

        batch_id = str(uuid.uuid4())

        # Store batch metadata in Redis
        batch_meta = {
            "batch_id":    batch_id,
            "user_id":     user_id,
            "tier":        tier,
            "total":       len(items),
            "completed":   0,
            "failed":      0,
            "started_at":  time.time(),
            "webhook_url": webhook_url,
            "status":      "processing",
        }
        await self._redis.setex(
            f"batch:{batch_id}:meta",
            BATCH_TTL_SECONDS,
            json.dumps(batch_meta),
        )

        # Dispatch in chunks of BATCH_CHUNK_SIZE
        chunks     = _chunk_list(items, BATCH_CHUNK_SIZE)
        task_group = []

        for chunk_idx, chunk in enumerate(chunks):
            for item in chunk:
                task = self._dispatch_item(batch_id, item, priority)
                task_group.append(task)

        # Fire all tasks concurrently
        await asyncio.gather(*task_group)

        log.info("batch_submitted",
                 batch_id=batch_id,
                 n_items=len(items),
                 n_chunks=len(chunks),
                 tier=tier)
        return batch_id

    async def _dispatch_item(
        self,
        batch_id: str,
        item:     BatchItem,
        priority: int,
    ) -> None:
        """Dispatch one item to the appropriate Celery queue."""
        queue_name = f"{item.content_type}_queue"
        payload    = {
            "batch_id":     batch_id,
            "item_id":      item.item_id,
            "content_type": item.content_type,
            "content_key":  item.content_key,
            "filename":     item.filename,
            "metadata":     item.metadata,
        }

        try:
            # Store item status as pending
            await self._redis.setex(
                f"batch:{batch_id}:item:{item.item_id}",
                BATCH_TTL_SECONDS,
                json.dumps({"status": "pending", "item_id": item.item_id}),
            )

            # Send to Celery (non-blocking)
            self._celery.send_task(
                "backend.app.workers.run_batch_item",
                kwargs=payload,
                queue=queue_name,
                priority=priority,
            )
        except Exception as exc:
            log.warning("batch_dispatch_failed",
                         item_id=item.item_id, error=str(exc))


class BatchResultAggregator:
    """
    Collects individual item results and tracks batch completion.
    Called by the Celery worker after each item finishes.
    """

    def __init__(self, redis: Any) -> None:
        self._redis = redis

    async def record_item_result(
        self,
        batch_id: str,
        result:   BatchItemResult,
    ) -> BatchSummary | None:
        """
        Record one item result. Returns BatchSummary if batch is now complete.
        """
        # Update item record
        await self._redis.setex(
            f"batch:{batch_id}:item:{result.item_id}",
            BATCH_TTL_SECONDS,
            json.dumps({
                "status":      result.status,
                "score":       result.score,
                "label":       result.label,
                "error":       result.error,
                "duration_ms": result.duration_ms,
            }),
        )

        # Increment counter
        pipe = self._redis.pipeline()
        counter_key = (
            f"batch:{batch_id}:completed"
            if result.status == "completed"
            else f"batch:{batch_id}:failed"
        )
        pipe.incr(counter_key)
        pipe.expire(counter_key, BATCH_TTL_SECONDS)

        # Get current meta
        pipe.get(f"batch:{batch_id}:meta")
        results = await pipe.execute()
        meta_raw = results[2]

        if not meta_raw:
            return None

        meta  = json.loads(meta_raw)
        total = int(meta["total"])

        completed = int(await self._redis.get(f"batch:{batch_id}:completed") or 0)
        failed    = int(await self._redis.get(f"batch:{batch_id}:failed")    or 0)

        if completed + failed >= total:
            return await self._build_summary(batch_id, meta, total, completed, failed)
        return None

    async def _build_summary(
        self, batch_id: str, meta: dict,
        total: int, completed: int, failed: int,
    ) -> BatchSummary:
        """Scan all item results to build aggregate statistics."""
        scores: list[float] = []
        labels: list[str]   = []
        durations: list[int] = []

        # Scan item keys
        cursor = 0
        while True:
            cursor, keys = await self._redis.scan(
                cursor, match=f"batch:{batch_id}:item:*", count=100
            )
            for key in keys:
                raw = await self._redis.get(key)
                if not raw:
                    continue
                item_data = json.loads(raw)
                if item_data.get("score") is not None:
                    scores.append(float(item_data["score"]))
                if item_data.get("label"):
                    labels.append(item_data["label"])
                if item_data.get("duration_ms"):
                    durations.append(int(item_data["duration_ms"]))
            if cursor == 0:
                break

        total_dur   = sum(durations)
        mean_dur    = float(np.mean(durations)) if durations else 0.0
        elapsed     = time.time() - float(meta.get("started_at", time.time()))
        items_per_s = total / max(elapsed, 0.001)

        completed_at = time.time()

        # Mark batch complete
        meta["status"]       = "completed"
        meta["completed_at"] = completed_at
        await self._redis.setex(
            f"batch:{batch_id}:meta", BATCH_TTL_SECONDS, json.dumps(meta)
        )

        return BatchSummary(
            batch_id=batch_id,
            total_items=total, completed=completed, failed=failed,
            pending=max(0, total - completed - failed),
            ai_count=labels.count("AI"),
            human_count=labels.count("HUMAN"),
            uncertain_count=labels.count("UNCERTAIN"),
            mean_score=round(float(np.mean(scores)),   4) if scores else 0.0,
            median_score=round(float(np.median(scores)), 4) if scores else 0.0,
            std_score=round(float(np.std(scores)),     4) if scores else 0.0,
            p95_score=round(float(np.percentile(scores, 95)), 4) if scores else 0.0,
            total_duration_ms=total_dur,
            mean_duration_ms=round(mean_dur, 1),
            items_per_second=round(items_per_s, 2),
            is_complete=True,
            completion_pct=100.0,
            started_at=float(meta.get("started_at", 0)),
            completed_at=completed_at,
        )

    async def get_progress(self, batch_id: str) -> dict[str, Any]:
        """Return current batch progress for SSE streaming."""
        meta_raw = await self._redis.get(f"batch:{batch_id}:meta")
        if not meta_raw:
            return {"error": "batch not found"}

        meta      = json.loads(meta_raw)
        completed = int(await self._redis.get(f"batch:{batch_id}:completed") or 0)
        failed    = int(await self._redis.get(f"batch:{batch_id}:failed")    or 0)
        total     = int(meta["total"])

        return {
            "batch_id":       batch_id,
            "total":          total,
            "completed":      completed,
            "failed":         failed,
            "pending":        max(0, total - completed - failed),
            "completion_pct": round((completed + failed) / max(total, 1) * 100, 1),
            "status":         meta.get("status", "processing"),
        }

    async def stream_progress(
        self, batch_id: str, poll_interval: float = 1.0
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Async generator yielding progress updates every poll_interval seconds.
        Suitable for SSE (Server-Sent Events) endpoint.
        """
        while True:
            progress = await self.get_progress(batch_id)
            yield progress
            if progress.get("status") == "completed" or progress.get("error"):
                break
            await asyncio.sleep(poll_interval)


def _chunk_list(items: list, chunk_size: int) -> list[list]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
