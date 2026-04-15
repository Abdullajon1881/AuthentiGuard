"""
Step 34: Redis 7 client for caching and queue coordination.
Separate logical databases: db=0 (cache), db=1 (Celery results).
"""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as aioredis
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from .config import get_settings

_redis_client: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    """Return the shared async Redis client (singleton).

    Configured to survive Redis restarts / transient failures without hanging:
      - bounded socket timeouts so no call blocks indefinitely
      - automatic retry with exponential backoff on ConnectionError / TimeoutError
      - health_check_interval pings idle connections so stale sockets are
        detected and recycled before the next real command
    """
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=200,
            socket_connect_timeout=5,
            socket_timeout=5,
            socket_keepalive=True,
            health_check_interval=30,
            retry=Retry(ExponentialBackoff(cap=2.0, base=0.1), retries=3),
            retry_on_error=[RedisConnectionError, RedisTimeoutError],
        )
    return _redis_client


async def cache_set(key: str, value: Any, ttl_seconds: int = 300) -> None:
    """Serialize value to JSON and cache with TTL."""
    client = get_redis()
    await client.setex(key, ttl_seconds, json.dumps(value))


async def cache_get(key: str) -> Any | None:
    """Retrieve and deserialize a cached value. Returns None if missing."""
    client = get_redis()
    raw = await client.get(key)
    if raw is None:
        return None
    return json.loads(raw)


async def cache_delete(key: str) -> None:
    client = get_redis()
    await client.delete(key)


async def reset_redis() -> None:
    """Tear down the cached client so the next get_redis() builds a fresh pool.

    Call this when a command fails at the connection layer even after retries
    (e.g. Redis was restarted, the pool's sockets are permanently bad, or DNS
    resolution changed). Every consumer re-enters through get_redis(), so no
    stale handle can outlive the reset.
    """
    global _redis_client
    client = _redis_client
    _redis_client = None
    if client is not None:
        try:
            await client.aclose()
        except Exception:
            # Best-effort teardown — a pool that is already dead can't harm us.
            pass


async def redis_ping() -> bool:
    """Health check — returns True if Redis is reachable.

    On a connection-level failure we reset the singleton so the next access
    starts a new pool. This turns retry_on_error from a per-command band-aid
    into a true singleton-level recovery path.
    """
    try:
        return await get_redis().ping()
    except (RedisConnectionError, RedisTimeoutError):
        await reset_redis()
        return False
    except Exception:
        return False
