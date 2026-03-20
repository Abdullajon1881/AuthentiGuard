"""
Step 34: Redis 7 client for caching and queue coordination.
Separate logical databases: db=0 (cache), db=1 (Celery results).
"""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as aioredis

from .config import get_settings

_redis_client: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    """Return the shared async Redis client (singleton)."""
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=50,
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


async def redis_ping() -> bool:
    """Health check — returns True if Redis is reachable."""
    try:
        return await get_redis().ping()
    except Exception:
        return False
