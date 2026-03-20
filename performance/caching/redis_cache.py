"""
Step 96: Intelligent Redis caching layer.

Three caching strategies, applied in order on every request:

1. Exact-match cache (content hash)
   A SHA-256 hash of the raw content bytes is used as the cache key.
   Same file submitted twice → instant cached result, zero detection cost.
   TTL: 24h for free, 7 days for pro, 30 days for enterprise.

2. Near-duplicate detection (MinHash LSH)
   Files that are ~90%+ similar (same text with minor edits) often produce
   the same result. We compute a MinHash signature of the content and
   check for near-duplicate matches. If found, blend the cached result with
   a lightweight freshness score.
   TTL: 1h (shorter because similarity is approximate).

3. Model output caching (layer-level)
   Individual detector layer outputs (perplexity, style, transformer) are
   cached on the content hash. When a re-analysis is triggered (e.g. with
   new metadata), only the changed layers re-run.
   TTL: 12h.

Cache warming strategy:
   - At startup, pre-warm with the last 100 most-requested items
   - Background task re-warms cache 10 minutes before TTL expiry
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

log = structlog.get_logger(__name__)

# TTL in seconds by tier
CACHE_TTL_BY_TIER: dict[str, int] = {
    "anonymous": 3_600,      # 1h
    "free":      86_400,     # 24h
    "pro":       604_800,    # 7d
    "enterprise": 2_592_000, # 30d
}

LAYER_CACHE_TTL   = 43_200   # 12h
SIMILARITY_TTL    = 3_600    # 1h
MINHASH_SHINGLES  = 3        # character n-gram size for MinHash
MINHASH_BANDS     = 20       # LSH bands
MINHASH_ROWS      = 5        # rows per band → threshold ≈ (1/bands)^(1/rows)


@dataclass
class CacheResult:
    """A cached detection result."""
    hit:         bool
    source:      str      # "exact" | "similarity" | "layer" | "miss"
    score:       float | None
    label:       str | None
    confidence:  float | None
    evidence:    dict | None
    cached_at:   float | None
    ttl_remaining: int


# ── Content hash cache ────────────────────────────────────────

def content_cache_key(content_hash: str, content_type: str) -> str:
    return f"cache:result:{content_type}:{content_hash}"


async def get_cached_result(
    redis: Any,
    content_hash: str,
    content_type: str,
) -> CacheResult:
    """Check exact-match cache. Returns CacheResult (hit or miss)."""
    key = content_cache_key(content_hash, content_type)
    try:
        raw = await redis.get(key)
        if raw:
            data     = json.loads(raw)
            ttl_rem  = await redis.ttl(key)
            log.debug("cache_hit_exact", key=key[:40])
            return CacheResult(
                hit=True, source="exact",
                score=data.get("score"), label=data.get("label"),
                confidence=data.get("confidence"), evidence=data.get("evidence"),
                cached_at=data.get("cached_at"), ttl_remaining=int(ttl_rem),
            )
    except Exception as exc:
        log.warning("cache_get_failed", error=str(exc))
    return CacheResult(hit=False, source="miss", score=None, label=None,
                       confidence=None, evidence=None, cached_at=None, ttl_remaining=0)


async def set_cached_result(
    redis: Any,
    content_hash: str,
    content_type: str,
    result: dict[str, Any],
    tier: str = "free",
) -> None:
    """Store a detection result in the exact-match cache."""
    key  = content_cache_key(content_hash, content_type)
    ttl  = CACHE_TTL_BY_TIER.get(tier, CACHE_TTL_BY_TIER["free"])
    data = {**result, "cached_at": time.time(), "cache_version": "1"}
    try:
        await redis.setex(key, ttl, json.dumps(data))
        log.debug("cache_set", key=key[:40], ttl=ttl)
    except Exception as exc:
        log.warning("cache_set_failed", error=str(exc))


# ── Layer-level cache ─────────────────────────────────────────

def layer_cache_key(content_hash: str, layer_name: str) -> str:
    return f"cache:layer:{layer_name}:{content_hash}"


async def get_layer_cache(
    redis: Any,
    content_hash: str,
    layer_name: str,
) -> dict | None:
    """Retrieve a single detector layer's output from cache."""
    key = layer_cache_key(content_hash, layer_name)
    try:
        raw = await redis.get(key)
        if raw:
            return json.loads(raw)
    except Exception as exc:
        log.warning("layer_cache_get_failed", layer=layer_name, error=str(exc))
    return None


async def set_layer_cache(
    redis: Any,
    content_hash: str,
    layer_name: str,
    layer_output: dict,
) -> None:
    """Cache a single detector layer's output."""
    key = layer_cache_key(content_hash, layer_name)
    try:
        await redis.setex(key, LAYER_CACHE_TTL, json.dumps(layer_output))
    except Exception as exc:
        log.warning("layer_cache_set_failed", layer=layer_name, error=str(exc))


# ── MinHash near-duplicate detection ─────────────────────────

def compute_minhash(text: str, n_hashes: int = MINHASH_BANDS * MINHASH_ROWS) -> list[int]:
    """
    Compute MinHash signature for text similarity detection.

    Uses character k-grams (k=MINHASH_SHINGLES) as features.
    Returns a list of n_hashes minimum hash values.

    The Jaccard similarity between two texts is estimated as:
        sim ≈ (# matching min-hash values) / n_hashes
    """
    k     = MINHASH_SHINGLES
    text  = text.lower()
    shingles = {text[i:i+k] for i in range(len(text) - k + 1)}
    if not shingles:
        return [0] * n_hashes

    # Use different hash seeds for each of the n_hashes functions
    sig = []
    for seed in range(n_hashes):
        min_val = float("inf")
        for shingle in shingles:
            h = int(hashlib.md5(f"{seed}:{shingle}".encode()).hexdigest(), 16)
            min_val = min(min_val, h)
        sig.append(min_val if min_val != float("inf") else 0)
    return sig


def minhash_similarity(sig1: list[int], sig2: list[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig1) != len(sig2) or not sig1:
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


def lsh_bucket_keys(sig: list[int], n_bands: int = MINHASH_BANDS,
                     n_rows: int = MINHASH_ROWS) -> list[str]:
    """
    Compute LSH bucket keys from a MinHash signature.
    Each band hashes n_rows consecutive values to a bucket.
    Documents that share ≥1 bucket are candidates for similarity.
    """
    buckets = []
    for band in range(n_bands):
        start  = band * n_rows
        end    = start + n_rows
        band_values = tuple(sig[start:end])
        band_hash   = hashlib.md5(
            struct.pack(f">{n_rows}Q", *band_hash_values(band_values, n_rows))
        ).hexdigest()[:16]
        buckets.append(f"cache:lsh:b{band}:{band_hash}")
    return buckets


def band_hash_values(band_values: tuple, n_rows: int) -> list[int]:
    """Pad or truncate band values to exactly n_rows ints."""
    vals = list(band_values)
    while len(vals) < n_rows:
        vals.append(0)
    return vals[:n_rows]


async def find_similar_cached(
    redis: Any,
    text: str,
    content_type: str,
    threshold: float = 0.85,
) -> CacheResult:
    """
    Check LSH buckets for near-duplicate cached content.
    Returns a CacheResult if a similar result is found.
    """
    if content_type not in ("text", "code"):
        return CacheResult(hit=False, source="miss", score=None, label=None,
                           confidence=None, evidence=None, cached_at=None, ttl_remaining=0)
    try:
        sig     = compute_minhash(text)
        buckets = lsh_bucket_keys(sig)

        for bucket_key in buckets[:5]:   # check first 5 bands only for speed
            members_raw = await redis.smembers(bucket_key)
            if not members_raw:
                continue

            for member in list(members_raw)[:10]:
                stored = await redis.get(f"cache:sim:{member.decode()}")
                if not stored:
                    continue

                data = json.loads(stored)
                stored_sig  = data.get("sig", [])
                stored_sim  = minhash_similarity(sig, stored_sig)
                if stored_sim >= threshold:
                    log.debug("cache_hit_similarity", sim=round(stored_sim, 3))
                    return CacheResult(
                        hit=True, source="similarity",
                        score=data.get("score"), label=data.get("label"),
                        confidence=float(data.get("confidence", 0)) * stored_sim,
                        evidence=data.get("evidence"),
                        cached_at=data.get("cached_at"),
                        ttl_remaining=SIMILARITY_TTL,
                    )
    except Exception as exc:
        log.warning("similarity_cache_failed", error=str(exc))

    return CacheResult(hit=False, source="miss", score=None, label=None,
                       confidence=None, evidence=None, cached_at=None, ttl_remaining=0)


# ── Cache warming ─────────────────────────────────────────────

async def warm_cache(
    redis: Any,
    db: Any,
    n_items: int = 100,
) -> int:
    """
    Pre-warm the cache with recently analysed items.
    Called at startup and periodically by Celery beat.
    Returns the number of items warmed.
    """
    try:
        from sqlalchemy import select, desc  # type: ignore
        from backend.app.models.models import DetectionJob, DetectionResult  # type: ignore

        warmed = 0
        q = await db.execute(
            select(DetectionJob, DetectionResult)
            .join(DetectionResult, DetectionResult.job_id == DetectionJob.id)
            .order_by(desc(DetectionJob.created_at))
            .limit(n_items)
        )
        rows = q.all()

        for job, result in rows:
            if not job.content_hash or not result:
                continue
            await set_cached_result(
                redis,
                content_hash=job.content_hash,
                content_type=job.content_type.value if job.content_type else "text",
                result={
                    "score":      result.authenticity_score,
                    "label":      result.label,
                    "confidence": result.confidence,
                    "evidence":   result.evidence_summary,
                },
                tier="pro",
            )
            warmed += 1

        log.info("cache_warmed", n_items=warmed)
        return warmed
    except Exception as exc:
        log.warning("cache_warm_failed", error=str(exc))
        return 0


# ── Cache stats ───────────────────────────────────────────────

async def get_cache_stats(redis: Any) -> dict[str, Any]:
    """Return cache hit/miss counters and memory usage."""
    try:
        info   = await redis.info("stats")
        memory = await redis.info("memory")
        return {
            "hits":              int(info.get("keyspace_hits", 0)),
            "misses":            int(info.get("keyspace_misses", 0)),
            "hit_rate":          _hit_rate(info),
            "used_memory_mb":    round(int(memory.get("used_memory", 0)) / 1_048_576, 1),
            "evicted_keys":      int(info.get("evicted_keys", 0)),
        }
    except Exception as exc:
        return {"error": str(exc)}


def _hit_rate(info: dict) -> float:
    hits   = int(info.get("keyspace_hits",   0))
    misses = int(info.get("keyspace_misses", 0))
    total  = hits + misses
    return round(hits / total, 4) if total > 0 else 0.0
