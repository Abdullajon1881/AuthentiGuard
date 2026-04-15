"""
Focused tests for the final hardening pass:

  Fix 1: base_worker._claim_job stale PROCESSING reclaim
  Fix 2: config.py S3 creds resolution from *_FILE (Docker secrets)
  Fix 3: redis.py singleton reset on connection failure

These tests exercise the exact behaviours the hardening pass added.
They use in-memory fakes — no Docker, no Postgres, no Redis required.
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Make `app.*` importable from repo root when pytest is run from backend/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ═════════════════════════════════════════════════════════════
# Fix 1 — _claim_job stale PROCESSING reclaim
# ═════════════════════════════════════════════════════════════

class _JobStatus:
    PENDING = SimpleNamespace(value="pending")
    PROCESSING = SimpleNamespace(value="processing")
    COMPLETED = SimpleNamespace(value="completed")
    FAILED = SimpleNamespace(value="failed")


class _FakeResult:
    def __init__(self, rowcount=0, scalar=None):
        self.rowcount = rowcount
        self._scalar = scalar

    def scalar_one_or_none(self):
        return self._scalar


class _FakeDB:
    """Minimal AsyncSession stub that plays back a scripted sequence of
    results for `.execute()` calls, in order.
    """

    def __init__(self, scripted_results):
        self._scripted = list(scripted_results)
        self.commits = 0

    async def execute(self, *_args, **_kwargs):
        return self._scripted.pop(0)

    async def commit(self):
        self.commits += 1


def _import_base_worker():
    # Patched import: the real module imports models & structlog. Provide
    # lightweight fakes only where needed.
    from app.workers import base_worker as bw  # type: ignore
    return bw


@pytest.mark.asyncio
async def test_claim_job_succeeds_on_first_attempt():
    bw = _import_base_worker()

    job = SimpleNamespace(
        id=uuid.uuid4(),
        version=1,
        status=bw.JobStatus.PENDING,
    )
    db = _FakeDB([_FakeResult(rowcount=1)])

    new_version = await bw._claim_job(db, job, str(job.id), "text")
    assert new_version == 2
    assert db.commits == 1


@pytest.mark.asyncio
async def test_claim_job_skips_when_row_is_live_processing():
    """PROCESSING with fresh started_at must return _ALREADY_HANDLED — the
    row is owned by another live worker, we don't reclaim."""
    bw = _import_base_worker()

    job = SimpleNamespace(id=uuid.uuid4(), version=1, status=bw.JobStatus.PENDING)
    # Re-fetched row shows another worker has claimed it just now.
    fresh = SimpleNamespace(
        id=job.id,
        version=2,
        status=bw.JobStatus.PROCESSING,
        started_at=datetime.now(timezone.utc),  # fresh → live
    )
    db = _FakeDB([
        _FakeResult(rowcount=0),          # our PENDING→PROCESSING loses
        _FakeResult(scalar=fresh),        # re-fetch returns live PROCESSING
    ])

    result = await bw._claim_job(db, job, str(job.id), "text")
    assert result is bw._ALREADY_HANDLED


@pytest.mark.asyncio
async def test_claim_job_reclaims_stale_processing():
    """PROCESSING with a started_at older than STALE_PROCESSING_MINUTES must
    be reclaimed (version bump + new started_at), NOT skipped."""
    bw = _import_base_worker()

    job = SimpleNamespace(id=uuid.uuid4(), version=1, status=bw.JobStatus.PENDING)
    stale_at = datetime.now(timezone.utc) - timedelta(
        minutes=bw.STALE_PROCESSING_MINUTES + 1
    )
    fresh = SimpleNamespace(
        id=job.id,
        version=2,
        status=bw.JobStatus.PROCESSING,
        started_at=stale_at,
    )
    db = _FakeDB([
        _FakeResult(rowcount=0),          # PENDING→PROCESSING loses (row is PROCESSING)
        _FakeResult(scalar=fresh),        # re-fetch: stale PROCESSING
        _FakeResult(rowcount=1),          # reclaim update wins
    ])

    new_version = await bw._claim_job(db, job, str(job.id), "text")
    assert new_version == 3  # fresh.version (2) + 1
    assert db.commits == 2   # initial claim + reclaim


@pytest.mark.asyncio
async def test_claim_job_treats_null_started_at_as_stale():
    """Defensive: if started_at is None, treat as stale and reclaim."""
    bw = _import_base_worker()

    job = SimpleNamespace(id=uuid.uuid4(), version=1, status=bw.JobStatus.PENDING)
    fresh = SimpleNamespace(
        id=job.id, version=2, status=bw.JobStatus.PROCESSING, started_at=None,
    )
    db = _FakeDB([
        _FakeResult(rowcount=0),
        _FakeResult(scalar=fresh),
        _FakeResult(rowcount=1),
    ])

    new_version = await bw._claim_job(db, job, str(job.id), "text")
    assert new_version == 3


@pytest.mark.asyncio
async def test_claim_job_terminal_states_return_already_handled():
    bw = _import_base_worker()

    for status in (bw.JobStatus.COMPLETED, bw.JobStatus.FAILED):
        job = SimpleNamespace(id=uuid.uuid4(), version=1, status=bw.JobStatus.PENDING)
        fresh = SimpleNamespace(
            id=job.id, version=2, status=status,
            started_at=datetime.now(timezone.utc),
        )
        db = _FakeDB([
            _FakeResult(rowcount=0),
            _FakeResult(scalar=fresh),
        ])
        result = await bw._claim_job(db, job, str(job.id), "text")
        assert result is bw._ALREADY_HANDLED, f"expected skip for {status}"


# ═════════════════════════════════════════════════════════════
# Fix 2 — config.py resolves S3 creds from *_FILE envs
# ═════════════════════════════════════════════════════════════

def _build_env_for_config(tmp_path: Path, use_file: bool) -> dict:
    """Build a minimal env dict that satisfies the Settings model."""
    base = {
        "APP_SECRET_KEY": "x" * 32,
        "DATABASE_URL": "postgresql+asyncpg://u:p@h/db",
        "REDIS_URL": "redis://localhost:6379/0",
        "JWT_SECRET_KEY": "y" * 64,
        "CELERY_BROKER_URL": "redis://localhost:6379/0",
        "CELERY_RESULT_BACKEND": "redis://localhost:6379/1",
        "ENCRYPTION_KEY": "z" * 32,
        # Explicitly clear any plain values that may leak from the process env:
        "AWS_ACCESS_KEY_ID": "",
        "AWS_SECRET_ACCESS_KEY": "",
    }
    if use_file:
        user_file = tmp_path / "akid"
        secret_file = tmp_path / "secret"
        user_file.write_text("svcacct-app-key")
        secret_file.write_text("svcacct-app-secret\n")  # trailing newline stripped
        base["AWS_ACCESS_KEY_ID_FILE"] = str(user_file)
        base["AWS_SECRET_ACCESS_KEY_FILE"] = str(secret_file)
    else:
        base["AWS_ACCESS_KEY_ID"] = "plain-key"
        base["AWS_SECRET_ACCESS_KEY"] = "plain-secret"
    return base


def test_config_reads_s3_creds_from_file(tmp_path):
    # Reload module with a clean lru_cache
    import importlib
    from app.core import config as cfg
    importlib.reload(cfg)

    env = _build_env_for_config(tmp_path, use_file=True)
    with patch.dict(os.environ, env, clear=False):
        cfg.get_settings.cache_clear()
        settings = cfg.get_settings()
        assert settings.AWS_ACCESS_KEY_ID == "svcacct-app-key"
        assert settings.AWS_SECRET_ACCESS_KEY == "svcacct-app-secret"


def test_config_plain_env_still_works_for_dev(tmp_path):
    import importlib
    from app.core import config as cfg
    importlib.reload(cfg)

    env = _build_env_for_config(tmp_path, use_file=False)
    # Explicitly unset *_FILE so plain path is taken
    env["AWS_ACCESS_KEY_ID_FILE"] = ""
    env["AWS_SECRET_ACCESS_KEY_FILE"] = ""
    with patch.dict(os.environ, env, clear=False):
        cfg.get_settings.cache_clear()
        settings = cfg.get_settings()
        assert settings.AWS_ACCESS_KEY_ID == "plain-key"
        assert settings.AWS_SECRET_ACCESS_KEY == "plain-secret"


def test_config_raises_when_no_creds_anywhere(tmp_path):
    import importlib
    from app.core import config as cfg
    importlib.reload(cfg)

    env = _build_env_for_config(tmp_path, use_file=False)
    env["AWS_ACCESS_KEY_ID"] = ""
    env["AWS_SECRET_ACCESS_KEY"] = ""
    env["AWS_ACCESS_KEY_ID_FILE"] = ""
    env["AWS_SECRET_ACCESS_KEY_FILE"] = ""
    with patch.dict(os.environ, env, clear=False):
        cfg.get_settings.cache_clear()
        with pytest.raises(Exception):  # pydantic wraps as ValidationError
            cfg.get_settings()


# ═════════════════════════════════════════════════════════════
# Fix 3 — redis.py singleton reset on connection failure
# ═════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_reset_redis_clears_singleton_and_closes_client():
    from app.core import redis as redis_mod

    fake = MagicMock()
    fake.aclose = AsyncMock()
    redis_mod._redis_client = fake

    await redis_mod.reset_redis()

    assert redis_mod._redis_client is None
    fake.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_redis_ping_resets_singleton_on_connection_error():
    from app.core import redis as redis_mod
    from redis.exceptions import ConnectionError as RedisConnectionError

    bad_client = MagicMock()
    bad_client.ping = AsyncMock(side_effect=RedisConnectionError("boom"))
    bad_client.aclose = AsyncMock()
    redis_mod._redis_client = bad_client

    result = await redis_mod.redis_ping()

    assert result is False
    assert redis_mod._redis_client is None     # singleton cleared
    bad_client.aclose.assert_awaited_once()    # torn down


@pytest.mark.asyncio
async def test_redis_ping_does_not_reset_on_generic_exception():
    """Unknown errors shouldn't nuke the singleton — only connection-level."""
    from app.core import redis as redis_mod

    client = MagicMock()
    client.ping = AsyncMock(side_effect=ValueError("unrelated"))
    client.aclose = AsyncMock()
    redis_mod._redis_client = client

    result = await redis_mod.redis_ping()

    assert result is False
    assert redis_mod._redis_client is client   # still cached
    client.aclose.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_redis_after_reset_builds_fresh_client():
    """After reset, the next get_redis() must create a new pool, not reuse
    the dead one."""
    from app.core import redis as redis_mod

    # First: seed a 'dead' client and reset
    dead = MagicMock()
    dead.aclose = AsyncMock()
    redis_mod._redis_client = dead
    await redis_mod.reset_redis()
    assert redis_mod._redis_client is None

    # Now patch aioredis.from_url so get_redis() doesn't try a real connect
    sentinel = object()
    with patch.object(redis_mod.aioredis, "from_url", return_value=sentinel):
        # Provide minimal settings stub
        fake_settings = SimpleNamespace(REDIS_URL="redis://localhost:6379/0")
        with patch.object(redis_mod, "get_settings", return_value=fake_settings):
            new_client = redis_mod.get_redis()
            assert new_client is sentinel
            assert redis_mod._redis_client is sentinel

    # Cleanup so other tests aren't polluted
    redis_mod._redis_client = None
