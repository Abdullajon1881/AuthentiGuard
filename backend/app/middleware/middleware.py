"""
Step 25: Rate limiting per user tier using Redis sliding window.
Step 36: Audit logging middleware — logs every request with user, IP, outcome.
"""

from __future__ import annotations

import time
import uuid
from typing import Callable

import structlog
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..core.config import get_settings
from ..core.redis import get_redis

log = structlog.get_logger(__name__)

TIER_LIMITS = {
    "free":       1000,
    "pro":        1000,
    "enterprise": 1000,
    "anonymous":  1000,
}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiter using Redis.
    Rate limits are applied per user (authenticated) or per IP (anonymous).
    Limits are configurable per tier in settings.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks, docs, CORS preflight, and static files
        if request.url.path in {"/health", "/docs", "/redoc", "/openapi.json"}:
            return await call_next(request)
        if request.method == "OPTIONS":
            return await call_next(request)
        if not request.url.path.startswith("/api/"):
            return await call_next(request)

        tier, identifier = self._get_tier_and_id(request)
        limit  = TIER_LIMITS.get(tier, TIER_LIMITS["anonymous"])
        window = 60   # 1-minute sliding window

        is_limited, current, reset_at = await self._check_rate_limit(
            identifier, limit, window
        )

        response = await call_next(request) if not is_limited else JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error":   "rate_limit_exceeded",
                "message": f"Rate limit of {limit} requests/minute exceeded.",
                "retry_after": reset_at,
            },
        )

        # Always add rate-limit headers
        response.headers["X-RateLimit-Limit"]     = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, limit - current))
        response.headers["X-RateLimit-Reset"]      = str(reset_at)

        return response

    def _get_tier_and_id(self, request: Request) -> tuple[str, str]:
        """Extract user tier and identifier from request state (set by auth middleware)."""
        user = getattr(request.state, "user", None)
        if user:
            tier = getattr(user, "tier", "free")
            return tier, f"user:{user.id}"
        # Anonymous: rate limit by IP
        ip = request.client.host if request.client else "unknown"
        return "anonymous", f"ip:{ip}"

    async def _check_rate_limit(
        self,
        identifier: str,
        limit: int,
        window: int,
    ) -> tuple[bool, int, int]:
        """
        Sliding window rate limit check using Redis sorted sets.
        Returns (is_limited, current_count, reset_timestamp).
        """
        redis = get_redis()
        key   = f"ratelimit:{identifier}"
        now   = int(time.time() * 1000)   # milliseconds
        window_ms = window * 1000

        pipe = redis.pipeline()
        pipe.zremrangebyscore(key, 0, now - window_ms)   # remove expired entries
        pipe.zadd(key, {str(uuid.uuid4()): now})          # add current request
        pipe.zcard(key)                                    # count in window
        pipe.expire(key, window + 1)
        results = await pipe.execute()

        current  = int(results[2])
        reset_at = (now + window_ms) // 1000   # unix timestamp

        return current > limit, current, reset_at


class AuditLogMiddleware(BaseHTTPMiddleware):
    """
    Step 36: Log every API call to the audit_logs table.
    Non-blocking — audit writes happen in the background.
    Errors in audit logging never affect the API response.
    """

    # Paths that don't need audit logging
    _SKIP_PATHS = {"/health", "/docs", "/redoc", "/openapi.json",
                   "/metrics", "/favicon.ico"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self._SKIP_PATHS:
            return await call_next(request)

        start_ns = time.perf_counter_ns()
        response: Response | None = None
        error_msg: str | None = None

        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            error_msg = str(exc)
            raise
        finally:
            elapsed_ms = (time.perf_counter_ns() - start_ns) // 1_000_000
            status_code = response.status_code if response else 500
            success = 200 <= status_code < 400

            user = getattr(request.state, "user", None)
            log.info(
                "api_request",
                method=request.method,
                path=request.url.path,
                status=status_code,
                user_id=str(user.id) if user else None,
                ip=request.client.host if request.client else None,
                ms=elapsed_ms,
                success=success,
            )

            # Non-blocking async audit write
            import asyncio
            asyncio.create_task(
                self._write_audit_log(request, status_code, success, error_msg, elapsed_ms)
            )

    async def _write_audit_log(
        self,
        request: Request,
        status_code: int,
        success: bool,
        error_msg: str | None,
        elapsed_ms: int,
    ) -> None:
        try:
            from ..core.database import AsyncSessionLocal
            from ..models.models import AuditLog

            user = getattr(request.state, "user", None)

            async with AsyncSessionLocal() as db:
                log_entry = AuditLog(
                    user_id=user.id if user else None,
                    action=f"{request.method}:{request.url.path}",
                    resource_type=self._infer_resource(request.url.path),
                    ip_address=request.client.host if request.client else None,
                    user_agent=request.headers.get("user-agent"),
                    details={
                        "status_code":  status_code,
                        "elapsed_ms":   elapsed_ms,
                        "query_params": self._sanitize_params(dict(request.query_params)),
                    },
                    success=success,
                    error_msg=error_msg,
                )
                db.add(log_entry)
                await db.commit()
        except Exception as exc:
            # Audit logging failures must never impact the API
            log.warning("audit_log_write_failed", error=str(exc))

    _SENSITIVE_KEYS = {"password", "token", "secret", "key", "api_key", "bearer",
                       "refresh_token", "access_token", "authorization", "credential"}

    @classmethod
    def _sanitize_params(cls, params: dict) -> dict:
        """Redact query params that may contain secrets."""
        return {
            k: "***REDACTED***" if any(s in k.lower() for s in cls._SENSITIVE_KEYS) else v
            for k, v in params.items()
        }

    @staticmethod
    def _infer_resource(path: str) -> str:
        parts = [p for p in path.split("/") if p]
        # Skip api version prefix (e.g. "api", "v1") to find actual resource
        for part in parts:
            if part not in {"api", "v1", "v2"}:
                return part
        return "unknown"
