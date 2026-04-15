"""
FastAPI application — the main entry point for the AuthentiGuard backend.
All middleware, routers, and startup/shutdown hooks are registered here.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.api.v1.endpoints.routes import router as api_router
from app.core.config import get_settings
from app.core.database import engine, Base
from app.middleware.middleware import AuditLogMiddleware, RateLimitMiddleware

log = structlog.get_logger(__name__)


DEMO_USER_ID = uuid.UUID("00000000-0000-4000-a000-000000000001")
DEMO_USER_EMAIL = "demo@authentiguard.local"


async def _ensure_db_tables():
    """Dev-only: create tables via SQLAlchemy metadata.

    Alembic is the source of truth for schema in every non-dev environment.
    This helper is only invoked from the lifespan when
    `settings.ALLOW_DB_CREATE_ALL` is explicitly True, which is rejected by
    the Settings validator when `APP_ENV=production`. Production boots must
    run `alembic upgrade head` before the API starts.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("db_tables_ensured_via_create_all_dev_only")


async def _ensure_minio_buckets(settings):
    """Create S3/MinIO buckets if they don't exist."""
    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import ClientError

    kwargs = {
        "region_name": settings.AWS_REGION,
        "aws_access_key_id": settings.AWS_ACCESS_KEY_ID,
        "aws_secret_access_key": settings.AWS_SECRET_ACCESS_KEY,
        "config": BotoConfig(connect_timeout=5, read_timeout=10, retries={"max_attempts": 2}),
    }
    if settings.S3_ENDPOINT_URL:
        kwargs["endpoint_url"] = settings.S3_ENDPOINT_URL

    s3 = boto3.client("s3", **kwargs)
    for bucket in [settings.S3_BUCKET_UPLOADS, settings.S3_BUCKET_REPORTS]:
        try:
            s3.head_bucket(Bucket=bucket)
            log.info("s3_bucket_exists", bucket=bucket)
        except ClientError:
            s3.create_bucket(Bucket=bucket)
            log.info("s3_bucket_created", bucket=bucket)


async def _ensure_demo_user():
    """Seed a system demo user for anonymous API access."""
    from sqlalchemy import select
    from app.core.database import AsyncSessionLocal
    from app.models.models import User, UserRole, UserTier
    from app.core.security import hash_password

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(User).where(User.id == DEMO_USER_ID)
        )
        if result.scalar_one_or_none() is None:
            demo_user = User(
                id=DEMO_USER_ID,
                email=DEMO_USER_EMAIL,
                hashed_password=hash_password("__demo_nologin__"),
                full_name="Demo User",
                role=UserRole.API_CONSUMER,
                tier=UserTier.FREE,
                is_active=True,
                is_verified=True,
            )
            session.add(demo_user)
            await session.commit()
            log.info("demo_user_created", user_id=str(DEMO_USER_ID))
        else:
            log.info("demo_user_exists")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    settings = get_settings()
    log.info("starting_up", env=settings.APP_ENV, version=settings.APP_VERSION)

    # Schema bootstrap.
    # Alembic is the source of truth. We only fall back to create_all when
    # `ALLOW_DB_CREATE_ALL=true`, which is blocked in production by the
    # Settings validator. Any other path (staging, prod) must have already
    # run `alembic upgrade head` as part of the deploy.
    if settings.ALLOW_DB_CREATE_ALL:
        log.warning(
            "using_create_all_bootstrap_dev_only",
            env=settings.APP_ENV,
            hint="Run `alembic upgrade head` instead for any non-dev environment.",
        )
        await _ensure_db_tables()
    else:
        log.info("skipping_create_all_alembic_is_source_of_truth", env=settings.APP_ENV)

    # Ensure S3/MinIO buckets exist
    try:
        await _ensure_minio_buckets(settings)
    except Exception as exc:
        log.warning("minio_bucket_init_failed", error=str(exc))

    # Seed demo user for anonymous demo access
    try:
        await _ensure_demo_user()
    except Exception as exc:
        log.warning("demo_user_seed_failed", error=str(exc))

    yield

    log.info("shutting_down")
    await engine.dispose()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="AI content authenticity detection across text, image, video, audio, and code.",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept", "X-Request-ID"],
    )

    # ── Correlation ID ────────────────────────────────────────
    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        structlog.contextvars.unbind_contextvars("request_id")
        return response

    # ── Security headers ────────────────────────────────────────
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if settings.is_production:
            response.headers["Strict-Transport-Security"] = (
                "max-age=63072000; includeSubDomains; preload"
            )
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self'; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "img-src 'self' data: https:; "
                "font-src 'self' https://fonts.gstatic.com; "
                "connect-src 'self'"
            )
        return response

    # ── Lightweight auth extraction (populates request.state.user for rate limiter) ──
    @app.middleware("http")
    async def extract_user_from_jwt(request: Request, call_next):
        """
        Best-effort JWT decode to populate request.state.user before
        the rate limiter runs. Does NOT enforce auth — that's deps.py's job.
        """
        request.state.user = None
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                from app.core.security import decode_access_token
                payload = decode_access_token(auth_header[7:])

                class _TokenUser:
                    """Minimal user-like object for rate limiting."""
                    __slots__ = ("id", "tier")
                    def __init__(self, uid: str, tier: str):
                        self.id = uid
                        self.tier = tier

                request.state.user = _TokenUser(payload["sub"], payload.get("tier", "free"))
            except Exception:
                pass  # Invalid/expired token — fall through to anonymous rate limiting
        return await call_next(request)

    # ── Request body size limit (10 MB) ─────────────────────────
    from starlette.middleware.base import BaseHTTPMiddleware

    MAX_REQUEST_BODY_BYTES = 10 * 1024 * 1024  # 10 MB

    class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > MAX_REQUEST_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"error": "request_too_large", "message": "Request body exceeds 10 MB limit"},
                )
            return await call_next(request)

    app.add_middleware(RequestSizeLimitMiddleware)

    # ── Rate limiting + audit logging ─────────────────────────
    app.add_middleware(AuditLogMiddleware)
    app.add_middleware(RateLimitMiddleware)

    # ── Prometheus metrics ──────────────────────────────────────
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    from app.core.metrics import DETECTOR_FALLBACK  # noqa: F811
    import time as _time

    REQUEST_LATENCY = Histogram(
        "http_request_duration_seconds",
        "HTTP request latency",
        ["method", "path", "status"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )
    REQUEST_COUNT = Counter(
        "http_requests_total",
        "Total HTTP requests",
        ["method", "path", "status"],
    )

    @app.middleware("http")
    async def prometheus_middleware(request: Request, call_next):
        if request.url.path in {"/metrics", "/health"}:
            return await call_next(request)
        start = _time.perf_counter()
        response = await call_next(request)
        elapsed = _time.perf_counter() - start
        # Normalize path to avoid cardinality explosion (strip UUIDs)
        import re
        path = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "{id}",
            request.url.path,
        )
        REQUEST_LATENCY.labels(
            method=request.method, path=path, status=response.status_code,
        ).observe(elapsed)
        REQUEST_COUNT.labels(
            method=request.method, path=path, status=response.status_code,
        ).inc()
        return response

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        # Update detector fallback gauge on each scrape
        try:
            from app.workers.text_worker import get_detector_mode
            DETECTOR_FALLBACK.set(1 if get_detector_mode() == "fallback" else 0)
        except Exception:
            pass
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    # ── Health check ─────────────────────────────────────────
    @app.get("/health", include_in_schema=False)
    async def health():
        checks: dict[str, str] = {}

        # Database
        try:
            async with engine.connect() as conn:
                await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
            checks["database"] = "ok"
        except Exception as exc:
            checks["database"] = f"error: {exc}"

        # Redis
        try:
            from app.core.redis import redis_ping
            if await redis_ping():
                checks["redis"] = "ok"
            else:
                checks["redis"] = "error: ping returned False"
        except Exception as exc:
            checks["redis"] = f"error: {exc}"

        # Detector mode (informational — not a hard failure)
        try:
            from app.workers.text_worker import get_detector_mode
            checks["detector_mode"] = get_detector_mode()
        except Exception:
            checks["detector_mode"] = "unknown"

        all_ok = checks.get("database") == "ok" and checks.get("redis") == "ok"
        if all_ok:
            return {"status": "healthy", "checks": checks}
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "checks": checks},
        )

    # ── Routes ────────────────────────────────────────────────
    app.include_router(api_router, prefix="/api/v1")

    # ── Serve frontend landing page (dev mode) ───────────────
    import os
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../frontend/public"))
    if os.path.isdir(frontend_dir):
        @app.get("/", include_in_schema=False)
        async def root():
            return FileResponse(os.path.join(frontend_dir, "landing.html"))

        app.mount("/", StaticFiles(directory=frontend_dir), name="frontend")

    # ── Global error handlers ─────────────────────────────────
    @app.exception_handler(Exception)
    async def global_error_handler(request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled_exception", path=request.url.path, error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "internal_error", "message": "An unexpected error occurred"},
        )

    return app


app = create_app()
