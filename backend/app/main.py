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
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.endpoints.routes import router as api_router
from app.core.config import get_settings
from app.core.database import engine, Base
from app.middleware.middleware import AuditLogMiddleware, RateLimitMiddleware

log = structlog.get_logger(__name__)


DEMO_USER_ID = uuid.UUID("00000000-0000-4000-a000-000000000001")
DEMO_USER_EMAIL = "demo@authentiguard.local"


async def _ensure_db_tables():
    """Create all tables if they don't exist (idempotent)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("db_tables_ensured")


async def _ensure_minio_buckets(settings):
    """Create S3/MinIO buckets if they don't exist."""
    import boto3
    from botocore.exceptions import ClientError

    kwargs = {
        "region_name": settings.AWS_REGION,
        "aws_access_key_id": settings.AWS_ACCESS_KEY_ID,
        "aws_secret_access_key": settings.AWS_SECRET_ACCESS_KEY,
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

    # Ensure DB tables exist (idempotent — safe for all environments)
    await _ensure_db_tables()

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

    # ── Rate limiting + audit logging ─────────────────────────
    app.add_middleware(AuditLogMiddleware)
    app.add_middleware(RateLimitMiddleware)

    # ── Health check ─────────────────────────────────────────
    @app.get("/health", include_in_schema=False)
    async def health():
        try:
            async with engine.connect() as conn:
                await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
            return {"status": "healthy"}
        except Exception:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "unhealthy"},
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
