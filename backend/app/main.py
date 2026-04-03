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
from fastapi.responses import JSONResponse

from .api.v1.endpoints.routes import router as api_router
from .core.config import get_settings
from .core.database import engine, Base
from .middleware.middleware import AuditLogMiddleware, RateLimitMiddleware

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    settings = get_settings()
    log.info("starting_up", env=settings.APP_ENV, version=settings.APP_VERSION)

    # Create DB tables (in production, use Alembic migrations instead)
    if settings.APP_ENV == "development":
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        log.info("db_tables_created")

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
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self'; "
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
                from .core.security import decode_access_token
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

    # ── Routes ────────────────────────────────────────────────
    app.include_router(api_router, prefix="/api/v1")

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
