"""
FastAPI application — the main entry point for the AuthentiGuard backend.
All middleware, routers, and startup/shutdown hooks are registered here.
"""

from __future__ import annotations

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
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
