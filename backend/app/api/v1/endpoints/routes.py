"""
API endpoints — auth, analysis submission, results polling, reports, webhooks.
Every endpoint has input validation, error handling, and rate limiting (via middleware).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.core.security import (
    create_access_token,
    create_refresh_token,
    hash_password,
    rotate_refresh_token,
    verify_password,
)
from app.models.models import (
    ContentType, DetectionJob, DetectionResult, JobStatus, User, UserRole, UserTier,
)
from app.schemas.schemas import (
    AnalysisJobResponse,
    DetectionResultResponse,
    ErrorResponse,
    JobStatusResponse,
    LayerScoresSchema,
    LoginRequest,
    ModelAttributionSchema,
    ForgotPasswordRequest,
    RefreshRequest,
    RegisterRequest,
    ResetPasswordRequest,
    TextSubmitRequest,
    TokenResponse,
    UrlSubmitRequest,
    UserResponse,
    UsageStatsResponse,
    WebhookCreateRequest,
    WebhookResponse,
)
from app.services.upload_service import store_upload
from app.workers.celery_app import CONTENT_TYPE_TO_QUEUE, TIER_TO_PRIORITY
from app.workers.text_worker import run_text_detection
from app.api.v1.deps import CurrentUser

router = APIRouter()


# ═══════════════════════════════════════════════════════════════
# AUTH
# ═══════════════════════════════════════════════════════════════

@router.post(
    "/auth/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["auth"],
)
async def register(
    body: RegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> User:
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )
    user = User(
        email=body.email,
        hashed_password=hash_password(body.password),
        full_name=body.full_name,
        role=UserRole.API_CONSUMER,
        tier=UserTier.FREE,
        consent_given=body.consent_given,
        consent_at=datetime.now(timezone.utc) if body.consent_given else None,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@router.post("/auth/login", response_model=TokenResponse, tags=["auth"])
async def login(
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is inactive")

    from app.core.config import get_settings
    settings = get_settings()

    access_token  = create_access_token(str(user.id), user.role.value, user.email, user.tier.value)
    refresh_token = await create_refresh_token(str(user.id))

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/auth/refresh", response_model=TokenResponse, tags=["auth"])
async def refresh_token(
    body: RefreshRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """Step 25: Refresh token rotation — invalidate old, issue new pair."""
    try:
        user_id, new_refresh = await rotate_refresh_token(body.refresh_token)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))

    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    from app.core.config import get_settings
    settings = get_settings()

    return TokenResponse(
        access_token=create_access_token(str(user.id), user.role.value, user.email, user.tier.value),
        refresh_token=new_refresh,
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT, tags=["auth"],
              response_class=JSONResponse)
async def logout(body: RefreshRequest):
    from app.core.security import revoke_refresh_token
    await revoke_refresh_token(body.refresh_token)
    return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content=None)


@router.post("/auth/forgot-password", status_code=status.HTTP_200_OK, tags=["auth"])
async def forgot_password(
    body: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Request a password reset token. Always returns 200 to prevent email enumeration.
    In production, this would send an email with the reset link.
    """
    from app.core.security import create_password_reset_token

    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()

    if user:
        token = await create_password_reset_token(str(user.id))
        # TODO: Send email with reset link containing the token.
        # For now, log it (dev only — remove before production).
        import structlog
        structlog.get_logger().info("password_reset_token_created", user_id=str(user.id), token=token)

    # Always return success to prevent email enumeration
    return {"message": "If an account with that email exists, a password reset link has been sent."}


@router.post("/auth/reset-password", status_code=status.HTTP_200_OK, tags=["auth"])
async def reset_password(
    body: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Reset password using a valid reset token."""
    from app.core.security import validate_password_reset_token, hash_password as _hash

    try:
        user_id = await validate_password_reset_token(body.token)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    user.hashed_password = _hash(body.new_password)
    await db.commit()

    # Revoke all existing refresh tokens for security
    from app.core.security import revoke_all_refresh_tokens
    await revoke_all_refresh_tokens(user_id)

    return {"message": "Password has been reset successfully. Please log in with your new password."}


# ═══════════════════════════════════════════════════════════════
# ANALYSIS — SUBMISSION
# ═══════════════════════════════════════════════════════════════

@router.post(
    "/analyze/text",
    response_model=AnalysisJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["analysis"],
)
async def submit_text(
    body: TextSubmitRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> AnalysisJobResponse:
    """Submit text/code paste for analysis. Returns a job_id to poll."""
    import hashlib
    content_type = ContentType(body.content_type)
    content_hash = hashlib.sha256(body.text.encode()).hexdigest()

    job = DetectionJob(
        user_id=current_user.id,
        content_type=content_type,
        input_text=body.text,
        content_hash=content_hash,
        status=JobStatus.PENDING,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Dispatch to Celery queue
    priority = TIER_TO_PRIORITY.get(current_user.tier.value, 1)
    task = run_text_detection.apply_async(
        args=[str(job.id)],
        queue="text",
        priority=priority,
    )
    job.celery_task_id = task.id
    await db.commit()

    return AnalysisJobResponse(
        job_id=job.id,
        status=job.status.value,
        content_type=content_type.value,
        created_at=job.created_at,
        poll_url=f"/api/v1/jobs/{job.id}",
    )


@router.post(
    "/analyze/file",
    response_model=AnalysisJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["analysis"],
)
async def submit_file(
    current_user: CurrentUser,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> AnalysisJobResponse:
    """Upload a file (text, image, audio, video, code) for analysis."""
    s3_key, content_hash, content_type, file_size = await store_upload(
        file, str(current_user.id)
    )

    job = DetectionJob(
        user_id=current_user.id,
        content_type=content_type,
        s3_key=s3_key,
        file_name=file.filename,
        file_size=file_size,
        content_hash=content_hash,
        status=JobStatus.PENDING,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Route to appropriate queue based on content type
    queue    = CONTENT_TYPE_TO_QUEUE.get(content_type.value, "text")
    priority = TIER_TO_PRIORITY.get(current_user.tier.value, 1)

    if queue == "text":
        task = run_text_detection.apply_async(
            args=[str(job.id)], queue=queue, priority=priority
        )
    else:
        # Audio/video/image workers are added in Phases 6–8
        from app.workers.celery_app import celery_app
        task = celery_app.send_task(
            f"workers.{queue}_worker.run_{queue}_detection",
            args=[str(job.id)],
            queue=queue,
            priority=priority,
        )

    job.celery_task_id = task.id
    await db.commit()

    return AnalysisJobResponse(
        job_id=job.id,
        status=job.status.value,
        content_type=content_type.value,
        created_at=job.created_at,
        poll_url=f"/api/v1/jobs/{job.id}",
    )


# ═══════════════════════════════════════════════════════════════
# JOBS — POLLING
# ═══════════════════════════════════════════════════════════════

@router.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["jobs"])
async def get_job_status(
    job_id: uuid.UUID,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> JobStatusResponse:
    job = await _get_job_or_404(job_id, current_user.id, db)
    return JobStatusResponse(
        job_id=job.id,
        status=job.status.value,
        progress=None,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


@router.get(
    "/jobs/{job_id}/result",
    response_model=DetectionResultResponse,
    tags=["jobs"],
)
async def get_job_result(
    job_id: uuid.UUID,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> DetectionResultResponse:
    job = await _get_job_or_404(job_id, current_user.id, db)

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job is not yet complete. Current status: {job.status.value}",
        )
    if not job.result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Result not found for this job",
        )

    r = job.result
    return DetectionResultResponse(
        job_id=job.id,
        status=job.status.value,
        content_type=job.content_type.value,
        authenticity_score=r.authenticity_score,
        confidence=r.confidence,
        label=r.label,
        layer_scores=LayerScoresSchema(**r.layer_scores),
        sentence_scores=r.sentence_scores,
        top_signals=r.evidence_summary.get("top_signals", []),
        model_attribution=ModelAttributionSchema(**r.model_attribution),
        processing_ms=r.processing_ms,
        report_url=f"/api/v1/jobs/{job.id}/report" if r.report_s3_key else None,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )


# ═══════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════

@router.get("/dashboard/stats", response_model=UsageStatsResponse, tags=["dashboard"])
async def get_usage_stats(
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> UsageStatsResponse:
    from sqlalchemy import func
    from app.core.config import get_settings

    settings = get_settings()

    # First of current month for monthly filtering
    now = datetime.now(timezone.utc)
    first_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    # Count all completed jobs
    all_jobs_q = await db.execute(
        select(DetectionJob).where(
            DetectionJob.user_id == current_user.id,
            DetectionJob.status == JobStatus.COMPLETED,
        )
    )
    all_jobs = all_jobs_q.scalars().all()

    # Count this month's jobs
    monthly_jobs_q = await db.execute(
        select(func.count(DetectionJob.id)).where(
            DetectionJob.user_id == current_user.id,
            DetectionJob.created_at >= first_of_month,
        )
    )
    scans_this_month = monthly_jobs_q.scalar() or 0

    # Get results for label counts
    job_ids = [j.id for j in all_jobs]
    results_q = await db.execute(
        select(DetectionResult).where(DetectionResult.job_id.in_(job_ids))
    ) if job_ids else None

    results = results_q.scalars().all() if results_q else []
    labels  = [r.label for r in results]
    scores  = [r.authenticity_score for r in results]

    tier_limit_map = {
        "free": settings.RATE_LIMIT_FREE_TIER,
        "pro":  settings.RATE_LIMIT_PRO_TIER,
        "enterprise": settings.RATE_LIMIT_ENTERPRISE_TIER,
    }

    return UsageStatsResponse(
        total_scans=len(all_jobs),
        scans_this_month=scans_this_month,
        ai_detected=labels.count("AI"),
        human_detected=labels.count("HUMAN"),
        uncertain=labels.count("UNCERTAIN"),
        avg_score=round(sum(scores) / len(scores), 4) if scores else None,
        tier_limit=tier_limit_map.get(current_user.tier.value, 10),
        tier_used=scans_this_month,
    )


# ═══════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════

@router.get("/health", tags=["system"], include_in_schema=False)
async def health(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    """Health check — returns 200 if all services ok, 503 if any degraded."""
    checks: dict[str, str] = {}

    # Database check
    try:
        await db.execute(select(func.now()))
        checks["database"] = "ok"
    except Exception:
        checks["database"] = "degraded"

    # Redis check
    from app.core.redis import redis_ping
    checks["redis"] = "ok" if await redis_ping() else "degraded"

    # Celery check (are any workers alive?)
    try:
        from app.workers.celery_app import celery_app
        insp = celery_app.control.inspect(timeout=2.0)
        ping_result = insp.ping()
        checks["celery"] = "ok" if ping_result else "degraded"
    except Exception:
        checks["celery"] = "degraded"

    all_ok = all(v == "ok" for v in checks.values())

    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={
            "status": "ok" if all_ok else "degraded",
            "checks": checks,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


# ═══════════════════════════════════════════════════════════════
# ANALYSIS — URL
# ═══════════════════════════════════════════════════════════════

@router.post(
    "/analyze/url",
    response_model=AnalysisJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["analysis"],
)
async def submit_url(
    body: UrlSubmitRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> AnalysisJobResponse:
    """Submit a URL for content analysis. Fetches content and routes to the appropriate detector."""
    import hashlib
    from app.services.url_analyzer import fetch_and_analyze_url

    try:
        content_type, content, metadata = await fetch_and_analyze_url(body.url)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    content_hash = hashlib.sha256(
        content.encode() if isinstance(content, str) else content
    ).hexdigest()

    # For text content, store directly; for binary, upload to S3
    s3_key = None
    input_text = None
    file_size = None

    if isinstance(content, str):
        input_text = content
    else:
        from app.services.s3_service import upload_to_s3
        s3_key = f"urls/{current_user.id}/{content_hash[:16]}"
        await upload_to_s3(s3_key, content, metadata.get("content_type_header", "application/octet-stream"))
        file_size = len(content)

    job = DetectionJob(
        user_id=current_user.id,
        content_type=content_type,
        input_text=input_text,
        s3_key=s3_key,
        file_name=body.url,
        file_size=file_size,
        content_hash=content_hash,
        status=JobStatus.PENDING,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    queue    = CONTENT_TYPE_TO_QUEUE.get(content_type.value, "text")
    priority = TIER_TO_PRIORITY.get(current_user.tier.value, 1)

    if queue == "text":
        task = run_text_detection.apply_async(
            args=[str(job.id)], queue=queue, priority=priority
        )
    else:
        from app.workers.celery_app import celery_app
        task = celery_app.send_task(
            f"workers.{queue}_worker.run_{queue}_detection",
            args=[str(job.id)],
            queue=queue,
            priority=priority,
        )

    job.celery_task_id = task.id
    await db.commit()

    return AnalysisJobResponse(
        job_id=job.id,
        status=job.status.value,
        content_type=content_type.value,
        created_at=job.created_at,
        poll_url=f"/api/v1/jobs/{job.id}",
    )


# ═══════════════════════════════════════════════════════════════
# WEBHOOKS — CRUD
# ═══════════════════════════════════════════════════════════════

@router.post(
    "/webhooks",
    response_model=WebhookResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["webhooks"],
)
async def create_webhook(
    body: WebhookCreateRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> WebhookResponse:
    """Register a webhook endpoint to receive job notifications."""
    from app.models.webhook import Webhook

    # Limit webhooks per user
    existing = await db.execute(
        select(Webhook).where(Webhook.user_id == current_user.id, Webhook.is_active == True)
    )
    if len(existing.scalars().all()) >= 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 active webhooks per user",
        )

    webhook = Webhook(
        user_id=current_user.id,
        url=body.url,
        events=body.events,
        secret=body.secret,
    )
    db.add(webhook)
    await db.commit()
    await db.refresh(webhook)
    return webhook


@router.get("/webhooks", response_model=list[WebhookResponse], tags=["webhooks"])
async def list_webhooks(
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> list:
    """List all webhooks for the current user."""
    from app.models.webhook import Webhook
    result = await db.execute(
        select(Webhook).where(Webhook.user_id == current_user.id).order_by(Webhook.created_at.desc())
    )
    return result.scalars().all()


@router.delete(
    "/webhooks/{webhook_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["webhooks"],
    response_class=JSONResponse,
)
async def delete_webhook(
    webhook_id: uuid.UUID,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    """Delete a webhook."""
    from app.models.webhook import Webhook
    result = await db.execute(
        select(Webhook).where(Webhook.id == webhook_id, Webhook.user_id == current_user.id)
    )
    webhook = result.scalar_one_or_none()
    if not webhook:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Webhook not found")
    await db.delete(webhook)
    await db.commit()
    return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content=None)


@router.patch("/webhooks/{webhook_id}", response_model=WebhookResponse, tags=["webhooks"])
async def update_webhook(
    webhook_id: uuid.UUID,
    body: WebhookCreateRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
) -> WebhookResponse:
    """Update a webhook's URL, events, or secret."""
    from app.models.webhook import Webhook
    result = await db.execute(
        select(Webhook).where(Webhook.id == webhook_id, Webhook.user_id == current_user.id)
    )
    webhook = result.scalar_one_or_none()
    if not webhook:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Webhook not found")

    webhook.url = body.url
    webhook.events = body.events
    if body.secret is not None:
        webhook.secret = body.secret
    await db.commit()
    await db.refresh(webhook)
    return webhook


# ═══════════════════════════════════════════════════════════════
# PASSPORT — PUBLIC VERIFICATION
# ═══════════════════════════════════════════════════════════════

@router.get("/passport/{content_hash}", tags=["passport"])
async def get_passport(content_hash: str) -> dict:
    """
    Public endpoint — retrieve an authenticity passport by content hash.
    No authentication required.
    """
    try:
        from authenticity_passport.signer.passport import PassportRegistry  # type: ignore
        registry = PassportRegistry()
        passport = registry.get(content_hash)
        if not passport:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Passport not found")
        return passport.__dict__ if hasattr(passport, "__dict__") else passport
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Passport module not available",
        )


@router.post("/passport/verify", tags=["passport"])
async def verify_passport(body: dict) -> dict:
    """
    Public endpoint — verify an authenticity passport.
    Accepts passport JSON and returns verification result.
    """
    try:
        from authenticity_passport.signer.passport import PassportVerifier  # type: ignore
        verifier = PassportVerifier()
        result = verifier.verify(body)
        return {"verified": result.get("valid", False), "details": result}
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Passport module not available",
        )


# ═══════════════════════════════════════════════════════════════
# REPORTS — EXPORT
# ═══════════════════════════════════════════════════════════════

@router.get("/jobs/{job_id}/report", tags=["reports"])
async def get_report(
    job_id: uuid.UUID,
    current_user: CurrentUser,
    format: str = "json",
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Generate or retrieve a detection report.
    Supports format=json or format=pdf.
    """
    job = await _get_job_or_404(job_id, current_user.id, db)

    if job.status != JobStatus.COMPLETED or not job.result:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Job must be completed before generating a report",
        )

    r = job.result

    if format == "json":
        return {
            "report_type": "json",
            "job_id": str(job.id),
            "content_type": job.content_type.value,
            "authenticity_score": r.authenticity_score,
            "confidence": r.confidence,
            "label": r.label,
            "layer_scores": r.layer_scores,
            "evidence_summary": r.evidence_summary,
            "sentence_scores": r.sentence_scores,
            "model_attribution": r.model_attribution,
            "processing_ms": r.processing_ms,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        }
    elif format == "pdf":
        from app.services.report_service import generate_pdf_report
        try:
            pdf_url = await generate_pdf_report(job, r)
            return {"report_type": "pdf", "report_url": pdf_url}
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate PDF: {str(exc)}",
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Supported formats: json, pdf",
        )


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

async def _get_job_or_404(
    job_id: uuid.UUID,
    user_id: uuid.UUID,
    db: AsyncSession,
) -> DetectionJob:
    result = await db.execute(
        select(DetectionJob)
        .options(selectinload(DetectionJob.result))
        .where(
            DetectionJob.id == job_id,
            DetectionJob.user_id == user_id,
        )
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job
