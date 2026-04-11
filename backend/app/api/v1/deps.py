"""
Step 35: Role-Based Access Control.
FastAPI dependencies for authentication and authorization.
Roles: admin > analyst > api_consumer.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import decode_access_token
from app.models.models import User, UserRole

bearer_scheme = HTTPBearer(auto_error=False)


async def _get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Security(bearer_scheme),
    ],
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Resolve the current user from the Bearer JWT.
    Raises 401 if missing/invalid, 403 if inactive.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_access_token(credentials.credentials)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

    result = await db.execute(
        select(User).where(User.id == uuid.UUID(user_id))
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is inactive")

    return user


# ── Public dependency ──────────────────────────────────────────
CurrentUser = Annotated[User, Depends(_get_current_user)]


async def _get_optional_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Security(bearer_scheme),
    ],
    db: AsyncSession = Depends(get_db),
) -> User | None:
    """
    Resolve the current user from the Bearer JWT, or return None for anonymous access.
    Used by demo/public endpoints that work with or without authentication.
    """
    if credentials is None:
        return None

    try:
        payload = decode_access_token(credentials.credentials)
    except ValueError:
        return None

    user_id = payload.get("sub")
    if not user_id:
        return None

    result = await db.execute(
        select(User).where(User.id == uuid.UUID(user_id))
    )
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        return None

    return user


OptionalCurrentUser = Annotated[User | None, Depends(_get_optional_user)]


# ── Role guards ───────────────────────────────────────────────

def require_role(*roles: UserRole):
    """Factory: returns a FastAPI dependency that checks user role."""
    async def _check(user: CurrentUser) -> User:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {', '.join(r.value for r in roles)}",
            )
        return user
    return _check


AdminRequired   = Depends(require_role(UserRole.ADMIN))
AnalystRequired = Depends(require_role(UserRole.ADMIN, UserRole.ANALYST))

# Convenience aliases
RequireAdmin   = Annotated[User, AdminRequired]
RequireAnalyst = Annotated[User, AnalystRequired]
