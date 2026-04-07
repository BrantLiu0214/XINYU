"""FastAPI auth dependencies: extract and validate JWT from Bearer header."""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select

from backend.app.core.security import decode_token
from backend.app.dependencies.services import AppContainer, get_container

_oauth2 = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/visitor/login")


def _get_payload(
    token: str = Depends(_oauth2),
    container: AppContainer = Depends(get_container),
) -> dict:
    return decode_token(token, container.settings)


def require_visitor(payload: dict = Depends(_get_payload)) -> str:
    """Return visitor_id from a valid visitor JWT. Raises 403 if role mismatch."""
    if payload.get("role") != "visitor":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="需要访客身份")
    return payload["sub"]


def require_counselor(payload: dict = Depends(_get_payload)) -> str:
    """Return counselor_id from a valid counselor JWT. Raises 403 if role mismatch."""
    if payload.get("role") != "counselor":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="需要咨询师身份")
    return payload["sub"]


def require_super_admin(
    counselor_id: str = Depends(require_counselor),
    container: AppContainer = Depends(get_container),
) -> str:
    """Return counselor_id only if the counselor is a super admin (college IS NULL)."""
    from backend.app.models.counselor_account import CounselorAccount
    with container.session_factory() as db:
        counselor = db.get(CounselorAccount, counselor_id)
        if counselor is None or counselor.college is not None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="仅超级管理员可执行此操作")
    return counselor_id
