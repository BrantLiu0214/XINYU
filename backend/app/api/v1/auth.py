from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select

from backend.app.core.colleges import COLLEGES
from backend.app.core.security import create_token, hash_password, verify_password
from backend.app.dependencies.auth import _get_payload
from backend.app.dependencies.services import AppContainer, get_container
from backend.app.models.counselor_account import CounselorAccount
from backend.app.models.visitor_profile import VisitorProfile
from backend.app.schemas.auth import (
    CounselorLoginRequest,
    MeResponse,
    TokenResponse,
    VisitorLoginRequest,
    VisitorRegisterRequest,
)

router = APIRouter(tags=["auth"])

_GUEST_EXPIRE_MINUTES = 60 * 24  # 24 hours for guest tokens


@router.post("/visitor/guest", response_model=TokenResponse, status_code=201)
async def visitor_guest(
    container: AppContainer = Depends(get_container),
) -> TokenResponse:
    """Create an anonymous guest visitor profile and return a short-lived JWT."""
    with container.session_factory() as db:
        visitor = VisitorProfile(is_guest=True, consent_accepted=True)
        db.add(visitor)
        db.flush()
        visitor_id = visitor.id
        db.commit()

    token = create_token(visitor_id, "visitor", container.settings, expire_minutes=_GUEST_EXPIRE_MINUTES)
    return TokenResponse(access_token=token, role="visitor", user_id=visitor_id)


@router.post("/visitor/register", response_model=TokenResponse, status_code=201)
async def visitor_register(
    body: VisitorRegisterRequest,
    container: AppContainer = Depends(get_container),
) -> TokenResponse:
    if body.college not in COLLEGES:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="无效的学院名称")

    with container.session_factory() as db:
        existing = db.scalar(
            select(VisitorProfile).where(VisitorProfile.username == body.username)
        )
        if existing is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="用户名已被占用")

        visitor = VisitorProfile(
            username=body.username,
            password_hash=hash_password(body.password),
            display_name=body.display_name or body.username,
            real_name=body.real_name,
            college=body.college,
            student_id=body.student_id,
            consent_accepted=True,
        )
        db.add(visitor)
        db.flush()
        visitor_id = visitor.id
        db.commit()

    token = create_token(visitor_id, "visitor", container.settings)
    return TokenResponse(access_token=token, role="visitor", user_id=visitor_id)


@router.post("/visitor/login", response_model=TokenResponse)
async def visitor_login(
    body: VisitorLoginRequest,
    container: AppContainer = Depends(get_container),
) -> TokenResponse:
    with container.session_factory() as db:
        visitor = db.scalar(
            select(VisitorProfile).where(VisitorProfile.username == body.username)
        )
    if visitor is None or visitor.password_hash is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
    if not verify_password(body.password, visitor.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")

    token = create_token(visitor.id, "visitor", container.settings)
    return TokenResponse(access_token=token, role="visitor", user_id=visitor.id)


@router.post("/counselor/login", response_model=TokenResponse)
async def counselor_login(
    body: CounselorLoginRequest,
    container: AppContainer = Depends(get_container),
) -> TokenResponse:
    with container.session_factory() as db:
        counselor = db.scalar(
            select(CounselorAccount).where(CounselorAccount.username == body.username)
        )
    if counselor is None or not counselor.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")
    if not verify_password(body.password, counselor.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")

    token = create_token(counselor.id, "counselor", container.settings)
    return TokenResponse(access_token=token, role="counselor", user_id=counselor.id)


@router.get("/me", response_model=MeResponse)
async def get_me(
    payload: dict = Depends(_get_payload),
    container: AppContainer = Depends(get_container),
) -> MeResponse:
    role = payload.get("role")
    user_id = payload["sub"]

    with container.session_factory() as db:
        if role == "visitor":
            user = db.get(VisitorProfile, user_id)
            if user is None:
                raise HTTPException(status_code=404, detail="用户不存在")
            return MeResponse(
                user_id=user.id,
                role="visitor",
                username=user.username or "",
                display_name=user.display_name,
                real_name=user.real_name,
                college=user.college,
                student_id=user.student_id,
                is_guest=user.is_guest,
            )
        elif role == "counselor":
            user = db.get(CounselorAccount, user_id)
            if user is None:
                raise HTTPException(status_code=404, detail="用户不存在")
            return MeResponse(
                user_id=user.id,
                role="counselor",
                username=user.username,
                display_name=user.display_name,
                college=user.college,
            )

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效令牌")
