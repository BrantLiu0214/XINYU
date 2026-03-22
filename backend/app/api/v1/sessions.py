from __future__ import annotations

from sqlalchemy import func, select

from fastapi import APIRouter, Depends, HTTPException

from backend.app.dependencies.services import AppContainer, get_container
from backend.app.models.chat_message import ChatMessage
from backend.app.models.chat_session import ChatSession
from backend.app.models.visitor_profile import VisitorProfile
from backend.app.schemas.session import (
    CreateSessionRequest,
    CreateSessionResponse,
    SessionDetailResponse,
    VisitorSessionListResponse,
    VisitorSessionSummary,
)

router = APIRouter(tags=["sessions"])


@router.post("", response_model=CreateSessionResponse, status_code=201)
async def create_session(
    body: CreateSessionRequest,
    container: AppContainer = Depends(get_container),
) -> CreateSessionResponse:
    """Create a chat session. If visitor_id is provided, reuse that visitor; otherwise create a new one."""
    with container.session_factory() as db:
        if body.visitor_id:
            visitor = db.get(VisitorProfile, body.visitor_id)
            if visitor is None:
                raise HTTPException(status_code=404, detail="Visitor not found")
        else:
            visitor = VisitorProfile(
                display_name=body.display_name,
                consent_accepted=True,
            )
            db.add(visitor)
            db.flush()

        session = ChatSession(visitor_id=visitor.id)
        db.add(session)
        db.flush()

        visitor_id = visitor.id
        session_id = session.id
        db.commit()

    return CreateSessionResponse(visitor_id=visitor_id, session_id=session_id)


@router.get("", response_model=VisitorSessionListResponse)
async def list_visitor_sessions(
    visitor_id: str,
    container: AppContainer = Depends(get_container),
) -> VisitorSessionListResponse:
    """Return all sessions for a visitor, newest-first (limit 10)."""
    with container.session_factory() as db:
        msg_count_sq = (
            select(
                ChatMessage.session_id,
                func.count(ChatMessage.id).label("message_count"),
            )
            .group_by(ChatMessage.session_id)
            .subquery()
        )

        rows = db.execute(
            select(ChatSession, msg_count_sq.c.message_count)
            .outerjoin(msg_count_sq, ChatSession.id == msg_count_sq.c.session_id)
            .where(ChatSession.visitor_id == visitor_id)
            .order_by(ChatSession.started_at.desc())
            .limit(10)
        ).all()

    sessions = [
        VisitorSessionSummary(
            session_id=row.ChatSession.id,
            started_at=row.ChatSession.started_at,
            message_count=row.message_count or 0,
            latest_risk_level=row.ChatSession.latest_risk_level.value,
        )
        for row in rows
    ]
    return VisitorSessionListResponse(sessions=sessions)


@router.get("/{session_id}", response_model=SessionDetailResponse)
async def get_session(
    session_id: str,
    container: AppContainer = Depends(get_container),
) -> SessionDetailResponse:
    """Return basic info for an existing session."""
    with container.session_factory() as db:
        session = db.get(ChatSession, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return SessionDetailResponse(
            session_id=session.id,
            visitor_id=session.visitor_id,
            latest_risk_level=session.latest_risk_level.value,
            started_at=session.started_at,
        )
