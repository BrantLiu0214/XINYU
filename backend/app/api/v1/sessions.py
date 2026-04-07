from __future__ import annotations

from sqlalchemy import func, select

from fastapi import APIRouter, Depends, HTTPException

from backend.app.dependencies.auth import require_visitor
from backend.app.dependencies.services import AppContainer, get_container
from backend.app.models.chat_message import ChatMessage
from backend.app.models.chat_session import ChatSession
from backend.app.models.enums import ChatRole
from backend.app.schemas.session import (
    CreateSessionResponse,
    SessionDetailResponse,
    VisitorSessionListResponse,
    VisitorSessionSummary,
)

router = APIRouter(tags=["sessions"])


@router.post("", response_model=CreateSessionResponse, status_code=201)
async def create_session(
    visitor_id: str = Depends(require_visitor),
    container: AppContainer = Depends(get_container),
) -> CreateSessionResponse:
    """Create a new chat session for the authenticated visitor."""
    with container.session_factory() as db:
        session = ChatSession(visitor_id=visitor_id)
        db.add(session)
        db.flush()
        session_id = session.id
        db.commit()

    return CreateSessionResponse(visitor_id=visitor_id, session_id=session_id)


@router.get("", response_model=VisitorSessionListResponse)
async def list_visitor_sessions(
    visitor_id: str = Depends(require_visitor),
    container: AppContainer = Depends(get_container),
) -> VisitorSessionListResponse:
    """Return all sessions for the authenticated visitor, newest-first."""
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
    visitor_id: str = Depends(require_visitor),
    container: AppContainer = Depends(get_container),
) -> SessionDetailResponse:
    """Return basic info for a session owned by the authenticated visitor."""
    with container.session_factory() as db:
        session = db.get(ChatSession, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        if session.visitor_id != visitor_id:
            raise HTTPException(status_code=403, detail="Forbidden")
        return SessionDetailResponse(
            session_id=session.id,
            visitor_id=session.visitor_id,
            latest_risk_level=session.latest_risk_level.value,
            started_at=session.started_at,
        )


@router.get("/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    visitor_id: str = Depends(require_visitor),
    container: AppContainer = Depends(get_container),
) -> dict:
    """Return messages for a session owned by the authenticated visitor."""
    with container.session_factory() as db:
        session = db.get(ChatSession, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        if session.visitor_id != visitor_id:
            raise HTTPException(status_code=403, detail="Forbidden")
        msgs = db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.sequence_no)
        ).scalars().all()
    return {
        "messages": [
            {
                "message_id": m.id,
                "role": m.role.value if isinstance(m.role, ChatRole) else m.role,
                "content": m.content,
            }
            for m in msgs
        ]
    }
