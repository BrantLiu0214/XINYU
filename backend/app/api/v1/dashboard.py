from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select

from fastapi import APIRouter, Depends, HTTPException

from backend.app.dependencies.services import AppContainer, get_container
from backend.app.models.alert_event import AlertEvent
from backend.app.models.chat_message import ChatMessage
from backend.app.models.chat_session import ChatSession
from backend.app.models.enums import AlertStatus, RiskLevel
from backend.app.models.message_analysis import MessageAnalysis
from backend.app.schemas.dashboard import (
    AlertListResponse,
    AlertStatusUpdate,
    AlertSummary,
    AnalysisSummary,
    ChartsData,
    DashboardStats,
    EmotionCount,
    MessageListResponse,
    MessageWithAnalysis,
    RiskLevelCount,
    SessionListResponse,
    SessionSummary,
)

router = APIRouter(tags=["dashboard"])


@router.get("/stats", response_model=DashboardStats)
async def get_stats(
    container: AppContainer = Depends(get_container),
) -> DashboardStats:
    """Return aggregate counts for the dashboard summary cards."""
    with container.session_factory() as db:
        total_sessions = db.scalar(select(func.count(ChatSession.id))) or 0
        total_messages = db.scalar(select(func.count(ChatMessage.id))) or 0
        open_alerts = db.scalar(
            select(func.count(AlertEvent.id))
            .where(AlertEvent.status == AlertStatus.OPEN)
        ) or 0
        l3_alerts = db.scalar(
            select(func.count(AlertEvent.id))
            .where(AlertEvent.risk_level == RiskLevel.L3)
        ) or 0

    return DashboardStats(
        total_sessions=total_sessions,
        total_messages=total_messages,
        open_alerts=open_alerts,
        l3_alerts=l3_alerts,
    )


@router.get("/charts", response_model=ChartsData)
async def get_charts(
    container: AppContainer = Depends(get_container),
) -> ChartsData:
    """Return aggregated emotion distribution and risk level distribution for charts."""
    with container.session_factory() as db:
        emotion_rows = db.execute(
            select(
                MessageAnalysis.emotion_label,
                func.count(MessageAnalysis.id).label("cnt"),
            )
            .group_by(MessageAnalysis.emotion_label)
            .order_by(func.count(MessageAnalysis.id).desc())
        ).all()

        risk_rows = db.execute(
            select(
                ChatSession.latest_risk_level,
                func.count(ChatSession.id).label("cnt"),
            )
            .group_by(ChatSession.latest_risk_level)
            .order_by(ChatSession.latest_risk_level.asc())
        ).all()

    return ChartsData(
        emotion_distribution=[
            EmotionCount(emotion=row.emotion_label, count=row.cnt)
            for row in emotion_rows
        ],
        risk_distribution=[
            RiskLevelCount(risk_level=row.latest_risk_level, count=row.cnt)
            for row in risk_rows
        ],
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    container: AppContainer = Depends(get_container),
) -> SessionListResponse:
    """Return all sessions newest-first with message counts and dominant emotion."""
    with container.session_factory() as db:
        # Subquery: count messages per session.
        msg_count_sq = (
            select(
                ChatMessage.session_id,
                func.count(ChatMessage.id).label("message_count"),
            )
            .group_by(ChatMessage.session_id)
            .subquery()
        )

        # Subquery: rank emotion labels per session by frequency, then pick rank=1.
        emotion_count_sq = (
            select(
                ChatMessage.session_id,
                MessageAnalysis.emotion_label,
                func.count(MessageAnalysis.id).label("emotion_count"),
            )
            .join(MessageAnalysis, ChatMessage.id == MessageAnalysis.message_id)
            .group_by(ChatMessage.session_id, MessageAnalysis.emotion_label)
            .subquery()
        )
        rn_col = func.row_number().over(
            partition_by=emotion_count_sq.c.session_id,
            order_by=emotion_count_sq.c.emotion_count.desc(),
        ).label("rn")
        ranked_sq = select(
            emotion_count_sq.c.session_id,
            emotion_count_sq.c.emotion_label,
            rn_col,
        ).subquery()
        top_emotion_sq = (
            select(ranked_sq.c.session_id, ranked_sq.c.emotion_label)
            .where(ranked_sq.c.rn == 1)
            .subquery()
        )

        rows = db.execute(
            select(
                ChatSession,
                msg_count_sq.c.message_count,
                top_emotion_sq.c.emotion_label,
            )
            .outerjoin(msg_count_sq, ChatSession.id == msg_count_sq.c.session_id)
            .outerjoin(top_emotion_sq, ChatSession.id == top_emotion_sq.c.session_id)
            .order_by(ChatSession.started_at.desc())
            .limit(100)
        ).all()

    sessions = [
        SessionSummary(
            session_id=row.ChatSession.id,
            visitor_id=row.ChatSession.visitor_id,
            latest_risk_level=row.ChatSession.latest_risk_level.value,
            started_at=row.ChatSession.started_at,
            message_count=row.message_count or 0,
            dominant_emotion=row.emotion_label,
        )
        for row in rows
    ]
    return SessionListResponse(sessions=sessions)


@router.get("/sessions/{session_id}/messages", response_model=MessageListResponse)
async def get_session_messages(
    session_id: str,
    container: AppContainer = Depends(get_container),
) -> MessageListResponse:
    """Return all messages for a session with their NLP analyses."""
    with container.session_factory() as db:
        if db.get(ChatSession, session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found")

        rows = db.execute(
            select(ChatMessage, MessageAnalysis)
            .outerjoin(
                MessageAnalysis,
                ChatMessage.id == MessageAnalysis.message_id,
            )
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.sequence_no.asc())
        ).all()

    messages = [
        MessageWithAnalysis(
            message_id=row.ChatMessage.id,
            role=row.ChatMessage.role.value,
            content=row.ChatMessage.content,
            sequence_no=row.ChatMessage.sequence_no,
            safety_mode=row.ChatMessage.safety_mode.value,
            created_at=row.ChatMessage.created_at,
            analysis=AnalysisSummary(
                emotion_label=row.MessageAnalysis.emotion_label,
                intent_label=row.MessageAnalysis.intent_label,
                intensity_score=row.MessageAnalysis.intensity_score,
                risk_score=row.MessageAnalysis.risk_score,
            ) if row.MessageAnalysis else None,
        )
        for row in rows
    ]
    return MessageListResponse(messages=messages)


@router.get("/alerts", response_model=AlertListResponse)
async def list_alerts(
    container: AppContainer = Depends(get_container),
) -> AlertListResponse:
    """Return all alert events newest-first."""
    with container.session_factory() as db:
        rows = db.scalars(
            select(AlertEvent)
            .order_by(AlertEvent.created_at.desc())
            .limit(200)
        ).all()

    alerts = [
        AlertSummary(
            alert_id=row.id,
            session_id=row.session_id,
            risk_level=row.risk_level.value,
            reasons=row.reasons,
            status=row.status.value,
            created_at=row.created_at,
        )
        for row in rows
    ]
    return AlertListResponse(alerts=alerts)


@router.patch("/alerts/{alert_id}", response_model=AlertSummary)
async def update_alert_status(
    alert_id: str,
    body: AlertStatusUpdate,
    container: AppContainer = Depends(get_container),
) -> AlertSummary:
    """Update an alert's status to acknowledged or resolved."""
    with container.session_factory() as db:
        alert = db.get(AlertEvent, alert_id)
        if alert is None:
            raise HTTPException(status_code=404, detail="Alert not found")

        alert.status = AlertStatus(body.status)
        if body.status == 'resolved':
            alert.resolved_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(alert)

        return AlertSummary(
            alert_id=alert.id,
            session_id=alert.session_id,
            risk_level=alert.risk_level.value,
            reasons=alert.reasons,
            status=alert.status.value,
            created_at=alert.created_at,
        )
