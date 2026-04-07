from __future__ import annotations

import io
from datetime import date, datetime, time as dtime, timezone

import openpyxl
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from backend.app.core.colleges import COLLEGES
from backend.app.core.security import hash_password

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from backend.app.dependencies.auth import require_counselor, require_super_admin
from backend.app.dependencies.services import AppContainer, get_container
from backend.app.models.alert_event import AlertEvent
from backend.app.models.chat_message import ChatMessage
from backend.app.models.chat_session import ChatSession
from backend.app.models.counselor_account import CounselorAccount
from backend.app.models.enums import AlertStatus, RiskLevel
from backend.app.models.message_analysis import MessageAnalysis
from backend.app.models.visitor_profile import VisitorProfile
from backend.app.schemas.dashboard import (
    AlertListResponse,
    AlertStatusUpdate,
    AlertSummary,
    AnalysisSummary,
    ChartsData,
    CounselorListResponse,
    CounselorSummary,
    CreateCounselorRequest,
    DashboardStats,
    EmotionCount,
    MessageListResponse,
    MessageWithAnalysis,
    RiskLevelCount,
    SessionListResponse,
    SessionSummary,
    VisitorDetailResponse,
    VisitorListResponse,
    VisitorSummary,
)

router = APIRouter(tags=["dashboard"])

_RISK_ORDER = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


def _get_counselor_college(db: Session, counselor_id: str) -> str | None:
    """Return the counselor's college, or None for global access."""
    counselor = db.get(CounselorAccount, counselor_id)
    return counselor.college if counselor else None


def _visitor_filter(college: str | None):
    """Return a SQLAlchemy WHERE condition for college-scoped access, or None (no filter)."""
    if college is None:
        return None
    return (VisitorProfile.college == college) | (VisitorProfile.is_guest == True)  # noqa: E712


@router.get("/stats", response_model=DashboardStats)
async def get_stats(
    counselor_id: str = Depends(require_counselor),
    container: AppContainer = Depends(get_container),
) -> DashboardStats:
    """Return aggregate counts for the dashboard summary cards."""
    with container.session_factory() as db:
        college = _get_counselor_college(db, counselor_id)
        vf = _visitor_filter(college)

        if vf is None:
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
        else:
            session_sq = (
                select(ChatSession.id)
                .join(VisitorProfile, ChatSession.visitor_id == VisitorProfile.id)
                .where(vf)
                .subquery()
            )
            total_sessions = db.scalar(select(func.count(session_sq.c.id))) or 0
            total_messages = db.scalar(
                select(func.count(ChatMessage.id))
                .where(ChatMessage.session_id.in_(select(session_sq.c.id)))
            ) or 0
            open_alerts = db.scalar(
                select(func.count(AlertEvent.id))
                .where(AlertEvent.status == AlertStatus.OPEN)
                .where(AlertEvent.session_id.in_(select(session_sq.c.id)))
            ) or 0
            l3_alerts = db.scalar(
                select(func.count(AlertEvent.id))
                .where(AlertEvent.risk_level == RiskLevel.L3)
                .where(AlertEvent.session_id.in_(select(session_sq.c.id)))
            ) or 0

    return DashboardStats(
        total_sessions=total_sessions,
        total_messages=total_messages,
        open_alerts=open_alerts,
        l3_alerts=l3_alerts,
    )


@router.get("/charts", response_model=ChartsData)
async def get_charts(
    counselor_id: str = Depends(require_counselor),
    container: AppContainer = Depends(get_container),
) -> ChartsData:
    """Return aggregated emotion distribution and risk level distribution for charts."""
    with container.session_factory() as db:
        college = _get_counselor_college(db, counselor_id)
        vf = _visitor_filter(college)

        if vf is None:
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
        else:
            session_sq = (
                select(ChatSession.id)
                .join(VisitorProfile, ChatSession.visitor_id == VisitorProfile.id)
                .where(vf)
                .subquery()
            )
            msg_sq = (
                select(ChatMessage.id)
                .where(ChatMessage.session_id.in_(select(session_sq.c.id)))
                .subquery()
            )
            emotion_rows = db.execute(
                select(
                    MessageAnalysis.emotion_label,
                    func.count(MessageAnalysis.id).label("cnt"),
                )
                .where(MessageAnalysis.message_id.in_(select(msg_sq.c.id)))
                .group_by(MessageAnalysis.emotion_label)
                .order_by(func.count(MessageAnalysis.id).desc())
            ).all()

            risk_rows = db.execute(
                select(
                    ChatSession.latest_risk_level,
                    func.count(ChatSession.id).label("cnt"),
                )
                .where(ChatSession.id.in_(select(session_sq.c.id)))
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


def _build_session_summaries(db: Session, college: str | None, limit: int = 100) -> list[SessionSummary]:
    """Build SessionSummary list, filtered by counselor college."""
    vf = _visitor_filter(college)

    msg_count_sq = (
        select(
            ChatMessage.session_id,
            func.count(ChatMessage.id).label("message_count"),
        )
        .group_by(ChatMessage.session_id)
        .subquery()
    )

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
    severity_col = case(
        (emotion_count_sq.c.emotion_label == "hopelessness", 4),
        (emotion_count_sq.c.emotion_label.in_(["sadness", "anger", "shame"]), 3),
        (emotion_count_sq.c.emotion_label.in_(["anxiety", "fear"]), 2),
        else_=0,
    )
    rn_col = func.row_number().over(
        partition_by=emotion_count_sq.c.session_id,
        order_by=[emotion_count_sq.c.emotion_count.desc(), severity_col.desc()],
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

    base_query = (
        select(
            ChatSession,
            msg_count_sq.c.message_count,
            top_emotion_sq.c.emotion_label,
            VisitorProfile.username,
            VisitorProfile.real_name,
            VisitorProfile.college,
            VisitorProfile.student_id,
            VisitorProfile.is_guest,
        )
        .outerjoin(msg_count_sq, ChatSession.id == msg_count_sq.c.session_id)
        .outerjoin(top_emotion_sq, ChatSession.id == top_emotion_sq.c.session_id)
        .outerjoin(VisitorProfile, ChatSession.visitor_id == VisitorProfile.id)
        .order_by(ChatSession.started_at.desc())
        .limit(limit)
    )

    if vf is not None:
        base_query = base_query.where(vf)

    rows = db.execute(base_query).all()

    return [
        SessionSummary(
            session_id=row.ChatSession.id,
            visitor_id=row.ChatSession.visitor_id,
            latest_risk_level=row.ChatSession.latest_risk_level.value,
            started_at=row.ChatSession.started_at,
            message_count=row.message_count or 0,
            dominant_emotion=row.emotion_label,
            visitor_username=row.username,
            visitor_real_name=row.real_name,
            visitor_college=row.college,
            visitor_student_id=row.student_id,
            visitor_is_guest=row.is_guest or False,
        )
        for row in rows
    ]


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    counselor_id: str = Depends(require_counselor),
    container: AppContainer = Depends(get_container),
) -> SessionListResponse:
    """Return all sessions newest-first with message counts and dominant emotion."""
    with container.session_factory() as db:
        college = _get_counselor_college(db, counselor_id)
        sessions = _build_session_summaries(db, college)
    return SessionListResponse(sessions=sessions)


@router.get("/sessions/{session_id}/messages", response_model=MessageListResponse)
async def get_session_messages(
    session_id: str,
    _: str = Depends(require_counselor),
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
    counselor_id: str = Depends(require_counselor),
    container: AppContainer = Depends(get_container),
) -> AlertListResponse:
    """Return all alert events newest-first."""
    with container.session_factory() as db:
        college = _get_counselor_college(db, counselor_id)
        vf = _visitor_filter(college)

        base_query = (
            select(AlertEvent)
            .order_by(AlertEvent.created_at.desc())
            .limit(200)
        )

        if vf is not None:
            session_sq = (
                select(ChatSession.id)
                .join(VisitorProfile, ChatSession.visitor_id == VisitorProfile.id)
                .where(vf)
                .subquery()
            )
            base_query = base_query.where(AlertEvent.session_id.in_(select(session_sq.c.id)))

        rows = db.scalars(base_query).all()

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
    _: str = Depends(require_counselor),
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


@router.get("/visitors", response_model=VisitorListResponse)
async def list_visitors(
    counselor_id: str = Depends(require_counselor),
    container: AppContainer = Depends(get_container),
) -> VisitorListResponse:
    """Return all registered (non-guest) visitors, optionally filtered by counselor college."""
    with container.session_factory() as db:
        college = _get_counselor_college(db, counselor_id)

        session_count_sq = (
            select(
                ChatSession.visitor_id,
                func.count(ChatSession.id).label("session_count"),
            )
            .group_by(ChatSession.visitor_id)
            .subquery()
        )

        # Peak risk across all sessions: take MAX by ordinal
        peak_risk_sq = (
            select(
                ChatSession.visitor_id,
                func.max(
                    case(
                        (ChatSession.latest_risk_level == RiskLevel.L3, 3),
                        (ChatSession.latest_risk_level == RiskLevel.L2, 2),
                        (ChatSession.latest_risk_level == RiskLevel.L1, 1),
                        else_=0,
                    )
                ).label("peak_ord"),
            )
            .group_by(ChatSession.visitor_id)
            .subquery()
        )

        base_query = (
            select(
                VisitorProfile,
                session_count_sq.c.session_count,
                peak_risk_sq.c.peak_ord,
            )
            .outerjoin(session_count_sq, VisitorProfile.id == session_count_sq.c.visitor_id)
            .outerjoin(peak_risk_sq, VisitorProfile.id == peak_risk_sq.c.visitor_id)
            .where(VisitorProfile.is_guest == False)  # noqa: E712
            .order_by(VisitorProfile.created_at.desc())
            .limit(200)
        )

        if college is not None:
            base_query = base_query.where(VisitorProfile.college == college)

        rows = db.execute(base_query).all()

    _ORD_TO_LEVEL = {0: "L0", 1: "L1", 2: "L2", 3: "L3"}

    visitors = [
        VisitorSummary(
            visitor_id=row.VisitorProfile.id,
            username=row.VisitorProfile.username,
            real_name=row.VisitorProfile.real_name,
            college=row.VisitorProfile.college,
            student_id=row.VisitorProfile.student_id,
            is_guest=row.VisitorProfile.is_guest,
            created_at=row.VisitorProfile.created_at,
            session_count=row.session_count or 0,
            latest_risk_level=_ORD_TO_LEVEL.get(row.peak_ord, "L0") if row.peak_ord is not None else None,
        )
        for row in rows
    ]
    return VisitorListResponse(visitors=visitors)


@router.get("/visitors/{visitor_id}", response_model=VisitorDetailResponse)
async def get_visitor_detail(
    visitor_id: str,
    counselor_id: str = Depends(require_counselor),
    container: AppContainer = Depends(get_container),
) -> VisitorDetailResponse:
    """Return a visitor's profile and their recent sessions."""
    with container.session_factory() as db:
        college = _get_counselor_college(db, counselor_id)
        visitor = db.get(VisitorProfile, visitor_id)
        if visitor is None:
            raise HTTPException(status_code=404, detail="访客不存在")

        # Access control: check counselor can see this visitor
        if college is not None and not visitor.is_guest and visitor.college != college:
            raise HTTPException(status_code=403, detail="无权访问该访客")

        session_count_sq = (
            select(func.count(ChatSession.id))
            .where(ChatSession.visitor_id == visitor_id)
        )
        session_count = db.scalar(session_count_sq) or 0

        peak_ord = db.scalar(
            select(
                func.max(
                    case(
                        (ChatSession.latest_risk_level == RiskLevel.L3, 3),
                        (ChatSession.latest_risk_level == RiskLevel.L2, 2),
                        (ChatSession.latest_risk_level == RiskLevel.L1, 1),
                        else_=0,
                    )
                )
            ).where(ChatSession.visitor_id == visitor_id)
        )
        _ORD_TO_LEVEL = {0: "L0", 1: "L1", 2: "L2", 3: "L3"}
        peak_risk = _ORD_TO_LEVEL.get(peak_ord, "L0") if peak_ord is not None else None

        summary = VisitorSummary(
            visitor_id=visitor.id,
            username=visitor.username,
            real_name=visitor.real_name,
            college=visitor.college,
            student_id=visitor.student_id,
            is_guest=visitor.is_guest,
            created_at=visitor.created_at,
            session_count=session_count,
            latest_risk_level=peak_risk,
        )

        sessions = _build_session_summaries(db, college=None, limit=20)
        # Filter to this visitor only
        sessions = [s for s in sessions if s.visitor_id == visitor_id]

    return VisitorDetailResponse(visitor=summary, sessions=sessions)


@router.get("/export")
async def export_excel(
    counselor_id: str = Depends(require_counselor),
    container: AppContainer = Depends(get_container),
    # Session sheet filters
    search: str | None = Query(None),
    risk_level: str | None = Query(None),
    date_from: date | None = Query(None),
    date_to: date | None = Query(None),
    # Alert sheet filters
    alert_status: AlertStatus | None = Query(None),
    alert_risk_level: RiskLevel | None = Query(None),
    alert_date_from: date | None = Query(None),
    alert_date_to: date | None = Query(None),
) -> StreamingResponse:
    """Export sessions and alerts as an Excel workbook."""
    _EMOTION_ZH = {
        "neutral": "平静", "sadness": "悲伤", "anxiety": "焦虑",
        "anger": "愤怒", "fear": "恐惧", "shame": "羞耻", "hopelessness": "绝望",
    }

    with container.session_factory() as db:
        college = _get_counselor_college(db, counselor_id)
        sessions = _build_session_summaries(db, college, limit=10000)

        # Apply session filters in Python (post-fetch)
        if search:
            q = search.lower()
            sessions = [s for s in sessions if
                q in (s.visitor_real_name or '').lower()
                or q in (s.visitor_username or '').lower()
                or q in (s.visitor_student_id or '').lower()]
        if risk_level:
            sessions = [s for s in sessions if s.latest_risk_level == risk_level]
        if date_from:
            sessions = [s for s in sessions if s.started_at and s.started_at.date() >= date_from]
        if date_to:
            sessions = [s for s in sessions if s.started_at and s.started_at.date() <= date_to]

        vf = _visitor_filter(college)
        alert_query = (
            select(AlertEvent)
            .order_by(AlertEvent.created_at.desc())
            .limit(10000)
        )
        if vf is not None:
            session_sq = (
                select(ChatSession.id)
                .join(VisitorProfile, ChatSession.visitor_id == VisitorProfile.id)
                .where(vf)
                .subquery()
            )
            alert_query = alert_query.where(AlertEvent.session_id.in_(select(session_sq.c.id)))
        if alert_status:
            alert_query = alert_query.where(AlertEvent.status == alert_status)
        if alert_risk_level:
            alert_query = alert_query.where(AlertEvent.risk_level == alert_risk_level)
        if alert_date_from:
            alert_query = alert_query.where(AlertEvent.created_at >= datetime.combine(alert_date_from, dtime.min, tzinfo=timezone.utc))
        if alert_date_to:
            alert_query = alert_query.where(AlertEvent.created_at <= datetime.combine(alert_date_to, dtime.max, tzinfo=timezone.utc))
        alerts = db.scalars(alert_query).all()

    wb = openpyxl.Workbook()

    # Sheet 1: sessions
    ws1 = wb.active
    ws1.title = "会话列表"
    ws1.append(["会话ID", "学生姓名", "学号", "学院", "是否游客", "开始时间", "消息数", "主导情绪", "最高风险等级"])
    for s in sessions:
        display_name = s.visitor_real_name or s.visitor_username or ("匿名" if s.visitor_is_guest else "未知")
        ws1.append([
            s.session_id,
            display_name,
            s.visitor_student_id or "",
            s.visitor_college or "",
            "是" if s.visitor_is_guest else "否",
            s.started_at.strftime("%Y-%m-%d %H:%M") if s.started_at else "",
            s.message_count,
            _EMOTION_ZH.get(s.dominant_emotion or "", s.dominant_emotion or ""),
            s.latest_risk_level,
        ])

    # Sheet 2: alerts
    ws2 = wb.create_sheet("预警记录")
    ws2.append(["预警ID", "会话ID", "风险等级", "触发原因", "状态", "创建时间"])
    _STATUS_ZH = {"open": "待处理", "acknowledged": "已确认", "resolved": "已解决"}
    for a in alerts:
        ws2.append([
            a.id,
            a.session_id,
            a.risk_level.value,
            "；".join(a.reasons) if a.reasons else "",
            _STATUS_ZH.get(a.status.value, a.status.value),
            a.created_at.strftime("%Y-%m-%d %H:%M") if a.created_at else "",
        ])

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    filename = f"xinyu_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return StreamingResponse(
        content=buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/counselors", response_model=CounselorListResponse)
async def list_counselors(
    _: str = Depends(require_super_admin),
    container: AppContainer = Depends(get_container),
) -> CounselorListResponse:
    """List all counselor accounts (super admin only)."""
    with container.session_factory() as db:
        rows = db.scalars(
            select(CounselorAccount).order_by(CounselorAccount.created_at.asc())
        ).all()
    return CounselorListResponse(counselors=[
        CounselorSummary(
            counselor_id=c.id,
            username=c.username,
            display_name=c.display_name,
            college=c.college,
            is_active=c.is_active,
            created_at=c.created_at,
        )
        for c in rows
    ])


@router.post("/counselors", response_model=CounselorSummary, status_code=201)
async def create_counselor(
    body: CreateCounselorRequest,
    _: str = Depends(require_super_admin),
    container: AppContainer = Depends(get_container),
) -> CounselorSummary:
    """Create a new college-level counselor account (super admin only)."""
    if body.college not in COLLEGES:
        raise HTTPException(status_code=422, detail="无效的学院名称")

    with container.session_factory() as db:
        existing = db.scalar(
            select(CounselorAccount).where(CounselorAccount.username == body.username)
        )
        if existing is not None:
            raise HTTPException(status_code=409, detail="用户名已被占用")

        counselor = CounselorAccount(
            username=body.username,
            password_hash=hash_password(body.password),
            display_name=body.display_name,
            college=body.college,
            is_active=True,
        )
        db.add(counselor)
        db.flush()
        db.commit()
        db.refresh(counselor)

        return CounselorSummary(
            counselor_id=counselor.id,
            username=counselor.username,
            display_name=counselor.display_name,
            college=counselor.college,
            is_active=counselor.is_active,
            created_at=counselor.created_at,
        )


@router.patch("/counselors/{counselor_id}", response_model=CounselorSummary)
async def toggle_counselor_active(
    counselor_id: str,
    _: str = Depends(require_super_admin),
    container: AppContainer = Depends(get_container),
) -> CounselorSummary:
    """Toggle a counselor's is_active status (super admin only)."""
    with container.session_factory() as db:
        counselor = db.get(CounselorAccount, counselor_id)
        if counselor is None:
            raise HTTPException(status_code=404, detail="咨询师不存在")
        if counselor.college is None:
            raise HTTPException(status_code=403, detail="不可停用超级管理员账户")

        counselor.is_active = not counselor.is_active
        db.commit()
        db.refresh(counselor)

        return CounselorSummary(
            counselor_id=counselor.id,
            username=counselor.username,
            display_name=counselor.display_name,
            college=counselor.college,
            is_active=counselor.is_active,
            created_at=counselor.created_at,
        )
