from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SessionSummary(BaseModel):
    session_id: str
    visitor_id: str
    latest_risk_level: str
    started_at: datetime
    message_count: int
    dominant_emotion: str | None = None
    visitor_username: str | None = None
    visitor_real_name: str | None = None
    visitor_college: str | None = None
    visitor_student_id: str | None = None
    visitor_is_guest: bool = False


class AnalysisSummary(BaseModel):
    emotion_label: str
    intent_label: str
    intensity_score: float
    risk_score: float


class MessageWithAnalysis(BaseModel):
    message_id: str
    role: str
    content: str
    sequence_no: int
    safety_mode: str
    created_at: datetime
    analysis: AnalysisSummary | None = None


class AlertSummary(BaseModel):
    alert_id: str
    session_id: str
    risk_level: str
    reasons: list[str]
    status: str
    created_at: datetime


class AlertStatusUpdate(BaseModel):
    status: Literal['acknowledged', 'resolved']


class EmotionCount(BaseModel):
    emotion: str
    count: int


class RiskLevelCount(BaseModel):
    risk_level: str
    count: int


class ChartsData(BaseModel):
    emotion_distribution: list[EmotionCount]
    risk_distribution: list[RiskLevelCount]


class DashboardStats(BaseModel):
    total_sessions: int
    total_messages: int
    open_alerts: int
    l3_alerts: int


class VisitorSummary(BaseModel):
    visitor_id: str
    username: str | None
    real_name: str | None
    college: str | None
    student_id: str | None
    is_guest: bool
    created_at: datetime
    session_count: int
    latest_risk_level: str | None


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]


class MessageListResponse(BaseModel):
    messages: list[MessageWithAnalysis]


class AlertListResponse(BaseModel):
    alerts: list[AlertSummary]


class VisitorListResponse(BaseModel):
    visitors: list[VisitorSummary]


class VisitorDetailResponse(BaseModel):
    visitor: VisitorSummary
    sessions: list[SessionSummary]


class CounselorSummary(BaseModel):
    counselor_id: str
    username: str
    display_name: str | None
    college: str | None
    is_active: bool
    created_at: datetime


class CounselorListResponse(BaseModel):
    counselors: list[CounselorSummary]


class CreateCounselorRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=50)
    password: str = Field(..., min_length=6, max_length=100)
    college: str = Field(..., max_length=100)
    display_name: str | None = Field(None, max_length=100)
