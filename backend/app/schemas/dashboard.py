from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class SessionSummary(BaseModel):
    session_id: str
    visitor_id: str
    latest_risk_level: str
    started_at: datetime
    message_count: int
    dominant_emotion: str | None = None


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


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]


class MessageListResponse(BaseModel):
    messages: list[MessageWithAnalysis]


class AlertListResponse(BaseModel):
    alerts: list[AlertSummary]
