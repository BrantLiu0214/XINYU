from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class CreateSessionRequest(BaseModel):
    display_name: str | None = None
    visitor_id: str | None = None  # if provided, reuse existing visitor


class CreateSessionResponse(BaseModel):
    visitor_id: str
    session_id: str


class SessionDetailResponse(BaseModel):
    session_id: str
    visitor_id: str
    latest_risk_level: str
    started_at: datetime


class VisitorSessionSummary(BaseModel):
    session_id: str
    started_at: datetime
    message_count: int
    latest_risk_level: str


class VisitorSessionListResponse(BaseModel):
    sessions: list[VisitorSessionSummary]
