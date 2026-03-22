from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, Enum, ForeignKey, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base
from backend.app.models.enums import AlertStatus, RiskLevel


class AlertEvent(Base):
    __tablename__ = "alert_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id"), nullable=False, index=True)
    message_id: Mapped[str | None] = mapped_column(ForeignKey("chat_messages.id"), nullable=True, index=True)
    risk_level: Mapped[RiskLevel] = mapped_column(Enum(RiskLevel, native_enum=False), nullable=False)
    reasons: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    status: Mapped[AlertStatus] = mapped_column(
        Enum(AlertStatus, native_enum=False),
        nullable=False,
        default=AlertStatus.OPEN,
        server_default=AlertStatus.OPEN.value,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
