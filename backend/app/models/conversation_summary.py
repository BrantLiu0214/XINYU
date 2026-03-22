from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base
from backend.app.models.enums import RiskLevel


class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id"), nullable=False, unique=True, index=True)
    summary_text: Mapped[str] = mapped_column(Text, nullable=False)
    summary_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    covered_until_message_id: Mapped[str | None] = mapped_column(ForeignKey("chat_messages.id"), nullable=True)
    last_risk_level: Mapped[RiskLevel] = mapped_column(
        Enum(RiskLevel, native_enum=False),
        nullable=False,
        default=RiskLevel.L0,
        server_default=RiskLevel.L0.value,
    )
    open_topics: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    carry_over_advice: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
