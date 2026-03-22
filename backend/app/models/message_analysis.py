from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, Float, ForeignKey, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base


class MessageAnalysis(Base):
    __tablename__ = "message_analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    message_id: Mapped[str] = mapped_column(ForeignKey("chat_messages.id"), nullable=False, unique=True, index=True)
    emotion_label: Mapped[str] = mapped_column(String(50), nullable=False)
    emotion_scores: Mapped[dict[str, float] | None] = mapped_column(JSON, nullable=True)
    intent_label: Mapped[str] = mapped_column(String(50), nullable=False)
    intent_scores: Mapped[dict[str, float] | None] = mapped_column(JSON, nullable=True)
    intensity_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    keyword_hits: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
