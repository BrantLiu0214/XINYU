from datetime import datetime
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.base import Base


class VisitorProfile(Base):
    __tablename__ = "visitor_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    username: Mapped[str | None] = mapped_column(String(100), nullable=True, unique=True)
    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    display_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    real_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    college: Mapped[str | None] = mapped_column(String(100), nullable=True)
    student_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    is_guest: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    consent_accepted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
