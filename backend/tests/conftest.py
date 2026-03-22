"""Shared pytest fixtures available to all backend tests."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.db.base import Base
from backend.app.models import ChatMessage, ChatSession, ConversationSummary, VisitorProfile
from backend.app.models.enums import ChatRole, RiskLevel, SafetyMode


@pytest.fixture
def sqlite_session_factory():
    """Isolated in-memory SQLite session factory, recreated for each test."""
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        class_=Session,
    )
    yield factory
    engine.dispose()


@pytest.fixture
def seed_session(sqlite_session_factory):
    """Return a callable that seeds a chat session with messages and an optional summary.

    Usage::

        def test_something(seed_session, sqlite_session_factory):
            session_id, message_ids = seed_session(message_count=10, with_summary=True)
            service = SomeService(session_factory=sqlite_session_factory)
            ...
    """

    def _seed(
        *,
        message_count: int,
        latest_risk_level: RiskLevel = RiskLevel.L1,
        with_summary: bool = False,
        covered_until_index: int = 4,
    ) -> tuple[str, list[str]]:
        with sqlite_session_factory.begin() as session:
            visitor = VisitorProfile(display_name="Test Visitor", consent_accepted=True)
            session.add(visitor)
            session.flush()

            chat_session = ChatSession(
                visitor_id=visitor.id,
                latest_risk_level=latest_risk_level,
            )
            session.add(chat_session)
            session.flush()

            message_ids: list[str] = []
            for sequence_no in range(1, message_count + 1):
                role = ChatRole.USER if sequence_no % 2 else ChatRole.ASSISTANT
                message = ChatMessage(
                    session_id=chat_session.id,
                    sequence_no=sequence_no,
                    role=role,
                    content=f"message-{sequence_no}-{role.value}",
                    safety_mode=SafetyMode.STANDARD,
                )
                session.add(message)
                session.flush()
                message_ids.append(message.id)

            if with_summary:
                session.add(
                    ConversationSummary(
                        session_id=chat_session.id,
                        summary_text="existing summary",
                        summary_version=2,
                        covered_until_message_id=message_ids[covered_until_index - 1],
                        last_risk_level=latest_risk_level,
                        open_topics=["previous-topic"],
                        carry_over_advice=["previous-advice"],
                    )
                )

            return chat_session.id, message_ids

    return _seed
