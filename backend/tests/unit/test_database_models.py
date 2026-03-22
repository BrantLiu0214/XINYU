from backend.app.db.base import Base
from backend.app.dependencies.services import build_container
from backend.app.models import (
    AlertEvent,
    ChatMessage,
    ChatSession,
    ConversationSummary,
    CounselorAccount,
    MessageAnalysis,
    ResourceCatalog,
    VisitorProfile,
)


def test_metadata_contains_core_tables() -> None:
    table_names = set(Base.metadata.tables.keys())
    assert table_names == {
        "visitor_profiles",
        "counselor_accounts",
        "chat_sessions",
        "chat_messages",
        "message_analyses",
        "alert_events",
        "conversation_summaries",
        "resource_catalog",
    }


def test_container_exposes_session_factory() -> None:
    container = build_container()
    assert container.session_factory is not None


def test_core_models_are_importable() -> None:
    assert VisitorProfile.__tablename__ == "visitor_profiles"
    assert CounselorAccount.__tablename__ == "counselor_accounts"
    assert ChatSession.__tablename__ == "chat_sessions"
    assert ChatMessage.__tablename__ == "chat_messages"
    assert MessageAnalysis.__tablename__ == "message_analyses"
    assert AlertEvent.__tablename__ == "alert_events"
    assert ConversationSummary.__tablename__ == "conversation_summaries"
    assert ResourceCatalog.__tablename__ == "resource_catalog"
