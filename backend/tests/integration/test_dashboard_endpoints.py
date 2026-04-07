"""Integration tests for Module 07 dashboard read endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.core.config import Settings
from backend.app.db.base import Base
from backend.app.dependencies.auth import require_counselor, require_visitor
from backend.app.dependencies.services import AppContainer, get_container
from backend.app.main import create_app
from backend.app.models import (  # noqa: F401
    AlertEvent,
    ChatMessage,
    ChatSession,
    ConversationSummary,
    CounselorAccount,
    MessageAnalysis,
    ResourceCatalog,
    VisitorProfile,
)
from backend.app.schemas.analysis import AnalysisResult
from backend.app.services.chat_service import ChatService
from backend.app.services.context_service import ContextService
from backend.app.services.llm_service import FakeLLMProvider
from backend.app.services.nlp_service import StubNLPService
from backend.app.services.prompt_service import PromptService
from backend.app.services.resource_service import ResourceService
from backend.app.services.risk_service import RiskService


@pytest.fixture
def shared_session_factory():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
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


def _make_container(session_factory) -> AppContainer:
    nlp = StubNLPService()
    ctx = ContextService(session_factory=session_factory)
    ps = PromptService(context_service=ctx)
    rs = RiskService()
    rsvc = ResourceService(session_factory=session_factory)
    llm = FakeLLMProvider()
    cs = ChatService(
        session_factory=session_factory,
        nlp_service=nlp,
        risk_service=rs,
        resource_service=rsvc,
        prompt_service=ps,
        llm_provider=llm,
    )
    return AppContainer(
        settings=Settings(),
        session_factory=session_factory,
        context_service=ctx,
        prompt_service=ps,
        nlp_service=nlp,
        risk_service=rs,
        resource_service=rsvc,
        llm_provider=llm,
        chat_service=cs,
    )


@pytest.fixture
def test_visitor_id(shared_session_factory) -> str:
    """Pre-seed a visitor profile and return its ID for auth dependency override."""
    with shared_session_factory.begin() as session:
        visitor = VisitorProfile(display_name="Test Visitor", consent_accepted=True)
        session.add(visitor)
        session.flush()
        return visitor.id


@pytest.fixture
def client(shared_session_factory, test_visitor_id):
    app = create_app()
    container = _make_container(shared_session_factory)
    app.dependency_overrides[get_container] = lambda: container
    app.dependency_overrides[require_visitor] = lambda: test_visitor_id
    app.dependency_overrides[require_counselor] = lambda: "test-counselor-id"
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_list_sessions_empty(client):
    resp = client.get("/api/v1/dashboard/sessions")
    assert resp.status_code == 200
    assert resp.json() == {"sessions": []}


def test_list_sessions_after_create(client):
    client.post("/api/v1/sessions")
    resp = client.get("/api/v1/dashboard/sessions")
    assert resp.status_code == 200
    sessions = resp.json()["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["latest_risk_level"] == "L0"
    assert sessions[0]["message_count"] == 0


def test_list_sessions_message_count(client):
    r = client.post("/api/v1/sessions")
    session_id = r.json()["session_id"]
    client.post(f"/api/v1/chat/{session_id}/stream", json={"message": "浣犲ソ"})

    resp = client.get("/api/v1/dashboard/sessions")
    session = resp.json()["sessions"][0]
    assert session["message_count"] == 2  # user + assistant


def test_get_session_messages_empty(client):
    r = client.post("/api/v1/sessions")
    session_id = r.json()["session_id"]
    resp = client.get(f"/api/v1/dashboard/sessions/{session_id}/messages")
    assert resp.status_code == 200
    assert resp.json() == {"messages": []}


def test_get_session_messages_after_turn(client):
    r = client.post("/api/v1/sessions")
    session_id = r.json()["session_id"]
    client.post(f"/api/v1/chat/{session_id}/stream", json={"message": "鎴戝緢闅惧彈"})

    resp = client.get(f"/api/v1/dashboard/sessions/{session_id}/messages")
    assert resp.status_code == 200
    messages = resp.json()["messages"]
    assert len(messages) == 2
    user_msg = messages[0]
    assert user_msg["role"] == "user"
    assert user_msg["analysis"] is not None
    assert user_msg["analysis"]["emotion_label"] == "anxiety"
    assistant_msg = messages[1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["analysis"] is None  # only user messages get analysis


def test_get_session_messages_not_found(client):
    resp = client.get("/api/v1/dashboard/sessions/nonexistent/messages")
    assert resp.status_code == 404


def test_list_alerts_empty(client):
    resp = client.get("/api/v1/dashboard/alerts")
    assert resp.status_code == 200
    assert resp.json() == {"alerts": []}


def test_list_alerts_after_crisis_turn(client):
    # Use a high-risk NLP stub to trigger an alert event.
    from backend.app.main import create_app as _create_app
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(
        bind=engine, autoflush=False, autocommit=False,
        expire_on_commit=False, class_=Session,
    )

    high_risk_analysis = AnalysisResult(
        emotion_label="hopelessness",
        emotion_scores={"hopelessness": 0.90, "sadness": 0.10},
        intent_label="crisis",
        intent_scores={"crisis": 0.90, "venting": 0.10},
        intensity_score=0.95,
        risk_aux_score=0.95,
        keyword_hits=["鎯虫"],
    )
    container = _make_container.__wrapped__(factory) if hasattr(_make_container, '__wrapped__') else None
    # Build container manually with high-risk NLP stub
    nlp = StubNLPService(fixed_result=high_risk_analysis)
    ctx = ContextService(session_factory=factory)
    ps = PromptService(context_service=ctx)
    rs = RiskService()
    rsvc = ResourceService(session_factory=factory)
    llm = FakeLLMProvider()
    cs = ChatService(session_factory=factory, nlp_service=nlp, risk_service=rs,
                     resource_service=rsvc, prompt_service=ps, llm_provider=llm)
    hr_container = AppContainer(settings=Settings(), session_factory=factory,
                                context_service=ctx, prompt_service=ps, nlp_service=nlp,
                                risk_service=rs, resource_service=rsvc, llm_provider=llm,
                                chat_service=cs)

    with factory.begin() as s:
        visitor = VisitorProfile(display_name="HR Visitor", consent_accepted=True)
        s.add(visitor)
        s.flush()
        hr_visitor_id = visitor.id

    app = _create_app()
    app.dependency_overrides[get_container] = lambda: hr_container
    app.dependency_overrides[require_visitor] = lambda: hr_visitor_id
    app.dependency_overrides[require_counselor] = lambda: "test-counselor-id"
    with TestClient(app) as c:
        session_id = c.post("/api/v1/sessions").json()["session_id"]
        c.post(f"/api/v1/chat/{session_id}/stream", json={"message": "我不想活了"})
        resp = c.get("/api/v1/dashboard/alerts")

    assert resp.status_code == 200
    alerts = resp.json()["alerts"]
    assert len(alerts) == 1
    assert alerts[0]["risk_level"] == "L3"
    assert alerts[0]["status"] == "open"
    assert alerts[0]["session_id"] == session_id

