"""Integration tests for session and chat HTTP endpoints."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.core.config import Settings
from backend.app.db.base import Base
from backend.app.dependencies.auth import require_visitor
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


_HIGH_RISK_ANALYSIS = AnalysisResult(
    emotion_label="hopelessness",
    emotion_scores={"hopelessness": 0.90, "sadness": 0.10},
    intent_label="crisis",
    intent_scores={"crisis": 0.90, "venting": 0.10},
    intensity_score=0.95,
    risk_aux_score=0.95,
    keyword_hits=["想死"],
)


def _make_container(session_factory, nlp_service=None) -> AppContainer:
    nlp = nlp_service or StubNLPService()
    context_service = ContextService(session_factory=session_factory)
    prompt_service = PromptService(context_service=context_service)
    risk_service = RiskService()
    resource_service = ResourceService(session_factory=session_factory)
    llm_provider = FakeLLMProvider()
    chat_service = ChatService(
        session_factory=session_factory,
        nlp_service=nlp,
        risk_service=risk_service,
        resource_service=resource_service,
        prompt_service=prompt_service,
        llm_provider=llm_provider,
    )
    return AppContainer(
        settings=Settings(),
        session_factory=session_factory,
        context_service=context_service,
        prompt_service=prompt_service,
        nlp_service=nlp,
        risk_service=risk_service,
        resource_service=resource_service,
        llm_provider=llm_provider,
        chat_service=chat_service,
    )


def _parse_sse(text: str) -> list[dict]:
    text = text.replace("\r\n", "\n")
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_type = None
        data_str = None
        for line in block.splitlines():
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_str = line[len("data:"):].strip()
        if event_type and data_str:
            events.append({"event": event_type, "data": json.loads(data_str)})
    return events


@pytest.fixture
def test_visitor_id(shared_session_factory) -> str:
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
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def high_risk_client(shared_session_factory, test_visitor_id):
    app = create_app()
    container = _make_container(
        shared_session_factory,
        nlp_service=StubNLPService(fixed_result=_HIGH_RISK_ANALYSIS),
    )
    app.dependency_overrides[get_container] = lambda: container
    app.dependency_overrides[require_visitor] = lambda: test_visitor_id
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_create_session(client):
    resp = client.post("/api/v1/sessions")
    assert resp.status_code == 201
    body = resp.json()
    assert "visitor_id" in body and body["visitor_id"]
    assert "session_id" in body and body["session_id"]


def test_create_session_with_display_name(client):
    resp = client.post("/api/v1/sessions")
    assert resp.status_code == 201
    assert resp.json()["session_id"]


def test_get_session(client):
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    resp = client.get(f"/api/v1/sessions/{session_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == session_id
    assert body["latest_risk_level"] == "L0"
    assert "started_at" in body


def test_get_session_not_found(client):
    resp = client.get("/api/v1/sessions/nonexistent-session-id")
    assert resp.status_code == 404


def test_list_sessions_returns_all_history(client):
    session_ids = [client.post("/api/v1/sessions").json()["session_id"] for _ in range(12)]

    resp = client.get("/api/v1/sessions")
    assert resp.status_code == 200
    sessions = resp.json()["sessions"]

    assert len(sessions) == 12
    assert {s["session_id"] for s in sessions} == set(session_ids)


def test_stream_normal_turn(client):
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    resp = client.post(
        f"/api/v1/chat/{session_id}/stream",
        json={"message": "我最近睡不好觉，感觉很焦虑"},
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)

    event_types = [e["event"] for e in events]
    assert event_types[0] == "meta"
    assert "token" in event_types
    assert event_types[-1] == "complete"
    assert "alert" not in event_types

    meta = events[0]["data"]
    assert meta["risk_level"] == "L1"
    assert meta["emotion"] == "anxiety"

    complete = events[-1]["data"]
    assert "message_id" in complete
    assert complete["latency_ms"] >= 0


def test_stream_high_risk_turn(high_risk_client):
    session_id = high_risk_client.post("/api/v1/sessions").json()["session_id"]
    resp = high_risk_client.post(
        f"/api/v1/chat/{session_id}/stream",
        json={"message": "我真的不想活了"},
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)

    event_types = [e["event"] for e in events]
    assert event_types[0] == "meta"
    assert event_types[1] == "alert"
    assert "token" in event_types
    assert event_types[-1] == "complete"

    meta = events[0]["data"]
    assert meta["risk_level"] == "L3"

    alert = events[1]["data"]
    assert alert["risk_level"] == "L3"


def test_stream_unknown_session(client):
    resp = client.post(
        "/api/v1/chat/nonexistent-id/stream",
        json={"message": "你好"},
    )
    assert resp.status_code == 404


def test_stream_empty_message(client):
    session_id = client.post("/api/v1/sessions").json()["session_id"]
    resp = client.post(
        f"/api/v1/chat/{session_id}/stream",
        json={"message": ""},
    )
    assert resp.status_code == 422


def test_stream_persists_messages(client, shared_session_factory):
    from sqlalchemy import select
    from backend.app.models.chat_message import ChatMessage

    session_id = client.post("/api/v1/sessions").json()["session_id"]
    client.post(
        f"/api/v1/chat/{session_id}/stream",
        json={"message": "今天心情不太好"},
    )

    with shared_session_factory() as db:
        count = db.execute(
            select(ChatMessage).where(ChatMessage.session_id == session_id)
        ).scalars().all()
    assert len(count) == 2
