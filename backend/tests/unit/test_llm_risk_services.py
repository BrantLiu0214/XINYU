"""Unit tests for Module 04: LLM provider, risk engine, and chat orchestration."""
from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.chat_message import ChatMessage
from backend.app.models.chat_session import ChatSession
from backend.app.models.enums import ChatRole, RiskLevel
from backend.app.models.message_analysis import MessageAnalysis
from backend.app.schemas.analysis import AnalysisResult, RiskAssessment
from backend.app.schemas.prompt import PromptBundle
from backend.app.schemas.stream import AlertEvent, CompleteEvent, MetaEvent, TokenEvent
from backend.app.services.chat_service import ChatService
from backend.app.services.context_service import ContextService
from backend.app.services.llm_service import FakeLLMProvider
from backend.app.services.nlp_service import StubNLPService
from backend.app.services.prompt_service import PromptService
from backend.app.services.resource_service import ResourceService
from backend.app.services.risk_service import RiskService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LOW_RISK_ANALYSIS = AnalysisResult(
    emotion_label="anxiety",
    emotion_scores={"anxiety": 0.65, "neutral": 0.35},
    intent_label="venting",
    intent_scores={"venting": 0.80},
    intensity_score=0.40,
    risk_aux_score=0.10,
    keyword_hits=[],
)

_L3_KEYWORD_ANALYSIS = AnalysisResult(
    emotion_label="hopelessness",
    emotion_scores={"hopelessness": 0.90},
    intent_label="crisis",
    intent_scores={"crisis": 0.90},
    intensity_score=0.85,
    risk_aux_score=0.75,
    keyword_hits=["自杀"],
)

_L2_SCORE_ANALYSIS = AnalysisResult(
    emotion_label="sadness",
    emotion_scores={"sadness": 0.70},
    intent_label="venting",
    intent_scores={"venting": 0.80},
    intensity_score=0.55,
    risk_aux_score=0.55,
    keyword_hits=[],
)


def _make_chat_service(session_factory, nlp_analysis: AnalysisResult | None = None) -> ChatService:
    context_service = ContextService(session_factory=session_factory)
    prompt_service = PromptService(context_service=context_service)
    return ChatService(
        session_factory=session_factory,
        nlp_service=StubNLPService(fixed_result=nlp_analysis),
        risk_service=RiskService(),
        resource_service=ResourceService(session_factory=session_factory),
        prompt_service=prompt_service,
        llm_provider=FakeLLMProvider(),
    )


async def _collect_events(service: ChatService, session_id: str, message: str) -> list:
    events = []
    async for event in service.stream_chat(session_id, message):
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# RiskService — unit tests
# ---------------------------------------------------------------------------


def test_risk_l3_on_explicit_keyword():
    risk = RiskService().evaluate(_L3_KEYWORD_ANALYSIS, recent_levels=[])
    assert risk.risk_level == "L3"
    assert risk.risk_score >= 0.85
    assert any("自杀" in r for r in risk.reasons)


def test_risk_l3_on_crisis_intent_and_high_aux():
    analysis = AnalysisResult(
        emotion_label="fear",
        intent_label="crisis",
        intensity_score=0.70,
        risk_aux_score=0.70,
        keyword_hits=[],
    )
    risk = RiskService().evaluate(analysis, recent_levels=[])
    assert risk.risk_level == "L3"


def test_risk_l3_on_hopelessness_and_max_intensity():
    analysis = AnalysisResult(
        emotion_label="hopelessness",
        intent_label="venting",
        intensity_score=0.95,
        risk_aux_score=0.20,
        keyword_hits=[],
    )
    risk = RiskService().evaluate(analysis, recent_levels=[])
    assert risk.risk_level == "L3"


def test_risk_l2_on_explicit_keyword():
    analysis = AnalysisResult(
        emotion_label="sadness",
        intent_label="venting",
        intensity_score=0.40,
        risk_aux_score=0.20,
        keyword_hits=["绝望"],
    )
    risk = RiskService().evaluate(analysis, recent_levels=[])
    assert risk.risk_level == "L2"


def test_risk_l2_on_high_aux_score():
    risk = RiskService().evaluate(_L2_SCORE_ANALYSIS, recent_levels=[])
    assert risk.risk_level == "L2"
    assert risk.risk_score >= 0.55


def test_risk_l1_on_moderate_intensity():
    analysis = AnalysisResult(
        emotion_label="anxiety",
        intent_label="venting",
        intensity_score=0.65,
        risk_aux_score=0.10,
        keyword_hits=[],
    )
    risk = RiskService().evaluate(analysis, recent_levels=[])
    assert risk.risk_level == "L1"


def test_risk_l0_on_normal_input():
    risk = RiskService().evaluate(_LOW_RISK_ANALYSIS, recent_levels=[])
    assert risk.risk_level == "L0"
    assert risk.risk_score < 0.25


def test_risk_escalating_trend_promotes_to_l2():
    """Two recent high-risk levels should trigger L2 even on low-score analysis."""
    analysis = AnalysisResult(
        emotion_label="anxiety",
        intent_label="venting",
        intensity_score=0.40,
        risk_aux_score=0.10,
        keyword_hits=[],
    )
    risk = RiskService().evaluate(analysis, recent_levels=["L2", "L3"])
    assert risk.risk_level == "L2"
    assert any("持续上升" in r for r in risk.reasons)


def test_risk_reasons_populated():
    risk = RiskService().evaluate(_LOW_RISK_ANALYSIS, recent_levels=[])
    assert len(risk.reasons) >= 1


# ---------------------------------------------------------------------------
# FakeLLMProvider — deterministic behaviour
# ---------------------------------------------------------------------------


def _make_bundle(system_prompt: str) -> PromptBundle:
    analysis = AnalysisResult(
        emotion_label="neutral", intent_label="venting",
        intensity_score=0.30, risk_aux_score=0.10, keyword_hits=[],
    )
    risk = RiskAssessment(
        risk_score=0.10, risk_level="L0", reasons=[], suggested_resource_level="none",
    )
    return PromptBundle(system_prompt=system_prompt, user_message="hi", analysis=analysis, risk=risk)


async def test_fake_llm_normal_path_yields_tokens():
    fake = FakeLLMProvider()
    tokens = [t async for t in fake.stream_reply(_make_bundle("normal support path"))]
    assert tokens == ["听起来你最近压力不小，", "先给自己一些空间。"]


async def test_fake_llm_crisis_path_yields_tokens():
    fake = FakeLLMProvider()
    tokens = [t async for t in fake.stream_reply(_make_bundle("this is the high-risk support path"))]
    assert tokens == ["我听到你了，", "这一刻一定很难熬。", "你现在安全吗？"]


# ---------------------------------------------------------------------------
# StubNLPService — deterministic behaviour
# ---------------------------------------------------------------------------


async def test_stub_nlp_default_returns_fixed_result():
    stub = StubNLPService()
    result = await stub.analyze("anything")
    assert result.emotion_label == "anxiety"
    assert result.risk_aux_score == 0.20


async def test_stub_nlp_custom_result_overrides_default():
    stub = StubNLPService(fixed_result=_L3_KEYWORD_ANALYSIS)
    result = await stub.analyze("whatever")
    assert result.keyword_hits == ["自杀"]


# ---------------------------------------------------------------------------
# ChatService — normal path
# ---------------------------------------------------------------------------


async def test_chat_service_normal_path_event_order(sqlite_session_factory, seed_session):
    session_id, _ = seed_session(message_count=2)
    service = _make_chat_service(sqlite_session_factory, nlp_analysis=_LOW_RISK_ANALYSIS)

    events = await _collect_events(service, session_id, "我今天有点累")

    types = [type(e) for e in events]
    assert types[0] is MetaEvent
    assert AlertEvent not in types
    assert TokenEvent in types
    assert types[-1] is CompleteEvent


async def test_chat_service_normal_path_token_content(sqlite_session_factory, seed_session):
    session_id, _ = seed_session(message_count=2)
    service = _make_chat_service(sqlite_session_factory, nlp_analysis=_LOW_RISK_ANALYSIS)

    events = await _collect_events(service, session_id, "我今天有点累")
    token_texts = [e.data.text for e in events if isinstance(e, TokenEvent)]

    assert token_texts == ["听起来你最近压力不小，", "先给自己一些空间。"]


async def test_chat_service_normal_path_meta_content(sqlite_session_factory, seed_session):
    session_id, _ = seed_session(message_count=2)
    service = _make_chat_service(sqlite_session_factory, nlp_analysis=_LOW_RISK_ANALYSIS)

    events = await _collect_events(service, session_id, "我今天有点累")
    meta = events[0]

    assert isinstance(meta, MetaEvent)
    assert meta.data.emotion == "anxiety"
    assert meta.data.intent == "venting"
    assert meta.data.risk_level == "L0"


# ---------------------------------------------------------------------------
# ChatService — high-risk path
# ---------------------------------------------------------------------------


async def test_chat_service_high_risk_path_event_order(sqlite_session_factory, seed_session):
    session_id, _ = seed_session(message_count=2)
    service = _make_chat_service(sqlite_session_factory, nlp_analysis=_L3_KEYWORD_ANALYSIS)

    events = await _collect_events(service, session_id, "我不想活了")

    types = [type(e) for e in events]
    assert types[0] is MetaEvent
    assert types[1] is AlertEvent
    assert TokenEvent in types
    assert types[-1] is CompleteEvent


async def test_chat_service_alert_precedes_all_tokens(sqlite_session_factory, seed_session):
    """Alert event must appear before every token event (crisis-card-first guarantee)."""
    session_id, _ = seed_session(message_count=2)
    service = _make_chat_service(sqlite_session_factory, nlp_analysis=_L3_KEYWORD_ANALYSIS)

    events = await _collect_events(service, session_id, "我不想活了")

    alert_index = next(i for i, e in enumerate(events) if isinstance(e, AlertEvent))
    first_token_index = next(i for i, e in enumerate(events) if isinstance(e, TokenEvent))
    assert alert_index < first_token_index


async def test_chat_service_high_risk_meta_risk_level(sqlite_session_factory, seed_session):
    session_id, _ = seed_session(message_count=2)
    service = _make_chat_service(sqlite_session_factory, nlp_analysis=_L3_KEYWORD_ANALYSIS)

    events = await _collect_events(service, session_id, "我不想活了")
    meta = events[0]

    assert isinstance(meta, MetaEvent)
    assert meta.data.risk_level == "L3"


async def test_chat_service_crisis_tokens_on_high_risk(sqlite_session_factory, seed_session):
    """High-risk path uses FakeLLMProvider crisis tokens because the system prompt
    contains the 'high-risk' marker inserted by PromptService."""
    session_id, _ = seed_session(message_count=2)
    service = _make_chat_service(sqlite_session_factory, nlp_analysis=_L3_KEYWORD_ANALYSIS)

    events = await _collect_events(service, session_id, "我不想活了")
    token_texts = [e.data.text for e in events if isinstance(e, TokenEvent)]

    assert token_texts == ["我听到你了，", "这一刻一定很难熬。", "你现在安全吗？"]


# ---------------------------------------------------------------------------
# ChatService — session validation
# ---------------------------------------------------------------------------


async def test_chat_service_raises_for_missing_session(sqlite_session_factory):
    service = _make_chat_service(sqlite_session_factory)

    with pytest.raises(ValueError, match="does not exist"):
        async for _ in service.stream_chat("nonexistent-id", "hello"):
            pass


# ---------------------------------------------------------------------------
# ChatService — DB persistence
# ---------------------------------------------------------------------------


async def test_chat_service_persists_messages(sqlite_session_factory, seed_session):
    session_id, _ = seed_session(message_count=2)
    service = _make_chat_service(sqlite_session_factory, nlp_analysis=_LOW_RISK_ANALYSIS)

    await _collect_events(service, session_id, "我今天有点累")

    with sqlite_session_factory() as session:
        msgs = session.scalars(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.sequence_no)
        ).all()

    # 2 pre-seeded + 1 user + 1 assistant
    assert len(msgs) == 4
    assert msgs[2].role == ChatRole.USER
    assert msgs[2].content == "我今天有点累"
    assert msgs[3].role == ChatRole.ASSISTANT


async def test_chat_service_persists_analysis(sqlite_session_factory, seed_session):
    session_id, _ = seed_session(message_count=2)
    service = _make_chat_service(sqlite_session_factory, nlp_analysis=_LOW_RISK_ANALYSIS)

    await _collect_events(service, session_id, "我今天有点累")

    with sqlite_session_factory() as session:
        msgs = session.scalars(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id, ChatMessage.role == ChatRole.USER)
            .order_by(ChatMessage.sequence_no)
        ).all()
        user_msg = msgs[-1]
        analysis_row = session.scalar(
            select(MessageAnalysis).where(MessageAnalysis.message_id == user_msg.id)
        )

    assert analysis_row is not None
    assert analysis_row.emotion_label == "anxiety"
    assert analysis_row.risk_score == pytest.approx(0.10, abs=0.01)


async def test_chat_service_updates_latest_risk_level(sqlite_session_factory, seed_session):
    session_id, _ = seed_session(message_count=2, latest_risk_level=RiskLevel.L0)
    service = _make_chat_service(sqlite_session_factory, nlp_analysis=_L2_SCORE_ANALYSIS)

    await _collect_events(service, session_id, "我感觉撑不下去了")

    with sqlite_session_factory() as session:
        chat_session = session.get(ChatSession, session_id)

    assert chat_session.latest_risk_level == RiskLevel.L2


async def test_chat_service_complete_event_has_message_id(sqlite_session_factory, seed_session):
    session_id, _ = seed_session(message_count=2)
    service = _make_chat_service(sqlite_session_factory, nlp_analysis=_LOW_RISK_ANALYSIS)

    events = await _collect_events(service, session_id, "你好")
    complete = events[-1]

    assert isinstance(complete, CompleteEvent)
    assert complete.data.message_id  # non-empty string
    assert complete.data.latency_ms >= 0
