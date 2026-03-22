import logging

from sqlalchemy import select

from backend.app.dependencies.services import build_container
from backend.app.models import ChatMessage, ConversationSummary
from backend.app.models.enums import ChatRole, SafetyMode
from backend.app.schemas.analysis import AnalysisResult, RiskAssessment
from backend.app.services.context_service import ContextService
from backend.app.services.prompt_service import PromptService


async def test_context_service_uses_summary_plus_recent_messages(sqlite_session_factory, seed_session) -> None:
    session_id, message_ids = seed_session(message_count=10, with_summary=True, covered_until_index=4)
    service = ContextService(session_factory=sqlite_session_factory)

    window = await service.build_window(session_id, "current user input")

    assert window.conversation_summary == "existing summary"
    assert window.summary_version == 2
    assert window.covered_until_message_id == message_ids[3]
    assert window.latest_risk_level == "L1"
    assert [item["message_id"] for item in window.recent_messages] == message_ids[4:]


async def test_context_service_falls_back_to_latest_ten_messages_without_summary(sqlite_session_factory, seed_session) -> None:
    session_id, message_ids = seed_session(message_count=12, with_summary=False)
    service = ContextService(session_factory=sqlite_session_factory)

    window = await service.build_window(session_id, "current user input")

    assert window.conversation_summary is None
    assert window.summary_version is None
    assert len(window.recent_messages) == 10
    assert [item["message_id"] for item in window.recent_messages] == message_ids[-10:]


async def test_context_service_logs_stale_summary_and_falls_back_to_latest_messages(
    sqlite_session_factory, seed_session, caplog
) -> None:
    session_id, message_ids = seed_session(message_count=8, with_summary=True, covered_until_index=4)
    service = ContextService(session_factory=sqlite_session_factory)

    with sqlite_session_factory.begin() as session:
        summary = session.scalar(select(ConversationSummary).where(ConversationSummary.session_id == session_id))
        assert summary is not None
        summary.covered_until_message_id = "missing-message-id"
        session.add(summary)

    with caplog.at_level(logging.WARNING):
        window = await service.build_window(session_id, "current user input")

    assert [item["message_id"] for item in window.recent_messages] == message_ids[-6:]
    assert "stale" in caplog.text


async def test_refresh_summary_if_needed_creates_and_updates_summary(sqlite_session_factory, seed_session) -> None:
    session_id, message_ids = seed_session(message_count=10, with_summary=False)
    service = ContextService(session_factory=sqlite_session_factory)

    await service.refresh_summary_if_needed(session_id)

    with sqlite_session_factory() as session:
        summary = session.scalar(select(ConversationSummary).where(ConversationSummary.session_id == session_id))
        assert summary is not None
        assert summary.summary_version == 1
        assert summary.covered_until_message_id == message_ids[-1]
        assert "Conversation now has 10 messages." in summary.summary_text

    await service.refresh_summary_if_needed(session_id)

    with sqlite_session_factory() as session:
        summary = session.scalar(select(ConversationSummary).where(ConversationSummary.session_id == session_id))
        assert summary is not None
        assert summary.summary_version == 1

        for offset in range(4):
            sequence_no = 11 + offset
            role = ChatRole.USER if sequence_no % 2 else ChatRole.ASSISTANT
            session.add(
                ChatMessage(
                    session_id=session_id,
                    sequence_no=sequence_no,
                    role=role,
                    content=f"message-{sequence_no}-{role.value}",
                    safety_mode=SafetyMode.STANDARD,
                )
            )
        session.commit()

    await service.refresh_summary_if_needed(session_id)

    with sqlite_session_factory() as session:
        summary = session.scalar(select(ConversationSummary).where(ConversationSummary.session_id == session_id))
        assert summary is not None
        assert summary.summary_version == 2
        assert summary.covered_until_message_id is not None
        assert "Conversation now has 14 messages." in summary.summary_text


async def test_prompt_service_builds_complete_bundle_and_container_exposes_services(
    sqlite_session_factory, seed_session
) -> None:
    session_id, message_ids = seed_session(message_count=10, with_summary=True, covered_until_index=4)
    context_service = ContextService(session_factory=sqlite_session_factory)
    prompt_service = PromptService(context_service=context_service)
    analysis = AnalysisResult(
        emotion_label="anxiety",
        emotion_scores={"anxiety": 0.84},
        intent_label="venting",
        intent_scores={"venting": 0.77},
        intensity_score=0.7,
        risk_aux_score=0.3,
        keyword_hits=["insomnia"],
    )
    risk = RiskAssessment(
        risk_score=0.86,
        risk_level="L3",
        reasons=["explicit self-harm ideation"],
        suggested_resource_level="urgent",
    )

    bundle = await prompt_service.build(session_id, "I am overwhelmed right now", analysis, risk)
    container = build_container()

    assert bundle.conversation_summary is not None
    assert bundle.covered_until_message_id == message_ids[3]
    assert bundle.user_message == "I am overwhelmed right now"
    assert bundle.analysis == analysis
    assert bundle.risk == risk
    assert "high-risk support path" in bundle.system_prompt
    assert "Do not claim to diagnose" in bundle.system_prompt
    assert container.context_service is not None
    assert container.prompt_service is not None
