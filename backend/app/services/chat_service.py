from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.models.alert_event import AlertEvent as AlertEventORM
from backend.app.models.chat_message import ChatMessage
from backend.app.models.chat_session import ChatSession
from backend.app.models.enums import AlertStatus, ChatRole, RiskLevel, SafetyMode
from backend.app.models.message_analysis import MessageAnalysis
from backend.app.schemas.stream import (
    AlertEvent,
    AlertEventPayload,
    CompleteEvent,
    CompleteEventPayload,
    MetaEvent,
    MetaEventPayload,
    StreamEvent,
    TokenEvent,
    TokenEventPayload,
)
from backend.app.services.llm_service import LLMProvider
from backend.app.services.nlp_service import NLPService
from backend.app.services.prompt_service import PromptService
from backend.app.services.resource_service import ResourceService
from backend.app.services.risk_service import RiskService

_HIGH_RISK_LEVELS: frozenset[str] = frozenset({"L2", "L3"})
_RECENT_RISK_HISTORY_LIMIT = 6


class ChatService:
    """Orchestrates the full chat turn pipeline and yields SSE stream events.

    Event order for a normal turn:  meta → token… → complete
    Event order for a high-risk turn: meta → alert → token… → complete

    The alert event is emitted before any token events so the frontend can
    display crisis resources immediately when a high-risk response begins.
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        nlp_service: NLPService,
        risk_service: RiskService,
        resource_service: ResourceService,
        prompt_service: PromptService,
        llm_provider: LLMProvider,
    ) -> None:
        self._session_factory = session_factory
        self._nlp = nlp_service
        self._risk = risk_service
        self._resources = resource_service
        self._prompt = prompt_service
        self._llm = llm_provider

    async def stream_chat(
        self,
        session_id: str,
        user_message: str,
    ) -> AsyncGenerator[StreamEvent, None]:
        started_ms = time.monotonic_ns() // 1_000_000

        # 1. Verify session exists; raise immediately if not.
        self._require_session(session_id)

        # 2. NLP analysis.
        analysis = await self._nlp.analyze(user_message)

        # 3. Load recent risk history (L2/L3 turns from alert_events).
        recent_levels = self._load_recent_risk_levels(session_id)

        # 4. Evaluate risk.
        risk = self._risk.evaluate(analysis, recent_levels)

        # 5. Update session risk level before building prompt so ContextService
        #    sees the current risk state on this and subsequent turns.
        self._update_session_risk_level(session_id, risk.risk_level)

        # 6. Build prompt.  The user message has not been persisted yet, so
        #    ContextService will not include it in recent_messages — it arrives
        #    separately as PromptBundle.user_message.  This prevents duplication
        #    in the message list sent to the LLM.
        prompt = await self._prompt.build(session_id, user_message, analysis, risk)

        # Pre-generate the user message ID so it can be referenced by the alert
        # record before the user message row is committed.
        user_msg_id = str(uuid4())

        # 7. Emit meta event.
        yield MetaEvent(
            data=MetaEventPayload(
                emotion=analysis.emotion_label,
                intent=analysis.intent_label,
                intensity=analysis.intensity_score,
                risk_level=risk.risk_level,
            )
        )

        # 8. For high-risk paths, emit alert event before any token events.
        is_high_risk = risk.risk_level in _HIGH_RISK_LEVELS
        if is_high_risk:
            resources = self._resources.get_for_risk_level(risk.risk_level)
            yield AlertEvent(
                data=AlertEventPayload(
                    risk_level=risk.risk_level,
                    resources=resources,
                )
            )

        # 9. Stream tokens.
        reply_parts: list[str] = []
        async for token in self._llm.stream_reply(prompt):
            reply_parts.append(token)
            yield TokenEvent(data=TokenEventPayload(text=token))

        # 10. Persist all DB records after streaming completes.
        full_reply = "".join(reply_parts)
        latency_ms = (time.monotonic_ns() // 1_000_000) - started_ms
        safety = SafetyMode.CRISIS if is_high_risk else SafetyMode.STANDARD

        assistant_msg_id = self._persist_turn(
            session_id=session_id,
            user_msg_id=user_msg_id,
            user_message=user_message,
            full_reply=full_reply,
            safety=safety,
            analysis=analysis,
            risk=risk,
            latency_ms=latency_ms,
            persist_alert=is_high_risk,
        )

        # 11. Emit complete event.
        yield CompleteEvent(
            data=CompleteEventPayload(
                message_id=assistant_msg_id,
                latency_ms=latency_ms,
            )
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_session(self, session_id: str) -> None:
        with self._session_factory() as session:
            if session.get(ChatSession, session_id) is None:
                raise ValueError(f"Session '{session_id}' does not exist.")

    def _load_recent_risk_levels(self, session_id: str) -> list[str]:
        """Return risk-level strings from the most recent alert events.

        Only L2/L3 turns produce alert records, so only high-risk levels appear
        here.  RiskService._is_escalating treats two or more such entries as an
        escalating pattern, which is exactly what alert records represent.
        """
        with self._session_factory() as session:
            rows = session.scalars(
                select(AlertEventORM.risk_level)
                .where(AlertEventORM.session_id == session_id)
                .order_by(AlertEventORM.created_at.desc())
                .limit(_RECENT_RISK_HISTORY_LIMIT)
            ).all()
        return [str(r) for r in rows]

    def _update_session_risk_level(self, session_id: str, risk_level: str) -> None:
        with self._session_factory() as session:
            chat_session = session.get(ChatSession, session_id)
            if chat_session is not None:
                chat_session.latest_risk_level = RiskLevel(risk_level)
                session.commit()

    def _persist_turn(
        self,
        *,
        session_id: str,
        user_msg_id: str,
        user_message: str,
        full_reply: str,
        safety: SafetyMode,
        analysis,
        risk,
        latency_ms: int,
        persist_alert: bool,
    ) -> str:
        """Persist user message, assistant message, analysis, and (optionally)
        alert event in a single session.  Returns the assistant message ID."""
        assistant_msg_id = str(uuid4())

        with self._session_factory() as session:
            user_seq = self._next_sequence_no(session, session_id)
            session.add(
                ChatMessage(
                    id=user_msg_id,
                    session_id=session_id,
                    sequence_no=user_seq,
                    role=ChatRole.USER,
                    content=user_message,
                    safety_mode=safety,
                )
            )
            # Flush to assign sequence_no before computing the next one.
            session.flush()

            assistant_seq = self._next_sequence_no(session, session_id)
            session.add(
                ChatMessage(
                    id=assistant_msg_id,
                    session_id=session_id,
                    sequence_no=assistant_seq,
                    role=ChatRole.ASSISTANT,
                    content=full_reply,
                    safety_mode=safety,
                    latency_ms=latency_ms,
                )
            )

            session.add(
                MessageAnalysis(
                    message_id=user_msg_id,
                    emotion_label=analysis.emotion_label,
                    emotion_scores=analysis.emotion_scores,
                    intent_label=analysis.intent_label,
                    intent_scores=analysis.intent_scores,
                    intensity_score=analysis.intensity_score,
                    risk_score=risk.risk_score,
                    keyword_hits=analysis.keyword_hits,
                )
            )

            if persist_alert:
                session.add(
                    AlertEventORM(
                        session_id=session_id,
                        message_id=user_msg_id,
                        risk_level=RiskLevel(risk.risk_level),
                        reasons=risk.reasons,
                        status=AlertStatus.OPEN,
                    )
                )

            session.commit()

        return assistant_msg_id

    def _next_sequence_no(self, session: Session, session_id: str) -> int:
        result = session.execute(
            select(func.max(ChatMessage.sequence_no)).where(
                ChatMessage.session_id == session_id
            )
        ).scalar()
        return (result or 0) + 1
