from __future__ import annotations

import logging
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from backend.app.models import ChatMessage, ChatSession, ConversationSummary
from backend.app.models.enums import ChatRole
from backend.app.schemas.prompt import ContextWindow

logger = logging.getLogger(__name__)

SUMMARY_TRIGGER_THRESHOLD = 10
SUMMARY_REFRESH_INTERVAL = 4
SUMMARY_WINDOW_LIMIT = 6
RECENT_WINDOW_FALLBACK_LIMIT = 10
SUMMARY_SNIPPET_LIMIT = 80


class ContextService:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    async def build_window(self, session_id: str, current_message: str) -> ContextWindow:
        # current_message is part of the ContextService Protocol contract and is reserved
        # for future relevance-based message selection (e.g. ranking recent messages by
        # semantic similarity to the current turn). Not used in this implementation.
        del current_message

        with self._session_factory() as session:
            chat_session = self._get_chat_session(session, session_id)
            summary = self._get_summary(session, session_id)
            messages = self._list_messages(session, session_id)

        if summary is None:
            return ContextWindow(
                recent_messages=self._serialize_messages(messages[-RECENT_WINDOW_FALLBACK_LIMIT:]),
                latest_risk_level=chat_session.latest_risk_level.value,
            )

        recent_messages = self._select_recent_messages(messages, summary)
        return ContextWindow(
            conversation_summary=summary.summary_text,
            summary_version=summary.summary_version,
            covered_until_message_id=summary.covered_until_message_id,
            recent_messages=self._serialize_messages(recent_messages),
            latest_risk_level=chat_session.latest_risk_level.value,
        )

    async def refresh_summary_if_needed(self, session_id: str) -> None:
        with self._session_factory() as session:
            chat_session = self._get_chat_session(session, session_id)
            summary = self._get_summary(session, session_id)
            messages = self._list_messages(session, session_id)

            if len(messages) < SUMMARY_TRIGGER_THRESHOLD:
                return

            if summary is None:
                self._create_summary(session, chat_session, messages)
                session.commit()
                return

            covered_index = self._find_message_index(messages, summary.covered_until_message_id)
            if covered_index is None and summary.covered_until_message_id is not None:
                logger.warning(
                    "conversation summary anchor missing for session %s; forcing rebuild",
                    session_id,
                )
                self._update_summary(session, summary, chat_session, messages)
                session.commit()
                return

            unsummarized_count = len(messages) if covered_index is None else len(messages) - covered_index - 1
            if unsummarized_count == 0:
                return
            if covered_index is not None and unsummarized_count % SUMMARY_REFRESH_INTERVAL != 0:
                return
            # Note: when covered_index is None here, covered_until_message_id was also None
            # (the None-with-non-None-id branch already returned above), meaning the summary
            # was created without an anchor. In that case we always attempt a refresh so the
            # anchor gets populated. _update_summary will skip silently if no progress was made.

            self._update_summary(session, summary, chat_session, messages)
            session.commit()

    def _get_chat_session(self, session: Session, session_id: str) -> ChatSession:
        chat_session = session.get(ChatSession, session_id)
        if chat_session is None:
            raise ValueError(f"Session '{session_id}' does not exist.")
        return chat_session

    def _get_summary(self, session: Session, session_id: str) -> ConversationSummary | None:
        statement = select(ConversationSummary).where(ConversationSummary.session_id == session_id)
        return session.scalar(statement)

    def _list_messages(self, session: Session, session_id: str) -> list[ChatMessage]:
        statement = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.sequence_no.asc())
        )
        return list(session.scalars(statement))

    def _select_recent_messages(
        self,
        messages: Sequence[ChatMessage],
        summary: ConversationSummary,
    ) -> list[ChatMessage]:
        covered_index = self._find_message_index(messages, summary.covered_until_message_id)
        if summary.covered_until_message_id is None or covered_index is None:
            logger.warning(
                "conversation summary is stale for session %s; falling back to latest messages",
                summary.session_id,
            )
            return list(messages[-SUMMARY_WINDOW_LIMIT:])

        return list(messages[covered_index + 1 :][-SUMMARY_WINDOW_LIMIT:])

    def _create_summary(
        self,
        session: Session,
        chat_session: ChatSession,
        messages: Sequence[ChatMessage],
    ) -> None:
        summary_text, open_topics, carry_over_advice = self._build_summary_parts(
            chat_session.latest_risk_level.value,
            messages,
        )
        session.add(
            ConversationSummary(
                session_id=chat_session.id,
                summary_text=summary_text,
                summary_version=1,
                covered_until_message_id=messages[-1].id,
                last_risk_level=chat_session.latest_risk_level,
                open_topics=open_topics,
                carry_over_advice=carry_over_advice,
            )
        )

    def _update_summary(
        self,
        session: Session,
        summary: ConversationSummary,
        chat_session: ChatSession,
        messages: Sequence[ChatMessage],
    ) -> None:
        if summary.covered_until_message_id == messages[-1].id:
            return

        summary_text, open_topics, carry_over_advice = self._build_summary_parts(
            chat_session.latest_risk_level.value,
            messages,
        )
        summary.summary_text = summary_text
        summary.summary_version += 1
        summary.covered_until_message_id = messages[-1].id
        summary.last_risk_level = chat_session.latest_risk_level
        summary.open_topics = open_topics
        summary.carry_over_advice = carry_over_advice
        session.add(summary)

    def _build_summary_parts(
        self,
        latest_risk_level: str,
        messages: Sequence[ChatMessage],
    ) -> tuple[str, list[str], list[str]]:
        user_topics = self._extract_snippets(messages, ChatRole.USER, limit=3)
        assistant_support = self._extract_snippets(messages, ChatRole.ASSISTANT, limit=2)

        topic_text = "; ".join(user_topics) if user_topics else "no clear topic yet"
        advice_text = "; ".join(assistant_support) if assistant_support else "no carry-over advice yet"
        summary_text = (
            f"Conversation now has {len(messages)} messages. "
            f"Recent user concerns: {topic_text}. "
            f"Recent assistant support: {advice_text}. "
            f"Recorded risk level: {latest_risk_level}."
        )
        return summary_text, user_topics, assistant_support

    def _extract_snippets(
        self,
        messages: Sequence[ChatMessage],
        role: ChatRole,
        *,
        limit: int,
    ) -> list[str]:
        snippets: list[str] = []
        for message in messages:
            if message.role != role:
                continue
            snippet = self._truncate_text(message.content)
            if not snippet:
                continue
            snippets.append(snippet)

        return snippets[-limit:]

    def _serialize_messages(self, messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
        return [
            {
                "message_id": message.id,
                "role": message.role.value,
                "content": message.content,
            }
            for message in messages
        ]

    def _find_message_index(self, messages: Sequence[ChatMessage], message_id: str | None) -> int | None:
        if message_id is None:
            return None

        for index, message in enumerate(messages):
            if message.id == message_id:
                return index
        return None

    def _truncate_text(self, text: str) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= SUMMARY_SNIPPET_LIMIT:
            return normalized
        return f"{normalized[: SUMMARY_SNIPPET_LIMIT - 3]}..."
