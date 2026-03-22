"""initial core tables

Revision ID: 20260309_01
Revises: None
Create Date: 2026-03-09 10:55:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "20260309_01"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    risk_level_enum = sa.Enum("L0", "L1", "L2", "L3", name="risklevel", native_enum=False)
    chat_role_enum = sa.Enum("user", "assistant", "system", name="chatrole", native_enum=False)
    safety_mode_enum = sa.Enum("standard", "crisis", name="safetymode", native_enum=False)
    alert_status_enum = sa.Enum("open", "acknowledged", "resolved", name="alertstatus", native_enum=False)

    op.create_table(
        "visitor_profiles",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("display_name", sa.String(length=100), nullable=True),
        sa.Column("consent_accepted", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_visitor_profiles")),
    )

    op.create_table(
        "counselor_accounts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("username", sa.String(length=100), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_counselor_accounts")),
        sa.UniqueConstraint("username", name=op.f("uq_counselor_accounts_username")),
    )

    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("visitor_id", sa.String(length=36), nullable=False),
        sa.Column("latest_risk_level", risk_level_enum, server_default="L0", nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["visitor_id"], ["visitor_profiles.id"], name=op.f("fk_chat_sessions_visitor_id_visitor_profiles")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chat_sessions")),
    )
    op.create_index(op.f("ix_chat_sessions_visitor_id"), "chat_sessions", ["visitor_id"], unique=False)

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("sequence_no", sa.Integer(), nullable=False),
        sa.Column("role", chat_role_enum, nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("safety_mode", safety_mode_enum, server_default="standard", nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["chat_sessions.id"], name=op.f("fk_chat_messages_session_id_chat_sessions")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chat_messages")),
        sa.UniqueConstraint("session_id", "sequence_no", name="uq_chat_messages_session_id_sequence_no"),
    )
    op.create_index(op.f("ix_chat_messages_session_id"), "chat_messages", ["session_id"], unique=False)

    op.create_table(
        "message_analyses",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("message_id", sa.String(length=36), nullable=False),
        sa.Column("emotion_label", sa.String(length=50), nullable=False),
        sa.Column("emotion_scores", sa.JSON(), nullable=True),
        sa.Column("intent_label", sa.String(length=50), nullable=False),
        sa.Column("intent_scores", sa.JSON(), nullable=True),
        sa.Column("intensity_score", sa.Float(), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("keyword_hits", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["message_id"], ["chat_messages.id"], name=op.f("fk_message_analyses_message_id_chat_messages")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_message_analyses")),
        sa.UniqueConstraint("message_id", name=op.f("uq_message_analyses_message_id")),
    )
    op.create_index(op.f("ix_message_analyses_message_id"), "message_analyses", ["message_id"], unique=True)

    op.create_table(
        "alert_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("message_id", sa.String(length=36), nullable=True),
        sa.Column("risk_level", risk_level_enum, nullable=False),
        sa.Column("reasons", sa.JSON(), nullable=False),
        sa.Column("status", alert_status_enum, server_default="open", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["message_id"], ["chat_messages.id"], name=op.f("fk_alert_events_message_id_chat_messages")),
        sa.ForeignKeyConstraint(["session_id"], ["chat_sessions.id"], name=op.f("fk_alert_events_session_id_chat_sessions")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_alert_events")),
    )
    op.create_index(op.f("ix_alert_events_message_id"), "alert_events", ["message_id"], unique=False)
    op.create_index(op.f("ix_alert_events_session_id"), "alert_events", ["session_id"], unique=False)

    op.create_table(
        "conversation_summaries",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("summary_text", sa.Text(), nullable=False),
        sa.Column("summary_version", sa.Integer(), server_default="1", nullable=False),
        sa.Column("covered_until_message_id", sa.String(length=36), nullable=True),
        sa.Column("last_risk_level", risk_level_enum, server_default="L0", nullable=False),
        sa.Column("open_topics", sa.JSON(), nullable=True),
        sa.Column("carry_over_advice", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.ForeignKeyConstraint(["covered_until_message_id"], ["chat_messages.id"], name=op.f("fk_conversation_summaries_covered_until_message_id_chat_messages")),
        sa.ForeignKeyConstraint(["session_id"], ["chat_sessions.id"], name=op.f("fk_conversation_summaries_session_id_chat_sessions")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_conversation_summaries")),
        sa.UniqueConstraint("session_id", name=op.f("uq_conversation_summaries_session_id")),
    )
    op.create_index(op.f("ix_conversation_summaries_session_id"), "conversation_summaries", ["session_id"], unique=True)

    op.create_table(
        "resource_catalog",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("title", sa.String(length=150), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("phone", sa.String(length=50), nullable=True),
        sa.Column("link_url", sa.String(length=255), nullable=True),
        sa.Column("risk_level", risk_level_enum, nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_resource_catalog")),
    )


def downgrade() -> None:
    op.drop_table("resource_catalog")
    op.drop_index(op.f("ix_conversation_summaries_session_id"), table_name="conversation_summaries")
    op.drop_table("conversation_summaries")
    op.drop_index(op.f("ix_alert_events_session_id"), table_name="alert_events")
    op.drop_index(op.f("ix_alert_events_message_id"), table_name="alert_events")
    op.drop_table("alert_events")
    op.drop_index(op.f("ix_message_analyses_message_id"), table_name="message_analyses")
    op.drop_table("message_analyses")
    op.drop_index(op.f("ix_chat_messages_session_id"), table_name="chat_messages")
    op.drop_table("chat_messages")
    op.drop_index(op.f("ix_chat_sessions_visitor_id"), table_name="chat_sessions")
    op.drop_table("chat_sessions")
    op.drop_table("counselor_accounts")
    op.drop_table("visitor_profiles")
