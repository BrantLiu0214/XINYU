"""add student profile fields to visitor_profiles and college/display_name to counselor_accounts

Revision ID: 20260401_03
Revises: 20260326_02
Create Date: 2026-04-01 10:00:00
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260401_03"
down_revision: Union[str, Sequence[str], None] = "20260326_02"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # visitor_profiles: add real_name, college, student_id, is_guest
    op.add_column("visitor_profiles", sa.Column("real_name", sa.String(length=100), nullable=True))
    op.add_column("visitor_profiles", sa.Column("college", sa.String(length=100), nullable=True))
    op.add_column("visitor_profiles", sa.Column("student_id", sa.String(length=50), nullable=True))
    op.add_column(
        "visitor_profiles",
        sa.Column("is_guest", sa.Boolean(), nullable=False, server_default="false"),
    )

    # counselor_accounts: add display_name, college
    op.add_column("counselor_accounts", sa.Column("display_name", sa.String(length=100), nullable=True))
    op.add_column("counselor_accounts", sa.Column("college", sa.String(length=100), nullable=True))


def downgrade() -> None:
    op.drop_column("counselor_accounts", "college")
    op.drop_column("counselor_accounts", "display_name")
    op.drop_column("visitor_profiles", "is_guest")
    op.drop_column("visitor_profiles", "student_id")
    op.drop_column("visitor_profiles", "college")
    op.drop_column("visitor_profiles", "real_name")
