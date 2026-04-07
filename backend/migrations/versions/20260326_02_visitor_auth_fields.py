"""add username and password_hash to visitor_profiles

Revision ID: 20260326_02
Revises: 20260309_01
Create Date: 2026-03-26 12:00:00
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260326_02"
down_revision: Union[str, Sequence[str], None] = "20260309_01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "visitor_profiles",
        sa.Column("username", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "visitor_profiles",
        sa.Column("password_hash", sa.String(length=255), nullable=True),
    )
    op.create_unique_constraint(
        op.f("uq_visitor_profiles_username"),
        "visitor_profiles",
        ["username"],
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("uq_visitor_profiles_username"), "visitor_profiles", type_="unique"
    )
    op.drop_column("visitor_profiles", "password_hash")
    op.drop_column("visitor_profiles", "username")
