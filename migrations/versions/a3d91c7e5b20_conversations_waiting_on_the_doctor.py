"""conversations waiting on the doctor

A conversation the assistant handed to the doctor, or one a patient wrote
into while staff had it, is pinned to the top of the dashboard's list until
a person answers. This is when that started, or NULL when nobody is owed a
reply.

Revision ID: a3d91c7e5b20
Revises: f7a2d94e1c05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a3d91c7e5b20"
down_revision: str | Sequence[str] | None = "f7a2d94e1c05"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "conversations",
        sa.Column("needs_doctor_since", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("conversations", "needs_doctor_since")
