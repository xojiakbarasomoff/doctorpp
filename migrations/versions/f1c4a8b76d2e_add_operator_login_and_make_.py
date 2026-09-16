"""add operator login fields and make appointment patient optional

Revision ID: f1c4a8b76d2e
Revises: a3f9c2d15e88
Create Date: 2026-08-19 06:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f1c4a8b76d2e"
down_revision: str | Sequence[str] | None = "a3f9c2d15e88"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # operators.credentials was never used by any code path (only
    # channels.credentials — the IG access token — was) so this repurposes
    # it directly as the bcrypt password hash rather than leaving a
    # vestigial unused column alongside a new one.
    op.alter_column("operators", "credentials", new_column_name="password_hash")
    op.add_column(
        "operators", sa.Column("username", sa.String(length=255), nullable=False, server_default="")
    )
    # server_default only exists to satisfy the NOT NULL constraint while
    # backfilling pre-dashboard rows (dev/test seed data) — no real operator
    # accounts exist yet, so there's nothing meaningful to backfill. Dropped
    # immediately after so it can't mask a missing username on a future
    # insert.
    op.alter_column("operators", "username", server_default=None)
    op.create_unique_constraint("uq_operators_username", "operators", ["username"])

    # Nullable so an operator can book a walk-in/phone patient who has no
    # User row (User requires a channel_id — i.e. a prior IG conversation).
    op.alter_column("appointments", "user_id", nullable=True)


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column("appointments", "user_id", nullable=False)
    op.drop_constraint("uq_operators_username", "operators", type_="unique")
    op.drop_column("operators", "username")
    op.alter_column("operators", "password_hash", new_column_name="credentials")
