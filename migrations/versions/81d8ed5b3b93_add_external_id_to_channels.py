"""add external_id to channels

Revision ID: 81d8ed5b3b93
Revises: 9f512d8d9aa0
Create Date: 2026-08-14 14:17:09.748937

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "81d8ed5b3b93"
down_revision: str | Sequence[str] | None = "9f512d8d9aa0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("channels", sa.Column("external_id", sa.String(length=255), nullable=False))
    op.create_unique_constraint(
        op.f("uq_channels_type_external_id"), "channels", ["type", "external_id"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(op.f("uq_channels_type_external_id"), "channels", type_="unique")
    op.drop_column("channels", "external_id")
