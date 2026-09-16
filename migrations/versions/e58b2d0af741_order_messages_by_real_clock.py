"""order messages by real clock

messages.created_at defaulted to now(), which in Postgres is
transaction_timestamp(): every row inserted inside one transaction receives
the same value. Ordering a transcript by it then falls through to the
tiebreak — a random UUID primary key — and the conversation comes back in an
arbitrary order.

That was harmless while nothing read the transcript back. It stops being
harmless now that the reply pipeline generates each answer from the ordered
history (see app.services.conversation.recent_history): a reordered
conversation is one where the clinic appears to have answered a question
before it was asked.

clock_timestamp() reads the actual clock at insert time, so rows written in
one transaction are still ordered by when each was written.

Revision ID: e58b2d0af741
Revises: d47a1c9e6b30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e58b2d0af741"
down_revision: str | Sequence[str] | None = "d47a1c9e6b30"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "messages",
        "created_at",
        server_default=sa.text("clock_timestamp()"),
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "messages",
        "created_at",
        server_default=sa.text("now()"),
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=False,
    )
