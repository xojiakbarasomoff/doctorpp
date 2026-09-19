"""Keep undelivered replies, and the language the patient asked for

Two columns and one more on users, all additive.

messages.delivery_status / delivery_error: a reply that could not be
delivered was not stored at all, so a clinic opening one of those chats saw
the patient's questions and no answers -- and had no way to tell a bot that
said nothing from a bot whose answer never left the building. Existing rows
are backfilled as "sent", which is what they were: nothing was written
unless it went out.

users.preferred_language: a patient who asks to be answered in Russian and
then sends "ok" was answered in Uzbek again, because the language was read
from each message rather than remembered.

Revision ID: c8f3a25d1b47
Revises: b4d21f7c9a15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c8f3a25d1b47"
down_revision: str | Sequence[str] | None = "b4d21f7c9a15"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "messages",
        sa.Column(
            "delivery_status",
            sa.String(length=20),
            nullable=False,
            server_default="sent",
        ),
    )
    op.add_column(
        "messages", sa.Column("delivery_error", sa.String(length=255), nullable=True)
    )
    op.add_column(
        "users", sa.Column("preferred_language", sa.String(length=16), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("users", "preferred_language")
    op.drop_column("messages", "delivery_error")
    op.drop_column("messages", "delivery_status")
