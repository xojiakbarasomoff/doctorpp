"""Remember where a conversation got to

One row per conversation holding what it is doing and what it has settled:
the day and time a patient chose, the appointment a cancellation is about,
and the last thing the backend actually did for them. Until now all of that
lived in the transcript, and the transcript the assistant is given is the
last ten turns -- so a patient who chose Thursday on Tuesday was offered
tomorrow on Wednesday, and one whose booking was finished was asked for
their name again.

Additive. Nothing is dropped, nothing is rewritten: conversations that
exist simply have no state row until their next message, which reads as an
idle conversation, which is what they are.

Revision ID: b4d21f7c9a15
Revises: e3a9b6c1d472
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "b4d21f7c9a15"
down_revision: str | Sequence[str] | None = "e3a9b6c1d472"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "conversation_states",
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(length=40), nullable=False, server_default="idle"),
        sa.Column("intent", sa.String(length=40), nullable=False, server_default="none"),
        sa.Column("awaiting_field", sa.String(length=40), nullable=True),
        sa.Column("requested_date", sa.Date(), nullable=True),
        sa.Column("requested_time", sa.Time(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("appointment_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "last_completed_action", sa.String(length=40), nullable=False, server_default="none"
        ),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["appointment_id"], ["appointments.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("conversation_id"),
    )
    # Every read is "this tenant's conversation", the same shape as every
    # other tenant-scoped query in this schema.
    op.create_index(
        "ix_conversation_states_tenant_id", "conversation_states", ["tenant_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_conversation_states_tenant_id", table_name="conversation_states")
    op.drop_table("conversation_states")
