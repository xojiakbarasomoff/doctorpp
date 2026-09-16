"""add appointment booking fields and slot uniqueness

Revision ID: a3f9c2d15e88
Revises: cb58f7d9c82c
Create Date: 2026-08-18 05:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a3f9c2d15e88"
down_revision: str | Sequence[str] | None = "cb58f7d9c82c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("appointments", sa.Column("conversation_id", sa.UUID(), nullable=True))
    op.add_column("appointments", sa.Column("patient_name", sa.String(length=255), nullable=True))
    op.add_column(
        "appointments",
        sa.Column(
            "source", sa.String(length=20), server_default=sa.text("'operator'"), nullable=False
        ),
    )
    op.add_column(
        "appointments",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.add_column(
        "appointments",
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_foreign_key(
        op.f("fk_appointments_conversation_id_conversations"),
        "appointments",
        "conversations",
        ["conversation_id"],
        ["id"],
    )
    op.create_index(
        op.f("ix_appointments_conversation_id"), "appointments", ["conversation_id"], unique=False
    )
    # The double-booking guard: at most one *active* (status='scheduled')
    # appointment per exact slot per tenant. Partial, not a plain unique
    # constraint, so cancelling (or a future no-show) frees the slot for
    # rebooking instead of blocking it forever.
    op.create_index(
        "uq_appointments_tenant_id_scheduled_at",
        "appointments",
        ["tenant_id", "scheduled_at"],
        unique=True,
        postgresql_where=sa.text("status = 'scheduled'"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("uq_appointments_tenant_id_scheduled_at", table_name="appointments")
    op.drop_index(op.f("ix_appointments_conversation_id"), table_name="appointments")
    op.drop_constraint(
        op.f("fk_appointments_conversation_id_conversations"), "appointments", type_="foreignkey"
    )
    op.drop_column("appointments", "updated_at")
    op.drop_column("appointments", "created_at")
    op.drop_column("appointments", "source")
    op.drop_column("appointments", "patient_name")
    op.drop_column("appointments", "conversation_id")
