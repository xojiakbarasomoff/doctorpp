"""suggestions from the nightly review

Fixes the review proposes from the day's conversations -- a rule, or a
question and answer -- waiting for somebody at the clinic to accept or
reject them.

Revision ID: b7e2c4f19a30
Revises: a3d91c7e5b20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "b7e2c4f19a30"
down_revision: str | Sequence[str] | None = "a3d91c7e5b20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "suggestions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenants.id"), nullable=False
        ),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("problem", sa.Text(), nullable=False),
        sa.Column("rule_text", sa.Text(), nullable=True),
        sa.Column("question", sa.Text(), nullable=True),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column(
            "evidence",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "decided_by",
            postgresql.UUID(as_uuid=True),
            # Staff are deleted outright; the decision outlives who made it.
            sa.ForeignKey("operators.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index("ix_suggestions_tenant_id", "suggestions", ["tenant_id"])
    op.create_index("ix_suggestions_tenant_status", "suggestions", ["tenant_id", "status"])


def downgrade() -> None:
    op.drop_index("ix_suggestions_tenant_status", table_name="suggestions")
    op.drop_index("ix_suggestions_tenant_id", table_name="suggestions")
    op.drop_table("suggestions")
