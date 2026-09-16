"""keep the photos patients send

A patient who sends a picture -- an analysis result, a scan -- is waiting for
a person to look at it. Until now the webhook dropped attachments on the
floor. This keeps them, image bytes included, because Instagram's signed
attachment links expire long before a doctor gets round to next week's.

Revision ID: e3a9b6c1d472
Revises: d5b7e04a1c63
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "e3a9b6c1d472"
down_revision: str | Sequence[str] | None = "d5b7e04a1c63"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "patient_media",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("external_id", sa.String(length=255), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("content", sa.LargeBinary(), nullable=True),
        sa.Column("content_type", sa.String(length=100), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=20), server_default=sa.text("'new'"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name=op.f("fk_patient_media_tenant_id_tenants")
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_patient_media_user_id_users")
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name=op.f("fk_patient_media_conversation_id_conversations"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_patient_media")),
        sa.UniqueConstraint("external_id", name=op.f("uq_patient_media_external_id")),
    )
    op.create_index(op.f("ix_patient_media_tenant_id"), "patient_media", ["tenant_id"])
    op.create_index(op.f("ix_patient_media_user_id"), "patient_media", ["user_id"])
    op.create_index(op.f("ix_patient_media_created_at"), "patient_media", ["created_at"])


def downgrade() -> None:
    op.drop_index(op.f("ix_patient_media_created_at"), table_name="patient_media")
    op.drop_index(op.f("ix_patient_media_user_id"), table_name="patient_media")
    op.drop_index(op.f("ix_patient_media_tenant_id"), table_name="patient_media")
    op.drop_table("patient_media")
