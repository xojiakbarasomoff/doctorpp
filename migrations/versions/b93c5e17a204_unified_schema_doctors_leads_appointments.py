"""unified schema: doctors, leads, and the merged appointment shape

First migration of the Instagram/Telegram merge. It brings the schema up to
the union of what the two bots each needed, so one database can serve both.

What it adds, and where each piece came from:

* doctors, leads - real tables on the Telegram side, but created there by
  raw SQL at application startup and by hand-run scripts, never by a
  migration. They are declared as models here and created properly.
* appointments.doctor -> doctor_name, plus doctor_id, patient_phone, notes
  and the two reminder flags. The rename is to the Telegram name because
  the column now sits beside doctor_id, and "doctor" alone would read as
  the foreign key rather than the text.
* appointments' partial unique index widens from status = 'scheduled' to
  the active set ('scheduled', 'confirmed'). A confirmed booking holds its
  slot exactly as firmly as an unconfirmed one; under the old predicate a
  confirmed appointment looked like a free slot and could be double-booked.
* channels.config - per-channel platform settings (a Telegram webhook
  secret, bot admin ids), which do not belong in tenants.settings because
  one clinic can run two accounts on the same platform.
* channels.created_at - the Telegram side had it, the Instagram side did not.
* users.is_admin - from the Telegram side: a patient who is also clinic
  staff and may use the bot's admin commands.

No data migration is involved: neither deployment holds production rows,
which is what makes the appointments rename and the index change safe to do
directly rather than through a copy-and-backfill.

Revision ID: b93c5e17a204
Revises: e58b2d0af741
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "b93c5e17a204"
down_revision: str | Sequence[str] | None = "e58b2d0af741"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Mirrors app.models.appointment.ACTIVE_STATUSES. Written out rather than
# imported: a migration describes the schema as it was at this revision, and
# must not change meaning later when the application's own constant does.
_ACTIVE_STATUSES = "'confirmed', 'scheduled'"


def upgrade() -> None:
    # -- doctors ---------------------------------------------------------
    op.create_table(
        "doctors",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("specialty", sa.String(length=255), nullable=False),
        sa.Column("phone", sa.String(length=50), nullable=True),
        sa.Column("working_hours", sa.String(length=255), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name=op.f("fk_doctors_tenant_id_tenants")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_doctors")),
    )
    op.create_index(op.f("ix_doctors_tenant_id"), "doctors", ["tenant_id"], unique=False)

    # -- leads -----------------------------------------------------------
    op.create_table(
        "leads",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("patient_name", sa.String(length=255), nullable=True),
        sa.Column("phone", sa.String(length=50), nullable=True),
        sa.Column("topic", sa.String(length=255), nullable=True),
        sa.Column("convenient_time", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=50), server_default=sa.text("'new'"), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name=op.f("fk_leads_conversation_id_conversations"),
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name=op.f("fk_leads_tenant_id_tenants")
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name=op.f("fk_leads_user_id_users")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_leads")),
    )
    op.create_index(op.f("ix_leads_tenant_id"), "leads", ["tenant_id"], unique=False)
    op.create_index(op.f("ix_leads_user_id"), "leads", ["user_id"], unique=False)
    op.create_index(op.f("ix_leads_conversation_id"), "leads", ["conversation_id"], unique=False)

    # -- channels --------------------------------------------------------
    op.add_column(
        "channels",
        sa.Column(
            "config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column(
        "channels",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    # -- users -----------------------------------------------------------
    op.add_column(
        "users",
        sa.Column("is_admin", sa.Boolean(), server_default=sa.text("false"), nullable=False),
    )

    # -- appointments ----------------------------------------------------
    op.alter_column("appointments", "doctor", new_column_name="doctor_name")
    op.add_column(
        "appointments", sa.Column("doctor_id", postgresql.UUID(as_uuid=True), nullable=True)
    )
    op.add_column("appointments", sa.Column("patient_phone", sa.String(length=50), nullable=True))
    op.add_column("appointments", sa.Column("notes", sa.Text(), nullable=True))
    op.add_column(
        "appointments",
        sa.Column(
            "reminder_24h_sent", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
    )
    op.add_column(
        "appointments",
        sa.Column(
            "reminder_2h_sent", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
    )
    op.create_index(op.f("ix_appointments_doctor_id"), "appointments", ["doctor_id"], unique=False)
    op.create_foreign_key(
        op.f("fk_appointments_doctor_id_doctors"),
        "appointments",
        "doctors",
        ["doctor_id"],
        ["id"],
    )

    # Widen the double-booking guard to every status that holds a slot.
    op.drop_index("uq_appointments_tenant_id_scheduled_at", table_name="appointments")
    op.create_index(
        "uq_appointments_tenant_id_scheduled_at",
        "appointments",
        ["tenant_id", "scheduled_at"],
        unique=True,
        postgresql_where=sa.text("status IN (" + _ACTIVE_STATUSES + ")"),
    )


def downgrade() -> None:
    op.drop_index("uq_appointments_tenant_id_scheduled_at", table_name="appointments")
    op.create_index(
        "uq_appointments_tenant_id_scheduled_at",
        "appointments",
        ["tenant_id", "scheduled_at"],
        unique=True,
        postgresql_where=sa.text("status = 'scheduled'"),
    )
    op.drop_constraint(
        op.f("fk_appointments_doctor_id_doctors"), "appointments", type_="foreignkey"
    )
    op.drop_index(op.f("ix_appointments_doctor_id"), table_name="appointments")
    op.drop_column("appointments", "reminder_2h_sent")
    op.drop_column("appointments", "reminder_24h_sent")
    op.drop_column("appointments", "notes")
    op.drop_column("appointments", "patient_phone")
    op.drop_column("appointments", "doctor_id")
    op.alter_column("appointments", "doctor_name", new_column_name="doctor")

    op.drop_column("users", "is_admin")
    op.drop_column("channels", "created_at")
    op.drop_column("channels", "config")

    op.drop_index(op.f("ix_leads_conversation_id"), table_name="leads")
    op.drop_index(op.f("ix_leads_user_id"), table_name="leads")
    op.drop_index(op.f("ix_leads_tenant_id"), table_name="leads")
    op.drop_table("leads")

    op.drop_index(op.f("ix_doctors_tenant_id"), table_name="doctors")
    op.drop_table("doctors")
