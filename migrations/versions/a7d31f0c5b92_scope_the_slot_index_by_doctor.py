"""Scope the appointment slot index by doctor

The old index made a time slot unique per clinic, so a practice with three
urologists could still only see one patient at 11:00. Adding doctor_id makes
the slot unique per clinician instead.

COALESCE rather than the bare column: NULLs are distinct in a unique index, so
two bookings with nobody assigned would each be allowed at the same time --
exactly the double-booking the index exists to prevent. Folding NULL onto a
fixed uuid leaves unassigned bookings behaving as they did.

Creating the new index before dropping the old one would fail on any clinic
that already has two same-time bookings under different doctors, which none do
today; the old index guarantees it. Dropping first is therefore safe and keeps
the window where neither index guards the table down to a single statement.

Revision ID: a7d31f0c5b92
Revises: 527feaadcded
"""

from collections.abc import Sequence

from alembic import op

revision: str = "a7d31f0c5b92"
down_revision: str | None = "527feaadcded"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ACTIVE = "status IN ('confirmed', 'scheduled')"
_SENTINEL = "00000000-0000-0000-0000-000000000000"

_OLD = "uq_appointments_tenant_id_scheduled_at"
_NEW = "uq_appointments_tenant_doctor_scheduled_at"


def upgrade() -> None:
    op.drop_index(_OLD, table_name="appointments")
    op.execute(
        f"CREATE UNIQUE INDEX {_NEW} ON appointments "
        f"(tenant_id, COALESCE(doctor_id, '{_SENTINEL}'::uuid), scheduled_at) "
        f"WHERE {_ACTIVE}"
    )


def downgrade() -> None:
    op.drop_index(_NEW, table_name="appointments")
    # Going back narrows what the table allows, so a clinic that has since
    # booked two doctors into the same slot cannot downgrade without first
    # cancelling one of them. Left to fail loudly rather than deleting rows.
    op.execute(
        f"CREATE UNIQUE INDEX {_OLD} ON appointments "
        f"(tenant_id, scheduled_at) WHERE {_ACTIVE}"
    )
