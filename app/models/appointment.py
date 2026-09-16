import uuid
from datetime import datetime
from enum import StrEnum

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, Text, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class AppointmentStatus(StrEnum):
    """The life of a booking.

    The union of what the two bots each used. The Instagram side had
    `scheduled` and `cancelled`; the Telegram side had `pending`,
    `confirmed` and `cancelled`, where `pending` meant the same thing
    `scheduled` did — booked, nobody has called to confirm it yet. One name
    survives for that state, and `confirmed` joins it as the distinct thing
    it actually is: the clinic has spoken to the patient.
    """

    SCHEDULED = "scheduled"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    NO_SHOW = "no_show"


# The statuses that hold a slot against other bookings. Cancelling, or
# marking a no-show, releases the time for somebody else; a completed
# appointment is in the past and cannot collide with anything new.
#
# Defined once and rendered into both the partial unique index below and
# AppointmentRepository's queries, so the constraint the database enforces
# and the rows the application treats as busy can never disagree.
ACTIVE_STATUSES: frozenset[str] = frozenset(
    {AppointmentStatus.SCHEDULED, AppointmentStatus.CONFIRMED}
)

_ACTIVE_STATUS_SQL = ", ".join(f"'{status}'" for status in sorted(ACTIVE_STATUSES))

# Stands in for doctor_id in the unique index below when no doctor has
# been assigned. Any fixed uuid would do; the all-zero one is recognisable
# in an index definition as a placeholder rather than a real row.
_NO_DOCTOR_SENTINEL = "00000000-0000-0000-0000-000000000000"


class Appointment(Base):
    __tablename__ = "appointments"
    __table_args__ = (
        # A slot counts as "taken" only while the booking is active — this is
        # partial, not a plain unique constraint, so cancelling (or a
        # no-show) drops the row out of the index and frees the slot for
        # rebooking. This index is the actual double-booking guard:
        # check_availability()/find_next_free_slot() in
        # app.services.appointment are UX only (they can go stale between
        # the check and the insert); this constraint is what makes a race
        # between two simultaneous bookings resolve to exactly one winner.
        #
        # The Telegram side had no equivalent, which is why two patients
        # could be booked into the same time there.
        # Scoped by doctor, so a slot holds one booking per clinician rather
        # than one for the whole clinic. Without doctor_id in here, a clinic
        # with three urologists could still only see one patient at 11:00 --
        # two thirds of its capacity was unreachable.
        #
        # COALESCE rather than a plain column, because NULLs are distinct in
        # a unique index: two bookings with no doctor assigned would each be
        # allowed at the same time, which is the double-booking this index
        # exists to prevent. Folding NULL onto a fixed uuid keeps "nobody
        # assigned" behaving exactly as it did before -- one such booking per
        # slot -- while each named doctor gets their own.
        Index(
            "uq_appointments_tenant_doctor_scheduled_at",
            "tenant_id",
            text(f"COALESCE(doctor_id, '{_NO_DOCTOR_SENTINEL}'::uuid)"),
            "scheduled_at",
            unique=True,
            postgresql_where=text(f"status IN ({_ACTIVE_STATUS_SQL})"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True
    )
    # Nullable: an operator booking a walk-in/phone patient who has never
    # messaged the clinic has no User row to attach (User requires a
    # channel_id) — patient_name is that booking's identity instead. A bot
    # booking always has both a user_id and a conversation_id.
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True
    )
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=True, index=True
    )
    # Nullable, and paired with doctor_name below rather than replacing it:
    # a booking made before the clinic listed its doctors, or with someone
    # who has since been removed, still has to say who it was with.
    doctor_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("doctors.id"), nullable=True, index=True
    )
    # The name as it stood when the booking was made. Kept alongside
    # doctor_id, not derived from it: a doctor row can be renamed or
    # deactivated, and an old appointment should still read the way it was
    # agreed with the patient.
    doctor_name: Mapped[str] = mapped_column(String(255), nullable=False)
    # The name to book under — may differ from user.name (the account
    # holder may be booking for a spouse/child), and is the only patient
    # identity at all when user_id is None. See
    # app.services.appointment.create_appointment for the "one of user_id /
    # patient_name is required" rule.
    patient_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # From the Telegram side. The number a booking was made on is not
    # necessarily the number on the User row (an operator takes it over the
    # phone), so it is stored per booking.
    patient_phone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    # "bot" | "operator" | "webapp" — which side created this booking.
    source: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'operator'")
    )
    # Set once each reminder has gone out, so the cron that scans for due
    # appointments cannot send the same reminder twice. From the Telegram
    # side, where the columns were added by a hand-run script rather than a
    # migration.
    reminder_24h_sent: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    reminder_2h_sent: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), onupdate=func.now(), nullable=False
    )
