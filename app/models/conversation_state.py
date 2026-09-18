"""Where a conversation has got to, kept out of the transcript.

The assistant is given the last few turns of a conversation and nothing
else, so everything it was told earlier is gone: the day a patient chose on
Tuesday, the number they typed twenty messages ago, the fact that the visit
they are asking about has already been booked. It filled the gap by asking
again, which is what a form does and what a front desk never does.

This is the other half of the answer to that, beside users.name and
users.phone: one row per conversation saying what is being done and how far
it has got. The transcript stays what it is -- what was said -- and this is
what is true.

One row per conversation, replaced in place, with a version column so two
messages arriving at once cannot each write their own idea of the state
over the other's (see app.repositories.conversation_state).
"""

import uuid
from datetime import date, datetime, time
from enum import StrEnum

from sqlalchemy import Date, DateTime, ForeignKey, Integer, String, Text, Time, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class FlowStatus(StrEnum):
    """What the conversation is in the middle of, if anything.

    IDLE is the normal state of a conversation: somebody asking about the
    clinic is not "in" anything. The rest exist so that the next message can
    be read as the answer to the last question rather than guessed at.
    """

    IDLE = "idle"
    COLLECTING = "collecting"  # name / phone / reason still wanted
    AWAITING_DATE = "awaiting_date"
    AWAITING_TIME = "awaiting_time"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    AWAITING_CANCEL_CHOICE = "awaiting_cancel_choice"  # which of several
    AWAITING_CANCEL_CONFIRM = "awaiting_cancel_confirm"


class FlowIntent(StrEnum):
    """Which flow the status belongs to."""

    NONE = "none"
    BOOKING = "booking"
    RESCHEDULE = "reschedule"
    CANCELLATION = "cancellation"


class CompletedAction(StrEnum):
    """The last thing the backend actually did for this patient.

    Recorded so that "rahmat" after a booking is answered as thanks rather
    than as the next field of a form that has already finished: the flow is
    idle, and this says why.
    """

    NONE = "none"
    BOOKING_CREATED = "booking_created"
    BOOKING_MOVED = "booking_moved"
    BOOKING_CANCELLED = "booking_cancelled"
    CALLBACK_RECORDED = "callback_recorded"


class ConversationState(Base):
    """One row per conversation: what it is doing and what it has settled."""

    __tablename__ = "conversation_states"

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        primary_key=True,
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(40), nullable=False, default=FlowStatus.IDLE)
    intent: Mapped[str] = mapped_column(String(40), nullable=False, default=FlowIntent.NONE)
    # Which of name / phone / reason / date / time the last question asked
    # for, so the next message is read as its answer.
    awaiting_field: Mapped[str | None] = mapped_column(String(40), nullable=True)

    # Canonical, not the words the patient used. "Keyingi hafta payshanba"
    # is resolved once, here, and every later turn reads this date -- which
    # is what stops it quietly becoming tomorrow.
    requested_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    requested_time: Mapped[time | None] = mapped_column(Time, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # The appointment this flow is about: the one being rescheduled, the one
    # being cancelled, or the one just made.
    appointment_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("appointments.id", ondelete="SET NULL"), nullable=True
    )
    last_completed_action: Mapped[str] = mapped_column(
        String(40), nullable=False, default=CompletedAction.NONE
    )

    # Bumped on every write. A second worker holding an older version loses
    # and re-reads rather than overwriting what the first one decided.
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
