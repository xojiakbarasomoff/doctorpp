"""Cancelling a patient's own appointment, from the database outwards.

Three things had to be true before this could exist at all, and all three
are about not trusting the conversation:

* which appointments a patient has comes from the database, never from what
  anybody remembers being said;
* when they have more than one, the patient says which -- nothing here
  guesses, because guessing means cancelling a visit somebody still wants;
* the patient is told it is cancelled only after the row says it is.

The model's part is the sentence. This decides what is true, and returns
the facts the sentence is written from.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.appointment import Appointment, AppointmentStatus
from app.repositories.appointment import AppointmentRepository
from app.services.appointment import CLINIC_TIMEZONE

logger = logging.getLogger(__name__)


class Outcome(StrEnum):
    NOTHING_TO_CANCEL = "nothing_to_cancel"
    NEEDS_CONFIRMATION = "needs_confirmation"
    NEEDS_CHOICE = "needs_choice"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class Result:
    outcome: Outcome
    appointment: Appointment | None = None
    choices: tuple[Appointment, ...] = ()

    @property
    def cancelled(self) -> bool:
        return self.outcome is Outcome.CANCELLED


def describe(appointment: Appointment) -> str:
    """One appointment as a person reads it, for the prompt and the sheet."""
    local = appointment.scheduled_at.astimezone(CLINIC_TIMEZONE)
    return f"{local:%d.%m.%Y %H:%M}"


async def begin(
    session: AsyncSession, *, user_id: uuid.UUID, now: datetime | None = None
) -> Result:
    """What a request to cancel means for this patient, right now.

    Exactly one booking: ask them to confirm that one. Several: ask which.
    None: say so -- which is the answer a patient who has already cancelled
    should get, rather than a confirmation of something that is not there.
    """
    appointments = await AppointmentRepository(session).list_active_for_user(
        user_id, after=now or datetime.now(UTC)
    )
    if not appointments:
        return Result(Outcome.NOTHING_TO_CANCEL)
    if len(appointments) == 1:
        return Result(Outcome.NEEDS_CONFIRMATION, appointment=appointments[0])
    return Result(Outcome.NEEDS_CHOICE, choices=tuple(appointments))


async def cancel(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    appointment_id: uuid.UUID,
    reason: str = "Bemor bekor qildi",
) -> Result:
    """Cancel one appointment, and only if it is really this patient's.

    The id comes from state the backend wrote, not from anything the model
    produced; it is checked against the patient's own live bookings anyway,
    because an id that reaches this function from somewhere else must not be
    able to cancel a stranger's visit.
    """
    repo = AppointmentRepository(session)
    live = await repo.list_active_for_user(user_id, after=datetime.now(UTC))
    target = next((a for a in live if a.id == appointment_id), None)
    if target is None:
        logger.warning(
            "cancellation_target_missing",
            extra={"user_id": str(user_id), "appointment_id": str(appointment_id)},
        )
        return Result(Outcome.NOTHING_TO_CANCEL)

    try:
        target.status = AppointmentStatus.CANCELLED
        target.notes = f"{target.notes or ''}\n[bekor: {reason}]".strip()
        await session.flush()
    except Exception:  # noqa: BLE001 - the patient must not be told it worked
        logger.exception("cancellation_failed", extra={"appointment_id": str(appointment_id)})
        return Result(Outcome.FAILED, appointment=target)

    logger.info(
        "appointment_cancelled_by_patient",
        extra={"appointment_id": str(target.id), "user_id": str(user_id)},
    )
    return Result(Outcome.CANCELLED, appointment=target)
