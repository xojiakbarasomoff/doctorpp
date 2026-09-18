"""What the clinic already knows, assembled from the places that hold it.

This used to read the conversation and guess. It no longer guesses: the
three things a booking needs come from the patient's record and the row the
backend writes, and the transcript is only consulted for the reason they
gave in this conversation.

    users.name / users.phone        who they are          (patient_profile)
    conversation_states             where the flow is     (ConversationState)
    appointments                    what is actually booked

Rendered into the prompt as facts, with the one open question named, so the
model's job is the sentence and not the bookkeeping. The order matters:
where the structured state and the transcript disagree, the structured
state is what is written down and the transcript is what somebody said.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, time

from app.models.appointment import Appointment
from app.models.conversation_state import CompletedAction, ConversationState, FlowStatus
from app.rag.llm import ChatMessage
from app.services import when as when_service
from app.services.appointment import CLINIC_TIMEZONE
from app.services.patient_profile import Profile


@dataclass(frozen=True)
class Booked:
    """One appointment this patient holds, as the database has it."""

    appointment_id: uuid.UUID
    when: str
    name: str | None = None
    phone: str | None = None

    @classmethod
    def of(cls, appointment: Appointment) -> "Booked":
        local = appointment.scheduled_at.astimezone(CLINIC_TIMEZONE)
        return cls(
            appointment_id=appointment.id,
            when=f"{local:%d.%m.%Y %H:%M}",
            name=appointment.patient_name,
            phone=appointment.patient_phone,
        )


@dataclass(frozen=True)
class BookingState:
    """Everything the next reply may rely on, from the backend's own records."""

    name: str | None
    phone: str | None
    reason: str | None
    in_progress: bool
    status: FlowStatus = FlowStatus.IDLE
    requested_date: date | None = None
    requested_time: time | None = None
    last_action: CompletedAction = CompletedAction.NONE
    # Every future appointment this patient holds, soonest first.
    appointments: tuple[Booked, ...] = ()

    @property
    def booked(self) -> Booked | None:
        return self.appointments[0] if self.appointments else None

    @property
    def next_needed(self) -> str | None:
        """The one question still open, in the order the clinic asks them.

        Nothing is "needed" outside a booking, and nothing already known is
        ever needed again -- which is the whole point of reading it from the
        patient's record rather than from ten turns of transcript.
        """
        if not self.in_progress:
            return None
        if self.name is None:
            return "name"
        if self.phone is None:
            return "phone"
        if self.reason is None:
            return "reason"
        if self.requested_date is None:
            return "date"
        if self.requested_time is None:
            return "time"
        return None


def read(
    *,
    profile: Profile,
    state: ConversationState | None,
    appointments: Sequence[Appointment] = (),
    history: Sequence[ChatMessage] | None = None,
    user_message: str = "",
    booking_in_progress: bool | None = None,
) -> BookingState:
    """Assemble the state of play for one turn.

    `booking_in_progress` comes from the intent router: a patient asking
    about the doctor is not in a booking, whatever the row says about a flow
    somebody abandoned last week.
    """
    status = FlowStatus(state.status) if state is not None else FlowStatus.IDLE
    last_action = (
        CompletedAction(state.last_completed_action) if state is not None else CompletedAction.NONE
    )
    in_progress = (
        booking_in_progress
        if booking_in_progress is not None
        else status
        in {
            FlowStatus.COLLECTING,
            FlowStatus.AWAITING_DATE,
            FlowStatus.AWAITING_TIME,
            FlowStatus.AWAITING_CONFIRMATION,
        }
    )

    reason = state.reason if state is not None else None
    if reason is None:
        reason = _reason_from(history, user_message)

    return BookingState(
        name=profile.name,
        phone=profile.phone,
        reason=reason,
        in_progress=in_progress,
        status=status,
        requested_date=state.requested_date if state is not None else None,
        requested_time=state.requested_time if state is not None else None,
        last_action=last_action,
        appointments=tuple(Booked.of(appointment) for appointment in appointments),
    )


def _reason_from(history: Sequence[ChatMessage] | None, user_message: str) -> None:
    """Deliberately nothing.

    The reason is written to the state when the assistant asks for it and
    the patient answers; inferring one from the transcript is how "Ha" and
    "assalomu alaykum" ended up in the clinic's own column.
    """
    return None


_LABELS = {
    "name": "their full name — and whose name, if they are booking for somebody else",
    "phone": "their telephone number",
    "reason": "the reason they are coming, in their own words",
    "date": "which day suits them",
    "time": "which of the free times suits them",
}

_ANSWER_FIRST = (
    "\n- This is a list, not a script. If their message asked you anything "
    "at all — a time, the address, whether you do something — answer that "
    "first, in the same message, and then ask for what is missing. A "
    "patient whose question is met with the next form field has been told "
    "nobody read it."
)


def render(state: BookingState) -> str:
    """The state as a prompt section: facts first, then the one open question."""
    lines: list[str] = []
    if state.name:
        lines.append(f"- Their name: {state.name}")
    if state.phone:
        lines.append(f"- Their telephone number: {state.phone}")
    if state.reason:
        lines.append(f"- Why they are coming: {state.reason}")
    if state.requested_date:
        lines.append(
            f"- The day they have chosen: {state.requested_date:%d.%m.%Y}"
            f" ({when_service.spoken(state.requested_date, _today())})."
            " This is settled. Do not offer another day and do not work it"
            " out again from what they wrote earlier."
        )
    if state.requested_time:
        lines.append(f"- The time they have chosen: {state.requested_time:%H:%M}")

    section = "\n\nWHAT THE CLINIC ALREADY KNOWS"
    section += "\n" + "\n".join(lines) if lines else "\n- Nothing about this patient yet."
    if lines:
        section += (
            "\n- These come from the clinic's own records, not from this "
            "conversation. Never ask for one of them again, and never ask "
            "the patient to confirm one."
        )

    if state.appointments:
        section += "\n\nAPPOINTMENTS THIS PATIENT ALREADY HAS (from the database)"
        for booked in state.appointments:
            section += f"\n- {booked.when}"
        section += (
            "\n- This is the whole list and it is authoritative. If they ask "
            "what they have booked, answer from it and from nothing else."
        )

    if state.last_action is CompletedAction.BOOKING_CREATED and not state.in_progress:
        section += (
            "\n\nTHE BOOKING IS DONE. Their last booking was written down and "
            "confirmed. They are not in the middle of anything: answer what "
            "they wrote — thanks, a question, small talk — and do not ask for "
            "a name, a number, a reason or a time."
        )
    elif state.last_action is CompletedAction.BOOKING_CANCELLED and not state.in_progress:
        section += (
            "\n\nTHEIR APPOINTMENT WAS CANCELLED just now, as they asked. Do "
            "not offer to book them again unless they ask for it."
        )

    if state.in_progress:
        if state.next_needed is None:
            section += (
                "\n- You have everything. Confirm the day and time they chose "
                "and write the booking marker; ask for nothing else."
            )
        else:
            section += (
                f"\n- The one thing still missing is {_LABELS[state.next_needed]}. "
                "Ask for that one thing only — never two of them in one message."
            )
        section += _ANSWER_FIRST
    return section


def _today() -> date:
    from datetime import datetime

    return datetime.now(CLINIC_TIMEZONE).date()
