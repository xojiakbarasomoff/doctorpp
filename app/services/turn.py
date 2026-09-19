"""One patient turn, from what is true to what is said.

The order here is the whole design. Everything the clinic acts on -- who the
patient is, what day they chose, which appointments exist, whether one was
cancelled -- is settled by the backend before the model is asked for a
sentence, and written back after. The model is given facts and writes
language; it is never asked what is true.

    lock the conversation
    → read the patient's record, the flow state, their real appointments
    → classify the intent from their own words
    → remember anything new they just told us (name, number, day, time)
    → run the deterministic part of the flow (cancellation, in particular)
    → ask the model for the sentence, with all of the above as facts
    → let the backend settle any booking the reply agreed to
    → write the state back

app.workers.tasks used to do the middle of this implicitly, through a prompt.
The failures that followed were all the same failure: the transcript is ten
turns long, so anything older had simply gone.
"""

import logging
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.models.appointment import Appointment
from app.models.conversation_state import CompletedAction, ConversationState, FlowIntent, FlowStatus
from app.rag.embeddings import EmbeddingProvider
from app.rag.llm import ChatMessage, LLMProvider
from app.repositories.appointment import AppointmentRepository
from app.repositories.conversation_state import ConversationStateRepository
from app.services import booking as booking_service
from app.services import booking_state, cancellation, patient_profile
from app.services import when as when_service
from app.repositories.doctor import DoctorRepository
from app.services.answer import generate_answer
from app.services.appointment import (
    CLINIC_TIMEZONE,
    SlotAlreadyBookedError,
    assign_doctor,
    create_appointment,
    days_off,
    is_within_working_hours,
)
from app.services.booking import settle as settle_booking
from app.services.intent import BOOKING_INTENTS, Intent, classify

logger = logging.getLogger(__name__)


@dataclass
class TurnResult:
    reply: str
    intent: Intent
    appointment: Appointment | None = None
    cancelled: Appointment | None = None
    state_before: str = ""
    state_after: str = ""
    profile_used: bool = False
    facts: list[str] = field(default_factory=list)


async def lock_conversation(session: AsyncSession, conversation_id: uuid.UUID) -> None:
    """Serialise this conversation for the rest of the transaction.

    Two bubbles half a second apart are answered by two workers. Without
    this they both read the same state, both decide the same question is
    still open, and the second one writes its answer over the first's --
    which is a patient asked for their telephone number twice in four
    seconds.

    A transaction-scoped advisory lock: Postgres releases it at commit or
    rollback, so nothing can be left holding it, and it costs one row lock
    rather than a table.
    """
    await session.execute(
        sql_text("SELECT pg_advisory_xact_lock(hashtext(:key))"),
        {"key": str(conversation_id)},
    )


async def respond(
    session: AsyncSession,
    *,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    message: str,
    history: Sequence[ChatMessage] | None = None,
    source: str = "instagram",
    settings: Settings | None = None,
    llm_provider: LLMProvider | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    now: datetime | None = None,
) -> TurnResult:
    """Answer one message, and leave the world consistent with the answer."""
    resolved = settings or get_settings()
    moment = now or datetime.now(UTC)
    await lock_conversation(session, conversation_id)

    states = ConversationStateRepository(session)
    state = await states.get_or_create(conversation_id)
    before = f"{state.status}/{state.awaiting_field or '-'}"

    # What they just told us about themselves, stored before anything reads it.
    found = await patient_profile.remember(
        session, user_id=user_id, message=message, history=history
    )
    profile = await patient_profile.load(session, user_id)

    appointments = await AppointmentRepository(session).list_active_for_user(
        user_id, after=moment
    )
    intent = classify(message, state=state)

    facts: list[str] = []
    cancelled: Appointment | None = None
    appointment: Appointment | None = None

    if intent is Intent.CANCEL_REQUEST:
        state, facts = await _begin_cancellation(
            session, states, state, user_id=user_id, moment=moment
        )
    elif intent is Intent.CANCEL_CONFIRM:
        state, facts, cancelled = await _finish_cancellation(
            session, states, state, user_id=user_id
        )
        if cancelled is not None:
            appointments = await AppointmentRepository(session).list_active_for_user(
                user_id, after=moment
            )
    elif intent is Intent.BOOKING_CONFIRM:
        # They said yes to the summary the assistant read back. Everything
        # the row needs is in the state, so it is written here -- before the
        # model is asked for a sentence, so that "tasdiqlandi" is only ever
        # said about a booking the diary already holds.
        appointment, facts = await _book_from_state(
            session,
            state,
            user_id=user_id,
            conversation_id=conversation_id,
            source=source,
            profile=profile,
        )
        if appointment is not None:
            state = await states.clear_flow(
                state, action=CompletedAction.BOOKING_CREATED, appointment_id=appointment.id
            )
            appointments = await AppointmentRepository(session).list_active_for_user(
                user_id, after=moment
            )
    elif intent in BOOKING_INTENTS or FlowStatus(state.status) in {
        FlowStatus.COLLECTING,
        FlowStatus.AWAITING_DATE,
        FlowStatus.AWAITING_TIME,
    }:
        state = await _advance_booking(
            states,
            state,
            message=message,
            intent=intent,
            profile=profile,
            moment=moment,
        )

    booking = booking_state.read(
        profile=profile,
        state=state,
        appointments=appointments,
        history=history,
        user_message=message,
        booking_in_progress=_is_booking(state, intent),
        other_phone=found.other_phone,
    )

    reply = await generate_answer(
        session,
        message,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        settings=resolved,
        history=list(history or []),
        booking=booking,
        intent=intent,
        facts=facts,
    )

    if FlowStatus(state.status) is FlowStatus.AWAITING_CONFIRMATION:
        # The summary has been read back and the patient has not said yes
        # yet. A marker written now is the model booking on its own
        # authority, which is the one thing it may never do: it is dropped,
        # the patient sees only the question, and the row waits for the
        # confirmation that creates it.
        cleaned, marked, _ = booking_service.extract(reply)
        if marked is not None:
            logger.warning(
                "booking_marker_ignored_before_confirmation",
                extra={"conversation_id": str(conversation_id), "slot": marked.isoformat()},
            )
        reply = cleaned
    elif appointment is None and resolved.booking_enabled:
        reply, appointment = await settle_booking(
            AppointmentRepository(session),
            reply,
            user_id=user_id,
            conversation_id=conversation_id,
            source=source,
            patient_name=profile.name,
            # Only an explicit "yana bir kunga ham" writes a second row; a
            # re-confirmation of the booking just made moves it instead.
            allow_second=intent is Intent.BOOK_NEW,
        )
        if appointment is not None:
            state = await states.clear_flow(
                state, action=CompletedAction.BOOKING_CREATED, appointment_id=appointment.id
            )
            reply = _say_the_day_that_was_booked(reply, appointment, moment)

    after = f"{state.status}/{state.awaiting_field or '-'}"
    logger.info(
        "turn_handled",
        extra={
            "conversation_id": str(conversation_id),
            "intent": str(intent),
            "state_before": before,
            "state_after": after,
            "profile_reused": bool(profile.name or profile.phone),
            "profile_learned": bool(found.name or found.phone),
            "appointments_held": len(appointments),
            "booking_created": appointment is not None,
            "booking_cancelled": cancelled is not None,
        },
    )
    return TurnResult(
        reply=reply,
        intent=intent,
        appointment=appointment,
        cancelled=cancelled,
        state_before=before,
        state_after=after,
        profile_used=bool(profile.name or profile.phone),
        facts=facts,
    )


# --- the deterministic halves ------------------------------------------------


async def _book_from_state(
    session: AsyncSession,
    state: ConversationState,
    *,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    source: str,
    profile: patient_profile.Profile,
) -> tuple[Appointment | None, list[str]]:
    """Write the booking the patient has just confirmed.

    Everything comes from the state and the patient's record: the day, the
    time, the name, the number. The model is not consulted and its marker is
    not needed -- this is the "tool" the clinic asked for, and the only
    thing that may make the assistant say a booking exists.
    """
    if state.requested_date is None or state.requested_time is None:
        return None, [
            "There is nothing to confirm yet: the day and the time are not "
            "both settled. Ask for the missing one."
        ]

    slot = datetime.combine(
        state.requested_date, state.requested_time, tzinfo=CLINIC_TIMEZONE
    )
    if not is_within_working_hours(slot) or slot.date() in await days_off(
        AppointmentRepository(session)
    ):
        return None, [
            f"{slot:%d.%m.%Y %H:%M} is NOT a time this clinic can give: it is "
            "outside working hours or on a closed day. Say so and offer a time "
            "from THE APPOINTMENT BOOK."
        ]

    repo = AppointmentRepository(session)
    doctors = await DoctorRepository(session).list_active()
    try:
        doctor_id, doctor_name = await assign_doctor(repo, doctors, slot.astimezone(UTC))
        appointment = await create_appointment(
            repo,
            scheduled_at=slot.astimezone(UTC),
            source=source,
            doctor_id=doctor_id,
            doctor_name=doctor_name,
            user_id=user_id,
            conversation_id=conversation_id,
            patient_name=profile.name,
            patient_phone=profile.phone,
            notes=state.reason,
        )
    except SlotAlreadyBookedError:
        logger.warning(
            "confirmed_slot_taken",
            extra={"conversation_id": str(conversation_id), "slot": slot.isoformat()},
        )
        return None, [
            f"{slot:%d.%m.%Y %H:%M} was taken while you were confirming it. Say "
            "so plainly and offer the nearest free times."
        ]

    logger.info(
        "booking_created_on_confirmation",
        extra={"appointment_id": str(appointment.id), "slot": slot.isoformat()},
    )
    return appointment, [
        "The booking IS NOW IN THE DIARY: "
        f"{slot:%d.%m.%Y} at {slot:%H:%M}, with {doctor_name}. Confirm it to "
        "them in one short line, with the date and the time."
    ]


async def _begin_cancellation(
    session: AsyncSession,
    states: ConversationStateRepository,
    state: ConversationState,
    *,
    user_id: uuid.UUID,
    moment: datetime,
) -> tuple[ConversationState, list[str]]:
    result = await cancellation.begin(session, user_id=user_id, now=moment)

    if result.outcome is cancellation.Outcome.NOTHING_TO_CANCEL:
        return state, [
            "This patient has no appointment to cancel. Say so plainly and "
            "do not offer to cancel anything."
        ]
    if result.outcome is cancellation.Outcome.NEEDS_CONFIRMATION:
        assert result.appointment is not None
        state = await states.save(
            state,
            status=FlowStatus.AWAITING_CANCEL_CONFIRM,
            intent=FlowIntent.CANCELLATION,
            awaiting_field="cancel_confirm",
            appointment_id=result.appointment.id,
        )
        return state, [
            "This patient has ONE appointment: "
            f"{cancellation.describe(result.appointment)}. They have asked to "
            "cancel it. Ask them, in one line, to confirm that this is the "
            "one — and do NOT say it is cancelled yet, because it is not.",
        ]

    state = await states.save(
        state,
        status=FlowStatus.AWAITING_CANCEL_CHOICE,
        intent=FlowIntent.CANCELLATION,
        awaiting_field="cancel_choice",
        appointment_id=None,
    )
    listed = ", ".join(cancellation.describe(a) for a in result.choices)
    return state, [
        f"This patient has SEVERAL appointments: {listed}. Ask which one they "
        "want cancelled. Do not cancel anything and do not guess."
    ]


async def _finish_cancellation(
    session: AsyncSession,
    states: ConversationStateRepository,
    state: ConversationState,
    *,
    user_id: uuid.UUID,
) -> tuple[ConversationState, list[str], Appointment | None]:
    if state.appointment_id is None:
        return state, ["There is nothing waiting to be cancelled."], None

    result = await cancellation.cancel(
        session, user_id=user_id, appointment_id=state.appointment_id
    )
    if not result.cancelled:
        return (
            state,
            [
                "The cancellation did NOT go through. Tell them you could not "
                "cancel it and give the clinic's number so a person can."
            ],
            None,
        )

    assert result.appointment is not None
    state = await states.clear_flow(
        state, action=CompletedAction.BOOKING_CANCELLED, appointment_id=result.appointment.id
    )
    return (
        state,
        [
            "Their appointment for "
            f"{cancellation.describe(result.appointment)} has been CANCELLED, "
            "just now, in the clinic's diary. Tell them it is done, in one line."
        ],
        result.appointment,
    )


async def _advance_booking(
    states: ConversationStateRepository,
    state: ConversationState,
    *,
    message: str,
    intent: Intent,
    profile: patient_profile.Profile,
    moment: datetime,
) -> ConversationState:
    """Record what this message settled, and name what is still open.

    The day and the time are resolved once, here, and stored as a real date
    and a real time. Every later turn reads those -- which is what stops
    "keyingi hafta payshanba" quietly becoming tomorrow.
    """
    found = when_service.read(message, now=moment.astimezone(CLINIC_TIMEZONE))

    requested_date = state.requested_date
    requested_time = state.requested_time
    if intent is Intent.BOOK_NEW:
        # A second appointment is a fresh flow: the day they chose for the
        # first one must not be carried into it.
        requested_date, requested_time = None, None
    if found.day is not None:
        requested_date = found.day
    if found.at is not None:
        requested_time = found.at

    reason = state.reason
    if state.awaiting_field == "reason" and message.strip() and found.empty:
        reason = message.strip()

    provisional = booking_state.BookingState(
        name=profile.name,
        phone=profile.phone,
        reason=reason,
        in_progress=True,
        requested_date=requested_date,
        requested_time=requested_time,
    )
    needed = provisional.next_needed
    status = {
        "name": FlowStatus.COLLECTING,
        "phone": FlowStatus.COLLECTING,
        "reason": FlowStatus.COLLECTING,
        "date": FlowStatus.AWAITING_DATE,
        "time": FlowStatus.AWAITING_TIME,
        None: FlowStatus.AWAITING_CONFIRMATION,
    }[needed]

    return await states.save(
        state,
        status=status,
        intent=FlowIntent.RESCHEDULE
        if intent is Intent.RESCHEDULE_EXISTING
        else FlowIntent.BOOKING,
        awaiting_field=needed,
        requested_date=requested_date,
        requested_time=requested_time,
        reason=reason,
        last_completed_action=CompletedAction.NONE
        if intent is Intent.BOOK_NEW
        else None,
    )


# "Bugun 16:20" agreed, and the confirmation said "ertaga 16:20". The row
# was right; the sentence was not, and the sentence is what the patient
# acts on. Every day-word in a confirmation is checked against the row that
# was actually written, and a wrong one is replaced by the date itself --
# unambiguous, and never the wrong day.
_DAY_WORDS = {
    "bugun": 0,
    "бугун": 0,
    "сегодня": 0,
    "ertaga": 1,
    "эртага": 1,
    "завтра": 1,
    "indinga": 2,
    "индинга": 2,
    "послезавтра": 2,
}
_DAY_WORD_RE = re.compile(r"\b(" + "|".join(_DAY_WORDS) + r")\w*", re.IGNORECASE)


def _say_the_day_that_was_booked(
    reply: str, appointment: Appointment, moment: datetime
) -> str:
    """Correct a confirmation that names the wrong day.

    Only when it is wrong: a confirmation that says "bugun" for a booking
    made today is left exactly as the model wrote it.
    """
    booked = appointment.scheduled_at.astimezone(CLINIC_TIMEZONE).date()
    today = moment.astimezone(CLINIC_TIMEZONE).date()
    offset = (booked - today).days

    def fix(match: "re.Match[str]") -> str:
        word = match.group(1).lower()
        if _DAY_WORDS[word] == offset:
            return match.group(0)
        logger.warning(
            "confirmation_day_corrected",
            extra={
                "appointment_id": str(appointment.id),
                "said": match.group(0),
                "booked": booked.isoformat(),
            },
        )
        return when_service.spoken(booked, today)

    return _DAY_WORD_RE.sub(fix, reply)


def _is_booking(state: ConversationState, intent: Intent) -> bool:
    if intent in BOOKING_INTENTS:
        return True
    return FlowStatus(state.status) in {
        FlowStatus.COLLECTING,
        FlowStatus.AWAITING_DATE,
        FlowStatus.AWAITING_TIME,
        FlowStatus.AWAITING_CONFIRMATION,
    }
