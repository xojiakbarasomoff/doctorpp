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
from app.services import booking_state, cancellation, patient_profile
from app.services import when as when_service
from app.services.answer import generate_answer
from app.services.appointment import CLINIC_TIMEZONE
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

    appointment: Appointment | None = None
    if resolved.booking_enabled:
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


def _is_booking(state: ConversationState, intent: Intent) -> bool:
    if intent in BOOKING_INTENTS:
        return True
    return FlowStatus(state.status) in {
        FlowStatus.COLLECTING,
        FlowStatus.AWAITING_DATE,
        FlowStatus.AWAITING_TIME,
        FlowStatus.AWAITING_CONFIRMATION,
    }
