"""One patient turn: ask the model for the reply, then write down any booking
it made.

    lock the conversation
    → a 👍 or "rahmat" that closes an exchange: say nothing
    → remember anything new they told us about themselves (name, number)
    → ask the model for the reply
    → greet only where a greeting belongs (app.services.greeting)
    → book the slot its [[BOOK:...]] marker names, if it wrote one
"""

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.models.appointment import Appointment
from app.rag.embeddings import EmbeddingProvider
from app.rag.llm import ChatMessage, LLMProvider
from app.repositories.appointment import AppointmentRepository
from app.services import greeting, handoff, patient_profile, small_talk
from app.services.answer import generate_answer
from app.services.booking import extract as extract_booking
from app.services.booking import settle as settle_booking
from app.services.conversation import hours_since_last_contact

logger = logging.getLogger(__name__)


@dataclass
class TurnResult:
    reply: str
    appointment: Appointment | None = None
    # The reply hands the patient to the doctor; the conversation is flagged.
    needs_doctor: bool = False
    # Nothing to send: the message closed the exchange (app.services.small_talk).
    silent: bool = False


async def lock_conversation(session: AsyncSession, conversation_id: uuid.UUID) -> None:
    """Serialise this conversation for the rest of the transaction.

    Two bubbles half a second apart are answered by two workers; without
    this both can book from the same reply. A transaction-scoped advisory
    lock: Postgres releases it at commit or rollback.
    """
    await session.execute(
        sql_text("SELECT pg_advisory_xact_lock(hashtext(:key))"),
        {"key": str(conversation_id)},
    )


def _greet_where_it_belongs(
    reply: str,
    *,
    message: str,
    history: Sequence[ChatMessage] | None,
    long_gap: bool,
) -> str:
    """The reply with its hello -- and its "I am the administrator" -- only
    where a person at the front desk would say them."""
    said_hello = greeting.patient_greeted(message)
    mode = greeting.mode_for(first=not history, long_gap=long_gap, patient_greeted=said_hello)
    earlier = [turn["content"] for turn in history or [] if turn["role"] == "assistant"]
    tidied = greeting.tidy(
        reply,
        mode=mode,
        patient_greeted=said_hello,
        introduced_before=greeting.introduced(earlier),
        asked_who=greeting.asked_who(message),
    )
    if tidied != reply:
        logger.info("reply_greeting_tidied", extra={"mode": str(mode)})
    return tidied


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
) -> TurnResult:
    """Answer one message, and write down the booking the reply made."""
    resolved = settings or get_settings()
    await lock_conversation(session, conversation_id)

    gap_hours = await hours_since_last_contact(session, conversation_id, now=datetime.now(UTC))
    long_gap = gap_hours is not None and gap_hours >= 24
    plan = small_talk.plan(message, list(history or []), long_gap=long_gap)
    if plan.silent:
        logger.info("turn_left_unanswered", extra={"conversation_id": str(conversation_id)})
        return TurnResult(reply="", silent=True)

    await patient_profile.remember(session, user_id=user_id, message=message, history=history)
    profile = await patient_profile.load(session, user_id)

    reply = await generate_answer(
        session,
        message,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        settings=resolved,
        history=list(history or []),
        patient=profile,
        # A gap this long is a new conversation in every way that matters
        # for how it opens, even though there is history to read: a front
        # desk greets somebody back who wrote yesterday, and stops doing
        # that only within one sitting. See PatientState.long_gap.
        long_gap=long_gap,
        situation=plan.note,
    )
    reply = _greet_where_it_belongs(reply, message=message, history=history, long_gap=long_gap)

    appointment: Appointment | None = None
    if resolved.booking_enabled:
        reply, appointment = await settle_booking(
            AppointmentRepository(session),
            reply,
            user_id=user_id,
            conversation_id=conversation_id,
            source=source,
            patient_name=profile.name,
        )
    else:
        reply = extract_booking(reply)[0]

    reply, needs_doctor = handoff.extract(reply)
    if needs_doctor:
        await handoff.flag(session, conversation_id)

    logger.info(
        "turn_handled",
        extra={
            "conversation_id": str(conversation_id),
            "booking_created": appointment is not None,
            "handed_to_doctor": needs_doctor,
        },
    )
    return TurnResult(reply=reply, appointment=appointment, needs_doctor=needs_doctor)
