"""One patient turn: ask the model for the reply, then write down any booking
it made.

    lock the conversation
    → remember anything new they told us about themselves (name, number)
    → ask the model for the reply
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
from app.services import patient_profile
from app.services.answer import generate_answer
from app.services.booking import extract as extract_booking
from app.services.booking import settle as settle_booking
from app.services.conversation import hours_since_last_contact

logger = logging.getLogger(__name__)


@dataclass
class TurnResult:
    reply: str
    appointment: Appointment | None = None


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

    await patient_profile.remember(session, user_id=user_id, message=message, history=history)
    profile = await patient_profile.load(session, user_id)
    gap_hours = await hours_since_last_contact(
        session, conversation_id, now=datetime.now(UTC)
    )

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
        long_gap=gap_hours is not None and gap_hours >= 24,
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
        )
    else:
        reply = extract_booking(reply)[0]

    logger.info(
        "turn_handled",
        extra={
            "conversation_id": str(conversation_id),
            "booking_created": appointment is not None,
        },
    )
    return TurnResult(reply=reply, appointment=appointment)
