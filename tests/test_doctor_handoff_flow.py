"""A conversation the assistant hands to the doctor is pinned until a person
answers, and only a person answering unpins it.

Driven through the real turn and the real echo job against the test
database; the model is a scripted fake.
"""

import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractContextManager, asynccontextmanager
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation
from app.models.message import MessageSender
from app.repositories.message import MessageRepository
from app.services import turn
from app.workers.tasks import note_clinic_reply
from tests.conftest import Seed
from tests.test_conversation_memory import NoEmbeddings, ScriptedLLM, _settings


async def _respond(session: AsyncSession, seed: Seed, reply: str) -> turn.TurnResult:
    llm = ScriptedLLM()
    llm.say(reply)
    return await turn.respond(
        session,
        conversation_id=seed.a.conversation.id,
        user_id=seed.a.user.id,
        message="Nega bunday bo'lyapti, doktor o'zi aytsin",
        history=[],
        source="instagram",
        settings=_settings(),  # type: ignore[arg-type]
        llm_provider=llm,
        embedding_provider=NoEmbeddings(),
    )


async def _flag_state(session: AsyncSession, seed: Seed) -> datetime | None:
    conversation = await session.get(Conversation, seed.a.conversation.id)
    assert conversation is not None
    await session.refresh(conversation)
    return conversation.needs_doctor_since


def _sessions(session: AsyncSession) -> Callable[[], AbstractContextManager[AsyncSession]]:
    @asynccontextmanager
    async def factory() -> AsyncIterator[AsyncSession]:
        yield session

    return factory  # type: ignore[return-value]


async def _echo(session: AsyncSession, seed: Seed, text: str | None) -> None:
    await note_clinic_reply(
        {},
        str(seed.tenant_a.id),
        str(seed.a.channel.id),
        seed.a.user.external_id,
        text,
        session_factory=_sessions(session),
    )


# --- raising the flag ---------------------------------------------------------


async def test_a_marked_handover_pins_the_conversation_and_hides_the_marker(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
) -> None:
    with as_tenant(seed.tenant_a.id):
        result = await _respond(db_session, seed, "Savolingizni doktorga yetkazaman. [[DOCTOR]]")
        since = await _flag_state(db_session, seed)

    assert result.reply == "Savolingizni doktorga yetkazaman."
    assert since is not None


async def test_a_malformed_marker_never_reaches_the_patient(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
) -> None:
    with as_tenant(seed.tenant_a.id):
        result = await _respond(db_session, seed, "Doktor javob beradi [[DOCTOR: sabab so'raldi]")
        since = await _flag_state(db_session, seed)

    assert "[" not in result.reply and "DOCTOR" not in result.reply
    assert since is not None


async def test_an_ordinary_reply_pins_nothing(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
) -> None:
    with as_tenant(seed.tenant_a.id):
        result = await _respond(
            db_session, seed, "Buni ko'rmasdan aytib bo'lmaydi, qabulga yozaymi?"
        )
        since = await _flag_state(db_session, seed)

    assert result.reply == "Buni ko'rmasdan aytib bo'lmaydi, qabulga yozaymi?"
    assert since is None


async def test_a_second_handover_keeps_the_time_it_first_started_waiting(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
) -> None:
    with as_tenant(seed.tenant_a.id):
        await _respond(db_session, seed, "Doktorning o'zi javob beradi. [[DOCTOR]]")
        first = await _flag_state(db_session, seed)
        await _respond(db_session, seed, "Doktorning o'zi javob beradi. [[DOCTOR]]")
        second = await _flag_state(db_session, seed)

    assert first is not None and first == second


# --- the doctor answering from the Instagram app --------------------------------


async def _pin_after_bot_said(session: AsyncSession, seed: Seed, bot_said: str) -> None:
    conversation = await session.get(Conversation, seed.a.conversation.id)
    assert conversation is not None
    conversation.needs_doctor_since = datetime.now(UTC) - timedelta(minutes=5)
    await MessageRepository(session).create(
        conversation_id=seed.a.conversation.id,
        sender=MessageSender.BOT,
        content=bot_said,
        channel="instagram",
    )
    await session.flush()


async def test_the_echo_of_the_assistants_own_message_keeps_the_pin(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
) -> None:
    with as_tenant(seed.tenant_a.id):
        await _pin_after_bot_said(db_session, seed, "Savolingizni doktorga yetkazaman.")
        await _echo(db_session, seed, "  Savolingizni doktorga   yetkazaman. ")
        assert await _flag_state(db_session, seed) is not None


async def test_the_doctor_typing_in_the_app_unpins_it(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
) -> None:
    with as_tenant(seed.tenant_a.id):
        await _pin_after_bot_said(db_session, seed, "Savolingizni doktorga yetkazaman.")
        await _echo(db_session, seed, "Assalomu alaykum, men doktor Temur. Ertaga kelib ko'ring.")
        assert await _flag_state(db_session, seed) is None


async def test_a_short_reply_contained_in_a_bot_message_still_counts_as_a_person(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
) -> None:
    """ "Ha" appears inside plenty of the assistant's sentences; it is still
    the doctor answering."""
    with as_tenant(seed.tenant_a.id):
        await _pin_after_bot_said(db_session, seed, "Ha, albatta, doktorga yetkazaman.")
        await _echo(db_session, seed, "Ha")
        assert await _flag_state(db_session, seed) is None


async def test_a_photo_from_the_app_unpins_it(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
) -> None:
    with as_tenant(seed.tenant_a.id):
        await _pin_after_bot_said(db_session, seed, "Savolingizni doktorga yetkazaman.")
        await _echo(db_session, seed, None)
        assert await _flag_state(db_session, seed) is None


async def test_an_echo_for_an_unpinned_conversation_changes_nothing(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
) -> None:
    with as_tenant(seed.tenant_a.id):
        await _echo(db_session, seed, "Salom")
        assert await _flag_state(db_session, seed) is None


async def test_an_echo_to_somebody_who_never_wrote_is_ignored(
    db_session: AsyncSession,
    seed: Seed,
) -> None:
    await note_clinic_reply(
        {},
        str(seed.tenant_a.id),
        str(seed.a.channel.id),
        "nobody-with-this-id",
        "Salom",
        session_factory=_sessions(db_session),
    )
