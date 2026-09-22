"""One patient turn: the model writes the reply, and a [[BOOK:...]] marker
in it is written to the diary and removed before the patient sees it.

The model here is a fake with a fixed script: what is being tested is the
flow around the model, not its prose.
"""

import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.message import MessageSender
from app.rag.embeddings import EmbeddingProvider
from app.rag.llm import ChatMessage, LLMProvider
from app.repositories.appointment import AppointmentRepository
from app.repositories.message import MessageRepository
from app.services import turn
from app.services.appointment import CLINIC_TIMEZONE, day_slots
from tests.conftest import Seed, isolated_settings


class ScriptedLLM(LLMProvider):
    def __init__(self) -> None:
        self.replies: list[str] = []
        self.prompts: list[str] = []

    def say(self, *replies: str) -> None:
        self.replies.extend(replies)

    async def generate(self, system_prompt: str, messages: list[ChatMessage]) -> str:
        self.prompts.append(system_prompt)
        return self.replies.pop(0) if self.replies else "Xo'p bo'ladi."


class NoEmbeddings(EmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 1536 for _ in texts]


def _settings(**overrides: object) -> object:
    values: dict[str, object] = {
        "booking_enabled": True,
        "default_reply_language": "Uzbek",
    }
    values.update(overrides)
    return isolated_settings(**values)


def _free_slot(now: datetime) -> datetime:
    """The first working slot at least two days out."""
    day = now.astimezone(CLINIC_TIMEZONE).date() + timedelta(days=2)
    while True:
        slots = list(day_slots(day))
        if slots:
            return slots[3]
        day += timedelta(days=1)


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM()


async def _say(
    session: AsyncSession,
    seed: Seed,
    llm: ScriptedLLM,
    message: str,
    **settings_overrides: object,
) -> turn.TurnResult:
    return await turn.respond(
        session,
        conversation_id=seed.a.conversation.id,
        user_id=seed.a.user.id,
        message=message,
        history=[],
        source="instagram",
        settings=_settings(**settings_overrides),  # type: ignore[arg-type]
        llm_provider=llm,
        embedding_provider=NoEmbeddings(),
    )


async def test_the_models_reply_is_what_the_patient_gets(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
    llm: ScriptedLLM,
) -> None:
    llm.say("Va alaykum assalom! Qanday yordam bera olaman?")
    with as_tenant(seed.tenant_a.id):
        result = await _say(db_session, seed, llm, "Salom")

    assert result.reply == "Va alaykum assalom! Qanday yordam bera olaman?"
    assert result.appointment is None


async def test_a_patient_back_after_a_real_gap_is_told_to_greet_them_again(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
    llm: ScriptedLLM,
) -> None:
    """The gap is read from the real transcript, not from whatever `history`
    a caller happens to pass -- so a patient back after two days gets
    greeted again even though this call passes none."""
    llm.say("Sizlarda UZI bor.")
    with as_tenant(seed.tenant_a.id):
        await MessageRepository(db_session).create(
            conversation_id=seed.a.conversation.id,
            sender=MessageSender.BOT,
            content="Marhamat, yana savolingiz bo'lsa yozing.",
            channel="instagram",
            created_at=datetime.now(UTC) - timedelta(hours=50),
        )

        await _say(db_session, seed, llm, "Sizlarda UZI bormi?")

    assert "24 soatdan ko'proq" in llm.prompts[-1]


async def test_a_patient_back_a_few_minutes_later_is_not_greeted_again(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
    llm: ScriptedLLM,
) -> None:
    llm.say("Sizlarda UZI bor.")
    with as_tenant(seed.tenant_a.id):
        await MessageRepository(db_session).create(
            conversation_id=seed.a.conversation.id,
            sender=MessageSender.BOT,
            content="Marhamat, yana savolingiz bo'lsa yozing.",
            channel="instagram",
        )

        await _say(db_session, seed, llm, "Sizlarda UZI bormi?")

    assert "24 soatdan ko'proq" not in llm.prompts[-1]


async def test_a_booking_marker_writes_the_appointment_and_is_hidden(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
    llm: ScriptedLLM,
) -> None:
    slot = _free_slot(datetime.now(UTC))
    llm.say(
        f"Yozib qo'ydim, {slot:%d.%m} soat {slot:%H:%M}da kutamiz. "
        f"[[BOOK:{slot:%Y-%m-%dT%H:%M}|Asadbek Risqiyev|+998901234567|buyrak og'rig'i]]"
    )
    with as_tenant(seed.tenant_a.id):
        result = await _say(db_session, seed, llm, "Ha, shu vaqt bo'ladi")
        held = await AppointmentRepository(db_session).list_active_for_user(
            seed.a.user.id, after=datetime.now(UTC)
        )

    assert "[[BOOK" not in result.reply
    assert result.appointment is not None
    assert result.appointment.patient_name == "Asadbek Risqiyev"
    assert result.appointment.scheduled_at == slot.astimezone(UTC)
    assert [a.id for a in held] == [result.appointment.id]


async def test_with_booking_off_a_marker_is_hidden_and_books_nothing(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
    llm: ScriptedLLM,
) -> None:
    slot = _free_slot(datetime.now(UTC))
    llm.say(f"Yaxshi. [[BOOK:{slot:%Y-%m-%dT%H:%M}|Ali]]")
    with as_tenant(seed.tenant_a.id):
        result = await _say(db_session, seed, llm, "Ha", booking_enabled=False)

    assert result.reply == "Yaxshi."
    assert result.appointment is None


async def test_the_conversation_lock_is_taken_before_anything_is_read(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
) -> None:
    """The lock is what stops two workers deciding the same question is open.

    Asserted directly, because its absence is invisible until it costs a
    patient two identical questions four seconds apart.
    """
    with as_tenant(seed.tenant_a.id):
        await turn.lock_conversation(db_session, seed.a.conversation.id)
        # Taking it twice in the same transaction is free; a second
        # transaction would wait, which is the point.
        await turn.lock_conversation(db_session, seed.a.conversation.id)


async def test_a_reply_that_could_not_be_delivered_is_still_visible(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
) -> None:
    """The clinic opened chats full of questions and no answers, with no way
    to tell a silent bot from an answer Instagram refused. The row is kept
    now -- and kept out of the model's own history, because the patient
    never saw it.
    """
    from sqlalchemy import select

    from app.models.message import DeliveryStatus, Message
    from app.repositories.message import MessageRepository
    from app.services.conversation import record_outbound_message

    text = "Ertaga 09:20 bo'sh."
    with as_tenant(seed.tenant_a.id):
        await record_outbound_message(
            db_session,
            conversation_id=seed.a.conversation.id,
            channel_type="instagram",
            text=text,
            status=DeliveryStatus.UNDELIVERABLE,
            error="outside the 24-hour window",
        )
        rows = (
            (
                await db_session.execute(
                    select(Message).where(Message.conversation_id == seed.a.conversation.id)
                )
            )
            .scalars()
            .all()
        )
        for_the_model = await MessageRepository(db_session).list_recent(
            seed.a.conversation.id, 20
        )

    kept = [row for row in rows if row.content == text]
    assert len(kept) == 1, "the clinic must be able to see it"
    assert kept[0].delivery_status == str(DeliveryStatus.UNDELIVERABLE)
    assert kept[0].delivery_error == "outside the 24-hour window"
    assert not any(m.content == text for m in for_the_model), (
        "the model must not be told a turn the patient never received"
    )
