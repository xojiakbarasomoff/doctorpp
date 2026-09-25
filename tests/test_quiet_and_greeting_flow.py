"""The whole path, worker to Instagram: who gets greeted, who gets silence,
and what a video, a file or a shared reel is answered with."""

from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager, asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation
from app.models.lead import Lead
from app.models.message import Message
from app.rag.llm import ChatMessage, LLMProvider
from app.repositories.message import MessageRepository
from app.services import message_labels, voice_notes
from app.services.patient_media import ACKNOWLEDGEMENTS
from app.workers.tasks import (
    SHARE_QUESTIONS,
    answer_attachment,
    answer_voice_note,
    process_inbound_message,
)
from tests.conftest import Seed
from tests.test_worker_tasks import QUERY_VECTOR, FakeEmbeddingProvider, _fake_adapter

AsTenant = Callable[[UUID], AbstractContextManager[None]]
SENDER = "sender-1"


class FakeLLM(LLMProvider):
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, list[ChatMessage]]] = []

    async def generate(self, system_prompt: str, messages: list[ChatMessage]) -> str:
        self.calls.append((system_prompt, messages))
        return self.reply


class FakeRedis:
    def __init__(self) -> None:
        self.keys: set[str] = set()

    async def set(self, key: str, value: str, *, ex: int, nx: bool) -> bool:
        if key in self.keys:
            return False
        self.keys.add(key)
        return True


def _factory(session: AsyncSession) -> Callable[[], AbstractAsyncContextManager[AsyncSession]]:
    @asynccontextmanager
    async def _session() -> AsyncIterator[AsyncSession]:
        yield session

    return _session


async def _conversation(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: AsTenant,
    lines: Sequence[tuple[str, str]],
    *,
    hours_ago: float = 0.1,
) -> None:
    """Replace the seeded conversation with these lines, oldest first; the
    last one is the patient's message being answered, sent just now."""
    await db_session.delete(seed.a.message)
    await db_session.flush()
    start = datetime.now(UTC) - timedelta(hours=hours_ago)
    with as_tenant(seed.tenant_a.id):
        for position, (sender, content) in enumerate(lines[:-1]):
            await MessageRepository(db_session).create(
                conversation_id=seed.a.conversation.id,
                sender=sender,
                content=content,
                channel="instagram",
                created_at=start + timedelta(seconds=position),
            )
        sender, content = lines[-1]
        await MessageRepository(db_session).create(
            conversation_id=seed.a.conversation.id,
            sender=sender,
            content=content,
            channel="instagram",
            created_at=datetime.now(UTC),
        )
    await db_session.flush()


async def _answer(
    db_session: AsyncSession, seed: Seed, message: str, llm: FakeLLM
) -> list[tuple[str, str, str]]:
    adapter, client = _fake_adapter()
    await process_inbound_message(
        {},
        str(seed.tenant_a.id),
        str(seed.a.channel.id),
        str(seed.a.conversation.id),
        SENDER,
        message,
        session_factory=_factory(db_session),
        embedding_provider=FakeEmbeddingProvider(QUERY_VECTOR),
        llm_provider=llm,
        adapter=adapter,
    )
    return client.calls


# --- silence ---------------------------------------------------------------------


@pytest.mark.parametrize("message", ["👍", "ok", "Rahmat katta", "❤️"])
async def test_a_thumbs_up_after_we_are_done_gets_no_reply_and_costs_nothing(
    db_session: AsyncSession, seed: Seed, as_tenant: AsTenant, message: str
) -> None:
    await _conversation(
        db_session,
        seed,
        as_tenant,
        [
            ("patient", "Qabulga yozing"),
            ("bot", "Ertaga soat 10:00 da kutamiz."),
            ("patient", message),
        ],
    )
    llm = FakeLLM("Eshitaman.")

    sent = await _answer(db_session, seed, message, llm)

    assert sent == []
    assert llm.calls == []
    conversation = await db_session.get(Conversation, seed.a.conversation.id)
    assert conversation is not None and conversation.needs_doctor_since is None


async def test_silence_writes_down_no_callback_from_earlier_words(
    db_session: AsyncSession, seed: Seed, as_tenant: AsTenant
) -> None:
    await _conversation(
        db_session,
        seed,
        as_tenant,
        [
            ("patient", "Menga qo'ng'iroq qiling +998901234567"),
            ("bot", "Albatta, doktor sizga qo'ng'iroq qiladi."),
            ("patient", "rahmat"),
        ],
    )
    leads = select(Lead.id).where(Lead.conversation_id == seed.a.conversation.id)
    before = list((await db_session.execute(leads)).scalars())

    await _answer(db_session, seed, "rahmat", FakeLLM("x"))

    assert list((await db_session.execute(leads)).scalars()) == before


async def test_a_thumbs_up_to_a_question_is_answered_as_a_yes(
    db_session: AsyncSession, seed: Seed, as_tenant: AsTenant
) -> None:
    await _conversation(
        db_session,
        seed,
        as_tenant,
        [
            ("patient", "Qabulga yozing"),
            ("bot", "Ertaga soat 10:00 sizga qulaymi?"),
            ("patient", "👍"),
        ],
    )
    llm = FakeLLM("Yaxshi, ertaga 10:00 ga yozdim.")

    sent = await _answer(db_session, seed, "👍", llm)

    assert [text for _, _, text in sent] == ["Yaxshi, ertaga 10:00 ga yozdim."]
    [(system_prompt, _)] = llm.calls
    assert "ha, roziman" in system_prompt


# --- greetings -----------------------------------------------------------------


async def test_a_patient_mid_conversation_is_not_greeted_or_introduced_to_again(
    db_session: AsyncSession, seed: Seed, as_tenant: AsTenant
) -> None:
    await _conversation(
        db_session,
        seed,
        as_tenant,
        [
            ("patient", "Assalomu alaykum"),
            ("bot", "Va alaykum assalom, men doktorning administratoriman. Eshitaman."),
            ("patient", "Doktor manda hojatga chiqishda muammo bor"),
        ],
    )
    llm = FakeLLM(
        "Va alaykum assalom, Assalomu alaykum, man urolog-androlog Temur Axmadaliyevning "
        "yordamchilari bo'laman, tushundim. Qachondan beri?"
    )

    sent = await _answer(db_session, seed, "Doktor manda hojatga chiqishda muammo bor", llm)

    assert [text for _, _, text in sent] == ["Tushundim. Qachondan beri?"]
    with as_tenant(seed.tenant_a.id):
        recorded = await MessageRepository(db_session).list_recent(seed.a.conversation.id, 1)
    assert recorded[-1].content == "Tushundim. Qachondan beri?"


async def test_a_first_message_without_a_salom_is_greeted_not_greeted_back(
    db_session: AsyncSession, seed: Seed, as_tenant: AsTenant
) -> None:
    await _conversation(db_session, seed, as_tenant, [("patient", "Тимур яхшимисиз")])
    llm = FakeLLM("Ва алейкум ассалом. Сизга қандай ёрдам бера оламан?")

    sent = await _answer(db_session, seed, "Тимур яхшимисиз", llm)

    assert [text for _, _, text in sent] == ["Ассалому алайкум. Сизга қандай ёрдам бера оламан?"]


async def test_a_salom_mid_conversation_is_always_returned(
    db_session: AsyncSession, seed: Seed, as_tenant: AsTenant
) -> None:
    await _conversation(
        db_session,
        seed,
        as_tenant,
        [("patient", "Qabul kerak"), ("bot", "Qaysi kun qulay?"), ("patient", "Assalomu alaykum")],
    )
    llm = FakeLLM("Xo'p, qaysi kun qulay?")

    sent = await _answer(db_session, seed, "Assalomu alaykum", llm)

    assert [text for _, _, text in sent] == ["Va alaykum assalom. Xo'p, qaysi kun qulay?"]


async def test_back_after_a_day_the_greeting_is_kept(
    db_session: AsyncSession, seed: Seed, as_tenant: AsTenant
) -> None:
    await _conversation(
        db_session,
        seed,
        as_tenant,
        [("patient", "Qabul kerak"), ("bot", "Ertaga kutamiz."), ("patient", "Yana savolim bor")],
        hours_ago=30,
    )
    llm = FakeLLM("Assalomu alaykum! Eshitaman.")

    sent = await _answer(db_session, seed, "Yana savolim bor", llm)

    assert [text for _, _, text in sent] == ["Assalomu alaykum! Eshitaman."]


async def test_an_emoji_that_opens_a_conversation_asks_for_a_short_hello(
    db_session: AsyncSession, seed: Seed, as_tenant: AsTenant
) -> None:
    await _conversation(db_session, seed, as_tenant, [("patient", "👍")])
    llm = FakeLLM("Assalomu alaykum! Qanday yordam bera olaman?")

    sent = await _answer(db_session, seed, "👍", llm)

    assert [text for _, _, text in sent] == ["Assalomu alaykum! Qanday yordam bera olaman?"]
    assert "salomlashing" in llm.calls[0][0]


# --- videos, files, shared posts, voice notes ---------------------------------------


async def _attachment(
    db_session: AsyncSession, seed: Seed, kind: str, redis: FakeRedis | None = None
) -> list[tuple[str, str, str]]:
    adapter, client = _fake_adapter()
    import app.services.patient_media as patient_media

    original = patient_media.send_reply

    async def _send(*args: Any, **kwargs: Any) -> Any:
        return await original(*args, **{**kwargs, "adapter": adapter})

    patient_media.send_reply = _send  # type: ignore[assignment]
    try:
        await answer_attachment(
            {"redis": redis or FakeRedis()},
            str(seed.tenant_a.id),
            str(seed.a.channel.id),
            str(seed.a.conversation.id),
            SENDER,
            kind,
            session_factory=_factory(db_session),
            adapter=adapter,
        )
    finally:
        patient_media.send_reply = original  # type: ignore[assignment]
    return client.calls


@pytest.mark.parametrize("kind", [message_labels.VIDEO, message_labels.FILE])
async def test_a_video_or_a_file_waits_on_the_doctor_and_is_acknowledged(
    db_session: AsyncSession, seed: Seed, as_tenant: AsTenant, kind: str
) -> None:
    await _conversation(
        db_session, seed, as_tenant, [("patient", "Salom"), ("patient", message_labels.TEXT[kind])]
    )

    sent = await _attachment(db_session, seed, kind)

    assert [text for _, _, text in sent] == [ACKNOWLEDGEMENTS["uz-latn"]]
    conversation = await db_session.get(Conversation, seed.a.conversation.id)
    assert conversation is not None and conversation.needs_doctor_since is not None


async def test_a_second_video_in_the_same_burst_is_not_acknowledged_twice(
    db_session: AsyncSession, seed: Seed, as_tenant: AsTenant
) -> None:
    await _conversation(db_session, seed, as_tenant, [("patient", "🎞 Video yubordi")])
    redis = FakeRedis()

    first = await _attachment(db_session, seed, message_labels.VIDEO, redis)
    second = await _attachment(db_session, seed, message_labels.VIDEO, redis)

    assert len(first) == 1 and second == []


async def test_a_shared_reel_gets_one_question_a_day(
    db_session: AsyncSession, seed: Seed, as_tenant: AsTenant
) -> None:
    await _conversation(
        db_session,
        seed,
        as_tenant,
        [("patient", "Здравствуйте"), ("patient", "🔗 Post yoki reels ulashdi")],
    )

    first = await _attachment(db_session, seed, message_labels.SHARE)
    second = await _attachment(db_session, seed, message_labels.SHARE)

    assert [text for _, _, text in first] == [SHARE_QUESTIONS["ru"]]
    assert second == []
    conversation = await db_session.get(Conversation, seed.a.conversation.id)
    assert conversation is not None and conversation.needs_doctor_since is None


async def test_nothing_is_said_where_staff_have_taken_over_but_a_video_still_pins(
    db_session: AsyncSession, seed: Seed, as_tenant: AsTenant
) -> None:
    await _conversation(db_session, seed, as_tenant, [("patient", "🎞 Video yubordi")])
    conversation = await db_session.get(Conversation, seed.a.conversation.id)
    assert conversation is not None
    conversation.is_bot_enabled = False
    await db_session.flush()

    video = await _attachment(db_session, seed, message_labels.VIDEO)
    share = await _attachment(db_session, seed, message_labels.SHARE)

    assert video == [] and share == []
    assert conversation.needs_doctor_since is not None


@pytest.mark.parametrize("kind", [message_labels.STICKER, message_labels.STORY, "bogus"])
async def test_nothing_else_is_ever_answered(
    db_session: AsyncSession, seed: Seed, as_tenant: AsTenant, kind: str
) -> None:
    await _conversation(db_session, seed, as_tenant, [("patient", "🏷 Stiker yubordi")])

    assert await _attachment(db_session, seed, kind) == []


async def test_the_voice_note_explanation_is_given_once_a_day(
    db_session: AsyncSession, seed: Seed, as_tenant: AsTenant
) -> None:
    await _conversation(
        db_session,
        seed,
        as_tenant,
        [
            ("patient", "🎤 Ovozli xabar"),
            ("bot", voice_notes.REPLIES["uz-latn"]),
            ("patient", "Qabul kerak"),
            ("bot", "Qaysi kun qulay?"),
            ("patient", "🎤 Ovozli xabar"),
        ],
        hours_ago=2,
    )
    adapter, client = _fake_adapter()

    await answer_voice_note(
        {},
        str(seed.tenant_a.id),
        str(seed.a.channel.id),
        str(seed.a.conversation.id),
        SENDER,
        session_factory=_factory(db_session),
        adapter=adapter,
    )

    assert client.calls == []


async def test_a_reaction_does_not_reopen_the_reply_window(
    db_session: AsyncSession, seed: Seed, as_tenant: AsTenant
) -> None:
    await _conversation(
        db_session,
        seed,
        as_tenant,
        [("patient", "Qabul kerak"), ("bot", "Kutamiz."), ("patient", "💟 Reaksiya bildirdi: ❤️")],
        hours_ago=30,
    )
    with as_tenant(seed.tenant_a.id):
        last = await MessageRepository(db_session).last_inbound_at(seed.a.conversation.id)
    assert last is not None and last < datetime.now(UTC) - timedelta(hours=29)
    rows = (
        await db_session.execute(
            select(Message.content).where(Message.conversation_id == seed.a.conversation.id)
        )
    ).scalars()
    assert any(r.startswith("💟") for r in rows)
