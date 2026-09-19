"""The conversation the clinic reported, from beginning to end.

One patient, one thread: a symptom, a name, a number, a day named a week
ahead, a time, a booking, thanks, a cancellation and a question about what
is left. Every failure this file asserts against actually happened in
production, and every one of them had the same shape -- something the
patient had already said had scrolled out of the ten turns the model is
given, so it was asked for again.

The model here is a fake with a fixed script. That is the point: what is
being tested is the backend's memory and the flow around the model, not the
model's prose.
"""

import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.appointment import AppointmentStatus
from app.models.conversation_state import CompletedAction, FlowStatus
from app.rag.embeddings import EmbeddingProvider
from app.rag.llm import ChatMessage, LLMProvider
from app.repositories.appointment import AppointmentRepository
from app.repositories.conversation_state import ConversationStateRepository
from app.services import turn
from app.services.appointment import CLINIC_TIMEZONE
from app.services.intent import Intent
from tests.conftest import Seed, isolated_settings


class ScriptedLLM(LLMProvider):
    """Says whatever the test tells it to, and records what it was told.

    The prompt it receives is the assertion surface: this is how the tests
    below check that the backend put the patient's name, their number and
    the day they chose in front of the model rather than hoping it
    remembered them.
    """

    def __init__(self) -> None:
        self.replies: list[str] = []
        self.prompts: list[str] = []

    def say(self, *replies: str) -> None:
        self.replies.extend(replies)

    async def generate(self, system_prompt: str, messages: list[ChatMessage]) -> str:
        self.prompts.append(system_prompt)
        return self.replies.pop(0) if self.replies else "Xo'p bo'ladi."

    @property
    def last_prompt(self) -> str:
        return self.prompts[-1]


class NoEmbeddings(EmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 1536 for _ in texts]


def _settings() -> object:
    return isolated_settings(
        database_url="postgresql+asyncpg://x/y",
        redis_url="redis://x",
        webhook_verify_token="t",
        meta_app_secret="s",
        encryption_key="0" * 43 + "=",
        session_secret_key="secret-secret-secret",
        openai_api_key="sk-test",
        answer_without_faq=True,
        booking_enabled=True,
        default_reply_language="Uzbek",
        clinic_phone_numbers="+998 70 310 40 40",
        doctor_name="Axmadaliyev Temur G'iyosiddin o'g'li",
        doctor_specialty="urolog-androlog",
    )


def _next_thursday(now: datetime) -> datetime:
    local = now.astimezone(CLINIC_TIMEZONE)
    ahead = (3 - local.weekday()) % 7 or 7
    return (local + timedelta(days=ahead)).replace(hour=12, minute=0, second=0, microsecond=0)


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM()


async def _say(
    session: AsyncSession,
    seed: Seed,
    llm: ScriptedLLM,
    message: str,
    *,
    history: list[ChatMessage] | None = None,
) -> turn.TurnResult:
    return await turn.respond(
        session,
        conversation_id=seed.a.conversation.id,
        user_id=seed.a.user.id,
        message=message,
        history=history or [],
        source="instagram",
        settings=_settings(),  # type: ignore[arg-type]
        llm_provider=llm,
        embedding_provider=NoEmbeddings(),
    )


# --- the reported conversation -------------------------------------------


async def test_the_whole_reported_conversation(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
    llm: ScriptedLLM,
) -> None:
    thursday = _next_thursday(datetime.now(UTC))
    history: list[ChatMessage] = []

    with as_tenant(seed.tenant_a.id):
        # 1. A symptom. This must not start collecting anything.
        llm.say("Tushundim. Doktor buni ko'rikda aniqlaydi.")
        result = await _say(db_session, seed, llm, "Buyragim og'riyapti", history=history)
        assert result.intent is Intent.MEDICAL_QUESTION
        assert "The one thing still missing" not in llm.last_prompt
        history += [
            {"role": "user", "content": "Buyragim og'riyapti"},
            {"role": "assistant", "content": result.reply},
        ]

        # 2. Now they ask to be booked, and give their name and number.
        llm.say("Ism-familiyangizni yozing.")
        await _say(db_session, seed, llm, "qabulga yozilmoqchiman", history=history)
        history += [
            {"role": "user", "content": "qabulga yozilmoqchiman"},
            {"role": "assistant", "content": "Ism-familiyangizni yozing."},
        ]

        llm.say("Rahmat. Telefon raqamingizni yozing.")
        await _say(db_session, seed, llm, "Asadbek Risqiyev", history=history)
        history += [
            {"role": "user", "content": "Asadbek Risqiyev"},
            {"role": "assistant", "content": "Rahmat. Telefon raqamingizni yozing."},
        ]

        llm.say("Qabul sababini yozing.")
        await _say(db_session, seed, llm, "93 951 11 11", history=history)
        history += [
            {"role": "user", "content": "93 951 11 11"},
            {"role": "assistant", "content": "Qabul sababini yozing."},
        ]

        llm.say("Qaysi kun qulay?")
        await _say(db_session, seed, llm, "buyrak og'rig'i", history=history)
        history += [
            {"role": "user", "content": "buyrak og'rig'i"},
            {"role": "assistant", "content": "Qaysi kun qulay?"},
        ]

        # 3. A day a week away, in words.
        llm.say("Payshanba kuni bo'sh vaqtlar bor. Soat nechada?")
        result = await _say(db_session, seed, llm, "keyingi hafta payshanba", history=history)
        state = await ConversationStateRepository(db_session).get(seed.a.conversation.id)
        assert state is not None
        assert state.requested_date == thursday.date(), "the day must be resolved and stored"
        assert FlowStatus(state.status) is FlowStatus.AWAITING_TIME
        history += [
            {"role": "user", "content": "keyingi hafta payshanba"},
            {"role": "assistant", "content": result.reply},
        ]

        # 4. Six more turns of chatter, so the name, the number and the day
        # all fall out of the ten-turn window the model is given.
        for filler in ("uzr", "yaxshi", "tushunarli", "hop", "mayli", "rahmat sizga"):
            llm.say("Albatta.")
            await _say(db_session, seed, llm, filler, history=history)
            history += [
                {"role": "user", "content": filler},
                {"role": "assistant", "content": "Albatta."},
            ]
        history = history[-10:]

        # 5. The time. The day must still be Thursday -- not tomorrow.
        llm.say("Payshanba 12:00 ga yozaymi?")
        result = await _say(db_session, seed, llm, "12:00", history=history)

        prompt = llm.last_prompt
        assert "Asadbek Risqiyev" in prompt, "the name must survive the window"
        assert "+998939511111" in prompt, "the number must survive the window"
        assert f"{thursday:%d.%m.%Y}" in prompt, "the chosen day must survive the window"
        assert result.appointment is None, "nothing is written before they confirm"

        # ...and the confirmation is what writes it.
        llm.say("Yozib qo'ydim.")
        result = await _say(db_session, seed, llm, "ha", history=history)

        assert result.appointment is not None
        assert result.appointment.scheduled_at == thursday.astimezone(UTC)

        state = await ConversationStateRepository(db_session).get(seed.a.conversation.id)
        assert state is not None
        assert FlowStatus(state.status) is FlowStatus.IDLE
        assert CompletedAction(state.last_completed_action) is CompletedAction.BOOKING_CREATED

        # 6. "Rahmat" after a finished booking is thanks, not a form.
        llm.say("Arzimaydi, salomat bo'ling.")
        result = await _say(db_session, seed, llm, "rahmat", history=history)
        assert result.intent is Intent.THANKS
        assert "THE BOOKING IS DONE" in llm.last_prompt
        assert "The one thing still missing" not in llm.last_prompt

        # 7. A cancellation: asked for, confirmed, and only then done.
        llm.say("Payshanba 12:00 dagi qabulni bekor qilaymi?")
        result = await _say(db_session, seed, llm, "qabulni bekor qilmoqchiman", history=history)
        assert result.intent is Intent.CANCEL_REQUEST
        assert result.cancelled is None, "nothing is cancelled before they confirm"
        assert "ONE appointment" in llm.last_prompt

        llm.say("Bekor qildim.")
        result = await _say(db_session, seed, llm, "ha", history=history)
        assert result.intent is Intent.CANCEL_CONFIRM
        assert result.cancelled is not None
        assert result.cancelled.status == AppointmentStatus.CANCELLED

        # 8. And what is left comes from the database.
        llm.say("Hozircha qabulingiz yo'q.")
        result = await _say(db_session, seed, llm, "boshqa qabulim bormi?", history=history)
        assert result.intent is Intent.EXISTING_BOOKING_QUERY
        assert "APPOINTMENTS THIS PATIENT ALREADY HAS" not in llm.last_prompt


# --- the pieces, on their own --------------------------------------------


async def test_a_name_given_once_is_never_asked_for_again(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
    llm: ScriptedLLM,
) -> None:
    with as_tenant(seed.tenant_a.id):
        llm.say("Rahmat.")
        await _say(
            db_session,
            seed,
            llm,
            "Asadbek Risqiyev",
            history=[{"role": "assistant", "content": "Ism-familiyangizni yozing."}],
        )

        llm.say("Telefon raqamingizni yozing.")
        await _say(db_session, seed, llm, "qabulga yozilmoqchiman", history=[])

    assert "Their name: Asadbek Risqiyev" in llm.last_prompt
    assert "still missing is their full name" not in llm.last_prompt


async def test_tepada_yozdimku_does_not_ask_again(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
    llm: ScriptedLLM,
) -> None:
    """The patient's own complaint, in their own words."""
    with as_tenant(seed.tenant_a.id):
        llm.say("Rahmat.")
        await _say(
            db_session,
            seed,
            llm,
            "Asadbek Risqiyev 93 951 11 11",
            history=[{"role": "assistant", "content": "Ism-familiyangizni yozing."}],
        )

        llm.say("Kechirasiz, raqamingiz bor ekan.")
        result = await _say(db_session, seed, llm, "tepada yozdimku", history=[])

    assert result.intent is Intent.REFERENCE_PREVIOUS
    assert "+998939511111" in llm.last_prompt
    assert "still missing is their telephone number" not in llm.last_prompt


async def test_a_medical_question_does_not_start_a_booking(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
    llm: ScriptedLLM,
) -> None:
    with as_tenant(seed.tenant_a.id):
        llm.say("Tushundim.")
        result = await _say(db_session, seed, llm, "Menda prostatada muammo bor", history=[])

    assert result.intent is Intent.MEDICAL_QUESTION
    assert "The one thing still missing" not in llm.last_prompt
    state_words = ("WHAT THE CLINIC ALREADY KNOWS",)
    assert all(word in llm.last_prompt for word in state_words)


async def test_the_model_cannot_invent_availability(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
    llm: ScriptedLLM,
) -> None:
    """A marker for a time outside working hours books nothing, and the
    patient is told the slot is gone rather than that they are booked.
    """
    tomorrow = (datetime.now(CLINIC_TIMEZONE) + timedelta(days=1)).replace(
        hour=3, minute=0, second=0, microsecond=0
    )
    with as_tenant(seed.tenant_a.id):
        llm.say(f"Ertaga 03:00 ga yozdim. [[BOOK:{tomorrow:%Y-%m-%dT%H:%M}]]")
        result = await _say(db_session, seed, llm, "ertaga 03:00 ga yozing", history=[])

    assert result.appointment is None
    assert "[[BOOK" not in result.reply


# --- several appointments at once ----------------------------------------


async def test_a_patient_with_two_appointments_is_asked_which_one(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
    llm: ScriptedLLM,
) -> None:
    """Both must exist, and neither may be cancelled on a guess."""
    first = _next_thursday(datetime.now(UTC))
    second = first + timedelta(days=7, hours=-1)

    with as_tenant(seed.tenant_a.id):
        for slot in (first, second):
            llm.say(f"Yozib qo'ydim. [[BOOK:{slot:%Y-%m-%dT%H:%M}|Asadbek Risqiyev]]")
            result = await _say(
                db_session, seed, llm, f"yana bir kunga qabulga yozing {slot:%H:%M}", history=[]
            )
            assert result.appointment is not None

        live = await AppointmentRepository(db_session).list_active_for_user(
            seed.a.user.id, after=datetime.now(UTC)
        )
        assert len(live) == 2, "a new booking must not replace the one they have"

        llm.say("Qaysi birini bekor qilay?")
        result = await _say(db_session, seed, llm, "qabulimni bekor qilmoqchiman", history=[])

        assert result.cancelled is None
        assert "SEVERAL appointments" in llm.last_prompt
        state = await ConversationStateRepository(db_session).get(seed.a.conversation.id)
        assert state is not None
        assert FlowStatus(state.status) is FlowStatus.AWAITING_CANCEL_CHOICE


# --- two messages at once -------------------------------------------------


async def test_two_messages_at_once_leave_one_consistent_state(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
    llm: ScriptedLLM,
) -> None:
    """Both turns run against the same conversation; the state that comes out
    is one of them, whole, rather than half of each.

    One session here, so this exercises the ordering and the version column
    rather than two live connections -- the advisory lock itself is what
    serialises the real pair, and it is taken inside respond().
    """
    with as_tenant(seed.tenant_a.id):
        llm.say("Ism-familiyangizni yozing.", "Telefon raqamingizni yozing.")
        await _say(db_session, seed, llm, "qabulga yozilmoqchiman", history=[])
        await _say(
            db_session,
            seed,
            llm,
            "Asadbek Risqiyev",
            history=[{"role": "assistant", "content": "Ism-familiyangizni yozing."}],
        )

        state = await ConversationStateRepository(db_session).get(seed.a.conversation.id)
        assert state is not None
        # Two writes, two versions: nothing was silently overwritten.
        assert state.version >= 2
        assert FlowStatus(state.status) is FlowStatus.COLLECTING


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


# --- the clinic's second list of complaints --------------------------------


async def test_everything_in_one_message_is_taken_apart(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
    llm: ScriptedLLM,
) -> None:
    """Name, number, reason and a time, all in one bubble."""
    thursday = _next_thursday(datetime.now(UTC))

    with as_tenant(seed.tenant_a.id):
        llm.say("Yozib qo'ydim.")
        await _say(
            db_session,
            seed,
            llm,
            f"Asadbek Risqiyev 93 951 11 11, buyrak og'rig'i, {thursday:%d}-sentabr 12:00 ga "
            "qabulga yozilmoqchiman",
            history=[],
        )

        prompt = llm.last_prompt
        assert "Asadbek Risqiyev" in prompt
        assert "+998939511111" in prompt
        state = await ConversationStateRepository(db_session).get(seed.a.conversation.id)
        assert state is not None
        assert state.requested_time is not None and state.requested_time.hour == 12


async def test_a_confirmation_cannot_name_a_day_the_booking_is_not_on(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
    llm: ScriptedLLM,
) -> None:
    """Live: "Bugun 16:20" was agreed and the confirmation said "ertaga
    16:20". The row was right and the sentence was wrong, and the sentence
    is the one the patient acts on.
    """
    today = datetime.now(CLINIC_TIMEZONE).replace(hour=16, minute=20, second=0, microsecond=0)
    if today <= datetime.now(CLINIC_TIMEZONE) + timedelta(minutes=40):
        today = today + timedelta(days=1)  # too close to now to be offered
    marker = f"[[BOOK:{today:%Y-%m-%dT%H:%M}|Asadbek Risqiyev|+998939511111|buyrak]]"

    with as_tenant(seed.tenant_a.id):
        # The model names the wrong day on purpose.
        wrong = "ertaga" if today.date() == datetime.now(CLINIC_TIMEZONE).date() else "bugun"
        llm.say(f"Yaxshi, {wrong} 16:20 ga yozib qo'ydim. {marker}")
        result = await _say(db_session, seed, llm, "16:20 bo'ladi", history=[])

    assert result.appointment is not None
    assert wrong not in result.reply.lower(), "the wrong day word must not survive"


async def test_the_language_a_patient_asks_for_is_remembered(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
    llm: ScriptedLLM,
) -> None:
    """They asked for Russian, then sent "ok", and were answered in Uzbek."""
    with as_tenant(seed.tenant_a.id):
        llm.say("Хорошо, отвечаю по-русски.")
        await _say(db_session, seed, llm, "давайте по-русски", history=[])

        llm.say("Записать вас?")
        await _say(db_session, seed, llm, "ok", history=[])

    assert "CYRILLIC" in llm.last_prompt or "Russian" in llm.last_prompt
    user = await db_session.get(__import__("app.models.user", fromlist=["User"]).User, seed.a.user.id)
    assert user is not None and user.preferred_language == "ru"


async def test_a_second_number_is_questioned_not_swapped(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
    llm: ScriptedLLM,
) -> None:
    """The clinic rings these numbers. A patient who types a second one gets
    asked which is current; the first is not silently thrown away.
    """
    from app.models.user import User

    with as_tenant(seed.tenant_a.id):
        llm.say("Rahmat.")
        await _say(
            db_session,
            seed,
            llm,
            "93 951 11 11",
            history=[{"role": "assistant", "content": "Telefon raqamingizni yozing."}],
        )

        llm.say("Qaysi raqamga qo'ng'iroq qilaylik?")
        await _say(db_session, seed, llm, "90 777 88 99 ham bor", history=[])

        user = await db_session.get(User, seed.a.user.id)

    assert user is not None and user.phone == "+998939511111", "the first number stands"
    assert "SECOND NUMBER" in llm.last_prompt


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


async def test_nothing_is_booked_until_the_patient_confirms(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
    llm: ScriptedLLM,
) -> None:
    """The clinic's rule: read the booking back, ask, and only then write it.

    The model is not trusted to decide when: even if it writes a booking
    marker while the flow is still waiting for a yes, nothing is created.
    """
    from app.models.conversation_state import FlowStatus
    from app.repositories.appointment import AppointmentRepository

    thursday = _next_thursday(datetime.now(UTC))
    marker = f"[[BOOK:{thursday:%Y-%m-%dT%H:%M}|Asadbek Risqiyev|+998939511111|uzi]]"

    with as_tenant(seed.tenant_a.id):
        llm.say("Telefon raqamingizni yozing.")
        await _say(
            db_session,
            seed,
            llm,
            "Asadbek Risqiyev, qabulga yozilmoqchiman",
            history=[{"role": "assistant", "content": "Ism-familiyangizni yozing."}],
        )
        llm.say("Sababini yozing.")
        await _say(db_session, seed, llm, "93 951 11 11", history=[])
        llm.say("Qaysi kun?")
        await _say(
            db_session,
            seed,
            llm,
            "uzi",
            history=[{"role": "assistant", "content": "Qabul sababini yozing."}],
        )
        llm.say("Soat nechada?")
        await _say(db_session, seed, llm, f"{thursday:%d}-sentabr", history=[])

        # The model jumps the gun and writes a marker before any confirmation.
        llm.say(f"Yozib qo'ydim. {marker}")
        result = await _say(db_session, seed, llm, "12:00", history=[])

        state = await ConversationStateRepository(db_session).get(seed.a.conversation.id)
        assert state is not None
        assert FlowStatus(state.status) is FlowStatus.AWAITING_CONFIRMATION
        live = await AppointmentRepository(db_session).list_active_for_user(
            seed.a.user.id, after=datetime.now(UTC)
        )
        assert live == [], "nothing may be written before they say yes"
        assert result.appointment is None

        # Now they confirm, and the backend writes it -- no marker needed.
        llm.say("Tasdiqladim.")
        result = await _say(db_session, seed, llm, "ha", history=[])

        assert result.intent is Intent.BOOKING_CONFIRM
        assert result.appointment is not None
        assert "IS NOW IN THE DIARY" in llm.last_prompt


async def test_a_day_the_clinic_is_shut_is_refused_when_it_is_named(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
    llm: ScriptedLLM,
) -> None:
    """Live: on a Saturday, "ertaga 15:00" is Sunday. It was accepted, read
    back in a summary, confirmed, and the patient was told they were booked
    on a day nobody is at the clinic.
    """
    from app.models.conversation_state import FlowStatus

    sunday = datetime.now(CLINIC_TIMEZONE).date()
    while sunday.weekday() != 6:
        sunday += timedelta(days=1)

    with as_tenant(seed.tenant_a.id):
        llm.say("Yakshanba dam olish kuni.")
        await _say(
            db_session,
            seed,
            llm,
            f"{sunday:%d}-{'sentabr' if sunday.month == 9 else 'oktabr'} 15:00 ga yozing",
            history=[],
        )

        state = await ConversationStateRepository(db_session).get(seed.a.conversation.id)
        assert state is not None
        assert state.requested_date != sunday, "a closed day must not enter the state"
        assert FlowStatus(state.status) is not FlowStatus.AWAITING_CONFIRMATION
        assert "CLOSED" in llm.last_prompt


async def test_a_reply_that_claims_a_booking_nobody_made_is_not_sent(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
    llm: ScriptedLLM,
) -> None:
    """The failure that ends with somebody at a locked door: the assistant
    said "tasdiqlayman — yozildingiz" about a row that was never written.
    """
    with as_tenant(seed.tenant_a.id):
        # It claims a booking twice; the second is the rewrite it is given.
        llm.say(
            "Tasdiqlayman — ertaga soat 15:00ga yozildingiz.",
            "Yozib qo'ydim, kutamiz.",
        )
        result = await _say(db_session, seed, llm, "ha", history=[])

    assert result.appointment is None
    assert "yozildingiz" not in result.reply
    assert "Yozib qo'ydim" not in result.reply
    # The last resort: a fixed, honest line rather than a false one.
    assert "yozib bo'lmadi" in result.reply
