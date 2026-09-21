from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models.doctor import Doctor
from app.rag.embeddings import EMBEDDING_DIMENSIONS, EmbeddingProvider
from app.rag.llm import ChatMessage, LLMProvider
from app.repositories.knowledge_base import KnowledgeBaseRepository
from app.services.answer import _build_system_prompt, generate_answer
from tests.conftest import Seed, isolated_settings

# A fixed, non-zero direction. Distance to itself is 0.0, so any FAQ seeded
# with this embedding is a guaranteed match for a FakeEmbeddingProvider that
# returns it.
QUERY_VECTOR = [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, vector: list[float]) -> None:
        self._vector = vector
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self._vector for _ in texts]


class FakeLLMProvider(LLMProvider):
    def __init__(self, reply: str = "Sure, here's the answer.") -> None:
        self._reply = reply
        self.calls: list[tuple[str, list[ChatMessage]]] = []

    async def generate(self, system_prompt: str, messages: list[ChatMessage]) -> str:
        self.calls.append((system_prompt, messages))
        return self._reply


async def _make_faq(db_session: AsyncSession, question: str, answer: str) -> None:
    await KnowledgeBaseRepository(db_session).create(
        question=question, answer=answer, embedding=QUERY_VECTOR
    )


def _settings(**overrides: object) -> Settings:
    return isolated_settings(**overrides)


async def _ask(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    message: str,
    *,
    reply: str = "Sure, here's the answer.",
    with_faq: bool = False,
    doctors: Sequence[tuple[str, str, str]] = (),
    history: Sequence[ChatMessage] | None = None,
    **settings_overrides: object,
) -> tuple[str, FakeLLMProvider]:
    llm_provider = FakeLLMProvider(reply=reply)
    with as_tenant(seed.tenant_a.id):
        if with_faq:
            await _make_faq(db_session, "What are your hours?", "9 to 5, Mon-Sat.")
        for name, specialty, hours in doctors:
            db_session.add(
                Doctor(
                    tenant_id=seed.tenant_a.id,
                    name=name,
                    specialty=specialty,
                    working_hours=hours,
                    is_active=True,
                )
            )
        await db_session.flush()
        result = await generate_answer(
            db_session,
            message,
            embedding_provider=FakeEmbeddingProvider(QUERY_VECTOR),
            llm_provider=llm_provider,
            settings=_settings(**settings_overrides),
            history=history,
        )
    return result, llm_provider


async def test_faq_context_reaches_the_model_and_its_reply_is_returned(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    result, llm = await _ask(
        db_session,
        seed,
        as_tenant,
        "What time do you open?",
        reply="We're open 9 to 5, Monday to Saturday.",
        with_faq=True,
    )

    assert result == "We're open 9 to 5, Monday to Saturday."
    system_prompt, messages = llm.calls[0]
    assert "Q: What are your hours?" in system_prompt
    assert "A: 9 to 5, Mon-Sat." in system_prompt
    assert messages == [{"role": "user", "content": "What time do you open?"}]


async def test_the_model_answers_even_when_no_faq_matches(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    result, llm = await _ask(db_session, seed, as_tenant, "Do you offer teeth whitening?")

    assert result == "Sure, here's the answer."
    assert len(llm.calls) == 1
    assert "Clinic information:" not in llm.calls[0][0]


@pytest.mark.parametrize(
    "message",
    ["/start", "/help", "qon ketyapti, nima qilay?", "qanaqa dori ichay?", "narxi qancha?"],
)
async def test_every_message_goes_to_the_model(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    message: str,
) -> None:
    """No canned answers: buttons, emergencies, medicine and price questions
    are all the model's to answer."""
    result, llm = await _ask(db_session, seed, as_tenant, message, reply="model reply")

    assert result == "model reply"
    assert len(llm.calls) == 1


async def test_the_reply_is_sent_exactly_as_the_model_wrote_it(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    written = (
        "Va alaykum assalom! Tushundim. Kuniga 2 mahal ichib turing? Yana nima? "
        "Ismingizni yozing? Raqamingizni yozing? " + "Uzun matn. " * 40
    )
    result, llm = await _ask(db_session, seed, as_tenant, "Salom", reply=written)

    assert result == written
    assert len(llm.calls) == 1


async def test_history_is_passed_to_the_model(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    history: list[ChatMessage] = [
        {"role": "user", "content": "Salom"},
        {"role": "assistant", "content": "Va alaykum assalom"},
    ]
    _, llm = await _ask(db_session, seed, as_tenant, "Narxi qancha?", history=history)

    assert llm.calls[0][1] == [*history, {"role": "user", "content": "Narxi qancha?"}]


async def test_clinic_details_and_doctors_are_given_to_the_model(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    _, llm = await _ask(
        db_session,
        seed,
        as_tenant,
        "Salom",
        doctors=[("Dr. Karimova N.S.", "Urolog", "09:00 - 17:00")],
        clinic_address="Toshkent, Chilonzor 1",
        clinic_phone_numbers="+998 71 200 03 93",
        clinic_work_hours="Du-Sh 09:00-18:00",
        default_reply_language="Uzbek",
    )

    prompt = llm.calls[0][0]
    assert "Address: Toshkent, Chilonzor 1" in prompt
    assert "Phone: +998 71 200 03 93" in prompt
    assert "Open: Du-Sh 09:00-18:00" in prompt
    assert "- Dr. Karimova N.S. — Urolog — 09:00 - 17:00" in prompt
    assert "reply in Uzbek" in prompt


async def test_the_appointment_book_is_given_only_when_booking_is_on(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    _, off = await _ask(db_session, seed, as_tenant, "Salom")
    _, on = await _ask(db_session, seed, as_tenant, "Salom", booking_enabled=True)

    assert "THE APPOINTMENT BOOK" not in off.calls[0][0]
    assert "THE APPOINTMENT BOOK" in on.calls[0][0]
    assert "[[BOOK:" in on.calls[0][0]


def _prompt_for(**kwargs: object) -> str:
    return _build_system_prompt(
        [],
        default_language="Uzbek",
        clinic_phone_numbers=None,
        clinic_address=None,
        **kwargs,  # type: ignore[arg-type]
    )


def test_a_doctors_inbox_speaks_as_that_doctors_administrator() -> None:
    prompt = _prompt_for(doctor_name="Axmadaliyev Temur", doctor_specialty="urolog-androlog")

    assert "Axmadaliyev Temur, urolog-androlog" in prompt
    assert "administrator" in prompt
    assert "front desk" not in prompt


@pytest.mark.parametrize(
    "persona",
    [{}, {"doctor_name": "Axmadaliyev Temur"}, {"doctor_specialty": "urolog-androlog"}],
)
def test_without_a_whole_doctor_the_clinic_voice_is_kept(persona: dict[str, str]) -> None:
    assert "front desk" in _prompt_for(**persona)


def test_a_brace_in_the_doctors_name_is_not_a_template_field() -> None:
    prompt = _prompt_for(doctor_name="Dr {x}", doctor_specialty="urolog")

    assert "Dr {x}, urolog" in prompt


def test_the_clinics_own_instructions_reach_the_prompt() -> None:
    prompt = _prompt_for(clinic_rules=["Shanba kuni ham ishlaymiz"])

    assert "- Shanba kuni ham ishlaymiz" in prompt
