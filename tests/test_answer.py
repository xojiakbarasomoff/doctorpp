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
from app.services.answer import (
    _build_system_prompt,
    _clinic_facts_block,
    _doctor_roster,
    HELP_RESPONSES,
    NO_MATCH_RESPONSE,
    NO_MATCH_RESPONSES,
    START_RESPONSES,
    client_command_response,
    generate_answer,
    is_client_command,
)
from app.services.conversation_signals import read_signals
from app.services.guardrail import EMERGENCY_RESPONSES
from tests.conftest import Seed, isolated_settings

# A fixed, non-zero direction. Distance to itself is 0.0, well inside
# DEFAULT_MAX_DISTANCE whatever it is set to, so any FAQ seeded with this
# embedding is a guaranteed match for a FakeEmbeddingProvider that returns it.
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


# --- normal flow: answer grounded in FAQ context ---


async def test_generate_answer_uses_faq_context_and_returns_llm_output(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    embedding_provider = FakeEmbeddingProvider(QUERY_VECTOR)
    llm_provider = FakeLLMProvider(reply="We're open 9 to 5, Monday to Saturday.")

    with as_tenant(seed.tenant_a.id):
        await _make_faq(db_session, "What are your hours?", "9 to 5, Mon-Sat.")

        result = await generate_answer(
            db_session,
            "What time do you open?",
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
        )

    assert result == "We're open 9 to 5, Monday to Saturday."
    assert embedding_provider.calls == [["What time do you open?"]]
    assert len(llm_provider.calls) == 1
    system_prompt, messages = llm_provider.calls[0]
    assert "Q: What are your hours?" in system_prompt
    assert "A: 9 to 5, Mon-Sat." in system_prompt
    assert messages == [{"role": "user", "content": "What time do you open?"}]


# --- empty retrieval: code-level short-circuit, LLM never called ---


async def test_generate_answer_returns_fixed_response_when_no_faq_matches(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    # No FAQ seeded with QUERY_VECTOR for this tenant, so search() finds
    # nothing within the default distance threshold (the `seed` fixture's
    # own placeholder FAQ has a zero-vector embedding, which is NaN distance
    # from any real query vector and gets filtered out).
    embedding_provider = FakeEmbeddingProvider(QUERY_VECTOR)
    llm_provider = FakeLLMProvider()

    with as_tenant(seed.tenant_a.id):
        result = await generate_answer(
            db_session,
            "Do you offer teeth whitening?",
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
        )

    assert result == NO_MATCH_RESPONSE
    assert llm_provider.calls == []


def _settings(**overrides: object) -> Settings:
    return isolated_settings(**overrides)


async def test_no_faq_match_asks_the_llm_when_answering_without_faq_is_enabled(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """ANSWER_WITHOUT_FAQ is what a deployment turns on before its knowledge
    base is populated, so that patients get a real reply instead of the same
    refusal to every message.
    """
    embedding_provider = FakeEmbeddingProvider(QUERY_VECTOR)
    llm_provider = FakeLLMProvider()

    with as_tenant(seed.tenant_a.id):
        result = await generate_answer(
            db_session,
            "Do you offer teeth whitening?",
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            settings=_settings(answer_without_faq=True),
        )

    assert result != NO_MATCH_RESPONSE
    assert len(llm_provider.calls) == 1


async def test_answering_without_faq_still_forbids_clinic_details_and_medical_advice(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """The relaxed path drops "answer only from the FAQ" and nothing else:
    an assistant free to improvise opening hours or medication is the
    failure this setting must not introduce.
    """
    embedding_provider = FakeEmbeddingProvider(QUERY_VECTOR)
    llm_provider = FakeLLMProvider()

    with as_tenant(seed.tenant_a.id):
        await generate_answer(
            db_session,
            "Do you offer teeth whitening?",
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            settings=_settings(answer_without_faq=True),
        )

    system_prompt = llm_provider.calls[0][0]
    assert "opening hours, prices" in system_prompt
    assert "Never state or guess any of" in system_prompt
    assert "You are not a medical professional" in system_prompt
    assert "Never claim or imply that you are a doctor" in system_prompt


async def test_disabled_by_default_so_an_unset_variable_cannot_relax_it(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    embedding_provider = FakeEmbeddingProvider(QUERY_VECTOR)
    llm_provider = FakeLLMProvider()

    with as_tenant(seed.tenant_a.id):
        result = await generate_answer(
            db_session,
            "Do you offer teeth whitening?",
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            settings=_settings(),
        )

    assert result == NO_MATCH_RESPONSE
    assert llm_provider.calls == []


async def test_default_reply_language_reaches_the_prompt(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """Patients open with "Salom", "Alik", "Nmagap" -- too short and too
    transliterated for the model to place, so it answers in English and the
    clinic looks like it is replying in the wrong language. The configured
    language is what it falls back to instead.
    """
    embedding_provider = FakeEmbeddingProvider(QUERY_VECTOR)
    llm_provider = FakeLLMProvider()

    with as_tenant(seed.tenant_a.id):
        await _make_faq(db_session, "What are your hours?", "9 to 5, Mon-Sat.")
        await generate_answer(
            db_session,
            "Nmagap",
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            settings=_settings(default_reply_language="Uzbek"),
        )

    system_prompt = llm_provider.calls[0][0]
    assert "reply in Uzbek" in system_prompt
    # The instruction to mirror the patient still comes first: the fallback
    # applies only when the language is unclear, it does not override a
    # message that plainly is in another language.
    assert system_prompt.index("same language the patient wrote in") < system_prompt.index(
        "reply in Uzbek"
    )


async def test_default_language_also_applies_without_a_faq_match(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    # The no-FAQ path is the one a not-yet-populated deployment actually
    # runs, so the language fallback has to be in that prompt too.
    embedding_provider = FakeEmbeddingProvider(QUERY_VECTOR)
    llm_provider = FakeLLMProvider()

    with as_tenant(seed.tenant_a.id):
        await generate_answer(
            db_session,
            "Nmagap",
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            settings=_settings(answer_without_faq=True, default_reply_language="Uzbek"),
        )

    assert "reply in Uzbek" in llm_provider.calls[0][0]


# --- how the reply must sound: alphabet, greeting, and the call-centre ask ---


async def _capture_system_prompt(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    *,
    with_faq: bool,
    doctors: Sequence[tuple[str, str, str]] = (),
    **settings_overrides: object,
) -> str:
    """Run generate_answer far enough to grab the system prompt it built.

    Every rule below is checked on both paths on purpose: a clinic whose
    knowledge base is not populated yet runs the no-FAQ prompt for every
    single message, so a rule that only made it into the FAQ prompt would be
    missing exactly where nobody would think to look for it.
    """
    embedding_provider = FakeEmbeddingProvider(QUERY_VECTOR)
    llm_provider = FakeLLMProvider()

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
        await generate_answer(
            db_session,
            "Assalom alaykum",
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            settings=_settings(answer_without_faq=not with_faq, **settings_overrides),
        )

    return llm_provider.calls[0][0]


@pytest.mark.parametrize("with_faq", [True, False], ids=["with_faq", "without_faq"])
async def test_prompt_requires_replying_in_the_alphabet_the_patient_used(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    with_faq: bool,
) -> None:
    """Uzbek is written in both Latin and Cyrillic, and "same language" alone
    does not settle which one to answer in -- a Cyrillic patient answered in
    Latin has been replied to in their language and still cannot comfortably
    read it.
    """
    system_prompt = await _capture_system_prompt(db_session, seed, as_tenant, with_faq=with_faq)

    assert "same alphabet they" in system_prompt
    assert "answer a Cyrillic message in Cyrillic and a Latin message in Latin" in system_prompt
    assert "Never transliterate a patient into the" in system_prompt


@pytest.mark.parametrize("with_faq", [True, False], ids=["with_faq", "without_faq"])
async def test_prompt_carries_the_expected_greeting_for_each_language(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    with_faq: bool,
) -> None:
    """The greeting "Assalom alaykum" has one correct answer, and a model left
    to its own devices returns "Salom" or the English "Hello" instead -- which
    reads, to the patient, as a clinic that did not greet them back.
    """
    system_prompt = await _capture_system_prompt(db_session, seed, as_tenant, with_faq=with_faq)

    assert "Va alaykum assalom" in system_prompt
    assert "Ва алайкум ассалом" in system_prompt
    assert "Здравствуйте" in system_prompt


@pytest.mark.parametrize("with_faq", [True, False], ids=["with_faq", "without_faq"])
async def test_prompt_does_not_ask_for_the_patients_number(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    with_faq: bool,
) -> None:
    """It used to, and asking was the point: a call-centre colleague picked
    the number up from here. The clinic books by telephone now, so the two
    requests compete -- told "ring us" and "leave your number" in one reply,
    a patient cannot tell which is actually going to happen, and production
    was doing exactly that to somebody writing about three years of
    infertility.
    """
    system_prompt = await _capture_system_prompt(db_session, seed, as_tenant, with_faq=with_faq)

    assert "Do not ask the patient for their telephone number" in system_prompt
    assert "reads as a runaround" in system_prompt
    # The one case that still needs it: they said they cannot ring.
    assert "they say plainly that they cannot ring" in system_prompt
    assert "offering the reason rather than the demand" in system_prompt


async def test_no_match_response_asks_for_a_phone_number(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """This path never reaches the LLM, so rule 4 in the prompt cannot apply
    to it -- and it is the case that needs a callback most, since the clinic
    has just failed to answer the patient at all.
    """
    embedding_provider = FakeEmbeddingProvider(QUERY_VECTOR)
    llm_provider = FakeLLMProvider()

    with as_tenant(seed.tenant_a.id):
        result = await generate_answer(
            db_session,
            "Buyrak toshini olasizmi?",
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            settings=_settings(),
        )

    assert result == NO_MATCH_RESPONSE
    assert llm_provider.calls == []
    assert "raqamingizni" in NO_MATCH_RESPONSE


async def test_the_no_match_line_is_written_in_the_patients_own_script(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """This path never reaches the model, so nothing downstream can put the
    reply back into the patient's language. It used to answer everyone in
    English, which is the most machine-like thing the assistant did.
    """
    embedding_provider = FakeEmbeddingProvider(QUERY_VECTOR)
    llm_provider = FakeLLMProvider()

    async def ask(message: str) -> str:
        with as_tenant(seed.tenant_a.id):
            return await generate_answer(
                db_session,
                message,
                embedding_provider=embedding_provider,
                llm_provider=llm_provider,
                settings=_settings(),
            )

    assert await ask("Сколько стоит приём уролога?") == NO_MATCH_RESPONSES["ru"]
    assert await ask("Буйрагим оғрияпти") == NO_MATCH_RESPONSES["uz-cyrl"]
    assert await ask("Qabulga yozilsam bo'ladimi?") == NO_MATCH_RESPONSES["uz-latn"]
    # None of the three asked the model anything.
    assert llm_provider.calls == []


# --- what the clinic does and does not offer, and what a price costs ---


async def test_faq_prompt_answers_service_questions_instead_of_deflecting(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """A question like "do you do implants?" is a yes/no question. Answering
    it with a booking offer reads as evasion, and the patient asks a
    competitor instead.
    """
    system_prompt = await _capture_system_prompt(db_session, seed, as_tenant, with_faq=True)

    assert "say plainly that yes, it is available" in system_prompt
    assert "Afsuski, bizda bunaqa xizmat hozircha yo'q" in system_prompt


async def test_faq_prompt_will_not_call_a_service_unavailable_on_silence(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """Retrieval returns the nearest FAQ entries, not the clinic's full
    service list, so a treatment missing from the context is unknown rather
    than absent. Announcing "we don't do that" from silence turns away a
    patient for a treatment the clinic may well perform.
    """
    system_prompt = await _capture_system_prompt(db_session, seed, as_tenant, with_faq=True)

    assert "does not mention the treatment either way, you do not know" in system_prompt
    assert "so do not say it is unavailable" in system_prompt


async def test_no_faq_prompt_claims_nothing_about_services_in_either_direction(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """With no knowledge base at all, every service question is the unknown
    case -- so this path must never confirm a treatment either.
    """
    system_prompt = await _capture_system_prompt(db_session, seed, as_tenant, with_faq=False)

    assert "never tell a patient that it does, and never tell" in system_prompt
    assert "Afsuski, bizda bunaqa xizmat hozircha yo'q" not in system_prompt


@pytest.mark.parametrize("with_faq", [True, False], ids=["with_faq", "without_faq"])
async def test_prompt_forbids_inventing_somewhere_else_to_go(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    with_faq: bool,
) -> None:
    """The no-FAQ path is the dangerous one here: it is explicitly allowed to
    use general knowledge, which is exactly what would produce a confident,
    fictional referral to a clinic across town.
    """
    system_prompt = await _capture_system_prompt(db_session, seed, as_tenant, with_faq=with_faq)

    assert "must NOT name another clinic" in system_prompt
    assert "Afsuski, bizda bunday ma'lumot yo'q" in system_prompt


@pytest.mark.parametrize("with_faq", [True, False], ids=["with_faq", "without_faq"])
async def test_configured_clinic_numbers_are_quoted_in_the_pricing_fallback(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    with_faq: bool,
) -> None:
    system_prompt = await _capture_system_prompt(
        db_session,
        seed,
        as_tenant,
        with_faq=with_faq,
        clinic_phone_numbers="+998 90 123 45 67",
    )

    assert "+998 90 123 45 67" in system_prompt
    assert "ushbu telefon raqamlariga qo'ng'iroq qiling" in system_prompt
    # The callback offer is not an either/or with the numbers: a patient who
    # is writing rather than calling is the one this whole rule exists for.
    assert "qachon gaplashish siz uchun qulay bo'lgan" in system_prompt


@pytest.mark.parametrize("with_faq", [True, False], ids=["with_faq", "without_faq"])
async def test_pricing_fallback_invents_no_number_when_none_is_configured(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    with_faq: bool,
) -> None:
    """CLINIC_PHONE_NUMBERS is unset by default, and a model asked to tell
    patients to "call these numbers" with no numbers given will produce a
    plausible +998 one. The prompt must drop that half of the offer instead.
    """
    system_prompt = await _capture_system_prompt(db_session, seed, as_tenant, with_faq=with_faq)

    assert "ushbu telefon raqamlariga qo'ng'iroq qiling" not in system_prompt
    assert "You have NOT been given a phone number" in system_prompt
    assert "never invent one" in system_prompt
    # The callback half survives -- it is the part that works without numbers.
    assert "qachon gaplashish siz uchun qulay bo'lgan" in system_prompt


@pytest.mark.parametrize("with_faq", [True, False], ids=["with_faq", "without_faq"])
async def test_prompt_forbids_dodging_a_price_with_an_estimate(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    with_faq: bool,
) -> None:
    """A quoted range the clinic never agreed to is worse than no answer: the
    patient arrives expecting it.
    """
    system_prompt = await _capture_system_prompt(db_session, seed, as_tenant, with_faq=with_faq)

    assert "do not give a range" in system_prompt
    assert "Do not say a doctor will decide it" in system_prompt


# --- clinic details that come from configuration, not the knowledge base ---

CLINIC_ADDRESS = "Toshkent, Yunusobod, Moyqo'rg'on 11A"
CLINIC_PHONES = "+998336677788"


@pytest.mark.parametrize("with_faq", [True, False], ids=["with_faq", "without_faq"])
async def test_configured_address_is_given_to_the_model_as_fact(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    with_faq: bool,
) -> None:
    """Both prompts otherwise forbid stating an address, which is right when
    retrieval is the only source. An operator-typed one is a different kind of
    fact, and a clinic with an empty knowledge base still has to be able to
    answer "qayerdasiz?".
    """
    system_prompt = await _capture_system_prompt(
        db_session,
        seed,
        as_tenant,
        with_faq=with_faq,
        clinic_address=CLINIC_ADDRESS,
        clinic_phone_numbers=CLINIC_PHONES,
    )

    assert f"Address: {CLINIC_ADDRESS}" in system_prompt
    assert f"Phone: {CLINIC_PHONES}" in system_prompt
    assert "never altered or added to" in system_prompt


@pytest.mark.parametrize("with_faq", [True, False], ids=["with_faq", "without_faq"])
async def test_configured_details_do_not_unlock_the_rest_of_the_clinic(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    with_faq: bool,
) -> None:
    """Being handed an address is not evidence of knowing the opening hours.
    The exemption has to stay scoped to exactly what was configured, or it
    becomes a licence to improvise every other clinic detail.
    """
    system_prompt = await _capture_system_prompt(
        db_session,
        seed,
        as_tenant,
        with_faq=with_faq,
        clinic_address=CLINIC_ADDRESS,
    )

    assert "the only details of this clinic you have been given" in system_prompt
    assert "Do not treat anything else about it as known" in system_prompt


@pytest.mark.parametrize("with_faq", [True, False], ids=["with_faq", "without_faq"])
async def test_no_facts_section_appears_when_nothing_is_configured(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    with_faq: bool,
) -> None:
    """An empty deployment must not be told it has details it does not have --
    an empty "Address:" line is an invitation to fill it in.
    """
    system_prompt = await _capture_system_prompt(db_session, seed, as_tenant, with_faq=with_faq)

    assert "These clinic details are given to you as fact" not in system_prompt
    assert "Address:" not in system_prompt


# --- the chat client's own buttons, not something a patient typed ---


@pytest.mark.parametrize(
    "message",
    ["/start", "/START", "  /start  ", "/start@medasistebot", "/help"],
)
def test_a_bare_client_command_is_recognised(message: str) -> None:
    """Telegram sends "/start" when somebody opens the chat and presses the
    button, addresses it to a named bot in groups, and clients pad it.
    """
    assert is_client_command(message)


@pytest.mark.parametrize(
    "message",
    ["/start bugun qabulga yozilsam", "salom", "startga bosdim", "/", "buyragim og'riyapti"],
)
def test_anything_a_patient_actually_wrote_is_left_to_the_model(message: str) -> None:
    """A patient who types past the command has said something, and swallowing
    it would lose the only thing they wrote.
    """
    assert not is_client_command(message)


@pytest.mark.parametrize(
    "message",
    ["/start clinic_ad_2", "/start@medasistebot ig_promo-7", "/start A1"],
)
def test_a_deep_link_campaign_tag_is_still_the_start_button(message: str) -> None:
    """Ad campaigns reach the clinic through t.me/<bot>?start=<payload>, which
    Telegram delivers as "/start clinic_ad_2". The payload is a campaign tag,
    not a question — handing it to the model asks it to answer an opaque
    token, which is the greeting-in-a-random-alphabet bug one keystroke away.
    """
    assert is_client_command(message)


@pytest.mark.parametrize(
    "message",
    ["/help clinic_ad_2", "/start bugun qabulga", "/start narx?", "/start " + "a" * 65],
)
def test_only_start_takes_a_payload_and_only_a_payload_shaped_one(message: str) -> None:
    """The deep-link exception is deliberately narrow: only "/start", only one
    trailing word, and only within the alphabet and length Telegram allows a
    payload. Everything else is a patient talking.
    """
    assert not is_client_command(message)


def test_pressing_help_is_answered_with_help_not_a_greeting() -> None:
    """A patient who presses Help halfway through a conversation is asking
    what this chat can do. Greeting them from scratch drops that question and
    reads as though the clinic forgot the conversation so far.
    """
    assert client_command_response("/help") == HELP_RESPONSES["uz-latn"]
    assert client_command_response("/start") == START_RESPONSES["uz-latn"]
    assert HELP_RESPONSES["uz-latn"] != START_RESPONSES["uz-latn"]


@pytest.mark.parametrize(
    ("default_language", "expected_script"),
    [("Russian", "ru"), ("russian", "ru"), ("Uzbek", "uz-latn"), ("English", "uz-latn")],
)
def test_a_button_is_answered_in_the_clinics_configured_language(
    default_language: str, expected_script: str
) -> None:
    """ "/start" has no letters in it at all, so script detection always lands
    on Uzbek Latin and DEFAULT_REPLY_LANGUAGE was ignored — a clinic
    configured for Russian greeted every patient in Uzbek on the one message
    that makes its first impression.

    A language these fixed lines don't exist in, English included, still falls
    back to the clinic's own rather than guessing.
    """
    assert client_command_response("/start", default_language) == START_RESPONSES[expected_script]
    assert client_command_response("/help", default_language) == HELP_RESPONSES[expected_script]


async def test_pressing_start_greets_once_without_asking_the_model(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """The bug this fixes was the clinic's first impression: "/start" went to
    the model, which greeted in Cyrillic, and the patient's own "Salom" a
    moment later drew a second greeting in Latin. Two hellos, two alphabets.

    Answering it here also spends no daily model allowance on the one message
    whose reply is entirely predictable.
    """
    llm_provider = FakeLLMProvider()

    with as_tenant(seed.tenant_a.id):
        reply = await generate_answer(
            db_session,
            "/start",
            embedding_provider=FakeEmbeddingProvider(QUERY_VECTOR),
            llm_provider=llm_provider,
            settings=_settings(answer_without_faq=True),
        )

    assert reply == START_RESPONSES["uz-latn"]
    assert llm_provider.calls == []
    # One greeting, in the clinic's own alphabet.
    assert "Assalom alaykum" in reply
    assert "Ассалом" not in reply


# --- the clinic's own clinicians ---

ROSTER = [
    ("Dr. Aliyev A.A.", "Urolog", "09:00 - 18:00 (Du-Ju)"),
    ("Dr. Karimova N.S.", "Urolog-androlog", "09:00 - 17:00 (Du-Ju)"),
]


@pytest.mark.parametrize("with_faq", [True, False], ids=["with_faq", "without_faq"])
async def test_the_clinics_doctors_are_given_to_the_model_as_fact(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    with_faq: bool,
) -> None:
    """ "Kim qabul qiladi?" is one of the first things a patient asks. The
    roster was already being read on every message to size the appointment
    book, and then thrown away, so both prompts' "you do not know this
    clinic's staff" rule made the bot refuse a question the database could
    answer.
    """
    system_prompt = await _capture_system_prompt(
        db_session, seed, as_tenant, with_faq=with_faq, doctors=ROSTER
    )

    for name, specialty, hours in ROSTER:
        assert f"- {name} — {specialty} — {hours}" in system_prompt
    assert "currently seeing patients here, given to you as fact" in system_prompt


@pytest.mark.parametrize("with_faq", [True, False], ids=["with_faq", "without_faq"])
async def test_the_roster_is_a_closed_list_with_nothing_added_to_it(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    with_faq: bool,
) -> None:
    """A name, a specialty and working hours is all the clinic said. Years of
    experience, where somebody trained, or which of them a patient "should"
    see are exactly the claims a patient picks a clinician on, and exactly
    what a model will supply unprompted.
    """
    system_prompt = await _capture_system_prompt(
        db_session, seed, as_tenant, with_faq=with_faq, doctors=ROSTER
    )

    assert "never name a doctor who is not on it" in system_prompt
    assert "their experience, their qualifications, where they studied" in system_prompt


@pytest.mark.parametrize("with_faq", [True, False], ids=["with_faq", "without_faq"])
async def test_working_hours_are_not_offered_as_free_appointment_times(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    with_faq: bool,
) -> None:
    """ "09:00 - 18:00" is when the doctor is in the building, not when the
    diary is empty. Read as availability it would have the bot offering hours
    that are already booked, and promising a named clinician for a slot the
    front desk assigns at settle time.
    """
    system_prompt = await _capture_system_prompt(
        db_session, seed, as_tenant, with_faq=with_faq, doctors=ROSTER
    )

    assert "They are not free appointment times" in system_prompt
    assert "do not promise a patient a particular doctor" in system_prompt


async def test_the_roster_survives_a_clinic_with_no_configured_details(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """Address and phone come from the environment, the roster from the
    database, and neither depends on the other. A deployment with no
    CLINIC_ADDRESS set must still be able to say who works there.
    """
    system_prompt = await _capture_system_prompt(
        db_session, seed, as_tenant, with_faq=False, doctors=ROSTER
    )

    assert "These clinic details are given to you as fact" not in system_prompt
    assert "Address:" not in system_prompt
    assert "Dr. Aliyev A.A." in system_prompt


@pytest.mark.parametrize("with_faq", [True, False], ids=["with_faq", "without_faq"])
async def test_a_clinic_that_has_listed_no_doctors_claims_none(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    with_faq: bool,
) -> None:
    """An empty heading is an invitation to fill it in, and an invented
    urologist is worse than a made-up address: the patient arrives asking for
    somebody by name.
    """
    system_prompt = await _capture_system_prompt(db_session, seed, as_tenant, with_faq=with_faq)

    assert "currently seeing patients here" not in system_prompt


# --- medical-advice: still goes through the LLM, with redirect framing enforced ---


async def test_generate_answer_medical_advice_message_gets_redirect_framing(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    embedding_provider = FakeEmbeddingProvider(QUERY_VECTOR)
    llm_provider = FakeLLMProvider(reply="Only a doctor can answer that at an appointment!")

    with as_tenant(seed.tenant_a.id):
        # Seeded so retrieval isn't empty — this test is about the
        # medical-advice reminder being added to the prompt, not about the
        # no-match short-circuit.
        await _make_faq(db_session, "What are your hours?", "9 to 5, Mon-Sat.")

        result = await generate_answer(
            db_session,
            "What antibiotic should I take for this?",
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
        )

    assert result == "Only a doctor can answer that at an appointment!"
    assert len(llm_provider.calls) == 1
    system_prompt, _messages = llm_provider.calls[0]
    assert "IMPORTANT: This message was flagged" in system_prompt


# --- emergency: fixed response, LLM (and embedding provider) never called ---


async def test_generate_answer_emergency_message_returns_fixed_response(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    embedding_provider = FakeEmbeddingProvider(QUERY_VECTOR)
    llm_provider = FakeLLMProvider()

    with as_tenant(seed.tenant_a.id):
        result = await generate_answer(
            db_session,
            "Severe pain and I can't stop bleeding, please help",
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
        )

    assert result == EMERGENCY_RESPONSES["uz-latn"]
    assert llm_provider.calls == []
    assert embedding_provider.calls == []


async def test_generate_answer_emergency_message_in_russian(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    embedding_provider = FakeEmbeddingProvider(QUERY_VECTOR)
    llm_provider = FakeLLMProvider()

    with as_tenant(seed.tenant_a.id):
        result = await generate_answer(
            db_session,
            "Не могу дышать, помогите!",
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
        )

    # Russian in, Russian out. This used to be one English sentence for
    # everybody, and a patient in Tashkent was told in English to call 103.
    assert result == EMERGENCY_RESPONSES["ru"]
    assert llm_provider.calls == []
    assert embedding_provider.calls == []


async def test_generate_answer_emergency_message_in_uzbek(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    embedding_provider = FakeEmbeddingProvider(QUERY_VECTOR)
    llm_provider = FakeLLMProvider()

    with as_tenant(seed.tenant_a.id):
        result = await generate_answer(
            db_session,
            "Qon to'xtamayapti, juda qo'rqinchli",
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
        )

    assert result == EMERGENCY_RESPONSES["uz-latn"]
    assert llm_provider.calls == []
    assert embedding_provider.calls == []


# --- the roster's own rendering ---------------------------------------------
#
# Pure functions, so these need no database: what is being checked is the
# shape of the text handed to the model, which is where the ugliness came
# from.


class _Listed:
    """A doctor row, as much of one as the roster reads."""

    def __init__(self, name: str, specialty: str, working_hours: str) -> None:
        self.name = name
        self.specialty = specialty
        self.working_hours = working_hours


_ALL_NINE_TO_SIX = [
    _Listed("Abdirova Umida Irsaliyevna", "Akusher-ginekolog", "09:00 - 18:00"),
    _Listed("Iriskulov Nabijan Kadirkulovich", "LOR", "09:00 - 18:00"),
    _Listed("Xusanov Sanjarbek Muhammadsohibovich", "Urolog-androlog", "09:00 - 18:00"),
]


def test_hours_every_doctor_shares_are_stated_once_not_per_name() -> None:
    """What the clinic complained about: asked for information about the
    clinic and its doctors, the reply came back as six bulleted names each
    ending "09:00-18:00", because that is exactly how the prompt handed the
    roster over.
    """
    roster = _doctor_roster(_ALL_NINE_TO_SIX)

    assert roster.count("09:00 - 18:00") == 1
    assert "- Iriskulov Nabijan Kadirkulovich — LOR" in roster
    assert "LOR — 09:00" not in roster
    assert "never after each name" in roster


def test_doctors_who_work_different_hours_keep_their_own() -> None:
    """The hours only collapse into one fact when they are one fact. A clinic
    whose LOR leaves at two has to be able to say so.
    """
    mixed = [
        _Listed("Dr. Aliyev A.A.", "Urolog", "09:00 - 18:00"),
        _Listed("Dr. Karimova N.S.", "LOR", "09:00 - 14:00"),
    ]

    roster = _doctor_roster(mixed)

    assert "- Dr. Aliyev A.A. — Urolog — 09:00 - 18:00" in roster
    assert "- Dr. Karimova N.S. — LOR — 09:00 - 14:00" in roster


def test_the_working_week_reaches_the_prompt_when_it_is_configured() -> None:
    """ "09:00-18:00" on its own never says which days, and the days were the
    part patients were being told wrongly.
    """
    facts = _clinic_facts_block(
        "Toshkent, Moyqo'rg'on 11A",
        "+998 71 200 03 93",
        _ALL_NINE_TO_SIX,
        "Dushanbadan shanbagacha 09:00 dan 18:00 gacha",
    )

    assert "Open: Dushanbadan shanbagacha 09:00 dan 18:00 gacha" in facts


def test_an_unconfigured_working_week_adds_no_line() -> None:
    facts = _clinic_facts_block("Toshkent, Moyqo'rg'on 11A", "+998 71 200 03 93", ())

    assert "Open:" not in facts


def _prompt_for(**persona: str | None) -> str:
    return _build_system_prompt(
        (),
        flagged_as_medical_advice=False,
        default_language="Uzbek",
        clinic_phone_numbers="+998 71 200 03 93",
        clinic_address=None,
        signals=read_signals([], "Salom"),
        **persona,
    )


def test_a_doctors_inbox_speaks_for_that_doctor_without_being_them() -> None:
    """The deployment on a doctor's own Instagram. The patient wrote to the
    doctor, so the doctor is who sees them -- but the reply comes from the
    person running the inbox, never in the doctor's first person, because a
    patient who thinks the doctor answered acts on it as medical word."""
    prompt = _prompt_for(doctor_name="Axmadaliyev Temur", doctor_specialty="urolog-androlog")

    assert "personal Instagram account of Axmadaliyev Temur, urolog-androlog" in prompt
    assert "You are not the doctor" in prompt
    assert "front desk of a medical clinic" not in prompt
    # The medical guard is the same guard whoever's inbox it is.
    assert "NEVER diagnose" in prompt


@pytest.mark.parametrize(
    "persona",
    [{}, {"doctor_name": "Axmadaliyev Temur"}, {"doctor_specialty": "urolog-androlog"}],
)
def test_without_a_whole_doctor_the_clinic_voice_is_kept(persona: dict[str, str]) -> None:
    prompt = _prompt_for(**persona)

    assert "front desk of a medical clinic" in prompt
    assert "personal Instagram account" not in prompt


def test_a_brace_in_the_doctors_name_is_not_a_template_field() -> None:
    prompt = _prompt_for(doctor_name="Dr {x}", doctor_specialty="urolog")

    assert "Dr {x}, urolog" in prompt
