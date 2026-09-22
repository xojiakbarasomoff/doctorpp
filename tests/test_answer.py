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
from app.repositories.knowledge_document import (
    KnowledgeChunkRepository,
    KnowledgeDocumentRepository,
)
from app.services.answer import generate_answer
from app.services.patient_profile import Profile
from app.services.persona import DEFAULT_PROMPT
from tests.conftest import Seed, isolated_settings

# A fixed, non-zero direction. Distance to itself is 0.0, so any FAQ seeded
# with this embedding is a guaranteed match for a FakeEmbeddingProvider that
# returns it.
QUERY_VECTOR = [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector for _ in texts]


class FakeLLMProvider(LLMProvider):
    """Returns `reply`, or -- given a sequence -- one reply per call, in
    order, holding on the last one for any call past the end. The second
    shape is what a test of a retry (app.services.answer._enforce) needs:
    a first reply that gets rejected, a second that does not.
    """

    def __init__(self, reply: str | Sequence[str] = "Sure, here's the answer.") -> None:
        self._replies = [reply] if isinstance(reply, str) else list(reply)
        self.calls: list[tuple[str, list[ChatMessage]]] = []

    async def generate(self, system_prompt: str, messages: list[ChatMessage]) -> str:
        self.calls.append((system_prompt, messages))
        index = min(len(self.calls) - 1, len(self._replies) - 1)
        return self._replies[index]


def _settings(**overrides: object) -> Settings:
    return isolated_settings(**overrides)


async def _ask(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    message: str,
    *,
    reply: str | Sequence[str] = "Sure, here's the answer.",
    with_faq: bool = False,
    files: Sequence[tuple[str, str | None, str]] = (),
    doctors: Sequence[tuple[str, str, str]] = (),
    history: Sequence[ChatMessage] | None = None,
    patient: Profile | None = None,
    tenant_settings: dict[str, object] | None = None,
    **settings_overrides: object,
) -> tuple[str, FakeLLMProvider]:
    llm_provider = FakeLLMProvider(reply=reply)
    with as_tenant(seed.tenant_a.id):
        if tenant_settings is not None:
            seed.tenant_a.settings = {**seed.tenant_a.settings, **tenant_settings}
        if with_faq:
            await KnowledgeBaseRepository(db_session).create(
                question="Konsultatsiya narxi qancha?",
                answer="Konsultatsiya 150 000 so'm.",
                embedding=QUERY_VECTOR,
            )
        for filename, label, text in files:
            document = await KnowledgeDocumentRepository(db_session).create(
                filename=filename, content_type=None, size_bytes=len(text), chunk_count=1
            )
            await KnowledgeChunkRepository(db_session).add_many(
                document.id,
                [{"ordinal": 0, "label": label, "content": text, "embedding": QUERY_VECTOR}],
            )
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
            patient=patient,
        )
    return result, llm_provider


async def test_the_models_reply_is_returned_exactly_as_written(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    written = "Va alaykum assalom! Tushundim, aytavering. " + "Uzun matn. " * 40

    result, llm = await _ask(db_session, seed, as_tenant, "Salom", reply=written)

    assert result == written
    assert len(llm.calls) == 1


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


async def test_a_matching_knowledge_base_row_is_in_the_prompt(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    _, llm = await _ask(db_session, seed, as_tenant, "Narxi qancha?", with_faq=True)

    prompt, messages = llm.calls[0]
    assert "# BILIMLAR BAZASI" in prompt
    assert "Savol: Konsultatsiya narxi qancha?\nJavob: Konsultatsiya 150 000 so'm." in prompt
    assert messages == [{"role": "user", "content": "Narxi qancha?"}]


async def test_with_no_match_the_model_is_told_not_to_guess(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    result, llm = await _ask(db_session, seed, as_tenant, "Implant qilasizmi?")

    assert result == "Sure, here's the answer."
    assert "Bu savolga mos yozuv topilmadi" in llm.calls[0][0]


async def test_the_default_persona_is_used_until_the_clinic_writes_its_own(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    _, llm = await _ask(db_session, seed, as_tenant, "Salom")

    assert llm.calls[0][0].startswith(DEFAULT_PROMPT.split("\n")[0])
    assert "# TAQIQLANGAN IBORALAR" in llm.calls[0][0]
    assert "# NAMUNA YOZISHMALAR" in llm.calls[0][0]


async def test_the_clinics_own_persona_and_examples_from_the_dashboard_are_used(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    _, llm = await _ask(
        db_session,
        seed,
        as_tenant,
        "Salom",
        tenant_settings={
            "assistant_prompt": "Siz Nigora ismli administratorsiz.",
            "assistant_examples": "Bemor: Salom\nAdmin: Salom, eshitaman.",
        },
    )

    prompt = llm.calls[0][0]
    assert prompt.startswith("Siz Nigora ismli administratorsiz.")
    assert "Admin: Salom, eshitaman." in prompt
    assert "Madina" not in prompt


async def test_one_clinics_persona_is_not_read_by_another(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    with as_tenant(seed.tenant_b.id):
        seed.tenant_b.settings = {**seed.tenant_b.settings, "assistant_prompt": "B klinikasi"}
        await db_session.flush()

    _, llm = await _ask(db_session, seed, as_tenant, "Salom")

    assert "B klinikasi" not in llm.calls[0][0]


async def test_the_dashboards_clinic_details_reach_the_model(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    _, llm = await _ask(
        db_session,
        seed,
        as_tenant,
        "Manzilingiz qayerda?",
        tenant_settings={
            "clinic_address": "Toshkent, Chilonzor 1",
            "clinic_landmark": "Metro yonida",
            "clinic_phone_numbers": "+998 71 200 03 93",
            "clinic_work_hours": "Du-Sh 09:00-17:00",
            "strict_rules": ["Shanba kuni ham ishlaymiz"],
        },
        doctors=[("Dr. Karimova N.S.", "Urolog", "09:00 - 17:00")],
    )

    prompt = llm.calls[0][0]
    assert "Manzil: Toshkent, Chilonzor 1" in prompt
    assert "Mo'ljal: Metro yonida" in prompt
    assert "Telefon: +998 71 200 03 93" in prompt
    assert "Ish vaqti: Du-Sh 09:00-17:00" in prompt
    assert "- Dr. Karimova N.S. — Urolog — 09:00 - 17:00" in prompt
    assert "- Shanba kuni ham ishlaymiz" in prompt


async def test_the_deployments_environment_fills_what_the_dashboard_left_empty(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    _, llm = await _ask(
        db_session,
        seed,
        as_tenant,
        "Salom",
        tenant_settings={"clinic_address": "Dashboarddagi manzil", "clinic_phone_numbers": "  "},
        clinic_address="Env manzil",
        clinic_phone_numbers="+998 70 310 40 40",
        doctor_name="Axmadaliyev Temur",
        doctor_specialty="urolog-androlog",
        doctor_background="Ish tajribasi: 5 yil",
    )

    prompt = llm.calls[0][0]
    assert "Manzil: Dashboarddagi manzil" in prompt
    assert "Env manzil" not in prompt
    assert "Telefon: +998 70 310 40 40" in prompt
    assert "Shifokor: Axmadaliyev Temur, urolog-androlog" in prompt
    assert "Shifokor haqida:\n- Ish tajribasi: 5 yil" in prompt


async def test_what_the_clinic_knows_about_the_patient_is_in_the_prompt(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    _, llm = await _ask(
        db_session,
        seed,
        as_tenant,
        "16:20",
        history=[{"role": "user", "content": "буйрагим оғрияпти"}],
        patient=Profile(name="Asadbek Risqiyev", phone="+998939511111"),
    )

    assert (
        "Ism: Asadbek Risqiyev | Telefon: +998939511111 | Til/yozuv: o'zbek, kirill yozuvi"
    ) in llm.calls[0][0]


async def test_a_language_the_patient_asked_for_outranks_the_alphabet_of_their_message(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    _, llm = await _ask(
        db_session,
        seed,
        as_tenant,
        "ok",
        patient=Profile(name=None, phone=None, language="ru"),
    )

    assert "Til/yozuv: rus, kirill yozuvi" in llm.calls[0][0]


async def test_a_conversation_with_history_is_told_not_to_restart(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """The bug this exists for: a patient who gave their name and number,
    then came back later with an unrelated question, was greeted again and
    asked to start booking from scratch."""
    history: list[ChatMessage] = [
        {"role": "user", "content": "Salom, Asadbek Risqiyev, 93 444 44 44"},
        {"role": "assistant", "content": "Va alaykum assalom, qaysi kun qulay?"},
    ]

    _, llm = await _ask(
        db_session, seed, as_tenant, "Sizlarda UZI ham bormi?", history=history
    )

    assert "birinchi xabar emas" in llm.calls[0][0]
    assert "qayta salomlashmang" in llm.calls[0][0]


async def test_the_opening_message_carries_no_dont_restart_note(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    _, llm = await _ask(db_session, seed, as_tenant, "Salom", history=None)

    assert "birinchi xabar emas" not in llm.calls[0][0]


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


async def test_the_booking_contract_and_the_book_appear_only_when_booking_is_on(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    _, off = await _ask(db_session, seed, as_tenant, "Salom")
    _, on = await _ask(db_session, seed, as_tenant, "Salom", booking_enabled=True)

    assert "THE APPOINTMENT BOOK" not in off.calls[0][0]
    assert "[[BOOK:" not in off.calls[0][0]
    assert "THE APPOINTMENT BOOK" in on.calls[0][0]
    assert "[[BOOK:YYYY-MM-DDTHH:MM|full name|telephone|reason]]" in on.calls[0][0]


async def test_a_persona_that_forgets_the_markers_still_gets_them(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """The markers are what writes bookings and callbacks down. They are not
    the clinic's text to delete."""
    _, llm = await _ask(
        db_session,
        seed,
        as_tenant,
        "Salom",
        tenant_settings={"assistant_prompt": "Faqat qisqa yozing."},
        booking_enabled=True,
    )

    prompt = llm.calls[0][0]
    assert "[[CALLBACK:" in prompt
    assert "[[BOOK:" in prompt


async def test_a_matching_piece_of_an_uploaded_file_is_in_the_prompt_with_its_source(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    _, llm = await _ask(
        db_session,
        seed,
        as_tenant,
        "UZI qancha turadi?",
        files=[("narxlar.pdf", "3-bet", "Xizmat: UZI; Narx: 150000")],
    )

    prompt = llm.calls[0][0]
    assert "[narxlar.pdf, 3-bet]\nXizmat: UZI; Narx: 150000" in prompt
    assert "Bu savolga mos yozuv topilmadi" not in prompt


async def test_a_file_without_a_page_is_named_by_its_file_alone(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    _, llm = await _ask(
        db_session, seed, as_tenant, "narxi", files=[("narxlar.csv", None, "UZI: 150000")]
    )

    assert "[narxlar.csv]\nUZI: 150000" in llm.calls[0][0]


async def test_the_bot_answers_from_the_faq_and_the_files_together(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    _, llm = await _ask(
        db_session,
        seed,
        as_tenant,
        "Narxi qancha?",
        with_faq=True,
        files=[("narxlar.pdf", None, "Xizmat: EKG; Narx: 80000")],
    )

    prompt = llm.calls[0][0]
    assert "Javob: Konsultatsiya 150 000 so'm." in prompt
    assert "[narxlar.pdf]\nXizmat: EKG; Narx: 80000" in prompt


async def test_one_clinics_files_do_not_reach_another_clinics_bot(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    with as_tenant(seed.tenant_b.id):
        document = await KnowledgeDocumentRepository(db_session).create(
            filename="ularniki.pdf", content_type=None, size_bytes=1, chunk_count=1
        )
        await KnowledgeChunkRepository(db_session).add_many(
            document.id,
            [{"ordinal": 0, "label": None, "content": "MAXFIY", "embedding": QUERY_VECTOR}],
        )

    _, llm = await _ask(db_session, seed, as_tenant, "narxi")

    assert "MAXFIY" not in llm.calls[0][0]


# --- the medical-safety check (app.services.medical_safety) ----------------


async def test_a_reply_that_prescribes_is_rewritten_before_it_is_sent(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    unsafe = "Amoksitsillin kuniga 2 mahal iching."
    safe = "Buni ko'rmasdan aytolmayman, qabulga yozib qo'yaymi?"

    result, llm = await _ask(
        db_session, seed, as_tenant, "Nima ichsam bo'ladi?", reply=[unsafe, safe]
    )

    assert result == safe
    assert len(llm.calls) == 2


async def test_the_second_try_is_told_plainly_what_the_first_one_did(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    unsafe = "Sizda prostatit bor."
    safe = "Buni ko'rmasdan aytolmayman."

    _, llm = await _ask(db_session, seed, as_tenant, "Menda nima?", reply=[unsafe, safe])

    second_prompt = llm.calls[1][0]
    assert "YOUR LAST REPLY WAS REJECTED" in second_prompt
    assert "told the patient what condition they have" in second_prompt
    # The first prompt did not carry the rejection -- only the second call did.
    assert "YOUR LAST REPLY WAS REJECTED" not in llm.calls[0][0]


async def test_a_reply_that_still_prescribes_after_one_retry_gets_the_fixed_safe_line(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    from app.services.medical_safety import SAFE_REPLIES

    result, llm = await _ask(
        db_session,
        seed,
        as_tenant,
        "Nima ichsam bo'ladi?",
        reply=["500 mg iching.", "Kuniga 2 marta iching."],
    )

    assert result == SAFE_REPLIES["uz-latn"]
    assert len(llm.calls) == 2


async def test_the_safe_line_is_in_the_patients_own_script(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    from app.services.medical_safety import SAFE_REPLIES

    result, _ = await _ask(
        db_session,
        seed,
        as_tenant,
        "буйрагим оғрияпти, нима ичсам бўлади?",
        reply=["Дорини ичинг.", "Кунига 2 марта ичинг."],
    )

    assert result == SAFE_REPLIES["uz-cyrl"]


async def test_an_ordinary_reply_costs_exactly_one_call(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """The common case: nothing is wrong, and the check adds no second call."""
    _, llm = await _ask(db_session, seed, as_tenant, "Salom", reply="Va alaykum assalom!")

    assert len(llm.calls) == 1


async def test_a_provider_failure_on_the_retry_still_returns_a_safe_line(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    from app.services.medical_safety import SAFE_REPLIES

    class FailsOnSecondCall(FakeLLMProvider):
        async def generate(self, system_prompt: str, messages: list[ChatMessage]) -> str:
            if self.calls:
                raise RuntimeError("provider is down")
            return await super().generate(system_prompt, messages)

    llm = FailsOnSecondCall(reply="500 mg iching.")
    with as_tenant(seed.tenant_a.id):
        await db_session.flush()
        result = await generate_answer(
            db_session,
            "Nima ichsam bo'ladi?",
            embedding_provider=FakeEmbeddingProvider(QUERY_VECTOR),
            llm_provider=llm,
            settings=_settings(),
        )

    assert result == SAFE_REPLIES["uz-latn"]
