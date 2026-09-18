"""Twenty conversations put through the real assistant, and marked.

Not part of the suite. It calls OpenAI with the deployment's own settings
and costs money, so it runs only when asked for:

    AUDIT_LIVE=1 pytest tests/audit_live.py -s

What it is for: the bugs this project has actually had were never visible
in a unit test. They were "the assistant greeted somebody who did not greet
it", "it asked for the number twice", "it answered a Latin message in
Cyrillic" — properties of a whole conversation, against the real model. So
each scenario below is a conversation, replayed turn by turn through
app.services.answer.generate_answer with the production prompt, and every
reply is marked against the rules the clinic actually asked for.

The failures are printed as a report rather than asserted one by one: the
point is a list of what is still wrong and how often, not a red build.
"""

import os
import re
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.embeddings import EmbeddingProvider
from app.rag.llm import ChatMessage, OpenAILLMProvider
from app.services import booking, callbacks, reply_style
from app.services.answer import conversation_script, generate_answer
from app.services.conversation_signals import looks_like_a_greeting, opens_with_a_greeting
from tests.conftest import Seed, isolated_settings

pytestmark = pytest.mark.skipif(
    os.environ.get("AUDIT_LIVE") != "1", reason="live audit: costs OpenAI credit"
)


# What the clinic has actually told the assistant, from the live
# deployment's own settings. Without these the audit measures a clinic that
# knows nothing -- and duly reports that it answers "bizda ma'lumot yo'q".
CLINIC_RULES = [
    "UZI ni shifokorning o'zi qiladi (buyrak, siydik pufagi, prostata). "
    "Bemor \"UZI qilasizmi?\" deb so'rasa, aniq \"ha, shifokorning o'zi qiladi\" deb javob bering.",
    "Shifokor kattalarni qabul qiladi — erkaklarni ham, ayollarni ham, urologik "
    "muammolar bilan. Bolalar uchun bolalar urologiga murojaat qilish kerakligini "
    "ayting va klinika raqamini bering.",
    "To'lovni karta bilan ham qilish mumkin.",
    "Yozilmasdan kelish ham mumkin, lekin navbat kutishga to'g'ri kelishi mumkin; "
    "vaqtga yozilgan bemor o'z vaqtida kiradi.",
    "Klinika yonida mashina qo'yish joyi bor.",
    "Narx, UZI ga tayyorgarlik va shunga o'xshash aniq savollar telefon orqali hal "
    "qilinadi: +998 70 310 40 40. Hech qachon \"tekshirib beraman\", \"aniqlab beraman\" "
    "deb va'da bermang — buni qila olmaysiz.",
]


async def prepare_tenant(session: object, tenant: object) -> None:
    """Make the fixture tenant look like the live one: the clinic's standing
    rules, and one doctor rather than the fixture's placeholder dentist."""
    from sqlalchemy import update

    from app.models.doctor import Doctor
    from app.models.tenant import Tenant

    tenant.settings = {**dict(tenant.settings or {}), "strict_rules": CLINIC_RULES}
    await session.execute(
        update(Doctor)
        .where(Doctor.tenant_id == tenant.id)
        .values(name="Axmadaliyev Temur G'iyosiddin o'g'li", specialty="urolog-androlog")
    )
    await session.flush()
    assert Tenant  # imported for the reader; the row above is already loaded


class _NoEmbeddings(EmbeddingProvider):
    """The knowledge base is empty in this deployment (ANSWER_WITHOUT_FAQ),
    so retrieval is a query that returns nothing; it still needs a vector."""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] * 1536 for _ in texts]


def _settings() -> object:
    return isolated_settings(
        database_url=os.environ["DATABASE_URL"],
        redis_url=os.environ["REDIS_URL"],
        webhook_verify_token="audit",
        meta_app_secret="audit",
        encryption_key=os.environ["ENCRYPTION_KEY"],
        session_secret_key="audit-audit-audit",
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
        openai_reasoning_effort=os.environ.get("OPENAI_REASONING_EFFORT", "low"),
        answer_without_faq=True,
        booking_enabled=True,
        default_reply_language="Uzbek",
        clinic_phone_numbers="+998 70 310 40 40",
        clinic_address="Toshkent shahri, Yunusobod tumani, 6-mavze, Moyqo'rg'on 11A",
        clinic_work_hours="Dushanbadan shanbagacha 09:00 dan 17:00 gacha, yakshanba dam olish",
        doctor_name="Axmadaliyev Temur G'iyosiddin o'g'li",
        doctor_specialty="urolog-androlog",
        doctor_background=(
            "Ish tajribasi: 5 yil | 2015-2021: Toshkent Tibbiyot akademiyasi, Davolash "
            "fakulteti | 2021-2025: Respublika urologiya markazi, magistratura | "
            "2025-2026: Respublika Malaka oshirish instituti, UZI sertifikat"
        ),
    )


# --- what counts as a fault -------------------------------------------------

# Whether the patient greeted, and whether the reply opens with one, are
# read with the application's own functions rather than a second regex
# written here. The first draft of this file had its own, it did not know
# "Ассалому алайкум", and it reported the assistant for greeting somebody
# back who had just said hello -- an audit that invents faults is worse
# than no audit.
_IDENTITY = re.compile(
    r"^\s*(?:men\s+)?(?:shifokorning|шифокорнинг)?\s*"
    r"(?:administrator|администратор)",
    re.IGNORECASE,
)
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_MACHINERY = re.compile(r"\bBOOK\b|\bCALLBACK\b|\[\[", re.IGNORECASE)
_DIAGNOSIS = re.compile(
    r"\bsizda\b[^.!?]{0,60}(?:prostatit|sistit|infeksiya|tosh)|у вас[^.!?]{0,60}(?:простатит|инфекц)",
    re.IGNORECASE,
)
_ENGLISH = re.compile(
    r"\b(?:appointment|reception|please|doctor will|available|schedule|sorry)\b", re.IGNORECASE
)


@dataclass
class Turn:
    patient: str
    # What this particular reply must and must not do, beyond the global rules.
    must_contain: tuple[str, ...] = ()
    must_not_contain: tuple[str, ...] = ()
    expect_booking: bool = False


@dataclass
class Scenario:
    name: str
    turns: list[Turn]
    faults: list[str] = field(default_factory=list)


def _global_faults(
    reply: str, *, patient: str, history: Sequence[ChatMessage], first: bool
) -> list[str]:
    found: list[str] = []
    # Exactly what the patient is sent: the worker strips both markers
    # before delivery (app.workers.tasks), so an audit that only strips one
    # reports a leak the patient never sees.
    visible = callbacks.extract(booking.extract(reply)[0])[0]
    script = conversation_script(list(history), patient)

    greeted = looks_like_a_greeting(patient)
    if opens_with_a_greeting(visible) and not greeted:
        found.append("greeted a patient who did not greet")
    if _IDENTITY.search(visible) and not first:
        found.append("re-introduced itself mid-conversation")
    if _ISO_DATE.search(visible):
        found.append("printed a machine date (2026-09-19)")
    if _MACHINERY.search(visible):
        found.append("leaked marker machinery to the patient")
    if _DIAGNOSIS.search(visible):
        found.append("told the patient what they have")
    if _ENGLISH.search(visible):
        found.append("used an English word")
    found.extend(reply_style.problems(visible, script=script, greeted=greeted))
    return found


def _selected(scenarios: list["Scenario"]) -> list["Scenario"]:
    """AUDIT_ONLY=12,13 runs those scenarios alone, for a re-check."""
    wanted = os.environ.get("AUDIT_ONLY", "").strip()
    if not wanted:
        return scenarios
    keys = {part.strip() for part in wanted.split(",") if part.strip()}
    return [s for s in scenarios if s.name.split(".")[0] in keys]


SCENARIOS: list[Scenario] = [
    Scenario(
        "1. Booking, Latin, no greeting",
        [
            Turn("Man kelasi seshanba 10:00ga qabulga yozilmoqchiman", must_not_contain=("assalom",)),
            Turn("Asadbek Risqiyev 93 9510000"),
            Turn("buyrak og'rig'i", must_not_contain=("ism", "familiy")),
            Turn("10:00", expect_booking=True),
        ],
    ),
    Scenario(
        "2. Booking, Cyrillic",
        [
            Turn("Ассалому алайкум, қабулга ёзилмоқчиман"),
            Turn("Жамшид Каримов"),
            Turn("90 123 45 67"),
            Turn("сийишда оғриқ"),
            Turn("эртага эрталабга"),
        ],
    ),
    Scenario(
        "3. Booking, Russian",
        [
            Turn("Здравствуйте, хочу записаться на приём"),
            Turn("Ирина Петрова"),
            Turn("90 555 44 33"),
            Turn("боли в почках"),
            Turn("завтра утром"),
        ],
    ),
    Scenario(
        "4. Name and number in one message",
        [
            Turn("qabulga yozing"),
            Turn("Sardor Tursunov 93 444 55 66", must_not_contain=("ismingiz", "familiya")),
            Turn("uzi qildirmoqchiman"),
        ],
    ),
    Scenario(
        "5. Asks the price",
        [Turn("qabul narxi qancha?"), Turn("qimmat ekan"), Turn("mayli, yozib qo'ying")],
    ),
    Scenario(
        "6. Asks the address and hours",
        [
            Turn("manzilingiz qayerda?"),
            Turn("nechigacha ishlaysiz?", must_not_contain=("ismingiz",)),
        ],
    ),
    Scenario(
        "7. Sunday",
        [Turn("yakshanba ishlaysizmi?"), Turn("dushanba qaysi vaqtlarga joy bor?")],
    ),
    Scenario(
        "8. Asks for medicine",
        [
            Turn("buyragim og'riyapti, qanday dori ichay?"),
            Turn("antibiotik kerakmi?"),
        ],
    ),
    Scenario(
        "9. Asks for a diagnosis",
        [Turn("menda prostatit bormi?"), Turn("analiz natijam yaxshimi?")],
    ),
    Scenario(
        "10. Asks about the doctor",
        [
            Turn("shifokor necha yillik tajribaga ega?", must_contain=("5",)),
            Turn("qayerda o'qigan?"),
        ],
    ),
    Scenario(
        "11. Asks who they are writing to",
        [Turn("siz kimsiz?"), Turn("doktorning o'zimi?")],
    ),
    Scenario(
        "12. Missed call",
        [Turn("telefon qildim, ko'tarmadingiz"), Turn("93 951 00 00")],
    ),
    Scenario(
        "13. Asks to be called",
        [Turn("menga qo'ng'iroq qiling"), Turn("90 111 22 33")],
    ),
    Scenario(
        "14. Greeting only",
        [Turn("Assalomu alaykum"), Turn("qabulga yozilsam bo'ladimi?")],
    ),
    Scenario(
        "15. Changes their mind about the time",
        [
            Turn("ertaga 09:00ga yozilmoqchiman"),
            Turn("Dilshod Rahimov"),
            Turn("93 777 88 99"),
            Turn("konsultatsiya"),
            Turn("09:00"),
            Turn("kechirasiz, 10:00ga o'zgartira olasizmi?"),
        ],
    ),
    Scenario(
        "16. Asks for a time that is taken or impossible",
        [
            Turn("bugun 03:00ga yozing"),
            Turn("unda yakshanba 11:00"),
        ],
    ),
    Scenario(
        "17. Mixed writing, short messages",
        [Turn("Nmagap"), Turn("qabul"), Turn("ha")],
    ),
    Scenario(
        "18. Emergency",
        [Turn("qon to'xtamayapti, nima qilay?")],
    ),
    Scenario(
        "19. Something outside urology",
        [Turn("tishim og'riyapti, qabulga yozing")],
    ),
    Scenario(
        "20. Asks about analysis results",
        [Turn("analiz natijamni ko'rib bering"), Turn("nima degani bu?")],
    ),
]


@pytest.mark.asyncio
async def test_audit(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    settings = _settings()
    provider = OpenAILLMProvider(settings)  # type: ignore[arg-type]
    embeddings = _NoEmbeddings()
    with as_tenant(seed.tenant_a.id):
        await prepare_tenant(db_session, seed.tenant_a)
    total_turns = 0

    for scenario in _selected(SCENARIOS):
        history: list[ChatMessage] = []
        print(f"\n\n=== {scenario.name} ===")
        for index, turn in enumerate(scenario.turns):
            with as_tenant(seed.tenant_a.id):
                reply = await generate_answer(
                    db_session,
                    turn.patient,
                    embedding_provider=embeddings,
                    llm_provider=provider,
                    settings=settings,  # type: ignore[arg-type]
                    history=list(history),
                )
            total_turns += 1
            visible, slot, _ = booking.extract(reply)
            faults = _global_faults(
                reply, patient=turn.patient, history=history, first=index == 0
            )
            for needle in turn.must_contain:
                if needle.lower() not in visible.lower():
                    faults.append(f"missing {needle!r}")
            for needle in turn.must_not_contain:
                if needle.lower() in visible.lower():
                    faults.append(f"should not say {needle!r}")
            if turn.expect_booking and slot is None:
                faults.append("no booking marker where the patient accepted a time")

            print(f"\n  P: {turn.patient}")
            print(f"  A: {visible}")
            if slot is not None:
                print(f"  [booked {slot:%Y-%m-%d %H:%M}]")
            if faults:
                print(f"  ⚠ {'; '.join(faults)}")
                scenario.faults.extend(f"turn {index + 1}: {fault}" for fault in faults)

            history.append(ChatMessage(role="user", content=turn.patient))
            history.append(ChatMessage(role="assistant", content=visible))

    print("\n\n================ AUDIT ================")
    audited = _selected(SCENARIOS)
    clean = [s for s in audited if not s.faults]
    print(f"turns: {total_turns} | clean scenarios: {len(clean)}/{len(audited)}")
    for scenario in audited:
        if scenario.faults:
            print(f"\n{scenario.name}")
            for fault in scenario.faults:
                print(f"  - {fault}")
