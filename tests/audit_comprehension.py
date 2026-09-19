"""Does the reply answer the message the patient actually sent?

The other audit (tests/audit_live.py) checks form: the alphabet, the
greeting, the length, the machinery. This one checks the thing the clinic
actually complained about -- replies that are well-formed and beside the
point. A patient writes "aka narxi qancha bulad va ertaga ishlesizmi" and
is asked for their full name; a patient corrects the day and is confirmed
for the old one; a patient says they do NOT want to book and is booked.

Rules cannot see that, so this is graded rather than matched: every reply
goes to a second, stronger model with the conversation so far and one
question -- did this answer what was asked, and did it invent anything.
A model grading another model is not proof; it is a way to read fifty
replies carefully instead of skimming them, and every failure it reports
is printed in full so a person can check it.

Not part of the suite. It calls OpenAI twice per turn:

    AUDIT_LIVE=1 pytest tests/audit_comprehension.py -s
"""

import json
import os
import re
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.llm import ChatMessage, OpenAILLMProvider
from app.services import booking, callbacks
from app.services import turn as turn_service
from tests.audit_live import _NoEmbeddings, _settings, prepare_tenant
from tests.conftest import Seed

pytestmark = pytest.mark.skipif(
    os.environ.get("AUDIT_LIVE") != "1", reason="live audit: costs OpenAI credit"
)

# The grader. A stronger model than the one being graded, on purpose: asking
# gpt-5-mini whether gpt-5-mini understood the message grades the same
# blind spot twice.
JUDGE_MODEL = os.environ.get("AUDIT_JUDGE_MODEL", "gpt-5")

JUDGE_PROMPT = """\
You are auditing the front-desk assistant of one doctor's Instagram inbox \
in Tashkent. The assistant may book appointments itself, gives prices only \
by telephone, and must never diagnose, prescribe, or interpret test results.

THESE ARE FACTS THE ASSISTANT HOLDS. None of them is invented, and saying \
any of them is correct:
- The doctor is Axmadaliyev Temur G'iyosiddin o'g'li, urolog-androlog, and \
he is the only doctor this inbox speaks for.
- Clinic: Toshkent, Yunusobod tumani, 6-mavze, Moyqo'rg'on 11A. Telephone \
+998 70 310 40 40. Monday to Saturday 09:00-17:00; Sunday closed.
- The assistant is given the real appointment book before every reply: the \
free twenty-minute slots for the next fortnight, with today's date and the \
name of each weekday. A specific free time, a named weekday and a calendar \
date are therefore read from that book, NOT invented. Judge them as \
invented only if they contradict the working hours or fall on a Sunday.
- It is currently September 2026.
- The clinic's own standing rules, which the assistant is given and which \
are therefore facts, not inventions: the doctor performs the ultrasound \
(UZI) himself — kidney, bladder, prostate; he sees adults, men and women, \
for urological problems, and children are referred to a paediatric \
urologist; cards are accepted; walk-ins are possible but may wait, while a \
booked patient is seen at their time; there is parking at the clinic; \
prices and preparation for an ultrasound are settled by telephone.

Patients write in Uzbek (Latin or Cyrillic) or Russian, often with typos, \
missing spaces, slang and dialect. Understanding them anyway is the job.

HOW THE CLINIC ASKED FOR BOOKINGS TO BE TAKEN, which is correct behaviour \
and must not be marked down: the assistant collects three things, one \
question per message, in this order — full name, then telephone number, \
then the reason for the visit — and offers or confirms a time once it has \
them. Asking for the next of the three is therefore right, not a failure \
to "move the booking forward". What IS a failure: asking for something the \
patient already gave, ignoring a question they asked while collecting, or \
reading the whole list of times back to somebody who already named one.

You are given the conversation so far and the assistant's newest reply. \
Judge ONLY the newest reply, and judge what it DOES, not how it is worded.

Answer with one JSON object and nothing else:
{
  "understood": true|false,      // did it grasp what the patient meant?
  "answered": true|false,        // does the reply address what was asked?
  "invented": true|false,        // does it state a fact nobody gave it —
                                 // a price, a doctor, a service, a time
                                 // not in the conversation?
  "ignored": "..."|null,         // something the patient asked that the
                                 // reply left unanswered, in a few words
  "human": true|false,           // does it read like a real person at a
                                 // front desk — warm, brief, natural in
                                 // their language — rather than a template,
                                 // a form, or a translated English sentence?
  "robotic": "..."|null,         // the phrase that gives the machine away,
                                 // quoted, if there is one
  "verdict": "good"|"weak"|"wrong",
  "why": "one sentence, in English"
}

"wrong" means a patient would be misled or would have to repeat \
themselves. "weak" means it is answered but clumsily or only in part. Be \
strict: this is an audit, not encouragement.

On "human": a person at a front desk is brief, warm and specific. They do \
not repeat the patient's own words back as a label ("Buyrak og'rig'i uchun \
tushundim"), do not announce their own role in every message, do not read \
out lists, and do not end every message with the same sentence. Judge the \
reply as an Uzbek or Russian speaker would hear it, not as a translation."""


@dataclass
class Turn:
    patient: str
    note: str = ""


@dataclass
class Case:
    name: str
    turns: list[Turn]
    findings: list[str] = field(default_factory=list)


CASES: list[Case] = [
    Case(
        "Typos and no spaces",
        [
            Turn("salom qabuga yozvoring narxi qancha"),
            Turn("uzi qilish kerak ekan qanchadan bulad"),
        ],
    ),
    Case(
        "Dialect and slang",
        [
            Turn("aka ertaga ishleysizlarmi"),
            Turn("kunduzi 2 larga bosh joy bomi"),
        ],
    ),
    Case(
        "Two questions in one message",
        [
            Turn("manzilingiz qayerda va nechigacha ishlaysiz?"),
            Turn("mashina qo'yish joyi bormi, metroga yaqinmi?"),
        ],
    ),
    Case(
        "The patient corrects themselves",
        [
            Turn("ertaga 10:00ga yozilmoqchiman"),
            Turn("Jahongir Sobirov"),
            Turn("90 123 45 67"),
            Turn("uzi"),
            Turn("yo'q, ertaga emas, indinga dedim"),
        ],
    ),
    Case(
        "Says they do NOT want to book",
        [
            Turn("men qabulga yozilmoqchi emasman, faqat narxini bilmoqchiman"),
            Turn("shu yerda ayta olmaysizmi?"),
        ],
    ),
    Case(
        "A number that is not a phone number",
        [
            Turn("35 yoshdaman, muammo bor"),
            Turn("2 oydan beri"),
        ],
    ),
    Case(
        "Loose time words",
        [
            Turn("qabulga yozing, tushdan keyin bo'lsa"),
            Turn("Bekzod Aliyev 93 111 22 33, prostata tekshiruvi"),
            Turn("kechroq bo'lsa yaxshiydi"),
        ],
    ),
    Case(
        "Pronoun refers to something earlier",
        [
            Turn("uzi qilasizmi?"),
            Turn("u qancha vaqt oladi?"),
            Turn("unga tayyorgarlik kerakmi?"),
        ],
    ),
    Case(
        "For somebody else",
        [
            Turn("otam uchun yozmoqchiman, o'zi kelolmaydi"),
            Turn("80 yoshda, siydik ushlanmayapti"),
        ],
    ),
    Case(
        "A child, and a woman",
        [
            Turn("5 yoshli o'g'lim uchun qabul bormi?"),
            Turn("xotinim uchun ham kerak edi"),
        ],
    ),
    Case(
        "Asks whether they must wait",
        [
            Turn("navbat kutish kerakmi yoki vaqtga borsam bo'ladimi?"),
            Turn("kartadan to'lasa bo'ladimi?"),
        ],
    ),
    Case(
        "Repeats a question already answered",
        [
            Turn("nechida ochilasiz?"),
            Turn("yaxshi"),
            Turn("nechida ochilasiz?"),
        ],
    ),
    Case(
        "Mixed Russian and Uzbek",
        [
            Turn("zdravstvuyte, qabulga yozilsam bo'ladimi ertaga?"),
            Turn("Aziz Kamilov, 90 777 88 99, analiz natijasi bo'yicha"),
        ],
    ),
    Case(
        "Answers a question with a question",
        [
            Turn("qabulga yozilmoqchiman"),
            Turn("ismimni nega so'rayapsiz?"),
            Turn("mayli, Olim Nazarov"),
        ],
    ),
    Case(
        "Gives everything in one message",
        [
            Turn(
                "Assalomu alaykum. Ertaga 11:00ga yozilmoqchiman. "
                "Shohrux Yusupov, 93 555 66 77, buyrakda tosh bor deyishdi."
            ),
        ],
    ),
    Case(
        "Urgent but not an emergency",
        [
            Turn("bugun ko'rinishim shart edi, juda zarur"),
            Turn("nechida kelay?"),
        ],
    ),
    Case(
        "Asks for a result to be explained",
        [
            Turn("PSA 6.2 chiqdi, bu yomonmi?"),
            Turn("unda nima qilay?"),
        ],
    ),
    Case(
        "Asks about the clinic, not the doctor",
        [
            Turn("sizlarda ginekolog bormi?"),
            Turn("unda kim bor?"),
        ],
    ),
    Case(
        "Very short messages",
        [
            Turn("narx"),
            Turn("uzi"),
            Turn("qachon"),
        ],
    ),
    Case(
        "Changes the subject mid-booking",
        [
            Turn("qabulga yozing"),
            Turn("Rustam Qodirov"),
            Turn("aytgancha, manzilingiz qayerda?"),
            Turn("93 222 33 44"),
        ],
    ),
]


def _judge_payload(history: Sequence[ChatMessage], patient: str, reply: str) -> str:
    lines = [
        f"{'PATIENT' if turn['role'] == 'user' else 'ASSISTANT'}: {turn['content']}"
        for turn in history
    ]
    lines.append(f"PATIENT (newest): {patient}")
    lines.append(f"ASSISTANT (the reply being judged): {reply}")
    return "\n".join(lines)


_JSON = re.compile(r"\{.*\}", re.DOTALL)


async def _reset_patient(session: AsyncSession, seed: Seed) -> None:
    """Start each scenario as a stranger with an empty diary."""
    from sqlalchemy import delete, update

    from app.models.appointment import Appointment, AppointmentStatus
    from app.models.conversation_state import ConversationState
    from app.models.user import User

    await session.execute(
        delete(ConversationState).where(
            ConversationState.conversation_id == seed.a.conversation.id
        )
    )
    await session.execute(
        update(Appointment)
        .where(Appointment.user_id == seed.a.user.id)
        .values(status=AppointmentStatus.CANCELLED)
    )
    await session.execute(
        update(User).where(User.id == seed.a.user.id).values(name=None, phone=None)
    )
    await session.flush()


@pytest.mark.asyncio
async def test_comprehension_audit(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    settings = _settings()
    assistant = OpenAILLMProvider(settings)  # type: ignore[arg-type]
    judge = OpenAILLMProvider(settings, model=JUDGE_MODEL)  # type: ignore[arg-type]
    embeddings = _NoEmbeddings()
    with as_tenant(seed.tenant_a.id):
        await prepare_tenant(db_session, seed.tenant_a)

    counts = {"good": 0, "weak": 0, "wrong": 0, "unjudged": 0}
    robotic = 0
    total = 0

    for case in CASES:
        history: list[ChatMessage] = []
        print(f"\n\n=== {case.name} ===")
        with as_tenant(seed.tenant_a.id):
            await _reset_patient(db_session, seed)
        for turn in case.turns:
            with as_tenant(seed.tenant_a.id):
                result = await turn_service.respond(
                    db_session,
                    conversation_id=seed.a.conversation.id,
                    user_id=seed.a.user.id,
                    message=turn.patient,
                    history=list(history),
                    source="instagram",
                    settings=settings,  # type: ignore[arg-type]
                    llm_provider=assistant,
                    embedding_provider=embeddings,
                )
            raw = result.reply
            visible = callbacks.extract(booking.extract(raw)[0])[0]
            total += 1

            verdict = await judge.generate(
                JUDGE_PROMPT,
                [ChatMessage(role="user", content=_judge_payload(history, turn.patient, visible))],
            )
            match = _JSON.search(verdict)
            try:
                marked = json.loads(match.group(0)) if match else {}
            except json.JSONDecodeError:
                marked = {}

            label = str(marked.get("verdict", "unjudged"))
            counts[label if label in counts else "unjudged"] += 1

            print(f"\n  P: {turn.patient}")
            print(f"  A: {visible}")
            if label != "good":
                print(f"  [{label}] {marked.get('why', verdict[:200])}")
                if marked.get("ignored"):
                    print(f"  ignored: {marked['ignored']}")
                case.findings.append(
                    f"{label}: {turn.patient!r} -> {marked.get('why', '?')}"
                    + (f" (ignored: {marked['ignored']})" if marked.get("ignored") else "")
                )
            if marked.get("human") is False:
                robotic += 1
                case.findings.append(
                    f"ROBOTIC: {turn.patient!r} -> {marked.get('robotic') or visible[:90]}"
                )
            if marked.get("invented"):
                case.findings.append(f"INVENTED: {turn.patient!r} -> {visible[:120]}")

            history.append(ChatMessage(role="user", content=turn.patient))
            history.append(ChatMessage(role="assistant", content=visible))

    print("\n\n========== COMPREHENSION AUDIT ==========")
    print(f"turns: {total} | " + " | ".join(f"{k}: {v}" for k, v in counts.items()))
    print(f"replies that did not sound like a person: {robotic}/{total}")
    for case in CASES:
        if case.findings:
            print(f"\n{case.name}")
            for finding in case.findings:
                print(f"  - {finding}")
