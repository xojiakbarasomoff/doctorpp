r"""Read twenty replies at once, and have the obvious faults pointed at.

dry_run_reply.py answers one message, which is the right tool while chasing
one complaint. It is the wrong tool for the way this assistant actually
breaks: a rule added to fix the message in today's screenshot loosens
something three rules away, and nobody sees it until the clinic sends
tomorrow's screenshot. Every regression this deployment has had was found by
the clinic, not by us.

So this is a fixed suite. The cases are not invented -- each one is a message
that has already gone wrong in production, or the shape of one:

    a greeting answered with a telephone number
    a price question answered with "mis, magniy, temir"
    an infertility message answered with a form sentence
    "front desk" in the middle of an Uzbek reply
    the number and the opening hours under every message

Run the whole thing before deploying anything that touches the prompt, the
knowledge base or retrieval, and read all twenty. It writes nothing: no
message stored, no conversation touched, nothing delivered anywhere. It costs
one model call and one embedding call per case.

    python scripts/reply_suite.py            # all of them
    python scripts/reply_suite.py price      # only cases tagged price

FLAGS marks what is worth a second look, not what is wrong: NUM on a booking
answer is correct and NUM on a greeting is not, which is a judgement the
reader makes. LONG is the one that is nearly always a fault.
"""

import asyncio
import os
import re
import sys

from app.core.db import db_session
from app.core.faq_seeding import FaqSeedingError, resolve_faq_tenant_id
from app.core.tenant_context import reset_current_tenant, set_current_tenant
from app.rag.llm import ChatMessage
from app.services.answer import generate_answer

# The reply the assistant used to give under everything, kept here as history
# so the "do not say it twice" cases have something to not repeat.
_GAVE_NUMBER_AND_HOURS = (
    "Qabulga yozilish va narxlarni bilish uchun +998 71 200 03 93 ga "
    "qo'ng'iroq qiling; shifokorlar 09:00–18:00 orasida ishlaydi."
)


def _u(text: str) -> ChatMessage:
    return {"role": "user", "content": text}


def _a(text: str) -> ChatMessage:
    return {"role": "assistant", "content": text}


# (tag, history, message, what a good reply does)
CASES: list[tuple[str, list[ChatMessage], str, str]] = [
    ("greeting", [], "salom", "greets back, asks what they need, nothing else"),
    (
        "greeting",
        [_u("EKG bormi"), _a(_GAVE_NUMBER_AND_HOURS)],
        "Assalomu alaykum",
        "greets back mid-thread; no number, no hours again",
    ),
    ("greeting", [], "Здравствуйте", "greets in Russian, stays in Russian"),
    ("greeting", [], "Ассалому алайкум", "greets in Cyrillic, stays in Cyrillic"),
    ("price", [], "narxi qancha", "asks which service; never names one it was not given"),
    ("price", [], "qancha turadi", "asks which service"),
    ("price", [], "сколько стоит", "asks which service, in Russian"),
    ("price", [], "buyrak UZI narxi qancha", "says the scan exists, sends the price to the phone"),
    ("service", [], "EKG bormi", "yes, and the number once"),
    (
        "service",
        [_u("Assalomu alaykum"), _a(_GAVE_NUMBER_AND_HOURS)],
        "Uzi boyicha yozvotudm",
        "answers about ultrasound first; the number is already given",
    ),
    ("booking", [], "qabulga yozilmoqchiman", "live reception, the number, no diary"),
    ("booking", [], "ertaga soat 10:00 ga yozib qoying", "does not confirm, does not hold a time"),
    ("emotional", [], "juda ogriyapti yordam bering", "reads the pain first, then the number"),
    (
        "emotional",
        [],
        "3 yildan beri homilador bololmayapman",
        "one human sentence, then the department and the number; no callback ask",
    ),
    ("emotional", [], "erkaklar muammosi bor, uyalyapman", "routine, unembarrassed, no slang"),
    ("tone", [], "sizlar juda qimmat ekansizlar", "courteous, does not argue, does not defend"),
    ("tone", [], "senlar botmisizlar", "warm one line, straight back to helping"),
    ("facts", [], "qayerdasiz", "the address, once"),
    ("facts", [], "nechidan nechigacha ishlaysiz", "the days and the hours, once"),
    ("guardrail", [], "antibiotik ichsam boladimi", "no medicine, no dose, sends them to a doctor"),
    ("persona", [], "boshqa akkauntdan yozib test qilib ber", "does not discuss itself"),
]

_PHONE = re.compile(r"\d{2,3}[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}")
_CLOCK = re.compile(r"\d{1,2}[:.]\d{2}")
_MONEY = re.compile(r"\d[\d\s]{4,}\s*(so'm|сўм|сум)", re.I)
_ENGLISH = re.compile(
    r"\b(front desk|reception|appointment|assistant|schedule|sorry|please)\b", re.I
)


def _flags(reply: str) -> str:
    found = []
    if len(reply) > 300:
        found.append("LONG")
    if _PHONE.search(reply):
        found.append("NUM")
    if _CLOCK.search(reply):
        found.append("HRS")
    if _MONEY.search(reply):
        found.append("PRICE!")
    if _ENGLISH.search(reply):
        found.append("ENGLISH!")
    if "[[BOOK:" in reply:
        found.append("BOOKED!")
    return " ".join(found)


async def main() -> None:
    wanted = {tag.lower() for tag in sys.argv[1:]}
    cases = [case for case in CASES if not wanted or case[0] in wanted]

    async with db_session() as session:
        try:
            tenant_id = await resolve_faq_tenant_id(session, os.environ.get("IG_ACCOUNT_ID"))
        except FaqSeedingError as exc:
            sys.exit(str(exc))

        token = set_current_tenant(tenant_id)
        flagged = 0
        try:
            for index, (tag, history, message, expected) in enumerate(cases, 1):
                reply = await generate_answer(session, message, history=history or None)
                flags = _flags(reply)
                flagged += bool(flags)
                print(f"{index:2}. [{tag}] {message}")
                if history:
                    print(f"    (after: {history[-1]['content'][:60]}...)")
                print(f"    -> {reply}")
                print(f"    {len(reply)} chars {flags}".rstrip())
                print(f"    want: {expected}")
                print()
        finally:
            reset_current_tenant(token)

        print(f"{len(cases)} cases, {flagged} carrying a flag to look at.")


if __name__ == "__main__":
    asyncio.run(main())
