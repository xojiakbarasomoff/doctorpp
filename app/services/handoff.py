"""Conversations a person has to answer, kept at the top of the dashboard.

Raised when the assistant hands a patient to the doctor -- it marks the reply
with [[DOCTOR]], or says as much in words the pattern below knows -- and when
a patient writes into a conversation staff have taken over. Cleared when a
person answers: from the dashboard, from the Instagram app itself, or by
marking it done in the dashboard after answering some other way.
"""

import re
import uuid
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation

# Tolerant of what a model actually writes -- "[[DOCTOR]", "[[ doctor: ... ]]"
# -- because any variant this misses is text the patient would see.
DOCTOR_MARKER = re.compile(r"\s*\[\[\s*DOCTOR\b[^\[\]]*\]?\]?\s*", re.IGNORECASE)

# Every apostrophe Uzbek Latin is typed with: ' ʻ ʼ ‘ ’ `.
_A = "['ʻʼ‘’`]?"

# A reply that tells the patient the doctor will answer them, without the
# marker. A handover the model forgot to mark is a patient who waits for an
# answer nobody knows is owed.
# The phrasings below are the ones real replies used: "doktor bilan aniqlab
# olaman", "sizni doktorga bildiraman", "klinika botiga yubordim", "рақамни
# докторга юбордим" -- each a promise that somebody would come back, on a
# conversation nobody had been told about.
_HANDOVER_WORDS = re.compile(
    rf"(doktor|shifokor)(ning)?\s+o{_A}z(i|lari)\s+(sizga\s+)?(javob|yoz|qo{_A}ng{_A}iroq)"
    rf"|(doktor|shifokor)(ga|imizga)\s+(yetkaz|bildir|yubor|xabar\s+ber)"
    rf"|(doktor|shifokor)\s+bilan\s+aniq(lab|lashtir)"
    rf"|(klinika\s+)?bot(i|ga|iga)\s+yubor"
    r"|(доктор|шифокор)(нинг)?\s+ўз(и|лари)\s+(сизга\s+)?(жавоб|ёз|қўнғироқ)"
    r"|(доктор|шифокор)(га|имизга)\s+(етказ|билдир|юбор|хабар\s+бер)"
    r"|(доктор|шифокор)\s+билан\s+ани(қ|к)(лаб|лаштир)"
    r"|(клиника\s+)?бот(и|га|ига)\s+юбор"
    r"|врач\s+(сам|лично)\s+(вам\s+)?(ответ|напиш|позвон)"
    r"|переда(м|дим|л[аи]?)\s+(ваш\w*\s+)?(вопрос\s+|номер\s+|контакт\w*\s+)?врачу"
    r"|уточн(ю|им)\s+у\s+врача|сообщ(у|им)\s+врачу",
    re.IGNORECASE,
)


def extract(reply: str) -> tuple[str, bool]:
    """The reply without the marker, and whether it hands the patient over."""
    marked = bool(DOCTOR_MARKER.search(reply))
    clean = DOCTOR_MARKER.sub(" ", reply).strip() if marked else reply
    return clean, marked or bool(_HANDOVER_WORDS.search(clean))


def same_message(a: str | None, b: str | None) -> bool:
    """Whether two message texts are the same message, whitespace aside."""
    return " ".join((a or "").split()) == " ".join((b or "").split())


async def flag(session: AsyncSession, conversation_id: uuid.UUID) -> None:
    """Mark this conversation as waiting on a person, keeping the earliest time."""
    conversation = await session.get(Conversation, conversation_id)
    if conversation is not None and conversation.needs_doctor_since is None:
        conversation.needs_doctor_since = datetime.now(UTC)
        await session.flush()


async def clear(session: AsyncSession, conversation_id: uuid.UUID) -> None:
    conversation = await session.get(Conversation, conversation_id)
    if conversation is not None and conversation.needs_doctor_since is not None:
        conversation.needs_doctor_since = None
        await session.flush()
