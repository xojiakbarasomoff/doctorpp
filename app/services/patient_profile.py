"""What the clinic knows about a patient, kept where it cannot scroll away.

The transcript is not memory. The model is given the tail of a conversation
-- ten turns -- and a name given twenty messages ago is simply not there any
more, which is why a patient who had already said "Asadbek Risqiyev" was
asked for it again, and again after that. The columns to fix it existed from
the first migration and nothing ever wrote to them: users.name and
users.phone.

This writes them, and it writes them carefully. A wrong name is worse than
no name -- it is what the clinic reads out when it rings somebody -- so a
value is only stored when the conversation makes it clear:

* the assistant asked for it and this message is the answer, or
* the patient volunteered it in the shape of a fact ("ismim Asadbek"), or
* it arrived beside a telephone number in a booking (people type both).

"Asadbek aka bilan gaplashmoqchiman" sets nobody's name. Neither does
"rahmat", "ha" or a greeting, all of which have the shape of a first name
and none of which is one.

Nothing here is overwritten by a guess. A stored name is replaced only when
the patient corrects it in as many words ("ismim aslida ..."), and a stored
number only by another number they actually typed.
"""

import logging
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.rag.llm import ChatMessage
from app.services.conversation_signals import looks_like_a_greeting

logger = logging.getLogger(__name__)

# +998 XX XXX XX XX. Stored in one shape so the same patient is one patient:
# the spreadsheet, the appointment row and the call-back list are all keyed
# on this string, and "93 951 11 11" and "+998939511111" are the same person.
_DIGITS = re.compile(r"\d")
_PHONE_RUN = re.compile(r"(?:\+?\d[\d\s().‐-―-]{6,}\d)")
_UZ_OPERATOR_CODES = frozenset(
    {"20", "33", "50", "55", "60", "61", "62", "65", "66", "67", "69", "70",
     "71", "73", "74", "75", "77", "78", "79", "88", "90", "91", "93", "94", "95", "97", "98", "99"}
)  # fmt: skip


def normalise_phone(text: str) -> str | None:
    """+998XXXXXXXXX, or None when this is not a telephone number.

    Nine digits is a subscriber number, twelve is one with the country code,
    and thirteen with a leading +. Anything else -- a year, a price, a
    passport number, "2026-09-24" -- is not a number anybody can ring.
    """
    run = _PHONE_RUN.search(text)
    if run is None:
        return None
    digits = "".join(_DIGITS.findall(run.group(0)))
    if digits.startswith("998"):
        digits = digits[3:]
    elif digits.startswith("00998"):
        digits = digits[5:]
    if len(digits) != 9:
        return None
    # An Uzbek subscriber number starts with a two-digit operator code. This
    # is what tells a telephone number from nine digits of something else.
    if digits[:2] not in _UZ_OPERATOR_CODES:
        return None
    return f"+998{digits}"


# "Ismim Asadbek", "mening ismim ...", "меня зовут ...". The patient saying
# it as a fact, which is as good as answering the question.
_SAYS_THEIR_NAME = re.compile(
    # "ismim aslida Asadbek" — the correction word belongs to the sentence,
    # not to the name, and a patient called "aslida Asadbek" is nobody.
    r"(?:mening\s+)?is[mi]\w*\s+(?:aslida\s+|—|-|:)?\s*"
    r"(?P<name>[^\W\d_][\w'’ʻʼ`-]*(?:\s+[^\W\d_][\w'’ʻʼ`-]*){0,3})"
    r"|мен(?:инг)?\s+исм\w*\s+(?P<cyr>[^\W\d_][\w'’ʻʼ`-]*(?:\s+[^\W\d_][\w'’ʻʼ`-]*){0,3})"
    r"|меня\s+зовут\s+(?P<ru>[^\W\d_][\w'’ʻʼ`-]*(?:\s+[^\W\d_][\w'’ʻʼ`-]*){0,3})",
    re.IGNORECASE,
)

# The assistant's own question, so that the message after it can be read as
# the answer to it.
_ASKED_FOR_A_NAME = re.compile(
    r"\bis[mi]\w*|\bfamiliya\w*|\bf\.?i\.?sh\b|исм\w*|фамили\w*|\bимя\b|\bзовут\b",
    re.IGNORECASE,
)

# Words that have the shape of a name and are not one. Kept short: this is
# the last line of defence, not the first -- the question above the message
# is what makes it an answer.
_NOT_A_NAME = frozenset(
    {
        "ha", "xa", "yoq", "yo'q", "ok", "okey", "mayli", "shu", "shuni",
        "rahmat", "raxmat", "tushunarli", "bo'ldi", "boldi", "zor", "yaxshi",
        "qabul", "narx", "uzi", "tekshiruv", "shifokor", "doktor", "salom",
        "да", "нет", "хорошо", "спасибо", "приём", "прием", "врач",
    }
)  # fmt: skip

# A name is one to four words of letters. Somebody who types a sentence is
# not answering "what is your name".
_NAME_SHAPE = re.compile(
    r"^[^\W\d_][\w'’ʻʼ`-]*(?:\s+[^\W\d_][\w'’ʻʼ`-]*){0,3}$", re.UNICODE
)
_MAX_NAME_LENGTH = 48

# A correction: "ismim aslida ...", "yo'q, ismim ...". Only this replaces a
# name the clinic already holds.
_CORRECTS_THEIR_NAME = re.compile(
    r"\b(?:aslida|noto'g'ri|xato|emas)\b|^\s*(?:yo'q|yoq|нет)\b", re.IGNORECASE
)


def looks_like_a_name(text: str) -> bool:
    stripped = text.strip().rstrip(".!,")
    if not stripped or len(stripped) > _MAX_NAME_LENGTH:
        return False
    if stripped.lower() in _NOT_A_NAME or looks_like_a_greeting(stripped):
        return False
    if normalise_phone(stripped) is not None or any(ch.isdigit() for ch in stripped):
        return False
    return bool(_NAME_SHAPE.match(stripped))


@dataclass(frozen=True)
class Found:
    """What this turn said about who the patient is."""

    name: str | None = None
    phone: str | None = None
    corrects: bool = False


def read_turn(message: str, *, asked_for_name: bool) -> Found:
    """The name and number in one message, or nothing.

    `asked_for_name` is whether the assistant's previous turn asked for it,
    which is what makes a bare "Asadbek Risqiyev" an answer rather than a
    passing mention.
    """
    phone = normalise_phone(message)
    corrects = bool(_CORRECTS_THEIR_NAME.search(message))

    said = _SAYS_THEIR_NAME.search(message)
    if said is not None:
        name = said.group("name") or said.group("cyr") or said.group("ru")
        if name and looks_like_a_name(name):
            return Found(name=name.strip(), phone=phone, corrects=corrects)

    if asked_for_name:
        # The answer to "ismingizni yozing" is whatever is left once the
        # telephone number they typed beside it is taken out: people send
        # "Asadbek Risqiyev 93 951 11 11" as one message.
        without_number = message
        if phone is not None:
            run = _PHONE_RUN.search(message)
            if run is not None:
                without_number = message.replace(run.group(0), " ")
        candidate = without_number.strip(" ,.-—\n")
        if looks_like_a_name(candidate):
            return Found(name=candidate, phone=phone, corrects=corrects)

    return Found(phone=phone, corrects=corrects)


def asked_for_a_name(history: Sequence[ChatMessage] | None) -> bool:
    """Whether the last thing the assistant said was asking for a name."""
    for turn in reversed(list(history or [])):
        if turn.get("role") == "assistant":
            return bool(_ASKED_FOR_A_NAME.search(turn.get("content") or ""))
    return False


async def remember(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    message: str,
    history: Sequence[ChatMessage] | None,
) -> Found:
    """Store what this turn said about the patient, and return it.

    Additive by default: an empty column is filled, a column that already
    holds something is left alone unless the patient is correcting it. The
    clinic rings these numbers, so the bar for replacing one is a patient
    saying so, not a regex being confident.
    """
    found = read_turn(message, asked_for_name=asked_for_a_name(history))
    if found.name is None and found.phone is None:
        return found

    user = await session.get(User, user_id)
    if user is None:  # pragma: no cover - the caller has just used this row
        return found

    changed: list[str] = []
    if found.name and (user.name is None or (found.corrects and found.name != user.name)):
        user.name = found.name
        changed.append("name")
    if found.phone and (user.phone is None or found.phone != user.phone):
        # A number is replaced by a number: they typed it, it is the one they
        # want to be rung on, and the old one is no use to anybody.
        user.phone = found.phone
        changed.append("phone")

    if changed:
        await session.flush()
        logger.info(
            "patient_profile_updated",
            extra={"user_id": str(user_id), "fields": ",".join(changed)},
        )
    return found


@dataclass(frozen=True)
class Profile:
    """The patient as the clinic holds them, for the prompt and the flow."""

    name: str | None
    phone: str | None

    @property
    def known(self) -> bool:
        return bool(self.name or self.phone)


async def load(session: AsyncSession, user_id: uuid.UUID) -> Profile:
    user = await session.get(User, user_id)
    if user is None:
        return Profile(name=None, phone=None)
    return Profile(name=user.name, phone=user.phone)
