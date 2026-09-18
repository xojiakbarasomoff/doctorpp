"""What the clinic already knows about a booking in progress.

The assistant collects three things before it offers a time: a full name, a
telephone number and the reason for the visit. Which of them it already has
was left to the model to work out from the conversation, and in a real
transcript it could not: it asked for the same number three times in four
messages, made the patient confirm "shunchaki konsultatsiya" twice, and then
asked for the name of a patient whose name was in the message it was
answering.

Self-inspection is the thing models are worst at, and this is the same
argument app.services.conversation_signals already makes about greetings:
"have you already asked?" is a question about text, and code reads text more
reliably than a model reads its own history.

So the three fields are extracted here, from the conversation, and written
into the prompt as facts -- with the one question that is still open named
outright. The model is left with what it is good at: the sentence.

Read only. Nothing here books anything; app.services.booking still settles
the marker, and the row in the database is still the only record.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.rag.llm import ChatMessage
from app.services.conversation_signals import (
    find_phone_number,
    looks_like_a_greeting,
    looks_like_a_phone_number,
)

# The assistant's own questions, as it asks them. Matched against its earlier
# turns, because what makes a patient's "Asadbek Risqiyev" a name rather than
# a complaint is the question above it.
_ASKED_NAME = re.compile(
    # Uzbek agglutinates: the question is "ismingizni yozing", so the stem is
    # matched with whatever is stuck to it. An exact "\bism\b" matched none
    # of the ways this assistant actually asks.
    r"\bis[mi]\w*|\bfamiliya\w*|\bf\.?i\.?sh\b|исм\w*|фамили\w*|\bимя\b",
    re.IGNORECASE,
)
_ASKED_PHONE = re.compile(
    r"\b(?:telefon|raqam\w*|nomer\w*)\b|телефон|рақам|номер", re.IGNORECASE
)
_ASKED_REASON = re.compile(
    r"\b(?:sabab\w*|shikoyat\w*|murojaat\w*)\b|сабаб|шикоят|шикаоят|жалоб|причин",
    re.IGNORECASE,
)

# A turn is only asking for something if it asks. Without this, handing a
# patient the clinic's telephone number -- a turn with the word "telefon" in
# it -- read as "we have started booking them", and the next reply asked a
# patient who wanted the opening hours for their full name.
_IS_A_QUESTION = re.compile(
    r"\?|\byoz(?:ing|asiz|ib)\b|\byubor(?:ing|asiz)\b|\bayting\b|\bqoldiring\b"
    r"|\bkirit(?:ing|asiz)\b|ёзинг|юборинг|айтинг|қолдиринг|киритинг"
    r"|напишите|отправьте|укажите|оставьте",
    re.IGNORECASE,
)

# A patient saying they want to come. Enough on its own to put the three
# questions in play, because it is what starts them at the front desk too.
_WANTS_TO_COME = re.compile(
    r"qabulga\s+yoz|yozilmoqchi|yozib\s+qo|navbat|band\s+qil"
    r"|қабулга\s+ёз|ёзилмоқчи|навбат"
    r"|запис(?:ать|аться|ите)|на\s+приём|на\s+прием",
    re.IGNORECASE,
)

# A name is short, has no digits in it, and is not a greeting. Two or three
# words in Uzbek ("Asadbek Risqiyev", "Xurshid Alimov o'g'li"); one is
# accepted because plenty of people answer with just their first name.
_NAME = re.compile(r"^[^\W\d_][\w'’ʻʼ`-]*(?:\s+[^\W\d_][\w'’ʻʼ`-]*){0,3}$", re.UNICODE)
_MAX_NAME_LETTERS = 48

# Everything a telephone number is made of, so what is left of a message
# that carried one can be read as the name beside it.
_DIGITS = re.compile(r"[+\d][\d\s().-]{4,}")

# "Ha", "shu", "tasdiqla" -- an answer to the question above it, not a new
# fact. Taken as a reason, they fill the clinic's column with the word "Ha".
_AGREEMENT = frozenset(
    {
        "ha",
        "xa",
        "ok",
        "okey",
        "mayli",
        "shu",
        "shuni",
        "tasdiqla",
        "tasdiqlayman",
        "да",
        "ха",
        "хорошо",
        "подтверждаю",
    }
)


def _is_agreement(text: str) -> bool:
    return text.strip().lower().strip("!?.,") in _AGREEMENT


def _looks_like_a_name(text: str) -> bool:
    stripped = text.strip().rstrip(".!,")
    if not stripped or len(stripped) > _MAX_NAME_LETTERS:
        return False
    if looks_like_a_phone_number(stripped) or looks_like_a_greeting(stripped):
        return False
    return bool(_NAME.match(stripped))


@dataclass(frozen=True)
class BookingState:
    """The three things a booking needs, and which of them are in hand."""

    name: str | None
    phone: str | None
    reason: str | None
    # Whether anything at all says this patient is being booked: they asked
    # to be, or the assistant has already started collecting.
    in_progress: bool

    @property
    def next_needed(self) -> str | None:
        """The one question still open, in the order the clinic asks them."""
        if self.name is None:
            return "name"
        if self.phone is None:
            return "phone"
        if self.reason is None:
            return "reason"
        return None


def read(history: Sequence[ChatMessage] | None, user_message: str) -> BookingState:
    """What has already been said, from the conversation itself.

    The patient's turns are read in order, each one in the light of the
    assistant's question before it -- which is what tells "Asadbek Risqiyev"
    apart from a complaint and "Buyrak ogrigi" apart from a name.
    """
    turns: list[ChatMessage] = [*(history or []), ChatMessage(role="user", content=user_message)]

    name: str | None = None
    phone: str | None = None
    reason: str | None = None
    asked_anything = False
    wants_to_come = False
    last_question = ""

    for turn in turns:
        text = (turn.get("content") or "").strip()
        if not text:
            continue
        if turn.get("role") == "assistant":
            asking = bool(_IS_A_QUESTION.search(text))
            last_question = text if asking else ""
            if asking and (
                _ASKED_NAME.search(text) or _ASKED_PHONE.search(text) or _ASKED_REASON.search(text)
            ):
                asked_anything = True
            continue

        if _WANTS_TO_COME.search(text):
            wants_to_come = True

        # A number is a number wherever it appears: patients send one
        # unprompted as often as they are asked for it.
        found = find_phone_number(text)
        if found and phone is None:
            phone = found
            # "Asadbek Risqiyev 93 9510000" is one message carrying two
            # answers. Taking the number and moving on left the assistant
            # asking for a name it was holding.
            text = _DIGITS.sub(" ", text).strip(" ,.-—")
            if not text:
                continue

        if name is None and _ASKED_NAME.search(last_question) and _looks_like_a_name(text):
            name = text.strip().rstrip(".!,")
            continue

        if reason is None and _ASKED_REASON.search(last_question) and not _is_agreement(text):
            if not looks_like_a_greeting(text) and not _looks_like_a_name(text):
                reason = text.strip()
                continue
            # "Buyrak" is one word and matches the name shape; a reason was
            # asked for, so that is what it is.
            if not looks_like_a_greeting(text):
                reason = text.strip()

    return BookingState(
        name=name,
        phone=phone,
        reason=reason,
        in_progress=wants_to_come or asked_anything,
    )


_LABELS = {
    "name": "their full name — and whose name, if they are booking for somebody else",
    "phone": "their telephone number",
    "reason": "the reason they are coming, in their own words",
}

# Said every time the section appears, because the section caused this.
#
# Naming the missing field turned the assistant into a form: a patient who
# asked "nechida kelay?" was answered "ismingizni yozing", and one who asked
# the address mid-booking got the next question instead of the address.
# Collecting is what it does while it answers, never instead of answering.
_ANSWER_FIRST = (
    "\n- This is a list, not a script. If their message asked you anything "
    "at all — a time, the address, whether you do something — answer that "
    "first, in the same message, and then ask for what is missing. A "
    "patient whose question is met with the next form field has been told "
    "nobody read it."
)


def render(state: BookingState) -> str:
    """The state as a prompt section, or nothing when no booking is running."""
    if not state.in_progress and state.next_needed == "name":
        return ""

    known = [
        f"- Their name: {state.name}" if state.name else None,
        f"- Their telephone number: {state.phone}" if state.phone else None,
        f"- Why they are coming: {state.reason}" if state.reason else None,
    ]
    lines = [line for line in known if line]

    section = "\n\nWHAT YOU ALREADY HAVE FOR THIS BOOKING"
    if lines:
        section += "\n" + "\n".join(lines)
        section += (
            "\n- Never ask for any of these again, and never ask them to "
            "confirm one. They said it; it is written down. Asking twice is "
            "how a patient learns that nobody is reading."
        )
    else:
        section += "\n- Nothing yet."

    if state.next_needed is None:
        section += (
            "\n- You have all three. Do not ask for anything else: offer a "
            "time from THE APPOINTMENT BOOK, and when they accept one, "
            "confirm it with the booking marker."
        )
    else:
        section += (
            f"\n- The one thing still missing is {_LABELS[state.next_needed]}. "
            "Ask for that one thing only — never two of them in one message."
        )
    return section + _ANSWER_FIRST
