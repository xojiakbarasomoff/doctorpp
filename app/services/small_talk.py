"""A thumbs-up, an "ok", a "rahmat": whether to answer at all, and how.

Answering every message is what a vending machine does. A patient who sends
👍 after "Ertaga soat 10:00 da kutamiz" has finished; the assistant writing
back "Eshitaman" -- or worse, greeting them from the top and introducing
itself, which real conversations showed it doing twice in a row to the same
👍 -- restarts a conversation that was over. A person at a front desk says
nothing, and so does this.

But the same 👍 after "Ertaga soat 10 sizga qulaymi?" is a yes, and silence
there loses a booking. What decides it is what the clinic said last: a
question waits for an answer, anything else does not. And a patient writing
only an emoji to start a conversation gets one short hello and a question,
not silence.

Decided in code, before the model is asked, because the one thing the model
cannot do is decide not to answer.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.rag.llm import ChatMessage
from app.services import message_labels

# Words that on their own say "understood / thanks / yes / bye" and nothing
# else. A message is an acknowledgement only if every word in it is one of
# these -- "ok, ertaga kelaman" is not.
_ACK_WORDS = frozenset(
    {
        # agreeing, understood
        "ok",
        "okay",
        "okey",
        "oke",
        "okk",
        "ок",
        "окей",
        "окк",
        "xop",
        "hop",
        "xo",
        "хоп",
        "хўп",
        "хуп",
        "ладно",
        "хорошо",
        "понятно",
        "понял",
        "поняла",
        "ясно",
        "tushunarli",
        "tushundim",
        "тушунарли",
        "тушундим",
        "mayli",
        "майли",
        "bopti",
        "бўпти",
        "бопти",
        "zor",
        "зўр",
        "зор",
        "super",
        "супер",
        "yaxshi",
        "яхши",
        "ha",
        "xa",
        "ха",
        "да",
        "ага",
        "yes",
        # thanks
        "rahmat",
        "raxmat",
        "rahmatt",
        "rahmet",
        "рахмат",
        "раҳмат",
        "рахмет",
        "спасибо",
        "спс",
        "благодарю",
        "большое",
        "thanks",
        "thank",
        "you",
        "katta",
        "kotta",
        "катта",
        "juda",
        "жуда",
        "sizga",
        "сизга",
        "ham",
        "хам",
        "ҳам",
        # goodbye
        "xayr",
        "хайр",
        "sog",
        "boling",
        "саломат",
        "бўлинг",
        "булинг",
    }
)
_MAX_ACK_WORDS = 5
_MAX_ACK_CHARS = 40
_WORD = re.compile(r"[^\W\d_]+")
_DIGIT = re.compile(r"\d")
_APOSTROPHES = re.compile(r"['’ʻ‘`]")


def is_acknowledgement(text: str) -> bool:
    """Only emoji or punctuation, or only words like ok / rahmat / ha."""
    stripped = (text or "").strip()
    if not stripped or len(stripped) > _MAX_ACK_CHARS or "?" in stripped:
        return False
    if _DIGIT.search(stripped):
        return False  # "10", a date, a telephone number: an answer
    words = _WORD.findall(_APOSTROPHES.sub("", stripped.lower()))
    if not words:
        return True  # 👍, ❤️🙏, "+", "..."
    return len(words) <= _MAX_ACK_WORDS and all(word in _ACK_WORDS for word in words)


def is_emoji_only(text: str) -> bool:
    stripped = (text or "").strip()
    return bool(stripped) and not _WORD.search(stripped) and not _DIGIT.search(stripped)


def _wordless(text: str) -> bool:
    """A patient turn with no words at all: emoji, or the label for a
    sticker or a reaction."""
    return is_emoji_only(text) or (
        message_labels.is_label(text)
        and text.lstrip().startswith((message_labels.STICKER, message_labels.REACTION))
    )


@dataclass(frozen=True)
class Plan:
    # Say nothing to this message.
    silent: bool = False
    # Or answer it, with this said to the model about what the message means.
    note: str | None = None


_ANSWER = Plan()

_STARTING = (
    "# BU XABAR HAQIDA\n"
    'Bemor so\'z yozmadi, faqat "{text}" yubordi -- suhbat shu bilan '
    "boshlanmoqda. Qisqa salomlashing va qanday yordam kerakligini bitta "
    "savol bilan so'rang. Boshqa hech narsa yozmang."
)
_REPLYING = (
    "# BU XABAR HAQIDA\n"
    'Bemor so\'z yozmadi, faqat "{text}" yubordi -- bu sizning oxirgi '
    "savolingizga javob. Savolingiz ha/yo'q savoli bo'lsa, buni \"ha, "
    "roziman\" deb tushuning va shunga mos davom eting. Aks holda o'sha "
    "savolni qayta bermang: bir gap bilan, boshqacha so'z bilan aniqlashtiring. "
    "Salomlashmang."
)


def plan(message: str, history: Sequence[ChatMessage], *, long_gap: bool) -> Plan:
    """What to do with this batch of the patient's messages."""
    if not is_acknowledgement(message):
        return _ANSWER
    if not history or long_gap:
        # Nothing of ours to acknowledge: this opens a conversation.
        if is_emoji_only(message):
            return Plan(note=_STARTING.format(text=message.strip()[:20]))
        return _ANSWER
    last = next((turn for turn in reversed(history) if turn["role"] == "assistant"), None)
    if last is None:
        return _ANSWER
    if "?" not in last["content"]:
        # Nothing was asked; "ok", "rahmat", 👍 close the exchange.
        return Plan(silent=True)
    if not is_emoji_only(message):
        # "ha", "ok", "rahmat" to a question: words, and the model reads them.
        return _ANSWER
    # An emoji to a question. If our question was itself the answer to an
    # emoji, this is not a reply to it -- answering again is how the
    # ping-pong starts.
    position = max(i for i, turn in enumerate(history) if turn is last)
    before = next((turn for turn in reversed(history[:position]) if turn["role"] == "user"), None)
    if before is not None and _wordless(before["content"]):
        return Plan(silent=True)
    return Plan(note=_REPLYING.format(text=message.strip()[:20]))
