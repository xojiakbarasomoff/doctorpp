import re
from collections.abc import Sequence

from app.rag.llm import ChatMessage

_UZBEK_CYRILLIC = frozenset("ўқғҳ")


def reply_script(user_message: str) -> str:
    """Which of "uz-latn", "uz-cyrl", "ru" to answer a fixed line in.

    A deliberately small rule rather than a language detector, for the few
    lines that are not written by the model (a voice note, a comment's
    public reply). Anything not Cyrillic is answered in Uzbek Latin.
    """
    lowered = user_message.lower()
    if any(letter in lowered for letter in _UZBEK_CYRILLIC):
        return "uz-cyrl"
    if any("Ѐ" <= character <= "ӿ" for character in lowered):
        return "ru"
    return "uz-latn"


# Whether a message has any letters at all: "93 444 444" and "11:00" are the
# two most common messages this inbox gets, and neither is evidence of an
# alphabet.
_HAS_LETTERS = re.compile(r"[^\W\d_]", re.UNICODE)


def conversation_script(history: Sequence[ChatMessage] | None, user_message: str) -> str:
    """The alphabet this conversation is being held in.

    Per conversation rather than per message: a patient who wrote Latin
    throughout and then sent a telephone number was answered in Cyrillic.
    The patient's own words decide it, newest first; what the assistant
    wrote does not, or one reply in the wrong alphabet would justify the
    next.
    """
    written = [user_message]
    if history:
        written.extend(m["content"] for m in reversed(history) if m.get("role") == "user")
    for message in written:
        if _HAS_LETTERS.search(message):
            return reply_script(message)
    return reply_script(user_message)
