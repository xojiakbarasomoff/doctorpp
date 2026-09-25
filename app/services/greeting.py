"""When the assistant may say hello, decided in code.

The prompt already says "greet once" -- and on real conversations the model
still opened replies with "Va alaykum assalom, Assalomu alaykum, man
urolog-andrologning yordamchisiman..." to a patient in the middle of
describing their complaint, and to a thumbs-up. Whether this is the first
reply, whether the patient came back after a day, whether they greeted just
now: those are facts the backend knows exactly, so the backend decides, and
the reply is tidied to match. Free -- no second call to the model -- and
it never adds words the model did not write, apart from turning a "Va
alaykum assalom" nobody asked for into the "Assalomu alaykum" it should
have been.

    FULL          first reply, or back after 24 hours: greet as the model likes
    RETURN_ONLY   they said salom mid-conversation: greet them back, once
    NONE          mid-conversation: no greeting at all

In every mode, an introduction the patient has already had in this
conversation goes -- unless they have just asked who they are talking to.
"""

import re
from enum import StrEnum

from app.services.conversation_signals import looks_like_a_greeting
from app.services.language import reply_script


class Mode(StrEnum):
    FULL = "full"
    RETURN_ONLY = "return_only"
    NONE = "none"


def mode_for(*, first: bool, long_gap: bool, patient_greeted: bool) -> Mode:
    if first or long_gap:
        return Mode.FULL
    if patient_greeted:
        return Mode.RETURN_ONLY
    return Mode.NONE


# One greeting, at the very start of the text. Spelled the ways the model
# spells them: Latin and Cyrillic Uzbek, Russian, with and without the "va"
# of a return greeting.
_GREETING = re.compile(
    r"^[\W_]*?(?P<g>"
    # Latin Uzbek
    r"(?P<ret_l>va\s*)?(?:as{1,2}al[oa]mu?\s*(?:u\s*)?al[ae]y?ku[mn]"
    r"|al[ae]y?ku[mn](?:\s*as{1,2}al[oa]m)?)"
    r"|salom(?:\s+al[ae]y?kum)?|xayrli\s+(?:kun|tong|kech)|hayrli\s+(?:kun|tong|kech)"
    # Cyrillic Uzbek
    r"|(?P<ret_c>ва\s*)?(?:ас{1,2}ал[оа]му?\s*ал[ае]й?ку[мн]|ал[ае]й?ку[мн](?:\s*ас{1,2}ал[оа]м)?)"
    r"|салом(?:\s+ал[ае]й?кум)?|хайрли\s+(?:кун|тонг|кеч)"
    # Russian
    r"|здравствуйте|здравствуй|добрый\s+(?:день|вечер)|доброе\s+утро|привет"
    # ...and a whole word: "Salomatligingiz" is not a greeting.
    r")(?![^\W_])(?P<sep>[\s,!.;:—–-]*)",
    re.IGNORECASE,
)
# "Assalomu alaykum, Aziz aka! ..." -- the name after a comma belongs to the
# greeting, and goes with it.
_ADDRESS = re.compile(r"^(?:[^\W\d_][\w'’ʻ‘-]*\s*){1,3}!\s*")

# Introducing itself: "Men administratorman", "man urolog-androlog Temur
# Axmadaliyevning yordamchilari bo'laman,", "Я администратор врача." -- and,
# straight after a greeting, without the pronoun: "Va alaykum assalom,
# shifokorning administratori."
_INTRO_WORDS = r"(?:yordamchi|administrator|ёрдамчи|администратор|помощни)"
_INTRO = re.compile(
    r"^(?:men|man|мен|ман|я)\s+[^.!?\n]{0,120}?" + _INTRO_WORDS + r"[^.!?\n,]*[.!,]?\s*",
    re.IGNORECASE,
)
_APPOSITIVE = re.compile(
    r"^(?:[^\W\d_][\w'’ʻ‘-]*\s+){0,3}" + _INTRO_WORDS + r"\w*\s*(?:[.!,]|—|–)\s*",
    re.IGNORECASE,
)
_INTRODUCED = re.compile(_INTRO_WORDS, re.IGNORECASE)

# "Sen kimsan?" -- then who we are is the answer, and stays.
_ASKED_WHO = re.compile(
    r"kim\s*san|kim\s*siz|siz\s+kim|sen\s+kim|bot\s*mi|robot|odam\s*mi|ismingiz"
    r"|ким\s*сан|ким\s*сиз|сиз\s+ким|сен\s+ким|бот\s*ми|одам\s*ми|исмингиз"
    r"|кто\s+вы|вы\s+кто|ты\s+кто|кто\s+ты|\bбот\b|робот|как\s+вас\s+зовут",
    re.IGNORECASE,
)

SALOM_LATIN = "Assalomu alaykum"
SALOM_CYRILLIC = "Ассалому алайкум"


def introduced(earlier_replies: list[str]) -> bool:
    """Whether the assistant already said who it is in this conversation."""
    return any(_INTRODUCED.search(reply) for reply in earlier_replies)


def asked_who(message: str) -> bool:
    """Whether the patient asked who they are talking to."""
    return _ASKED_WHO.search(message) is not None


def _capitalise(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


RETURN = {"uz-latn": "Va alaykum assalom", "uz-cyrl": "Ва алайкум ассалом", "ru": "Здравствуйте"}


def tidy(
    reply: str,
    *,
    mode: Mode,
    patient_greeted: bool,
    introduced_before: bool,
    asked_who: bool = False,
) -> str:
    """The reply with its opening made to fit `mode` -- and a salom never
    left unanswered: the model, told not to greet mid-conversation, also
    stopped greeting back, and not returning a salom is rude. The clinic
    said as much in its own rules."""
    tidied = _tidy(
        reply,
        mode=mode,
        patient_greeted=patient_greeted,
        introduced_before=introduced_before,
        asked_who=asked_who,
    )
    if patient_greeted and mode is not Mode.NONE and _GREETING.match(tidied) is None:
        # In the reply's own alphabet, so a Cyrillic reply never opens in Latin.
        opener = RETURN[reply_script(tidied)]
        return f"{opener}. {_capitalise(tidied.lstrip())}"
    return tidied


def _tidy(
    reply: str,
    *,
    mode: Mode,
    patient_greeted: bool,
    introduced_before: bool,
    asked_who: bool,
) -> str:
    """The reply with its opening made to fit `mode`.

    Only ever removes words from the start of the reply -- repeated
    greetings, an introduction the patient has already had -- and, in one
    case, swaps a "Va alaykum assalom" for "Assalomu alaykum" in the same
    alphabet. Everything else the model wrote, punctuation included, stays.
    """
    rest = reply
    greetings: list[re.Match[str]] = []
    while (match := _GREETING.match(rest)) is not None and match.group("g"):
        greetings.append(match)
        rest = rest[match.end() :]
        if "," in match.group("sep") and (address := _ADDRESS.match(rest)):
            rest = rest[address.end() :]

    stripped_intro = False
    intro = None
    if introduced_before and not asked_who:
        intro = _INTRO.match(rest) or (_APPOSITIVE.match(rest) if greetings else None)
    elif greetings and mode is Mode.NONE and not asked_who:
        # "Va alaykum assalom, shifokorning administratori." with the
        # greeting gone leaves "Shifokorning administratori." standing alone.
        intro = _APPOSITIVE.match(rest)
    if intro is not None:
        rest = rest[intro.end() :]
        stripped_intro = True

    rest = rest.lstrip()
    if (not greetings and not stripped_intro) or not rest:
        # Nothing to tidy -- or nothing but a greeting, which is sent as it
        # was rather than sent empty.
        return reply
    if not greetings or mode is Mode.NONE:
        return _capitalise(rest)

    first = greetings[0]
    said = first.group("g")
    swapped = False
    if mode is Mode.FULL and not patient_greeted and (first.group("ret_l") or first.group("ret_c")):
        # Greeted back somebody who did not greet: greet them instead.
        said = SALOM_CYRILLIC if first.group("ret_c") else SALOM_LATIN
        swapped = True
    if len(greetings) == 1 and not stripped_intro and not swapped:
        return reply
    # Where an introduction came out, a sentence ended; otherwise the model's
    # own punctuation after its greeting stands.
    mark = (first.group("sep").strip() or ".")[0]
    if stripped_intro and mark in ",—–-":
        mark = "."
    return f"{_capitalise(said)}{mark} {rest if mark == ',' else _capitalise(rest)}"


# "Salomatligim" (my health) carries "salom" inside it and is no greeting.
_HEALTH = re.compile(r"sa?lomat|саломат", re.IGNORECASE)


def patient_greeted(message: str) -> bool:
    """Whether the patient said hello in this message."""
    return looks_like_a_greeting(_HEALTH.sub(" ", message))
