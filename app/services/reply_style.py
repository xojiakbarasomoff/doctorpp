"""What the patient is allowed to be sent, checked after the model wrote it.

The system prompt says all of this already: greet somebody who greeted you
and nobody else, say who you are once, ask one question at a time, write
dates the way a person says them, stay in one alphabet. A prompt is a
request, and across one real conversation of ninety-one messages the
request lost: the assistant opened thirteen messages in a row with "Ва
алайкум ассалом, шифокорнинг администратори" to a patient who had not said
hello since the first day, asked for the same telephone number three times,
printed "2026-09-19" at a patient, and said "men CALLBACK yozaman" out loud.

So this is the part that holds, in two halves.

`tidy` repairs what can be repaired without a second opinion -- a greeting
nobody gave, a self-introduction in the fortieth message, the word CALLBACK.
Cutting an opening clause cannot change what the reply means, which is what
makes it safe to do silently.

`problems` names what cannot be repaired that way -- two alphabets in one
sentence, two questions in one message, a reply four hundred characters
long. Those need the sentence rewritten rather than trimmed, and the caller
(app.services.answer) is what asks the model to rewrite it, once. A reply
that comes back broken twice is still sent, tidied: a patient waiting on an
answer is worse served by silence than by an inelegant message.

Nothing here is about medicine. app.services.guardrail is what stands
between a patient and a dose; this is about a front desk that sounds like a
person rather than a script with a loop in it.
"""

import logging
import re

logger = logging.getLogger(__name__)

# The markers are machinery: [[BOOK:...]] and [[CALLBACK:...]]. Everything
# below runs on the text with them lifted out, or a rule about the word
# "CALLBACK" would eat the marker that makes the call-back happen.
_MARKER = re.compile(r"\[\[[^\]]*\]\]")
_PLACEHOLDER = "\x00marker%d\x00"

# A greeting at the front of a message: "Assalomu alaykum", "Ва алайкум
# ассалом", "Здравствуйте", with whatever punctuation follows it. Anchored,
# because a greeting in the middle of a sentence is the patient's own words
# being quoted back and is none of this function's business.
_OPENING_GREETING = re.compile(
    r"^\s*(?:va\s*)?(?:a?ssalom\w*|вa?а?лейкум|ваалейкум|ва\s*алайкум|ассалом\w*"
    r"|алайкум\s*ассалом|алейкум\s*ассалом|здравствуйте|привет|salom|салом)"
    r"[\s,!.—-]*(?:alaykum|aleykum|алайкум|алейкум|ассалом)?[\s,!.—-]*",
    re.IGNORECASE,
)

# "shifokorning administratori", in every way this inbox says it: Uzbek in
# both alphabets and Russian. The Cyrillic Uzbek spelling matters most --
# that is the one the assistant opened thirteen messages with.
_IDENTITY = re.compile(
    r"(?:men\s+|мен\s+)?(?:shifokorning|шифокор(?:нинг)?|доктор(?:нинг)?|врача)?\s*"
    r"(?:administrator(?:i|man|iman)?|администратор(?:и|ман|иман)?)"
    r"(?:\s+(?:shifokorning|шифокорнинг|врача))?[\s,.!—-]*",
    re.IGNORECASE,
)

# Somebody asking who they are writing to. Then the introduction is the
# answer to their question, not a tic, and it stays.
_ASKS_WHO = re.compile(
    r"\bkim(?:siz|san|bilan|man)?\b|\bkim gapiryapti\b|\bbu kim\b"
    r"|\bкто (?:вы|это)\b|\bс кем\b|\bким(?:сиз)?\b",
    re.IGNORECASE,
)

# The words that only exist inside this codebase. A patient reading "men
# CALLBACK yozaman" has been shown the wiring.
_MACHINERY = re.compile(
    r"\s*\[?\[?\b(?:BOOK|CALLBACK|MARKER|SYSTEM|PROMPT)\b\]?\]?\s*", re.IGNORECASE
)

# 2026-09-19 — a machine's way of saying a date, forbidden in the prompt and
# written anyway. Rewritten rather than flagged: the day is correct, only
# its clothes are wrong.
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

_LATIN = re.compile(r"[a-z]", re.IGNORECASE)
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")

# How much of the other alphabet is a slip rather than a mix. A brand, a
# handle or a word like "UZI" inside a Cyrillic sentence is not the failure
# this is looking for; half a message in the other script is.
_SCRIPT_TOLERANCE = 0.15

# Long enough for a time, a day and a question; short enough to read on a
# phone between patients. The replies that went wrong were consistently the
# long ones.
MAX_CHARS = 400

# One question per message is a rule of the prompt. Three question marks in
# one reply is the assistant asking for the name, the number and the reason
# at once, which is how a patient answers only the last of them.
MAX_QUESTIONS = 1


def _mask(reply: str) -> tuple[str, list[str]]:
    markers: list[str] = []

    def take(match: re.Match[str]) -> str:
        markers.append(match.group(0))
        return _PLACEHOLDER % (len(markers) - 1)

    return _MARKER.sub(take, reply), markers


def _unmask(reply: str, markers: list[str]) -> str:
    for index, marker in enumerate(markers):
        reply = reply.replace(_PLACEHOLDER % index, marker)
    return reply


def _strip_opening(text: str, pattern: re.Pattern[str]) -> str:
    match = pattern.match(text)
    if match is None or not match.group(0).strip():
        return text
    rest = text[match.end() :].lstrip(" ,.!—-")
    # Never leave the patient with nothing. A reply that was only a greeting
    # is a reply whose greeting was the point.
    return rest or text


def tidy(reply: str, *, greeted: bool, opening: bool, user_message: str = "") -> str:
    """The reply with the openings it should not have had taken off.

    Only the front of the message and only fixed phrases, so what the
    assistant actually said is untouched.

    `opening` is no longer enough on its own to keep the introduction. A
    live audit marked sixteen replies in forty-nine as sounding like a
    machine, and every one of them was the same thing: "Shifokorning
    administratori —" stuck on the front of an answer nobody had asked to
    be introduced to. A person says who they are when they are greeted or
    when they are asked. The rest of the time they just answer.
    """
    body, markers = _mask(reply)

    if not greeted:
        body = _strip_opening(body, _OPENING_GREETING)
    if not (greeted and opening) and not _ASKS_WHO.search(user_message):
        body = _strip_opening(body, _IDENTITY)
        # The same phrase as a sentence of its own, mid-conversation: the
        # assistant took to prefixing every message with it once the greeting
        # was gone.
        body = re.sub(
            r"(?m)^\s*(?:men\s+|мен\s+)?(?:shifokorning\s+administratori"
            r"|шифокорнинг\s+администратори)[\s,.!—-]*",
            "",
            body,
            flags=re.IGNORECASE,
        )

    body = _MACHINERY.sub(" ", body)
    body = _ISO_DATE.sub(lambda m: f"{m.group(3)}.{m.group(2)}.{m.group(1)}", body)
    # A sentence that ends on a colon promised something that never came:
    # "Iltimos kelish eslatmasi:" was sent to a patient exactly like that.
    body = re.sub(r"[:\s]+$", "", body)
    body = re.sub(r"[ \t]{2,}", " ", body).strip()
    body = _capitalised(body)
    return _unmask(body, markers) if body else reply


def _capitalised(text: str) -> str:
    """A message starts with a capital letter.

    Trimming an opening clause leaves whatever followed it, and what
    followed it was written as the middle of a sentence: cutting
    "Shifokorning administratori — " off the front sent a patient "siz
    Axmadaliyev Temur G'iyosiddin o'g'liga yozayapsiz", lowercase. Nobody
    types that, which makes it exactly the kind of tell this file exists
    to remove.
    """
    for index, character in enumerate(text):
        if character.isalpha():
            return text[:index] + character.upper() + text[index + 1 :]
        if not character.isspace() and character not in "\"'“«(":
            break
    return text


def _script_mix(text: str) -> float:
    latin = len(_LATIN.findall(text))
    cyrillic = len(_CYRILLIC.findall(text))
    total = latin + cyrillic
    return min(latin, cyrillic) / total if total else 0.0


def dominant_script(text: str) -> str | None:
    latin = len(_LATIN.findall(text))
    cyrillic = len(_CYRILLIC.findall(text))
    if latin == cyrillic:
        return None
    return "latin" if latin > cyrillic else "cyrillic"


def problems(reply: str, *, script: str, greeted: bool) -> list[str]:
    """What is wrong with this reply that trimming cannot fix.

    Each entry is written for the model to read: it goes back as the
    instruction for the rewrite.
    """
    body, _ = _mask(reply)
    found: list[str] = []

    if _script_mix(body) > _SCRIPT_TOLERANCE:
        found.append(
            "You mixed two alphabets in one message. Write every word in one alphabet."
        )
    wanted = "cyrillic" if script in {"uz-cyrl", "ru"} else "latin"
    actual = dominant_script(body)
    if actual is not None and actual != wanted:
        found.append(
            f"You answered in the {actual} alphabet; this patient writes in the "
            f"{wanted} one. Rewrite it in {wanted} letters."
        )
    if body.count("?") > MAX_QUESTIONS:
        found.append(
            "You asked more than one question. Ask one thing, in one sentence, "
            "and wait for the answer."
        )
    if len(body) > MAX_CHARS:
        found.append(
            f"The reply is {len(body)} characters. Say the same thing in under "
            f"{MAX_CHARS}, in at most three short sentences."
        )
    if not greeted and _OPENING_GREETING.match(body):
        found.append("They did not greet you. Do not open with a greeting.")
    if _RECEIPT.search(body):
        found.append(
            "You opened by reading the patient's own words back to them "
            '("35 yosh ekansiz", "2 oydan beri ekanini tushundim"). That is '
            "a receipt, not an answer. Start with the answer."
        )
    if _BAND_ON_A_FREE_SLOT.search(body):
        found.append(
            'You called free times "band". "Band" means taken; a free slot '
            'is "bo\'sh". Say it the way the patient will read it.'
        )
    if _OFFERS_TO_FIND_OUT.search(body):
        found.append(
            "You offered to find something out, ring somebody, or come back "
            "later. You cannot do any of those. Say plainly what you do not "
            "know, give the clinic's number once, and answer the rest."
        )
    return found


# "Tekshirib beraman", "so'rab qo'yaman", "aniqlab beramiz", "узнаю" -- a
# promise nobody in this inbox can keep. The patient waits for an answer
# that is never coming, which is worse than being told to ring.
# The receipt: the patient's own words read back before the answer. "Siz 35
# yosh ekansiz va muammo bor ekan — tushundim", "Sizning so'rovingiz:
# Online korik". Narrow on purpose -- "Tushundim, buyrak og'rig'i" is the
# assistant showing a frightened patient it read them, and that stays.
_RECEIPT = re.compile(
    r"\bekan(?:siz|ini|ligini)\b[^.!?]{0,30}tushundim"
    # And the same sentence written the other way round, which is how it
    # came back after the first version of this rule: "Tushundim — 2 oydan
    # beri muammo bor ekan", "80 yoshda va siydik ushlanmayapti — tushundim".
    r"|tushundim[^.!?]{0,60}\bekan\b"
    r"|^[^.!?]{0,60}\s[—-]\s*tushundim\b"
    r"|^\s*siz\b[^.!?]{0,60}\bekansiz\b"
    r"|sizning so[o'’ʻ]?rovingiz\s*[:\"«]"
    r"|\bдеб ёздингиз\b|\bваш запрос\s*:",
    re.IGNORECASE,
)

# "Ertaga 09:00, 09:20 yoki 09:40 band" -- said of slots that are free.
# "Band" is what a taken slot is, and a patient reading it is being told
# the opposite of what was meant.
_BAND_ON_A_FREE_SLOT = re.compile(
    r"\d{1,2}:\d{2}[^.!?]{0,60}\bband\b(?!\s*emas)"
    r"|\bband\b(?!\s*emas)[^.!?]{0,30}\d{1,2}:\d{2}[^.!?]{0,30}\d{1,2}:\d{2}",
    re.IGNORECASE,
)

_OFFERS_TO_FIND_OUT = re.compile(
    r"tekshir\w*\s+(?:ber|qo)\w*|aniqla\w*\s+(?:ber|qo)\w*|so[o'’ʻ]?ra\w*\s+(?:ber|qo)\w*"
    r"|bilib\s+(?:ber|ol)\w*|текшир\w*\s+бер\w*|аниқла\w*\s+бер\w*|сўра\w*\s+(?:бер|қў)\w*"
    r"|узна(?:ю|ем)|уточн(?:ю|им)\s|перезвон(?:ю|им)",
    re.IGNORECASE,
)


REWRITE_INSTRUCTION = (
    "\n\nYOUR PREVIOUS REPLY WAS REJECTED\n"
    "It broke these rules:\n{problems}\n"
    "Write the same answer again, correctly. Same facts, same language, same "
    "booking marker if there was one — only the wording changes. Send only the "
    "corrected reply."
)
