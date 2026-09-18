"""What the assistant can already tell about a conversation before it answers.

The prompt used to state its two social rules unconditionally: return the
greeting, and close by asking for a phone number. Applied to every turn,
that produces a correspondent who says hello five times and ends every
single message asking for a number nobody is refusing to give -- which reads
as a machine running a script, and makes the request itself easy to ignore.

Both rules are really conditional, and the conditions are things code can
establish more reliably than a model reading its own history: is this the
first message, has a number already been given, and have we asked before.
Establishing them here keeps the prompt free of "work out whether you have
already asked", which is exactly the kind of self-inspection models are
worst at.

None of this is a hard gate. The signals are rendered into the prompt as
facts, and the rules that use them are written so the failure mode is
asking too little rather than too much -- a patient who is never asked can
still be asked next turn, while a patient asked in every message has already
decided what this is.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.rag.llm import ChatMessage

# Nine digits is the shortest real Uzbek subscriber number (90 123 45 67),
# and twelve covers +998 with the country code. The separators people type
# between the groups are stripped first.
#
# Deliberately not matching shorter runs: a clinic that quotes "3 500 000
# so'm" or answers "09:00 - 20:00" would otherwise look like a patient who
# has already left a number, and the assistant would stop asking.
_SEPARATORS = re.compile(r"[\s\-()+.]")
_PHONE = re.compile(r"\d{9,12}")

# Words an assistant turn asking for a number almost always contains, in the
# languages and alphabets this deployment answers in.
#
# The callback words matter as much as the phone ones. Rule 7 asks for the
# number by offering the thing the patient wants -- "tell me a time that
# suits you and a colleague will call" -- and that sentence need never
# contain "raqam" at all. A list of only phone words would therefore miss
# exactly the phrasing the prompt recommends, leave the counter at zero, and
# let the assistant ask again in the very next message: the behaviour this
# module exists to stop.
#
# Still a heuristic, and known to be one. It is allowed to be rough because
# of which way it is wrong: over-counting holds back a request that can be
# made on any later turn, while under-counting produces the repetition.
_ASK_MARKERS = (
    # the number itself
    "raqam",
    "nomer",
    "номер",
    "телефон",
    "telefon",
    "phone",
    # ...and asking for it by promising the call
    "qongiroq",
    "bogla",
    "звон",
    "свяж",
    "call you",
    "call back",
)

# Apostrophes are written at least four ways in Uzbek Latin ("qo'ng'iroq",
# "qoʻngʻiroq", "qo‘ng‘iroq", "qo`ng`iroq"), and which one arrives depends on
# the patient's keyboard and on the model. Removing them entirely, on both
# sides of the comparison, is what makes one spelling of the marker enough.
_APOSTROPHES = re.compile(r"['‘’ʻʼ`´]")


def _normalise(text: str) -> str:
    return _APOSTROPHES.sub("", text.lower())


# What a greeting looks like, in every alphabet and spelling this deployment
# receives one in. Matched as substrings against text with its non-letters
# stripped, because the greeting a patient actually types is "Osalamayalaykum",
# "assalomu alaykum!!", "Ассалому алайкум" or "salam alikum" -- the word is
# recognisable, the spelling is never the same twice, and a list of exact forms
# would miss nearly all of them.
#
# "salam" rather than "salom" as the Latin stem, because the vowel is the thing
# patients vary; both are covered, along with the "alaykum" half on its own, so
# a message that mangles the first word still matches on the second.
_GREETING_MARKERS = (
    "salom",
    "salam",
    "alaykum",
    "aleykum",
    "alikum",
    "салом",
    "салам",
    "алайкум",
    "алейкум",
    "привет",
    "здравств",
    "zdravstv",
    "dobriy",
    "добр",
    "hello",
)

# Everything that is not a letter, so punctuation, emoji and digits cannot
# come between a marker and the text it should have matched.
_NON_LETTERS = re.compile(r"[^a-zЀ-ӿ]+")


def _letters(text: str) -> str:
    return _NON_LETTERS.sub("", _normalise(text))


def looks_like_a_greeting(text: str) -> bool:
    letters = _letters(text)
    return any(marker in letters for marker in _GREETING_MARKERS)


# A greeting is something a message opens with. Read further than that and an
# assistant turn quoting the clinic back to a patient -- or simply using the
# word later in a sentence -- counts as having said hello, and the next real
# greeting goes unanswered: the bug this module is being changed to fix.
_OPENING_LETTERS = 24


def opens_with_a_greeting(text: str) -> bool:
    return looks_like_a_greeting(_letters(text)[:_OPENING_LETTERS])


# Clinic details the assistant has already handed over. Repeating one is what
# turns a front desk into a leaflet: the transcript that prompted this answered
# "Osalamayalaykum", "Assalomu alaykum" and "Uzi boyicha yozvotudm" with the
# same telephone number and the same opening hours three times running, which
# reads as a machine with one sentence in it -- and buried the fact that the
# third message was a question about ultrasound that never got answered.
#
# Detected from our own earlier replies rather than asked of the model,
# for the same reason as everything else here: "have you already said this?"
# is a question about text, and code reads text more reliably than a model
# reads its own history.

# 09:00, 09.00, 9:00-18:00 -- an hour written with a separator. A bare "18"
# is not enough: the price rows and the doctor list are full of loose numbers.
_CLOCK = re.compile(r"\d{1,2}[:.]\d{2}")

_ADDRESS_MARKERS = (
    "kocha",
    "kochasi",
    "manzil",
    "tuman",
    "shahri",
    "кўча",
    "куча",
    "улиц",
    "адрес",
    "район",
)


@dataclass(frozen=True)
class ConversationSignals:
    """Facts about the conversation so far, from the assistant's side."""

    is_opening: bool
    patient_left_number: bool
    times_asked_for_number: int
    # Whether the message being answered right now is itself a greeting, and
    # whether we have ever returned one in this conversation.
    #
    # These exist because is_opening was doing this job and doing it wrongly.
    # Greeting was tied to "this is their first message", so a patient who
    # said hello again -- the next morning, or after the thread had gone quiet
    # for a week -- was told the opening hours with no hello in front of them.
    # An Instagram thread is one conversation forever; a person at a front
    # desk answers "Assalomu alaykum" with "Va alaykum assalom" every single
    # time it is said to them, and only starts sounding like a machine if they
    # say it twice in a row within one exchange.
    patient_greeted_now: bool
    already_greeted: bool
    # Whether we have already given each of these in this conversation.
    number_already_given: bool
    hours_already_given: bool
    address_already_given: bool


def looks_like_a_phone_number(text: str) -> bool:
    return find_phone_number(text) is not None


def find_phone_number(text: str) -> str | None:
    """The number itself, normalised, or None.

    Normalised because the same patient writes "+998 90 123 45 67" today and
    "998901234567" next week, and the clinic's spreadsheet is keyed on this
    string — two spellings would be two rows for one person.
    """
    match = _PHONE.search(_SEPARATORS.sub("", text))
    return match.group(0) if match else None


def read_signals(
    history: Sequence[ChatMessage] | None, user_message: str | None = None
) -> ConversationSignals:
    """Read the conversation's earlier turns, plus the message being answered.

    `history` excludes the message being answered right now (see
    app.services.conversation.context_for_reply), so an empty history is
    precisely "this is the first thing they have said to us".

    `user_message` is that excluded message, and it counts towards whether a
    number has been given. A patient who types their number unprompted has
    given it, and a facts block still saying "they have not given you a
    phone number yet" would be contradicting the message printed directly
    below it — which is exactly the disagreement that gets a patient asked
    for something they just typed.

    It deliberately does not count towards `is_opening` or towards the ask
    count: the first thing somebody says is still the opening, and the
    patient's own message is not us asking.
    """
    turns = list(history or [])
    patient_left_number = any(
        turn["role"] == "user" and looks_like_a_phone_number(turn["content"]) for turn in turns
    ) or bool(user_message and looks_like_a_phone_number(user_message))
    times_asked = sum(
        1
        for turn in turns
        if turn["role"] == "assistant"
        and any(m in _normalise(turn["content"]) for m in _ASK_MARKERS)
    )
    # Only the most recent assistant turn: greeting twice inside one exchange
    # is the mechanical thing, greeting again after the patient has come back
    # and said hello is not.
    ours = [turn["content"] for turn in turns if turn["role"] == "assistant"]
    last_assistant = ours[-1] if ours else ""
    return ConversationSignals(
        is_opening=not turns,
        patient_left_number=patient_left_number,
        times_asked_for_number=times_asked,
        patient_greeted_now=bool(user_message and looks_like_a_greeting(user_message)),
        already_greeted=opens_with_a_greeting(last_assistant),
        number_already_given=any(looks_like_a_phone_number(turn) for turn in ours),
        hours_already_given=any(_CLOCK.search(turn) for turn in ours),
        address_already_given=any(
            marker in _letters(turn) for turn in ours for marker in _ADDRESS_MARKERS
        ),
    )


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def render(signals: ConversationSignals) -> str:
    """The signals as a prompt section.

    Written as plain statements of fact rather than as instructions: the
    rules that act on them live in the prompt proper, so that reading the
    rules shows the whole policy in one place instead of half of it here.
    """
    lines = [
        (
            "This is the patient's first message to you."
            if signals.is_opening
            else "You are already in the middle of this conversation — "
            "they have written to you before."
        ),
        (
            "They have already given you a phone number."
            if signals.patient_left_number
            else "They have not given you a phone number yet."
        ),
    ]
    if signals.patient_greeted_now and not signals.already_greeted:
        lines.append(
            "They have greeted you in this message, and you have not greeted "
            "them back yet — return the greeting before anything else."
        )
    elif signals.patient_greeted_now:
        lines.append(
            "They have greeted you in this message, but your own last message "
            "already opened with a greeting — do not open with one again."
        )
    else:
        # Said out loud, because the rule on its own was not holding. "Man
        # kelasi seshanba 10:00ga qabulga yozilmoqchiman" — no greeting in it
        # anywhere — was answered "Ва алайкум ассалом", which answers a
        # message the patient never sent and leaves the one they did send
        # unanswered.
        lines.append(
            "There is no greeting in what they just wrote — do not open with "
            'one, and never with "Va alaykum assalom", which is the answer to a '
            "greeting nobody gave you. Begin with what they actually asked."
        )
    already = [
        label
        for label, given in (
            ("the clinic's telephone number", signals.number_already_given),
            ("the opening hours", signals.hours_already_given),
            ("the clinic's address", signals.address_already_given),
        )
        if given
    ]
    if already:
        lines.append(
            "You have already given them "
            + _join(already)
            + " in this conversation — do not print "
            + ("them" if len(already) > 1 else "it")
            + " again unless they ask."
        )
    if signals.times_asked_for_number == 0:
        lines.append("You have not asked them for their number in this conversation.")
    elif signals.times_asked_for_number == 1:
        lines.append("You have already asked them for their number once.")
    else:
        lines.append(
            f"You have already asked them for their number {signals.times_asked_for_number} times."
        )
    body = "\n".join(f"- {line}" for line in lines)
    return f"\n\nWHERE THIS CONVERSATION STANDS\n{body}"
