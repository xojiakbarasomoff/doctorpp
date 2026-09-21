"""Recognising a telephone number or a greeting in what a patient wrote."""

import re

# Nine digits is the shortest real Uzbek subscriber number (90 123 45 67),
# and twelve covers +998 with the country code. The separators people type
# between the groups are stripped first.
#
# Deliberately not matching shorter runs: a clinic that quotes "3 500 000
# so'm" or answers "09:00 - 20:00" would otherwise look like a patient who
# has already left a number, and the assistant would stop asking.
_SEPARATORS = re.compile(r"[\s\-()+.]")
_PHONE = re.compile(r"\d{9,12}")

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
