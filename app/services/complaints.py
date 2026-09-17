"""What patients write in about, sorted into the doctor's own categories.

Keyword rules rather than a model. A report is read to make decisions -- which
conditions to post about, what to prepare for -- and a count that changes when
the same data is recounted, or costs an API call per patient per page load,
is not a count anyone should act on. The rules are visible, cheap, and give
the same answer every time; a message nothing matches is counted honestly as
"Aniqlanmagan" rather than guessed at.

Written for how patients in Tashkent actually type: Uzbek in Latin (with and
without the apostrophes), Uzbek in Cyrillic, and Russian, including the
common misspellings. Matching is on word stems, so "og'riyapti",
"ogriyapti" and "og'riq" all land in the same place.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    key: str
    label: str
    pattern: re.Pattern[str]


def _stems(*stems: str) -> re.Pattern[str]:
    return re.compile("|".join(stems), re.IGNORECASE)


# Ordered from the most specific to the most general: a patient who writes
# "buyragimda tosh bor, siyganda achishadi" is counted under both stones and
# urinary infection -- a category is a topic mentioned, not a diagnosis.
CATEGORIES: tuple[Category, ...] = (
    Category(
        "stones",
        "Buyrak toshi, buyrak og'rig'i",
        _stems(
            r"tosh(?!kent)", r"buyra[kg]", r"буйрак", r"тош\b", r"почк", r"камн", r"камен", r"колик"
        ),
    ),
    Category(
        "prostate",
        "Prostata",
        _stems(r"prostat", r"простат", r"\bpsa\b", r"\bпса\b", r"prostotit"),
    ),
    Category(
        "erectile",
        "Erektil disfunksiya, potensiya",
        _stems(
            r"erek",
            r"potens",
            r"potents",
            r"jinsiy\s+(zaif|kuch|quvvat)",
            r"эрекц",
            r"потенц",
            r"эректил",
            r"tez\s+tugat",
            r"eyakul",
            r"эякул",
            r"семяизверж",
        ),
    ),
    Category(
        "infertility",
        "Bepushtlik, spermogramma",
        _stems(
            r"bepusht",
            r"farzand",
            r"spermogram",
            r"spermo",
            r"бесплод",
            r"сперм",
            r"бепушт",
            r"фарзанд",
            r"homilador\s+bo'?lma",
        ),
    ),
    Category(
        "urinary_infection",
        "Siydik yo'li yallig'lanishi, achishish",
        _stems(
            r"achish",
            r"ачиш",
            r"sistit",
            r"цистит",
            r"uretrit",
            r"уретрит",
            r"жжен",
            r"рези\b",
            r"siyganda\s+og'?ri",
            r"siydik\s+yo'?l",
            r"сийдик",
            r"yallig'?lan",
        ),
    ),
    Category(
        "frequent_urination",
        "Tez-tez siyish, siydik tutolmaslik",
        _stems(
            r"tez-?tez\s+siy",
            r"tutolma",
            r"tuta\s+olma",
            r"kechasi\s+siy",
            r"недерж",
            r"частое\s+моче",
            r"часто\s+в\s+туалет",
            r"hojatxonaga\s+tez",
        ),
    ),
    Category(
        "blood",
        "Siydikda qon",
        _stems(r"siydikda\s+qon", r"qon\s+keld", r"гематур", r"кровь\s+в\s+моч", r"сийдикда\s+қон"),
    ),
    Category(
        "scrotum",
        "Moyak, varikosel, churra",
        _stems(
            r"moyak",
            r"varik",
            r"варик",
            r"яичк",
            r"варикоцел",
            r"мошонк",
            r"gidrosel",
            r"гидроцел",
        ),
    ),
    Category(
        "sti",
        "Jinsiy yo'l infeksiyalari",
        _stems(
            r"xlamid",
            r"хламид",
            r"gonor",
            r"гонор",
            r"sifilis",
            r"сифил",
            r"ureaplazm",
            r"уреаплазм",
            r"mikoplazm",
            r"микоплазм",
            r"иппп",
            r"zppp",
            r"trixomon",
            r"трихомон",
        ),
    ),
    Category(
        "pain",
        "Og'riq (joyi aniqlanmagan)",
        _stems(r"og'?ri", r"огри", r"оғри", r"боль", r"болит"),
    ),
)

UNKNOWN = Category("unknown", "Aniqlanmagan", re.compile(r"(?!)"))

# Messages that are about the practice rather than a condition. Counted
# separately: "qancha turadi?" is a real demand signal, but it is not a
# medical complaint and must not dilute the chart of what patients suffer from.
ENQUIRY = re.compile(
    r"narx|qancha|necha\s+pul|qabul|yozil|manzil|qayerda|telefon|цена|стоим|сколько|"
    r"запис|адрес|где\s+наход|нарх|қабул|ёзил",
    re.IGNORECASE,
)


# Lines the system itself appends to an appointment's notes (see
# app.services.doctor_telegram) -- not the patient's words.
_SYSTEM_NOTES = re.compile(r"^Doktor bu (kuni|vaqtda) qabul qila olmadi\.?$", re.MULTILINE)


def patient_words(text: str | None) -> str:
    """An appointment's notes with the system's own lines taken out."""
    return _SYSTEM_NOTES.sub("", text or "").strip()


def categorise(texts: Iterable[str]) -> list[Category]:
    """Every category the patient's words mention, in CATEGORIES order."""
    joined = " \n ".join(t for t in texts if t)
    found = [category for category in CATEGORIES if category.pattern.search(joined)]
    # "Og'riq" on its own is a symptom without a place; once a place is named
    # it is that category's pain, not a separate one.
    if len(found) > 1:
        found = [category for category in found if category.key != "pain"]
    return found


def is_enquiry(texts: Iterable[str]) -> bool:
    return any(ENQUIRY.search(t) for t in texts if t)
