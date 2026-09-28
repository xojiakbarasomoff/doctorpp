"""The reason for a visit, in the words a urologist files it under.

Patients describe one condition a dozen ways -- "tez ejakulatsiya", "тез
бошаниш", "erta tugash", "tez bo'shanish" -- and the appointment book's
Izoh column was a dozen different words for the same thing, which nobody
can count or scan. So a complaint is written into the book under its
medical name.

Two rules keep this honest:

* The doctor's own grouping wins: everything about sexual function --
  early ejaculation as well as weak erection -- is "Jinsiy zaiflik", as he
  asked, although textbooks keep the two apart.
* Only what the patient actually named is renamed. A symptom maps to the
  symptom's name ("siyganda achishadi" -> "Dizuriya"), never to a diagnosis
  it might point to; and a phrase this list does not know ("siydikdagi
  noqulaylik") is kept exactly as written rather than guessed at.

For the clinic's sheet only. Nothing here is said to a patient.
"""

import re

# Canonical name -> patterns, matched against the text lower-cased with
# apostrophes removed ("bo'shanish" -> "boshanish"), in all three scripts
# patients write in.
_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Jinsiy zaiflik",
        (
            r"(erta|tez)\s*(bo?shan|boʻshan|tugash|tugay|tugab|ejakul|eyakul|chiq)",
            r"\b(ejakul|eyakul)",
            r"jinsiy\s*(zaif|quvvat)",
            r"\berek(ts|s|t)i",
            r"\bimpoten",
            r"\bpotent",
            r"\bturma(yapti|ydi|di|yotgan)",
            r"(тез|эрта)\s*(бўшан|бошан|бушан|туга|эякул)",
            r"жинсий\s*(заиф|кувват|қувват)",
            r"эякул",
            r"эрекц",
            r"импотен",
            r"потенц",
            r"преждевремен",
            r"турма(япти|йди|ди)",
        ),
    ),
    ("Prostata adenomasi", (r"\badenom", r"аденом", r"\bдгпж\b", r"гиперплази")),
    ("Prostatit", (r"\bprostatit", r"простатит")),
    ("Varikotsele", (r"\bvarik[ao]", r"варик[ао]")),
    ("Gidrotsele", (r"\bgidrots", r"гидроц")),
    ("Fimoz", (r"\bfimoz", r"фимоз")),
    ("Sistit", (r"\bsistit", r"цистит")),
    ("Uretrit", (r"\buretrit", r"уретрит")),
    (
        "Siydik-tosh kasalligi",
        (
            r"buyra[kg]\w*\s*tosh",
            r"tosh\w*\s*buyra[kg]",
            r"буйра[кг]\w*\s*тош",
            r"камн\w*\s*в\s*почк",
            r"почечн\w*\s*камн",
            r"мочекамен",
            r"urolitiaz",
            r"уролитиаз",
        ),
    ),
    ("Erkak bepushtligi", (r"\bbepusht", r"бепушт", r"бесплод", r"farzand\w*\s*bo?lma")),
    (
        "Dizuriya",
        (
            r"siy(ganda|ishda|gan\s*vaqt)\w*\s*(achish|ogri|og'ri|kuy)",
            r"дизури",
            r"(жжени\w*|бол\w*)\s*при\s*мочеисп",
            r"сий(ганда|ишда)\s*(ачиш|оғри|огри)",
        ),
    ),
    (
        "Gematuriya",
        (r"siydik\w*\s*qon", r"кровь\s*в\s*моче", r"сийдик\w*\s*(қон|кон)", r"гематури"),
    ),
    # "kechasi 3 marta siyaman": a few words may stand between.
    (
        "Nikturiya",
        (
            r"kechas\w*\b.{0,20}\bsiy",
            r"ноч\w*\b.{0,20}мочеисп",
            r"кечас\w*\b.{0,20}сий",
            r"никтури",
        ),
    ),
    (
        "Pollakiuriya (tez-tez siyish)",
        (r"tez[\s-]*tez\s*siy", r"част\w*\s*мочеисп", r"тез[\s-]*тез\s*сий", r"поллакиури"),
    ),
    ("Siydik tuta olmaslik", (r"siydik\w*\s*tut\w*\s*olm", r"недержан")),
    ("Chov sohasida og'riq", (r"\bchov\b", r"\bpax\w*\s*(sohasi|ogri)", r"\bпах", r"\bчов\b")),
)
_COMPILED = tuple(
    (name, tuple(re.compile(pattern) for pattern in patterns)) for name, patterns in _TERMS
)

# How a complaint note splits into separate complaints.
_PARTS = re.compile(r"\s*(?:,|;|/|\n|\s+va\s+|\s+hamda\s+|\s+и\s+|\s+ва\s+)\s*", re.IGNORECASE)
_APOSTROPHES = re.compile(r"['’ʻ‘`ʼ]")
# A cancellation's reason goes first in the Izoh, ahead of " · ".
_JOIN = " · "


def _names(fragment: str) -> list[str]:
    text = _APOSTROPHES.sub("", fragment.lower())
    return [name for name, patterns in _COMPILED if any(p.search(text) for p in patterns)]


def canonical(note: str) -> str:
    """The note with every complaint it names under its medical name.

    Each complaint (split on commas, "va", "и", ...) is renamed on its own;
    one this list does not know stays as the patient wrote it. A note in
    which nothing is recognised comes back exactly as it went in.
    """
    parts = [part for part in _PARTS.split(note or "") if part.strip()]
    renamed: list[str] = []
    changed = False
    for part in parts:
        names = _names(part)
        if names:
            changed = True
            renamed.extend(names)
        else:
            renamed.append(part.strip())
    if not changed:
        return note
    return ", ".join(dict.fromkeys(renamed))


def canonical_izoh(izoh: str) -> str:
    """An Izoh cell as the book stores it: a "Bekor: ..." reason, if any,
    kept as it is, and the complaint after it renamed."""
    pieces = (izoh or "").split(_JOIN)
    return _JOIN.join(piece if piece.startswith("Bekor:") else canonical(piece) for piece in pieces)
