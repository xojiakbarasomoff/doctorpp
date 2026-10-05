"""Which sums of money a reply may say: only the ones the clinic's rules give.

The persona forbids prices, and the clinic's rules may name a few it does
want said ("qabul 300 000 so'm", "amaliyot taxminan 30 mln"). Told that in
the prompt, the model still read leave to quote one price as leave to quote
the knowledge base's: asked "UZI narxi qancha?", it answered with the
uploaded price list two times in eight. So it is checked here, by pattern,
the way app.services.medical_safety checks a prescription.

What counts as a sum: a number with a unit of money or a multiplier after it
("300 ming", "30 mln", "250 000 so'm", "$500"), or a number grouped in
thousands that ends in zeros ("200 000"). That is how prices are written
here, and it leaves alone what else a reply carries numbers for: telephone
numbers ("+998 71 200 03 93"), times, dates, ages.

A sum is allowed when the same amount -- not the same spelling -- appears in
one of the clinic's rules: "300 ming" in a reply matches "300 000 so'm" in a
rule.
"""

import re
from collections.abc import Iterable, Sequence
from decimal import Decimal, InvalidOperation

_MULTIPLIERS = {
    "ming": 1_000,
    "минг": 1_000,
    "тыс": 1_000,
    "тысяч": 1_000,
    "k": 1_000,
    "mln": 1_000_000,
    "million": 1_000_000,
    "миллион": 1_000_000,
    "млн": 1_000_000,
    "mlrd": 1_000_000_000,
    "milliard": 1_000_000_000,
    "млрд": 1_000_000_000,
    "миллиард": 1_000_000_000,
}
_CURRENCY = r"so['‘ʻ’`]?m|сўм|сум|sum|uzs|\$|usd|dollar|доллар|руб|rubl|€|eur"
_MULTIPLIER = "|".join(sorted(map(re.escape, _MULTIPLIERS), key=len, reverse=True))
_NUMBER = r"\d+(?:[ \u00a0\u202f.,]\d{3})*(?:[.,]\d{1,2})?"

# A number followed by a multiplier and/or a currency, or a currency sign in
# front of one. The lookarounds keep "k" and "sum" from matching inside words.
_WITH_UNIT = re.compile(
    rf"(?<![\w+])(?P<cur>\$|€)?\s?(?P<num>{_NUMBER})\s*"
    rf"(?:(?P<mul>{_MULTIPLIER})\.?(?![\w]))?\s*(?P<unit>(?:{_CURRENCY})(?![\w]))?",
    re.IGNORECASE,
)
# "200 000" with nothing after it: thousands groups ending in 000.
_BARE = re.compile(r"(?<![\w+\d])\d{1,3}(?:[ \u00a0\u202f.,]\d{3})*[ \u00a0\u202f.,]000(?![\d])")


def _value(number: str, multiplier: str | None) -> Decimal | None:
    digits = re.sub(r"[ \u00a0\u202f]", "", number)
    # "1,5 mln" / "1.5 mln": a decimal separator before one or two digits;
    # "250.000" / "250,000": a thousands separator before three.
    parts = re.split(r"[.,]", digits)
    if len(parts) == 2 and len(parts[1]) in (1, 2):
        text = f"{parts[0]}.{parts[1]}"
    else:
        text = "".join(parts)
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    if multiplier:
        value *= _MULTIPLIERS[multiplier.lower()]
    return value


def amounts(text: str) -> list[tuple[str, Decimal]]:
    """Every sum of money in `text`, as written and as a number."""
    found: list[tuple[str, Decimal]] = []
    taken: list[tuple[int, int]] = []
    for match in _WITH_UNIT.finditer(text):
        if not (match.group("cur") or match.group("mul") or match.group("unit")):
            continue
        value = _value(match.group("num"), match.group("mul"))
        if value is not None:
            found.append((match.group(0).strip(), value))
            taken.append(match.span())
    for match in _BARE.finditer(text):
        if any(start <= match.start() < end for start, end in taken):
            continue
        value = _value(match.group(0), None)
        if value is not None:
            found.append((match.group(0), value))
    return found


def allowed(rules: Iterable[str]) -> set[Decimal]:
    """The sums the clinic's rules name."""
    return {value for rule in rules for _, value in amounts(rule)}


def check(reply: str, rules: Sequence[str]) -> list[str]:
    """The sums in `reply` that no rule gives, as written. Empty when fine."""
    permitted = allowed(rules)
    return [written for written, value in amounts(reply) if value not in permitted]


SAFE_REPLIES = {
    "uz-latn": (
        "Narx bo'yicha aniq ma'lumotni klinikamizga qo'ng'iroq qilib bilib olishingiz mumkin."
    ),
    "uz-cyrl": "Нарх бўйича аниқ маълумотни клиникамизга қўнғироқ қилиб билиб олишингиз мумкин.",
    "ru": "Точную стоимость можно узнать, позвонив в нашу клинику.",
}
