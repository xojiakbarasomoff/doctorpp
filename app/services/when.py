"""Turning what a patient wrote about time into a date the clinic can hold.

"Keyingi hafta payshanba" is a sentence. 2026-09-24 is a date. Until now the
sentence was all there was: it went into the prompt, the model worked out
what it meant, and worked it out again on the next turn from a shorter
transcript -- which is how a patient booked for Thursday the 24th was
offered "ertaga 09:00" two messages later.

Resolved once, here, against the clinic's own timezone, and stored as a real
date (app.models.conversation_state). Everything after that reads the date.

Deliberately not a general natural-language date parser. It covers what
patients of this clinic actually type, in the three ways they type it, and
returns None for anything it is not sure about -- an unresolved date is a
question the assistant can ask, while a wrong one is a patient arriving on
the wrong day.
"""

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from app.services.appointment import CLINIC_TIMEZONE

# Monday is 0, the way date.weekday() counts.
_WEEKDAYS: dict[str, int] = {
    "dushanba": 0, "душанба": 0, "понедельник": 0,
    "seshanba": 1, "сешанба": 1, "вторник": 1,
    "chorshanba": 2, "чоршанба": 2, "среда": 2, "среду": 2,
    "payshanba": 3, "пайшанба": 3, "четверг": 3,
    "juma": 4, "жума": 4, "пятница": 4, "пятницу": 4,
    "shanba": 5, "шанба": 5, "суббота": 5, "субботу": 5,
    "yakshanba": 6, "якшанба": 6, "воскресенье": 6,
}  # fmt: skip

# "Shanba" is inside "yakshanba" and "seshanba"; the longest name wins.
_WEEKDAY_RE = re.compile(
    r"\b(" + "|".join(sorted(_WEEKDAYS, key=len, reverse=True)) + r")\w*", re.IGNORECASE
)

_MONTHS: dict[str, int] = {
    "yanvar": 1, "январ": 1, "fevral": 2, "феврал": 2, "mart": 3, "март": 3,
    "aprel": 4, "апрел": 4, "may": 5, "мая": 5, "май": 5, "iyun": 6, "июн": 6,
    "iyul": 7, "июл": 7, "avgust": 8, "август": 8, "sentabr": 9, "sentyabr": 9,
    "сентябр": 9, "oktabr": 10, "октябр": 10, "noyabr": 11, "ноябр": 11,
    "dekabr": 12, "декабр": 12,
}  # fmt: skip

_DAY_MONTH = re.compile(
    r"\b(?P<day>[0-3]?\d)\s*[-\s]?\s*(?P<month>" + "|".join(_MONTHS) + r")\w*", re.IGNORECASE
)
# 24.09, 01.10, 24/09/2026 -- but never "PSA 6.2", which is a test result a
# patient is frightened about, not a date in February.
_NUMERIC_DATE = re.compile(
    r"\b(?P<day>[0-3]\d)[./](?P<month>[01]\d)(?:[./](?P<year>\d{2,4}))?\b"
    r"|\b(?P<day2>[0-3]?\d)[./](?P<month2>[01]?\d)[./](?P<year2>\d{2,4})\b"
)

_TODAY = re.compile(r"\bbugun\w*|\bбугун\w*|\bсегодня\b", re.IGNORECASE)
_TOMORROW = re.compile(r"\bertaga\w*|\bэртага\w*|\bзавтра\b", re.IGNORECASE)
_DAY_AFTER = re.compile(r"\bindin\w*|\bиндин\w*|\bпослезавтра\b", re.IGNORECASE)

# "Keyingi payshanba", "kelasi hafta payshanba", "следующий четверг". The
# week after the coming one, which is not the same thing as the next
# Thursday when today is Tuesday.
_NEXT_WEEK = re.compile(
    r"\b(?:keyingi|kelasi|kelgusi)\s+(?:hafta\w*\s+)?|\bкейинги\s+|\bследующ\w+\s+",
    re.IGNORECASE,
)
_EXPLICIT_WEEK = re.compile(r"\b(?:keyingi|kelasi|kelgusi)\s+hafta|\bследующ\w+\s+недел", re.IGNORECASE)

_TIME = re.compile(r"\b(?P<hour>[0-2]?\d)[:.](?P<minute>[0-5]\d)\b")
# "soat 12 da", "12 larga" -- an hour with no minutes, which patients write
# far more often than they write 12:00.
_BARE_HOUR = re.compile(
    r"\b(?:soat\s*)?(?P<hour>[01]?\d|2[0-3])\s*(?:larga|lar|ga|da|ta|\-?ga)\b", re.IGNORECASE
)


@dataclass(frozen=True)
class When:
    """What one message said about when, canonically."""

    day: date | None = None
    at: time | None = None

    @property
    def empty(self) -> bool:
        return self.day is None and self.at is None


def _weekday_after(today: date, weekday: int, *, next_week: bool) -> date:
    """The next such weekday, or the one in the week after this one.

    "Payshanba" on a Tuesday means this Thursday. "Keyingi hafta payshanba"
    means the Thursday after that, and getting this wrong books somebody a
    week away from where they meant to be.
    """
    ahead = (weekday - today.weekday()) % 7
    if ahead == 0:
        ahead = 7
    day = today + timedelta(days=ahead)
    if next_week:
        # Explicitly "next week": from the start of the coming week.
        start_of_next_week = today + timedelta(days=7 - today.weekday())
        day = start_of_next_week + timedelta(days=weekday)
        if day <= today:  # pragma: no cover - defensive
            day += timedelta(days=7)
    return day


def read(text: str, *, now: datetime | None = None) -> When:
    """The day and the time in this message, in the clinic's own timezone."""
    local_now = (now or datetime.now(CLINIC_TIMEZONE)).astimezone(CLINIC_TIMEZONE)
    today = local_now.date()

    day: date | None = None
    if _TODAY.search(text):
        day = today
    elif _TOMORROW.search(text):
        day = today + timedelta(days=1)
    elif _DAY_AFTER.search(text):
        day = today + timedelta(days=2)

    if day is None:
        named = _DAY_MONTH.search(text)
        if named is not None:
            month = next(
                number
                for stem, number in _MONTHS.items()
                if named.group("month").lower().startswith(stem[:4])
            )
            day = _with_year(int(named.group("day")), month, today)

    if day is None:
        numeric = _NUMERIC_DATE.search(text)
        if numeric is not None:
            year = numeric.group("year") or numeric.group("year2")
            try:
                day = date(
                    int(year) if year and len(year) == 4 else today.year,
                    int(numeric.group("month") or numeric.group("month2")),
                    int(numeric.group("day") or numeric.group("day2")),
                )
            except ValueError:
                day = None
            else:
                if day < today:
                    day = day.replace(year=day.year + 1)

    if day is None:
        weekday = _WEEKDAY_RE.search(text)
        if weekday is not None:
            stem = weekday.group(1).lower()
            before = text[: weekday.start()]
            next_week = bool(_EXPLICIT_WEEK.search(before)) or bool(
                _NEXT_WEEK.search(before[-24:])
            )
            day = _weekday_after(today, _WEEKDAYS[stem], next_week=next_week)

    at: time | None = None
    clock = _TIME.search(text)
    if clock is not None:
        hour, minute = int(clock.group("hour")), int(clock.group("minute"))
        if hour < 24:
            at = time(hour, minute)
    else:
        bare = _BARE_HOUR.search(text)
        if bare is not None:
            hour = int(bare.group("hour"))
            # "2 larga" in the afternoon means 14:00: a clinic that opens at
            # nine and closes at five has no two o'clock in the morning.
            if 1 <= hour <= 8:
                hour += 12
            at = time(hour, 0)

    return When(day=day, at=at)


def _with_year(day_number: int, month: int, today: date) -> date | None:
    """"24-sentabr" without a year means the next one there is."""
    for year in (today.year, today.year + 1):
        try:
            candidate = date(year, month, day_number)
        except ValueError:
            return None
        if candidate >= today:
            return candidate
    return None  # pragma: no cover - the loop above always returns


def spoken(day: date, today: date) -> str:
    """The day as a person says it, for a sentence the patient reads."""
    if day == today:
        return "bugun"
    if day == today + timedelta(days=1):
        return "ertaga"
    if day == today + timedelta(days=2):
        return "indinga"
    names = ["dushanba", "seshanba", "chorshanba", "payshanba", "juma", "shanba", "yakshanba"]
    months = [
        "yanvar", "fevral", "mart", "aprel", "may", "iyun",
        "iyul", "avgust", "sentabr", "oktabr", "noyabr", "dekabr",
    ]  # fmt: skip
    return f"{names[day.weekday()]}, {day.day}-{months[day.month - 1]}"
