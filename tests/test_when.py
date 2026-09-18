"""What the patient wrote about time, as a date the clinic can hold.

Every case runs against a frozen clock: Friday 18 September 2026, which is
the day the transcript that prompted this was recorded.
"""

from datetime import date, datetime, time

import pytest

from app.services.appointment import CLINIC_TIMEZONE
from app.services.when import read, spoken

# Friday, 18 September 2026, mid-afternoon in Tashkent.
NOW = datetime(2026, 9, 18, 15, 30, tzinfo=CLINIC_TIMEZONE)
TODAY = NOW.date()


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("bugun bo'sh vaqt bormi?", date(2026, 9, 18)),
        ("ertaga yozilmoqchiman", date(2026, 9, 19)),
        ("indinga bo'ladimi", date(2026, 9, 20)),
        # Friday: the coming Thursday is the 24th.
        ("payshanba kuni", date(2026, 9, 24)),
        # ...and "next week Thursday" is the same day here, because the
        # coming Thursday already falls in next week.
        ("keyingi hafta payshanba", date(2026, 9, 24)),
        ("kelasi seshanba", date(2026, 9, 22)),
        ("24-sentabr", date(2026, 9, 24)),
        ("24 sentabrga yozing", date(2026, 9, 24)),
        ("01.10 ga", date(2026, 10, 1)),
        ("завтра утром", date(2026, 9, 19)),
        ("в четверг", date(2026, 9, 24)),
    ],
)
def test_the_day_a_patient_named(written: str, expected: date) -> None:
    assert read(written, now=NOW).day == expected


def test_a_date_already_past_this_year_means_next_year() -> None:
    assert read("3-mart", now=NOW).day == date(2027, 3, 3)


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("12:00", time(12, 0)),
        ("11.40 da", time(11, 40)),
        ("soat 12 da kelaman", time(12, 0)),
        # "2 larga" at a clinic open 09:00-17:00 is the afternoon.
        ("kunduzi 2 larga bosh joy bomi", time(14, 0)),
    ],
)
def test_the_time_a_patient_named(written: str, expected: time) -> None:
    assert read(written, now=NOW).at == expected


def test_a_message_about_nothing_in_particular_resolves_to_nothing() -> None:
    assert read("buyragim og'riyapti", now=NOW).empty
    assert read("rahmat", now=NOW).empty


def test_the_day_and_the_time_together() -> None:
    when = read("keyingi hafta payshanba 12:00 ga yozing", now=NOW)

    assert when.day == date(2026, 9, 24)
    assert when.at == time(12, 0)


def test_the_day_is_read_back_the_way_a_person_says_it() -> None:
    assert spoken(TODAY, TODAY) == "bugun"
    assert spoken(date(2026, 9, 19), TODAY) == "ertaga"
    assert spoken(date(2026, 9, 24), TODAY) == "payshanba, 24-sentabr"
