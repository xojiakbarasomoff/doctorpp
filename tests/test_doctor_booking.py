"""Dr. Temur's week, the booking marker, and the two prompt modes.

The rules here are the ones a patient notices when they break: being offered
a Sunday, being booked into 17:00 when the doctor leaves at five, being told
they are booked by an assistant that was meant to send them to the telephone,
or being read the doctor's CV in answer to "salom".
"""

from datetime import date, datetime

import pytest

from app.services.answer import _build_system_prompt
from app.services.appointment import CLINIC_TIMEZONE, day_slots, is_within_working_hours
from app.services.booking import extract, marker_details, render
from app.services.conversation_signals import read_signals
from app.services.patient_media import ACKNOWLEDGEMENTS, caption

FRIDAY = date(2026, 9, 18)
SUNDAY = date(2026, 9, 20)


def _at(day: date, hour: int, minute: int) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=CLINIC_TIMEZONE)


# --- the week ---------------------------------------------------------------------


def test_a_working_day_is_twenty_four_twenty_minute_slots_from_nine_to_four_forty() -> None:
    slots = list(day_slots(FRIDAY))

    assert len(slots) == 24
    assert slots[0] == _at(FRIDAY, 9, 0)
    assert slots[1] == _at(FRIDAY, 9, 20)
    assert slots[-1] == _at(FRIDAY, 16, 40)


def test_sunday_has_no_slots_at_all() -> None:
    assert list(day_slots(SUNDAY)) == []
    assert not is_within_working_hours(_at(SUNDAY, 10, 0))


@pytest.mark.parametrize(
    ("hour", "minute", "bookable"),
    [(9, 0, True), (16, 40, True), (17, 0, False), (8, 40, False), (9, 30, False), (10, 20, True)],
)
def test_only_times_on_the_grid_inside_the_day_are_bookable(
    hour: int, minute: int, bookable: bool
) -> None:
    assert is_within_working_hours(_at(FRIDAY, hour, minute)) is bookable


# --- the marker -------------------------------------------------------------------


def test_the_marker_carries_name_phone_and_reason() -> None:
    reply = (
        "Yaxshi, ertaga 09:20 da doktor sizni kutadi. "
        "[[BOOK:2026-09-18T09:20|Ali Valiyev|+998 90 123 45 67|siydik yo'lida og'riq]]"
    )

    text, slot, name = extract(reply)
    phone, reason = marker_details(reply)

    assert "[[" not in text
    assert slot == _at(FRIDAY, 9, 20)
    assert name == "Ali Valiyev"
    assert phone == "+998 90 123 45 67"
    assert reason == "siydik yo'lida og'riq"


def test_a_marker_with_only_a_time_still_parses() -> None:
    assert marker_details("[[BOOK:2026-09-18T09:20]]") == (None, None)
    assert extract("[[BOOK:2026-09-18T09:20]]")[1] == _at(FRIDAY, 9, 20)


def test_placeholder_words_are_not_stored_as_a_phone_or_a_reason() -> None:
    assert marker_details("[[BOOK:2026-09-18T09:20|Ali|telephone|reason]]") == (None, None)


def test_the_book_tells_the_model_how_long_an_appointment_is() -> None:
    assert "Each lasts 20 minutes" in render([_at(FRIDAY, 9, 0)], _at(FRIDAY, 8, 0))


# --- the prompt -------------------------------------------------------------------


def _prompt(**overrides: object) -> str:
    values: dict[str, object] = {
        "flagged_as_medical_advice": False,
        "default_language": "Uzbek",
        "clinic_phone_numbers": "+998 70 310 40 40",
        "clinic_address": None,
        "signals": read_signals([], "Salom"),
        "doctor_name": "Axmadaliyev Temur",
        "doctor_specialty": "urolog-androlog",
    }
    values.update(overrides)
    return _build_system_prompt((), **values)  # type: ignore[arg-type]


def test_without_a_book_the_assistant_sends_patients_to_the_telephone() -> None:
    prompt = _prompt()

    assert "Never book, never hold" in prompt
    assert "[[BOOK:YYYY" not in prompt
    assert "THE APPOINTMENT BOOK" not in prompt


def test_with_a_book_it_collects_name_phone_and_complaint_then_books() -> None:
    prompt = _prompt(appointment_book=render([_at(FRIDAY, 9, 0)], _at(FRIDAY, 8, 0)))

    assert "Never book, never hold" not in prompt
    assert "Do not ask the patient for their telephone number" not in prompt
    assert "[[BOOK:YYYY-MM-DDTHH:MM|full name|telephone|reason]]" in prompt
    name, phone, reason = (
        prompt.index("full name (first name"),
        prompt.index("then their telephone number"),
        prompt.index("then the reason they are coming"),
    )
    assert name < phone < reason
    # The book is the last thing the model reads.
    assert prompt.rstrip().endswith("from the list.") or "THE APPOINTMENT BOOK" in prompt[-3000:]


def test_the_assistant_introduces_itself_as_the_doctors_administrator() -> None:
    assert "shifokorning administratori" in _prompt()


def test_the_doctors_background_is_given_but_only_for_when_asked() -> None:
    prompt = _prompt(doctor_background="- Ish tajribasi: 5 yil")

    assert "- Ish tajribasi: 5 yil" in prompt
    assert "ONLY when the patient asks" in prompt
    assert "Ish tajribasi" not in _prompt()


def test_medical_advice_stays_forbidden_in_booking_mode() -> None:
    prompt = _prompt(appointment_book=render([], _at(FRIDAY, 8, 0)))

    assert "NEVER diagnose" in prompt
    assert "nothing about treating it (rule 3)" in prompt


# --- photos -----------------------------------------------------------------------


def test_the_photo_acknowledgement_exists_in_every_script() -> None:
    assert ACKNOWLEDGEMENTS["uz-latn"] == "Hozir ko'rib beramiz."
    assert set(ACKNOWLEDGEMENTS) == {"uz-latn", "uz-cyrl", "ru"}


def test_the_doctor_is_told_who_sent_the_photo() -> None:
    class Patient:
        name = "Aziz"
        username = "aziz_uz"

    text = caption(Patient(), 2)  # type: ignore[arg-type]

    assert "Bemor tomonidan 2 ta rasm yuborildi" in text
    assert "Aziz (@aziz_uz)" in text
