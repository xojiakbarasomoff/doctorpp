"""Dr. Temur's week, the booking marker, and the two prompt modes.

The rules here are the ones a patient notices when they break: being offered
a Sunday, being booked into 17:00 when the doctor leaves at five, being told
they are booked by an assistant that was meant to send them to the telephone,
or being read the doctor's CV in answer to "salom".
"""

from datetime import date, datetime, timedelta

import pytest

from app.rag.llm import ChatMessage
from app.services.answer import _build_system_prompt, conversation_script
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


# --- one conversation, one alphabet -------------------------------------


def test_the_alphabet_is_the_conversations_own_not_the_last_message() -> None:
    """A telephone number and a time have no alphabet. Deciding per message
    answered "93 444 444" in Cyrillic in the middle of a Latin conversation,
    and the message after it went back to Latin.
    """
    history: list[ChatMessage] = [
        {"role": "user", "content": "Man kelasi seshanba 10:00ga qabulga yozilmoqchiman"},
        {"role": "assistant", "content": "Исмингизни ёзинг"},
        {"role": "user", "content": "Asadbek Risqiyev"},
    ]

    assert conversation_script(history, "93 444 444") == "uz-latn"
    assert conversation_script(history, "11:00") == "uz-latn"
    # What the assistant wrote does not count, or one reply in the wrong
    # alphabet would justify the next one.
    assert conversation_script(history, "Buyrak ogrigi") == "uz-latn"


def test_a_patient_who_writes_cyrillic_is_answered_in_cyrillic() -> None:
    history: list[ChatMessage] = [{"role": "user", "content": "буйрагим оғрияпти"}]

    assert conversation_script(history, "16:20") == "uz-cyrl"
    assert conversation_script(None, "Здравствуйте") == "ru"


def test_both_cyrillic_scripts_are_told_to_answer_in_cyrillic() -> None:
    """reply_script() reads Cyrillic without Uzbek's own letters -- "Ассалом
    алайкум" -- as Russian. The section only claims what it knows: the
    alphabet. Which language is written in it is decided elsewhere.
    """
    from app.services.answer import _SCRIPT_NAMES

    assert "CYRILLIC" in _SCRIPT_NAMES["uz-cyrl"]
    assert "CYRILLIC" in _SCRIPT_NAMES["ru"]
    assert "LATIN" in _SCRIPT_NAMES["uz-latn"]


# --- how far ahead the book goes ----------------------------------------


def test_a_named_weekday_next_week_is_inside_the_book() -> None:
    """"Man kelasi seshanba 10:00ga yozilmoqchiman" was answered "seshanba
    11:00 bu jadvalda yo'q" on a Friday, with next Tuesday entirely free:
    the book only looked three days ahead. People book around their own
    week, and a named weekday is further off than "ertaga" every time.
    """
    from app.services.booking import HORIZON_DAYS

    assert HORIZON_DAYS >= 8


def test_a_time_that_starts_in_two_minutes_is_not_offered() -> None:
    """At 15:58 the book still held 16:00 and it was offered. Nobody can
    keep that time; they either miss it or arrive to a doctor who is with
    somebody else.
    """
    from app.services.booking import BOOKING_LEAD

    assert BOOKING_LEAD >= timedelta(minutes=20)
