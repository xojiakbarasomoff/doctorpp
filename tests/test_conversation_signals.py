"""Recognising a telephone number or a greeting in what a patient wrote."""

import pytest

from app.services.conversation_signals import looks_like_a_greeting, looks_like_a_phone_number


@pytest.mark.parametrize(
    "text",
    [
        "+998 90 123 45 67",
        "998901234567",
        "901234567",
        "90 123 45 67",
        "raqamim: +998-90-123-45-67",
        "(90) 123-45-67 shu raqamga qo'ng'iroq qiling",
    ],
)
def test_a_number_written_the_way_people_write_it_is_recognised(text: str) -> None:
    assert looks_like_a_phone_number(text)


@pytest.mark.parametrize(
    "text",
    [
        "implant qancha turadi?",
        "3 500 000 so'm",
        "narxi 3500000",
        "09:00 dan 20:00 gacha",
        "2026-yil 24-avgust",
        "5 ta tishim og'riyapti",
        "",
    ],
)
def test_ordinary_numbers_in_a_message_are_not_a_phone_number(text: str) -> None:
    """A price or an opening time reading as a phone number would silently
    stop the assistant ever asking for one — the failure would look like the
    assistant simply not doing its job.
    """
    assert not looks_like_a_phone_number(text)


@pytest.mark.parametrize(
    "message",
    [
        "Assalomu alaykum",
        "assalomu alaykum!!",
        # What patients actually type. The first word is misspelt past
        # recognition and the greeting still has to be answered.
        "Osalamayalaykum",
        "salam alikum",
        "Ассалому алайкум",
        "Салом",
        "Здравствуйте",
        "Привет!",
    ],
)
def test_a_greeting_is_recognised_however_it_is_spelt(message: str) -> None:
    assert looks_like_a_greeting(message)


@pytest.mark.parametrize(
    "message",
    [
        "EKG qancha turadi",
        "buyragim og'riyapti",
        "+998901234567",
        "Qabulga yozilmoqchiman",
        "Rahmat",
    ],
)
def test_an_ordinary_message_is_not_a_greeting(message: str) -> None:
    assert not looks_like_a_greeting(message)
