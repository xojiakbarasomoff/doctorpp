"""A price question with nothing to price.

The detector decides whether the model is handed any clinic rows at all, so
both ways of being wrong are visible to a patient: fire when it should not
and a real question ("buyrak UZI narxi") loses its answer; fail to fire and
the patient is asked whether they meant copper or magnesium.
"""

import pytest

from app.services.question_shape import asks_a_price, names_nothing_to_price


@pytest.mark.parametrize(
    "message",
    [
        "narxi qancha",
        "Narxi qancha?",
        "qancha turadi",
        "necha pul",
        "qancha pul",
        "narxlari qancha",
        "Menga narxini ayting",
        "qancha turadi iltimos ayting",
        # "which service" is the whole question, so the service is not named.
        "xizmat qancha turadi",
        # A greeting in front of it changes nothing.
        "salom narxi qancha",
        "сколько стоит",
        "цена какая",
        "какая цена",
        "нархи қанча",
    ],
)
def test_a_price_question_with_no_service_in_it(message: str) -> None:
    assert names_nothing_to_price(message)


@pytest.mark.parametrize(
    "message",
    [
        "buyrak UZI narxi",
        "UZI qancha turadi",
        "EKG narxi qancha",
        "ginekolog qabuli qancha",
        "spermogramma necha pul",
        "analiz qancha turadi",
        "konsultatsiya qancha",
        "сколько стоит УЗИ почек",
        "цена УЗИ",
    ],
)
def test_a_named_service_is_left_alone(message: str) -> None:
    """These retrieve properly, so nothing should be taken away from them."""
    assert asks_a_price(message)
    assert not names_nothing_to_price(message)


@pytest.mark.parametrize(
    "message",
    [
        "salom",
        "qayerdasiz",
        "qabulga yozilmoqchiman",
        "ish vaqtingiz qanday",
        "buyragim og'riyapti",
        "shifokorlar haqida ayting",
        "EKG bormi",
    ],
)
def test_a_message_that_is_not_about_price_is_not_touched(message: str) -> None:
    assert not names_nothing_to_price(message)


def test_the_apostrophe_a_patient_types_does_not_matter() -> None:
    """Uzbek Latin writes the same word four ways depending on the keyboard."""
    for spelling in ("qancha turadi", "qanchа turadi".replace("а", "a")):
        assert names_nothing_to_price(spelling)


def test_hours_are_not_mistaken_for_a_price_question() -> None:
    """ "qanday" shares three letters with "qancha" and means something else
    entirely; a clinic asked its opening hours must still be able to answer.
    """
    assert not asks_a_price("ish vaqtingiz qanday")
    assert not names_nothing_to_price("nechidan nechigacha ishlaysiz")
