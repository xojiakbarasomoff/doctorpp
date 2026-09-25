"""👍, "ok", "rahmat": answered only where the patient is waiting for us."""

import pytest

from app.rag.llm import ChatMessage
from app.services.small_talk import is_acknowledgement, is_emoji_only, plan


def patient(text: str) -> ChatMessage:
    return ChatMessage(role="user", content=text)


def clinic(text: str) -> ChatMessage:
    return ChatMessage(role="assistant", content=text)


@pytest.mark.parametrize(
    "text",
    [
        "👍",
        "👍🏼👍🏼",
        "❤️🙏",
        "+",
        "...",
        "ok",
        "Ok rahmat",
        "Rahmat katta",
        "Xo'p",
        "xop",
        "Хорошо, спасибо",
        "Спасибо большое",
        "Рахмат",
        "ha",
        "да",
        "Tushunarli, sog' bo'ling",
        "Mayli",
        "Katta rahmat sizga ham",
    ],
)
def test_these_say_nothing_but_ok_or_thanks(text: str) -> None:
    assert is_acknowledgement(text)


@pytest.mark.parametrize(
    "text",
    [
        "?",
        "10",
        "+998901234567",
        "ok ertaga kelaman",
        "Rahmat, yana bir savol",
        "Rahmat, narxi qancha?",
        "Salom",
        "Ha, lekin og'riq kuchaydi",
        "",
        "   ",
        "ok " * 20,
        "Tushunarli, 25-sentyabr bo'ladimi",
    ],
)
def test_these_have_something_in_them(text: str) -> None:
    assert not is_acknowledgement(text)


def test_emoji_only_means_no_letters_and_no_digits() -> None:
    assert is_emoji_only("👍") and is_emoji_only("+") and is_emoji_only("❤️🙏")
    assert not is_emoji_only("ok") and not is_emoji_only("1") and not is_emoji_only("")


BOOKED = [patient("Qabulga yozilmoqchiman"), clinic("Ertaga soat 10:00 da kutamiz.")]
ASKED = [patient("Qabulga yozilmoqchiman"), clinic("Ertaga soat 10:00 sizga qulaymi?")]


@pytest.mark.parametrize("message", ["👍", "ok", "rahmat", "Спасибо", "❤️", "+"])
def test_after_a_statement_an_acknowledgement_gets_silence(message: str) -> None:
    assert plan(message, BOOKED, long_gap=False).silent


@pytest.mark.parametrize("message", ["👍", "+"])
def test_after_a_question_an_emoji_is_read_as_yes(message: str) -> None:
    result = plan(message, ASKED, long_gap=False)
    assert not result.silent
    assert result.note is not None and "ha, roziman" in result.note


@pytest.mark.parametrize("message", ["ha", "ok", "rahmat", "да"])
def test_after_a_question_words_go_to_the_model_as_they_are(message: str) -> None:
    assert plan(message, ASKED, long_gap=False) == plan("Ertaga kelaman", ASKED, long_gap=False)


def test_an_emoji_opening_a_conversation_gets_a_short_hello() -> None:
    result = plan("👍", [], long_gap=False)
    assert not result.silent and result.note is not None
    assert "salomlashing" in result.note


def test_an_emoji_after_a_day_away_opens_the_conversation_again() -> None:
    result = plan("👍", BOOKED, long_gap=True)
    assert not result.silent and result.note is not None and "salomlashing" in result.note


def test_thanks_as_the_very_first_message_is_answered() -> None:
    assert plan("rahmat", [], long_gap=False) == plan("Salom", [], long_gap=False)


@pytest.mark.parametrize(
    "before",
    ["👍", "🏷 Stiker yubordi", "💟 Reaksiya bildirdi: ❤️"],
)
def test_the_second_emoji_of_a_ping_pong_is_not_answered(before: str) -> None:
    """Real conversation: 👍 → greeting + "qanday yordam?" → 👍 → the same
    greeting again. The second 👍 answers nothing."""
    history = [patient(before), clinic("Assalomu alaykum! Qanday yordam bera olaman?")]
    assert plan("👍", history, long_gap=False).silent


def test_words_after_our_question_to_an_emoji_are_still_answered() -> None:
    """👍 → "10:00 qulaymi?" → "ha" is a booking, not a ping-pong."""
    history = [patient("👍"), clinic("Ertaga soat 10:00 qulaymi?")]
    result = plan("ha", history, long_gap=False)
    assert not result.silent


def test_an_ordinary_message_is_always_answered() -> None:
    for history in ([], BOOKED, ASKED):
        assert plan("Qachon kelsam bo'ladi?", history, long_gap=False) == plan(
            "Narxi qancha?", history, long_gap=False
        )
        assert not plan("Qachon kelsam bo'ladi?", history, long_gap=False).silent


def test_a_staff_message_counts_as_ours() -> None:
    """An operator's words reach the model as the assistant's turn."""
    history = [patient("Qabul kerak"), clinic("Doktor ertaga sizga qo'ng'iroq qiladi.")]
    assert plan("rahmat", history, long_gap=False).silent


def test_the_note_cannot_be_stretched_by_a_long_message() -> None:
    result = plan("👍" * 30, ASKED, long_gap=False)
    assert result.note is not None and result.note.count("👍") <= 20
