"""Reading a conversation before answering it.

These decide whether the assistant greets again and whether it asks for a
number again, so the interesting cases are the ones where getting it wrong
is visible to a patient: greeting somebody for the fifth time, or asking for
a number they already typed.
"""

import pytest

from app.rag.llm import ChatMessage
from app.services.conversation_signals import (
    looks_like_a_greeting,
    looks_like_a_phone_number,
    read_signals,
    render,
)


def _user(content: str) -> ChatMessage:
    return {"role": "user", "content": content}


def _assistant(content: str) -> ChatMessage:
    return {"role": "assistant", "content": content}


# --- is this the first message ---------------------------------------------


def test_no_history_is_the_opening_message() -> None:
    assert read_signals(None).is_opening
    assert read_signals([]).is_opening


def test_one_earlier_turn_is_already_a_conversation() -> None:
    assert not read_signals([_user("salom")]).is_opening


# --- has the patient given a number ----------------------------------------


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


def test_a_number_the_clinic_quoted_is_not_a_number_the_patient_gave() -> None:
    """Only the patient's own turns count. The assistant reading out the
    clinic's phone number must not convince it that the patient has left
    theirs.
    """
    history = [
        _user("telefon raqamingiz bormi?"),
        _assistant("Albatta — +998 33 667 77 88 raqamiga qo'ng'iroq qilsangiz bo'ladi."),
    ]

    assert not read_signals(history).patient_left_number


def test_a_number_given_earlier_is_remembered() -> None:
    history = [_user("salom"), _assistant("Va alaykum assalom!"), _user("+998901234567")]

    assert read_signals(history).patient_left_number


# --- how often we have already asked ---------------------------------------


def test_asking_is_counted_so_it_is_not_repeated() -> None:
    history = [
        _user("implant qancha?"),
        _assistant(
            "Implantatsiya 3 500 000 so'mdan. Qulay vaqtingizni ayting, hamkasbim "
            "qo'ng'iroq qilib kelishib oladi."
        ),
        _user("o'ylab ko'raman"),
        _assistant("Albatta. Telefon raqamingizni qoldirsangiz, o'zimiz bog'lanamiz."),
    ]

    assert read_signals(history).times_asked_for_number == 2


def test_a_reply_that_never_mentions_a_number_is_not_counted_as_asking() -> None:
    history = [
        _user("ish vaqtingiz?"),
        _assistant("Har kuni 09:00 dan 20:00 gacha ishlaymiz."),
    ]

    assert read_signals(history).times_asked_for_number == 0


def test_asking_is_counted_in_russian_and_in_cyrillic_uzbek() -> None:
    """The clinic answers in three written forms, and a counter that only
    understood one of them would let the assistant ask again in the very
    next message — the behaviour this exists to prevent.
    """
    russian = [_user("сколько стоит?"), _assistant("Оставьте номер, коллега перезвонит.")]
    cyrillic = [_user("нарх?"), _assistant("Телефон рақамингизни қолдиринг.")]

    assert read_signals(russian).times_asked_for_number == 1
    assert read_signals(cyrillic).times_asked_for_number == 1


# --- how it reaches the prompt ---------------------------------------------


def test_the_opening_message_is_stated_plainly() -> None:
    rendered = render(read_signals([]))

    assert "first message" in rendered
    assert "not given you a phone number yet" in rendered
    assert "not asked them for their number" in rendered


def test_a_continuing_conversation_says_so() -> None:
    rendered = render(read_signals([_user("salom"), _assistant("Va alaykum assalom!")]))

    assert "middle of this conversation" in rendered
    assert "first message" not in rendered


def test_a_number_already_given_is_stated_so_it_is_not_asked_for_again() -> None:
    rendered = render(read_signals([_user("+998901234567")]))

    assert "already given you a phone number" in rendered


@pytest.mark.parametrize(("asks", "expected"), [(1, "once"), (2, "2 times"), (3, "3 times")])
def test_the_number_of_previous_asks_is_stated(asks: int, expected: str) -> None:
    history: list[ChatMessage] = []
    for _ in range(asks):
        history.append(_user("hmm"))
        history.append(_assistant("Telefon raqamingizni qoldiring."))

    assert expected in render(read_signals(history))


@pytest.mark.parametrize(
    "reply",
    [
        "Qulay vaqtingizni ayting, hamkasbim qo'ng'iroq qilib kelishib oladi.",
        "Qulay vaqtingizni ayting, hamkasbim qoʻngʻiroq qilib kelishib oladi.",
        "Qulay vaqtingizni ayting, hamkasbim qo‘ng‘iroq qilib kelishib oladi.",
        "Скажите удобное время, коллега перезвонит.",
        "Qachon qulay bo'lsa ayting, o'zimiz bog'lanamiz.",
    ],
)
def test_asking_by_offering_the_call_is_counted_even_without_the_word_number(
    reply: str,
) -> None:
    """The phrasing rule 7 actually recommends — offer the callback, not the
    demand — need never contain "raqam" at all. A counter that only knew the
    phone words would read this as "never asked" and let the assistant ask
    again in the very next message, which is the whole problem.

    The apostrophe is written at least three ways in Uzbek Latin depending
    on the keyboard, and all of them have to count.
    """
    history = [_user("implant qancha?"), _assistant(reply)]

    assert read_signals(history).times_asked_for_number == 1


def test_a_number_typed_in_the_message_being_answered_counts_right_away() -> None:
    """The number arrives in the turn being answered, not in history, so a
    facts block reading only history would say "they have not given you a
    phone number yet" directly above a message containing one — and the
    reply would ask for something the patient had just typed.
    """
    signals = read_signals([_user("qabulga yozmoqchiman")], "+998901234567, ertalab qulay")

    assert signals.patient_left_number
    assert "already given you a phone number" in render(signals)


def test_the_current_message_does_not_make_an_opening_look_like_a_continuation() -> None:
    """Someone who opens with their number is still opening the
    conversation, and should still be greeted.
    """
    assert read_signals([], "+998901234567").is_opening


def test_the_patients_own_message_is_never_counted_as_us_asking() -> None:
    assert read_signals([], "telefon raqamingiz bormi?").times_asked_for_number == 0


# --- greeting ---------------------------------------------------------------


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


def test_a_greeting_is_returned_even_in_the_middle_of_a_thread() -> None:
    """The bug this was written for.

    An Instagram thread is one conversation for as long as the account
    exists, so tying the greeting to is_opening meant a patient who came
    back a week later and said "Assalomu alaykum" was answered with the
    clinic's opening hours and no hello at all.
    """
    history = [_user("EKG bormi"), _assistant("Ha, EKG bizda bor.")]

    signals = read_signals(history, "Assalomu alaykum")

    assert signals.patient_greeted_now
    assert not signals.already_greeted
    assert "return the greeting" in render(signals)


def test_a_greeting_already_returned_is_not_returned_twice() -> None:
    history = [_user("Salom"), _assistant("Va alaykum assalom! Nima bo'yicha yordam beray?")]

    signals = read_signals(history, "Assalomu alaykum")

    assert signals.already_greeted
    assert "do not open with one again" in render(signals)


def test_a_patient_who_did_not_greet_is_not_greeted() -> None:
    signals = read_signals([_user("Salom"), _assistant("Va alaykum assalom!")], "EKG bormi")

    assert not signals.patient_greeted_now
    assert "greeted you in this message" not in render(signals)


def test_the_word_further_down_a_reply_is_not_a_greeting_we_gave() -> None:
    """Only the opening counts. A reply that happens to contain the word
    later would otherwise spend the greeting the patient has not had yet.
    """
    history = [
        _user("EKG bormi"),
        _assistant("Ha, bor. Kelganingizda registraturada salom deb ayting, kutib olamiz."),
    ]

    assert not read_signals(history, "Assalomu alaykum").already_greeted


# --- clinic details already given -------------------------------------------


def test_a_detail_we_have_already_given_is_marked_as_given() -> None:
    """The transcript this was written for put the same number and the same
    opening hours under three consecutive messages, one of which was a
    question about ultrasound that never got answered.
    """
    history = [
        _user("qabulga yozilmoqchiman"),
        _assistant(
            "Qabulga yozilish uchun +998 71 200 03 93 ga qo'ng'iroq qiling; "
            "shifokorlar 09:00-18:00 orasida ishlaydi."
        ),
    ]

    signals = read_signals(history, "Uzi boyicha yozvotudm")

    assert signals.number_already_given
    assert signals.hours_already_given
    assert not signals.address_already_given
    rendered = render(signals)
    assert "telephone number and the opening hours" in rendered
    assert "do not print them again" in rendered


def test_nothing_given_yet_says_nothing() -> None:
    signals = read_signals([_user("salom"), _assistant("Va alaykum assalom!")], "UZI bormi?")

    assert not signals.number_already_given
    assert not signals.hours_already_given
    assert "already given them" not in render(signals)


def test_the_patients_own_number_is_not_a_number_we_gave() -> None:
    """Only our side counts. A patient who types their own number has not
    been given the clinic's, and would otherwise never receive it.
    """
    history = [_user("+998901234567"), _assistant("Rahmat, hamkasbim bog'lanadi.")]

    assert not read_signals(history, "UZI qancha?").number_already_given


def test_the_address_is_recognised_from_our_own_wording() -> None:
    history = [
        _user("qayerdasiz"),
        _assistant("Toshkent shahri, Yunusobod tumani, Moyqo'rg'on ko'chasi 11A uy."),
    ]

    signals = read_signals(history, "UZI bormi?")

    assert signals.address_already_given
    assert "the clinic's address" in render(signals)


def test_a_loose_number_is_not_read_as_the_hours() -> None:
    """The FAQ is full of numbers that are not clock times -- room numbers,
    house numbers, how many days a result takes -- and reading one as the
    opening hours would suppress the hours the patient still needs.
    """
    history = [_user("qayerdasiz"), _assistant("Moyqo'rg'on ko'chasi 11A uy, 2-qavat.")]

    assert not read_signals(history, "nechida ochilasiz?").hours_already_given
