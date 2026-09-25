"""When the assistant may greet: decided in code, applied to what it wrote.

The replies below are the assistant's own, from real conversations, where
it greeted a patient mid-complaint and introduced itself for the third time.
"""

import pytest

from app.services.greeting import Mode, asked_who, introduced, mode_for, patient_greeted, tidy

REPEAT = (
    "Va alaykum assalom, Assalomu alaykum, man urolog-androlog Temur Axmadaliyevning "
    "yordamchilari bo'laman, tushundim. Qachondan beri hojatga chiqishda qiyinchilik bor?"
)


def _tidy(
    reply: str,
    mode: Mode,
    *,
    greeted: bool = False,
    before: bool = False,
    who: bool = False,
) -> str:
    return tidy(reply, mode=mode, patient_greeted=greeted, introduced_before=before, asked_who=who)


@pytest.mark.parametrize(
    ("first", "long_gap", "greeted", "mode"),
    [
        (True, False, False, Mode.FULL),
        (True, False, True, Mode.FULL),
        (False, True, False, Mode.FULL),
        (False, False, True, Mode.RETURN_ONLY),
        (False, False, False, Mode.NONE),
    ],
)
def test_the_mode_follows_what_the_backend_knows(
    first: bool, long_gap: bool, greeted: bool, mode: Mode
) -> None:
    assert mode_for(first=first, long_gap=long_gap, patient_greeted=greeted) is mode


def test_mid_conversation_the_greeting_and_the_introduction_go() -> None:
    assert _tidy(REPEAT, Mode.NONE, before=True) == (
        "Tushundim. Qachondan beri hojatga chiqishda qiyinchilik bor?"
    )


def test_a_patient_saying_salom_mid_conversation_gets_one_short_return() -> None:
    reply = (
        "Va alaykum assalom, Assalomu alaykum, man urolog-androlog Temur Axmadaliyevning "
        "yordamchilari bo'laman, qanday yordam berolamiz?"
    )
    assert _tidy(reply, Mode.RETURN_ONLY, greeted=True, before=True) == (
        "Va alaykum assalom. Qanday yordam berolamiz?"
    )


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (
            "Ва алейкум ассалом. Мен доктор администраторйман. Сизга қандай ёрдам бера оламан?",
            "Ассалому алайкум. Мен доктор администраторйман. Сизга қандай ёрдам бера оламан?",
        ),
        (
            "Va alaykum assalom! Qanday yordam bera olaman?",
            "Assalomu alaykum! Qanday yordam bera olaman?",
        ),
    ],
)
def test_nobody_is_greeted_back_who_did_not_greet(reply: str, expected: str) -> None:
    """ "Тимур яхшимисиз" was answered "Ва алейкум ассалом"."""
    assert _tidy(reply, Mode.FULL, greeted=False) == expected


def test_a_first_reply_keeps_its_greeting_and_introduction() -> None:
    reply = "Assalomu alaykum! Men administratorman. Qanday yordam kerak?"
    assert _tidy(reply, Mode.FULL, greeted=True) == reply


def test_back_after_a_day_the_greeting_stays_but_not_a_second_introduction() -> None:
    reply = "Assalomu alaykum! Men administratorman. Qanday yordam kerak?"
    assert _tidy(reply, Mode.FULL, greeted=True, before=True) == (
        "Assalomu alaykum! Qanday yordam kerak?"
    )


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("Assalomu alaykum, Aziz aka! Qabulga qaysi kun qulay?", "Qabulga qaysi kun qulay?"),
        ("😊 Assalomu alaykum! Qanday yordam bera olaman?", "Qanday yordam bera olaman?"),
        # Never introduced here before: the introduction stays, the greeting goes.
        (
            "Здравствуйте! Я администратор врача. Чем могу помочь?",
            "Я администратор врача. Чем могу помочь?",
        ),
        (
            "Добрый день! Запишу вас на завтра. [[BOOK:2026-09-26T10:00]]",
            "Запишу вас на завтра. [[BOOK:2026-09-26T10:00]]",
        ),
        ("Xayrli kun! Ertaga kelasizmi?", "Ertaga kelasizmi?"),
        (
            "Assalomu alaykum. Minnatdorman, yana savolingiz bo'lsa yozing.",
            "Minnatdorman, yana savolingiz bo'lsa yozing.",
        ),
    ],
)
def test_greetings_in_every_script_are_recognised(reply: str, expected: str) -> None:
    assert _tidy(reply, Mode.NONE) == expected


@pytest.mark.parametrize(
    "reply",
    [
        # "Salom" inside a word is not a greeting.
        "Salomatligingiz uchun ertaga kelishingiz yaxshi.",
        "Tushundim. Qaysi kun qulay?",
        # An introduction that is not at the start is the model's own sentence.
        "Ertaga kelsangiz, men administrator sifatida yozib qo'yaman.",
        "Приветствую ваше решение, запишу вас.",
        "",
    ],
)
def test_a_reply_with_no_opening_greeting_is_left_exactly_as_it_was(reply: str) -> None:
    assert _tidy(reply, Mode.NONE, before=True) == reply


def test_a_reply_that_is_only_a_greeting_is_sent_rather_than_sent_empty() -> None:
    assert _tidy("Va alaykum assalom!", Mode.NONE) == "Va alaykum assalom!"


def test_a_booking_marker_survives_the_tidy() -> None:
    reply = "Assalomu alaykum! Ertaga 10:00 ga yozdim. [[BOOK:2026-09-26T10:00]]"
    assert _tidy(reply, Mode.NONE).endswith("[[BOOK:2026-09-26T10:00]]")


@pytest.mark.parametrize(
    ("message", "greeted"),
    [
        ("Assalomu alaykum doktor", True),
        ("Solamalekun", True),
        ("Ассалом алекум", True),
        ("Здрасте", True),
        ("Salom, salomatligim yomon", True),
        ("salomatligim yomon", False),
        ("Саломатлигим яхши эмас", False),
        ("Тимур яхшимисиз", False),
        ("Doktor manda hojatga kopro qatshda muammo bor", False),
    ],
)
def test_what_counts_as_the_patient_saying_hello(message: str, greeted: bool) -> None:
    assert patient_greeted(message) is greeted


def test_an_introduction_in_any_earlier_reply_counts() -> None:
    assert introduced(["Salom!", "Men administratorman, yordam beraman."])
    assert introduced(["Мен доктор администраторйман."])
    assert not introduced(["Ertaga 10:00 da kutamiz."])
    assert not introduced([])


# --- from replaying every real conversation through these rules ------------------


def test_a_patient_asking_who_we_are_keeps_the_answer() -> None:
    """ "Sen kimsan" -- the introduction is the answer."""
    reply = "Men Doctor Temur klinikasining administratori, ismim Madina. Yozing, yordam beraman."
    assert asked_who("Sen kimsan") and asked_who("сен кимсан") and asked_who("Вы бот?")
    assert not asked_who("Qachon kelsam bo'ladi?")
    assert _tidy(reply, Mode.NONE, before=True, who=True) == reply


def test_the_title_after_a_greeting_goes_with_it() -> None:
    reply = "Va alaykum assalom, shifokorning administratori. Qanday maqsadda qo'ng'iroq qilaylik?"
    assert _tidy(reply, Mode.NONE) == "Qanday maqsadda qo'ng'iroq qilaylik?"


def test_a_return_greeting_stays_in_the_alphabet_the_model_wrote_it_in() -> None:
    """A patient greeting in Latin, a reply in Cyrillic: no Latin greeting
    bolted onto a Cyrillic reply."""
    reply = "Ва алайкум ассалом, шифокорнинг администратори. Сизни бугун 16:20га ёзайми?"
    assert _tidy(reply, Mode.RETURN_ONLY, greeted=True, before=True) == (
        "Ва алайкум ассалом. Сизни бугун 16:20га ёзайми?"
    )


@pytest.mark.parametrize(
    "reply",
    [
        "Va alaykum assalom! Qanday yordam bera olaman?",
        "Добрый день. Какая у вас жалоба?",
        "Assalomu alaykum, man urolog-androlog Temur Axmadaliyevning yordamchisiman.",
    ],
)
def test_a_single_greeting_to_a_greeting_is_left_alone(reply: str) -> None:
    assert _tidy(reply, Mode.RETURN_ONLY, greeted=True) == reply
    assert _tidy(reply, Mode.FULL, greeted=True) == reply


def test_a_doubled_greeting_keeps_the_first_and_its_comma() -> None:
    reply = (
        "Va alaykum assalom, Assalomu alaykum, man urolog-androlog Temur Axmadaliyevning "
        "yordamchilari bo'laman, qanday yordam berolamiz?"
    )
    assert _tidy(reply, Mode.FULL, greeted=True) == (
        "Va alaykum assalom, man urolog-androlog Temur Axmadaliyevning "
        "yordamchilari bo'laman, qanday yordam berolamiz?"
    )


def test_an_introduction_never_given_before_is_not_taken_away() -> None:
    reply = "Va alaykum assalom. Men doktorning administratoriman. Qanday yordam kerak?"
    assert _tidy(reply, Mode.RETURN_ONLY, greeted=True, before=False) == reply


# --- a salom is always returned --------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("Xo'p, qaysi kun qulay?", "Va alaykum assalom. Xo'p, qaysi kun qulay?"),
        ("Хўп, қайси кун қулай?", "Ва алайкум ассалом. Хўп, қайси кун қулай?"),
        ("Хорошо, какой день вам удобен?", "Здравствуйте. Хорошо, какой день вам удобен?"),
    ],
)
def test_a_salom_mid_conversation_is_returned_even_when_the_model_forgot(
    reply: str, expected: str
) -> None:
    """Live model, told not to greet mid-conversation, answered "Assalomu
    alaykum" with "Xo'p, qaysi kun qulay?". The return greeting is in the
    reply's own alphabet."""
    got = tidy(reply, mode=Mode.RETURN_ONLY, patient_greeted=True, introduced_before=True)
    assert got == expected


def test_a_first_salom_is_returned_too() -> None:
    got = tidy(
        "qanday yordam kerak?",
        mode=Mode.FULL,
        patient_greeted=True,
        introduced_before=False,
    )
    assert got == "Va alaykum assalom. Qanday yordam kerak?"


def test_nothing_is_added_where_nobody_said_salom() -> None:
    for mode in Mode:
        assert (
            tidy("Qaysi kun qulay?", mode=mode, patient_greeted=False, introduced_before=True)
            == "Qaysi kun qulay?"
        )


def test_a_greeting_the_model_already_returned_is_not_doubled() -> None:
    reply = "Va alaykum assalom! Qaysi kun qulay?"
    assert tidy(reply, mode=Mode.RETURN_ONLY, patient_greeted=True, introduced_before=True) == reply
