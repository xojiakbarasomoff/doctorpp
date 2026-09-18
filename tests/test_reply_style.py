"""The last thing between the model and the patient.

Every example here is a message that actually went out, taken from one
conversation of ninety-one. That is the point of the file: these are not
hypotheses about how a model might misbehave, they are the ways this one
did.
"""

import pytest

from app.services.reply_style import MAX_CHARS, problems, tidy

# --- what is cut without asking anybody --------------------------------------


def test_a_greeting_nobody_gave_is_taken_off() -> None:
    """Thirteen messages in a row opened this way, to a patient who had said
    hello once, on the previous day.
    """
    out = tidy(
        "Ва алайкум ассалом, шифокорнинг администратори. Исмингизни ёзинг, илтимос.",
        greeted=False,
        opening=False,
    )

    assert out == "Исмингизни ёзинг, илтимос."


def test_the_greeting_stays_when_they_greeted_you() -> None:
    reply = "Va alaykum assalom! Sizga qanday yordam bera olaman?"

    assert tidy(reply, greeted=True, opening=True) == reply


def test_the_introduction_stays_in_the_first_reply() -> None:
    reply = "Assalomu alaykum, shifokorning administratori. Sizga qanday yordam beray?"

    assert tidy(reply, greeted=True, opening=True) == reply


def test_the_introduction_stays_when_they_asked_who_they_are_writing_to() -> None:
    reply = "Men shifokorning administratoriman."

    assert tidy(reply, greeted=False, opening=False, user_message="Siz kimsiz?") == reply


def test_the_introduction_is_dropped_in_the_fortieth_message() -> None:
    out = tidy(
        "Shifokorning administratori. Rahmat, telefonni oldim.",
        greeted=False,
        opening=False,
        user_message="93 951 00 00",
    )

    assert out == "Rahmat, telefonni oldim."


def test_the_machinery_never_reaches_the_patient() -> None:
    """Sent to a real patient: "...yoki hozir bergan raqamni tasdiqlang — men
    CALLBACK yozaman."
    """
    out = tidy(
        "To'liq raqamingizni yozing — men CALLBACK yozaman.", greeted=False, opening=False
    )

    assert "CALLBACK" not in out
    assert "raqamingizni" in out


def test_the_booking_marker_itself_survives() -> None:
    """It contains the word BOOK, and it is the only reason the appointment
    is written down at all.
    """
    reply = (
        "Shifokorning administratori. Ertaga 16:20 ga yozib qo'ydim. "
        "[[BOOK:2026-09-19T16:20|Asadbek Risqiyev|+998939510000|buyrak]]"
    )

    out = tidy(reply, greeted=False, opening=False)

    assert "[[BOOK:2026-09-19T16:20|Asadbek Risqiyev|+998939510000|buyrak]]" in out
    assert not out.startswith("Shifokorning")


def test_a_machine_date_is_written_the_way_a_person_writes_it() -> None:
    out = tidy("Ertaga — 2026-09-19 soat 16:20da kutamiz.", greeted=False, opening=False)

    assert "2026-09-19" not in out
    assert "19.09.2026" in out


def test_a_sentence_that_promised_something_and_stopped_is_closed() -> None:
    """"Iltimos kelish eslatmasi:" — and then nothing, to a real patient."""
    out = tidy("Yozib qo'ydim. Iltimos kelish eslatmasi:", greeted=False, opening=False)

    assert not out.endswith(":")


def test_a_reply_that_was_only_a_greeting_is_left_alone() -> None:
    """Trimming it would leave the patient with an empty message."""
    assert tidy("Assalomu alaykum!", greeted=False, opening=False) == "Assalomu alaykum!"


# --- what has to be written again --------------------------------------------


def test_two_alphabets_in_one_message_is_a_rewrite() -> None:
    """Sent as-is: "Энг яқин бўш вақтлар: ertaga (shanba) 09:20 ёки 09:40"."""
    faults = problems(
        "Энг яқин бўш вақтлар: ertaga (shanba) 09:20 ёки 09:40",
        script="uz-cyrl",
        greeted=False,
    )

    assert any("two alphabets" in fault for fault in faults)


def test_answering_a_latin_patient_in_cyrillic_is_a_rewrite() -> None:
    faults = problems(
        "Раҳмат, Асадбек. Энди телефон рақамингизни ёзинг.", script="uz-latn", greeted=False
    )

    assert any("latin" in fault for fault in faults)


def test_one_question_per_message() -> None:
    """Asking for the name, the number and the reason at once is how a
    patient answers only the last of them.
    """
    faults = problems(
        "Ismingiz nima? Telefon raqamingiz? Nima bilan murojaat qilyapsiz?",
        script="uz-latn",
        greeted=False,
    )

    assert any("more than one question" in fault for fault in faults)


def test_a_wall_of_text_is_a_rewrite() -> None:
    faults = problems("Salom. " + "juda uzun matn " * 40, script="uz-latn", greeted=True)

    assert any(str(MAX_CHARS) in fault for fault in faults)


@pytest.mark.parametrize(
    "reply",
    [
        "Ertaga 09:20 bo'sh. Sizga qulaymi?",
        "Раҳмат. Телефон рақамингизни ёзинг.",
        "Yakshanba — dam olish kuni. Dushanba 09:00 bo'sh.",
    ],
)
def test_an_ordinary_reply_has_nothing_wrong_with_it(reply: str) -> None:
    script = "uz-cyrl" if "Раҳмат" in reply else "uz-latn"

    assert problems(reply, script=script, greeted=False) == []


def test_what_is_left_after_trimming_still_starts_like_a_sentence() -> None:
    """Live audit, scenario 11: cutting the introduction off the front left
    "siz Axmadaliyev Temur G'iyosiddin o'g'liga yozayapsiz" — lowercase,
    which nobody types.
    """
    out = tidy(
        "Shifokorning administratori — siz Axmadaliyev Temurga yozayapsiz.",
        greeted=False,
        opening=False,
    )

    assert out.startswith("Siz Axmadaliyev")
