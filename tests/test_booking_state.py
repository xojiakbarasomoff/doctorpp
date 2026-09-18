"""Which of the three things the clinic already has.

The sequences here are lifted from a real conversation in which the
assistant asked one patient for their telephone number three times in four
messages and made them confirm the same reason twice.
"""

from app.rag.llm import ChatMessage
from app.services.booking_state import BookingState, read, render


def _user(content: str) -> ChatMessage:
    return ChatMessage(role="user", content=content)


def _bot(content: str) -> ChatMessage:
    return ChatMessage(role="assistant", content=content)


# --- reading the conversation -------------------------------------------


def test_the_three_things_are_read_out_of_the_conversation() -> None:
    history = [
        _user("Ertaga 12ga yozib qoyin"),
        _bot("Yaxshi, to'liq ism va familiyangizni yuboring, iltimos."),
        _user("Asadbek Risqiyev"),
        _bot("Rahmat. Iltimos, telefon raqamingizni yuboring."),
        _user("93 951 4499"),
        _bot("Iltimos, qabulga kelish sababingizni yozing."),
    ]

    state = read(history, "Konsultatsiya")

    assert state.name == "Asadbek Risqiyev"
    assert state.phone is not None and state.phone.endswith("939514499")
    assert state.reason == "Konsultatsiya"
    assert state.next_needed is None


def test_a_number_already_given_is_never_asked_for_again() -> None:
    """From the transcript: the number arrived, and the next three replies
    asked for it in "+998..." form, then asked to confirm it, then asked
    again.
    """
    history = [
        _user("Assalomu alaykum"),
        _bot("Ismingizni yozing."),
        _user("Asadbek Risqiyev 93 9510000"),
    ]

    state = read(history, "93 951 00 00")

    assert state.phone is not None
    assert "Never ask for any of these again" in render(state)


def test_a_bare_yes_is_not_a_reason() -> None:
    """"Ha" as the complaint would put the word "Ha" in the clinic's column
    where the reason for the visit belongs.
    """
    history = [
        _bot("Ismingizni yozing."),
        _user("Asadbek Risqiyev"),
        _bot("Telefon raqamingizni yozing."),
        _user("93 951 4499"),
        _bot("Qabul sababini yozing."),
    ]

    assert read(history, "Ha").reason is None
    assert read(history, "Buyrak ogrigi").reason == "Buyrak ogrigi"


def test_the_questions_are_asked_in_the_clinics_own_order() -> None:
    assert read([], "Qabulga yozilmoqchiman").next_needed == "name"

    with_name = read([_bot("Ismingizni yozing."), _user("Asadbek Risqiyev")], "ok")
    assert with_name.next_needed == "phone"


def test_a_patient_who_is_not_booking_is_not_being_collected_from() -> None:
    """The section must not appear for somebody asking the opening hours, or
    the assistant will start asking them for a name.
    """
    state = read([], "Yakshanba kunichi?")

    assert state.in_progress is False
    assert render(state) == ""


def test_asking_to_come_starts_the_three_questions() -> None:
    state = read([], "Man kelasi seshanba 10:00ga qabulga yozilmoqchiman")

    assert state.in_progress is True
    assert "still missing" in render(state)


# --- what the model is told ---------------------------------------------


def test_everything_in_hand_means_offer_a_time_not_another_question() -> None:
    section = render(
        BookingState(
            name="Asadbek Risqiyev",
            phone="+998939514499",
            reason="Buyrak og'rig'i",
            in_progress=True,
        )
    )

    assert "You have all three" in section
    assert "Asadbek Risqiyev" in section
    assert "+998939514499" in section
    assert "Buyrak og'rig'i" in section


def test_the_open_question_is_named_outright() -> None:
    section = render(
        BookingState(name="Asadbek Risqiyev", phone=None, reason=None, in_progress=True)
    )

    assert "telephone number" in section
    assert "Asadbek Risqiyev" in section


def test_giving_out_the_clinics_number_is_not_the_start_of_a_booking() -> None:
    """A turn with the word "telefon" in it is not a question. Without that
    distinction, a patient who asked for the phone number was answered, and
    then asked for their full name.
    """
    history = [
        _user("Telefon raqamingiz bormi?"),
        _bot("Klinika telefoni: +998 70 310 40 40."),
    ]

    state = read(history, "rahmat")

    assert state.in_progress is False
    assert render(state) == ""


def test_a_time_the_patient_already_named_is_carried() -> None:
    """Live audit: a patient who opened with "ertaga 10:00ga yozilmoqchiman"
    was read the whole list back once the three questions were answered,
    instead of simply being booked for the time they asked for.
    """
    history = [
        _user("ertaga 10:00ga yozilmoqchiman"),
        _bot("Ismingizni yozing."),
        _user("Jahongir Sobirov"),
        _bot("Telefon raqamingizni yozing."),
        _user("90 123 45 67"),
        _bot("Qabul sababini yozing."),
    ]

    state = read(history, "uzi")

    assert state.wanted_when is not None and "10:00" in state.wanted_when
    section = render(state)
    assert "already said when they want to come" in section
    assert "do not read the list back" in section


def test_the_latest_time_wins_when_they_change_it() -> None:
    history = [
        _user("ertaga 10:00ga yozilmoqchiman"),
        _bot("Ismingizni yozing."),
        _user("Jahongir Sobirov"),
    ]

    state = read(history, "yo'q, ertaga emas, indinga dedim")

    assert state.wanted_when is not None and "indinga" in state.wanted_when


def test_somebody_who_is_not_booking_has_no_wanted_time() -> None:
    assert read([], "ertaga ishlaysizmi?").wanted_when is None


def test_a_patient_who_is_already_booked_is_not_collected_from_again() -> None:
    """From production: the patient booked, said "rahmat", and was asked for
    their name again — the booking had scrolled out of the nine-turn window
    the model is given, so the code decided nothing had been collected.
    """
    from app.services.booking_state import Booked

    booked = Booked(when="19.09.2026 11:40", name="Asadbek Risqiyev", phone="+998939510000")
    history = [_bot("Ismingizni yozing."), _user("ha")]

    state = read(history, "rahmat", booked=booked)

    assert state.next_needed is None
    assert state.name == "Asadbek Risqiyev"
    section = render(state)
    assert "ALREADY BOOKED: 19.09.2026 11:40" in section
    assert "do not start a new booking" in section
    assert "still missing" not in section
