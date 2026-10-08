import pytest

from app.services import handoff


def test_the_marker_is_removed_and_hands_the_patient_over() -> None:
    reply, needs_doctor = handoff.extract("Savolingizni doktorga yetkazaman. [[DOCTOR]]")

    assert needs_doctor
    assert reply == "Savolingizni doktorga yetkazaman."
    assert "[[" not in reply


@pytest.mark.parametrize(
    "reply",
    [
        "Doktorga yetkazaman [[DOCTOR]",
        "Doktorga yetkazaman [[doctor]]",
        "Doktorga yetkazaman [[ DOCTOR : narx so'raldi ]]",
        "[[DOCTOR]] Doktorga yetkazaman",
    ],
)
def test_every_shape_of_the_marker_is_removed(reply: str) -> None:
    clean, needs_doctor = handoff.extract(reply)

    assert needs_doctor
    assert clean == "Doktorga yetkazaman"


def test_other_markers_are_left_for_their_own_readers() -> None:
    reply = "Qo'ng'iroq qilamiz. [[CALLBACK:+998901234567|narx]]"

    assert handoff.extract(reply) == (reply, False)


def test_messages_are_the_same_whitespace_aside_and_only_whole() -> None:
    assert handoff.same_message("  Salom,\n doktor ", "Salom, doktor")
    assert not handoff.same_message("Ha", "Ha, albatta")
    assert not handoff.same_message(None, "Salom")


@pytest.mark.parametrize(
    "reply",
    [
        "Bu haqda doktorning o'zi sizga javob beradi.",
        "Bu haqda doktorning o‘zi sizga javob beradi.",
        "Shifokorning oʻzi sizga qoʻngʻiroq qiladi.",
        "Doktorning ozi javob beradi.",
        "Shifokorga yetkazaman, tez orada yozishadi.",
        "Доктор ўзи жавоб беради.",
        "Саволингизни шифокорга етказаман.",
        "Врач сам вам ответит.",
        "Передам ваш вопрос врачу.",
        # The promises real replies made without pinning anything.
        "Yaxshi, doktor bilan aniqlab olaman, bir daqiqa iltimos.",
        "Mayli, sizni doktorga bildiraman.",
        "Sizning so‘rovingizni klinika botiga yubordik.",
        "Суҳбатингизни клиника ботига юбордим.",
        "Тушунарли. Сизнинг рақамни докторга юбордим, WhatsApp орқали боғланиш учун.",
        "Я уточню у врача и напишу вам.",
    ],
)
def test_a_handover_in_words_is_caught_without_the_marker(reply: str) -> None:
    assert handoff.extract(reply) == (reply, True)


@pytest.mark.parametrize(
    "reply",
    [
        "Buni ko'rmasdan aytib bo'lmaydi, qabulga yozib qo'yaymi?",
        "Doktor dushanbadan shanbagacha 09:00 dan 17:00 gacha ishlaydi.",
        "Doktor bilan uchrashuvga yozilmoqchimisiz?",
        "Bot emas, men klinika administratoriman.",
        "Доктор билан қабулга ёзилишни хоҳлайсизми?",
        "Ertaga 11:00 ga yozib qo'ydim.",
        "Врач принимает с 9 до 17.",
    ],
)
def test_ordinary_replies_are_not_handovers(reply: str) -> None:
    assert handoff.extract(reply) == (reply, False)


@pytest.mark.parametrize(
    "reply",
    [
        "Aniqlab, hozir yozaman.",
        "Hozir aniqlab beraman.",
        "Bir daqiqa, ko'rib chiqib yozaman.",
        "Аниқлаб, ҳозир ёзаман.",
        "Сейчас уточню и напишу.",
        "Минутку, узнаю.",
    ],
)
def test_a_promise_to_come_back_is_kept_by_a_person(reply: str) -> None:
    """The assistant only speaks when the patient writes. "Aniqlab, hozir
    yozaman" -- which the persona once told it to say -- is a promise only a
    person can keep, so the conversation goes to the top of the inbox."""
    assert handoff.extract(reply)[1] is True


@pytest.mark.parametrize(
    "reply",
    [
        "Ertaga 10:00 ga hozir yozib qo'yaman.",
        "Hozir ish vaqti tugagan, ertaga 09:00 da ochamiz.",
        "Narxni telefon orqali aniqlashtirib beramiz.",
        "Telefon orqali aniqlab olishingiz mumkin.",
        "Записать вас на завтра?",
    ],
)
def test_booking_and_telephone_lines_are_not_promises(reply: str) -> None:
    assert handoff.extract(reply)[1] is False


@pytest.mark.parametrize(
    "reply",
    [
        "Albatta, administratorimiz shu yerda sizga javob beradi.",
        "Hozir sizni administratorimizga ulayman.",
        "Албатта, администраторимиз шу ерда сизга жавоб беради.",
        "Конечно, администратор вам здесь ответит.",
    ],
)
def test_a_handover_to_the_administrator_flags_the_conversation(reply: str) -> None:
    """What the persona says to "odam bilan gaplashmoqchiman". The model left
    the marker off every time it said it in testing, so nobody was told."""
    assert handoff.extract(reply)[1]
