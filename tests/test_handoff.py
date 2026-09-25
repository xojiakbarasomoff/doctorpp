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
