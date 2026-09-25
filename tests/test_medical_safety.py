"""The one output check that is code, not a prompt: the reply must never
prescribe, dose, diagnose, or claim to be the clinician.

The interesting cases are the two directions this can fail. A pattern too
loose withholds an ordinary answer -- a refusal that merely mentions "dori"
as a class, a booking confirmation with a time in it, the roster naming a
specialty. A pattern too tight lets a real prescription through dressed in
a slightly different sentence. Both are tested here, deliberately more of
the second than the first: the shipped defaults lean inclusive, and it is
the false negatives that end with somebody standing at a locked door, or
worse, having taken something they should not have.
"""

import pytest

from app.services.medical_safety import DESCRIPTIONS, SAFE_REPLIES, Violation, check

# --- true positives: every category, in Uzbek Latin, Cyrillic and Russian --


@pytest.mark.parametrize(
    ("reply", "category"),
    [
        # A named medicine, by its ending.
        ("Amoksitsillin iching, yordam beradi.", "dori_nomi"),
        ("Ципрофлоксацин ичинг.", "dori_nomi"),
        ("Nolitsin va Omepразол buyuraman.", "dori_nomi"),
        # A dose.
        ("Kuniga 500 mg iching.", "doza"),
        ("2 tabletka iching.", "doza"),
        ("500мг дан 2 марта ичинг.", "doza"),
        # A schedule.
        ("Kuniga ikki mahal iching.", "jadval"),
        ("Kuniga 3 marta qabul qiling.", "jadval"),
        ("Принимайте 2 раза в день.", "jadval"),
        # An instruction to take something.
        ("Dorini ichib turing, unutmang.", "ichish"),
        ("Antibiotikni -- bu muhim -- ichib turing.", "ichish"),
        ("Дорини ичиб туринг.", "ichish"),
        ("Пейте таблетки после еды.", "ichish"),
        # Prescribing in the passive voice.
        ("Bunga antibiotik buyuriladi.", "buyurish"),
        ("Sizga preparat tavsiya qilaman.", "buyurish"),
        # A diagnosis, attributed to the patient.
        ("Sizda prostatit bor, xavotir olmang.", "tashxis"),
        ("Sizning tahlilingizda infeksiya bor.", "tashxis"),
        ("У вас цистит.", "tashxis"),
        ("Albatta sizda saraton bor.", "tashxis"),
        # Telling the patient they need not come.
        ("Bu jiddiy emas, o'zi o'tib ketadi.", "kelmasa_boladi"),
        ("Klinikaga kelish shart emas.", "kelmasa_boladi"),
        ("Ничего страшного, само пройдёт.", "kelmasa_boladi"),
        # Writing a prescription.
        ("Sizga retsept yozib beraman.", "retsept"),
        ("Мен рецепт напишу.", "retsept"),
        # Claiming to be the clinician.
        ("Men shifokorman, sizni ko'raman.", "ozini_shifokor_deb_korsatish"),
        ("Я врач, приходите.", "ozini_shifokor_deb_korsatish"),
    ],
)
def test_every_category_is_caught(reply: str, category: str) -> None:
    violation = check(reply)

    assert violation is not None, f"expected a {category} violation in {reply!r}"
    assert violation.category == category


# --- true negatives: the ordinary, safe replies this must never touch -----


@pytest.mark.parametrize(
    "reply",
    [
        # A refusal that talks about medicine as a class, not an instruction.
        "Dori haqida faqat shifokor, ko'rikdan keyin gapira oladi.",
        "Antibiotik kerakmi, buni urolog hal qiladi.",
        "Buni ko'rmasdan aytib bo'lmaydi, qabulga yozib qo'yaymi?",
        # Naming who treats a condition, not diagnosing the patient with it.
        "Prostatit bilan urolog shug'ullanadi.",
        "С циститом обращайтесь к урологу.",
        # A booking confirmation, with a time in it.
        "Yaxshi, ertaga 16:20 da kutamiz. [[BOOK:2026-09-22T16:20|Ali|+998901234567|tekshiruv]]",
        "Bugun 09.20 da yozib qo'ydim.",
        # A price, a service, a specialty -- none of them a medicine.
        "UZI narxi 150 000 so'm, konsultatsiya bilan birga.",
        "Axmadaliyev Temur, urolog-androlog, dushanbadan shanbagacha ishlaydi.",
        "Sizni bugun 16:20ga yozdim, doktor kutadi.",
        # An ordinary greeting and a short, factual answer.
        "Va alaykum assalom! Qanday yordam bera olaman?",
        "Ha, klinikada bepul avtoturargoh bor.",
        # Reassurance that names no condition and instructs nothing.
        "Xavotir olmang, shifokor ko'rib chiqadi.",
    ],
)
def test_ordinary_replies_are_never_flagged(reply: str) -> None:
    assert check(reply) is None


def test_a_clock_is_never_read_as_a_dose() -> None:
    """"16:20" once read as twenty grams; the class is closed, not the one report."""
    assert check("Sizni 16:20ga yozdim.") is None
    assert check("Ertaga 9.20 da keling.") is None


def test_the_booking_and_callback_markers_are_not_inspected() -> None:
    """The marker carries the patient's own words back -- a complaint, a
    reason for the visit -- and must never be judged as the assistant's."""
    reply = (
        "Bo'ladi, yozdim. "
        "[[BOOK:2026-09-22T09:20|Ali Valiyev|+998901234567|antibiotik ichyapman degan]]"
    )

    assert check(reply) is None


def test_the_first_matching_category_wins_by_the_fixed_order() -> None:
    """Two things wrong in one reply still returns one violation -- the
    caller only needs one reason to reject it."""
    violation = check("Amoksitsillin iching, kuniga 2 mahal.")

    assert violation is not None
    assert violation.category == "dori_nomi"


def test_the_match_is_short_enough_to_log() -> None:
    reply = "Amoksitsillin " + "x" * 200 + " iching."

    violation = check(reply)

    assert violation is not None
    assert len(violation.matched) <= 80


def test_every_category_has_a_description_and_a_reply_in_all_three_scripts() -> None:
    categories = {name for name, _ in _all_categories()}
    assert set(DESCRIPTIONS) == categories
    assert set(SAFE_REPLIES) == {"uz-latn", "uz-cyrl", "ru"}
    assert all(isinstance(v, str) and v for v in DESCRIPTIONS.values())
    assert all(isinstance(v, str) and v for v in SAFE_REPLIES.values())


def _all_categories() -> list[tuple[str, str]]:
    from app.services.medical_safety import _PATTERNS

    return [(name, pattern) for name, pattern in _PATTERNS]


def test_violation_is_a_plain_value() -> None:
    v = Violation(category="doza", matched="500 mg")
    assert v.category == "doza"
    assert v.matched == "500 mg"


@pytest.mark.parametrize(
    "reply",
    [
        "Men dori tavsiya qila olmayman, buni doktor aytadi.",
        "Afsuski, dori tavsiya qilmaymiz — shifokor ko'rigidan keyin aniqlanadi.",
        "Bemorga dori yoki doza tavsiya qilmang, doktorga yo'naltiring.",
    ],
)
def test_refusing_to_recommend_a_medicine_is_not_recommending_one(reply: str) -> None:
    assert check(reply) is None


@pytest.mark.parametrize(
    "reply",
    [
        "Sizga shu dorini tavsiya qilaman.",
        "Tabletka tavsiya qilamiz, kuniga bir marta.",
        "Dori tavsiya qila olaman: paratsetamol.",
    ],
)
def test_recommending_a_medicine_is_still_caught(reply: str) -> None:
    assert check(reply) is not None
