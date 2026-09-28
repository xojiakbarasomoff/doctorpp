"""Complaints under their medical names, in the appointment book's Izoh.

Most cases are the book's own Izoh cells on the day this was written.
"""

import pytest

from app.services.complaint_terms import canonical, canonical_izoh

# --- the doctor's grouping: all of it is "Jinsiy zaiflik" -------------------------------


@pytest.mark.parametrize(
    "said",
    [
        "tez ejakulatsiya",
        "тез бошаниш",
        "tez bo‘shanish",
        "tez bo'shanish",
        "erta tugash",
        "erta bo'shanish",
        "жинсий заифлик",
        "jinsiy zaiflik",
        "Турмаяпти",
        "turmayapti",
        "erektsiya yo'q",
        "эрта бўшаниш",
        "преждевременная эякуляция",
        "слабая эрекция",
        "импотенция",
        "potentsiya pasaygan",
    ],
)
def test_early_ejaculation_and_weak_erection_are_both_jinsiy_zaiflik(said: str) -> None:
    assert canonical(said) == "Jinsiy zaiflik"


# --- the rest under their medical names ----------------------------------------------------


@pytest.mark.parametrize(
    ("said", "name"),
    [
        ("Боли в паховом облости", "Chov sohasida og'riq"),
        ("chov og'riyapti", "Chov sohasida og'riq"),
        ("prostata adenomasi", "Prostata adenomasi"),
        ("аденома простаты", "Prostata adenomasi"),
        ("простатит", "Prostatit"),
        ("Варикацеле класлами?", "Varikotsele"),
        ("varikotsele", "Varikotsele"),
        ("гидроцеле", "Gidrotsele"),
        ("fimoz", "Fimoz"),
        ("sistit", "Sistit"),
        ("buyragimda tosh bor", "Siydik-tosh kasalligi"),
        ("камни в почках", "Siydik-tosh kasalligi"),
        ("siyganda achishadi", "Dizuriya"),
        ("жжение при мочеиспускании", "Dizuriya"),
        ("siydikda qon", "Gematuriya"),
        ("kechasi 3 marta siyaman", "Nikturiya"),
        ("tez-tez siyaman", "Pollakiuriya (tez-tez siyish)"),
        ("farzand bolmayapti", "Erkak bepushtligi"),
        ("бесплодие", "Erkak bepushtligi"),
    ],
)
def test_other_complaints_take_their_medical_name(said: str, name: str) -> None:
    assert canonical(said) == name


def test_each_complaint_in_a_note_is_renamed_on_its_own_and_duplicates_merge() -> None:
    assert canonical("prostata adenomasi, siyishda muammo va jinsiy zaiflik") == (
        "Prostata adenomasi, siyishda muammo, Jinsiy zaiflik"
    )
    assert canonical("эрта бўшаниш ва простатит") == "Jinsiy zaiflik, Prostatit"
    assert canonical("tez bo'shanish, erta tugash") == "Jinsiy zaiflik"


# --- what is not known is not guessed -------------------------------------------------------


@pytest.mark.parametrize(
    "said",
    [
        "qichishish va siydikdagi noqulaylik",  # vague: kept exactly, "va" and all
        "prostata og'rig'i",  # pain, not a named diagnosis
        "siyishda muammo",
        "Salom",
        "",
        "paxta",  # contains "pax", means cotton
        "kerak",  # contains "erek"
        "turmush o'rtog'im uchun",  # "turmush", not "turmaydi"
        "chovgum",
    ],
)
def test_a_note_with_nothing_recognised_comes_back_exactly(said: str) -> None:
    assert canonical(said) == said


def test_symptoms_are_named_as_symptoms_never_as_a_diagnosis() -> None:
    """Burning on urination is dysuria -- not cystitis, which is the doctor's call."""
    assert canonical("siyganda achishadi") == "Dizuriya"
    assert "Sistit" not in canonical("siyganda achishadi")


# --- an Izoh cell as the book stores it ------------------------------------------------------


@pytest.mark.parametrize(
    ("izoh", "expected"),
    [
        (
            "Bekor: Doktor bu kuni qabul qila olmadi. · prostata adenomasi, siyishda muammo "
            "va jinsiy zaiflik",
            "Bekor: Doktor bu kuni qabul qila olmadi. · Prostata adenomasi, siyishda muammo, "
            "Jinsiy zaiflik",
        ),
        (
            "Bekor: Doktor bu vaqtda qabul qila olmadi. · prostata og'rig'i",
            "Bekor: Doktor bu vaqtda qabul qila olmadi. · prostata og'rig'i",
        ),
        ("Bekor: tez tugab qoldi deb bekor qildi", "Bekor: tez tugab qoldi deb bekor qildi"),
        ("tez ejakulatsiya", "Jinsiy zaiflik"),
    ],
)
def test_the_cancellation_reason_is_left_alone_and_the_complaint_renamed(
    izoh: str, expected: str
) -> None:
    assert canonical_izoh(izoh) == expected
