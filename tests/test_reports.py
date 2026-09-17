"""The report's two judgement calls: what a patient's words are about, and
how the headline numbers are counted.

The classifier is tested on the way patients actually write -- Latin and
Cyrillic Uzbek, Russian, missing apostrophes -- and on the words that used to
fool it: "Toshkent" is not a kidney stone.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.admin.reports import _figures
from app.models.appointment import Appointment
from app.models.message import Message, MessageSender
from app.services.appointment import CLINIC_TIMEZONE
from app.services.complaints import categorise, is_enquiry, patient_words
from tests.conftest import Seed


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("buyragimda tosh bor, siyganda achishadi", ["stones", "urinary_infection"]),
        ("Toshkentdanman, qabulga yozilmoqchiman", []),
        ("prostatit bo'lsa kerak", ["prostate"]),
        ("эрекция слабая", ["erectile"]),
        ("бола бўлмаяпти, спермограмма топширмоқчиман", ["infertility"]),
        ("tez-tez siyaman kechasi", ["frequent_urination"]),
        ("siydikda qon keldi", ["blood"]),
        ("Varik", ["scrotum"]),
        ("belim og'riyapti", ["pain"]),
        ("belim ogriyapti, buyrak", ["stones"]),
        ("Assalomu alaykum", []),
    ],
)
def test_complaints_are_sorted_into_the_doctors_categories(text: str, expected: list[str]) -> None:
    assert [category.key for category in categorise([text])] == expected


def test_the_systems_own_note_is_not_the_patients_complaint() -> None:
    assert patient_words("Varik\nDoktor bu kuni qabul qila olmadi.") == "Varik"
    assert patient_words("Doktor bu vaqtda qabul qila olmadi.") == ""


def test_a_price_question_is_an_enquiry_not_a_complaint() -> None:
    assert is_enquiry(["UZI qancha turadi?"])
    assert categorise(["UZI qancha turadi?"]) == []


async def test_the_period_counts_patients_messages_bookings_and_response_time(
    db_session: AsyncSession, seed: Seed
) -> None:
    today = datetime.now(CLINIC_TIMEZONE).date()
    written = datetime.now(UTC) - timedelta(hours=1)
    conversation = seed.a.conversation
    db_session.add_all(
        [
            Message(
                conversation_id=conversation.id,
                sender=MessageSender.PATIENT,
                content="prostatit, qabulga yozilmoqchiman",
                channel="instagram",
                created_at=written,
            ),
            Message(
                conversation_id=conversation.id,
                sender=MessageSender.BOT,
                content="Ismingizni ayting",
                channel="instagram",
                created_at=written + timedelta(seconds=30),
            ),
            Appointment(
                tenant_id=seed.tenant_a.id,
                user_id=seed.a.user.id,
                conversation_id=conversation.id,
                doctor_name="Test",
                notes="prostata",
                scheduled_at=written + timedelta(days=1),
                status="scheduled",
                source="bot",
            ),
        ]
    )
    await db_session.flush()

    figures = await _figures(db_session, seed.tenant_a.id, today, today)

    assert seed.a.user.id in figures["active"]
    assert figures["patient_messages"] >= 1
    assert any(a.notes == "prostata" for a in figures["appointments"])
    assert figures["conversion"] > 0
    assert figures["median_response"] == pytest.approx(30, abs=1)
    assert figures["complaint_counts"]["prostate"] == 1
    assert figures["enquiry_patients"] >= 1
