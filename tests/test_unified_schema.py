"""The schema changes the Instagram/Telegram merge brought in.

Covers the two tables that arrived from the Telegram side (doctors, leads)
and the one behavioural change to an existing table: a confirmed appointment
now holds its slot.
"""

from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.appointment import ACTIVE_STATUSES, AppointmentStatus
from app.models.lead import LeadStatus
from app.repositories.appointment import AppointmentRepository
from app.repositories.doctor import DoctorRepository
from app.repositories.lead import LeadRepository
from app.services.appointment import (
    SlotAlreadyBookedError,
    cancel_appointment,
    check_availability,
    confirm_appointment,
    create_appointment,
    is_within_working_hours,
)
from tests.conftest import Seed


def _next_free_slot() -> datetime:
    """A grid-aligned slot inside working hours, comfortably in the future.

    Walks forward rather than picking a fixed offset, because the seed
    fixture already books one appointment and the working-hours window does
    not cover every hour of the day.
    """
    candidate = datetime.now(UTC) + timedelta(days=1)
    candidate = candidate.replace(minute=0, second=0, microsecond=0)
    for _ in range(96):
        if is_within_working_hours(candidate):
            return candidate
        candidate += timedelta(minutes=30)
    raise AssertionError("no bookable slot found within 48 hours")


# --- a confirmed booking still holds its slot ---


async def test_confirmed_status_is_treated_as_holding_the_slot() -> None:
    """The set the partial unique index is built from.

    The Instagram side counted only `scheduled`; the Telegram side used
    `pending` and `confirmed` with no uniqueness at all. Reading only
    `scheduled` would report a confirmed appointment's time as free.
    """
    assert AppointmentStatus.SCHEDULED in ACTIVE_STATUSES
    assert AppointmentStatus.CONFIRMED in ACTIVE_STATUSES
    assert AppointmentStatus.CANCELLED not in ACTIVE_STATUSES
    assert AppointmentStatus.NO_SHOW not in ACTIVE_STATUSES
    assert AppointmentStatus.COMPLETED not in ACTIVE_STATUSES


async def test_confirming_a_booking_does_not_free_its_slot(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    slot = _next_free_slot()

    with as_tenant(seed.tenant_a.id):
        repo = AppointmentRepository(db_session)
        booking = await create_appointment(
            repo, scheduled_at=slot, source="bot", patient_name="Aziza"
        )
        await confirm_appointment(repo, booking)

        assert booking.status == AppointmentStatus.CONFIRMED
        assert await check_availability(repo, slot) is False

        with pytest.raises(SlotAlreadyBookedError):
            await create_appointment(
                repo, scheduled_at=slot, source="operator", patient_name="Boshqa bemor"
            )


async def test_cancelling_a_confirmed_booking_frees_its_slot(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    slot = _next_free_slot()

    with as_tenant(seed.tenant_a.id):
        repo = AppointmentRepository(db_session)
        booking = await create_appointment(
            repo, scheduled_at=slot, source="bot", patient_name="Aziza"
        )
        await confirm_appointment(repo, booking)
        await cancel_appointment(repo, booking)

        assert await check_availability(repo, slot) is True
        rebooked = await create_appointment(
            repo, scheduled_at=slot, source="operator", patient_name="Keyingi bemor"
        )

    assert rebooked.status == AppointmentStatus.SCHEDULED


async def test_booking_records_the_doctor_by_id_and_by_name(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    """Both, not one or the other: the id is the live link, the name is what
    was agreed with the patient and must survive the doctor being renamed or
    deactivated.
    """
    slot = _next_free_slot()

    with as_tenant(seed.tenant_a.id):
        booking = await create_appointment(
            AppointmentRepository(db_session),
            scheduled_at=slot,
            source="operator",
            patient_name="Aziza",
            doctor_id=seed.a.doctor.id,
            doctor_name=seed.a.doctor.name,
            patient_phone="+998 90 123 45 67",
            notes="Implant bo'yicha maslahat",
        )

        # Renaming the doctor must not rewrite what the booking says.
        await DoctorRepository(db_session).update(seed.a.doctor, name="Dr. Smith-Karimova")

    assert booking.doctor_id == seed.a.doctor.id
    assert booking.doctor_name == "Dr. Smith"
    assert booking.patient_phone == "+998 90 123 45 67"
    assert booking.notes == "Implant bo'yicha maslahat"


async def test_a_new_booking_starts_with_both_reminders_unsent(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    with as_tenant(seed.tenant_a.id):
        booking = await create_appointment(
            AppointmentRepository(db_session),
            scheduled_at=_next_free_slot(),
            source="bot",
            patient_name="Aziza",
        )

    assert booking.reminder_24h_sent is False
    assert booking.reminder_2h_sent is False


# --- doctors ---


async def test_list_active_returns_only_this_tenants_working_doctors(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    with as_tenant(seed.tenant_a.id):
        repo = DoctorRepository(db_session)
        await repo.create(name="Dr. Anvar", specialty="Implantolog", working_hours="09:00 - 16:00")
        retired = await repo.create(
            name="Dr. Retired", specialty="Ortodont", working_hours="10:00 - 17:00"
        )
        await repo.update(retired, is_active=False)

        active = await repo.list_active()

    names = [doctor.name for doctor in active]
    # Sorted by name, so the booking UI and the bot offer the same order.
    assert names == ["Dr. Anvar", "Dr. Smith"]
    assert seed.b.doctor.id not in {doctor.id for doctor in active}


# --- leads ---


async def test_a_new_lead_starts_as_new(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    with as_tenant(seed.tenant_a.id):
        lead = await LeadRepository(db_session).create(
            patient_name="Aziza",
            phone="+998 90 123 45 67",
            topic="implant narxi",
            convenient_time="kechqurun 6 dan keyin",
        )

    assert lead.status == LeadStatus.NEW
    assert lead.convenient_time == "kechqurun 6 dan keyin"


async def test_leads_are_listed_newest_first_and_filterable_by_status(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    with as_tenant(seed.tenant_a.id):
        repo = LeadRepository(db_session)
        first = await repo.create(patient_name="Birinchi", phone="+998 90 000 00 01")
        second = await repo.create(patient_name="Ikkinchi", phone="+998 90 000 00 02")
        await repo.update(first, status=LeadStatus.CONTACTED)

        recent = await repo.list_recent()
        contacted = await repo.list_recent(status=LeadStatus.CONTACTED)

    # The seed fixture leaves one lead of its own, so only the ordering of
    # the two created here is asserted.
    ids = [lead.id for lead in recent]
    assert ids.index(second.id) < ids.index(first.id)
    assert [lead.id for lead in contacted] == [first.id]


async def test_a_conversations_lead_is_found_so_it_is_not_raised_twice(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    """A patient who leaves a number and keeps chatting must not generate a
    second lead — the call centre would ring them twice.
    """
    with as_tenant(seed.tenant_a.id):
        found = await LeadRepository(db_session).get_open_for_conversation(seed.a.conversation.id)

    assert found is not None
    assert found.id == seed.a.lead.id


async def test_another_tenants_conversation_has_no_lead_here(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    with as_tenant(seed.tenant_a.id):
        found = await LeadRepository(db_session).get_open_for_conversation(seed.b.conversation.id)

    assert found is None
