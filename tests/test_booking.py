"""Offering a real slot and writing the appointment it turns into.

The cases worth having are the ones a patient would notice: being offered a
time that has already passed, being offered one somebody else has, being
told they are booked when they are not, and seeing the machinery.
"""

import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.appointment import AppointmentStatus
from app.repositories.appointment import AppointmentRepository
from app.services.appointment import CLINIC_TIMEZONE, create_appointment
from app.services.booking import (
    HORIZON_DAYS,
    MAX_SLOTS,
    SLOT_LOST_NOTICE,
    extract,
    free_slots,
    render,
    settle,
)
from tests.conftest import Seed


def _local(*args: int) -> datetime:
    return datetime(*args, tzinfo=CLINIC_TIMEZONE)  # type: ignore[arg-type]


# --- reading the diary ------------------------------------------------------


async def test_only_slots_still_ahead_are_offered(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    """Mid-morning, this morning's slots are gone. Offering one is how a
    patient is told to come at a time that has already passed.
    """
    now = _local(2026, 9, 7, 12, 40)
    with as_tenant(seed.tenant_a.id):
        slots = await free_slots(AppointmentRepository(db_session), now)

    assert slots
    assert all(slot > now for slot in slots)
    assert slots[0] == _local(2026, 9, 7, 13, 0)


async def test_a_booked_slot_is_not_offered(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    now = _local(2026, 9, 7, 12, 40)
    taken = _local(2026, 9, 7, 13, 0)
    with as_tenant(seed.tenant_a.id):
        repo = AppointmentRepository(db_session)
        await create_appointment(
            repo, scheduled_at=taken.astimezone(UTC), source="operator", patient_name="Someone"
        )
        slots = await free_slots(repo, now)

    assert taken not in slots
    assert slots[0] == _local(2026, 9, 7, 13, 30)


async def test_a_cancelled_appointment_frees_its_slot_again(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    now = _local(2026, 9, 7, 12, 40)
    slot = _local(2026, 9, 7, 13, 0)
    with as_tenant(seed.tenant_a.id):
        repo = AppointmentRepository(db_session)
        appointment = await create_appointment(
            repo, scheduled_at=slot.astimezone(UTC), source="operator", patient_name="Someone"
        )
        appointment.status = AppointmentStatus.CANCELLED
        await db_session.flush()
        slots = await free_slots(repo, now)

    assert slot in slots


async def test_the_list_is_capped_so_it_cannot_crowd_out_the_answer(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    """An empty diary over the whole horizon is sixty free slots. All of them
    in the prompt would push out the FAQ context that answers what the
    patient actually asked.
    """
    with as_tenant(seed.tenant_a.id):
        slots = await free_slots(AppointmentRepository(db_session), _local(2026, 9, 7, 9, 0))

    assert len(slots) == MAX_SLOTS


async def test_another_clinics_bookings_do_not_block_this_ones_diary(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    now = _local(2026, 9, 7, 12, 40)
    slot = _local(2026, 9, 7, 13, 0)
    with as_tenant(seed.tenant_b.id):
        await create_appointment(
            AppointmentRepository(db_session),
            scheduled_at=slot.astimezone(UTC),
            source="operator",
            patient_name="Other clinic",
        )
    with as_tenant(seed.tenant_a.id):
        slots = await free_slots(AppointmentRepository(db_session), now)

    assert slot in slots


# --- what the model is told -------------------------------------------------


def test_the_current_time_is_stated_because_bugun_depends_on_it() -> None:
    rendered = render([_local(2026, 9, 7, 13, 0)], _local(2026, 9, 7, 12, 40))

    assert "2026-09-07 12:40" in rendered
    assert "13:00" in rendered


def test_slots_are_grouped_by_day_so_today_and_tomorrow_are_distinguishable() -> None:
    rendered = render(
        [_local(2026, 9, 7, 18, 0), _local(2026, 9, 8, 9, 0)], _local(2026, 9, 7, 12, 40)
    )

    assert "2026-09-07: 18:00" in rendered
    assert "2026-09-08: 09:00" in rendered


def test_today_and_tomorrow_are_named_rather_than_left_to_be_worked_out() -> None:
    """Asked to derive the day word from the date, the model got it backwards
    in the direction that costs a booking: on 31 August a patient was offered
    "ertaga 2026-08-31", said no to a day that was actually today, and the
    slot everyone thought was agreed was never written down.

    The date stays on the line because rule 8's marker is written from it.
    """
    rendered = render(
        [_local(2026, 9, 7, 18, 0), _local(2026, 9, 8, 9, 0), _local(2026, 9, 9, 9, 0)],
        _local(2026, 9, 7, 12, 40),
    )

    assert "TODAY — Monday 2026-09-07: 18:00" in rendered
    assert "TOMORROW — Tuesday 2026-09-08: 09:00" in rendered
    # The third day gets no word: "u kun" is not a thing every language has,
    # and a wrong one is worse than none.
    assert "Wednesday 2026-09-09: 09:00" in rendered
    assert "TODAY — Wednesday" not in rendered


def test_the_patient_is_not_read_a_machine_readable_date() -> None:
    """ "ertaga 2026-09-01 kuni soat 09:00" is what the model wrote unprompted,
    and it reads as notes rather than as somebody typing.
    """
    rendered = render([_local(2026, 9, 7, 18, 0)], _local(2026, 9, 7, 12, 40))

    assert "Never read a date out" in rendered


def test_an_empty_diary_tells_the_model_not_to_offer_anything() -> None:
    """Without this the model has an empty list and a rule telling it to
    offer a time, which is the shape that produces an invented one.
    """
    rendered = render([], _local(2026, 9, 7, 12, 40))

    assert "nothing free" in rendered
    assert "Do not offer a time" in rendered
    assert str(HORIZON_DAYS + 1) in rendered


# --- the marker -------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "Bo'ldi, 13:00 ga yozdim. [[BOOK:2026-09-07T13:00]]",
        "Bo'ldi, 13:00 ga yozdim.[[BOOK: 2026-09-07T13:00 ]]",
        "Bo'ldi, 13:00 ga yozdim. [[ BOOK : 2026-09-07 13:00 ]]",
    ],
)
def test_the_slot_is_read_and_the_marker_never_reaches_the_patient(reply: str) -> None:
    text, slot, _ = extract(reply)

    assert slot == _local(2026, 9, 7, 13, 0)
    assert "BOOK" not in text
    assert "[[" not in text
    assert text == "Bo'ldi, 13:00 ga yozdim."


def test_a_reply_without_a_marker_is_left_alone() -> None:
    text, slot, _ = extract("Klinika 09:00 dan 19:00 gacha ishlaydi.")

    assert slot is None
    assert text == "Klinika 09:00 dan 19:00 gacha ishlaydi."


def test_a_malformed_marker_is_still_stripped_from_what_the_patient_sees() -> None:
    """Showing a patient "[[BOOK:" is worse than failing to book: it is the
    one thing that says out loud that nobody is typing.
    """
    text, slot, _ = extract("Yozib qo'ydim. [[BOOK:tomorrow at two]]")

    assert slot is None
    assert "[[BOOK" not in text


# --- writing the appointment ------------------------------------------------


async def test_an_accepted_slot_reaches_the_clinics_book(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    slot = datetime.now(UTC).astimezone(CLINIC_TIMEZONE) + timedelta(days=1)
    slot = slot.replace(hour=13, minute=0, second=0, microsecond=0)

    with as_tenant(seed.tenant_a.id):
        repo = AppointmentRepository(db_session)
        text, appointment = await settle(
            repo,
            f"Bo'ldi, {slot:%H:%M} ga yozdim. [[BOOK:{slot:%Y-%m-%dT%H:%M}]]",
            user_id=seed.a.user.id,
            conversation_id=seed.a.conversation.id,
            source="telegram",
        )

    assert appointment is not None
    assert appointment.scheduled_at == slot.astimezone(UTC)
    assert appointment.status == AppointmentStatus.SCHEDULED
    assert appointment.source == "telegram"
    assert appointment.conversation_id == seed.a.conversation.id
    assert "BOOK" not in text


async def test_a_slot_taken_since_it_was_offered_is_admitted_not_confirmed(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    """The one failure that ends with somebody standing in a waiting room:
    being told they are booked when they are not.
    """
    slot = datetime.now(UTC).astimezone(CLINIC_TIMEZONE) + timedelta(days=1)
    slot = slot.replace(hour=14, minute=0, second=0, microsecond=0)

    with as_tenant(seed.tenant_a.id):
        repo = AppointmentRepository(db_session)
        await create_appointment(
            repo, scheduled_at=slot.astimezone(UTC), source="operator", patient_name="Faster"
        )
        text, appointment = await settle(
            repo,
            f"Bo'ldi, {slot:%H:%M} ga yozdim. [[BOOK:{slot:%Y-%m-%dT%H:%M}]]",
            user_id=seed.a.user.id,
            conversation_id=seed.a.conversation.id,
            source="telegram",
        )

    assert appointment is None
    assert text.endswith(SLOT_LOST_NOTICE.strip())


async def test_a_reply_with_no_booking_writes_nothing(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    with as_tenant(seed.tenant_a.id):
        repo = AppointmentRepository(db_session)
        before = len(await repo.list())
        text, appointment = await settle(
            repo,
            "Klinika 09:00 dan 19:00 gacha ishlaydi.",
            user_id=seed.a.user.id,
            conversation_id=seed.a.conversation.id,
            source="telegram",
        )
        after = len(await repo.list())

    assert appointment is None
    assert before == after
    assert text == "Klinika 09:00 dan 19:00 gacha ishlaydi."


async def test_a_slot_outside_working_hours_is_refused_rather_than_booked(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    """The model is told to pick from the list, but the list is not what
    enforces this — an appointment at three in the morning must fail on the
    way to the table, not on the way into the prompt.
    """
    slot = datetime.now(UTC).astimezone(CLINIC_TIMEZONE) + timedelta(days=1)
    slot = slot.replace(hour=3, minute=0, second=0, microsecond=0)

    with as_tenant(seed.tenant_a.id):
        text, appointment = await settle(
            AppointmentRepository(db_session),
            f"Bo'ldi. [[BOOK:{slot:%Y-%m-%dT%H:%M}]]",
            user_id=seed.a.user.id,
            conversation_id=seed.a.conversation.id,
            source="telegram",
        )

    assert appointment is None
    assert "BOOK" not in text


async def test_a_booking_failure_never_costs_the_patient_their_reply(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    """A booking that did not happen is recoverable by a human reading the
    dashboard. A message that never arrives is not.
    """
    slot = datetime.now(UTC).astimezone(CLINIC_TIMEZONE) + timedelta(days=1)
    slot = slot.replace(hour=15, minute=0, second=0, microsecond=0)

    with as_tenant(seed.tenant_a.id):
        text, appointment = await settle(
            AppointmentRepository(db_session),
            f"Ertaga {slot:%H:%M} da kutamiz. [[BOOK:{slot:%Y-%m-%dT%H:%M}]]",
            user_id=uuid.uuid4(),  # no such patient — the insert will fail
            conversation_id=seed.a.conversation.id,
            source="telegram",
        )

    assert appointment is None
    assert text.startswith("Ertaga")


def test_the_marker_can_carry_the_patients_name() -> None:
    """A row in the clinic's diary with no name is one the front desk cannot
    act on, so the assistant asks and passes it along.
    """
    text, slot, name = extract("Yozib qo'ydim. [[BOOK:2026-09-07T13:00|Nodira Karimova]]")

    assert name == "Nodira Karimova"
    assert slot == _local(2026, 9, 7, 13, 0)
    assert text == "Yozib qo'ydim."


def test_a_booking_without_a_name_is_still_a_booking() -> None:
    """Refusing to write the appointment because nobody said their name
    would throw away the appointment to protect a tidy column.
    """
    _, slot, name = extract("Yozib qo'ydim. [[BOOK:2026-09-07T13:00]]")

    assert slot is not None
    assert name is None


async def test_the_name_reaches_the_clinics_book(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    slot = datetime.now(UTC).astimezone(CLINIC_TIMEZONE) + timedelta(days=1)
    slot = slot.replace(hour=16, minute=0, second=0, microsecond=0)

    with as_tenant(seed.tenant_a.id):
        _, appointment = await settle(
            AppointmentRepository(db_session),
            f"Bo'ldi. [[BOOK:{slot:%Y-%m-%dT%H:%M}|Nodira Karimova]]",
            user_id=seed.a.user.id,
            conversation_id=seed.a.conversation.id,
            source="telegram",
        )

    assert appointment is not None
    assert appointment.patient_name == "Nodira Karimova"


@pytest.mark.parametrize("written", ["Name", "name", "<ism>", "[Bemor]", "имя", "F.I.O"])
def test_the_placeholder_is_not_recorded_as_somebody_s_name(written: str) -> None:
    """A model copying the marker's shape rather than filling it in writes
    the placeholder itself. "Name" in the clinic's diary is worse than an
    empty cell: the front desk cannot tell it is not a person.
    """
    _, _, name = extract(f"Yozdim. [[BOOK:2026-09-07T13:00|{written}]]")

    assert name is None


async def test_reconfirming_a_booking_does_not_tell_the_patient_they_lost_it(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    """The turn after booking, the patient answers "my name is ..." and the
    assistant confirms again, marker and all. Against a slot this very
    conversation owns, "sorry, that time was just taken" is the most
    confusing sentence this code could produce — and it is the common case,
    not the rare one.
    """
    slot = datetime.now(UTC).astimezone(CLINIC_TIMEZONE) + timedelta(days=1)
    slot = slot.replace(hour=11, minute=0, second=0, microsecond=0)
    marker = f"[[BOOK:{slot:%Y-%m-%dT%H:%M}]]"

    with as_tenant(seed.tenant_a.id):
        repo = AppointmentRepository(db_session)
        _, first = await settle(
            repo,
            f"Yozib qo'ydim. {marker}",
            user_id=seed.a.user.id,
            conversation_id=seed.a.conversation.id,
            source="telegram",
        )
        text, second = await settle(
            repo,
            f"Rahmat, Nodira. Tasdiqladim. [[BOOK:{slot:%Y-%m-%dT%H:%M}|Nodira Karimova]]",
            user_id=seed.a.user.id,
            conversation_id=seed.a.conversation.id,
            source="telegram",
        )

    assert first is not None
    assert second is not None
    assert second.id == first.id
    assert SLOT_LOST_NOTICE.strip() not in text
    # The name arrived only with the second marker, and is kept.
    assert second.patient_name == "Nodira Karimova"


async def test_a_real_name_replaces_a_placeholder_that_slipped_through(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    """Catching every placeholder is a losing game — the model has written
    "Name", "Ismingiz" and "Ism-sharif", each a way of saying "put the name
    here". So when one gets past, the real name arriving a turn later must
    be able to overwrite it rather than find the column already full.
    """
    slot = datetime.now(UTC).astimezone(CLINIC_TIMEZONE) + timedelta(days=1)
    slot = slot.replace(hour=17, minute=0, second=0, microsecond=0)

    with as_tenant(seed.tenant_a.id):
        repo = AppointmentRepository(db_session)
        await create_appointment(
            repo,
            scheduled_at=slot.astimezone(UTC),
            source="telegram",
            user_id=seed.a.user.id,
            conversation_id=seed.a.conversation.id,
            patient_name="Ism-sharif",
        )
        _, updated = await settle(
            repo,
            f"Rahmat. [[BOOK:{slot:%Y-%m-%dT%H:%M}|Nodira Karimova]]",
            user_id=seed.a.user.id,
            conversation_id=seed.a.conversation.id,
            source="telegram",
        )

    assert updated is not None
    assert updated.patient_name == "Nodira Karimova"


async def test_another_patients_booking_is_still_reported_as_lost(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    """The guard above must not swallow a genuine race — a slot held by
    somebody else is still gone.
    """
    slot = datetime.now(UTC).astimezone(CLINIC_TIMEZONE) + timedelta(days=1)
    slot = slot.replace(hour=12, minute=0, second=0, microsecond=0)

    with as_tenant(seed.tenant_a.id):
        repo = AppointmentRepository(db_session)
        await create_appointment(
            repo, scheduled_at=slot.astimezone(UTC), source="operator", patient_name="Somebody else"
        )
        text, appointment = await settle(
            repo,
            f"Yozib qo'ydim. [[BOOK:{slot:%Y-%m-%dT%H:%M}]]",
            user_id=seed.a.user.id,
            conversation_id=seed.a.conversation.id,
            source="telegram",
        )

    assert appointment is None
    assert SLOT_LOST_NOTICE.strip() in text


async def test_one_conversation_never_ends_up_with_two_appointments(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    """Live, the assistant booked 14:30, then confirmed again a turn later
    naming 15:00 — and the patient held two appointments, one of which
    nobody would come to, while the diary disagreed with what they had been
    told. The second marker moves the booking instead of adding one.
    """
    base = datetime.now(UTC).astimezone(CLINIC_TIMEZONE) + timedelta(days=1)
    first_slot = base.replace(hour=14, minute=30, second=0, microsecond=0)
    second_slot = base.replace(hour=15, minute=0, second=0, microsecond=0)

    with as_tenant(seed.tenant_a.id):
        repo = AppointmentRepository(db_session)
        _, first = await settle(
            repo,
            f"Yozib qo'ydim. [[BOOK:{first_slot:%Y-%m-%dT%H:%M}]]",
            user_id=seed.a.user.id,
            conversation_id=seed.a.conversation.id,
            source="telegram",
        )
        _, second = await settle(
            repo,
            f"Ertaga kutamiz. [[BOOK:{second_slot:%Y-%m-%dT%H:%M}|Nodira Karimova]]",
            user_id=seed.a.user.id,
            conversation_id=seed.a.conversation.id,
            source="telegram",
        )
        live = [
            appointment
            for appointment in await repo.list()
            if appointment.conversation_id == seed.a.conversation.id
            and appointment.status == AppointmentStatus.SCHEDULED
        ]

    assert first is not None and second is not None
    assert second.id == first.id
    assert len(live) == 1
    # The diary says what the patient was told.
    assert second.scheduled_at == second_slot.astimezone(UTC)
    assert second.patient_name == "Nodira Karimova"


async def test_a_move_onto_somebody_elses_slot_keeps_the_booking_they_have(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    """Losing the appointment they already hold, to a time that was never
    available, is worse than the reply naming a time the diary does not.
    """
    base = datetime.now(UTC).astimezone(CLINIC_TIMEZONE) + timedelta(days=1)
    theirs = base.replace(hour=9, minute=30, second=0, microsecond=0)
    taken = base.replace(hour=10, minute=0, second=0, microsecond=0)

    with as_tenant(seed.tenant_a.id):
        repo = AppointmentRepository(db_session)
        await create_appointment(
            repo, scheduled_at=taken.astimezone(UTC), source="operator", patient_name="Someone else"
        )
        _, mine = await settle(
            repo,
            f"Yozib qo'ydim. [[BOOK:{theirs:%Y-%m-%dT%H:%M}]]",
            user_id=seed.a.user.id,
            conversation_id=seed.a.conversation.id,
            source="telegram",
        )
        _, after = await settle(
            repo,
            f"Yaxshi. [[BOOK:{taken:%Y-%m-%dT%H:%M}]]",
            user_id=seed.a.user.id,
            conversation_id=seed.a.conversation.id,
            source="telegram",
        )

    assert after is not None
    assert after.id == mine.id
    assert after.scheduled_at == theirs.astimezone(UTC)
