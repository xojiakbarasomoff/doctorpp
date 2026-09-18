"""What the model is told about the patient, assembled from the records.

The old version of this file read the transcript and guessed. What it was
guessing at now comes from users.name, users.phone, conversation_states and
the appointments table, so what is tested here is the assembly and the
wording of the section -- the guessing is gone, along with the failures it
caused.
"""

import uuid
from datetime import UTC, date, datetime, time

from app.models.appointment import Appointment, AppointmentStatus
from app.models.conversation_state import CompletedAction, ConversationState, FlowStatus
from app.services.booking_state import Booked, BookingState, read, render
from app.services.patient_profile import Profile


def _state(**over: object) -> ConversationState:
    fields: dict[str, object] = {
        "status": str(FlowStatus.IDLE),
        "intent": "none",
        "last_completed_action": str(CompletedAction.NONE),
        "version": 1,
    }
    fields.update(over)
    return ConversationState(**fields)  # type: ignore[arg-type]


def _appointment(when: datetime) -> Appointment:
    return Appointment(
        id=uuid.uuid4(),
        scheduled_at=when,
        status=AppointmentStatus.SCHEDULED,
        patient_name="Asadbek Risqiyev",
        patient_phone="+998939511111",
    )


# --- what is known ------------------------------------------------------


def test_the_patients_record_is_what_is_known_about_them() -> None:
    state = read(
        profile=Profile(name="Asadbek Risqiyev", phone="+998939511111"),
        state=_state(status=str(FlowStatus.COLLECTING)),
        booking_in_progress=True,
    )

    assert state.name == "Asadbek Risqiyev"
    assert state.phone == "+998939511111"
    # Name and number in hand: the next thing wanted is why they are coming.
    assert state.next_needed == "reason"


def test_nothing_is_wanted_outside_a_booking() -> None:
    """A patient asking about the doctor is not halfway through a form."""
    state = read(profile=Profile(name=None, phone=None), state=_state(), booking_in_progress=False)

    assert state.next_needed is None
    assert "The one thing still missing" not in render(state)


def test_the_day_they_chose_is_stated_as_settled() -> None:
    state = read(
        profile=Profile(name="Asadbek", phone="+998939511111"),
        state=_state(
            status=str(FlowStatus.AWAITING_TIME),
            reason="buyrak og'rig'i",
            requested_date=date(2026, 9, 24),
        ),
        booking_in_progress=True,
    )

    section = render(state)

    assert "24.09.2026" in section
    assert "This is settled" in section
    assert state.next_needed == "time"


def test_what_is_known_is_never_asked_for_again() -> None:
    section = render(
        read(
            profile=Profile(name="Asadbek Risqiyev", phone="+998939511111"),
            state=_state(status=str(FlowStatus.COLLECTING), reason="uzi"),
            booking_in_progress=True,
        )
    )

    assert "Never ask for one of them again" in section
    assert "Their name: Asadbek Risqiyev" in section


# --- what is booked -----------------------------------------------------


def test_every_appointment_the_patient_holds_is_listed() -> None:
    first = _appointment(datetime(2026, 9, 24, 7, 0, tzinfo=UTC))
    second = _appointment(datetime(2026, 10, 1, 6, 40, tzinfo=UTC))

    section = render(
        read(profile=Profile(None, None), state=_state(), appointments=[first, second])
    )

    assert "24.09.2026 12:00" in section
    assert "01.10.2026 11:40" in section
    assert "authoritative" in section


def test_a_finished_booking_says_so_instead_of_asking_for_a_name() -> None:
    """"Rahmat" after a booking was answered "ism-familiyangizni yozing"."""
    section = render(
        read(
            profile=Profile(name="Asadbek", phone="+998939511111"),
            state=_state(last_completed_action=str(CompletedAction.BOOKING_CREATED)),
            booking_in_progress=False,
        )
    )

    assert "THE BOOKING IS DONE" in section
    assert "The one thing still missing" not in section


def test_a_cancelled_booking_is_not_rebooked_unasked() -> None:
    section = render(
        read(
            profile=Profile(name="Asadbek", phone=None),
            state=_state(last_completed_action=str(CompletedAction.BOOKING_CANCELLED)),
            booking_in_progress=False,
        )
    )

    assert "CANCELLED" in section


def test_the_section_always_says_to_answer_the_question_first() -> None:
    section = render(
        BookingState(
            name=None,
            phone=None,
            reason=None,
            in_progress=True,
            status=FlowStatus.COLLECTING,
        )
    )

    assert "This is a list, not a script" in section


def test_a_booked_appointment_is_summarised_the_way_a_person_reads_it() -> None:
    booked = Booked.of(_appointment(datetime(2026, 9, 24, 7, 0, tzinfo=UTC)))

    assert booked.when == "24.09.2026 12:00"
    assert booked.name == "Asadbek Risqiyev"


def test_the_time_they_chose_is_carried_with_the_day() -> None:
    state = read(
        profile=Profile(name="A", phone="+998939511111"),
        state=_state(
            status=str(FlowStatus.AWAITING_CONFIRMATION),
            reason="uzi",
            requested_date=date(2026, 9, 24),
            requested_time=time(12, 0),
        ),
        booking_in_progress=True,
    )

    assert state.next_needed is None
    assert "12:00" in render(state)
    assert "You have everything" in render(state)
