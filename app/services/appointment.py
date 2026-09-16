import uuid
from collections import Counter
from collections.abc import Collection, Iterator, Sequence
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy.exc import IntegrityError

from app.models.appointment import Appointment, AppointmentStatus
from app.models.doctor import Doctor
from app.repositories.appointment import AppointmentRepository

# TODO(IGB-?): move onto Tenant.settings once the admin/dashboard panel
# exists, so each clinic can tune its own hours/slot length instead of every
# tenant sharing these — same pattern as
# core.config.Settings.debounce_window_seconds.
CLINIC_TIMEZONE = ZoneInfo("Asia/Tashkent")
SLOT_MINUTES = 30
WORK_START = time(9, 0)
WORK_END = time(19, 0)
# Used when a booking names no doctor at all. The clinic's real clinicians
# now live in the doctors table (app.models.doctor), so this is no longer a
# stand-in for "multi-doctor support does not exist" — it is the fallback for
# a booking taken before anyone was assigned, which an operator resolves
# later from the dashboard.
UNASSIGNED_DOCTOR_NAME = "Tayinlanmagan"
DEFAULT_SEARCH_HORIZON_DAYS = 14


class SlotAlreadyBookedError(Exception):
    """create_appointment() lost the race (or simply arrived second) for
    this exact slot. The DB's partial unique index is what actually
    enforces this — this exception is just the graceful, catchable form of
    the IntegrityError it raises.
    """

    def __init__(self, scheduled_at: datetime) -> None:
        self.scheduled_at = scheduled_at
        super().__init__(f"slot already booked: {scheduled_at.isoformat()}")


class OutsideWorkingHoursError(Exception):
    """scheduled_at falls outside working hours, or isn't aligned to the slot grid."""

    def __init__(self, scheduled_at: datetime) -> None:
        self.scheduled_at = scheduled_at
        super().__init__(f"outside working hours: {scheduled_at.isoformat()}")


class NoAvailabilityError(Exception):
    """No free slot was found anywhere in the search horizon."""

    def __init__(self, from_datetime: datetime, horizon_days: int) -> None:
        self.from_datetime = from_datetime
        self.horizon_days = horizon_days
        super().__init__(
            f"no free slot found within {horizon_days} days of {from_datetime.isoformat()}"
        )


class MissingPatientIdentityError(Exception):
    """create_appointment() was given neither user_id nor patient_name — a
    booking must be identifiable as *someone's* appointment.
    """

    def __init__(self) -> None:
        super().__init__("create_appointment requires at least one of user_id or patient_name")


def _to_local(scheduled_at: datetime) -> datetime:
    return scheduled_at.astimezone(CLINIC_TIMEZONE)


def is_within_working_hours(scheduled_at: datetime) -> bool:
    """Whether scheduled_at (any tz) lands on a bookable slot: inside
    [WORK_START, WORK_END) local clinic time, and aligned to the
    SLOT_MINUTES grid measured from WORK_START.

    A time that's merely *inside* working hours but off-grid (e.g. 9:17) is
    not a valid slot — the bot/dashboard only ever offer grid-aligned times,
    so an off-grid scheduled_at reaching here means the caller built it
    wrong.
    """
    local = _to_local(scheduled_at)
    if not (WORK_START <= local.time() < WORK_END):
        return False
    if local.second or local.microsecond:
        return False
    minutes_since_open = (local.hour * 60 + local.minute) - (
        WORK_START.hour * 60 + WORK_START.minute
    )
    return minutes_since_open % SLOT_MINUTES == 0


def day_slots(local_date: date) -> Iterator[datetime]:
    """Every bookable local slot start on local_date, tz-aware in CLINIC_TIMEZONE.

    Public because app.services.booking walks the same grid to list what is
    free for the assistant to offer — one definition of "a slot", so the
    times a patient is offered cannot drift from the times that can be
    booked.
    """
    current = datetime.combine(local_date, WORK_START, tzinfo=CLINIC_TIMEZONE)
    end = datetime.combine(local_date, WORK_END, tzinfo=CLINIC_TIMEZONE)
    step = timedelta(minutes=SLOT_MINUTES)
    while current < end:
        yield current
        current += step


def _first_slot_on_or_after(local_dt: datetime) -> datetime:
    """The first grid-aligned local slot start that is >= local_dt — rolls
    to the next day's opening slot if local_dt is after today's last slot.
    """
    for slot in day_slots(local_dt.date()):
        if slot >= local_dt:
            return slot
    return next(iter(day_slots(local_dt.date() + timedelta(days=1))))


def slot_capacity(doctor_count: int) -> int:
    """How many bookings one time slot can hold.

    One per active doctor -- and one when the clinic has listed none at all.
    Without that floor, a practice that has not filled in its staff would be
    able to book nobody, which is a worse failure than the one it replaces.
    """
    return max(doctor_count, 1)


def first_free_doctor(
    doctors: Sequence[Doctor], taken: Collection[uuid.UUID | None]
) -> Doctor | None:
    """The first doctor with nothing at this slot, or None if all are busy.

    In listed order rather than by load: a clinic that lists its senior first
    means it, and evening out a rota is a scheduling decision the front desk
    makes, not one to bury in a booking helper.
    """
    busy = set(taken)
    return next((doctor for doctor in doctors if doctor.id not in busy), None)


async def assign_doctor(
    repo: AppointmentRepository,
    doctors: Sequence[Doctor],
    scheduled_at: datetime,
) -> tuple[uuid.UUID | None, str]:
    """Which doctor takes this slot, and the name to record on the booking.

    Returns (None, UNASSIGNED_DOCTOR_NAME) when the clinic has listed no
    doctors -- the state every booking has been in so far, and the reason
    patients were reminded of an appointment with "Tayinlanmagan".
    """
    if not doctors:
        return None, UNASSIGNED_DOCTOR_NAME
    taken = [appt.doctor_id for appt in await repo.list_active_at(scheduled_at)]
    chosen = first_free_doctor(doctors, taken)
    if chosen is None:
        raise SlotAlreadyBookedError(scheduled_at)
    return chosen.id, chosen.name


async def check_availability(
    repo: AppointmentRepository, scheduled_at: datetime, *, capacity: int = 1
) -> bool:
    """Whether this exact slot could be booked right now.

    UX only — a True here is not a hold on the slot, so create_appointment()
    can still lose a race against a booking that lands between this check
    and the insert. That race is resolved by the DB's partial unique index,
    not by this function.
    """
    if not is_within_working_hours(scheduled_at):
        return False
    return len(await repo.list_active_at(scheduled_at)) < capacity


async def find_next_free_slot(
    repo: AppointmentRepository,
    from_datetime: datetime,
    *,
    horizon_days: int = DEFAULT_SEARCH_HORIZON_DAYS,
    capacity: int = 1,
) -> datetime:
    """The first free, working-hours, grid-aligned slot at or after
    from_datetime, searching up to horizon_days ahead.

    Fetches the whole window's busy slots in a single query, then walks the
    local slot grid in memory — not one query per candidate slot.
    """
    local_start = _first_slot_on_or_after(_to_local(from_datetime))
    last_day = local_start.date() + timedelta(days=horizon_days)
    # Bounded by the *end* of the last day checked (WORK_END), not
    # local_start's time-of-day N days out — otherwise the busy-set query
    # would cut off mid-afternoon on the last day and a real booking late in
    # the horizon could be missed, making an actually-taken slot look free.
    window_end_local = datetime.combine(last_day, WORK_END, tzinfo=CLINIC_TIMEZONE)

    # Counted, not collected into a set: a slot is full only once every
    # doctor in it is taken.
    booked = Counter(
        appt.scheduled_at
        for appt in await repo.list_active_between(
            local_start.astimezone(UTC), window_end_local.astimezone(UTC)
        )
    )

    for day_offset in range(horizon_days + 1):
        day = local_start.date() + timedelta(days=day_offset)
        for slot in day_slots(day):
            if slot < local_start:
                continue
            candidate_utc = slot.astimezone(UTC)
            if booked[candidate_utc] < capacity:
                return candidate_utc

    raise NoAvailabilityError(from_datetime, horizon_days)


async def create_appointment(
    repo: AppointmentRepository,
    *,
    scheduled_at: datetime,
    source: str,
    user_id: uuid.UUID | None = None,
    conversation_id: uuid.UUID | None = None,
    patient_name: str | None = None,
    doctor_name: str = UNASSIGNED_DOCTOR_NAME,
    doctor_id: uuid.UUID | None = None,
    patient_phone: str | None = None,
    notes: str | None = None,
) -> Appointment:
    """Books scheduled_at, refusing to double-book.

    user_id is optional: a bot booking always has one (there's no
    appointment without a prior IG conversation), but an operator booking a
    walk-in or phone patient who's never messaged the clinic has no User row
    to attach — patient_name is that booking's identity instead. At least
    one of the two is required, checked here rather than left as a DB-level
    surprise.

    The is_within_working_hours check up front just gives an obviously-wrong
    caller a clear error early. The actual double-booking guard is the
    partial unique index on (tenant_id, scheduled_at): a violation surfaces
    here as IntegrityError and is translated to SlotAlreadyBookedError so
    callers (bot/operator/dashboard) never see a raw DB error.
    """
    if user_id is None and not patient_name:
        raise MissingPatientIdentityError()
    if not is_within_working_hours(scheduled_at):
        raise OutsideWorkingHoursError(scheduled_at)
    try:
        # A dedicated SAVEPOINT for just this insert attempt: on
        # IntegrityError, SQLAlchemy rolls back to it automatically before
        # re-raising, undoing only this attempt. Rolling back the session
        # itself (session.rollback()) would instead unwind everything since
        # the session's own autobegin — including any earlier, already-
        # successful work in the same request/session — which is not what a
        # "this one slot was taken" error should do.
        async with repo.session.begin_nested():
            return await repo.create(
                user_id=user_id,
                scheduled_at=scheduled_at,
                status=AppointmentStatus.SCHEDULED,
                source=source,
                conversation_id=conversation_id,
                patient_name=patient_name,
                doctor_name=doctor_name,
                doctor_id=doctor_id,
                patient_phone=patient_phone,
                notes=notes,
            )
    except IntegrityError:
        raise SlotAlreadyBookedError(scheduled_at) from None


async def cancel_appointment(repo: AppointmentRepository, appointment: Appointment) -> Appointment:
    """Cancels a booking, which — via the partial unique index — immediately
    frees its slot for rebooking.
    """
    return await repo.update(appointment, status=AppointmentStatus.CANCELLED)


async def confirm_appointment(repo: AppointmentRepository, appointment: Appointment) -> Appointment:
    """Marks a booking as confirmed with the patient.

    Still holds its slot (see ACTIVE_STATUSES) — confirming records that
    somebody spoke to the patient, it does not change what the time is
    booked for.
    """
    return await repo.update(appointment, status=AppointmentStatus.CONFIRMED)
