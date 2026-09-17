"""The "Hisobot" screen: how many patients, what they come with, what came of it.

One endpoint returns the whole report for a period, with the same figures for
the period before it so each headline number can say whether it went up or
down. Computed on request from the rows that already exist -- messages,
conversations, appointments, photos -- so there is no second copy of the
truth to drift out of step, and a period can be any length.

Every time is bucketed in the clinic's own time zone. A patient who writes at
00:30 Tashkent time wrote on that day, not on the previous one in UTC.
"""

import statistics
import uuid
from collections import Counter, defaultdict
from datetime import UTC, date, datetime, time, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.admin.deps import require_patient_access
from app.core.db import get_db_session
from app.core.tenant_context import get_current_tenant
from app.models.appointment import Appointment, AppointmentStatus
from app.models.conversation import Conversation
from app.models.message import Message, MessageSender
from app.models.operator import Operator
from app.models.patient_media import PatientMedia
from app.models.user import User
from app.services.appointment import CLINIC_TIMEZONE
from app.services.complaints import CATEGORIES, UNKNOWN, categorise, is_enquiry, patient_words

router = APIRouter(prefix="/api/admin/reports", tags=["Admin — Reports"])

MAX_DAYS = 366
# A reply more than a day after the question is somebody coming back to an
# old chat, not a response time; counting it would make one weekend the
# whole average.
RESPONSE_CAP = timedelta(hours=24)


class Kpi(BaseModel):
    value: float
    previous: float


class DayPoint(BaseModel):
    day: date
    patients: int
    appointments: int


class Bucket(BaseModel):
    key: str
    label: str
    count: int


class ComplaintExample(BaseModel):
    at: datetime
    patient: str
    text: str
    categories: list[str]


class Report(BaseModel):
    start: date
    end: date
    days: int
    active_patients: Kpi
    new_patients: Kpi
    patient_messages: Kpi
    appointments_created: Kpi
    conversion_percent: Kpi
    photo_patients: Kpi
    median_response_seconds: Kpi | None
    human_share_percent: Kpi
    daily: list[DayPoint]
    by_hour: list[int]
    by_weekday: list[int]
    complaints: list[Bucket]
    complaint_patients: int
    enquiry_patients: int
    appointment_status: list[Bucket]
    appointment_source: list[Bucket]
    recent_complaints: list[ComplaintExample]


_STATUS_LABELS = {
    AppointmentStatus.SCHEDULED: "Kutilmoqda",
    AppointmentStatus.CONFIRMED: "Tasdiqlangan",
    AppointmentStatus.COMPLETED: "Keldi",
    AppointmentStatus.NO_SHOW: "Kelmadi",
    AppointmentStatus.CANCELLED: "Bekor qilindi",
}
_SOURCE_LABELS = {
    "bot": "Instagram bot",
    "instagram": "Instagram bot",
    "operator": "Dashboard",
    "webapp": "Web-ilova",
    "telegram": "Telegram",
}


def _bounds(start: date, end: date) -> tuple[datetime, datetime]:
    return (
        datetime.combine(start, time.min, tzinfo=CLINIC_TIMEZONE).astimezone(UTC),
        datetime.combine(end + timedelta(days=1), time.min, tzinfo=CLINIC_TIMEZONE).astimezone(UTC),
    )


def _local(moment: datetime) -> datetime:
    return moment.astimezone(CLINIC_TIMEZONE)


def _percent(part: int, whole: int) -> float:
    return round(100 * part / whole, 1) if whole else 0.0


async def _figures(session: AsyncSession, tenant_id: uuid.UUID, start: date, end: date) -> dict:
    """Everything the report counts, for one period."""
    since, until = _bounds(start, end)

    messages = (
        await session.execute(
            select(Message.conversation_id, Message.sender, Message.content, Message.created_at)
            .join(Conversation, Conversation.id == Message.conversation_id)
            .where(
                Conversation.tenant_id == tenant_id,
                Message.created_at >= since,
                Message.created_at < until,
            )
            .order_by(Message.conversation_id, Message.created_at)
        )
    ).all()
    conversation_user = dict(
        (
            await session.execute(
                select(Conversation.id, Conversation.user_id).where(
                    Conversation.tenant_id == tenant_id,
                    Conversation.id.in_({m.conversation_id for m in messages} or {uuid.uuid4()}),
                )
            )
        ).all()
    )

    patient_texts: dict[uuid.UUID, list[str]] = defaultdict(list)
    first_seen: dict[uuid.UUID, datetime] = {}
    by_day_patients: dict[date, set[uuid.UUID]] = defaultdict(set)
    by_hour = [0] * 24
    by_weekday = [0] * 7
    response_seconds: list[float] = []
    bot_replies = human_replies = patient_messages = 0
    waiting_since: datetime | None = None
    current_conversation = None

    for m in messages:
        user_id = conversation_user.get(m.conversation_id)
        if m.conversation_id != current_conversation:
            current_conversation, waiting_since = m.conversation_id, None
        if m.sender == MessageSender.PATIENT:
            patient_messages += 1
            if user_id is not None:
                patient_texts[user_id].append(m.content)
                first_seen.setdefault(user_id, m.created_at)
                by_day_patients[_local(m.created_at).date()].add(user_id)
            local = _local(m.created_at)
            by_hour[local.hour] += 1
            by_weekday[local.weekday()] += 1
            waiting_since = waiting_since or m.created_at
        else:
            if m.sender == MessageSender.BOT:
                bot_replies += 1
            else:
                human_replies += 1
            if waiting_since is not None:
                delay = m.created_at - waiting_since
                if delay <= RESPONSE_CAP:
                    response_seconds.append(delay.total_seconds())
                waiting_since = None

    active = set(patient_texts)
    earliest = dict(
        (
            await session.execute(
                select(Conversation.user_id, func.min(Message.created_at))
                .join(Message, Message.conversation_id == Conversation.id)
                .where(
                    Conversation.tenant_id == tenant_id,
                    Conversation.user_id.in_(active or {uuid.uuid4()}),
                    Message.sender == MessageSender.PATIENT,
                )
                .group_by(Conversation.user_id)
            )
        ).all()
    )
    new_patients = {u for u in active if earliest.get(u) and earliest[u] >= since}

    appointments = (
        (
            await session.execute(
                select(Appointment).where(
                    Appointment.tenant_id == tenant_id,
                    Appointment.created_at >= since,
                    Appointment.created_at < until,
                )
            )
        )
        .scalars()
        .all()
    )
    appointment_texts: dict[uuid.UUID, list[str]] = defaultdict(list)
    for appointment in appointments:
        words = patient_words(appointment.notes)
        if appointment.user_id and words:
            appointment_texts[appointment.user_id].append(words)
    booked_patients = {a.user_id for a in appointments if a.user_id}
    by_day_appointments = Counter(_local(a.created_at).date() for a in appointments)

    photo_patients = (
        await session.execute(
            select(func.count(func.distinct(PatientMedia.user_id))).where(
                PatientMedia.tenant_id == tenant_id,
                PatientMedia.created_at >= since,
                PatientMedia.created_at < until,
            )
        )
    ).scalar_one()

    complaint_counts: Counter[str] = Counter()
    complaint_patients = enquiry_patients = 0
    user_categories: dict[uuid.UUID, list[str]] = {}
    for user_id in active | set(appointment_texts):
        texts = patient_texts.get(user_id, []) + appointment_texts.get(user_id, [])
        found = categorise(texts)
        if found:
            complaint_patients += 1
            complaint_counts.update(category.key for category in found)
            user_categories[user_id] = [category.label for category in found]
        else:
            complaint_counts[UNKNOWN.key] += 1
        if is_enquiry(texts):
            enquiry_patients += 1

    replies = bot_replies + human_replies
    return {
        "since": since,
        "until": until,
        "active": active,
        "new": new_patients,
        "patient_messages": patient_messages,
        "appointments": appointments,
        "conversion": _percent(len(booked_patients & active), len(active)),
        "photo_patients": photo_patients,
        "median_response": statistics.median(response_seconds) if response_seconds else None,
        "human_share": _percent(human_replies, replies),
        "by_day_patients": by_day_patients,
        "by_day_appointments": by_day_appointments,
        "by_hour": by_hour,
        "by_weekday": by_weekday,
        "complaint_counts": complaint_counts,
        "complaint_patients": complaint_patients,
        "enquiry_patients": enquiry_patients,
        "user_categories": user_categories,
        "appointment_texts": appointment_texts,
    }


@router.get("", response_model=Report)
async def report(
    days: int = Query(default=30, ge=1, le=MAX_DAYS),
    end: date | None = Query(default=None),
    operator: Operator = Depends(require_patient_access),
    session: AsyncSession = Depends(get_db_session),
) -> Report:
    tenant_id = get_current_tenant()
    last = end or datetime.now(CLINIC_TIMEZONE).date()
    first = last - timedelta(days=days - 1)
    if first < date(2000, 1, 1):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Noto'g'ri davr")

    now = await _figures(session, tenant_id, first, last)
    before = await _figures(
        session, tenant_id, first - timedelta(days=days), first - timedelta(days=1)
    )

    labels = {category.key: category.label for category in (*CATEGORIES, UNKNOWN)}
    complaints = [
        Bucket(key=key, label=labels[key], count=count)
        for key, count in now["complaint_counts"].most_common()
    ]
    # Unknown last whatever its size: it is the remainder, not a finding.
    complaints.sort(key=lambda b: (b.key == UNKNOWN.key, -b.count))

    status_counts = Counter(a.status for a in now["appointments"])
    source_counts = Counter(a.source for a in now["appointments"])

    names: dict[uuid.UUID, str] = {}
    wanted = list(now["user_categories"])[:200]
    if wanted:
        for user in (await session.execute(select(User).where(User.id.in_(wanted)))).scalars():
            names[user.id] = user.name or (f"@{user.username}" if user.username else "Bemor")
    recent = sorted(
        (
            ComplaintExample(
                at=a.created_at,
                patient=a.patient_name or names.get(a.user_id, "Bemor"),
                text=patient_words(a.notes),
                categories=[c.label for c in categorise([patient_words(a.notes)])],
            )
            for a in now["appointments"]
            if patient_words(a.notes)
        ),
        key=lambda e: e.at,
        reverse=True,
    )[:12]

    def kpi(key: str, value_of=lambda v: v) -> Kpi:
        return Kpi(value=value_of(now[key]), previous=value_of(before[key]))

    return Report(
        start=first,
        end=last,
        days=days,
        active_patients=kpi("active", len),
        new_patients=kpi("new", len),
        patient_messages=kpi("patient_messages"),
        appointments_created=kpi("appointments", len),
        conversion_percent=kpi("conversion"),
        photo_patients=kpi("photo_patients"),
        median_response_seconds=(
            Kpi(value=now["median_response"], previous=before["median_response"] or 0)
            if now["median_response"] is not None
            else None
        ),
        human_share_percent=kpi("human_share"),
        daily=[
            DayPoint(
                day=first + timedelta(days=i),
                patients=len(now["by_day_patients"].get(first + timedelta(days=i), ())),
                appointments=now["by_day_appointments"].get(first + timedelta(days=i), 0),
            )
            for i in range(days)
        ],
        by_hour=now["by_hour"],
        by_weekday=now["by_weekday"],
        complaints=complaints,
        complaint_patients=now["complaint_patients"],
        enquiry_patients=now["enquiry_patients"],
        appointment_status=[
            Bucket(key=str(s), label=_STATUS_LABELS.get(s, str(s)), count=c)
            for s, c in status_counts.most_common()
        ],
        appointment_source=[
            Bucket(key=str(s), label=_SOURCE_LABELS.get(s, str(s)), count=c)
            for s, c in source_counts.most_common()
        ],
        recent_complaints=recent,
    )
