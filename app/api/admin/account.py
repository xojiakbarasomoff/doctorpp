"""An operator's own account, and getting the day's list out of the browser."""

import csv
import io
from datetime import UTC, date, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.admin.deps import require_patient_access, verify_csrf_header
from app.api.auth import get_current_operator
from app.core.db import get_db_session
from app.core.passwords import MIN_PASSWORD_LENGTH, hash_password, verify_password
from app.models.operator import Operator
from app.repositories.appointment import AppointmentRepository
from app.services.appointment import CLINIC_TIMEZONE

router = APIRouter(prefix="/api/admin", tags=["Admin — Account"])


class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(min_length=MIN_PASSWORD_LENGTH)


@router.post("/password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    payload: PasswordChange,
    operator: Operator = Depends(get_current_operator),
    session: AsyncSession = Depends(get_db_session),
    _: None = Depends(verify_csrf_header),
) -> None:
    """Change your own password. A `doctor` may do this too — it is their
    account, not clinic data.

    The current password is required even though the session already proves
    who you are: it is what stops a borrowed, unlocked browser from being
    turned into permanent access.
    """
    if not verify_password(payload.current_password, operator.password_hash):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Joriy parol noto'g'ri")
    if payload.new_password == payload.current_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Yangi parol eskisidan farq qilsin"
        )

    operator.password_hash = hash_password(payload.new_password)
    await session.commit()


# Patient names and phone numbers leave the building in this file, so it is
# gated like any other patient data rather than like a report.
@router.get("/export/appointments.csv")
async def export_appointments(
    operator: Operator = Depends(require_patient_access),
    session: AsyncSession = Depends(get_db_session),
    day: date | None = Query(default=None, alias="date"),
    days: int = Query(default=1, ge=1, le=92),
) -> Response:
    """The schedule as a CSV file.

    CSV rather than the .xlsx the Telegram dashboard produced: it needs no
    dependency, opens in Excel and Google Sheets alike, and cannot carry a
    formula that runs when somebody opens it.

    Times are written in clinic-local time, because that is the only form
    anybody reading a printed day sheet can act on.
    """
    start_date = day or datetime.now(CLINIC_TIMEZONE).date()
    start = datetime.combine(start_date, datetime.min.time(), tzinfo=CLINIC_TIMEZONE)
    end = start + timedelta(days=days)

    appointments = await AppointmentRepository(session).list_between(
        start.astimezone(UTC), end.astimezone(UTC)
    )

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Sana", "Vaqt", "Bemor", "Telefon", "Shifokor", "Holat", "Manba", "Izoh"])
    for appointment in appointments:
        local = appointment.scheduled_at.astimezone(CLINIC_TIMEZONE)
        writer.writerow(
            [
                local.strftime("%d.%m.%Y"),
                local.strftime("%H:%M"),
                appointment.patient_name or "",
                appointment.patient_phone or "",
                appointment.doctor_name,
                appointment.status,
                appointment.source,
                appointment.notes or "",
            ]
        )

    filename = f"qabullar-{start_date.isoformat()}.csv"
    return Response(
        # BOM first: Excel on Windows reads a CSV as the system codepage
        # without it, and every Uzbek name with an oʻ or gʻ in it comes out
        # mangled.
        content="﻿" + buffer.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
