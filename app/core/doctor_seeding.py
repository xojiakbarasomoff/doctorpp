"""Loading a clinic's clinicians into the doctors table from inside the app.

The same reason app.core.faq_seeding exists: on a managed host the database
is reachable only from inside the cluster's private network, so the only
process that can write these rows is the application itself, driven by
configuration.

The roster is not cosmetic. app.services.answer reads it to tell a patient
who works here, and app.services.appointment reads it to decide how many
bookings one slot holds -- with an empty table the assistant cannot answer
"qaysi shifokor qabul qiladi?" and the appointment book offers a single
seat per time, which is wrong for a clinic with six clinicians.

Set SEED_DOCTORS_FROM to a path (e.g. "data/doctors.json") to arm it; leave
it unset and startup skips this entirely. Like PROVISION_TENANT_NAME it is
an instruction to seed, not a description of the running system.

Re-running is safe and cheap: a clinician is matched on their name, and a
match updates the specialty and hours in place rather than inserting a
second row. Nobody is ever deactivated or deleted from here -- a doctor who
has left is a decision for the dashboard, not a side effect of a file no
longer mentioning them, because the row carries appointments.
"""

import asyncio
import json
import logging
import uuid
from pathlib import Path

from pydantic import BaseModel, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.db import db_session
from app.core.faq_seeding import FaqSeedingError, resolve_faq_tenant_id
from app.models.doctor import Doctor

logger = logging.getLogger(__name__)

# Six rows and no embedding call, so this is a database round trip per
# clinician and nothing else -- far below faq_seeding's budget, but bounded
# for the same reason: it sits in front of the port opening.
_STARTUP_TIMEOUT_SECONDS = 15.0


class DoctorImport(BaseModel):
    """One clinician. Hours are free text, matching the column."""

    name: str
    specialty: str
    working_hours: str = "09:00 - 18:00"

    @field_validator("name", "specialty", "working_hours")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


def load_doctors(path: Path) -> list[DoctorImport]:
    """Read and validate the whole file before writing any of it."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FaqSeedingError(f"No such file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FaqSeedingError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(raw, list):
        raise FaqSeedingError(
            f"{path} must contain a JSON array of doctor objects, got {type(raw).__name__}."
        )

    try:
        return [DoctorImport.model_validate(item) for item in raw]
    except Exception as exc:  # noqa: BLE001 - pydantic's own message is the useful part
        raise FaqSeedingError(f"Invalid doctor entry in {path}: {exc}") from exc


async def seed_doctors(
    session: AsyncSession, doctors: list[DoctorImport], ig_account_id: str | None = None
) -> tuple[uuid.UUID, int, int]:
    """Write `doctors` for the resolved tenant. Returns the tenant, how many
    were created and how many already existed and were refreshed.
    """
    tenant_id = await resolve_faq_tenant_id(session, ig_account_id)
    existing = {
        row.name: row
        for row in (
            await session.execute(select(Doctor).where(Doctor.tenant_id == tenant_id))
        ).scalars()
    }

    created = updated = 0
    for doctor in doctors:
        row = existing.get(doctor.name)
        if row is None:
            session.add(
                Doctor(
                    tenant_id=tenant_id,
                    name=doctor.name,
                    specialty=doctor.specialty,
                    working_hours=doctor.working_hours,
                    is_active=True,
                )
            )
            created += 1
        else:
            row.specialty = doctor.specialty
            row.working_hours = doctor.working_hours
            updated += 1
    await session.commit()
    return tenant_id, created, updated


async def seed_doctors_if_configured(settings: Settings) -> None:
    """Load SEED_DOCTORS_FROM into the doctors table when it is set.

    Never fatal, for the same reason faq_seeding is not: continuing to accept
    Meta's deliveries is worth more than refusing to start over a roster
    somebody can load afterwards.
    """
    configured = settings.seed_doctors_from
    if configured is None:
        return

    path = Path(configured)

    async def _run() -> None:
        doctors = load_doctors(path)
        if not doctors:
            logger.warning("doctor_seeding_skipped_empty_file path=%s", path)
            return
        async with db_session() as session:
            tenant_id, created, updated = await seed_doctors(
                session, doctors, settings.provision_ig_account_id
            )
        logger.warning(
            "doctor_seeding_complete tenant_id=%s created=%s updated=%s path=%s",
            tenant_id,
            created,
            updated,
            path,
        )

    try:
        await asyncio.wait_for(_run(), timeout=_STARTUP_TIMEOUT_SECONDS)
    except TimeoutError:
        logger.error("doctor_seeding_timed_out path=%s seconds=%s", path, _STARTUP_TIMEOUT_SECONDS)
    except FaqSeedingError as exc:
        logger.error("doctor_seeding_failed path=%s error=%s", path, exc)
    except Exception:
        logger.exception("doctor_seeding_failed path=%s", path)
