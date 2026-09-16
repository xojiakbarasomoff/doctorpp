"""Booking from the Telegram Mini App.

Every request proves it came from Telegram before anything is written. The
endpoint this replaces took a `tenant_id` in the request body and verified
nothing at all, so anyone who found the URL could create appointments in any
clinic's calendar, and look up a patient by their Telegram id across tenant
boundaries.

Telegram signs the `initData` string it hands the Mini App with a key derived
from the bot's own token. Checking that signature is what turns "somebody
posted JSON at us" into "this Telegram user, in this clinic's bot, right
now".
"""

import hashlib
import hmac
import json
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any
from urllib.parse import parse_qsl

from fastapi import APIRouter, Depends, HTTPException, Path, status
from pydantic import BaseModel, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.channels.base import ChannelType
from app.core.db import get_db_session
from app.core.encryption import decrypt
from app.core.tenant_context import reset_current_tenant, set_current_tenant
from app.models.channel import Channel
from app.repositories.appointment import AppointmentRepository
from app.repositories.doctor import DoctorRepository
from app.repositories.user import UserRepository
from app.services.appointment import (
    UNASSIGNED_DOCTOR_NAME,
    MissingPatientIdentityError,
    OutsideWorkingHoursError,
    SlotAlreadyBookedError,
    create_appointment,
)
from app.services.tenant_resolution import resolve_channel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/webapp", tags=["Telegram Mini App"])

# How old a signed initData may be. Telegram stamps auth_date when the Mini
# App is opened; without a bound, a captured string would let somebody book
# forever. Generous enough that a patient can fill the form unhurried.
MAX_AUTH_AGE_SECONDS = 24 * 60 * 60

# The constant Telegram specifies for deriving the signing key.
_SECRET_SALT = b"WebAppData"


class BookingRequest(BaseModel):
    """What the Mini App posts.

    Note what is *not* here: no tenant, and no patient identity. Both come
    from the signed initData instead, because anything the client sends is
    something the client can change.
    """

    init_data: str
    scheduled_at: datetime
    patient_name: str
    patient_phone: str
    doctor_id: uuid.UUID | None = None
    notes: str | None = None

    @field_validator("patient_name", "patient_phone")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be blank")
        return stripped


class BookingResponse(BaseModel):
    appointment_id: uuid.UUID
    scheduled_at: datetime
    doctor_name: str


def verify_init_data(init_data: str, bot_token: str, *, now: float | None = None) -> dict[str, Any]:
    """Check Telegram's signature over initData and return its fields.

    The algorithm is Telegram's: every field except `hash`, sorted by key and
    joined as `key=value` with newlines, HMAC-SHA256'd under a key that is
    itself HMAC-SHA256 of the bot token under the constant "WebAppData".
    Only somebody holding the bot token can produce that, which is what makes
    it proof of origin.

    Raises HTTPException on anything that fails, deliberately with the same
    message: a caller learning *why* their forgery was rejected learns how to
    improve it.
    """
    invalid = HTTPException(
        status_code=status.HTTP_403_FORBIDDEN, detail="Telegram ma'lumotlari tasdiqlanmadi"
    )

    fields = dict(parse_qsl(init_data, keep_blank_values=True))
    provided_hash = fields.pop("hash", None)
    if not provided_hash:
        raise invalid

    data_check_string = "\n".join(f"{key}={fields[key]}" for key in sorted(fields))
    secret_key = hmac.new(_SECRET_SALT, bot_token.encode("utf-8"), hashlib.sha256).digest()
    expected = hmac.new(secret_key, data_check_string.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, provided_hash):
        raise invalid

    # A valid signature over a stale payload is still a replay.
    try:
        auth_date = int(fields.get("auth_date", ""))
    except ValueError:
        raise invalid from None
    if (now or time.time()) - auth_date > MAX_AUTH_AGE_SECONDS:
        raise invalid

    return fields


def telegram_user_id(fields: dict[str, Any]) -> str:
    """The patient's Telegram id, taken from the signed payload.

    Taken from here rather than from the request body for the whole reason
    the signature exists: the body is whatever the client typed, and a
    booking attributed to a user id the client chose is a booking attributed
    to anybody they like.
    """
    raw_user = fields.get("user")
    if not raw_user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Telegram ma'lumotlari tasdiqlanmadi"
        )
    try:
        user = json.loads(raw_user)
        return str(user["id"])
    except (ValueError, KeyError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Telegram ma'lumotlari tasdiqlanmadi"
        ) from None


@router.post("/{bot_id}/book", response_model=BookingResponse)
async def book(
    payload: BookingRequest,
    bot_id: Annotated[str, Path(description="The bot's own Telegram id — Channel.external_id")],
    session: AsyncSession = Depends(get_db_session),
) -> BookingResponse:
    resolved = await resolve_channel(session, channel_type=ChannelType.TELEGRAM, external_id=bot_id)
    if resolved is None:
        logger.warning("webapp_unknown_bot", extra={"bot_id": bot_id})
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Telegram ma'lumotlari tasdiqlanmadi"
        )

    channel = await session.get(Channel, resolved.channel_id)
    assert channel is not None  # resolve_channel just found it
    fields = verify_init_data(payload.init_data, decrypt(channel.credentials))
    external_id = telegram_user_id(fields)

    token = set_current_tenant(resolved.tenant_id)
    try:
        doctor_name = UNASSIGNED_DOCTOR_NAME
        if payload.doctor_id is not None:
            doctor = await DoctorRepository(session).get(payload.doctor_id)
            if doctor is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="Shifokor topilmadi"
                )
            doctor_name = doctor.name

        # Scoped to this channel, so the same Telegram id under another
        # clinic's bot is a different patient — the endpoint this replaces
        # looked users up by external_id alone, across every tenant.
        user = await UserRepository(session).get_by_external_id(
            channel_id=resolved.channel_id, external_id=external_id
        )

        try:
            appointment = await create_appointment(
                AppointmentRepository(session),
                scheduled_at=payload.scheduled_at,
                source="webapp",
                user_id=user.id if user is not None else None,
                patient_name=payload.patient_name,
                patient_phone=payload.patient_phone,
                doctor_id=payload.doctor_id,
                doctor_name=doctor_name,
                notes=payload.notes,
            )
        except SlotAlreadyBookedError:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail="Bu vaqt allaqachon band"
            ) from None
        except OutsideWorkingHoursError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Bu vaqt ish vaqtidan tashqarida"
            ) from None
        except MissingPatientIdentityError:  # pragma: no cover - patient_name is required
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Bemor ismi kerak"
            ) from None

        await session.commit()
        logger.info(
            "webapp_booking_created",
            extra={
                "tenant_id": str(resolved.tenant_id),
                "appointment_id": str(appointment.id),
                "scheduled_at": appointment.scheduled_at.astimezone(UTC).isoformat(),
            },
        )
        return BookingResponse(
            appointment_id=appointment.id,
            scheduled_at=appointment.scheduled_at,
            doctor_name=appointment.doctor_name,
        )
    finally:
        reset_current_tenant(token)
