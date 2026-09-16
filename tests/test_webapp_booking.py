"""Booking from the Telegram Mini App.

The endpoint this replaces verified nothing: it took a `tenant_id` in the
body and created appointments on it. Everything here is about that being
closed — the signature, what it is checked against, and what the request is
no longer allowed to assert about itself.
"""

import hashlib
import hmac
import json
import time
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractContextManager
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlencode
from uuid import UUID

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.webapp import MAX_AUTH_AGE_SECONDS
from app.channels.base import ChannelType
from app.core.db import get_db_session
from app.core.encryption import encrypt
from app.main import app
from app.models.appointment import AppointmentStatus
from app.repositories.appointment import AppointmentRepository
from app.repositories.channel import ChannelRepository
from app.repositories.user import UserRepository
from app.services.appointment import is_within_working_hours
from tests.conftest import Seed

BOT_ID = "8123456789"
BOT_TOKEN = "8123456789:AAbbCCddEEff"
PATIENT_TG_ID = 555000111


def _sign_init_data(bot_token: str, fields: dict[str, str]) -> str:
    """Build an initData string signed the way Telegram signs it."""
    data_check_string = "\n".join(f"{key}={fields[key]}" for key in sorted(fields))
    secret_key = hmac.new(b"WebAppData", bot_token.encode(), hashlib.sha256).digest()
    signature = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()
    return urlencode({**fields, "hash": signature})


def _init_data(
    *,
    bot_token: str = BOT_TOKEN,
    user_id: int = PATIENT_TG_ID,
    auth_date: int | None = None,
) -> str:
    return _sign_init_data(
        bot_token,
        {
            "auth_date": str(auth_date if auth_date is not None else int(time.time())),
            "query_id": "AAF-test",
            "user": json.dumps({"id": user_id, "first_name": "Aziza"}),
        },
    )


def _next_free_slot(offset_days: int = 1) -> datetime:
    candidate = (datetime.now(UTC) + timedelta(days=offset_days)).replace(
        minute=0, second=0, microsecond=0
    )
    for _ in range(96):
        if is_within_working_hours(candidate):
            return candidate
        candidate += timedelta(minutes=30)
    raise AssertionError("no bookable slot found")


@pytest.fixture
async def telegram_channel(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> Any:
    with as_tenant(seed.tenant_a.id):
        return await ChannelRepository(db_session).create(
            type=ChannelType.TELEGRAM,
            external_id=BOT_ID,
            credentials=encrypt(BOT_TOKEN),
            config={"webhook_secret": "s"},
        )


@pytest.fixture
async def client(db_session: AsyncSession) -> AsyncIterator[httpx.AsyncClient]:
    async def _get_db_session_override() -> AsyncSession:
        return db_session

    app.dependency_overrides[get_db_session] = _get_db_session_override
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
    app.dependency_overrides.pop(get_db_session, None)


async def _book(client: httpx.AsyncClient, **overrides: Any) -> httpx.Response:
    payload: dict[str, Any] = {
        "init_data": _init_data(),
        "scheduled_at": _next_free_slot().isoformat(),
        "patient_name": "Aziza",
        "patient_phone": "+998 90 123 45 67",
    }
    payload.update(overrides)
    bot = payload.pop("bot", BOT_ID)
    return await client.post(f"/api/webapp/{bot}/book", json=payload)


# --- the signature is the whole point ---


async def test_a_correctly_signed_request_books(
    client: httpx.AsyncClient, telegram_channel: Any, seed: Seed
) -> None:
    response = await _book(client, doctor_id=str(seed.a.doctor.id))

    assert response.status_code == 200
    assert response.json()["doctor_name"] == "Dr. Smith"


async def test_an_unsigned_request_is_refused(
    client: httpx.AsyncClient, telegram_channel: Any
) -> None:
    response = await _book(client, init_data="auth_date=1&user=%7B%22id%22%3A1%7D")

    assert response.status_code == 403


async def test_a_request_signed_with_the_wrong_token_is_refused(
    client: httpx.AsyncClient, telegram_channel: Any
) -> None:
    """A signature only proves origin if it is checked against *this* bot's
    token. Signing with another bot's is the forgery this stops.
    """
    response = await _book(client, init_data=_init_data(bot_token="9999999999:someoneelse"))

    assert response.status_code == 403


async def test_a_tampered_field_invalidates_the_signature(
    client: httpx.AsyncClient, telegram_channel: Any
) -> None:
    signed = _init_data()
    tampered = signed.replace("Aziza", "Bobur")

    response = await _book(client, init_data=tampered)

    assert response.status_code == 403


async def test_a_stale_signature_is_refused(
    client: httpx.AsyncClient, telegram_channel: Any
) -> None:
    """A valid signature over an old payload is still a replay — without a
    bound, a captured initData would book forever.
    """
    long_ago = int(time.time()) - MAX_AUTH_AGE_SECONDS - 60

    response = await _book(client, init_data=_init_data(auth_date=long_ago))

    assert response.status_code == 403


async def test_an_unknown_bot_is_refused(client: httpx.AsyncClient, telegram_channel: Any) -> None:
    response = await _book(client, bot="1111111111")

    assert response.status_code == 403


# --- what the request may no longer assert about itself ---


async def test_the_booking_is_attributed_to_the_signed_user_not_a_claimed_one(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    telegram_channel: Any,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """The patient comes from the signed payload. The old endpoint took a
    `telegram_user_id` field from the body, so a booking could be attributed
    to anybody the caller named.
    """
    with as_tenant(seed.tenant_a.id):
        patient = await UserRepository(db_session).create(
            channel_id=telegram_channel.id, external_id=str(PATIENT_TG_ID)
        )

    response = await _book(client, telegram_user_id=999999999)

    assert response.status_code == 200
    with as_tenant(seed.tenant_a.id):
        booking = await AppointmentRepository(db_session).get(
            UUID(response.json()["appointment_id"])
        )
    assert booking is not None
    assert booking.user_id == patient.id


async def test_the_booking_lands_in_the_bots_own_clinic(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    telegram_channel: Any,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """The tenant comes from the channel the bot id resolves to. There is no
    request field that could point it at another clinic.
    """
    response = await _book(client, tenant_id=str(seed.tenant_b.id))

    assert response.status_code == 200
    with as_tenant(seed.tenant_a.id):
        booking = await AppointmentRepository(db_session).get(
            UUID(response.json()["appointment_id"])
        )
    assert booking is not None
    assert booking.tenant_id == seed.tenant_a.id


async def test_the_same_telegram_id_under_another_bot_is_a_different_patient(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    telegram_channel: Any,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """The old endpoint looked users up by external_id alone, across every
    tenant, so one clinic's booking could be linked to another's patient.
    """
    with as_tenant(seed.tenant_b.id):
        other_channel = await ChannelRepository(db_session).create(
            type=ChannelType.TELEGRAM,
            external_id="7000000000",
            credentials=encrypt("7000000000:token"),
            config={"webhook_secret": "s"},
        )
        await UserRepository(db_session).create(
            channel_id=other_channel.id, external_id=str(PATIENT_TG_ID)
        )

    response = await _book(client)

    assert response.status_code == 200
    with as_tenant(seed.tenant_a.id):
        booking = await AppointmentRepository(db_session).get(
            UUID(response.json()["appointment_id"])
        )
    assert booking is not None
    # No User row exists for this id under *this* bot, so the booking stands
    # on its patient_name rather than being linked to the other clinic's.
    assert booking.user_id is None
    assert booking.patient_name == "Aziza"


# --- the booking rules are the shared ones ---


async def test_the_mini_app_cannot_double_book_a_slot(
    client: httpx.AsyncClient, telegram_channel: Any
) -> None:
    slot = _next_free_slot(2).isoformat()

    first = await _book(client, scheduled_at=slot)
    second = await _book(client, scheduled_at=slot, patient_name="Boshqa bemor")

    assert first.status_code == 200
    assert second.status_code == 409


async def test_the_mini_app_cannot_book_outside_working_hours(
    client: httpx.AsyncClient, telegram_channel: Any
) -> None:
    at_night = (datetime.now(UTC) + timedelta(days=1)).replace(
        hour=3, minute=0, second=0, microsecond=0
    )

    response = await _book(client, scheduled_at=at_night.isoformat())

    assert response.status_code == 400


async def test_a_mini_app_booking_is_marked_as_coming_from_the_webapp(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    telegram_channel: Any,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    response = await _book(client, notes="Tishim og'riyapti")

    with as_tenant(seed.tenant_a.id):
        booking = await AppointmentRepository(db_session).get(
            UUID(response.json()["appointment_id"])
        )
    assert booking is not None
    assert booking.source == "webapp"
    assert booking.status == AppointmentStatus.SCHEDULED
    assert booking.notes == "Tishim og'riyapti"
    assert booking.patient_phone == "+998 90 123 45 67"
