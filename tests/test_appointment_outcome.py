"""Marking a visit "Keldi" / "Kelmadi": the rule, and the spreadsheet.

The endpoints are tested with the rest of the dashboard API, in
tests/test_admin_api.py. Without them the report's outcome chart had
nothing to count.
"""

import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch
from uuid import UUID

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.appointment import AppointmentStatus
from app.repositories.appointment import AppointmentRepository
from app.services.appointment import (
    OUTCOME_GRACE,
    CancelledAppointmentError,
    NotYetDueError,
    record_outcome,
)
from app.services.sheets import APPOINTMENT_HEADER
from tests.conftest import Seed
from tests.test_sheets import _appointment, _mirror

AsTenant = Callable[[UUID], AbstractContextManager[None]]


# --- the rule itself ----------------------------------------------------------------------


async def test_half_an_hour_early_counts_and_more_does_not(
    db_session: AsyncSession, seed: Seed, as_tenant: AsTenant
) -> None:
    now = datetime.now(UTC)
    with as_tenant(seed.tenant_a.id):
        repo = AppointmentRepository(db_session)
        appointment = seed.a.appointment
        appointment.scheduled_at = now + OUTCOME_GRACE - timedelta(minutes=1)
        await record_outcome(repo, appointment, AppointmentStatus.COMPLETED, now=now)
        assert appointment.status == AppointmentStatus.COMPLETED

        appointment.scheduled_at = now + OUTCOME_GRACE + timedelta(minutes=1)
        with pytest.raises(NotYetDueError):
            await record_outcome(repo, appointment, AppointmentStatus.NO_SHOW, now=now)


async def test_only_an_outcome_is_an_outcome(
    db_session: AsyncSession, seed: Seed, as_tenant: AsTenant
) -> None:
    with as_tenant(seed.tenant_a.id):
        repo = AppointmentRepository(db_session)
        with pytest.raises(ValueError):
            await record_outcome(repo, seed.a.appointment, AppointmentStatus.CANCELLED)
        seed.a.appointment.status = AppointmentStatus.CANCELLED
        with pytest.raises(CancelledAppointmentError):
            await record_outcome(repo, seed.a.appointment, AppointmentStatus.COMPLETED)


# --- the spreadsheet ----------------------------------------------------------------------


@pytest.mark.parametrize(("status", "label"), [("completed", "Keldi"), ("no_show", "Kelmadi")])
async def test_a_marked_outcome_is_written_into_the_status_cell(status: str, label: str) -> None:
    """Status is left alone on an ordinary rewrite -- it belongs to a person.
    "Keldi" / "Kelmadi" marked in the dashboard is that person's decision,
    so it is written."""
    written: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if request.method == "GET" and "A1%3AZ1" in url.replace(":", "%3A"):
            return httpx.Response(200, json={"values": [list(APPOINTMENT_HEADER)]})
        if request.method == "GET" and "/values/" in url:
            return httpx.Response(200, json={"values": [["Kod"], [_appointment().reference]]})
        if request.method == "GET":
            return httpx.Response(
                200, json={"sheets": [{"properties": {"sheetId": 0, "title": "Qabullar"}}]}
            )
        if "values:batchUpdate" in url:
            written.extend(json.loads(request.content)["data"])
        return httpx.Response(200, json={})

    mirror, _ = _mirror(handler)
    real_client = httpx.AsyncClient

    def fake(*args: object, **kwargs: object) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(handler))

    with patch("app.services.sheets.httpx.AsyncClient", fake):
        mirror._appointments_ready = False  # type: ignore[attr-defined]
        await mirror.upsert_appointment(_appointment(status=status))

    assert [entry["range"] for entry in written] == ["Qabullar!A2:H2"]
    assert written[0]["values"][0][APPOINTMENT_HEADER.index("Status")] == label
