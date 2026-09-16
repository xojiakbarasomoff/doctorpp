"""The dashboard's JSON API.

The security tests here are the point of the port: the Telegram admin API
this replaces took `tenant_id: int = Query(1)` on every endpoint, so any
authenticated operator could read and write another clinic's records by
changing a number in the URL.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractContextManager
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import httpx
import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

import app.services.knowledge_base as knowledge_base_service
from app.api.admin.deps import CSRF_HEADER
from app.core.db import get_db_session
from app.core.encryption import encrypt
from app.core.session import SESSION_COOKIE_NAME, create_session_cookie
from app.main import app
from app.models.appointment import AppointmentStatus
from app.models.lead import LeadStatus
from app.models.message import MessageSender
from app.rag.embeddings import EMBEDDING_DIMENSIONS, EmbeddingProvider
from app.repositories.appointment import AppointmentRepository
from app.repositories.doctor import DoctorRepository
from app.repositories.lead import LeadRepository
from app.repositories.operator import OperatorRepository
from app.services.appointment import CLINIC_TIMEZONE, is_within_working_hours
from tests.conftest import Seed


class FakeEmbeddingProvider(EmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * EMBEDDING_DIMENSIONS for _ in texts]


@pytest.fixture(autouse=True)
def _no_real_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The FAQ endpoint embeds what it stores, and a row without an embedding
    is invisible to retrieval. Nothing here should reach a real provider.

    Patched on the module rather than through app.dependency_overrides:
    ingest_faqs resolves its provider by calling get_embedding_provider()
    itself, which FastAPI's dependency system never sees.
    """
    monkeypatch.setattr(knowledge_base_service, "get_embedding_provider", FakeEmbeddingProvider)


@pytest.fixture
async def manager(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> Any:
    """An `admin` on tenant A: the account that may change the clinic itself.

    The seed fixture's own operator is a `doctor` — view-only by design, and
    exercised as such below — so anything testing a write needs its own
    account rather than quietly widening the shared one.
    """
    with as_tenant(seed.tenant_a.id):
        return await OperatorRepository(db_session).create(
            name="Admin A",
            role="admin",
            username=f"manager-{seed.tenant_a.id}",
            password_hash="x",
        )


@pytest.fixture
async def front_desk(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> Any:
    """An `operator` on tenant A: the front desk.

    Separate from `manager` because the boundary between them is the point
    of having three roles at all, and a boundary only one fixture ever
    crosses is not being tested.
    """
    with as_tenant(seed.tenant_a.id):
        return await OperatorRepository(db_session).create(
            name="Front Desk A",
            role="operator",
            username=f"frontdesk-{seed.tenant_a.id}",
            password_hash="x",
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


def _login_as(client: httpx.AsyncClient, operator_id: UUID) -> str:
    """Put a real signed session cookie on the client and return its CSRF
    token — the same pair a browser gets from POST /login.
    """
    cookie_value, csrf_token = create_session_cookie(operator_id)
    client.cookies.set(SESSION_COOKIE_NAME, cookie_value)
    return csrf_token


def _next_free_slot(offset_days: int = 1) -> datetime:
    candidate = (datetime.now(UTC) + timedelta(days=offset_days)).replace(
        minute=0, second=0, microsecond=0
    )
    for _ in range(96):
        if is_within_working_hours(candidate):
            return candidate
        candidate += timedelta(minutes=30)
    raise AssertionError("no bookable slot found")


# --- authentication and tenant scoping ---


async def test_the_api_is_closed_to_anyone_without_a_session(client: httpx.AsyncClient) -> None:
    response = await client.get("/api/admin/conversations")

    # The dashboard's exception handler turns "no session" into a redirect
    # to the login page rather than a JSON 401.
    assert response.status_code in (303, 307)
    assert "/login" in response.headers.get("location", "")


async def test_the_session_endpoint_reports_the_operators_own_tenant(
    client: httpx.AsyncClient, seed: Seed
) -> None:
    _login_as(client, seed.a.operator.id)

    body = (await client.get("/api/admin/session")).json()

    assert body["tenant_id"] == str(seed.tenant_a.id)
    assert body["tenant_name"] == "Clinic A"
    assert body["csrf_token"]


async def test_an_operator_only_ever_sees_their_own_clinics_records(
    client: httpx.AsyncClient, seed: Seed
) -> None:
    """The whole reason for the port. There is no request parameter that
    could widen this — the tenant comes from the operator's own row.
    """
    _login_as(client, seed.a.operator.id)

    conversations = (await client.get("/api/admin/conversations")).json()
    leads = (await client.get("/api/admin/leads")).json()
    doctors = (await client.get("/api/admin/doctors")).json()

    assert [c["id"] for c in conversations] == [str(seed.a.conversation.id)]
    assert str(seed.b.conversation.id) not in {c["id"] for c in conversations}
    assert str(seed.b.lead.id) not in {row["id"] for row in leads}
    assert str(seed.b.doctor.id) not in {row["id"] for row in doctors}


async def test_a_tenant_id_query_parameter_changes_nothing(
    client: httpx.AsyncClient, seed: Seed
) -> None:
    """A direct regression test for the vulnerability being replaced: the
    old endpoints took ?tenant_id= and honoured it.
    """
    _login_as(client, seed.a.operator.id)

    body = (await client.get(f"/api/admin/leads?tenant_id={seed.tenant_b.id}")).json()

    assert str(seed.b.lead.id) not in {row["id"] for row in body}
    assert str(seed.a.lead.id) in {row["id"] for row in body}


async def test_another_clinics_record_is_not_reachable_by_id(
    client: httpx.AsyncClient, seed: Seed
) -> None:
    _login_as(client, seed.a.operator.id)

    response = await client.get(f"/api/admin/conversations/{seed.b.conversation.id}")

    assert response.status_code == 404


# --- CSRF ---


async def test_a_write_without_the_csrf_header_is_refused(
    client: httpx.AsyncClient, seed: Seed, manager: Any
) -> None:
    _login_as(client, manager.id)

    response = await client.post(
        "/api/admin/doctors", json={"name": "Dr. New", "specialty": "Ortodont"}
    )

    assert response.status_code == 403
    assert CSRF_HEADER in response.json()["detail"]


async def test_a_write_with_the_wrong_csrf_token_is_refused(
    client: httpx.AsyncClient, seed: Seed, manager: Any
) -> None:
    _login_as(client, manager.id)

    response = await client.post(
        "/api/admin/doctors",
        json={"name": "Dr. New"},
        headers={CSRF_HEADER: "not-the-token"},
    )

    assert response.status_code == 403


# --- roles ---


async def test_a_doctor_account_can_read_but_not_write(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    with as_tenant(seed.tenant_a.id):
        viewer = await OperatorRepository(db_session).create(
            name="Dr. Viewer",
            role="doctor",
            username=f"viewer-{seed.tenant_a.id}",
            password_hash="x",
        )
    csrf = _login_as(client, viewer.id)

    assert (await client.get("/api/admin/appointments")).status_code == 200
    write = await client.post(
        "/api/admin/doctors", json={"name": "Dr. New"}, headers={CSRF_HEADER: csrf}
    )
    assert write.status_code == 403


# --- conversations and operator takeover ---


async def test_turning_the_bot_off_and_back_on(
    client: httpx.AsyncClient, seed: Seed, manager: Any
) -> None:
    csrf = _login_as(client, manager.id)
    url = f"/api/admin/conversations/{seed.a.conversation.id}/bot"

    off = await client.post(url, json={"is_bot_enabled": False}, headers={CSRF_HEADER: csrf})
    assert off.status_code == 200
    assert off.json()["is_bot_enabled"] is False

    on = await client.post(url, json={"is_bot_enabled": True}, headers={CSRF_HEADER: csrf})
    assert on.json()["is_bot_enabled"] is True


async def test_the_transcript_comes_back_oldest_first(
    client: httpx.AsyncClient, seed: Seed
) -> None:
    _login_as(client, seed.a.operator.id)

    body = (await client.get(f"/api/admin/conversations/{seed.a.conversation.id}")).json()

    assert body["conversation"]["id"] == str(seed.a.conversation.id)
    assert [m["content"] for m in body["messages"]] == ["Hello"]
    assert body["messages"][0]["sender"] == MessageSender.PATIENT


async def test_an_operator_reply_that_cannot_be_delivered_is_not_recorded(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    manager: Any,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """A message in the transcript the patient never received would have the
    operator believe they had answered.

    The channel is put back to an unconfigured credential first, which is
    what makes delivery refuse without any network call — the seeded value
    is a real-looking token, and the adapter would take it to Meta.
    """
    with as_tenant(seed.tenant_a.id):
        seed.a.channel.credentials = encrypt("pending")
        await db_session.flush()

    csrf = _login_as(client, manager.id)

    response = await client.post(
        f"/api/admin/conversations/{seed.a.conversation.id}/reply",
        json={"text": "Ertaga 10:00 ga yozdim"},
        headers={CSRF_HEADER: csrf},
    )

    assert response.status_code == 502
    detail = (await client.get(f"/api/admin/conversations/{seed.a.conversation.id}")).json()
    assert [m["content"] for m in detail["messages"]] == ["Hello"]


# --- doctors ---


async def test_creating_and_editing_a_doctor(
    client: httpx.AsyncClient, seed: Seed, manager: Any
) -> None:
    csrf = _login_as(client, manager.id)

    created = await client.post(
        "/api/admin/doctors",
        json={"name": "Dr. Anvar", "specialty": "Implantolog", "phone": "+998 90 111 22 33"},
        headers={CSRF_HEADER: csrf},
    )
    assert created.status_code == 201
    doctor_id = created.json()["id"]
    # Defaults come from the schema, not from the caller.
    assert created.json()["working_hours"] == "09:00 - 18:00"

    edited = await client.patch(
        f"/api/admin/doctors/{doctor_id}",
        json={"phone": "+998 90 999 88 77"},
        headers={CSRF_HEADER: csrf},
    )
    assert edited.json()["phone"] == "+998 90 999 88 77"
    # A partial update must not blank what it did not mention.
    assert edited.json()["specialty"] == "Implantolog"


async def test_deactivating_a_doctor_removes_them_from_the_default_list(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    manager: Any,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    csrf = _login_as(client, manager.id)

    await client.patch(
        f"/api/admin/doctors/{seed.a.doctor.id}",
        json={"is_active": False},
        headers={CSRF_HEADER: csrf},
    )

    active = (await client.get("/api/admin/doctors")).json()
    everyone = (await client.get("/api/admin/doctors?include_inactive=true")).json()

    assert str(seed.a.doctor.id) not in {row["id"] for row in active}
    assert str(seed.a.doctor.id) in {row["id"] for row in everyone}
    del db_session, as_tenant


async def test_editing_another_clinics_doctor_is_a_404(
    client: httpx.AsyncClient, seed: Seed, manager: Any
) -> None:
    csrf = _login_as(client, manager.id)

    response = await client.patch(
        f"/api/admin/doctors/{seed.b.doctor.id}",
        json={"name": "hijacked"},
        headers={CSRF_HEADER: csrf},
    )

    assert response.status_code == 404


# --- leads ---


async def test_working_a_lead_through_its_statuses(
    client: httpx.AsyncClient, seed: Seed, manager: Any
) -> None:
    csrf = _login_as(client, manager.id)

    created = await client.post(
        "/api/admin/leads",
        json={"patient_name": "Aziza", "phone": "+998 90 123 45 67", "topic": "implant"},
        headers={CSRF_HEADER: csrf},
    )
    assert created.status_code == 201
    assert created.json()["status"] == LeadStatus.NEW

    updated = await client.patch(
        f"/api/admin/leads/{created.json()['id']}",
        json={"status": LeadStatus.CONTACTED, "notes": "Ertaga qayta qo'ng'iroq"},
        headers={CSRF_HEADER: csrf},
    )
    assert updated.json()["status"] == LeadStatus.CONTACTED

    only_contacted = (await client.get(f"/api/admin/leads?status={LeadStatus.CONTACTED}")).json()
    assert [row["id"] for row in only_contacted] == [created.json()["id"]]


async def test_an_unknown_lead_status_is_rejected(
    client: httpx.AsyncClient, seed: Seed, manager: Any
) -> None:
    csrf = _login_as(client, manager.id)

    response = await client.patch(
        f"/api/admin/leads/{seed.a.lead.id}",
        json={"status": "nonsense"},
        headers={CSRF_HEADER: csrf},
    )

    assert response.status_code == 422


# --- knowledge base ---


async def test_a_faq_added_here_is_embedded_so_retrieval_can_find_it(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    manager: Any,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """A row without an embedding shows in the dashboard and is never used
    to answer anything, which is the worst of both.
    """
    csrf = _login_as(client, manager.id)

    created = await client.post(
        "/api/admin/knowledge-base",
        json={"question": "Implant narxi qancha?", "answer": "5 000 000 so'm."},
        headers={CSRF_HEADER: csrf},
    )

    assert created.status_code == 201
    from app.repositories.knowledge_base import KnowledgeBaseRepository

    with as_tenant(seed.tenant_a.id):
        faq = await KnowledgeBaseRepository(db_session).get(UUID(created.json()["id"]))
    assert faq is not None
    assert len(faq.embedding) == EMBEDDING_DIMENSIONS


async def test_deleting_a_faq_deactivates_it_rather_than_dropping_the_row(
    client: httpx.AsyncClient, seed: Seed, manager: Any
) -> None:
    csrf = _login_as(client, manager.id)

    response = await client.delete(
        f"/api/admin/knowledge-base/{seed.a.knowledge_base.id}", headers={CSRF_HEADER: csrf}
    )

    assert response.status_code == 204
    listed = (await client.get("/api/admin/knowledge-base")).json()
    [row] = [r for r in listed if r["id"] == str(seed.a.knowledge_base.id)]
    assert row["is_active"] is False


# --- settings ---


async def test_settings_merge_rather_than_replace(
    client: httpx.AsyncClient, seed: Seed, manager: Any
) -> None:
    csrf = _login_as(client, manager.id)

    await client.patch(
        "/api/admin/settings",
        json={"clinic_address": "Amir Temur 45", "debounce_seconds": 8},
        headers={CSRF_HEADER: csrf},
    )
    # A form that only edits the address must not blank the debounce window.
    after = await client.patch(
        "/api/admin/settings",
        json={"clinic_address": "Navoiy 12"},
        headers={CSRF_HEADER: csrf},
    )

    assert after.json()["clinic_address"] == "Navoiy 12"
    assert after.json()["debounce_seconds"] == 8


async def test_an_out_of_range_debounce_window_is_rejected(
    client: httpx.AsyncClient, seed: Seed, manager: Any
) -> None:
    csrf = _login_as(client, manager.id)

    response = await client.patch(
        "/api/admin/settings", json={"debounce_seconds": 9999}, headers={CSRF_HEADER: csrf}
    )

    assert response.status_code == 422


# --- appointments and analytics ---


async def test_booking_cancelling_and_confirming_from_the_dashboard(
    client: httpx.AsyncClient, seed: Seed, manager: Any
) -> None:
    csrf = _login_as(client, manager.id)
    slot = _next_free_slot()

    created = await client.post(
        "/api/admin/appointments",
        json={
            "scheduled_at": slot.isoformat(),
            "patient_name": "Aziza",
            "patient_phone": "+998 90 123 45 67",
            "doctor_id": str(seed.a.doctor.id),
        },
        headers={CSRF_HEADER: csrf},
    )
    assert created.status_code == 201
    assert created.json()["doctor_name"] == "Dr. Smith"
    assert created.json()["source"] == "operator"
    appointment_id = created.json()["id"]

    confirmed = await client.post(
        f"/api/admin/appointments/{appointment_id}/confirm", headers={CSRF_HEADER: csrf}
    )
    assert confirmed.json()["status"] == AppointmentStatus.CONFIRMED

    cancelled = await client.post(
        f"/api/admin/appointments/{appointment_id}/cancel", headers={CSRF_HEADER: csrf}
    )
    assert cancelled.json()["status"] == AppointmentStatus.CANCELLED


async def test_double_booking_a_slot_is_refused(
    client: httpx.AsyncClient, seed: Seed, manager: Any
) -> None:
    csrf = _login_as(client, manager.id)
    slot = _next_free_slot(2)
    payload = {"scheduled_at": slot.isoformat(), "patient_name": "Aziza"}

    first = await client.post("/api/admin/appointments", json=payload, headers={CSRF_HEADER: csrf})
    second = await client.post(
        "/api/admin/appointments",
        json={**payload, "patient_name": "Boshqa bemor"},
        headers={CSRF_HEADER: csrf},
    )

    assert first.status_code == 201
    assert second.status_code == 409


async def test_booking_outside_working_hours_is_refused(
    client: httpx.AsyncClient, seed: Seed, manager: Any
) -> None:
    csrf = _login_as(client, manager.id)
    # 03:00 UTC is 08:00 in Tashkent — before the clinic opens.
    at_night = (datetime.now(UTC) + timedelta(days=1)).replace(
        hour=3, minute=0, second=0, microsecond=0
    )

    response = await client.post(
        "/api/admin/appointments",
        json={"scheduled_at": at_night.isoformat(), "patient_name": "Aziza"},
        headers={CSRF_HEADER: csrf},
    )

    assert response.status_code == 400


async def test_analytics_counts_only_this_clinic(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    with as_tenant(seed.tenant_b.id):
        # Noise in the other clinic that must not show up in A's numbers.
        await LeadRepository(db_session).create(patient_name="B lead", phone="+998 90 000 00 00")
        await DoctorRepository(db_session).create(
            name="Dr. B", specialty="Ortodont", working_hours="09:00 - 18:00"
        )
        await AppointmentRepository(db_session).create(
            doctor_name="Dr. B",
            scheduled_at=_next_free_slot(3),
            status=AppointmentStatus.SCHEDULED,
            patient_name="B bemor",
        )

    _login_as(client, seed.a.operator.id)
    body = (await client.get("/api/admin/analytics")).json()

    assert body["conversations_total"] == 1
    assert body["appointments_total"] == 1
    assert body["leads_total"] == 1
    assert body["leads_new"] == 1
    assert body["faqs_active"] == 1


# --- account: password and export ---


async def test_changing_your_own_password(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    from app.core.passwords import hash_password, verify_password

    with as_tenant(seed.tenant_a.id):
        operator = await OperatorRepository(db_session).create(
            name="Operator A",
            role="operator",
            username=f"pw-{seed.tenant_a.id}",
            password_hash=hash_password("current-password"),
        )
    csrf = _login_as(client, operator.id)

    response = await client.post(
        "/api/admin/password",
        json={"current_password": "current-password", "new_password": "a-much-longer-one"},
        headers={CSRF_HEADER: csrf},
    )

    assert response.status_code == 204
    assert verify_password("a-much-longer-one", operator.password_hash)


async def test_the_current_password_is_required_to_change_it(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """A session alone is not enough: this is what stops a borrowed, unlocked
    browser from being turned into permanent access.
    """
    from app.core.passwords import hash_password

    with as_tenant(seed.tenant_a.id):
        operator = await OperatorRepository(db_session).create(
            name="Operator A",
            role="operator",
            username=f"pw2-{seed.tenant_a.id}",
            password_hash=hash_password("current-password"),
        )
    csrf = _login_as(client, operator.id)

    response = await client.post(
        "/api/admin/password",
        json={"current_password": "wrong", "new_password": "a-much-longer-one"},
        headers={CSRF_HEADER: csrf},
    )

    assert response.status_code == 400


async def test_a_short_new_password_is_rejected(
    client: httpx.AsyncClient, seed: Seed, manager: Any
) -> None:
    csrf = _login_as(client, manager.id)

    response = await client.post(
        "/api/admin/password",
        json={"current_password": "x", "new_password": "short"},
        headers={CSRF_HEADER: csrf},
    )

    assert response.status_code == 422


async def test_the_day_exports_as_csv(client: httpx.AsyncClient, seed: Seed, manager: Any) -> None:
    csrf = _login_as(client, manager.id)
    slot = _next_free_slot()
    await client.post(
        "/api/admin/appointments",
        json={
            "scheduled_at": slot.isoformat(),
            "patient_name": "Aziza Karimova",
            "patient_phone": "+998 90 123 45 67",
        },
        headers={CSRF_HEADER: csrf},
    )

    local_day = slot.astimezone(CLINIC_TIMEZONE).date().isoformat()
    response = await client.get(f"/api/admin/export/appointments.csv?date={local_day}")

    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert "attachment" in response.headers["content-disposition"]
    # A BOM, so Excel on Windows does not mangle every Uzbek name.
    assert response.text.startswith("\ufeff")
    assert "Aziza Karimova" in response.text
    assert "+998 90 123 45 67" in response.text


async def test_the_export_covers_only_this_clinic(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    manager: Any,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    slot = _next_free_slot(2)
    with as_tenant(seed.tenant_b.id):
        await AppointmentRepository(db_session).create(
            doctor_name="Dr. B",
            scheduled_at=slot,
            status=AppointmentStatus.SCHEDULED,
            patient_name="Boshqa klinikaning bemori",
        )

    _login_as(client, manager.id)
    local_day = slot.astimezone(CLINIC_TIMEZONE).date().isoformat()
    response = await client.get(f"/api/admin/export/appointments.csv?date={local_day}")

    assert "Boshqa klinikaning bemori" not in response.text


# --- the boundary between the three roles ----------------------------------


async def test_the_front_desk_can_work_a_lead(
    client: httpx.AsyncClient, seed: Seed, front_desk: Any
) -> None:
    """The front desk's own job. If this ever fails, the role has been
    narrowed into uselessness and everyone will just be given `admin`.
    """
    csrf = _login_as(client, front_desk.id)

    response = await client.post(
        "/api/admin/leads",
        json={"patient_name": "Nodira", "phone": "+998901112233", "topic": "Implant"},
        headers={CSRF_HEADER: csrf},
    )

    assert response.status_code == 201


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("post", "/api/admin/doctors", {"name": "Dr. New", "specialty": "Ortodont"}),
        ("post", "/api/admin/knowledge-base", {"question": "Narx?", "answer": "3 000 000"}),
        ("patch", "/api/admin/settings", {"clinic_address": "Yangi manzil"}),
    ],
)
async def test_the_front_desk_cannot_change_what_the_clinic_says(
    client: httpx.AsyncClient,
    seed: Seed,
    front_desk: Any,
    method: str,
    path: str,
    payload: dict[str, Any],
) -> None:
    """A wrong answer in the knowledge base is quoted to every patient who
    asks, by a bot, with nobody reading it first — so editing it is not part
    of a front-desk shift.
    """
    csrf = _login_as(client, front_desk.id)

    response = await getattr(client, method)(path, json=payload, headers={CSRF_HEADER: csrf})

    assert response.status_code == 403


async def test_a_role_nobody_defined_cannot_be_stored_at_all(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """The regression this whole change exists for, caught one layer lower
    than expected.

    Under the previous check (`if role == "doctor": deny`) an account with
    any other role held every right over the clinic, so a typo was an
    administrator. The permission layer now refuses an unknown role
    (tests/test_roles.py), and the table refuses to hold one in the first
    place — so this cannot be written through a script or a psql session
    either, which is where such a value would realistically come from.
    """
    with as_tenant(seed.tenant_a.id), pytest.raises(IntegrityError, match="ck_operators_role"):
        await OperatorRepository(db_session).create(
            name="Typo",
            role="operatr",
            username=f"typo-{seed.tenant_a.id}",
            password_hash="x",
        )
    await db_session.rollback()


async def test_the_csv_of_patient_names_and_phones_is_not_open_to_every_account(
    client: httpx.AsyncClient, seed: Seed
) -> None:
    """The export carries patient names and phone numbers out of the
    building, so it is gated as patient data rather than as a report. The
    seed operator is a `doctor`.
    """
    _login_as(client, seed.a.operator.id)

    response = await client.get("/api/admin/export/appointments.csv")

    assert response.status_code == 403
