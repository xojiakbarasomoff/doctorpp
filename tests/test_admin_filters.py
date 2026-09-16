"""Views the clinic saves for itself.

The point of the feature is that a new question about the clinic's own data
stops being a developer's job, so the cases worth having are the ones where
that could go wrong: a saved view outliving the control it was built on,
one clinic seeing another's, and anything that would let a saved value mean
more than a value.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractContextManager
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.admin.deps import CSRF_HEADER
from app.core.db import get_db_session
from app.core.session import SESSION_COOKIE_NAME, create_session_cookie
from app.main import app
from app.repositories.operator import OperatorRepository
from app.repositories.saved_filter import SavedFilterRepository
from tests.conftest import Seed


@pytest.fixture
async def client(db_session: AsyncSession) -> AsyncIterator[httpx.AsyncClient]:
    async def _get_db_session_override() -> AsyncSession:
        return db_session

    app.dependency_overrides[get_db_session] = _get_db_session_override
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
    app.dependency_overrides.pop(get_db_session, None)


@pytest.fixture
async def admin(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> Any:
    with as_tenant(seed.tenant_a.id):
        return await OperatorRepository(db_session).create(
            name="Admin A", role="admin", username=f"fadmin-{seed.tenant_a.id}", password_hash="x"
        )


def _login_as(client: httpx.AsyncClient, operator_id: UUID) -> str:
    cookie_value, csrf_token = create_session_cookie(operator_id)
    client.cookies.set(SESSION_COOKIE_NAME, cookie_value)
    return csrf_token


TODAYS_TELEGRAM = {
    "resource": "appointments",
    "name": "Bugungi Telegram qabullari",
    "params": {"source": "telegram", "status": "scheduled"},
}


# --- saving a view ----------------------------------------------------------


async def test_an_admin_saves_a_view_the_clinic_asks_for_every_morning(
    client: httpx.AsyncClient, seed: Seed, admin: Any
) -> None:
    csrf = _login_as(client, admin.id)

    response = await client.post(
        "/api/admin/filters", json=TODAYS_TELEGRAM, headers={CSRF_HEADER: csrf}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Bugungi Telegram qabullari"
    assert body["params"] == {"source": "telegram", "status": "scheduled"}


async def test_a_saved_view_replays_through_the_endpoint_it_was_saved_for(
    client: httpx.AsyncClient, seed: Seed, admin: Any
) -> None:
    """The whole design: a saved filter is parameters for an endpoint that
    already exists, so applying one is the same request a person clicking
    the controls makes. If this ever fails, there are two query paths and
    one of them will drift.
    """
    csrf = _login_as(client, admin.id)
    saved = (
        await client.post("/api/admin/filters", json=TODAYS_TELEGRAM, headers={CSRF_HEADER: csrf})
    ).json()

    replayed = await client.get("/api/admin/appointments", params=saved["params"])

    assert replayed.status_code == 200


async def test_two_views_cannot_share_a_name_on_the_same_list(
    client: httpx.AsyncClient, seed: Seed, admin: Any
) -> None:
    """The second would be unreachable in the interface, which is a worse
    answer than refusing to save it.
    """
    csrf = _login_as(client, admin.id)
    await client.post("/api/admin/filters", json=TODAYS_TELEGRAM, headers={CSRF_HEADER: csrf})

    again = await client.post(
        "/api/admin/filters", json=TODAYS_TELEGRAM, headers={CSRF_HEADER: csrf}
    )

    assert again.status_code == 409


async def test_the_same_name_on_a_different_list_is_a_different_question(
    client: httpx.AsyncClient, seed: Seed, admin: Any
) -> None:
    csrf = _login_as(client, admin.id)
    await client.post(
        "/api/admin/filters",
        json={"resource": "appointments", "name": "Bugun", "params": {}},
        headers={CSRF_HEADER: csrf},
    )

    other = await client.post(
        "/api/admin/filters",
        json={"resource": "leads", "name": "Bugun", "params": {"status": "new"}},
        headers={CSRF_HEADER: csrf},
    )

    assert other.status_code == 201


# --- what a saved view may contain ------------------------------------------


async def test_a_parameter_the_list_does_not_understand_is_dropped_not_refused(
    client: httpx.AsyncClient, seed: Seed, admin: Any
) -> None:
    """A view saved before a list gained or lost a control should keep
    working, showing the rows it still can — not become an error the clinic
    cannot fix from the interface.
    """
    csrf = _login_as(client, admin.id)

    response = await client.post(
        "/api/admin/filters",
        json={
            "resource": "leads",
            "name": "Yangi lidlar",
            "params": {"status": "new", "tenant_id": str(uuid4()), "order_by": "password"},
        },
        headers={CSRF_HEADER: csrf},
    )

    assert response.status_code == 201
    assert response.json()["params"] == {"status": "new"}


async def test_a_saved_view_cannot_become_a_place_to_keep_documents(
    client: httpx.AsyncClient, seed: Seed, admin: Any
) -> None:
    csrf = _login_as(client, admin.id)

    response = await client.post(
        "/api/admin/filters",
        json={"resource": "leads", "name": "Katta", "params": {"status": "x" * 500}},
        headers={CSRF_HEADER: csrf},
    )

    assert response.status_code == 422


async def test_a_nested_value_is_refused(client: httpx.AsyncClient, seed: Seed, admin: Any) -> None:
    """Query parameters are flat. Anything else is somebody storing
    structure where the API only ever reads a string.
    """
    csrf = _login_as(client, admin.id)

    response = await client.post(
        "/api/admin/filters",
        json={"resource": "leads", "name": "Ichma-ich", "params": {"status": {"$ne": None}}},
        headers={CSRF_HEADER: csrf},
    )

    assert response.status_code == 422


async def test_a_list_nobody_serves_is_refused(
    client: httpx.AsyncClient, seed: Seed, admin: Any
) -> None:
    csrf = _login_as(client, admin.id)

    response = await client.post(
        "/api/admin/filters",
        json={"resource": "operators", "name": "Xodimlar", "params": {}},
        headers={CSRF_HEADER: csrf},
    )

    assert response.status_code == 422


# --- who may do what --------------------------------------------------------


async def test_the_front_desk_can_use_the_views_but_not_write_them(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    admin: Any,
) -> None:
    """A view is only worth having if the people working the lists can click
    it; deciding which views exist is the admin's.
    """
    csrf = _login_as(client, admin.id)
    await client.post("/api/admin/filters", json=TODAYS_TELEGRAM, headers={CSRF_HEADER: csrf})

    with as_tenant(seed.tenant_a.id):
        desk = await OperatorRepository(db_session).create(
            name="Front desk",
            role="operator",
            username=f"fdesk-{seed.tenant_a.id}",
            password_hash="x",
        )
    desk_csrf = _login_as(client, desk.id)

    assert len((await client.get("/api/admin/filters")).json()) == 1
    refused = await client.post(
        "/api/admin/filters",
        json={"resource": "leads", "name": "Meniki", "params": {}},
        headers={CSRF_HEADER: desk_csrf},
    )
    assert refused.status_code == 403


async def test_one_clinics_views_are_not_another_clinics(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    admin: Any,
) -> None:
    csrf = _login_as(client, admin.id)
    await client.post("/api/admin/filters", json=TODAYS_TELEGRAM, headers={CSRF_HEADER: csrf})

    with as_tenant(seed.tenant_b.id):
        other_admin = await OperatorRepository(db_session).create(
            name="Admin B", role="admin", username=f"fadmin-{seed.tenant_b.id}", password_hash="x"
        )
    _login_as(client, other_admin.id)

    assert (await client.get("/api/admin/filters")).json() == []


async def test_another_clinics_view_cannot_be_deleted(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    admin: Any,
) -> None:
    with as_tenant(seed.tenant_b.id):
        theirs = await SavedFilterRepository(db_session).create(
            resource="leads", name="Ularniki", params={}, position=0
        )
    csrf = _login_as(client, admin.id)

    response = await client.delete(f"/api/admin/filters/{theirs.id}", headers={CSRF_HEADER: csrf})

    assert response.status_code == 404


async def test_a_write_without_the_csrf_header_is_refused(
    client: httpx.AsyncClient, seed: Seed, admin: Any
) -> None:
    _login_as(client, admin.id)

    response = await client.post("/api/admin/filters", json=TODAYS_TELEGRAM)

    assert response.status_code == 403


# --- keeping them tidy ------------------------------------------------------


async def test_a_view_can_be_renamed_and_reordered(
    client: httpx.AsyncClient, seed: Seed, admin: Any
) -> None:
    csrf = _login_as(client, admin.id)
    saved = (
        await client.post("/api/admin/filters", json=TODAYS_TELEGRAM, headers={CSRF_HEADER: csrf})
    ).json()

    response = await client.patch(
        f"/api/admin/filters/{saved['id']}",
        json={"name": "  Telegram   qabullari  ", "position": 3},
        headers={CSRF_HEADER: csrf},
    )

    assert response.status_code == 200
    # Whitespace a person typed is not part of the name.
    assert response.json()["name"] == "Telegram qabullari"
    assert response.json()["position"] == 3


async def test_a_deleted_view_is_gone_from_the_list(
    client: httpx.AsyncClient, seed: Seed, admin: Any
) -> None:
    csrf = _login_as(client, admin.id)
    saved = (
        await client.post("/api/admin/filters", json=TODAYS_TELEGRAM, headers={CSRF_HEADER: csrf})
    ).json()

    removed = await client.delete(f"/api/admin/filters/{saved['id']}", headers={CSRF_HEADER: csrf})

    assert removed.status_code == 204
    assert (await client.get("/api/admin/filters")).json() == []
