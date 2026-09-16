import re
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractContextManager
from datetime import date
from uuid import UUID

import httpx
import pytest
from arq.connections import ArqRedis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.db import get_db_session
from app.core.passwords import hash_password
from app.core.queue import get_arq_pool
from app.main import app
from app.models.operator import Operator
from app.repositories.operator import OperatorRepository
from app.services.login_rate_limit import MAX_LOGIN_ATTEMPTS
from tests.conftest import Seed

TEST_SETTINGS = Settings(
    database_url="postgresql+asyncpg://test:test@localhost/test",
    redis_url="redis://localhost:6379/0",
    openai_api_key="sk-test",
    webhook_verify_token="test-verify-token",
    meta_app_secret="test-app-secret",
)

# A fixed future date, arbitrary but deterministic — avoids "today" flakiness.
BASE_DATE = date(2026, 9, 2)

_CSRF_RE = re.compile(r'name="csrf_token" value="([^"]+)"')


@pytest.fixture(autouse=True)
def _override_settings() -> Iterator[None]:
    app.dependency_overrides[get_settings] = lambda: TEST_SETTINGS
    yield
    app.dependency_overrides.pop(get_settings, None)


@pytest.fixture
async def client(
    db_session: AsyncSession, redis_pool: ArqRedis
) -> AsyncIterator[httpx.AsyncClient]:
    """Same in-process ASGI pattern as tests/test_webhook.py's client fixture
    — shares this test's event loop (and so its db_session), unlike
    fastapi.testclient.TestClient. Needs the real redis_pool now too: login
    rate-limiting does real INCR/EXPIRE/GET/DELETE against Redis.
    """

    async def _get_db_session_override() -> AsyncSession:
        return db_session

    async def _get_arq_pool_override() -> ArqRedis:
        return redis_pool

    app.dependency_overrides[get_db_session] = _get_db_session_override
    app.dependency_overrides[get_arq_pool] = _get_arq_pool_override
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
    app.dependency_overrides.pop(get_db_session, None)
    app.dependency_overrides.pop(get_arq_pool, None)


async def _create_operator(
    db_session: AsyncSession,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    tenant_id: UUID,
    *,
    username: str,
    password: str,
    role: str,
) -> Operator:
    with as_tenant(tenant_id):
        return await OperatorRepository(db_session).create(
            name="Test Staff",
            role=role,
            username=username,
            password_hash=hash_password(password),
        )


async def _login(client: httpx.AsyncClient, username: str, password: str) -> httpx.Response:
    return await client.post(
        "/login", data={"username": username, "password": password}, follow_redirects=False
    )


def _extract_csrf(html: str) -> str:
    match = _CSRF_RE.search(html)
    assert match is not None, "csrf_token field not found in rendered page"
    return match.group(1)


# --- login ---


async def test_login_page_renders(client: httpx.AsyncClient) -> None:
    response = await client.get("/login")
    assert response.status_code == 200
    assert "csrf" not in response.text  # not logged in yet — no session to bind a CSRF token to


async def test_login_wrong_password_shows_error_and_no_cookie(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    await _create_operator(
        db_session,
        as_tenant,
        seed.tenant_a.id,
        username="op-wrongpw",
        password="right",
        role="operator",
    )
    response = await _login(client, "op-wrongpw", "wrong")
    assert response.status_code == 401
    assert "session" not in response.cookies


async def test_login_unknown_username_shows_error(client: httpx.AsyncClient) -> None:
    response = await _login(client, "no-such-user", "whatever")
    assert response.status_code == 401


async def test_login_success_sets_cookie_and_redirects(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    await _create_operator(
        db_session,
        as_tenant,
        seed.tenant_a.id,
        username="op-ok",
        password="correct-horse",
        role="operator",
    )
    response = await _login(client, "op-ok", "correct-horse")
    assert response.status_code == 303
    # The operator dashboard, not the appointments page: everything an
    # operator came for -- conversations, leads, doctors, FAQs, settings --
    # lives at /admin/, and a login that lands elsewhere hides all of it.
    assert response.headers["location"] == "/admin/"
    assert "session" in response.cookies


# --- login rate limiting ---


async def test_login_rate_limits_after_repeated_failures(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    await _create_operator(
        db_session,
        as_tenant,
        seed.tenant_a.id,
        username="op-throttled",
        password="right-password",
        role="operator",
    )

    for _ in range(MAX_LOGIN_ATTEMPTS):
        response = await _login(client, "op-throttled", "wrong-password")
        assert response.status_code == 401

    # Locked out now even with the correct password — rate limiting has to
    # block the attempt itself, not just count failures after the fact.
    response = await _login(client, "op-throttled", "right-password")
    assert response.status_code == 429
    assert "session" not in response.cookies


async def test_login_rate_limit_is_per_username(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    await _create_operator(
        db_session,
        as_tenant,
        seed.tenant_a.id,
        username="op-throttled-a",
        password="pw-a",
        role="operator",
    )
    await _create_operator(
        db_session,
        as_tenant,
        seed.tenant_a.id,
        username="op-throttled-b",
        password="pw-b",
        role="operator",
    )

    for _ in range(MAX_LOGIN_ATTEMPTS):
        await _login(client, "op-throttled-a", "wrong")

    # A different account's own attempts are unaffected by op-throttled-a's lockout.
    response = await _login(client, "op-throttled-b", "pw-b")
    assert response.status_code == 303


# --- the old appointments page is gone ---
# Its behaviour is not retested here, because it no longer exists: booking
# rules live in tests/test_appointment_service.py and the CSRF, role and
# tenant-isolation rules the page relied on are covered against the API the
# operator dashboard actually calls, in tests/test_admin_api.py.


@pytest.mark.parametrize(
    "path",
    ["/dashboard", "/dashboard/appointments", "/dashboard/appointments?date=2026-08-24"],
)
async def test_the_old_dashboard_paths_lead_to_the_operator_dashboard(
    client: httpx.AsyncClient, path: str
) -> None:
    """Bookmarks and browser history outlive a refactor, so the paths stay
    even though the page they served does not.
    """
    response = await client.get(path, follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "/admin/"


async def test_the_redirect_is_not_permanent(client: httpx.AsyncClient) -> None:
    """301 would be cached by the browser indefinitely, so anything served
    at /dashboard again later would need every operator to clear their
    cache. Not a cost worth paying for one round trip on an internal route.
    """
    response = await client.get("/dashboard/appointments", follow_redirects=False)
    assert response.status_code != 301


async def test_booking_through_the_old_page_is_not_quietly_forwarded(
    client: httpx.AsyncClient,
) -> None:
    """A form replayed from a stale page must fail rather than be pointed at
    /api/admin/, which takes a different shape and its own CSRF token.
    """
    response = await client.post(
        "/dashboard/appointments",
        data={"patient_name": "Someone", "time": "10:00"},
        follow_redirects=False,
    )
    assert response.status_code == 405
