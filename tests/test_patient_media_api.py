"""Patient photos: medical images, served from the dashboard's own origin.

Who can fetch them, and what they can be once fetched.
"""

import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractContextManager
from typing import Any

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.admin.deps import CSRF_HEADER
from app.core.db import get_db_session
from app.core.session import SESSION_COOKIE_NAME, create_session_cookie
from app.main import app
from app.models.patient_media import PatientMedia
from app.repositories.operator import OperatorRepository
from app.services import patient_media
from tests.conftest import Seed

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
SVG = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(document.cookie)</script></svg>'


@pytest.fixture
async def client(db_session: AsyncSession) -> AsyncIterator[httpx.AsyncClient]:
    async def _session() -> AsyncSession:
        return db_session

    app.dependency_overrides[get_db_session] = _session
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as ac:
        yield ac
    app.dependency_overrides.pop(get_db_session, None)


@pytest.fixture
async def front_desk(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[uuid.UUID], AbstractContextManager[None]],
) -> Any:
    with as_tenant(seed.tenant_a.id):
        return await OperatorRepository(db_session).create(
            name="Front Desk", role="operator", username=f"fd-{seed.tenant_a.id}", password_hash="x"
        )


def _login(client: httpx.AsyncClient, operator_id: uuid.UUID) -> str:
    cookie, csrf = create_session_cookie(operator_id)
    client.cookies.set(SESSION_COOKIE_NAME, cookie)
    return csrf


async def _photo(
    session: AsyncSession, tenant_id: uuid.UUID, user_id: uuid.UUID, content: bytes, kind: str
) -> uuid.UUID:
    media = PatientMedia(
        tenant_id=tenant_id,
        user_id=user_id,
        source_url="https://lookaside.fbsbx.com/x",
        content=content,
        content_type=kind,
        size_bytes=len(content),
    )
    session.add(media)
    await session.flush()
    return media.id


async def test_a_photo_is_served_with_headers_that_keep_it_inert(
    client: httpx.AsyncClient, db_session: AsyncSession, seed: Seed, front_desk: Any
) -> None:
    media_id = await _photo(db_session, seed.tenant_a.id, seed.a.user.id, PNG, "image/png")
    _login(client, front_desk.id)

    response = await client.get(f"/api/admin/media/{media_id}/file")

    assert response.status_code == 200
    assert response.content == PNG
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "sandbox" in response.headers["content-security-policy"]
    assert "private" in response.headers["cache-control"]


async def test_a_stored_svg_is_never_served_as_something_a_browser_runs(
    client: httpx.AsyncClient, db_session: AsyncSession, seed: Seed, front_desk: Any
) -> None:
    """A row from before SVGs were refused, or put there some other way."""
    media_id = await _photo(db_session, seed.tenant_a.id, seed.a.user.id, SVG, "image/svg+xml")
    _login(client, front_desk.id)

    response = await client.get(f"/api/admin/media/{media_id}/file")

    assert response.headers["content-type"] == "application/octet-stream"
    assert "sandbox" in response.headers["content-security-policy"]


async def test_another_clinics_photo_is_not_found(
    client: httpx.AsyncClient, db_session: AsyncSession, seed: Seed, front_desk: Any
) -> None:
    theirs = await _photo(db_session, seed.tenant_b.id, seed.b.user.id, PNG, "image/png")
    csrf = _login(client, front_desk.id)

    assert (await client.get(f"/api/admin/media/{theirs}/file")).status_code == 404
    changed = await client.post(
        f"/api/admin/media/{theirs}/status",
        json={"status": "reviewed"},
        headers={CSRF_HEADER: csrf},
    )
    assert changed.status_code == 404


async def test_a_photo_needs_a_session(
    client: httpx.AsyncClient, db_session: AsyncSession, seed: Seed
) -> None:
    media_id = await _photo(db_session, seed.tenant_a.id, seed.a.user.id, PNG, "image/png")

    response = await client.get(f"/api/admin/media/{media_id}/file", follow_redirects=False)

    assert response.status_code in {303, 401}
    assert response.content != PNG


async def test_a_view_only_account_cannot_open_patient_photos(
    client: httpx.AsyncClient, db_session: AsyncSession, seed: Seed
) -> None:
    media_id = await _photo(db_session, seed.tenant_a.id, seed.a.user.id, PNG, "image/png")
    _login(client, seed.a.operator.id)

    assert (await client.get(f"/api/admin/media/{media_id}/file")).status_code == 403


@pytest.mark.parametrize(
    ("content_type", "accepted"),
    [
        ("image/jpeg", True),
        ("image/png; charset=binary", True),
        ("IMAGE/WEBP", True),
        ("image/svg+xml", False),
        ("text/html", False),
        ("application/octet-stream", False),
        ("", False),
    ],
)
async def test_only_raster_images_are_downloaded(content_type: str, accepted: bool) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=PNG, headers={"content-type": content_type})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await patient_media.download("https://lookaside.fbsbx.com/x", http=http)

    assert (result is not None) is accepted


async def test_an_oversized_image_is_not_downloaded() -> None:
    big = b"\x00" * (patient_media.MAX_IMAGE_BYTES + 1)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=big, headers={"content-type": "image/jpeg"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        assert await patient_media.download("https://lookaside.fbsbx.com/x", http=http) is None
