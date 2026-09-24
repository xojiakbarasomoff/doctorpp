import hashlib
import hmac
import json
import logging
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractContextManager
from uuid import UUID

import httpx
import pytest
from arq.connections import ArqRedis
from arq.jobs import Job
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.db import get_db_session
from app.core.queue import get_arq_pool
from app.main import app
from app.models.message import MessageSender
from app.repositories.conversation import ConversationRepository
from app.repositories.message import MessageRepository
from app.repositories.user import UserRepository
from app.services.debounce import FIRE_DEBOUNCE_WINDOW_JOB
from tests.conftest import Seed

TEST_SETTINGS = Settings(
    database_url="postgresql+asyncpg://test:test@localhost/test",
    redis_url="redis://localhost:6379/0",
    openai_api_key="sk-test",
    webhook_verify_token="test-verify-token",
    meta_app_secret="test-app-secret",
)


@pytest.fixture(autouse=True)
def _override_settings() -> Iterator[None]:
    app.dependency_overrides[get_settings] = lambda: TEST_SETTINGS
    yield
    app.dependency_overrides.pop(get_settings, None)


@pytest.fixture
async def client(
    db_session: AsyncSession, redis_pool: ArqRedis
) -> AsyncIterator[httpx.AsyncClient]:
    """An httpx.AsyncClient driving the app in-process over ASGI, sharing
    this test's event loop (and so its db_session) — unlike
    fastapi.testclient.TestClient, which runs the app on a separate thread
    with its own event loop and would break the asyncpg connection bound to
    db_session's loop.

    Uses the real redis_pool fixture (dedicated test Redis DB) rather than a
    mock: handle_inbound_message does real RPUSH/LRANGE/Lua-script work that
    isn't meaningfully fakeable.
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


async def _queued_jobs(pool: ArqRedis) -> list[Job]:
    job_ids = await pool.zrange("arq:queue", 0, -1)
    return [Job(job_id.decode(), redis=pool, _queue_name="arq:queue") for job_id in job_ids]


async def _queued_function_names(pool: ArqRedis) -> list[str]:
    return [(await job.info()).function for job in await _queued_jobs(pool)]


async def _answer_jobs(pool: ArqRedis) -> list[Job]:
    """Queued jobs, less the username lookup every new patient also gets."""
    return [
        job for job in await _queued_jobs(pool) if (await job.info()).function != "resolve_username"
    ]


def _sign(body: bytes, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


# --- GET /webhook verification handshake ---


async def test_verify_webhook_with_valid_token_returns_challenge(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get(
        "/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "test-verify-token",
            "hub.challenge": "12345",
        },
    )
    assert response.status_code == 200
    assert response.text == "12345"


async def test_verify_webhook_with_invalid_token_returns_403(client: httpx.AsyncClient) -> None:
    response = await client.get(
        "/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong-token",
            "hub.challenge": "12345",
        },
    )
    assert response.status_code == 403


async def test_verify_webhook_with_wrong_mode_returns_403(client: httpx.AsyncClient) -> None:
    response = await client.get(
        "/webhook",
        params={
            "hub.mode": "unsubscribe",
            "hub.verify_token": "test-verify-token",
            "hub.challenge": "12345",
        },
    )
    assert response.status_code == 403


# --- POST /webhook signature validation ---


async def test_receive_webhook_with_valid_signature_returns_200(client: httpx.AsyncClient) -> None:
    body = json.dumps({"object": "instagram", "entry": []}).encode("utf-8")
    signature = _sign(body, "test-app-secret")

    response = await client.post(
        "/webhook", content=body, headers={"X-Hub-Signature-256": signature}
    )
    assert response.status_code == 200


async def test_receive_webhook_with_invalid_signature_returns_403(
    client: httpx.AsyncClient,
) -> None:
    body = json.dumps({"object": "instagram", "entry": []}).encode("utf-8")

    response = await client.post(
        "/webhook", content=body, headers={"X-Hub-Signature-256": "sha256=" + "0" * 64}
    )
    assert response.status_code == 403


async def test_receive_webhook_with_missing_signature_returns_403(
    client: httpx.AsyncClient,
) -> None:
    body = json.dumps({"object": "instagram", "entry": []}).encode("utf-8")

    response = await client.post("/webhook", content=body)
    assert response.status_code == 403


async def test_unenforced_invalid_signature_is_processed_not_rejected(
    client: httpx.AsyncClient,
) -> None:
    """WEBHOOK_SIGNATURE_ENFORCED=false is a diagnostic mode: the check
    still runs and still reports, but a request that fails it is handled
    instead of refused. Asserted explicitly because the whole point of the
    flag is to change the outcome of a failing check, and a regression
    that silently kept rejecting would look identical from the outside to
    the flag not being read at all.
    """
    unenforced = TEST_SETTINGS.model_copy(update={"webhook_signature_enforced": False})
    app.dependency_overrides[get_settings] = lambda: unenforced
    try:
        body = json.dumps({"object": "instagram", "entry": []}).encode("utf-8")
        response = await client.post(
            "/webhook", content=body, headers={"X-Hub-Signature-256": "sha256=" + "0" * 64}
        )
    finally:
        app.dependency_overrides[get_settings] = lambda: TEST_SETTINGS
    assert response.status_code == 200


def test_signature_enforcement_defaults_to_on() -> None:
    # The insecure mode must never be what you get by forgetting to set
    # the variable.
    assert TEST_SETTINGS.webhook_signature_enforced is True


async def test_receive_webhook_succeeds_with_no_session_cookie_and_no_csrf_token(
    client: httpx.AsyncClient,
) -> None:
    """The dashboard's session/CSRF layer (app.api.auth) must never reach
    the webhook — Meta calls this endpoint with no session cookie and no
    CSRF token, authenticating solely via X-Hub-Signature-256. Nothing here
    sends a cookie or a csrf_token field; a signed request must still
    succeed exactly as it did before that layer existed, proving isolation
    isn't just "no test caught a regression" but an explicit, permanent
    guarantee.
    """
    assert "session" not in client.cookies
    body = json.dumps({"object": "instagram", "entry": []}).encode("utf-8")
    signature = _sign(body, "test-app-secret")

    response = await client.post(
        "/webhook", content=body, headers={"X-Hub-Signature-256": signature}
    )

    assert response.status_code == 200
    assert "session" not in response.cookies


# --- tenant resolution + echo filtering + debounce registration ---


def _messaging_payload(
    page_id: str,
    sender_id: str,
    recipient_id: str,
    text: str | None,
    is_echo: bool = False,
    mid: str | None = None,
) -> bytes:
    message: dict[str, object] = {"text": text, "is_echo": is_echo}
    if mid is not None:
        message["mid"] = mid
    payload = {
        "object": "instagram",
        "entry": [
            {
                "id": page_id,
                "messaging": [
                    {
                        "sender": {"id": sender_id},
                        "recipient": {"id": recipient_id},
                        "message": message,
                    }
                ],
            }
        ],
    }
    return json.dumps(payload).encode("utf-8")


async def test_receive_webhook_skips_echo_event(
    client: httpx.AsyncClient,
    seed: Seed,
    redis_pool: ArqRedis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    page_id = seed.a.channel.external_id
    body = _messaging_payload(page_id, page_id, "user-1", "We'll be right with you", is_echo=True)
    signature = _sign(body, "test-app-secret")

    with caplog.at_level(logging.INFO, logger="app.api.webhook"):
        response = await client.post(
            "/webhook", content=body, headers={"X-Hub-Signature-256": signature}
        )

    assert response.status_code == 200
    assert "webhook_echo_skipped" in caplog.text
    assert "webhook_message_received" not in caplog.text
    # Nothing is answered; the only job is the check for a person replying.
    assert await _queued_function_names(redis_pool) == ["note_clinic_reply"]


async def test_receive_webhook_skips_event_from_own_account_without_echo_flag(
    client: httpx.AsyncClient,
    seed: Seed,
    redis_pool: ArqRedis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    page_id = seed.a.channel.external_id
    body = _messaging_payload(page_id, page_id, "user-1", "Auto message", is_echo=False)
    signature = _sign(body, "test-app-secret")

    with caplog.at_level(logging.INFO, logger="app.api.webhook"):
        response = await client.post(
            "/webhook", content=body, headers={"X-Hub-Signature-256": signature}
        )

    assert response.status_code == 200
    assert "webhook_echo_skipped" in caplog.text
    assert await _queued_function_names(redis_pool) == ["note_clinic_reply"]


async def test_receive_webhook_processes_genuine_inbound_message(
    client: httpx.AsyncClient,
    seed: Seed,
    redis_pool: ArqRedis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    page_id = seed.a.channel.external_id
    text = "Hi, do you have an opening tomorrow?"
    body = _messaging_payload(page_id, "user-1", page_id, text)
    signature = _sign(body, "test-app-secret")

    with caplog.at_level(logging.INFO, logger="app.api.webhook"):
        response = await client.post(
            "/webhook", content=body, headers={"X-Hub-Signature-256": signature}
        )

    assert response.status_code == 200
    assert "webhook_echo_skipped" not in caplog.text

    # Full patient message text must not appear in the INFO record — only
    # length + a short preview (TZ section 7, personal data).
    [record] = [r for r in caplog.records if r.message == "webhook_message_received"]
    assert record.tenant_id == str(seed.tenant_a.id)  # type: ignore[attr-defined]
    assert record.message_length == len(text)  # type: ignore[attr-defined]
    assert record.message_preview == text  # type: ignore[attr-defined]
    assert "message_text" not in record.__dict__

    # The webhook returns 200 without generating the reply inline (no
    # OpenAI call in the request path) — it hands off to the debounce layer,
    # which schedules a deferred fire_debounce_window job rather than
    # calling process_inbound_message directly.
    # The webhook returns 200 without generating the reply inline, and the
    # job it schedules names the channel and conversation the message
    # belongs to — so the worker replies over the account that was written
    # to instead of picking one from the tenant.
    [job] = await _answer_jobs(redis_pool)
    info = await job.info()
    assert info is not None
    assert info.function == FIRE_DEBOUNCE_WINDOW_JOB
    tenant_id, channel_id, conversation_id, sender, generation, reply_context = info.args
    assert tenant_id == str(seed.tenant_a.id)
    assert channel_id == str(seed.a.channel.id)
    assert (sender, generation) == ("user-1", 1)
    # Instagram routes a reply by the recipient's id alone.
    assert reply_context is None
    assert uuid.UUID(conversation_id)


async def test_receive_webhook_from_unknown_ig_account_is_skipped_and_logged(
    client: httpx.AsyncClient,
    redis_pool: ArqRedis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    body = _messaging_payload(
        "no-such-page", "user-1", "no-such-page", "Hi, do you have an opening tomorrow?"
    )
    signature = _sign(body, "test-app-secret")

    with caplog.at_level(logging.INFO, logger="app.api.webhook"):
        response = await client.post(
            "/webhook", content=body, headers={"X-Hub-Signature-256": signature}
        )

    assert response.status_code == 200
    assert "webhook_unknown_ig_account" in caplog.text
    assert "webhook_message_received" not in caplog.text
    assert await _queued_jobs(redis_pool) == []


async def test_receive_webhook_tenant_b_account_resolves_to_tenant_b_not_a(
    client: httpx.AsyncClient,
    seed: Seed,
    redis_pool: ArqRedis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    page_id = seed.b.channel.external_id
    text = "What are your hours?"
    body = _messaging_payload(page_id, "user-1", page_id, text)
    signature = _sign(body, "test-app-secret")

    with caplog.at_level(logging.INFO, logger="app.api.webhook"):
        response = await client.post(
            "/webhook", content=body, headers={"X-Hub-Signature-256": signature}
        )

    assert response.status_code == 200
    [record] = [r for r in caplog.records if r.message == "webhook_message_received"]
    assert record.tenant_id == str(seed.tenant_b.id)  # type: ignore[attr-defined]
    assert record.tenant_id != str(seed.tenant_a.id)  # type: ignore[attr-defined]

    [job] = await _answer_jobs(redis_pool)
    info = await job.info()
    assert info is not None
    assert info.args[:2] == (str(seed.tenant_b.id), str(seed.b.channel.id))


async def test_receive_webhook_attachment_only_message_is_skipped_and_not_enqueued(
    client: httpx.AsyncClient,
    seed: Seed,
    redis_pool: ArqRedis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    page_id = seed.a.channel.external_id
    body = _messaging_payload(page_id, "user-1", page_id, None)
    signature = _sign(body, "test-app-secret")

    with caplog.at_level(logging.INFO, logger="app.api.webhook"):
        response = await client.post(
            "/webhook", content=body, headers={"X-Hub-Signature-256": signature}
        )

    assert response.status_code == 200
    assert "webhook_attachment_only_skipped" in caplog.text
    assert "webhook_message_received" not in caplog.text
    assert await _queued_jobs(redis_pool) == []


async def test_a_photo_is_recorded_and_handed_to_the_media_job(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    redis_pool: ArqRedis,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """A patient's photo is kept, not skipped: it goes into the transcript,
    and the job that stores it, acknowledges it and tells the doctor is
    queued -- the model is never asked about it."""
    page_id = seed.a.channel.external_id
    payload = {
        "object": "instagram",
        "entry": [
            {
                "id": page_id,
                "time": 1,
                "messaging": [
                    {
                        "sender": {"id": "photo-patient"},
                        "recipient": {"id": page_id},
                        "timestamp": 1,
                        "message": {
                            "mid": "mid-photo-1",
                            "attachments": [
                                {"type": "image", "payload": {"url": "https://cdn.example/a.jpg"}}
                            ],
                        },
                    }
                ],
            }
        ],
    }
    body = json.dumps(payload).encode()

    response = await _post(client, body)

    assert response.status_code == 200
    functions = [(await job.info()).function for job in await _queued_jobs(redis_pool)]
    assert "handle_patient_media" in functions
    assert FIRE_DEBOUNCE_WINDOW_JOB not in functions
    with as_tenant(seed.tenant_a.id):
        user = await UserRepository(db_session).get_by_external_id(
            channel_id=seed.a.channel.id, external_id="photo-patient"
        )
        assert user is not None


# --- inbound messages are recorded, deduplicated, and respect operator takeover ---


async def _post(client: httpx.AsyncClient, body: bytes) -> httpx.Response:
    return await client.post(
        "/webhook", content=body, headers={"X-Hub-Signature-256": _sign(body, "test-app-secret")}
    )


async def test_receive_webhook_records_the_patient_and_their_message(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """The transcript is the point of the conversation store: a patient
    writing for the first time must end up with a User, an open
    Conversation, and their message on it.
    """
    page_id = seed.a.channel.external_id
    text = "Assalom alaykum, implant qilasizmi?"

    response = await _post(client, _messaging_payload(page_id, "new-patient", page_id, text))

    assert response.status_code == 200
    with as_tenant(seed.tenant_a.id):
        user = await UserRepository(db_session).get_by_external_id(
            channel_id=seed.a.channel.id, external_id="new-patient"
        )
        assert user is not None
        conversation = await ConversationRepository(db_session).get_open_for_user(user.id)
        assert conversation is not None
        messages = await MessageRepository(db_session).list_recent(conversation.id, 10)

    assert [(m.sender, m.content, m.channel) for m in messages] == [
        (MessageSender.PATIENT, text, "instagram")
    ]


async def test_two_messages_from_one_patient_share_a_user_and_conversation(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    page_id = seed.a.channel.external_id

    await _post(client, _messaging_payload(page_id, "new-patient", page_id, "Salom"))
    await _post(client, _messaging_payload(page_id, "new-patient", page_id, "narxi qancha?"))

    with as_tenant(seed.tenant_a.id):
        user = await UserRepository(db_session).get_by_external_id(
            channel_id=seed.a.channel.id, external_id="new-patient"
        )
        assert user is not None
        conversation = await ConversationRepository(db_session).get_open_for_user(user.id)
        assert conversation is not None
        messages = await MessageRepository(db_session).list_recent(conversation.id, 10)

    assert [m.content for m in messages] == ["Salom", "narxi qancha?"]


async def test_redelivered_message_is_not_recorded_or_answered_twice(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    redis_pool: ArqRedis,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Meta repeats a delivery whose 200 came back too slowly or not at all.
    Without a claim on the message id the patient is answered twice and the
    transcript records a message they only sent once.
    """
    page_id = seed.a.channel.external_id
    body = _messaging_payload(page_id, "new-patient", page_id, "Salom", mid="mid-abc123")

    with caplog.at_level(logging.INFO, logger="app.api.webhook"):
        first = await _post(client, body)
        second = await _post(client, body)

    assert (first.status_code, second.status_code) == (200, 200)
    assert "webhook_duplicate_skipped" in caplog.text

    with as_tenant(seed.tenant_a.id):
        user = await UserRepository(db_session).get_by_external_id(
            channel_id=seed.a.channel.id, external_id="new-patient"
        )
        assert user is not None
        conversation = await ConversationRepository(db_session).get_open_for_user(user.id)
        assert conversation is not None
        messages = await MessageRepository(db_session).list_recent(conversation.id, 10)

    assert [m.content for m in messages] == ["Salom"]
    assert len(await _answer_jobs(redis_pool)) == 1


async def test_distinct_message_ids_are_both_processed(
    client: httpx.AsyncClient,
    seed: Seed,
    redis_pool: ArqRedis,
) -> None:
    """The claim must key on the message, not on the sender — two genuine
    messages in a row are not a redelivery.
    """
    page_id = seed.a.channel.external_id

    await _post(client, _messaging_payload(page_id, "new-patient", page_id, "Salom", mid="mid-1"))
    await _post(client, _messaging_payload(page_id, "new-patient", page_id, "narxi?", mid="mid-2"))

    assert len(await _answer_jobs(redis_pool)) == 2


async def test_operator_takeover_records_the_message_but_does_not_answer_it(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    redis_pool: ArqRedis,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Conversation.is_bot_enabled existed as a column that nothing read, so
    an operator taking a conversation over had no effect at all. The bot
    must now stay quiet — while the patient's words still reach the
    transcript, since the operator needs to read them.
    """
    page_id = seed.a.channel.external_id
    with as_tenant(seed.tenant_a.id):
        user = await UserRepository(db_session).create(
            channel_id=seed.a.channel.id, external_id="taken-over"
        )
        conversation = await ConversationRepository(db_session).create(
            user_id=user.id, status="open", is_bot_enabled=False
        )

    with caplog.at_level(logging.INFO, logger="app.api.webhook"):
        response = await _post(
            client, _messaging_payload(page_id, "taken-over", page_id, "yana savolim bor")
        )

    assert response.status_code == 200
    assert "webhook_bot_disabled_for_conversation" in caplog.text
    assert await _answer_jobs(redis_pool) == []

    with as_tenant(seed.tenant_a.id):
        messages = await MessageRepository(db_session).list_recent(conversation.id, 10)
        await db_session.refresh(conversation)
    assert [m.content for m in messages] == ["yana savolim bor"]
    # Nobody but a person will answer now, so the conversation is pinned.
    assert conversation.needs_doctor_since is not None
