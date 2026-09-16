"""The Telegram inbound edge and adapter.

The pipeline behind this route is the same code the Instagram bot runs and is
covered by its own tests; what is exercised here is everything Telegram-
shaped — authentication, the update formats, and the business-connection
routing that has to survive the trip through the queue.
"""

import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import httpx
import pytest
from arq.connections import ArqRedis
from arq.jobs import Job
from sqlalchemy.ext.asyncio import AsyncSession

from app.channels.base import ChannelType, DeliveryBlocked
from app.channels.telegram.adapter import BUSINESS_CONNECTION_ID, TelegramAdapter
from app.channels.telegram.client import TelegramClient, is_placeholder_credential
from app.core.db import get_db_session
from app.core.encryption import encrypt
from app.core.queue import get_arq_pool
from app.main import app
from app.models.message import MessageSender
from app.repositories.channel import ChannelRepository
from app.repositories.conversation import ConversationRepository
from app.repositories.message import MessageRepository
from app.repositories.user import UserRepository
from app.services.debounce import FIRE_DEBOUNCE_WINDOW_JOB
from app.services.delivery import send_reply
from tests.conftest import Seed

BOT_ID = "8123456789"
WEBHOOK_SECRET = "a-real-webhook-secret"
CHAT_ID = 555000111


class FakeTelegramClient(TelegramClient):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, str | None]] = []

    async def send_message(
        self,
        *,
        bot_token: str,
        chat_id: str,
        text: str,
        business_connection_id: str | None = None,
    ) -> None:
        self.calls.append((bot_token, chat_id, text, business_connection_id))


@pytest.fixture
async def telegram_channel(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> Any:
    """A configured Telegram channel on tenant A, with a real secret."""
    with as_tenant(seed.tenant_a.id):
        return await ChannelRepository(db_session).create(
            type=ChannelType.TELEGRAM,
            external_id=BOT_ID,
            credentials=encrypt("8123456789:AAbbCCddEEff"),
            config={"webhook_secret": WEBHOOK_SECRET},
        )


@pytest.fixture
async def client(
    db_session: AsyncSession, redis_pool: ArqRedis
) -> AsyncIterator[httpx.AsyncClient]:
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


def _update(
    text: str | None = "Assalom alaykum",
    *,
    update_id: int = 1,
    message_id: int = 10,
    chat_id: int = CHAT_ID,
    business_connection_id: str | None = None,
    from_id: int | None = None,
    from_is_bot: bool = False,
    field: str = "message",
) -> bytes:
    message: dict[str, Any] = {"message_id": message_id, "chat": {"id": chat_id}}
    if text is not None:
        message["text"] = text
    if business_connection_id is not None:
        message["business_connection_id"] = business_connection_id
    if from_id is not None:
        message["from"] = {"id": from_id, "is_bot": from_is_bot}
    return json.dumps({"update_id": update_id, field: message}).encode("utf-8")


async def _post(
    client: httpx.AsyncClient,
    body: bytes,
    *,
    secret: str | None = WEBHOOK_SECRET,
    bot: str = BOT_ID,
) -> httpx.Response:
    headers = {"Content-Type": "application/json"}
    if secret is not None:
        headers["X-Telegram-Bot-Api-Secret-Token"] = secret
    return await client.post(f"/webhook/telegram/{bot}", content=body, headers=headers)


async def _queued_jobs(pool: ArqRedis) -> list[Job]:
    job_ids = await pool.zrange("arq:queue", 0, -1)
    return [Job(job_id.decode(), redis=pool, _queue_name="arq:queue") for job_id in job_ids]


# --- authentication ---


async def test_update_without_the_secret_is_rejected(
    client: httpx.AsyncClient, telegram_channel: Any, redis_pool: ArqRedis
) -> None:
    response = await _post(client, _update(), secret=None)

    assert response.status_code == 403
    assert await _queued_jobs(redis_pool) == []


async def test_update_with_the_wrong_secret_is_rejected(
    client: httpx.AsyncClient, telegram_channel: Any, redis_pool: ArqRedis
) -> None:
    response = await _post(client, _update(), secret="not-the-secret")

    assert response.status_code == 403
    assert await _queued_jobs(redis_pool) == []


async def test_a_channel_with_no_secret_configured_rejects_everything(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An endpoint that accepts unauthenticated updates lets anyone write
    into a clinic's patient transcript. Configuring the secret is part of
    registering the webhook, not optional hardening — so a channel without
    one fails closed rather than open.
    """
    with as_tenant(seed.tenant_a.id):
        await ChannelRepository(db_session).create(
            type=ChannelType.TELEGRAM,
            external_id="9000000000",
            credentials=encrypt("9000000000:token"),
        )

    with caplog.at_level(logging.INFO, logger="app.api.telegram_webhook"):
        response = await _post(client, _update(), bot="9000000000", secret=None)

    assert response.status_code == 403
    assert "telegram_webhook_secret_not_configured" in caplog.text


async def test_unknown_bot_id_is_refused_without_saying_so(
    client: httpx.AsyncClient, telegram_channel: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """Same 403 as a bad secret: an endpoint that distinguishes the two
    tells an unauthenticated caller which bot ids exist.
    """
    with caplog.at_level(logging.INFO, logger="app.api.telegram_webhook"):
        response = await _post(client, _update(), bot="1111111111")

    assert response.status_code == 403
    assert "telegram_webhook_unknown_bot" in caplog.text
    assert "Invalid webhook secret" in response.text


async def test_an_inactive_channel_stops_serving_its_webhook(
    client: httpx.AsyncClient, db_session: AsyncSession, telegram_channel: Any
) -> None:
    telegram_channel.is_active = False
    await db_session.flush()

    assert (await _post(client, _update())).status_code == 403


# --- update shapes ---


async def test_a_genuine_message_is_recorded_and_queued(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    telegram_channel: Any,
    redis_pool: ArqRedis,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    response = await _post(client, _update("Implant narxi qancha?"))

    assert response.status_code == 200

    with as_tenant(seed.tenant_a.id):
        user = await UserRepository(db_session).get_by_external_id(
            channel_id=telegram_channel.id, external_id=str(CHAT_ID)
        )
        assert user is not None
        conversation = await ConversationRepository(db_session).get_open_for_user(user.id)
        assert conversation is not None
        messages = await MessageRepository(db_session).list_recent(conversation.id, 10)

    assert [(m.sender, m.content, m.channel) for m in messages] == [
        (MessageSender.PATIENT, "Implant narxi qancha?", ChannelType.TELEGRAM)
    ]

    [job] = await _queued_jobs(redis_pool)
    info = await job.info()
    assert info is not None
    assert info.function == FIRE_DEBOUNCE_WINDOW_JOB
    tenant_id, channel_id, _conversation_id, sender, _generation, reply_context = info.args
    assert tenant_id == str(seed.tenant_a.id)
    assert channel_id == str(telegram_channel.id)
    assert sender == str(CHAT_ID)
    # A direct chat carries no business connection, so nothing to route by.
    assert reply_context is None


async def test_a_business_message_is_handled_like_a_direct_one(
    client: httpx.AsyncClient, telegram_channel: Any, redis_pool: ArqRedis
) -> None:
    """Telegram delivers a message to a Business account under its own
    field. It is the same thing to this pipeline — a patient talking to the
    clinic — and must not be dropped for arriving differently.
    """
    response = await _post(
        client,
        _update("Salom", business_connection_id="bc-1", from_id=CHAT_ID, field="business_message"),
    )

    assert response.status_code == 200
    [job] = await _queued_jobs(redis_pool)
    info = await job.info()
    assert info is not None
    assert info.args[-1] == {BUSINESS_CONNECTION_ID: "bc-1"}


async def test_a_message_the_clinic_typed_itself_is_not_answered(
    client: httpx.AsyncClient,
    telegram_channel: Any,
    redis_pool: ArqRedis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Over a business connection Telegram also delivers what the clinic's
    own staff type from their personal account. Answering it would have the
    bot talking to the receptionist.
    """
    with caplog.at_level(logging.INFO, logger="app.api.telegram_webhook"):
        response = await _post(
            client,
            _update(
                "Ertaga 10:00 ga yozdim",
                business_connection_id="bc-1",
                from_id=999888777,  # the operator, not the chat's patient
                field="business_message",
            ),
        )

    assert response.status_code == 200
    assert "telegram_business_operator_message_skipped" in caplog.text
    assert await _queued_jobs(redis_pool) == []


async def test_a_message_from_a_bot_is_skipped(
    client: httpx.AsyncClient,
    telegram_channel: Any,
    redis_pool: ArqRedis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="app.api.telegram_webhook"):
        response = await _post(client, _update("echo", from_id=CHAT_ID, from_is_bot=True))

    assert response.status_code == 200
    assert "telegram_bot_message_skipped" in caplog.text
    assert await _queued_jobs(redis_pool) == []


async def test_a_message_with_no_text_is_skipped(
    client: httpx.AsyncClient,
    telegram_channel: Any,
    redis_pool: ArqRedis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="app.api.telegram_webhook"):
        response = await _post(client, _update(None))

    assert response.status_code == 200
    assert "telegram_non_text_message_skipped" in caplog.text
    assert await _queued_jobs(redis_pool) == []


async def test_a_non_message_update_is_skipped(
    client: httpx.AsyncClient,
    telegram_channel: Any,
    redis_pool: ArqRedis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    body = json.dumps({"update_id": 7, "callback_query": {"id": "cb-1"}}).encode("utf-8")

    with caplog.at_level(logging.INFO, logger="app.api.telegram_webhook"):
        response = await _post(client, body)

    assert response.status_code == 200
    assert "telegram_non_message_update_skipped" in caplog.text
    assert await _queued_jobs(redis_pool) == []


async def test_an_unparseable_payload_is_acknowledged_not_retried(
    client: httpx.AsyncClient, telegram_channel: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """200, not 4xx: Telegram retries anything else, and a payload this
    deployment cannot parse will not parse on the retry either.
    """
    with caplog.at_level(logging.INFO, logger="app.api.telegram_webhook"):
        response = await _post(client, b"{not json")

    assert response.status_code == 200
    assert "telegram_webhook_payload_invalid" in caplog.text


# --- redelivery and takeover ---


async def test_a_redelivered_update_is_not_recorded_or_answered_twice(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    telegram_channel: Any,
    redis_pool: ArqRedis,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    body = _update("Salom", update_id=42, message_id=100)

    with caplog.at_level(logging.INFO, logger="app.api.telegram_webhook"):
        first = await _post(client, body)
        second = await _post(client, body)

    assert (first.status_code, second.status_code) == (200, 200)
    assert "telegram_duplicate_skipped" in caplog.text

    with as_tenant(seed.tenant_a.id):
        user = await UserRepository(db_session).get_by_external_id(
            channel_id=telegram_channel.id, external_id=str(CHAT_ID)
        )
        assert user is not None
        conversation = await ConversationRepository(db_session).get_open_for_user(user.id)
        assert conversation is not None
        messages = await MessageRepository(db_session).list_recent(conversation.id, 10)

    assert [m.content for m in messages] == ["Salom"]
    assert len(await _queued_jobs(redis_pool)) == 1


async def test_operator_takeover_records_the_message_but_does_not_answer_it(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    telegram_channel: Any,
    redis_pool: ArqRedis,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    with as_tenant(seed.tenant_a.id):
        user = await UserRepository(db_session).create(
            channel_id=telegram_channel.id, external_id=str(CHAT_ID)
        )
        conversation = await ConversationRepository(db_session).create(
            user_id=user.id, status="open", is_bot_enabled=False
        )

    with caplog.at_level(logging.INFO, logger="app.api.telegram_webhook"):
        response = await _post(client, _update("yana savolim bor"))

    assert response.status_code == 200
    assert "telegram_bot_disabled_for_conversation" in caplog.text
    assert await _queued_jobs(redis_pool) == []

    with as_tenant(seed.tenant_a.id):
        messages = await MessageRepository(db_session).list_recent(conversation.id, 10)
    assert [m.content for m in messages] == ["yana savolim bor"]


# --- one tenant's bot cannot be reached through another's ---


async def test_two_clinics_bots_resolve_to_their_own_tenants(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    telegram_channel: Any,
    redis_pool: ArqRedis,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    with as_tenant(seed.tenant_b.id):
        await ChannelRepository(db_session).create(
            type=ChannelType.TELEGRAM,
            external_id="7000000000",
            credentials=encrypt("7000000000:token"),
            config={"webhook_secret": "clinic-b-secret"},
        )

    await _post(client, _update("A klinikaga", update_id=1, message_id=1))
    await _post(
        client,
        _update("B klinikaga", update_id=2, message_id=2),
        bot="7000000000",
        secret="clinic-b-secret",
    )

    jobs = await _queued_jobs(redis_pool)
    tenants = set()
    for job in jobs:
        info = await job.info()
        assert info is not None
        tenants.add(info.args[0])

    assert tenants == {str(seed.tenant_a.id), str(seed.tenant_b.id)}


async def test_clinic_bs_secret_does_not_open_clinic_as_webhook(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    seed: Seed,
    telegram_channel: Any,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """The secret is per-channel, so one clinic's leaking cannot be used to
    write into another's transcript.
    """
    with as_tenant(seed.tenant_b.id):
        await ChannelRepository(db_session).create(
            type=ChannelType.TELEGRAM,
            external_id="7000000000",
            credentials=encrypt("7000000000:token"),
            config={"webhook_secret": "clinic-b-secret"},
        )

    response = await _post(client, _update(), secret="clinic-b-secret")

    assert response.status_code == 403


# --- the adapter ---


async def test_the_adapter_routes_a_reply_over_its_business_connection(
    db_session: AsyncSession,
    seed: Seed,
    telegram_channel: Any,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """The whole reason reply_context exists: without the connection id the
    reply goes out from the bot account instead of the clinic's own.
    """
    fake = FakeTelegramClient()

    with as_tenant(seed.tenant_a.id):
        delivered_over = await send_reply(
            db_session,
            channel_id=telegram_channel.id,
            recipient_external_id=str(CHAT_ID),
            text="Va alaykum assalom!",
            last_user_message_at=datetime.now(UTC),
            reply_context={BUSINESS_CONNECTION_ID: "bc-77"},
            adapter=TelegramAdapter(client=fake),
        )

    assert delivered_over == ChannelType.TELEGRAM
    assert fake.calls == [("8123456789:AAbbCCddEEff", str(CHAT_ID), "Va alaykum assalom!", "bc-77")]


async def test_a_direct_chat_reply_carries_no_business_connection(
    db_session: AsyncSession,
    seed: Seed,
    telegram_channel: Any,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    fake = FakeTelegramClient()

    with as_tenant(seed.tenant_a.id):
        await send_reply(
            db_session,
            channel_id=telegram_channel.id,
            recipient_external_id=str(CHAT_ID),
            text="Salom",
            last_user_message_at=datetime.now(UTC),
            adapter=TelegramAdapter(client=fake),
        )

    assert fake.calls[0][3] is None


async def test_a_malformed_reply_context_is_ignored_rather_than_sent(
    db_session: AsyncSession,
    seed: Seed,
    telegram_channel: Any,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """reply_context is rebuilt from job arguments that originated in a
    webhook payload, so a wrong type must not reach the API as a malformed
    field.
    """
    fake = FakeTelegramClient()

    with as_tenant(seed.tenant_a.id):
        await send_reply(
            db_session,
            channel_id=telegram_channel.id,
            recipient_external_id=str(CHAT_ID),
            text="Salom",
            last_user_message_at=datetime.now(UTC),
            reply_context={BUSINESS_CONNECTION_ID: 12345},
            adapter=TelegramAdapter(client=fake),
        )

    assert fake.calls[0][3] is None


async def test_a_placeholder_token_blocks_delivery_instead_of_calling_telegram(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    fake = FakeTelegramClient()

    with as_tenant(seed.tenant_a.id):
        channel = await ChannelRepository(db_session).create(
            type=ChannelType.TELEGRAM,
            external_id="6000000000",
            credentials=encrypt("pending"),
            config={"webhook_secret": "s"},
        )
        delivered_over = await send_reply(
            db_session,
            channel_id=channel.id,
            recipient_external_id=str(CHAT_ID),
            text="Salom",
            last_user_message_at=datetime.now(UTC),
            adapter=TelegramAdapter(client=fake),
        )

    assert delivered_over is None
    assert fake.calls == []


def test_telegram_has_no_messaging_window_unlike_instagram() -> None:
    """Meta refuses a plain text send outside 24 hours; Telegram does not
    restrict replying in a chat the user opened. The difference is exactly
    why this check sits behind the adapter rather than in the shared worker.
    """
    long_ago = datetime(2020, 1, 1, tzinfo=UTC)

    assert (
        TelegramAdapter().delivery_block_reason(
            credentials="8123456789:real", last_user_message_at=long_ago
        )
        is None
    )
    assert (
        TelegramAdapter().delivery_block_reason(
            credentials="pending", last_user_message_at=datetime.now(UTC)
        )
        is DeliveryBlocked.NOT_CONFIGURED
    )


def test_placeholder_credentials_are_recognised() -> None:
    assert is_placeholder_credential("pending")
    assert is_placeholder_credential("  CHANGEME ")
    assert is_placeholder_credential("")
    assert not is_placeholder_credential("8123456789:AAbbCCddEEff")
