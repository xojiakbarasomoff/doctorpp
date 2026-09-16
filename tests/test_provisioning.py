import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.encryption import decrypt, encrypt
from app.core.passwords import verify_password
from app.core.provisioning import (
    _provision,
    _provision_operator,
    _provision_telegram,
    provision_channel_if_configured,
    provision_operator_if_configured,
    provision_telegram_if_configured,
)
from app.models.channel import Channel
from app.models.operator import Operator
from app.models.tenant import Tenant
from tests.conftest import isolated_settings

IG_ACCOUNT_ID = "37823824730565264"


def _settings(**overrides: object) -> Settings:
    return isolated_settings(**overrides)


async def _channel(session: AsyncSession) -> Channel | None:
    result = await session.execute(
        select(Channel).where(Channel.type == "instagram", Channel.external_id == IG_ACCOUNT_ID)
    )
    return result.scalar_one_or_none()


async def test_creates_tenant_and_channel_with_token(db_session: AsyncSession) -> None:
    await _provision(
        db_session,
        ig_account_id=IG_ACCOUNT_ID,
        tenant_name="Klinika",
        access_token="IGAA-real-token",
    )

    channel = await _channel(db_session)
    assert channel is not None
    assert channel.is_active is True
    # Stored encrypted, not in plaintext: the whole point of channel
    # credentials living behind app.core.encryption.
    assert channel.credentials != "IGAA-real-token"
    assert decrypt(channel.credentials) == "IGAA-real-token"

    tenant = await db_session.get(Tenant, channel.tenant_id)
    assert tenant is not None
    assert tenant.name == "Klinika"


async def test_without_token_seeds_a_placeholder(db_session: AsyncSession) -> None:
    """A channel seeded before the token is known must be distinguishable
    from one carrying a real credential -- app.channels.instagram.client
    skips sending on a placeholder instead of calling Meta with garbage.
    """
    await _provision(
        db_session, ig_account_id=IG_ACCOUNT_ID, tenant_name="Klinika", access_token=None
    )

    channel = await _channel(db_session)
    assert channel is not None
    assert decrypt(channel.credentials) == "pending"


async def test_rerun_does_not_create_a_second_tenant(db_session: AsyncSession) -> None:
    """Provisioning runs on every web boot, so a redeploy must not keep
    stacking up duplicate tenants for the same Instagram account.
    """
    for _ in range(3):
        await _provision(
            db_session,
            ig_account_id=IG_ACCOUNT_ID,
            tenant_name="Klinika",
            access_token="IGAA-real-token",
        )

    tenants = (await db_session.execute(select(Tenant))).scalars().all()
    channels = (await db_session.execute(select(Channel))).scalars().all()
    assert len(tenants) == 1
    assert len(channels) == 1


async def test_placeholder_credential_is_upgraded_to_a_real_token(
    db_session: AsyncSession,
) -> None:
    """The recovery path that matters: a channel seeded without a token and
    left alone would never be able to reply, and would never say so.
    """
    await _provision(
        db_session, ig_account_id=IG_ACCOUNT_ID, tenant_name="Klinika", access_token=None
    )
    await _provision(
        db_session,
        ig_account_id=IG_ACCOUNT_ID,
        tenant_name="Klinika",
        access_token="IGAA-real-token",
    )

    channel = await _channel(db_session)
    assert channel is not None
    assert decrypt(channel.credentials) == "IGAA-real-token"


async def test_existing_real_token_is_not_overwritten(db_session: AsyncSession) -> None:
    """Provisioning must never clobber a credential someone set deliberately
    (scripts/set_channel_credentials.py, or a token rotated by hand) with a
    stale value still sitting in the deployment's environment.
    """
    tenant = Tenant(name="Klinika", status="active")
    db_session.add(tenant)
    await db_session.flush()
    db_session.add(
        Channel(
            tenant_id=tenant.id,
            type="instagram",
            external_id=IG_ACCOUNT_ID,
            credentials=encrypt("IGAA-token-set-by-hand"),
            is_active=True,
        )
    )
    await db_session.commit()

    await _provision(
        db_session,
        ig_account_id=IG_ACCOUNT_ID,
        tenant_name="Different Name",
        access_token="IGAA-stale-env-token",
    )

    channel = await _channel(db_session)
    assert channel is not None
    assert decrypt(channel.credentials) == "IGAA-token-set-by-hand"


async def test_unconfigured_settings_provision_nothing(db_session: AsyncSession) -> None:
    """The default must be "do nothing": an unset variable is not an
    instruction to invent a tenant.

    Asserts the behaviour rather than the field defaults, because Settings
    falls back to the real environment for anything not passed in -- a
    defaults assertion would fail on exactly the machines where these
    variables are set, which includes every machine running this deployment.
    """
    await provision_channel_if_configured(
        _settings(provision_ig_account_id=None, provision_tenant_name=None)
    )
    assert (await db_session.execute(select(Tenant))).scalars().all() == []


# --- helpers for the steps below ---

BOT_ID = "8123456789"
BOT_TOKEN = f"{BOT_ID}:AAbbCCddEEff"


async def _operator(session: AsyncSession, username: str) -> Operator | None:
    result = await session.execute(select(Operator).where(Operator.username == username))
    return result.scalar_one_or_none()


async def _telegram_channel(session: AsyncSession) -> Channel | None:
    result = await session.execute(
        select(Channel).where(Channel.type == "telegram", Channel.external_id == BOT_ID)
    )
    return result.scalar_one_or_none()


@asynccontextmanager
async def _fake_bot_api(
    calls: list[tuple[str, dict]], *, get_me_ok: bool = True
) -> AsyncIterator[httpx.AsyncClient]:
    """A stand-in for api.telegram.org that records what it was asked.

    Every response is HTTP 200 — including the failures. That is how the Bot
    API actually behaves, and asserting against a transport that gets it
    right is the only way these tests can prove the `ok` field is what gets
    checked.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        payload = json.loads(request.content or b"{}")
        calls.append((method, payload))
        if method == "getMe":
            if not get_me_ok:
                return httpx.Response(200, json={"ok": False, "description": "Unauthorized"})
            return httpx.Response(
                200, json={"ok": True, "result": {"id": int(BOT_ID), "username": "clinic_bot"}}
            )
        return httpx.Response(200, json={"ok": True, "result": True})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport, base_url="https://api.telegram.org"
    ) as client:
        yield client


# --- the first dashboard login ---


async def test_creates_the_first_operator(db_session: AsyncSession) -> None:
    """Without this a deployment on a host whose database is only reachable
    from inside the cluster has no way into its own dashboard at all.
    """
    await _provision_operator(
        db_session,
        tenant_name="Klinika",
        username="admin",
        password="a-real-password",
        name="Administrator",
        role="operator",
    )

    operator = await _operator(db_session, "admin")
    assert operator is not None
    assert operator.role == "operator"
    # Hashed, never stored in a form anything can read back.
    assert operator.password_hash != "a-real-password"
    assert verify_password("a-real-password", operator.password_hash)


async def test_the_operator_lands_on_the_same_tenant_as_the_channel(
    db_session: AsyncSession,
) -> None:
    """Two provisioning steps, one clinic. An operator on a tenant of its own
    would log in to a dashboard showing none of the conversations the bot is
    actually having.
    """
    await _provision(
        db_session, ig_account_id=IG_ACCOUNT_ID, tenant_name="Klinika", access_token=None
    )
    await _provision_operator(
        db_session,
        tenant_name="Klinika",
        username="admin",
        password="a-real-password",
        name="Administrator",
        role="operator",
    )

    channel = await _channel(db_session)
    operator = await _operator(db_session, "admin")
    assert channel is not None and operator is not None
    assert operator.tenant_id == channel.tenant_id
    assert len((await db_session.execute(select(Tenant))).scalars().all()) == 1


async def test_an_existing_operators_password_is_left_alone(db_session: AsyncSession) -> None:
    """The variables stay set in the host's environment long after the first
    boot. Re-hashing on every deploy would silently undo a password the
    operator changed through the dashboard, and put the old one back in reach
    of anyone who can read the host's config.
    """
    await _provision_operator(
        db_session,
        tenant_name="Klinika",
        username="admin",
        password="the-first-password",
        name="Administrator",
        role="operator",
    )
    await _provision_operator(
        db_session,
        tenant_name="Klinika",
        username="admin",
        password="a-stale-env-password",
        name="Someone Else",
        role="doctor",
    )

    operator = await _operator(db_session, "admin")
    assert operator is not None
    assert verify_password("the-first-password", operator.password_hash)
    assert operator.role == "operator"


async def test_a_short_bootstrap_password_is_refused(db_session: AsyncSession) -> None:
    """The first account is the one with nothing else guarding it, and the
    bot this replaces shipped with a hardcoded admin/admin.
    """
    with pytest.raises(ValueError, match="at least"):
        await _provision_operator(
            db_session,
            tenant_name="Klinika",
            username="admin",
            password="admin",
            name="Administrator",
            role="operator",
        )

    assert await _operator(db_session, "admin") is None


async def test_operator_provisioning_is_skipped_when_unconfigured(
    db_session: AsyncSession,
) -> None:
    await provision_operator_if_configured(
        _settings(
            provision_tenant_name="Klinika",
            provision_operator_username=None,
            provision_operator_password=None,
        )
    )

    assert (await db_session.execute(select(Operator))).scalars().all() == []


async def test_a_failing_operator_step_does_not_stop_startup(
    db_session: AsyncSession, caplog: pytest.LogCaptureFixture
) -> None:
    """Never fatal: refusing to boot over a seeding problem drops webhook
    traffic that cannot be recovered, to fix one that can.
    """
    with caplog.at_level(logging.ERROR, logger="app"):
        await provision_operator_if_configured(
            _settings(
                provision_tenant_name="Klinika",
                provision_operator_username="admin",
                provision_operator_password="short",
            )
        )

    assert "provisioning_failed" in caplog.text
    assert (await db_session.execute(select(Operator))).scalars().all() == []


async def test_the_bootstrap_password_is_never_logged(
    db_session: AsyncSession, caplog: pytest.LogCaptureFixture
) -> None:
    password = "a-very-distinctive-password"

    with caplog.at_level(logging.DEBUG, logger="app"):
        await provision_operator_if_configured(
            _settings(
                provision_tenant_name="Klinika",
                provision_operator_username="admin",
                provision_operator_password=password,
            )
        )

    assert password not in caplog.text


# --- the Telegram channel and its webhook ---


async def test_creates_the_telegram_channel_and_registers_the_webhook(
    db_session: AsyncSession,
) -> None:
    calls: list[tuple[str, dict]] = []

    async with _fake_bot_api(calls) as client:
        await _provision_telegram(
            db_session,
            client,
            token=BOT_TOKEN,
            tenant_name="Klinika",
            public_base_url="https://clinic.example.com",
        )

    channel = await _telegram_channel(db_session)
    assert channel is not None
    assert decrypt(channel.credentials) == BOT_TOKEN

    assert [method for method, _ in calls] == ["getMe", "setWebhook"]
    set_webhook = calls[1][1]
    assert set_webhook["url"] == f"https://clinic.example.com/webhook/telegram/{BOT_ID}"
    # The secret the endpoint checks on every delivery has to be the one
    # Telegram was told to sign with, or the endpoint refuses everything.
    assert set_webhook["secret_token"] == channel.config["webhook_secret"]


async def test_a_trailing_slash_does_not_double_up_in_the_webhook_url(
    db_session: AsyncSession,
) -> None:
    calls: list[tuple[str, dict]] = []

    async with _fake_bot_api(calls) as client:
        await _provision_telegram(
            db_session,
            client,
            token=BOT_TOKEN,
            tenant_name="Klinika",
            public_base_url="https://clinic.example.com/",
        )

    assert calls[1][1]["url"] == f"https://clinic.example.com/webhook/telegram/{BOT_ID}"


async def test_rerunning_keeps_the_webhook_secret(db_session: AsyncSession) -> None:
    """Rotating the secret on every boot would leave a window where Telegram
    is still signing with the old one and the endpoint rejects every delivery
    in it.
    """
    calls: list[tuple[str, dict]] = []

    async with _fake_bot_api(calls) as client:
        for _ in range(2):
            await _provision_telegram(
                db_session,
                client,
                token=BOT_TOKEN,
                tenant_name="Klinika",
                public_base_url="https://clinic.example.com",
            )

    channels = (await db_session.execute(select(Channel))).scalars().all()
    assert len(channels) == 1
    assert calls[1][1]["secret_token"] == calls[3][1]["secret_token"]


async def test_rerunning_does_not_drop_pending_updates(db_session: AsyncSession) -> None:
    """This runs on every boot, unlike the one-off script. Dropping the
    backlog on a redeploy would throw away messages patients sent while the
    new version was rolling out.
    """
    calls: list[tuple[str, dict]] = []

    async with _fake_bot_api(calls) as client:
        await _provision_telegram(
            db_session,
            client,
            token=BOT_TOKEN,
            tenant_name="Klinika",
            public_base_url="https://clinic.example.com",
        )

    assert calls[1][1]["drop_pending_updates"] is False


async def test_a_rotated_bot_token_replaces_the_stored_one(db_session: AsyncSession) -> None:
    calls: list[tuple[str, dict]] = []
    rotated = f"{BOT_ID}:BBccDDeeFF-rotated"

    async with _fake_bot_api(calls) as client:
        await _provision_telegram(
            db_session,
            client,
            token=BOT_TOKEN,
            tenant_name="Klinika",
            public_base_url="https://clinic.example.com",
        )
        await _provision_telegram(
            db_session,
            client,
            token=rotated,
            tenant_name="Klinika",
            public_base_url="https://clinic.example.com",
        )

    channel = await _telegram_channel(db_session)
    assert channel is not None
    assert decrypt(channel.credentials) == rotated


async def test_a_rejected_token_creates_no_channel(db_session: AsyncSession) -> None:
    """The Bot API answers a bad token with HTTP 200 and ok:false, so a
    channel row created before that answer was read would look configured and
    never work.
    """
    async with _fake_bot_api([], get_me_ok=False) as client:
        with pytest.raises(RuntimeError, match="Unauthorized"):
            await _provision_telegram(
                db_session,
                client,
                token=BOT_TOKEN,
                tenant_name="Klinika",
                public_base_url="https://clinic.example.com",
            )

    assert await _telegram_channel(db_session) is None


async def test_a_malformed_token_is_reported_as_such(db_session: AsyncSession) -> None:
    async with _fake_bot_api([]) as client:
        with pytest.raises(ValueError, match="<id>:<secret>"):
            await _provision_telegram(
                db_session,
                client,
                token="not-a-token",
                tenant_name="Klinika",
                public_base_url="https://clinic.example.com",
            )


async def test_telegram_provisioning_needs_an_https_base_url(
    db_session: AsyncSession, caplog: pytest.LogCaptureFixture
) -> None:
    """Telegram refuses a plaintext webhook outright. Saying so here beats
    letting setWebhook report it as a failed channel setup.
    """
    with caplog.at_level(logging.ERROR, logger="app"):
        await provision_telegram_if_configured(
            _settings(
                provision_tenant_name="Klinika",
                provision_telegram_bot_token=BOT_TOKEN,
                public_base_url="http://clinic.example.com",
            )
        )

    assert "public_base_url_not_https" in caplog.text
    assert await _telegram_channel(db_session) is None


async def test_telegram_provisioning_is_skipped_when_unconfigured(
    db_session: AsyncSession,
) -> None:
    await provision_telegram_if_configured(
        _settings(provision_tenant_name="Klinika", provision_telegram_bot_token=None)
    )

    assert (await db_session.execute(select(Channel))).scalars().all() == []
