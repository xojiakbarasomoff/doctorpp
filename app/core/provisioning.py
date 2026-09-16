"""First-run provisioning of everything a deployment needs to be usable:
the tenant, its Instagram and Telegram channels, and the first dashboard
login.

The scripts under scripts/ do the same things from an operator's shell, and
stay the right tool wherever one is available. This exists because on a
managed host the database is reachable only from inside the cluster's private
network: the sole process that can write these rows is the application itself,
so provisioning has to be something the app can do on its own behalf, driven
by configuration.

Runs on web startup only (app.main's lifespan), never in the worker, so two
processes booting from the same image cannot race to insert the same row.
Every step is idempotent and none of them is fatal -- a deployment that
cannot provision must still serve, because the webhook traffic it drops
while refusing to start is not recoverable and the provisioning problem is.

Four independent switches, each armed by setting its variables and safely
left armed afterwards:

* PROVISION_TENANT_NAME -- the clinic every other step attaches to. Nothing
  below happens without it.
* PROVISION_IG_ACCOUNT_ID (+ ACCESS_TOKEN) -- the Instagram channel.
* PROVISION_TELEGRAM_BOT_TOKEN (+ PUBLIC_BASE_URL) -- the Telegram channel,
  including registering the webhook with Telegram.
* PROVISION_OPERATOR_USERNAME + PROVISION_OPERATOR_PASSWORD -- the first
  dashboard login, without which a freshly deployed dashboard cannot be
  opened at all.
"""

import asyncio
import logging
import secrets
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.channels.base import ChannelType
from app.channels.instagram.client import is_placeholder_credential
from app.channels.telegram.client import BOT_API_BASE_URL
from app.core.config import Settings
from app.core.db import db_session
from app.core.encryption import decrypt, encrypt
from app.core.passwords import MIN_PASSWORD_LENGTH, hash_password
from app.models.channel import Channel
from app.models.operator import Operator
from app.models.tenant import Tenant

logger = logging.getLogger(__name__)

CHANNEL_TYPE = ChannelType.INSTAGRAM

# Where app.api.telegram_webhook looks for the per-channel secret it checks
# on every delivery. Same key scripts/setup_telegram_channel.py writes.
WEBHOOK_SECRET_KEY = "webhook_secret"

_WEBHOOK_SECRET_BYTES = 32

# What a channel's credentials hold until a real token is supplied.
# app.channels.instagram.client reads it as "not configured yet" and skips
# sending, rather than calling Meta with a value that cannot work.
_PLACEHOLDER = "pending"

# Long enough for a healthy database on the same private network, short
# enough that an unhealthy one does not keep the web process from serving.
_STARTUP_TIMEOUT_SECONDS = 15.0

# Telegram provisioning adds two round-trips to api.telegram.org on top of
# the database work, so it gets its own, longer bound. Still bounded: this
# runs before the port opens, and an unreachable Telegram is not a reason to
# hold a deployment closed.
_TELEGRAM_TIMEOUT_SECONDS = 30.0


async def _ensure_tenant(session: AsyncSession, name: str) -> Tenant:
    """The clinic this deployment serves, created on first boot.

    Looked up by name so the Instagram channel, the Telegram channel and the
    first operator all land on one tenant instead of each inventing its own.
    On the (misconfigured) case of several tenants sharing a name, the oldest
    wins -- deterministic beats arbitrary, and re-running must not walk a
    deployment across different clinics from boot to boot.
    """
    existing = (
        (
            await session.execute(
                select(Tenant).where(Tenant.name == name)
                # created_at then id: rows inserted in one transaction share
                # a now(), and id breaks that tie so the result cannot vary
                # between boots.
                .order_by(Tenant.created_at, Tenant.id)
            )
        )
        .scalars()
        .first()
    )
    if existing is not None:
        return existing

    tenant = Tenant(name=name, status="active")
    session.add(tenant)
    await session.flush()
    return tenant


def _credential_plaintext(channel: Channel) -> str:
    """The channel's credential in plaintext, or the placeholder when it
    cannot be read. A credential encrypted under a rotated-away key is not a
    reason to fail startup -- it is a reason to treat the channel as
    unconfigured, which is exactly what the placeholder means.
    """
    try:
        return decrypt(channel.credentials)
    except Exception:  # noqa: BLE001 - any decrypt failure means "unreadable"
        return _PLACEHOLDER


async def _provision(
    session: AsyncSession,
    *,
    ig_account_id: str,
    tenant_name: str,
    access_token: str | None,
) -> None:
    result = await session.execute(
        select(Channel).where(Channel.type == CHANNEL_TYPE, Channel.external_id == ig_account_id)
    )
    channel = result.scalar_one_or_none()

    if channel is None:
        # Resolved by name rather than always created, so a deployment that
        # provisions a Telegram channel and an Instagram one ends up with a
        # single clinic rather than two that each see half the patients.
        tenant = await _ensure_tenant(session, tenant_name)
        session.add(
            Channel(
                tenant_id=tenant.id,
                type=CHANNEL_TYPE,
                external_id=ig_account_id,
                credentials=encrypt(access_token or _PLACEHOLDER),
                is_active=True,
            )
        )
        await session.commit()
        # WARNING, not INFO: "this deployment created its tenant" is a line
        # worth finding later without having to widen a log filter.
        logger.warning(
            "provisioned_channel tenant_id=%s ig_account_id=%s has_token=%s",
            tenant.id,
            ig_account_id,
            bool(access_token),
        )
        return

    # Already provisioned. The one thing still worth doing is replacing a
    # placeholder credential with a real token: a channel seeded before the
    # token existed would otherwise stay permanently unable to reply, and
    # silently, since the client skips placeholder credentials rather than
    # raising.
    if access_token and is_placeholder_credential(_credential_plaintext(channel)):
        channel.credentials = encrypt(access_token)
        await session.commit()
        logger.warning(
            "provisioned_channel_token_filled channel_id=%s ig_account_id=%s",
            channel.id,
            ig_account_id,
        )
        return

    logger.warning(
        "provisioning_noop_channel_exists channel_id=%s ig_account_id=%s",
        channel.id,
        ig_account_id,
    )


async def _bot_api(client: httpx.AsyncClient, token: str, method: str, **payload: Any) -> Any:
    """One Bot API call, raising on anything that is not a success.

    The Bot API answers a rejected call with HTTP 200 and `ok: false`, so the
    status code alone proves nothing -- same check app.channels.telegram.client
    makes. The description ("Unauthorized", "Bad webhook: HTTPS url must be
    provided") names the problem and carries no credential, so it is safe to
    put in the message.
    """
    response = await client.post(f"/bot{token}/{method}", json=payload)
    body = response.json()
    if not body.get("ok"):
        raise RuntimeError(f"Telegram {method} failed: {body.get('description')}")
    return body["result"]


def _bot_id_from_token(token: str) -> str:
    """Telegram issues tokens as "<bot_id>:<secret>"; the id half is public
    and is what the webhook path carries.
    """
    bot_id, separator, _ = token.partition(":")
    if not separator or not bot_id.isdigit():
        raise ValueError("PROVISION_TELEGRAM_BOT_TOKEN is not a Telegram bot token (<id>:<secret>)")
    return bot_id


async def _provision_telegram(
    session: AsyncSession,
    client: httpx.AsyncClient,
    *,
    token: str,
    tenant_name: str,
    public_base_url: str,
) -> None:
    """Create or refresh the Telegram channel, and point Telegram at us.

    getMe first, so a mistyped token fails here with "Unauthorized" instead
    of later as updates that resolve to no channel.
    """
    bot_id = _bot_id_from_token(token)
    me = await _bot_api(client, token, "getMe")
    if str(me.get("id")) != bot_id:  # pragma: no cover - defensive
        raise RuntimeError("The token's bot id does not match getMe")

    found = await session.execute(
        select(Channel).where(Channel.type == ChannelType.TELEGRAM, Channel.external_id == bot_id)
    )
    channel = found.scalar_one_or_none()

    if channel is None:
        tenant = await _ensure_tenant(session, tenant_name)
        secret = secrets.token_urlsafe(_WEBHOOK_SECRET_BYTES)
        channel = Channel(
            tenant_id=tenant.id,
            type=ChannelType.TELEGRAM,
            external_id=bot_id,
            credentials=encrypt(token),
            config={WEBHOOK_SECRET_KEY: secret},
            is_active=True,
        )
        session.add(channel)
        await session.commit()
        logger.warning(
            "provisioned_telegram_channel tenant_id=%s bot_id=%s username=%s",
            tenant.id,
            bot_id,
            me.get("username"),
        )
    else:
        # Kept, not regenerated. Rotating the secret on every boot would
        # leave a window where Telegram is still signing with the old one
        # and app.api.telegram_webhook rejects every delivery in it.
        secret = str(channel.config.get(WEBHOOK_SECRET_KEY) or "")
        if not secret:
            secret = secrets.token_urlsafe(_WEBHOOK_SECRET_BYTES)
            channel.config = {**channel.config, WEBHOOK_SECRET_KEY: secret}
        # The token is refreshed: unlike the Instagram path there is no
        # "set by hand" case to protect, and a rotated @BotFather token has
        # to be able to reach the row somehow.
        channel.credentials = encrypt(token)
        channel.is_active = True
        await session.commit()
        logger.warning("provisioned_telegram_channel_refreshed bot_id=%s", bot_id)

    webhook_url = f"{public_base_url.rstrip('/')}/webhook/telegram/{bot_id}"
    await _bot_api(
        client,
        token,
        "setWebhook",
        url=webhook_url,
        secret_token=secret,
        # Everything else this pipeline ignores anyway; asking for less keeps
        # Telegram from spending retries on updates dropped on arrival.
        allowed_updates=["message", "business_message"],
        # False, unlike the one-off script: this runs on every boot, and
        # dropping the backlog on a redeploy would silently throw away
        # messages patients sent while the new version was rolling out.
        drop_pending_updates=False,
    )
    logger.warning("provisioned_telegram_webhook url=%s", webhook_url)


async def _provision_operator(
    session: AsyncSession,
    *,
    tenant_name: str,
    username: str,
    password: str,
    name: str,
    role: str,
) -> None:
    """Create the first dashboard login, once.

    Deliberately does *not* reset the password of an account that already
    exists, which is the one place this differs from
    scripts/create_operator.py. The variables stay set in the host's
    environment long after the first boot, so resetting would silently undo
    a password the operator changed through the dashboard, on every deploy,
    and put the old one back in reach of anyone who can read the host's
    config.
    """
    existing = (
        (await session.execute(select(Operator).where(Operator.username == username)))
        .scalars()
        .one_or_none()
    )
    if existing is not None:
        logger.warning("provisioning_noop_operator_exists username=%s", username)
        return

    if len(password) < MIN_PASSWORD_LENGTH:
        # Refused rather than truncated or accepted: the first account is
        # the one with nothing else guarding it, and the bot this replaces
        # shipped with a hardcoded admin/admin.
        raise ValueError(
            f"PROVISION_OPERATOR_PASSWORD must be at least {MIN_PASSWORD_LENGTH} characters"
        )

    tenant = await _ensure_tenant(session, tenant_name)
    session.add(
        Operator(
            tenant_id=tenant.id,
            name=name,
            role=role,
            username=username,
            password_hash=hash_password(password),
        )
    )
    await session.commit()
    logger.warning(
        "provisioned_operator tenant_id=%s username=%s role=%s", tenant.id, username, role
    )


async def _guarded(step: str, run: Any, *, timeout: float = _STARTUP_TIMEOUT_SECONDS) -> None:
    """Run one provisioning step, bounded and never fatal.

    Bounded because this sits in the startup path: an unreachable database
    would otherwise hold the port closed for asyncpg's own connect timeout,
    failing the platform's health check and rolling the deploy back -- the
    opposite of what "never fatal" is for.
    """
    try:
        await asyncio.wait_for(run(), timeout=timeout)
    except TimeoutError:
        logger.error("provisioning_timed_out step=%s seconds=%s", step, timeout)
    except Exception:
        logger.exception("provisioning_failed step=%s", step)


async def provision_channel_if_configured(settings: Settings) -> None:
    """Create the configured tenant/Instagram channel when it is missing.

    No-op when unconfigured, and never fatal: a provisioning failure must not
    stop the web process from serving the webhook, since continuing to accept
    Meta's deliveries is worth more than refusing to start over a seeding
    problem someone can fix afterwards.
    """
    ig_account_id = settings.provision_ig_account_id
    tenant_name = settings.provision_tenant_name
    if ig_account_id is None or tenant_name is None:
        return

    async def _run() -> None:
        async with db_session() as session:
            await _provision(
                session,
                ig_account_id=ig_account_id,
                tenant_name=tenant_name,
                access_token=settings.access_token,
            )

    await _guarded(f"instagram:{ig_account_id}", _run)


async def provision_telegram_if_configured(settings: Settings) -> None:
    """Register the configured Telegram bot and its webhook.

    Needs PUBLIC_BASE_URL as well as the token: a bot with a channel row but
    no webhook registered receives nothing, and the container has no way to
    learn the URL the outside world reaches it by.
    """
    token = settings.provision_telegram_bot_token
    tenant_name = settings.provision_tenant_name
    base_url = settings.public_base_url
    if token is None or tenant_name is None:
        return
    if base_url is None or not base_url.startswith("https://"):
        # Telegram refuses a plaintext webhook outright, so saying so here
        # beats letting setWebhook report it as a failed channel setup.
        logger.error("provisioning_skipped_telegram reason=public_base_url_not_https")
        return

    async def _run() -> None:
        async with (
            httpx.AsyncClient(base_url=BOT_API_BASE_URL, timeout=10.0) as client,
            db_session() as session,
        ):
            await _provision_telegram(
                session,
                client,
                token=token,
                tenant_name=tenant_name,
                public_base_url=base_url,
            )

    await _guarded("telegram", _run, timeout=_TELEGRAM_TIMEOUT_SECONDS)


async def provision_operator_if_configured(settings: Settings) -> None:
    """Create the first dashboard login when it is missing."""
    username = settings.provision_operator_username
    password = settings.provision_operator_password
    tenant_name = settings.provision_tenant_name
    if username is None or password is None or tenant_name is None:
        return

    async def _run() -> None:
        async with db_session() as session:
            await _provision_operator(
                session,
                tenant_name=tenant_name,
                username=username,
                password=password,
                name=settings.provision_operator_name,
                role=settings.provision_operator_role,
            )

    # The username, never the password -- not even its length.
    await _guarded(f"operator:{username}", _run)


async def provision_if_configured(settings: Settings) -> None:
    """Every provisioning step, in dependency order.

    Instagram first because it is the step that has always created the
    tenant, then Telegram, then the operator -- so a deployment configured
    for all three ends up with one clinic rather than three.
    """
    await provision_channel_if_configured(settings)
    await provision_telegram_if_configured(settings)
    await provision_operator_if_configured(settings)
