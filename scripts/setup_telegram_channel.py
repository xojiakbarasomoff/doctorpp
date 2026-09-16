r"""Register a clinic's Telegram bot: create its channel row and point
Telegram's webhook at this deployment.

Run from your own terminal. The bot token is read from an environment
variable you set yourself, so it never appears in chat, git history, shell
history as a command-line argument, or any log line here.

What it does, in order:

1. Calls getMe to check the token actually works, and to learn the bot's own
   id — that id is what the webhook path carries, so a typo is caught here
   rather than showing up later as updates that resolve to no channel.
2. Creates (or updates) the channel row for this tenant, with the token
   encrypted and a webhook secret in Channel.config.
3. Calls setWebhook so Telegram starts delivering to this deployment, with
   that same secret. The secret is what app.api.telegram_webhook verifies on
   every delivery; without it the endpoint refuses everything, which is
   deliberate — an endpoint that accepts unauthenticated updates lets anyone
   write into a clinic's patient transcript.

Usage (PowerShell), from the instagram/ directory:

    $env:TELEGRAM_BOT_TOKEN = "<paste the token from @BotFather>"
    $env:PUBLIC_BASE_URL = "https://your-deployment.example.com"
    $env:TENANT_NAME = "Smile Dental"       # or $env:TENANT_ID = "<uuid>"
    ./../.venv/Scripts/python.exe scripts/setup_telegram_channel.py
    Remove-Item Env:\TELEGRAM_BOT_TOKEN

A webhook secret is generated for you unless TELEGRAM_WEBHOOK_SECRET is set.
Re-running is safe: an existing channel has its token, secret and webhook
refreshed rather than being duplicated.
"""

import asyncio
import os
import secrets
import sys
import uuid

import httpx
from sqlalchemy import select

from app.channels.base import ChannelType
from app.channels.telegram.client import BOT_API_BASE_URL
from app.core.db import db_session
from app.core.encryption import encrypt
from app.models.channel import Channel
from app.models.tenant import Tenant

WEBHOOK_SECRET_KEY = "webhook_secret"

# Telegram accepts letters, digits, underscore and hyphen, 1-256 characters.
_SECRET_BYTES = 32


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"{name} is not set. See this script's docstring for usage.")
    return value


def _bot_id_from_token(token: str) -> str:
    """Telegram issues tokens as "<bot_id>:<secret>"; the id half is public.

    Taken from the token rather than only from getMe so a malformed token is
    reported as such instead of as a confusing API error.
    """
    bot_id, separator, _ = token.partition(":")
    if not separator or not bot_id.isdigit():
        sys.exit("TELEGRAM_BOT_TOKEN does not look like a Telegram bot token (<id>:<secret>).")
    return bot_id


async def _call(client: httpx.AsyncClient, token: str, method: str, **payload: object) -> dict:
    response = await client.post(f"/bot{token}/{method}", json=payload)
    try:
        body = response.json()
    except ValueError:
        sys.exit(f"{method} returned a non-JSON response (HTTP {response.status_code}).")
    if not body.get("ok"):
        # The description names the problem ("Unauthorized", "Bad webhook:
        # HTTPS url must be provided") and carries no credential.
        sys.exit(f"{method} failed: {body.get('description')}")
    return body


async def _resolve_tenant(session, name: str | None, tenant_id: str | None) -> Tenant:
    if tenant_id:
        tenant = await session.get(Tenant, uuid.UUID(tenant_id))
        if tenant is None:
            sys.exit(f"No tenant with id {tenant_id}.")
        return tenant

    assert name is not None
    existing = (await session.execute(select(Tenant).where(Tenant.name == name))).scalars().all()
    if len(existing) > 1:
        sys.exit(f"{len(existing)} tenants are named {name!r}. Set TENANT_ID instead.")
    if existing:
        return existing[0]

    tenant = Tenant(name=name, status="active")
    session.add(tenant)
    await session.flush()
    print(f"Created tenant {tenant.name!r} ({tenant.id}).")
    return tenant


async def main() -> None:
    token = _require_env("TELEGRAM_BOT_TOKEN")
    base_url = _require_env("PUBLIC_BASE_URL").rstrip("/")
    tenant_name = os.environ.get("TENANT_NAME", "").strip() or None
    tenant_id = os.environ.get("TENANT_ID", "").strip() or None
    if not tenant_name and not tenant_id:
        sys.exit("Set either TENANT_NAME or TENANT_ID.")

    if not base_url.startswith("https://"):
        # Telegram refuses a plaintext webhook outright; failing here says so
        # in one line instead of leaving you to read setWebhook's error.
        sys.exit("PUBLIC_BASE_URL must be an https:// URL — Telegram will not call http://.")

    bot_id = _bot_id_from_token(token)
    secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "").strip() or secrets.token_urlsafe(
        _SECRET_BYTES
    )

    async with httpx.AsyncClient(base_url=BOT_API_BASE_URL, timeout=15.0) as client:
        me = await _call(client, token, "getMe")
        username = me["result"].get("username")
        if str(me["result"].get("id")) != bot_id:  # pragma: no cover - defensive
            sys.exit("The token's bot id does not match getMe. Refusing to continue.")
        print(f"Token is valid for @{username} (id {bot_id}).")

        async with db_session() as session:
            tenant = await _resolve_tenant(session, tenant_name, tenant_id)

            found = await session.execute(
                select(Channel).where(
                    Channel.type == ChannelType.TELEGRAM, Channel.external_id == bot_id
                )
            )
            channel = found.scalar_one_or_none()

            if channel is None:
                channel = Channel(
                    tenant_id=tenant.id,
                    type=ChannelType.TELEGRAM,
                    external_id=bot_id,
                    credentials=encrypt(token),
                    config={WEBHOOK_SECRET_KEY: secret},
                    is_active=True,
                )
                session.add(channel)
                action = "Created"
            else:
                if channel.tenant_id != tenant.id:
                    sys.exit(
                        f"Bot {bot_id} already belongs to another tenant ({channel.tenant_id}). "
                        "Refusing to move it."
                    )
                channel.credentials = encrypt(token)
                channel.config = {**channel.config, WEBHOOK_SECRET_KEY: secret}
                channel.is_active = True
                action = "Updated"

            await session.commit()
            print(f"{action} the Telegram channel for tenant {tenant.name!r} ({channel.id}).")

        webhook_url = f"{base_url}/webhook/telegram/{bot_id}"
        await _call(
            client,
            token,
            "setWebhook",
            url=webhook_url,
            secret_token=secret,
            # Everything else this pipeline ignores anyway; asking for less
            # keeps Telegram from spending retries on updates that are
            # dropped on arrival.
            allowed_updates=["message", "business_message"],
            drop_pending_updates=True,
        )
        print(f"Webhook set to {webhook_url}")

    print(
        "\nDone. The webhook secret is stored in the channel row — it is not printed here, "
        "and nothing needs it in your environment."
    )


if __name__ == "__main__":
    asyncio.run(main())
