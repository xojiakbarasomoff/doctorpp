"""Keeping Telegram pointed at this deployment.

A webhook is registered once, at boot, by app.core.provisioning -- and after
that nothing checks it. That would be fine if the registration were ours
alone to change, but it is not: it lives in Telegram's account for the bot
token, and *any* process holding that token can move it or clear it. Calling
deleteWebhook, or starting a long-poller, is enough. That is not hypothetical
here; the repository still carries an older polling bot for the same token,
and a single run of it takes the clinic's Telegram offline until the next
redeploy.

The failure has no symptom on our side. Telegram simply stops delivering, the
logs stay clean because nothing arrives to log, and the health check passes.
The clinic discovers it when a patient mentions the bot never answered.

So the registration is treated as drifting state to be reconciled rather than
a step that happened once: read what Telegram currently has, compare it with
where this deployment lives, and put it back when they disagree. Read-only in
the common case -- getWebhookInfo costs one call and changes nothing, so this
can run often enough to keep an outage down to minutes.

The secret is not re-rolled when re-registering. The one in Channel.config is
what app.api.telegram_webhook validates against, and a fresh one would only
mean rejecting the deliveries this job exists to restore.
"""

import logging
from collections.abc import Sequence

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.channels.base import ChannelType
from app.channels.telegram.client import BOT_API_BASE_URL, is_placeholder_credential
from app.core.encryption import decrypt
from app.models.channel import Channel

logger = logging.getLogger(__name__)

# Same key app.core.provisioning writes and app.api.telegram_webhook reads.
WEBHOOK_SECRET_KEY = "webhook_secret"

# Matches what provisioning asks for. Registering the same URL with a
# different set would make every run look like drift and re-register forever.
ALLOWED_UPDATES = ["message", "business_message"]

_TIMEOUT_SECONDS = 20.0


class WebhookWatchRun:
    """What one pass found, for the caller to log."""

    def __init__(self) -> None:
        self.ok = 0
        self.repaired = 0
        self.skipped = 0
        self.failed = 0

    @property
    def notable(self) -> bool:
        """Whether the run is worth a log line.

        A pass that finds every webhook already correct is the expected case
        and says nothing new, so it stays silent.
        """
        return bool(self.repaired or self.failed)


def webhook_url(public_base_url: str, bot_id: str) -> str:
    """Where Telegram should be delivering this bot's updates.

    One definition, used both to compare and to re-register, so the check
    cannot drift from the repair.
    """
    return f"{public_base_url.rstrip('/')}/webhook/telegram/{bot_id}"


async def _telegram_channels(session: AsyncSession) -> Sequence[Channel]:
    """Every Telegram channel, across every tenant: this runs on a schedule,
    with no request and no operator to scope it to.
    """
    stmt = select(Channel).where(Channel.type == ChannelType.TELEGRAM, Channel.is_active.is_(True))
    return (await session.execute(stmt)).scalars().all()


async def _call(client: httpx.AsyncClient, token: str, method: str, **payload: object) -> object:
    """One Bot API call. Rejections arrive as HTTP 200 with ok:false, so the
    body decides, not the status code.
    """
    response = await client.post(f"{BOT_API_BASE_URL}/bot{token}/{method}", json=payload)
    body = response.json()
    if not body.get("ok"):
        raise RuntimeError(f"Telegram {method} failed: {body.get('description')}")
    return body["result"]


async def verify_telegram_webhooks(
    session: AsyncSession,
    *,
    public_base_url: str | None,
    client: httpx.AsyncClient | None = None,
) -> WebhookWatchRun:
    """Re-point Telegram at this deployment wherever it has drifted away."""
    run = WebhookWatchRun()
    if not public_base_url:
        # Without it there is no URL to compare against, and guessing one
        # would be worse than leaving the registration alone.
        return run

    channels = await _telegram_channels(session)
    if not channels:
        return run

    owned = client is None
    http = client or httpx.AsyncClient(timeout=_TIMEOUT_SECONDS)
    try:
        for channel in channels:
            token = decrypt(channel.credentials)
            secret = str((channel.config or {}).get(WEBHOOK_SECRET_KEY) or "")
            if is_placeholder_credential(token) or not secret:
                # A channel still waiting on its real token, or one whose
                # secret never got written: provisioning owns both cases, and
                # registering here would only invite deliveries this
                # deployment then rejects.
                run.skipped += 1
                continue

            expected = webhook_url(public_base_url, channel.external_id)
            try:
                info = await _call(http, token, "getWebhookInfo")
                current = (info or {}).get("url", "") if isinstance(info, dict) else ""
                if current == expected:
                    run.ok += 1
                    continue

                await _call(
                    http,
                    token,
                    "setWebhook",
                    url=expected,
                    secret_token=secret,
                    allowed_updates=ALLOWED_UPDATES,
                    # False, as at boot: whatever Telegram queued while the
                    # webhook was missing is exactly what this job is for.
                    drop_pending_updates=False,
                )
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                run.failed += 1
                logger.error(
                    "telegram_webhook_repair_failed channel=%s bot_id=%s error=%s",
                    channel.id,
                    channel.external_id,
                    exc,
                )
                continue

            run.repaired += 1
            # ERROR, not WARNING: reaching here means the bot had been
            # silently unreachable, and how long it stayed that way is the
            # question somebody will want the timestamps to answer.
            logger.error(
                "telegram_webhook_repaired channel=%s bot_id=%s was=%r now=%s",
                channel.id,
                channel.external_id,
                current,
                expected,
            )
    finally:
        if owned:
            await http.aclose()
    return run
