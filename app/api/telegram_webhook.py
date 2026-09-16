"""Telegram's inbound edge.

The mirror of app.api.webhook, and deliberately the same shape: authenticate
the delivery, parse it, work out which channel it belongs to, and hand each
genuine message to the shared services. Everything after that point —
deduplication, the conversation store, debouncing, retrieval, guardrails, the
answer prompt, delivery — is the same code the Instagram bot runs.

The route carries the bot's own id so one deployment can serve many clinics'
bots: Telegram sends no tenant of its own, and a single shared path would
leave the deployment guessing which clinic an update belongs to. The webhook
secret is per-channel too, held in Channel.config, so one clinic's secret
leaking cannot be used to post updates to another's.
"""

import hmac
import logging
from typing import Annotated, Any

from arq.connections import ArqRedis
from fastapi import APIRouter, Depends, Header, HTTPException, Path, Request, Response, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.channels.base import ChannelType
from app.channels.telegram.adapter import BUSINESS_CONNECTION_ID
from app.core.db import get_db_session
from app.core.queue import get_arq_pool
from app.core.redaction import preview
from app.core.tenant_context import reset_current_tenant, set_current_tenant
from app.models.channel import Channel
from app.services.conversation import register_inbound_message
from app.services.debounce import handle_inbound_message
from app.services.idempotency import claim_event
from app.services.profile import remember_username
from app.services.tenant_resolution import ResolvedChannel, bot_replies_enabled, resolve_channel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook/telegram", tags=["Telegram"])

# Where the per-channel webhook secret lives inside Channel.config. Telegram
# echoes it back on every delivery in X-Telegram-Bot-Api-Secret-Token, which
# is the only evidence an update actually came from Telegram.
WEBHOOK_SECRET_KEY = "webhook_secret"


class TelegramChat(BaseModel):
    id: int


class TelegramUser(BaseModel):
    id: int
    is_bot: bool = False
    # Telegram hands the handle over with the message, so this channel needs
    # no lookup at all -- unlike Instagram, which has to be asked (see
    # app.services.profile). Optional because a Telegram account need not
    # have one.
    username: str | None = None


class TelegramMessage(BaseModel):
    # Unique only per chat, not globally — see _event_id for how it is
    # combined into an idempotency key.
    message_id: int
    chat: TelegramChat
    text: str | None = None
    # Present when the bot reaches the chat through Telegram Business rather
    # than a direct conversation. The reply has to go back over the same
    # connection.
    business_connection_id: str | None = None
    from_: TelegramUser | None = None

    model_config = {"populate_by_name": True}

    def __init__(self, **data: Any) -> None:
        # Telegram's field is "from", which is a Python keyword.
        if "from" in data:
            data.setdefault("from_", data.pop("from"))
        super().__init__(**data)


class TelegramUpdate(BaseModel):
    update_id: int
    message: TelegramMessage | None = None
    # A message sent to a Telegram Business account the bot manages. Carried
    # separately by Telegram, but the same thing to this pipeline.
    business_message: TelegramMessage | None = None


def _inbound_message(update: TelegramUpdate) -> TelegramMessage | None:
    """The patient's message on this update, whichever field it arrived in.

    Edited messages are deliberately not handled: answering an edit again
    would send the patient a second reply to a question they only asked
    once, and the transcript would show the message twice.
    """
    return update.message or update.business_message


def _event_id(update: TelegramUpdate) -> str:
    """The idempotency key for this update.

    update_id alone would do for a normal webhook — Telegram increments it
    per bot — but a redelivery after a failed acknowledgement reuses it,
    which is exactly the case being guarded. Combined with chat and message
    id so a replayed update cannot collide with a different chat's.
    """
    message = _inbound_message(update)
    if message is None:
        return f"update:{update.update_id}"
    return f"update:{update.update_id}:{message.chat.id}:{message.message_id}"


async def _verify_webhook_secret(channel: Channel, provided: str | None, *, bot_id: str) -> None:
    """Reject an update whose secret does not match this channel's.

    A channel with no secret configured rejects everything rather than
    letting the update through. Telegram sets the header only when the
    webhook was registered with a secret, so "no secret configured" means
    the webhook was registered without one — and an endpoint that accepts
    unauthenticated updates would let anyone write into a clinic's patient
    transcript. Configuring the secret is part of registering the webhook,
    not an optional hardening step.
    """
    expected = channel.config.get(WEBHOOK_SECRET_KEY)
    if not isinstance(expected, str) or not expected:
        logger.error("telegram_webhook_secret_not_configured", extra={"bot_id": bot_id})
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Webhook secret not configured"
        )
    if provided is None or not hmac.compare_digest(provided, expected):
        logger.warning("telegram_webhook_secret_invalid", extra={"bot_id": bot_id})
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid webhook secret")


@router.post("/{bot_id}")
async def receive_telegram_webhook(
    request: Request,
    bot_id: Annotated[str, Path(description="The bot's own Telegram id — Channel.external_id")],
    x_telegram_bot_api_secret_token: Annotated[str | None, Header()] = None,
    session: AsyncSession = Depends(get_db_session),
    pool: ArqRedis = Depends(get_arq_pool),
) -> Response:
    resolved = await resolve_channel(session, channel_type=ChannelType.TELEGRAM, external_id=bot_id)
    if resolved is None:
        # Deliberately the same 403 an invalid secret gets, and logged
        # rather than answered in detail: an endpoint that distinguishes
        # "no such bot" from "wrong secret" tells an unauthenticated caller
        # which bot ids exist.
        logger.warning("telegram_webhook_unknown_bot", extra={"bot_id": bot_id})
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid webhook secret")

    channel = await session.get(Channel, resolved.channel_id)
    if channel is None:  # pragma: no cover - resolve_channel just found it
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid webhook secret")
    await _verify_webhook_secret(channel, x_telegram_bot_api_secret_token, bot_id=bot_id)

    raw_body = await request.body()
    try:
        update = TelegramUpdate.model_validate_json(raw_body)
    except ValueError:
        # 200, not 4xx: Telegram retries anything else, and an update this
        # deployment cannot parse will not parse on the retry either.
        logger.warning("telegram_webhook_payload_invalid", extra={"bot_id": bot_id})
        return Response(status_code=status.HTTP_200_OK)

    token = set_current_tenant(resolved.tenant_id)
    try:
        await _handle_update(session, pool, resolved, update)
    finally:
        reset_current_tenant(token)

    return Response(status_code=status.HTTP_200_OK)


async def _handle_update(
    session: AsyncSession,
    pool: ArqRedis,
    channel: ResolvedChannel,
    update: TelegramUpdate,
) -> None:
    message = _inbound_message(update)
    if message is None:
        # Not a message update (a callback query, a business_connection
        # status change, an edit) — nothing for the answer pipeline yet.
        logger.info("telegram_non_message_update_skipped", extra={"update_id": update.update_id})
        return

    if message.from_ is not None and message.from_.is_bot:
        # The bot's own message echoed back, or another bot in the chat.
        logger.info("telegram_bot_message_skipped", extra={"chat_id": message.chat.id})
        return

    if message.text is None:
        # A photo, sticker or voice note — nothing the FAQ/LLM pipeline can
        # answer yet, same as an attachment-only Instagram message.
        logger.info("telegram_non_text_message_skipped", extra={"chat_id": message.chat.id})
        return

    # A business message the clinic's own staff typed from their personal
    # account, not the patient. Telegram delivers it over the same
    # connection, and answering it would have the bot talking to the
    # receptionist. In a Business chat the patient is the chat itself, so a
    # sender who is not the chat is somebody on the clinic's side.
    if (
        message.business_connection_id
        and message.from_ is not None
        and str(message.from_.id) != str(message.chat.id)
    ):
        logger.info(
            "telegram_business_operator_message_skipped",
            extra={"chat_id": message.chat.id, "sender_id": message.from_.id},
        )
        return

    if not await claim_event(
        pool,
        tenant_id=channel.tenant_id,
        channel_type=channel.channel_type,
        event_id=_event_id(update),
    ):
        logger.info(
            "telegram_duplicate_skipped",
            extra={"update_id": update.update_id, "chat_id": message.chat.id},
        )
        return

    sender_external_id = str(message.chat.id)

    # Full message text stays out of INFO — it is patient content that
    # should not sit in logs that may ship to external monitoring (TZ
    # section 7, personal data). Length + a short truncated preview only.
    logger.info(
        "telegram_message_received",
        extra={
            "tenant_id": str(channel.tenant_id),
            "chat_id": sender_external_id,
            "message_length": len(message.text),
            "message_preview": preview(message.text),
        },
    )

    reply_context = (
        {BUSINESS_CONNECTION_ID: message.business_connection_id}
        if message.business_connection_id
        else None
    )

    inbound = await register_inbound_message(
        session,
        channel_id=channel.channel_id,
        channel_type=channel.channel_type,
        sender_external_id=sender_external_id,
        text=message.text,
        reply_context=reply_context,
    )
    # Free, and before the early returns for the same reason the Instagram
    # webhook enqueues its lookup there: a conversation an operator answers
    # by hand still needs a name on it in the dashboard.
    if message.from_ is not None:
        await remember_username(session, user_id=inbound.user_id, username=message.from_.username)
    await session.commit()

    if not inbound.is_bot_enabled:
        logger.info(
            "telegram_bot_disabled_for_conversation",
            extra={"conversation_id": str(inbound.conversation_id)},
        )
        return

    if not await bot_replies_enabled(session, channel.tenant_id):
        # The clinic's own switch, the same one the Instagram webhook reads:
        # one dashboard button silences every channel, which is what somebody
        # turning the assistant off actually means.
        logger.info(
            "telegram_bot_disabled_for_tenant",
            extra={
                "tenant_id": str(channel.tenant_id),
                "conversation_id": str(inbound.conversation_id),
            },
        )
        return

    await handle_inbound_message(
        pool,
        tenant_id=channel.tenant_id,
        channel_id=channel.channel_id,
        conversation_id=inbound.conversation_id,
        sender_external_id=sender_external_id,
        message_text=message.text,
        reply_context=reply_context,
    )
