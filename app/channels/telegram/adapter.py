"""Telegram's implementation of the ChannelAdapter contract.

Holds the one rule that is Telegram's rather than ours — a channel whose
credential is still a placeholder cannot send — and the one piece of routing
that is: a conversation reached through Telegram Business is answered over
that same business connection.

Telegram has no equivalent of Meta's 24-hour messaging window for a chat the
user themselves opened, so delivery_block_reason returns None once a real
token is configured. That difference is exactly why the window check lives
behind this method rather than in the shared worker, where it used to sit.
"""

from collections.abc import Mapping
from datetime import datetime
from functools import lru_cache
from typing import Any

from app.channels.base import ChannelAdapter, ChannelType, DeliveryBlocked
from app.channels.telegram.client import (
    TelegramClient,
    get_telegram_client,
    is_placeholder_credential,
)

# Key under which the inbound edge stores the business connection a message
# arrived on. Named here, next to the code that reads it, because the webhook
# writes it and this adapter is the only thing that may interpret it — the
# shared pipeline in between passes the mapping through untouched.
BUSINESS_CONNECTION_ID = "business_connection_id"


class TelegramAdapter(ChannelAdapter):
    channel_type = ChannelType.TELEGRAM

    def __init__(self, client: TelegramClient | None = None) -> None:
        # Resolved lazily rather than at construction, for the same reason
        # the Instagram adapter does it: registration runs at import time,
        # and building the real httpx client there would open a connection
        # pool in every process that merely imports app.channels.
        self._client = client

    def _resolve_client(self) -> TelegramClient:
        return self._client or get_telegram_client()

    async def send_text(
        self,
        *,
        credentials: str,
        recipient_external_id: str,
        text: str,
        reply_context: Mapping[str, Any] | None = None,
    ) -> None:
        business_connection_id = None
        if reply_context is not None:
            raw = reply_context.get(BUSINESS_CONNECTION_ID)
            # Guarded rather than trusted: reply_context is rebuilt from job
            # arguments that originated in a webhook payload, so a wrong
            # type here would otherwise reach the API as a malformed field.
            business_connection_id = raw if isinstance(raw, str) and raw else None

        await self._resolve_client().send_message(
            bot_token=credentials,
            chat_id=recipient_external_id,
            text=text,
            business_connection_id=business_connection_id,
        )

    def delivery_block_reason(
        self, *, credentials: str, last_user_message_at: datetime
    ) -> DeliveryBlocked | None:
        if is_placeholder_credential(credentials):
            return DeliveryBlocked.NOT_CONFIGURED
        return None


@lru_cache
def get_telegram_adapter() -> TelegramAdapter:
    return TelegramAdapter()
