"""Transport for the Telegram Bot API.

Deliberately dumb, the same way app.channels.instagram.client is: it knows
how to make the HTTP call and nothing about tenants, retrieval or guardrails.
Anything that is a business rule lives in app/services, where both bots reach
it.
"""

import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

import httpx

logger = logging.getLogger(__name__)

BOT_API_BASE_URL = "https://api.telegram.org"

# Telegram issues bot tokens as "<bot_id>:<secret>". A channel seeded before
# its real token arrived carries one of these instead, so the pipeline can
# tell "not configured yet" apart from "wrong token" and skip sending rather
# than calling the API with a value that cannot work. Mirrors
# app.channels.instagram.client._PLACEHOLDER_CREDENTIALS.
_PLACEHOLDER_CREDENTIALS = frozenset({"", "placeholder", "todo", "changeme", "pending"})


def is_placeholder_credential(credentials: str) -> bool:
    return credentials.strip().lower() in _PLACEHOLDER_CREDENTIALS


class TelegramSendError(Exception):
    """Raised when the Bot API rejects or fails a send request."""


class TelegramClient(ABC):
    """Abstraction over "send a text message to a Telegram chat", mirroring
    InstagramClient so the transport can be swapped for a test double without
    touching callers.
    """

    @abstractmethod
    async def send_message(
        self,
        *,
        bot_token: str,
        chat_id: str,
        text: str,
        business_connection_id: str | None = None,
    ) -> None:
        """Send `text` to `chat_id`, authenticated with `bot_token`.

        business_connection_id is set when the conversation reached the bot
        through Telegram Business rather than a direct chat — the reply has
        to go back over that same connection, or it is delivered from the bot
        account instead of the clinic's own account (or refused outright).

        Raises TelegramSendError on any transport failure or on a response
        the API marks as not ok.
        """


class BotAPITelegramClient(TelegramClient):
    """Real implementation: POSTs to api.telegram.org."""

    def __init__(self, *, base_url: str = BOT_API_BASE_URL, timeout: float = 10.0) -> None:
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def send_message(
        self,
        *,
        bot_token: str,
        chat_id: str,
        text: str,
        business_connection_id: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {"chat_id": _normalize_chat_id(chat_id), "text": text}
        if business_connection_id:
            payload["business_connection_id"] = business_connection_id

        # The token is a path segment for this API, so it would land in any
        # log line that records the URL. app.core.logging keeps httpx's own
        # request logging off for exactly this reason (see its module
        # docstring); nothing here logs the path either.
        response = await self._http.post(f"/bot{bot_token}/sendMessage", json=payload)

        # A logical failure comes back as HTTP 200 with ok=false, so the
        # status code alone is not enough to tell whether the message was
        # delivered.
        description: str | None = None
        ok = False
        try:
            body = response.json()
            ok = bool(body.get("ok"))
            description = body.get("description")
        except ValueError:
            ok = False

        if response.is_error or not ok:
            logger.error(
                "telegram_send_failed",
                extra={
                    "chat_id": chat_id,
                    "status_code": response.status_code,
                    # Telegram's description names the reason ("chat not
                    # found", "bot was blocked by the user") and carries no
                    # credential, unlike Meta's error payloads.
                    "description": description,
                    "over_business_connection": bool(business_connection_id),
                },
            )
            raise TelegramSendError(
                f"Telegram send failed with status {response.status_code}: {description}"
            )


@lru_cache
def get_telegram_client() -> TelegramClient:
    return BotAPITelegramClient()


def _normalize_chat_id(chat_id: str) -> int | str:
    """Telegram accepts a numeric chat id or an "@channelusername".

    Numeric ids are sent as integers rather than strings: the API accepts
    both, but a channel or supergroup id is negative and large, and sending
    it as a string has been a reliable source of "chat not found" against
    some deployments. Anything non-numeric (a username) is passed through.
    """
    candidate = chat_id.strip()
    if candidate.lstrip("-").isdigit():
        return int(candidate)
    return candidate
