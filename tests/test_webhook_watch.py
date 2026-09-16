"""The Telegram webhook watchdog, against each state it can find.

Stubbed transport and a stubbed session: what matters here is which answer
from Telegram leads to a setWebhook and which does not, and neither needs a
real channel row to be true.
"""

import uuid
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from app.core.encryption import encrypt
from app.services.webhook_watch import verify_telegram_webhooks, webhook_url

BASE = "https://clinic.example"
BOT_ID = "8803059390"
EXPECTED = f"{BASE}/webhook/telegram/{BOT_ID}"


class _Session:
    async def execute(self, _stmt: Any) -> Any:
        channels = self._channels
        return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: channels))

    def __init__(self, channels: list[Any]) -> None:
        self._channels = channels


def _channel(token: str = "8803059390:AAsecret", secret: str | None = "s3cr3t") -> Any:
    return SimpleNamespace(
        id=uuid.uuid4(),
        external_id=BOT_ID,
        credentials=encrypt(token),
        config={} if secret is None else {"webhook_secret": secret},
    )


class _Telegram:
    """Answers getWebhookInfo with `url` and records every setWebhook."""

    def __init__(self, url: str, *, set_ok: bool = True) -> None:
        self.url = url
        self.set_ok = set_ok
        self.set_calls: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/getWebhookInfo"):
            return httpx.Response(200, json={"ok": True, "result": {"url": self.url}})
        import json

        self.set_calls.append(json.loads(request.content))
        if not self.set_ok:
            return httpx.Response(200, json={"ok": False, "description": "Bad webhook"})
        return httpx.Response(200, json={"ok": True, "result": True})


async def _run(handler: Any, channels: list[Any], base: str | None = BASE) -> Any:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        return await verify_telegram_webhooks(
            _Session(channels),
            public_base_url=base,
            client=client,  # type: ignore[arg-type]
        )
    finally:
        await client.aclose()


def test_the_url_is_built_the_same_way_provisioning_builds_it() -> None:
    """The check and the repair share this, and so does app.core.provisioning.
    A trailing slash on the configured base must not produce a second one.
    """
    assert webhook_url(BASE, BOT_ID) == EXPECTED
    assert webhook_url(BASE + "/", BOT_ID) == EXPECTED


async def test_a_cleared_webhook_is_put_back() -> None:
    """The real failure: something called deleteWebhook, or polled, and
    Telegram now has no destination at all.
    """
    telegram = _Telegram(url="")

    run = await _run(telegram, [_channel()])

    assert run.repaired == 1
    assert len(telegram.set_calls) == 1
    sent = telegram.set_calls[0]
    assert sent["url"] == EXPECTED
    assert sent["secret_token"] == "s3cr3t"
    # The backlog Telegram held while the webhook was gone is the whole
    # reason to repair it, so it must not be dropped on the way back.
    assert sent["drop_pending_updates"] is False


async def test_a_webhook_pointing_somewhere_else_is_taken_back() -> None:
    run = await _run(_Telegram(url="https://someone-else.example/hook"), [_channel()])

    assert run.repaired == 1


async def test_a_correct_webhook_is_left_untouched() -> None:
    """The common case, and the one that runs every ten minutes: it must cost
    one read and write nothing.
    """
    telegram = _Telegram(url=EXPECTED)

    run = await _run(telegram, [_channel()])

    assert run.ok == 1
    assert run.repaired == 0
    assert telegram.set_calls == []


@pytest.mark.parametrize(
    ("channel", "why"),
    [
        (_channel(token="pending"), "no real token yet"),
        (_channel(secret=None), "no secret to validate deliveries against"),
    ],
)
async def test_a_half_configured_channel_is_skipped(channel: Any, why: str) -> None:
    """Registering either of these would invite deliveries the webhook route
    then rejects, which is worse than staying unregistered.
    """
    telegram = _Telegram(url="")

    run = await _run(telegram, [channel])

    assert run.skipped == 1, why
    assert telegram.set_calls == []


async def test_a_refused_registration_is_counted_and_logged_not_raised() -> None:
    """This runs on a cron across every tenant: one clinic's bad token must
    not stop the pass before it reaches the others.
    """
    run = await _run(_Telegram(url="", set_ok=False), [_channel()])

    assert run.failed == 1
    assert run.repaired == 0
    assert run.notable


async def test_without_a_public_url_nothing_is_registered() -> None:
    """A deployment that does not know its own address cannot compare, and a
    guessed URL would point Telegram at nothing.
    """
    telegram = _Telegram(url="")

    run = await _run(telegram, [_channel()], base=None)

    assert not run.notable
    assert telegram.set_calls == []
