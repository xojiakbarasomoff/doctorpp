import json
import logging
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from app.channels.instagram.client import (
    GraphAPIInstagramClient,
    InstagramSendError,
    is_placeholder_credential,
    is_within_messaging_window,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


# --- is_within_messaging_window ---


def test_within_window_just_under_24h() -> None:
    last_message_at = NOW - timedelta(hours=23, minutes=59)
    assert is_within_messaging_window(last_message_at, now=NOW) is True


def test_exactly_24h_is_still_within_window() -> None:
    last_message_at = NOW - timedelta(hours=24)
    assert is_within_messaging_window(last_message_at, now=NOW) is True


def test_outside_window_just_over_24h() -> None:
    last_message_at = NOW - timedelta(hours=24, minutes=1)
    assert is_within_messaging_window(last_message_at, now=NOW) is False


def test_outside_window_days_stale() -> None:
    last_message_at = NOW - timedelta(days=3)
    assert is_within_messaging_window(last_message_at, now=NOW) is False


# --- is_placeholder_credential ---


@pytest.mark.parametrize(
    "credentials", ["", "  ", "placeholder", "PLACEHOLDER", "TODO", "changeme", "pending"]
)
def test_placeholder_credentials_are_detected(credentials: str) -> None:
    assert is_placeholder_credential(credentials) is True


@pytest.mark.parametrize("credentials", ["token", "EAAG...realtoken", "PLACEHOLDERX"])
def test_real_looking_credentials_are_not_placeholders(credentials: str) -> None:
    assert is_placeholder_credential(credentials) is False


# --- GraphAPIInstagramClient ---


async def test_send_text_posts_recipient_and_message_with_token_as_query_param() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"recipient_id": "sender-1", "message_id": "mid.1"})

    client = GraphAPIInstagramClient()
    client._http = httpx.AsyncClient(
        base_url=client._http.base_url, transport=httpx.MockTransport(handler)
    )

    await client.send_text(access_token="secret-token", recipient_igsid="sender-1", text="Hi!")

    [request] = captured
    assert request.url.params["access_token"] == "secret-token"
    assert request.url.path.endswith("/me/messages")
    assert json.loads(request.content) == {
        "recipient": {"id": "sender-1"},
        "message": {"text": "Hi!"},
    }


async def test_send_text_raises_on_error_response_without_leaking_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "Invalid token", "code": 190}})

    client = GraphAPIInstagramClient()
    client._http = httpx.AsyncClient(
        base_url=client._http.base_url, transport=httpx.MockTransport(handler)
    )

    with pytest.raises(InstagramSendError):
        await client.send_text(access_token="bad-token", recipient_igsid="sender-1", text="Hi!")


async def test_send_failure_logs_the_subcode_that_says_what_went_wrong(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Meta answers "no such user", "outside the 24-hour messaging window"
    and "this app may not message this account" all with code 100. Without
    the subcode an operator reading the log cannot tell which of those
    happened, and the three have completely different fixes.

    The values here are a real response: subcode 2534014 is what the live
    Graph API returns for a recipient id it cannot resolve.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "Не удается найти запрошенного пользователя.",
                    "type": "IGApiException",
                    "code": 100,
                    "error_subcode": 2534014,
                    "fbtrace_id": "A0U1DBpPiP12byVCL0NKiF2",
                }
            },
        )

    client = GraphAPIInstagramClient()
    client._http = httpx.AsyncClient(
        base_url=client._http.base_url, transport=httpx.MockTransport(handler)
    )

    with caplog.at_level(logging.ERROR, logger="app"), pytest.raises(InstagramSendError):
        await client.send_text(
            access_token="a-real-looking-token", recipient_igsid="9900112233445566", text="Hi!"
        )

    record = next(r for r in caplog.records if r.message == "instagram_send_failed")
    assert record.error_code == 100
    assert record.error_subcode == 2534014
    # Still no body, and above all no token: Meta's error payloads can echo
    # request params back in the copy.
    assert "a-real-looking-token" not in caplog.text
    assert "fbtrace_id" not in caplog.text


# --- fetching the sender's handle ---


def _client_over(handler) -> GraphAPIInstagramClient:
    client = GraphAPIInstagramClient()
    client._http = httpx.AsyncClient(
        base_url=client._http.base_url, transport=httpx.MockTransport(handler)
    )
    return client


async def test_fetch_username_asks_the_senders_node_for_the_handle() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"username": "asomov", "id": "9900112233445566"})

    client = _client_over(handler)

    username = await client.fetch_username(access_token="secret-token", igsid="9900112233445566")

    assert username == "asomov"
    [request] = captured
    assert request.url.path.endswith("/9900112233445566")
    assert request.url.params["fields"] == "username"
    assert request.url.params["access_token"] == "secret-token"


async def test_fetch_username_strips_an_at_sign_if_meta_sends_one() -> None:
    client = _client_over(lambda request: httpx.Response(200, json={"username": " @asomov "}))

    assert await client.fetch_username(access_token="t", igsid="1") == "asomov"


@pytest.mark.parametrize(
    "response",
    [
        # The permission was not granted, or the id is stale.
        httpx.Response(400, json={"error": {"message": "Unsupported get request", "code": 100}}),
        httpx.Response(403, json={"error": {"message": "Permissions error", "code": 10}}),
        # Meta is having an afternoon.
        httpx.Response(500, text="upstream"),
        # A 200 with nothing useful in it: an account with no handle set.
        httpx.Response(200, json={"id": "9900112233445566"}),
        httpx.Response(200, json={"username": "   "}),
        httpx.Response(200, text="not json at all"),
    ],
)
async def test_a_handle_that_cannot_be_had_is_none_rather_than_an_error(
    response: httpx.Response,
) -> None:
    """None, never an exception. This is called on the way to answering a
    patient, and a nicer label in the dashboard is not worth dropping a
    message for.
    """
    client = _client_over(lambda request: response)

    assert await client.fetch_username(access_token="t", igsid="1") is None


async def test_a_network_failure_is_also_just_a_missing_handle() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    client = _client_over(handler)

    assert await client.fetch_username(access_token="t", igsid="1") is None
