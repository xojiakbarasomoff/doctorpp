import secrets
import uuid
from dataclasses import dataclass
from functools import lru_cache

import itsdangerous

from app.core.config import get_settings

SESSION_COOKIE_NAME = "session"
SESSION_MAX_AGE_SECONDS = 60 * 60 * 12  # 12h — long enough for a work shift, not indefinite.


@dataclass(frozen=True)
class Session:
    operator_id: uuid.UUID
    # Bound into the same signed cookie as operator_id, so it can't be
    # forged independently of a real login — compared against a hidden form
    # field on every state-changing request (the double-submit pattern),
    # without needing a second cookie or server-side session storage.
    csrf_token: str


@lru_cache
def _serializer() -> itsdangerous.URLSafeTimedSerializer:
    # Settings already validates session_secret_key is present at startup —
    # by the time this runs it can't fail on a missing key.
    return itsdangerous.URLSafeTimedSerializer(
        get_settings().session_secret_key, salt="operator-session"
    )


def create_session_cookie(operator_id: uuid.UUID) -> tuple[str, str]:
    """Returns (cookie_value, csrf_token) for a fresh login."""
    csrf_token = secrets.token_urlsafe(32)
    cookie_value = _serializer().dumps({"operator_id": str(operator_id), "csrf_token": csrf_token})
    return cookie_value, csrf_token


def read_session_cookie(cookie_value: str) -> Session | None:
    """Decodes and verifies a session cookie, or None if it's missing,
    expired (older than SESSION_MAX_AGE_SECONDS), tampered with, or
    malformed — one None for every failure mode, since callers only ever
    need to know "is this a valid session or not", not why.
    """
    try:
        payload = _serializer().loads(cookie_value, max_age=SESSION_MAX_AGE_SECONDS)
        return Session(
            operator_id=uuid.UUID(payload["operator_id"]), csrf_token=payload["csrf_token"]
        )
    except (itsdangerous.BadData, KeyError, ValueError, TypeError):
        return None
