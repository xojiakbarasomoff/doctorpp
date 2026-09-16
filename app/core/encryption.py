from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import get_settings


class DecryptionError(Exception):
    """Raised when a stored value can't be decrypted with the configured key —
    wrong key, corrupted/tampered ciphertext, or a value that was never
    encrypted in the first place. One exception type for all of those, so
    callers don't need to know which cryptography-library exception to
    catch.
    """


@lru_cache
def _default_fernet() -> Fernet:
    # Settings already validates ENCRYPTION_KEY is a well-formed Fernet key
    # at startup (see app.core.config._require_valid_encryption_key) — by
    # the time this runs, construction here can't fail on a malformed key,
    # only ever be called with whatever key the app actually started with.
    return Fernet(get_settings().encryption_key.encode("utf-8"))


def encrypt(plaintext: str, *, key: str | None = None) -> str:
    """Encrypts a secret (e.g. a channel access token) for storage.

    `key` is injectable so tests can encrypt/decrypt with an explicit key
    instead of depending on ENCRYPTION_KEY being set in the environment —
    same pattern as embedding_provider/llm_provider elsewhere in this
    codebase.
    """
    fernet = Fernet(key.encode("utf-8")) if key is not None else _default_fernet()
    return fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt(ciphertext: str, *, key: str | None = None) -> str:
    """Decrypts a value produced by encrypt(). Raises DecryptionError — never
    silently returns garbage or falls back to treating the input as
    plaintext — if the configured/given key can't open it.
    """
    fernet = Fernet(key.encode("utf-8")) if key is not None else _default_fernet()
    try:
        return fernet.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except InvalidToken as exc:
        raise DecryptionError(
            "credentials could not be decrypted with the configured ENCRYPTION_KEY"
        ) from exc
