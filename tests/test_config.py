import pytest
from pydantic import ValidationError

from app.core.config import Settings

_BASE_KWARGS = {
    "database_url": "postgresql+asyncpg://test:test@localhost/test",
    "redis_url": "redis://localhost:6379/0",
    "webhook_verify_token": "test-verify-token",
    "meta_app_secret": "test-app-secret",
    # A real (test-only) Fernet key, not an arbitrary string — encryption_key
    # is validated at construction (see test_encryption_key_* below), so
    # every other test in this file needs one that actually passes.
    "encryption_key": "Hq3_REB-V0twf7iBgCPCSUZQiG44egxyiZg9kOKRxUg=",
    # Required now that OpenAI makes every embedding. Given here rather than
    # per test so that a test which is about something else does not quietly
    # depend on the machine running it having a real key in the environment.
    "openai_api_key": "sk-test",
}


def test_without_an_openai_key_it_refuses_to_boot() -> None:
    """OpenAI makes every embedding in the system, so a deployment without a
    key cannot retrieve a single FAQ row whatever writes its replies. Better
    to fail at startup than to accept webhooks and answer none of them.

    openai_api_key must be forced to None explicitly, not just omitted:
    Settings falls back to the real OPENAI_API_KEY env var for any field not
    given a constructor value, which would silently defeat this test in an
    environment (like CI, or this test run) that has one set.
    """
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        Settings(**{**_BASE_KWARGS, "openai_api_key": None})


def test_an_openai_key_is_all_a_deployment_needs() -> None:
    settings = Settings(**_BASE_KWARGS)
    assert settings.openai_model == "gpt-5-mini"
    assert settings.llm_provider is None


def test_the_replies_can_be_moved_off_openai_on_their_own() -> None:
    """The escape hatch, for the afternoon OpenAI is unreachable. The
    embeddings stay where they are, because moving those means re-embedding
    the whole knowledge base.
    """
    settings = Settings(**_BASE_KWARGS, llm_provider="qwen", hf_token="hf")

    assert settings.llm_provider == "qwen"


def test_moving_the_replies_without_the_credential_raises() -> None:
    with pytest.raises(ValidationError, match="HF_TOKEN"):
        Settings(**_BASE_KWARGS, llm_provider="qwen", hf_token=None)


def test_gemini_is_no_longer_a_provider() -> None:
    """It was removed outright rather than left selectable: a deployment that
    could still be pointed at it would embed its FAQ into a vector space
    nothing else in the system shares, and the failure is silent.
    """
    with pytest.raises(ValidationError):
        Settings(**_BASE_KWARGS, llm_provider="gemini")


# --- encryption_key ---


def test_missing_encryption_key_raises() -> None:
    kwargs = dict(_BASE_KWARGS)
    del kwargs["encryption_key"]
    # Must be forced to None, not just omitted — same env-var-fallback trap
    # as OPENAI_API_KEY above.
    with pytest.raises(ValidationError, match="encryption_key"):
        Settings(**kwargs, encryption_key=None)  # type: ignore[arg-type]


def test_malformed_encryption_key_raises() -> None:
    with pytest.raises(ValidationError, match="ENCRYPTION_KEY must be a valid Fernet key"):
        Settings(
            **{**_BASE_KWARGS, "encryption_key": "not-a-valid-fernet-key"},
        )


def test_valid_encryption_key_is_accepted() -> None:
    settings = Settings(**_BASE_KWARGS)
    assert settings.encryption_key == _BASE_KWARGS["encryption_key"]


# --- database_url driver normalization ---


def test_driverless_postgres_url_gets_asyncpg_driver() -> None:
    settings = Settings(
        **{**_BASE_KWARGS, "database_url": "postgresql://u:p@host:5432/db"},
    )
    assert settings.database_url == "postgresql+asyncpg://u:p@host:5432/db"


def test_legacy_postgres_scheme_gets_asyncpg_driver() -> None:
    # Some managed hosts still hand out the older `postgres://` alias.
    settings = Settings(
        **{**_BASE_KWARGS, "database_url": "postgres://u:p@host:5432/db"},
    )
    assert settings.database_url == "postgresql+asyncpg://u:p@host:5432/db"


def test_explicit_driver_is_left_alone() -> None:
    settings = Settings(
        **{**_BASE_KWARGS, "database_url": "postgresql+psycopg://u:p@host:5432/db"},
    )
    assert settings.database_url == "postgresql+psycopg://u:p@host:5432/db"


def test_password_containing_the_scheme_is_not_mangled() -> None:
    # Only the leading scheme is rewritten — a `postgres://` sitting inside
    # the credentials must survive untouched.
    url = "postgresql+asyncpg://u:postgres%3A//p@host:5432/db"
    settings = Settings(
        **{**_BASE_KWARGS, "database_url": url},
    )
    assert settings.database_url == url


def test_clinic_phone_numbers_default_to_none() -> None:
    """Unset is the supported resting state: app.services.answer drops the
    "call these numbers" half of the pricing fallback rather than letting the
    model produce a number of its own.
    """
    settings = Settings(**_BASE_KWARGS, clinic_phone_numbers=None)
    assert settings.clinic_phone_numbers is None


def test_pasted_clinic_phone_numbers_are_stripped() -> None:
    """Pasting into a hosting dashboard picks up a trailing newline, and this
    value is read back to patients verbatim.
    """
    settings = Settings(
        **_BASE_KWARGS,
        clinic_phone_numbers="  +998 90 123 45 67\n",
    )
    assert settings.clinic_phone_numbers == "+998 90 123 45 67"
