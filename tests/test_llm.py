from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.rag.embeddings import OpenAIEmbeddingProvider, _select_embedding_provider
from app.rag.llm import (
    HF_ROUTER_BASE_URL,
    OpenAILLMProvider,
    QwenLLMProvider,
    _select_llm_provider,
)

TEST_SETTINGS = Settings(
    database_url="postgresql+asyncpg://test:test@localhost/test",
    redis_url="redis://localhost:6379/0",
    openai_api_key="sk-test",
    webhook_verify_token="test-verify-token",
    meta_app_secret="test-app-secret",
)


class _FakeCompletionsResource:
    def __init__(self, content: str | None) -> None:
        self.create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )
        )


class _FakeChat:
    def __init__(self, content: str | None) -> None:
        self.completions = _FakeCompletionsResource(content)


class _FakeAsyncOpenAI:
    def __init__(self, content: str | None) -> None:
        self.chat = _FakeChat(content)


# No test here ever talks to the real OpenAI API: AsyncOpenAI is monkeypatched
# at the point llm.py imports it, so OpenAILLMProvider.__init__ picks up the
# fake client instead of a real network client.


async def test_generate_prepends_system_prompt_and_returns_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeAsyncOpenAI("Sure, we're open 9 to 5!")
    monkeypatch.setattr("app.rag.llm.AsyncOpenAI", lambda **kwargs: fake_client)

    provider = OpenAILLMProvider(settings=TEST_SETTINGS)
    result = await provider.generate(
        "You are a helpful assistant.", [{"role": "user", "content": "What are your hours?"}]
    )

    assert result == "Sure, we're open 9 to 5!"
    fake_client.chat.completions.create.assert_awaited_once_with(
        model=TEST_SETTINGS.openai_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What are your hours?"},
        ],
        # A reasoning model is asked to think briefly: a patient waits for
        # every second of it (see Settings.openai_reasoning_effort).
        reasoning_effort=TEST_SETTINGS.openai_reasoning_effort,
    )


async def test_openai_model_comes_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """OPENAI_MODEL is a dashboard lever: a model that
    retires or answers badly has to be swappable while patients are waiting,
    not after a redeploy. It was pinned in code to gpt-4o-mini until now.
    """
    fake_client = _FakeAsyncOpenAI("ok")
    monkeypatch.setattr("app.rag.llm.AsyncOpenAI", lambda **kwargs: fake_client)

    settings = TEST_SETTINGS.model_copy(update={"openai_model": "gpt-5.6-luna"})
    provider = OpenAILLMProvider(settings=settings)
    await provider.generate("system", [{"role": "user", "content": "hi"}])

    assert fake_client.chat.completions.create.await_args.kwargs["model"] == "gpt-5.6-luna"


async def test_generate_raises_when_content_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeAsyncOpenAI(None)
    monkeypatch.setattr("app.rag.llm.AsyncOpenAI", lambda **kwargs: fake_client)

    provider = OpenAILLMProvider(settings=TEST_SETTINGS)
    with pytest.raises(ValueError, match="no text content"):
        await provider.generate("system", [{"role": "user", "content": "hi"}])


# --- Qwen, through Hugging Face's OpenAI-compatible router ---


def test_qwen_talks_to_the_hugging_face_router_with_the_hugging_face_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It is the OpenAI client, pointed elsewhere — so the thing worth
    asserting is that it is pointed elsewhere, and with the right credential.
    Sending an OpenAI key to Hugging Face, or an HF token to OpenAI, fails at
    the first patient message rather than here.
    """
    captured: dict[str, object] = {}

    def _fake_client(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("app.rag.llm.AsyncOpenAI", _fake_client)

    QwenLLMProvider(settings=TEST_SETTINGS.model_copy(update={"hf_token": "hf_test"}))

    assert captured["api_key"] == "hf_test"
    assert captured["base_url"] == HF_ROUTER_BASE_URL


def test_qwen_without_a_token_says_which_variable_is_missing() -> None:
    with pytest.raises(ValueError, match="HF_TOKEN"):
        QwenLLMProvider(settings=TEST_SETTINGS)


def test_llm_provider_overrides_the_backend_that_writes_replies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.rag.llm.AsyncOpenAI", lambda **kwargs: object())
    settings = TEST_SETTINGS.model_copy(update={"llm_provider": "qwen", "hf_token": "hf_test"})

    assert isinstance(_select_llm_provider(settings), QwenLLMProvider)


def test_overriding_the_reply_backend_leaves_embeddings_where_they_are() -> None:
    """The point of the override. Moving embeddings invalidates every vector
    in knowledge_base, so a deployment switching the model that writes
    replies must not drag the clinic's FAQ along with it.
    """
    moved = TEST_SETTINGS.model_copy(update={"llm_provider": "qwen", "hf_token": "hf_test"})

    assert isinstance(_select_embedding_provider(moved), OpenAIEmbeddingProvider)
