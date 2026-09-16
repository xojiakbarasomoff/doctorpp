from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.rag.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    OpenAIEmbeddingProvider,
    _select_embedding_provider,
)

TEST_SETTINGS = Settings(
    database_url="postgresql+asyncpg://test:test@localhost/test",
    redis_url="redis://localhost:6379/0",
    openai_api_key="sk-test",
    webhook_verify_token="test-verify-token",
    meta_app_secret="test-app-secret",
)


class _FakeEmbeddingsResource:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.create = AsyncMock(
            return_value=SimpleNamespace(
                data=[SimpleNamespace(embedding=vector) for vector in vectors]
            )
        )


class _FakeAsyncOpenAI:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.embeddings = _FakeEmbeddingsResource(vectors)


# No test here ever talks to the real OpenAI API: AsyncOpenAI is monkeypatched
# at the point embeddings.py imports it, so OpenAIEmbeddingProvider.__init__
# picks up the fake client instead of a real network client.


async def test_embed_returns_vectors_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    vectors = [[0.1] * EMBEDDING_DIMENSIONS, [0.2] * EMBEDDING_DIMENSIONS]
    fake_client = _FakeAsyncOpenAI(vectors)
    monkeypatch.setattr("app.rag.embeddings.AsyncOpenAI", lambda **kwargs: fake_client)

    provider = OpenAIEmbeddingProvider(settings=TEST_SETTINGS)
    result = await provider.embed(["hello", "world"])

    assert result == vectors
    fake_client.embeddings.create.assert_awaited_once_with(
        model="text-embedding-3-small",
        input=["hello", "world"],
        dimensions=EMBEDDING_DIMENSIONS,
    )


async def test_embed_empty_list_skips_api_call(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeAsyncOpenAI([])
    monkeypatch.setattr("app.rag.embeddings.AsyncOpenAI", lambda **kwargs: fake_client)

    provider = OpenAIEmbeddingProvider(settings=TEST_SETTINGS)
    result = await provider.embed([])

    assert result == []
    fake_client.embeddings.create.assert_not_awaited()


# --- provider selection ---


def test_embeddings_are_always_openais() -> None:
    """There is nothing to select between any more. LLM_PROVIDER can move the
    replies elsewhere in an emergency; the embeddings never move with them,
    because vectors from two models are not comparable and moving these means
    re-embedding the clinic's whole FAQ.
    """
    assert isinstance(_select_embedding_provider(TEST_SETTINGS), OpenAIEmbeddingProvider)

    moved_replies = TEST_SETTINGS.model_copy(update={"llm_provider": "qwen", "hf_token": "hf"})
    assert isinstance(_select_embedding_provider(moved_replies), OpenAIEmbeddingProvider)


def test_the_model_that_makes_the_vectors_is_named_and_fixed() -> None:
    """It is written onto every row it embeds, so that a later change of
    model is detected and repaired instead of silently emptying retrieval.
    """
    assert EMBEDDING_MODEL == "text-embedding-3-small"
    assert EMBEDDING_DIMENSIONS == 1536


# --- constructing a provider without its key raises (defense in depth,
# independent of Settings' own validator) ---


def test_openai_embedding_provider_without_key_raises() -> None:
    settings = TEST_SETTINGS.model_copy(update={"openai_api_key": None})
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingProvider(settings=settings)


class _EchoingOpenAIEmbeddings:
    """One vector per input text, so a chunked embed can be checked for both
    call count and ordering.
    """

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    async def create(self, *, model: str, input: list[str], dimensions: int):
        self.batches.append(list(input))
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[float(int(text))] * EMBEDDING_DIMENSIONS)
                for text in input
            ]
        )


async def test_openai_embed_splits_batches_over_the_api_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI takes at most 2048 inputs per request. The cap sat unnoticed
    because the clinic was under it -- and the knowledge base has been growing
    a wording at a time, from 98 rows to over 3500. ingest_faqs embeds a whole
    file in one call, so the deploy that crosses 2048 loses its entire seed.
    """
    embeddings = _EchoingOpenAIEmbeddings()
    fake_client = SimpleNamespace(embeddings=embeddings)
    monkeypatch.setattr("app.rag.embeddings.AsyncOpenAI", lambda **kwargs: fake_client)

    texts = [str(index) for index in range(5000)]
    provider = OpenAIEmbeddingProvider(settings=TEST_SETTINGS)
    result = await provider.embed(texts)

    assert [len(batch) for batch in embeddings.batches] == [2048, 2048, 904]
    assert len(result) == 5000
    # Order survives the split: vector i still belongs to text i.
    assert [vector[0] for vector in result] == [float(index) for index in range(5000)]
