from collections.abc import Callable
from contextlib import AbstractContextManager
from uuid import UUID

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.embeddings import EMBEDDING_DIMENSIONS, EmbeddingProvider
from app.repositories.base import CrossTenantAccessError
from app.repositories.knowledge_base import KnowledgeBaseRepository
from app.services.knowledge_base import FAQImport, ingest_faqs
from tests.conftest import Seed


class FakeEmbeddingProvider(EmbeddingProvider):
    """Records every batch it's asked to embed and returns deterministic,
    correctly-shaped vectors — no network call, so tests never touch OpenAI.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text))] * EMBEDDING_DIMENSIONS for text in texts]


# --- ingest inserts rows with (mocked) embeddings ---


async def test_ingest_faqs_inserts_rows_with_embeddings(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    provider = FakeEmbeddingProvider()
    faqs = [
        FAQImport(
            question="Do you take walk-ins?",
            answer="Yes, subject to availability.",
            category="general",
        ),
        FAQImport(
            question="What insurance do you accept?",
            answer="We accept most major PPOs.",
            category="billing",
        ),
    ]

    with as_tenant(seed.tenant_a.id):
        rows = await ingest_faqs(db_session, faqs, embedding_provider=provider)

    assert provider.calls == [[faq.question for faq in faqs]]
    assert len(rows) == 2
    for row, faq in zip(rows, faqs, strict=True):
        assert row.tenant_id == seed.tenant_a.id
        assert row.question == faq.question
        assert row.answer == faq.answer
        assert row.category == faq.category
        assert len(row.embedding) == EMBEDDING_DIMENSIONS


# --- re-import updates instead of duplicating ---


async def test_ingest_faqs_reimport_updates_existing_row_instead_of_duplicating(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    # `seed` already created a knowledge_base row for tenant A with this exact
    # question ("What are your hours?" -> "9 to 5."), so re-importing it with
    # a changed answer/category should update that row in place.
    updated = FAQImport(
        question="What are your hours?", answer="8am-6pm, Mon-Sat.", category="hours"
    )
    provider = FakeEmbeddingProvider()

    with as_tenant(seed.tenant_a.id):
        rows = await ingest_faqs(db_session, [updated], embedding_provider=provider)
        all_rows = await KnowledgeBaseRepository(db_session).list()

    assert len(rows) == 1
    assert rows[0].id == seed.a.knowledge_base.id
    assert rows[0].answer == "8am-6pm, Mon-Sat."
    assert rows[0].category == "hours"
    assert len(all_rows) == 1


# --- validation rejects bad rows ---


def test_faq_import_rejects_missing_question() -> None:
    with pytest.raises(ValidationError):
        FAQImport.model_validate({"answer": "Yes, we do."})


def test_faq_import_rejects_missing_answer() -> None:
    with pytest.raises(ValidationError):
        FAQImport.model_validate({"question": "Do you take walk-ins?"})


def test_faq_import_rejects_blank_question() -> None:
    with pytest.raises(ValidationError):
        FAQImport(question="   ", answer="Yes.")


async def test_ingest_faqs_rejects_bad_row_before_calling_embedding_provider(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    provider = FakeEmbeddingProvider()

    with as_tenant(seed.tenant_a.id), pytest.raises(ValidationError):
        await ingest_faqs(
            db_session, [{"question": "", "answer": "ok"}], embedding_provider=provider
        )

    assert provider.calls == []


# --- tenant isolation: ingest under tenant A doesn't leak to B ---


async def test_ingest_faqs_under_tenant_a_is_invisible_to_tenant_b(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    faq = FAQImport(question="Do you offer teeth whitening?", answer="Yes.", category="services")
    provider = FakeEmbeddingProvider()

    with as_tenant(seed.tenant_a.id):
        [inserted] = await ingest_faqs(db_session, [faq], embedding_provider=provider)

    with as_tenant(seed.tenant_b.id):
        leaked = await KnowledgeBaseRepository(db_session).get_by_question(
            "Do you offer teeth whitening?"
        )
        b_rows = await KnowledgeBaseRepository(db_session).list()

    assert leaked is None
    assert inserted.id not in {row.id for row in b_rows}
    assert inserted.tenant_id == seed.tenant_a.id


async def test_repository_update_on_other_tenant_row_raises(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """Guards KnowledgeBaseRepository.update() itself, independent of
    ingest_faqs: even if a caller got hold of another tenant's row (e.g. by
    skipping the tenant-scoped get_by_question() lookup), update() must
    refuse to write to it rather than trusting the caller.
    """
    with as_tenant(seed.tenant_a.id), pytest.raises(CrossTenantAccessError):
        await KnowledgeBaseRepository(db_session).update(seed.b.knowledge_base, answer="hacked")
