"""Files in the knowledge base: what is stored, what is found, whose it is."""

from collections.abc import Callable
from contextlib import AbstractContextManager
from uuid import UUID

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge_document import KnowledgeChunk, KnowledgeDocument
from app.rag.embeddings import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, EmbeddingProvider
from app.rag.retrieval import retrieve_knowledge
from app.repositories.base import CrossTenantAccessError
from app.repositories.knowledge_document import KnowledgeDocumentRepository
from app.services import knowledge_files
from app.services.documents import DocumentError
from app.services.knowledge_files import EmbeddingUnavailableError, ingest_document
from tests.conftest import Seed

_TOPICS = ("narx", "tayyor", "manzil")


class TopicEmbeddings(EmbeddingProvider):
    """Puts every text on the axis of the first topic word it contains.

    Texts about the same thing are then identical to the search and texts
    about different things are as far apart as they can be, which is all a
    test of who finds what needs from an embedding.
    """

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("embedding service down")
        vectors = []
        for text in texts:
            axis = next((i for i, t in enumerate(_TOPICS) if t in text.lower()), len(_TOPICS))
            vector = [0.0] * EMBEDDING_DIMENSIONS
            vector[axis] = 1.0
            vectors.append(vector)
        return vectors


PRICES = b"Xizmat,Narx\nUZI,150000\nEKG,80000"
PREPARATION = b"Tahlilga tayyorgarlik\nTahlildan oldin 8 soat ovqat yemang."


async def _upload(
    session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    filename: str,
    data: bytes,
    *,
    provider: EmbeddingProvider | None = None,
    tenant: UUID | None = None,
) -> KnowledgeDocument:
    with as_tenant(tenant or seed.tenant_a.id):
        return await ingest_document(
            session,
            filename=filename,
            content_type="text/plain",
            data=data,
            embedding_provider=provider or TopicEmbeddings(),
        )


async def _chunk_count(session: AsyncSession, tenant: UUID) -> int:
    stmt = (
        select(func.count()).select_from(KnowledgeChunk).where(KnowledgeChunk.tenant_id == tenant)
    )
    return int((await session.execute(stmt)).scalar_one())


async def test_an_uploaded_file_is_stored_with_its_chunks_and_the_model_that_embedded_them(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    document = await _upload(db_session, seed, as_tenant, "narxlar.csv", PRICES)

    assert document.filename == "narxlar.csv"
    assert document.size_bytes == len(PRICES)
    assert document.chunk_count == 1
    with as_tenant(seed.tenant_a.id):
        chunks = (
            (
                await db_session.execute(
                    select(KnowledgeChunk).where(KnowledgeChunk.document_id == document.id)
                )
            )
            .scalars()
            .all()
        )
    [stored] = chunks
    assert stored.content == "Xizmat: UZI; Narx: 150000\nXizmat: EKG; Narx: 80000"
    assert stored.embedding_model == EMBEDDING_MODEL
    assert stored.ordinal == 0


async def test_a_question_finds_the_piece_of_the_file_that_answers_it(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    await _upload(db_session, seed, as_tenant, "narxlar.csv", PRICES)
    await _upload(db_session, seed, as_tenant, "tayyorgarlik.txt", PREPARATION)

    with as_tenant(seed.tenant_a.id):
        knowledge = await retrieve_knowledge(db_session, "UZI narxi qancha?", TopicEmbeddings())

    [match] = knowledge.chunks
    assert match.filename == "narxlar.csv"
    assert "Narx: 150000" in match.chunk.content
    assert match.distance == pytest.approx(0.0)


async def test_a_message_about_nothing_in_the_files_finds_nothing(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    await _upload(db_session, seed, as_tenant, "narxlar.csv", PRICES)

    with as_tenant(seed.tenant_a.id):
        knowledge = await retrieve_knowledge(db_session, "salom, yaxshimisiz", TopicEmbeddings())

    assert knowledge.chunks == []


async def test_a_message_is_embedded_once_for_both_kinds_of_knowledge(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    provider = TopicEmbeddings()

    with as_tenant(seed.tenant_a.id):
        await retrieve_knowledge(db_session, "narxi qancha", provider)

    assert provider.calls == 1


async def test_uploading_a_name_again_replaces_the_old_file_rather_than_adding_to_it(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    first = await _upload(db_session, seed, as_tenant, "narxlar.csv", PRICES)
    second = await _upload(
        db_session, seed, as_tenant, "narxlar.csv", b"Xizmat,Narx\nUZI,999000"
    )

    with as_tenant(seed.tenant_a.id):
        documents = await KnowledgeDocumentRepository(db_session).list_newest()
        knowledge = await retrieve_knowledge(db_session, "UZI narxi", TopicEmbeddings())
        assert await _chunk_count(db_session, seed.tenant_a.id) == 1

    assert [d.id for d in documents] == [second.id]
    assert first.id != second.id
    [match] = knowledge.chunks
    assert "999000" in match.chunk.content
    assert "150000" not in match.chunk.content


async def test_deleting_a_file_removes_everything_the_bot_could_quote_from_it(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    document = await _upload(db_session, seed, as_tenant, "narxlar.csv", PRICES)

    with as_tenant(seed.tenant_a.id):
        await KnowledgeDocumentRepository(db_session).delete_with_chunks(document)
        knowledge = await retrieve_knowledge(db_session, "narxi", TopicEmbeddings())
        assert await _chunk_count(db_session, seed.tenant_a.id) == 0

    assert knowledge.chunks == []


async def test_one_clinics_files_are_never_found_by_another(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    await _upload(db_session, seed, as_tenant, "narxlar.csv", PRICES, tenant=seed.tenant_b.id)

    with as_tenant(seed.tenant_a.id):
        knowledge = await retrieve_knowledge(db_session, "narxi qancha", TopicEmbeddings())
        listed = await KnowledgeDocumentRepository(db_session).list_newest()
        assert await KnowledgeDocumentRepository(db_session).get_by_filename("narxlar.csv") is None
    with as_tenant(seed.tenant_b.id):
        theirs = await retrieve_knowledge(db_session, "narxi qancha", TopicEmbeddings())

    assert knowledge.chunks == []
    assert listed == []
    assert len(theirs.chunks) == 1


async def test_two_clinics_may_use_the_same_file_name(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    await _upload(db_session, seed, as_tenant, "narxlar.csv", PRICES)
    await _upload(db_session, seed, as_tenant, "narxlar.csv", PRICES, tenant=seed.tenant_b.id)

    assert await _chunk_count(db_session, seed.tenant_a.id) == 1
    assert await _chunk_count(db_session, seed.tenant_b.id) == 1


async def test_another_clinics_file_cannot_be_deleted(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    theirs = await _upload(
        db_session, seed, as_tenant, "narxlar.csv", PRICES, tenant=seed.tenant_b.id
    )

    with as_tenant(seed.tenant_a.id), pytest.raises(CrossTenantAccessError):
        await KnowledgeDocumentRepository(db_session).delete_with_chunks(theirs)


async def test_a_file_that_could_not_be_embedded_leaves_nothing_behind(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    with pytest.raises(EmbeddingUnavailableError):
        await _upload(
            db_session, seed, as_tenant, "narxlar.csv", PRICES, provider=TopicEmbeddings(fail=True)
        )

    with as_tenant(seed.tenant_a.id):
        assert await KnowledgeDocumentRepository(db_session).list_newest() == []
    assert await _chunk_count(db_session, seed.tenant_a.id) == 0


async def test_a_failed_replacement_keeps_the_file_it_was_meant_to_replace(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    await _upload(db_session, seed, as_tenant, "narxlar.csv", PRICES)

    with pytest.raises(EmbeddingUnavailableError):
        await _upload(
            db_session,
            seed,
            as_tenant,
            "narxlar.csv",
            b"Xizmat,Narx\nUZI,1",
            provider=TopicEmbeddings(fail=True),
        )

    with as_tenant(seed.tenant_a.id):
        knowledge = await retrieve_knowledge(db_session, "narxi", TopicEmbeddings())
    assert "150000" in knowledge.chunks[0].chunk.content


async def test_an_unusable_file_leaves_nothing_behind(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    with pytest.raises(DocumentError):
        await _upload(db_session, seed, as_tenant, "virus.exe", b"MZ")

    assert await _chunk_count(db_session, seed.tenant_a.id) == 0


async def test_there_is_a_ceiling_on_files_but_replacing_is_always_allowed(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(knowledge_files, "MAX_DOCUMENTS", 1)
    await _upload(db_session, seed, as_tenant, "a.txt", b"birinchi")

    with pytest.raises(DocumentError, match="oshmasligi kerak"):
        await _upload(db_session, seed, as_tenant, "b.txt", b"ikkinchi")
    await _upload(db_session, seed, as_tenant, "a.txt", b"yangilangan")

    with as_tenant(seed.tenant_a.id):
        assert await KnowledgeDocumentRepository(db_session).count() == 1


async def test_the_file_name_and_the_page_are_part_of_what_is_embedded(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """A question about "tayyorgarlik" should find a chunk of tayyorgarlik.txt
    even when the passage itself never uses the word."""
    await _upload(db_session, seed, as_tenant, "tayyorgarlik.txt", b"Ovqat yemang, suv iching.")

    with as_tenant(seed.tenant_a.id):
        knowledge = await retrieve_knowledge(db_session, "tayyorgarlik", TopicEmbeddings())

    assert [m.filename for m in knowledge.chunks] == ["tayyorgarlik.txt"]
