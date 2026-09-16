import math
from collections.abc import Callable
from contextlib import AbstractContextManager
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.embeddings import EMBEDDING_DIMENSIONS, EmbeddingProvider
from app.rag.retrieval import DEFAULT_MAX_DISTANCE, retrieve_relevant_faqs
from app.repositories.knowledge_base import KnowledgeBaseRepository
from tests.conftest import Seed


def _direction(*nonzero: tuple[int, float]) -> list[float]:
    """An EMBEDDING_DIMENSIONS-dim vector with the given (index, value) pairs
    set and every other component zero. Cosine distance depends only on angle, not
    magnitude, so putting all the "signal" in a couple of fixed dimensions
    gives exact, easy-to-reason-about distances between test vectors instead
    of relying on real (unpredictable) embeddings.
    """
    vector = [0.0] * EMBEDDING_DIMENSIONS
    for index, value in nonzero:
        vector[index] = value
    return vector


# Three mutually distinguishable directions: E0 and E1 are orthogonal
# (cosine distance 1 - cos(90 deg) = 1.0). MID sits 60 degrees from E0, for
# an exact cosine distance of 1 - cos(60 deg) = 0.5.
E0 = _direction((0, 1.0))
E1 = _direction((1, 1.0))
MID = _direction((0, 0.5), (1, math.sqrt(3) / 2))


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, vector: list[float]) -> None:
        self._vector = vector
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self._vector for _ in texts]


async def _make_faq(
    db_session: AsyncSession,
    question: str,
    embedding: list[float],
    is_active: bool = True,
) -> UUID:
    row = await KnowledgeBaseRepository(db_session).create(
        question=question,
        answer=f"Answer to: {question}",
        embedding=embedding,
        is_active=is_active,
    )
    return row.id


# --- KnowledgeBaseRepository.search() ---


async def test_search_ranks_by_cosine_similarity(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    with as_tenant(seed.tenant_a.id):
        closest_id = await _make_faq(db_session, "closest", E0)
        middle_id = await _make_faq(db_session, "middle", MID)
        farthest_id = await _make_faq(db_session, "farthest", E1)

        matches = await KnowledgeBaseRepository(db_session).search(E0, limit=10)

    # seed() also created one knowledge_base row for tenant A; it's an
    # arbitrary embedding so just confirm our three are in the right order
    # relative to each other, not that they're the only results.
    ids_in_order = [m.knowledge_base.id for m in matches]
    assert ids_in_order.index(closest_id) < ids_in_order.index(middle_id)
    assert ids_in_order.index(middle_id) < ids_in_order.index(farthest_id)

    by_id = {m.knowledge_base.id: m for m in matches}
    assert by_id[closest_id].distance == 0.0
    assert math.isclose(by_id[middle_id].distance, 0.5, abs_tol=1e-6)
    assert math.isclose(by_id[farthest_id].distance, 1.0, abs_tol=1e-6)
    assert math.isclose(by_id[closest_id].score, 1.0, abs_tol=1e-6)


async def test_search_respects_limit(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    with as_tenant(seed.tenant_a.id):
        for i in range(5):
            await _make_faq(db_session, f"faq {i}", E0)

        matches = await KnowledgeBaseRepository(db_session).search(E0, limit=3)

    assert len(matches) == 3


async def test_search_excludes_inactive_rows(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    with as_tenant(seed.tenant_a.id):
        inactive_id = await _make_faq(db_session, "inactive", E0, is_active=False)
        active_id = await _make_faq(db_session, "active", E1)

        matches = await KnowledgeBaseRepository(db_session).search(E0, limit=10)

    ids_seen = {m.knowledge_base.id for m in matches}
    assert inactive_id not in ids_seen
    assert active_id in ids_seen


async def test_search_never_returns_other_tenant_rows(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    with as_tenant(seed.tenant_b.id):
        other_tenant_id = await _make_faq(db_session, "tenant b faq", E0)

    with as_tenant(seed.tenant_a.id):
        matches = await KnowledgeBaseRepository(db_session).search(E0, limit=10)

    ids_seen = {m.knowledge_base.id for m in matches}
    assert other_tenant_id not in ids_seen


# --- retrieve_relevant_faqs() ---


async def test_retrieve_relevant_faqs_embeds_query_and_returns_ranked_matches(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    provider = FakeEmbeddingProvider(E0)
    with as_tenant(seed.tenant_a.id):
        closest_id = await _make_faq(db_session, "closest", E0)
        farthest_id = await _make_faq(db_session, "farthest", E1)

        matches = await retrieve_relevant_faqs(
            db_session, "what are your hours?", embedding_provider=provider, max_distance=None
        )

    assert provider.calls == [["what are your hours?"]]
    ids_in_order = [m.knowledge_base.id for m in matches]
    assert ids_in_order.index(closest_id) < ids_in_order.index(farthest_id)


async def test_retrieve_relevant_faqs_drops_matches_beyond_max_distance(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    provider = FakeEmbeddingProvider(E0)
    with as_tenant(seed.tenant_a.id):
        close_id = await _make_faq(db_session, "close", E0)
        far_id = await _make_faq(db_session, "far", E1)  # distance 1.0 from E0

        matches = await retrieve_relevant_faqs(
            db_session,
            "irrelevant text, provider is faked",
            embedding_provider=provider,
            max_distance=0.3,
        )

    ids_seen = {m.knowledge_base.id for m in matches}
    assert close_id in ids_seen
    assert far_id not in ids_seen


async def test_retrieve_relevant_faqs_tenant_scoped(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    provider = FakeEmbeddingProvider(E0)
    with as_tenant(seed.tenant_b.id):
        other_tenant_id = await _make_faq(db_session, "tenant b faq", E0)

    with as_tenant(seed.tenant_a.id):
        matches = await retrieve_relevant_faqs(
            db_session, "hours?", embedding_provider=provider, max_distance=None
        )

    ids_seen = {m.knowledge_base.id for m in matches}
    assert other_tenant_id not in ids_seen


def test_default_max_distance_sits_between_the_measured_extremes() -> None:
    """The cutoff is measured, not chosen, and the measurement is narrow.

    Against the live knowledge base, the furthest question that still wanted a
    real row landed at 0.3190 ("qon guruhini aniqlash" -> the blood group and
    rhesus test) and the nearest question with no answer in the clinic at all
    landed at 0.3759 ("tish oldirsam qancha bo'ladi" -> ear cleaning, from a
    clinic with no dentist). Moving the cutoff to either side of that gap
    picks a failure: below it a patient is refused an answer that is in the
    table, above it a patient is handed rows about a different part of the
    body. This test is here so that a later nudge to the number has to argue
    with the measurement rather than slide past it.
    """
    assert 0.3190 < DEFAULT_MAX_DISTANCE < 0.3759
