from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from app.core.tenant_context import get_current_tenant
from app.models.knowledge_base import KnowledgeBase
from app.repositories.base import CrossTenantAccessError, TenantScopedRepository


@dataclass(frozen=True)
class KnowledgeBaseMatch:
    """One search() hit: the row plus how far its embedding is from the
    query, in the metric pgvector's `<=>` operator computes.
    """

    knowledge_base: KnowledgeBase
    distance: float

    @property
    def score(self) -> float:
        """Cosine similarity (1.0 = identical direction, 0.0 = orthogonal),
        derived from distance since pgvector's cosine_distance is
        1 - cosine_similarity.
        """
        return 1.0 - self.distance


class KnowledgeBaseRepository(TenantScopedRepository[KnowledgeBase]):
    model = KnowledgeBase

    async def get_by_question(self, question: str) -> KnowledgeBase | None:
        """Exact-match lookup scoped to the current tenant, used to decide
        whether a FAQ import is a new row or an update to an existing one.
        """
        return await self._get(question=question, tenant_id=get_current_tenant())

    async def update(self, obj: KnowledgeBase, **values: Any) -> KnowledgeBase:
        """Mutates and flushes an already-loaded row.

        Callers are expected to have fetched `obj` through a tenant-scoped
        read (e.g. get_by_question()), but this doesn't trust that: it
        re-checks obj.tenant_id against the current tenant itself, the same
        way create() re-checks a caller-supplied tenant_id via
        _resolve_tenant_id(), so a future caller that skips the scoped fetch
        fails loudly instead of silently writing across tenants.
        """
        current_tenant_id = get_current_tenant()
        if obj.tenant_id != current_tenant_id:
            raise CrossTenantAccessError(
                f"obj.tenant_id={obj.tenant_id} does not match the current tenant "
                f"({current_tenant_id})"
            )
        for field, value in values.items():
            setattr(obj, field, value)
        await self.session.flush()
        return obj

    async def search(
        self, query_embedding: list[float], limit: int = 5
    ) -> list[KnowledgeBaseMatch]:
        """Tenant-scoped semantic search over active FAQs, closest first.

        Ranks by pgvector cosine distance (the `<=>` operator, exposed here
        via Vector.cosine_distance()) between each row's embedding and
        query_embedding — the ordering and the LIMIT both happen in SQL, so
        Postgres does the ranking rather than pulling every row into Python.
        Inactive rows and rows outside the current tenant are excluded from
        the query entirely, not filtered after the fact.
        """
        distance = KnowledgeBase.embedding.cosine_distance(query_embedding).label("distance")
        stmt = (
            select(KnowledgeBase, distance)
            .where(
                KnowledgeBase.tenant_id == get_current_tenant(),
                KnowledgeBase.is_active.is_(True),
            )
            .order_by(distance)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return [KnowledgeBaseMatch(knowledge_base=kb, distance=dist) for kb, dist in result.all()]
