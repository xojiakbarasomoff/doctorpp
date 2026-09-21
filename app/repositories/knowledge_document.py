import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import delete, func, select

from app.core.tenant_context import get_current_tenant
from app.models.knowledge_document import KnowledgeChunk, KnowledgeDocument
from app.repositories.base import CrossTenantAccessError, TenantScopedRepository


@dataclass(frozen=True)
class ChunkMatch:
    """One search hit: the piece of a file, the file it came from, and how far
    its embedding is from the query's."""

    chunk: KnowledgeChunk
    filename: str
    distance: float


class KnowledgeDocumentRepository(TenantScopedRepository[KnowledgeDocument]):
    model = KnowledgeDocument

    async def get_by_filename(self, filename: str) -> KnowledgeDocument | None:
        return await self._get(filename=filename, tenant_id=get_current_tenant())

    async def list_newest(self) -> Sequence[KnowledgeDocument]:
        stmt = (
            select(KnowledgeDocument)
            .where(KnowledgeDocument.tenant_id == get_current_tenant())
            .order_by(KnowledgeDocument.created_at.desc())
        )
        return (await self.session.execute(stmt)).scalars().all()

    async def count(self) -> int:
        stmt = select(func.count()).where(KnowledgeDocument.tenant_id == get_current_tenant())
        return int((await self.session.execute(stmt)).scalar_one())

    async def delete_with_chunks(self, document: KnowledgeDocument) -> None:
        """Remove a file and everything cut from it.

        The chunks are deleted explicitly, as well as by the foreign key's
        cascade: the ORM does not know about the cascade, and would otherwise
        be holding rows the database has already removed.
        """
        tenant_id = get_current_tenant()
        if document.tenant_id != tenant_id:
            raise CrossTenantAccessError(
                f"document.tenant_id={document.tenant_id} does not match the current "
                f"tenant ({tenant_id})"
            )
        await self.session.execute(
            delete(KnowledgeChunk).where(
                KnowledgeChunk.document_id == document.id, KnowledgeChunk.tenant_id == tenant_id
            )
        )
        await self.session.delete(document)
        await self.session.flush()


class KnowledgeChunkRepository(TenantScopedRepository[KnowledgeChunk]):
    model = KnowledgeChunk

    async def add_many(
        self, document_id: uuid.UUID, rows: Sequence[dict[str, object]]
    ) -> list[KnowledgeChunk]:
        tenant_id = self._resolve_tenant_id(None)
        chunks = [
            KnowledgeChunk(tenant_id=tenant_id, document_id=document_id, **row) for row in rows
        ]
        self.session.add_all(chunks)
        await self.session.flush()
        return chunks

    async def search(self, query_embedding: list[float], limit: int = 4) -> list[ChunkMatch]:
        """Tenant-scoped semantic search over uploaded files, closest first.

        The same shape as KnowledgeBaseRepository.search: ranked in SQL by
        pgvector's cosine distance, scoped to the current tenant in the query
        itself rather than after it.
        """
        distance = KnowledgeChunk.embedding.cosine_distance(query_embedding).label("distance")
        stmt = (
            select(KnowledgeChunk, KnowledgeDocument.filename, distance)
            .join(KnowledgeDocument, KnowledgeDocument.id == KnowledgeChunk.document_id)
            .where(KnowledgeChunk.tenant_id == get_current_tenant())
            .order_by(distance)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return [ChunkMatch(chunk=c, filename=name, distance=d) for c, name, d in result.all()]
