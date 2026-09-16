"""add HNSW cosine index to knowledge_base embedding

Revision ID: cb58f7d9c82c
Revises: 81d8ed5b3b93
Create Date: 2026-08-15 00:46:53.754792

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "cb58f7d9c82c"
down_revision: str | Sequence[str] | None = "81d8ed5b3b93"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # cosine ops to match KnowledgeBaseRepository.search()'s use of
    # Vector.cosine_distance() (the `<=>` operator). HNSW, not IVFFlat:
    # better recall/speed for our scale, no training step needed. The
    # embedding column stays Vector(1536): Gemini's gemini-embedding-001
    # natively outputs 3072 dims, explicitly truncated to 1536 via
    # output_dimensionality (see app.rag.embeddings.EMBEDDING_DIMENSIONS) —
    # pgvector's HNSW/IVFFlat indexes hard-cap at 2000 dims (verified
    # against pgvector 0.8.6), so the native 3072 couldn't be indexed at all.
    op.create_index(
        "ix_knowledge_base_embedding_hnsw_cosine",
        "knowledge_base",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "ix_knowledge_base_embedding_hnsw_cosine",
        table_name="knowledge_base",
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
