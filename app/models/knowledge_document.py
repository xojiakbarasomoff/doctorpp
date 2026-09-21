import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class KnowledgeDocument(Base):
    """A file the clinic uploaded to its knowledge base.

    Only what the dashboard lists lives here. The words themselves are in
    KnowledgeChunk, cut into pieces small enough to be found by a question
    and to sit in a prompt; the original file is not kept, because nothing
    reads it again -- a changed file is uploaded again, under the same name,
    and replaces this one.
    """

    __tablename__ = "knowledge_documents"
    __table_args__ = (
        UniqueConstraint("tenant_id", "filename", name="uq_knowledge_documents_tenant_filename"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )


class KnowledgeChunk(Base):
    """One retrievable piece of an uploaded file."""

    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        # Same index, same operator class as knowledge_base: the search
        # orders by Vector.cosine_distance(), and an index built with another
        # op class would not be used by it.
        Index(
            "ix_knowledge_chunks_embedding_hnsw_cosine",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    # Where in the file this came from -- "3-bet", "Narxlar" -- so the model
    # can be told, and the clinic can see, what a reply was built from.
    label: Mapped[str | None] = mapped_column(String(200), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536), nullable=False)
    # See KnowledgeBase.embedding_model: vectors from two models are not
    # comparable, so the model that made this one is written down.
    embedding_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
