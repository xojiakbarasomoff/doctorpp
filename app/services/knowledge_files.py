"""Putting an uploaded file into the knowledge base.

Read the file, cut it into chunks, embed every chunk, and only then write
anything: a file whose embedding failed leaves nothing behind, not a document
row that lists as ready and answers nothing. A file uploaded again under the
same name replaces the old one in the same transaction, so a price list that
is corrected is never present twice and never absent between the two.
"""

import asyncio
import logging
from pathlib import PurePath

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge_document import KnowledgeDocument
from app.rag.embeddings import EMBEDDING_MODEL, EmbeddingProvider, get_embedding_provider
from app.repositories.knowledge_document import (
    KnowledgeChunkRepository,
    KnowledgeDocumentRepository,
)
from app.services import documents
from app.services.documents import Chunk, DocumentError

logger = logging.getLogger(__name__)

# Enough for a clinic's price lists, preparation sheets and schedules; a
# limit at all so that one tenant cannot fill the vector index.
MAX_DOCUMENTS = 100


class EmbeddingUnavailableError(Exception):
    """The embedding service could not be reached or refused the request."""


def _embedding_text(filename: str, chunk: Chunk) -> str:
    """What is embedded for a chunk: its words, and where they came from.

    The file's name and the page or sheet are part of what a question is
    about ("narxlar ro'yxati", "tayyorgarlik"), so they are embedded with the
    passage rather than left out of it.
    """
    stem = PurePath(filename).stem.replace("_", " ").replace("-", " ")
    where = f"{stem}. {chunk.label}" if chunk.label else stem
    return f"{where}\n{chunk.text}"


async def ingest_document(
    session: AsyncSession,
    *,
    filename: str,
    content_type: str | None,
    data: bytes,
    embedding_provider: EmbeddingProvider | None = None,
) -> KnowledgeDocument:
    """Store a file in the current tenant's knowledge base.

    Raises DocumentError for a file that cannot be used and
    EmbeddingUnavailableError when it could be read but not indexed; in both
    cases nothing has been written.
    """
    name = documents.clean_filename(filename)
    # The parsing is CPU-bound and a large PDF takes seconds: off the event
    # loop, so the rest of the dashboard and the bot keep answering.
    chunks = await asyncio.to_thread(documents.parse, name, data)

    docs = KnowledgeDocumentRepository(session)
    existing = await docs.get_by_filename(name)
    if existing is None and await docs.count() >= MAX_DOCUMENTS:
        raise DocumentError(
            f"Fayllar soni {MAX_DOCUMENTS} tadan oshmasligi kerak. Keraksizlarini o'chiring."
        )

    provider = embedding_provider or get_embedding_provider()
    try:
        vectors = await provider.embed([_embedding_text(name, c) for c in chunks])
    except Exception as error:
        logger.exception("knowledge_file_embedding_failed filename=%s", name)
        raise EmbeddingUnavailableError from error

    if existing is not None:
        await docs.delete_with_chunks(existing)
    document = await docs.create(
        filename=name,
        content_type=content_type,
        size_bytes=len(data),
        chunk_count=len(chunks),
    )
    await KnowledgeChunkRepository(session).add_many(
        document.id,
        [
            {
                "ordinal": ordinal,
                "label": c.label,
                "content": c.text,
                "embedding": vector,
                "embedding_model": EMBEDDING_MODEL,
            }
            for ordinal, (c, vector) in enumerate(zip(chunks, vectors, strict=True))
        ],
    )
    logger.info(
        "knowledge_file_ingested",
        extra={"document": name, "chunks": len(chunks), "replaced": existing is not None},
    )
    return document
