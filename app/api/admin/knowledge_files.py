"""Files in the knowledge base: upload, list, delete.

The bot answers from these as well as from the question-and-answer rows
(app.rag.retrieval.retrieve_knowledge). Uploading is a clinic-level change --
whatever is in a file will be quoted to patients with nobody reading it first
-- so it takes the same permission as editing the FAQ.
"""

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.admin.deps import require_manage_clinic, verify_csrf_header
from app.api.auth import get_current_operator
from app.core.db import get_db_session
from app.models.knowledge_document import KnowledgeDocument
from app.models.operator import Operator
from app.repositories.knowledge_document import KnowledgeDocumentRepository
from app.services.documents import MAX_FILE_BYTES, DocumentError
from app.services.knowledge_files import EmbeddingUnavailableError, ingest_document

router = APIRouter(prefix="/api/admin/knowledge-files", tags=["Admin — Knowledge files"])


class KnowledgeFileOut(BaseModel):
    id: uuid.UUID
    filename: str
    content_type: str | None
    size_bytes: int
    chunk_count: int
    created_at: datetime


def _out(document: KnowledgeDocument) -> KnowledgeFileOut:
    return KnowledgeFileOut(
        id=document.id,
        filename=document.filename,
        content_type=document.content_type,
        size_bytes=document.size_bytes,
        chunk_count=document.chunk_count,
        created_at=document.created_at,
    )


@router.get("", response_model=list[KnowledgeFileOut])
async def list_files(
    operator: Operator = Depends(get_current_operator),
    session: AsyncSession = Depends(get_db_session),
) -> list[KnowledgeFileOut]:
    return [_out(d) for d in await KnowledgeDocumentRepository(session).list_newest()]


@router.post(
    "",
    response_model=KnowledgeFileOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_csrf_header)],
)
async def upload_file(
    file: UploadFile,
    operator: Operator = Depends(require_manage_clinic),
    session: AsyncSession = Depends(get_db_session),
) -> KnowledgeFileOut:
    """Read, cut up and index one file. A file with the same name replaces the
    one already there.

    Read one byte past the limit and no further: a request is not allowed to
    make the server hold whatever it was sent.
    """
    data = await file.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Fayl juda katta. Eng ko'pi {MAX_FILE_BYTES // 1024 // 1024} MB.",
        )
    try:
        document = await ingest_document(
            session,
            filename=file.filename or "fayl",
            content_type=file.content_type,
            data=data,
        )
        await session.commit()
    except DocumentError as error:
        # Nothing has been written: the file is read and the embeddings are
        # made before the first row is.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from None
    except EmbeddingUnavailableError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Indekslash xizmati javob bermadi. Birozdan keyin qayta urinib ko'ring.",
        ) from None
    except IntegrityError:
        # Two uploads of the same name at once: the second loses the race.
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Shu nomli fayl hozir yuklanmoqda. Birozdan keyin qayta urinib ko'ring.",
        ) from None
    return _out(document)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(verify_csrf_header)],
)
async def delete_file(
    document_id: uuid.UUID,
    operator: Operator = Depends(require_manage_clinic),
    session: AsyncSession = Depends(get_db_session),
) -> None:
    """Remove a file, and with it everything the bot could have quoted from it."""
    repo = KnowledgeDocumentRepository(session)
    document = await repo.get(document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Fayl topilmadi")
    await repo.delete_with_chunks(document)
    await session.commit()
