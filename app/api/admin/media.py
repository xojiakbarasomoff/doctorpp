"""The "Rasm yuborganlar" screen: patients who sent photos, and the photos.

Grouped by patient rather than listed photo by photo, because that is how
they are handled: somebody opens a patient, looks at everything they sent,
and marks them seen. A patient with anything unseen sorts to the top.
"""

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import undefer

from app.api.admin.deps import require_patient_access, verify_csrf_header
from app.core.db import get_db_session
from app.core.tenant_context import get_current_tenant
from app.models.operator import Operator
from app.models.patient_media import PatientMedia, PatientMediaStatus
from app.models.user import User
from app.services import patient_media

router = APIRouter(prefix="/api/admin/media", tags=["Admin — Patient photos"])


class MediaItemOut(BaseModel):
    id: uuid.UUID
    status: str
    created_at: datetime
    available: bool
    size_bytes: int | None


class MediaPatientOut(BaseModel):
    user_id: uuid.UUID
    conversation_id: uuid.UUID | None
    name: str | None
    username: str | None
    phone: str | None
    total: int
    unseen: int
    last_at: datetime
    items: list[MediaItemOut]


class StatusUpdate(BaseModel):
    status: PatientMediaStatus


@router.get("", response_model=list[MediaPatientOut])
async def list_patients(
    only_unseen: bool = Query(default=False),
    q: str | None = Query(default=None, max_length=100),
    operator: Operator = Depends(require_patient_access),
    session: AsyncSession = Depends(get_db_session),
) -> list[MediaPatientOut]:
    rows = (
        await session.execute(
            select(PatientMedia, User)
            .join(User, PatientMedia.user_id == User.id)
            .where(PatientMedia.tenant_id == get_current_tenant())
            .order_by(PatientMedia.created_at.desc())
            .limit(2000)
        )
    ).all()

    grouped: dict[uuid.UUID, MediaPatientOut] = {}
    for media, user in rows:
        entry = grouped.get(user.id)
        if entry is None:
            entry = grouped[user.id] = MediaPatientOut(
                user_id=user.id,
                conversation_id=media.conversation_id,
                name=user.name,
                username=user.username,
                phone=user.phone,
                total=0,
                unseen=0,
                last_at=media.created_at,
                items=[],
            )
        entry.total += 1
        entry.unseen += media.status == PatientMediaStatus.NEW
        entry.items.append(
            MediaItemOut(
                id=media.id,
                status=media.status,
                created_at=media.created_at,
                available=True,
                size_bytes=media.size_bytes,
            )
        )

    patients = list(grouped.values())
    if only_unseen:
        patients = [p for p in patients if p.unseen]
    if q:
        needle = q.strip().lstrip("@").lower()
        patients = [
            p
            for p in patients
            if needle in (p.name or "").lower()
            or needle in (p.username or "").lower()
            or needle in (p.phone or "")
        ]
    # Unseen first, then most recent.
    patients.sort(key=lambda p: (p.unseen == 0, -p.last_at.timestamp()))
    return patients


@router.get("/{media_id}/file")
async def file(
    media_id: uuid.UUID,
    operator: Operator = Depends(require_patient_access),
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    media = (
        await session.execute(
            select(PatientMedia)
            # The bytes are deferred so that lists never load them; here they
            # are the whole point, and a deferred column touched after the
            # query is a lazy load an async session refuses.
            .options(undefer(PatientMedia.content)).where(
                PatientMedia.id == media_id, PatientMedia.tenant_id == get_current_tenant()
            )
        )
    ).scalar_one_or_none()
    if media is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Rasm topilmadi")
    if media.content is None:
        # Not downloaded when it arrived: try once more now, through the
        # Graph API if the original link is refused.
        channel_id = (
            await session.execute(select(User.channel_id).where(User.id == media.user_id))
        ).scalar_one_or_none()
        if channel_id is None or not await patient_media.repair(session, media, channel_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Rasm yuklanmadi")
    content_type = (
        media.content_type if media.content_type in patient_media.SAFE_IMAGE_TYPES else None
    )
    return Response(
        content=media.content,
        media_type=content_type or "application/octet-stream",
        # Patient photos: never cached by a shared proxy, never sniffed into
        # something executable, and -- opened on their own in a tab, on the
        # dashboard's origin -- never able to run anything even if they were.
        headers={
            "Cache-Control": "private, max-age=3600",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'; img-src 'self'; sandbox",
        },
    )


@router.post("/{media_id}/status", dependencies=[Depends(verify_csrf_header)])
async def set_status(
    media_id: uuid.UUID,
    payload: StatusUpdate,
    operator: Operator = Depends(require_patient_access),
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, str]:
    result = await session.execute(
        update(PatientMedia)
        .where(PatientMedia.id == media_id, PatientMedia.tenant_id == get_current_tenant())
        .values(status=payload.status)
    )
    if result.rowcount == 0:  # type: ignore[attr-defined]
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Rasm topilmadi")
    await session.commit()
    return {"status": payload.status}


@router.post("/patients/{user_id}/seen", dependencies=[Depends(verify_csrf_header)])
async def mark_patient_seen(
    user_id: uuid.UUID,
    operator: Operator = Depends(require_patient_access),
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, int]:
    result = await session.execute(
        update(PatientMedia)
        .where(
            PatientMedia.user_id == user_id,
            PatientMedia.tenant_id == get_current_tenant(),
            PatientMedia.status == PatientMediaStatus.NEW,
        )
        .values(status=PatientMediaStatus.REVIEWED)
    )
    await session.commit()
    return {"updated": result.rowcount}  # type: ignore[attr-defined]
